//! Resampling strategies: cross-validation (plain, repeated, leave-one-out),
//! holdout, bootstrap, spatial, stratified, group, time-series.

pub mod spatial;
pub mod stratified;
pub mod time_series;

use crate::{Result, SmeltError};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

pub use spatial::{SpatialBlockCV, SpatialBufferCV};
pub use stratified::{GroupCV, StratifiedCV};
pub use time_series::TimeSeriesCV;

/// Trait for resampling strategies.
///
/// `Send + Sync` (all built-in implementers are plain data, trivially both)
/// so a `&dyn Resample` can be shared across threads -- e.g. by tuning
/// methods that evaluate independent hyperparameter candidates in parallel
/// with rayon.
pub trait Resample: Send + Sync {
    /// Generate train/test index splits. Fails if `n_samples` is
    /// inconsistent with this strategy's own configuration (e.g. a
    /// coordinate/group/label vector supplied at construction time whose
    /// length doesn't match, or a fold count that doesn't divide sensibly).
    fn splits(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>>;
}

/// K-fold cross-validation.
pub struct CrossValidation {
    /// Number of folds to split the data into.
    pub folds: usize,
    /// RNG seed used to shuffle samples before splitting into folds.
    pub seed: u64,
}

impl Default for CrossValidation {
    fn default() -> Self {
        Self { folds: 5, seed: 42 }
    }
}

impl CrossValidation {
    /// Create a K-fold cross-validation with the given number of folds.
    pub fn new(folds: usize) -> Self {
        Self { folds, seed: 42 }
    }
    /// Set the RNG seed used to shuffle samples before splitting into folds.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Resample for CrossValidation {
    fn splits(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if self.folds < 2 {
            return Err(SmeltError::InvalidParameter(format!(
                "CrossValidation requires at least 2 folds, got {}",
                self.folds
            )));
        }
        if n_samples < self.folds {
            return Err(SmeltError::InvalidParameter(format!(
                "CrossValidation with {} folds requires at least {} samples, got {n_samples}",
                self.folds, self.folds
            )));
        }
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = StdRng::seed_from_u64(self.seed);
        indices.shuffle(&mut rng);

        let fold_size = n_samples / self.folds;
        Ok((0..self.folds)
            .map(|fold| {
                let test_start = fold * fold_size;
                let test_end = if fold == self.folds - 1 {
                    n_samples
                } else {
                    test_start + fold_size
                };
                let test: Vec<usize> = indices[test_start..test_end].to_vec();
                let train: Vec<usize> = indices[..test_start]
                    .iter()
                    .chain(indices[test_end..].iter())
                    .copied()
                    .collect();
                (train, test)
            })
            .collect())
    }
}

/// Simple train/test split.
pub struct Holdout {
    /// Fraction of samples assigned to the training set, in (0, 1).
    pub ratio: f64,
    /// RNG seed used to shuffle samples before splitting.
    pub seed: u64,
}

impl Default for Holdout {
    fn default() -> Self {
        Self {
            ratio: 0.8,
            seed: 42,
        }
    }
}

impl Holdout {
    /// Create a train/test split with the given training-set ratio.
    pub fn new(ratio: f64) -> Self {
        Self { ratio, seed: 42 }
    }
    /// Set the RNG seed used to shuffle samples before splitting.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Resample for Holdout {
    fn splits(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if !(self.ratio > 0.0 && self.ratio < 1.0) {
            return Err(SmeltError::InvalidParameter(format!(
                "Holdout ratio must be in (0, 1), got {}",
                self.ratio
            )));
        }
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = StdRng::seed_from_u64(self.seed);
        indices.shuffle(&mut rng);

        let split = (n_samples as f64 * self.ratio) as usize;
        let train = indices[..split].to_vec();
        let test = indices[split..].to_vec();
        Ok(vec![(train, test)])
    }
}

/// Repeated K-fold cross-validation.
///
/// Runs plain [`CrossValidation`] `repeats` times, each with a *different*
/// shuffle seed, and concatenates the results — so a call yields
/// `folds * repeats` train/test splits. Averaging a measure over all of them
/// reduces the variance of the estimate that comes from a single arbitrary
/// fold assignment; it is scikit-learn's `RepeatedKFold` and mlr3's
/// `"repeated_cv"`. Each repeat is a full partition of the data (every sample
/// tested exactly once per repeat), unlike [`Bootstrap`]'s with-replacement
/// draws.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
///
/// let cv = RepeatedCV::new(5, 3); // 5 folds, 3 repeats
/// let splits = cv.splits(50).unwrap();
/// assert_eq!(splits.len(), 15); // folds * repeats
/// ```
pub struct RepeatedCV {
    /// Number of folds per repeat.
    pub folds: usize,
    /// How many times the whole K-fold partition is redrawn.
    pub repeats: usize,
    /// Base RNG seed; repeat `r` shuffles with a seed derived from it.
    pub seed: u64,
}

impl RepeatedCV {
    /// Create a repeated K-fold CV with the given folds and repeats.
    pub fn new(folds: usize, repeats: usize) -> Self {
        Self {
            folds,
            repeats,
            seed: 42,
        }
    }
    /// Set the base RNG seed (each repeat derives its own seed from it).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Resample for RepeatedCV {
    fn splits(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if self.repeats < 1 {
            return Err(SmeltError::InvalidParameter(format!(
                "RepeatedCV requires at least 1 repeat, got {}",
                self.repeats
            )));
        }
        let mut all = Vec::with_capacity(self.folds.saturating_mul(self.repeats));
        for r in 0..self.repeats {
            // Distinct per-repeat seeds so the partitions actually differ;
            // `wrapping_add` keeps them derived from `seed` yet reproducible.
            let cv = CrossValidation {
                folds: self.folds,
                seed: self.seed.wrapping_add(r as u64),
            };
            all.extend(cv.splits(n_samples)?);
        }
        Ok(all)
    }
}

/// Leave-one-out cross-validation (LOOCV).
///
/// The `n`-fold limit of [`CrossValidation`]: each of the `n` samples is the
/// sole test point of one split, trained on the other `n - 1`. Deterministic
/// — there is nothing to shuffle, so no seed. Nearly unbiased but expensive
/// (`n` model fits) and high-variance; mainly for small datasets. This is
/// scikit-learn's `LeaveOneOut` / mlr3's `"loo"`.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
///
/// let splits = LeaveOneOut.splits(4).unwrap();
/// assert_eq!(splits.len(), 4);
/// assert_eq!(splits[0].1, vec![0]); // test is a single held-out sample
/// assert_eq!(splits[0].0, vec![1, 2, 3]); // train is everything else
/// ```
pub struct LeaveOneOut;

impl Resample for LeaveOneOut {
    fn splits(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if n_samples < 2 {
            return Err(SmeltError::InvalidParameter(format!(
                "LeaveOneOut requires at least 2 samples, got {n_samples}"
            )));
        }
        Ok((0..n_samples)
            .map(|test_idx| {
                let train: Vec<usize> = (0..n_samples).filter(|&j| j != test_idx).collect();
                (train, vec![test_idx])
            })
            .collect())
    }
}

/// Bootstrap resampling (out-of-bag validation).
///
/// Each resample draws `n_samples` training indices **with replacement**
/// (so the train set has duplicates and is the same size as the data); the
/// samples never drawn — the "out-of-bag" (OOB) set, ~36.8% of the data on
/// average — become the test set. This is the resampling behind bagging's
/// OOB error and mlr3's `"bootstrap"` / scikit-learn's `resample`.
///
/// Draws whose OOB set happens to be empty are skipped (their probability,
/// `(1 - 1/n)^n`, is non-negligible for tiny `n` — e.g. ~25% at `n = 2`),
/// so exactly `n_resamples` usable splits are returned as long as
/// `n_samples >= 2`. A single continuous RNG stream keeps the whole thing
/// reproducible from `seed`.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
///
/// let boot = Bootstrap::new(10).with_seed(7);
/// let splits = boot.splits(100).unwrap();
/// assert_eq!(splits.len(), 10);
/// let (train, test) = &splits[0];
/// assert_eq!(train.len(), 100); // resampled to original size, with dups
/// assert!(!test.is_empty()); // the out-of-bag remainder
/// ```
pub struct Bootstrap {
    /// Number of bootstrap resamples (train/OOB-test pairs) to generate.
    pub n_resamples: usize,
    /// RNG seed for the (single, continuous) resampling stream.
    pub seed: u64,
}

impl Bootstrap {
    /// Create a bootstrap resampler generating `n_resamples` train/OOB pairs.
    pub fn new(n_resamples: usize) -> Self {
        Self {
            n_resamples,
            seed: 42,
        }
    }
    /// Set the RNG seed for the resampling stream.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Resample for Bootstrap {
    fn splits(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if self.n_resamples < 1 {
            return Err(SmeltError::InvalidParameter(format!(
                "Bootstrap requires at least 1 resample, got {}",
                self.n_resamples
            )));
        }
        if n_samples < 2 {
            // At n = 1 the OOB set is always empty: no valid test set exists.
            return Err(SmeltError::InvalidParameter(format!(
                "Bootstrap requires at least 2 samples, got {n_samples}"
            )));
        }
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut splits = Vec::with_capacity(self.n_resamples);
        while splits.len() < self.n_resamples {
            let train: Vec<usize> = (0..n_samples)
                .map(|_| rng.random_range(0..n_samples))
                .collect();
            let mut in_bag = vec![false; n_samples];
            for &i in &train {
                in_bag[i] = true;
            }
            let test: Vec<usize> = (0..n_samples).filter(|&i| !in_bag[i]).collect();
            if !test.is_empty() {
                splits.push((train, test));
            }
        }
        Ok(splits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn repeated_cv_yields_folds_times_repeats_splits() {
        let cv = RepeatedCV::new(5, 3);
        let splits = cv.splits(50).unwrap();
        assert_eq!(splits.len(), 15);
        // Each split is a valid train/test partition covering all indices.
        for (train, test) in &splits {
            assert_eq!(train.len() + test.len(), 50);
            let mut all: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
            all.sort_unstable();
            assert_eq!(all, (0..50).collect::<Vec<_>>());
        }
    }

    #[test]
    fn repeated_cv_repeats_actually_differ() {
        // The point of repeating is a *different* partition each time; with 3
        // repeats of 2-fold CV, the first fold's test set must not be
        // identical across all repeats.
        let cv = RepeatedCV::new(2, 3);
        let splits = cv.splits(20).unwrap();
        let rep0: HashSet<usize> = splits[0].1.iter().copied().collect();
        let rep1: HashSet<usize> = splits[2].1.iter().copied().collect(); // fold 0 of repeat 1
        assert_ne!(rep0, rep1, "repeats produced identical partitions");
    }

    #[test]
    fn repeated_cv_is_reproducible_and_seedable() {
        let a = RepeatedCV::new(4, 2).with_seed(7).splits(24).unwrap();
        let b = RepeatedCV::new(4, 2).with_seed(7).splits(24).unwrap();
        assert_eq!(a, b);
        let c = RepeatedCV::new(4, 2).with_seed(8).splits(24).unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn repeated_cv_rejects_zero_repeats() {
        assert!(RepeatedCV::new(5, 0).splits(50).is_err());
    }

    #[test]
    fn loo_holds_out_one_sample_per_split() {
        let splits = LeaveOneOut.splits(5).unwrap();
        assert_eq!(splits.len(), 5);
        // Union of the single test indices is exactly every sample, once.
        let mut tested: Vec<usize> = splits.iter().map(|(_, t)| t[0]).collect();
        tested.sort_unstable();
        assert_eq!(tested, (0..5).collect::<Vec<_>>());
        for (train, test) in &splits {
            assert_eq!(test.len(), 1);
            assert_eq!(train.len(), 4);
            assert!(!train.contains(&test[0]));
        }
    }

    #[test]
    fn loo_rejects_too_few_samples() {
        assert!(LeaveOneOut.splits(1).is_err());
        assert!(LeaveOneOut.splits(0).is_err());
    }

    #[test]
    fn bootstrap_train_is_full_size_with_replacement_and_test_is_oob() {
        let splits = Bootstrap::new(20).with_seed(1).splits(50).unwrap();
        assert_eq!(splits.len(), 20);
        for (train, test) in &splits {
            assert_eq!(train.len(), 50, "resampled to original size");
            assert!(!test.is_empty(), "OOB test never empty");
            // train and test are disjoint: OOB is exactly the not-drawn set.
            let in_bag: HashSet<usize> = train.iter().copied().collect();
            for &t in test {
                assert!(!in_bag.contains(&t));
            }
            // every index is either in-bag or OOB, none lost.
            assert_eq!(in_bag.len() + test.len(), 50);
        }
    }

    #[test]
    fn bootstrap_actually_draws_duplicates() {
        // With replacement, a full-size draw over 50 samples is essentially
        // certain to repeat at least one index.
        let splits = Bootstrap::new(1).with_seed(3).splits(50).unwrap();
        let train = &splits[0].0;
        let unique: HashSet<usize> = train.iter().copied().collect();
        assert!(unique.len() < train.len(), "expected duplicate draws");
    }

    #[test]
    fn bootstrap_is_reproducible_and_seedable() {
        let a = Bootstrap::new(5).with_seed(9).splits(30).unwrap();
        let b = Bootstrap::new(5).with_seed(9).splits(30).unwrap();
        assert_eq!(a, b);
        let c = Bootstrap::new(5).with_seed(10).splits(30).unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn bootstrap_skips_empty_oob_at_tiny_n() {
        // At n = 2, ~25% of draws have empty OOB; the skip logic must still
        // deliver exactly the requested count with non-empty test sets.
        let splits = Bootstrap::new(30).with_seed(5).splits(2).unwrap();
        assert_eq!(splits.len(), 30);
        assert!(splits.iter().all(|(_, test)| !test.is_empty()));
    }

    #[test]
    fn bootstrap_rejects_degenerate_configs() {
        assert!(Bootstrap::new(0).splits(50).is_err());
        assert!(Bootstrap::new(10).splits(1).is_err());
    }

    #[test]
    fn all_three_compose_as_trait_objects() {
        let resamplers: Vec<Box<dyn Resample>> = vec![
            Box::new(RepeatedCV::new(3, 2)),
            Box::new(LeaveOneOut),
            Box::new(Bootstrap::new(4)),
        ];
        for r in &resamplers {
            assert!(!r.splits(12).unwrap().is_empty());
        }
    }
}
