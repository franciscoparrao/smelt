//! Resampling strategies: cross-validation, holdout, bootstrap, spatial,
//! stratified, group.

pub mod spatial;
pub mod stratified;

use crate::{Result, SmeltError};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

pub use spatial::{SpatialBlockCV, SpatialBufferCV};
pub use stratified::{GroupCV, StratifiedCV};

/// Trait for resampling strategies.
pub trait Resample {
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
