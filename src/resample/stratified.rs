//! Label-aware and group-aware cross-validation.
//!
//! Plain [`crate::resample::CrossValidation`] shuffles all indices together,
//! so a fold can easily end up missing a minority class entirely (fatal for
//! imbalanced classification, e.g. landslide/no-landslide or presence/
//! absence datasets) or splitting a spatial/temporal group (e.g. all
//! measurements from one field site or one survey year) across train and
//! test, leaking information.

use super::Resample;
use crate::{Result, SmeltError};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// Stratified k-fold cross-validation: each fold has (approximately) the
/// same class proportions as the full dataset, computed by distributing
/// each class's own shuffled indices round-robin across the folds
/// independently (the standard approach, matching scikit-learn's
/// `StratifiedKFold`).
pub struct StratifiedCV {
    n_folds: usize,
    labels: Vec<usize>,
    seed: u64,
}

impl StratifiedCV {
    /// Create a stratified K-fold CV that preserves class proportions of
    /// `labels` in every fold.
    pub fn new(n_folds: usize, labels: Vec<usize>) -> Self {
        Self { n_folds, labels, seed: 42 }
    }

    /// Set the RNG seed used to shuffle within each class before
    /// distributing round-robin across folds.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Resample for StratifiedCV {
    fn splits(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if self.n_folds < 2 {
            return Err(SmeltError::InvalidParameter(format!(
                "StratifiedCV requires at least 2 folds, got {}",
                self.n_folds
            )));
        }
        if self.labels.len() != n_samples {
            return Err(SmeltError::DimensionMismatch {
                expected: n_samples,
                got: self.labels.len(),
            });
        }

        // Group sample indices by class, in deterministic class order.
        let mut by_class: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &label) in self.labels.iter().enumerate() {
            by_class.entry(label).or_default().push(i);
        }
        let mut classes: Vec<usize> = by_class.keys().copied().collect();
        classes.sort_unstable();

        for &class in &classes {
            let count = by_class[&class].len();
            if count < self.n_folds {
                return Err(SmeltError::InvalidParameter(format!(
                    "StratifiedCV: class {class} has only {count} samples, \
                     fewer than n_folds={}; reduce n_folds or merge rare classes",
                    self.n_folds
                )));
            }
        }

        // Shuffle within each class, then distribute round-robin: fold
        // assignment of the k-th sample of a class is k % n_folds. Since
        // this is done independently per class, every fold gets ~1/n_folds
        // of every class, preserving the overall class balance.
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut fold_of = vec![0usize; n_samples];
        for &class in &classes {
            let mut idx = by_class[&class].clone();
            idx.shuffle(&mut rng);
            for (k, &sample) in idx.iter().enumerate() {
                fold_of[sample] = k % self.n_folds;
            }
        }

        Ok((0..self.n_folds)
            .map(|fold| {
                let test: Vec<usize> = (0..n_samples).filter(|&i| fold_of[i] == fold).collect();
                let train: Vec<usize> = (0..n_samples).filter(|&i| fold_of[i] != fold).collect();
                (train, test)
            })
            .collect())
    }
}

/// Group k-fold cross-validation: every sample belonging to the same group
/// id is always in the same fold, so a fold split never separates two
/// samples from the same group into train and test. Groups (not samples)
/// are distributed round-robin across folds.
pub struct GroupCV {
    n_folds: usize,
    groups: Vec<usize>,
    seed: u64,
}

impl GroupCV {
    /// Create a group K-fold CV that keeps every sample sharing a group id
    /// in `groups` together in the same fold.
    pub fn new(n_folds: usize, groups: Vec<usize>) -> Self {
        Self { n_folds, groups, seed: 42 }
    }

    /// Set the RNG seed used to shuffle groups before distributing
    /// round-robin across folds.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Resample for GroupCV {
    fn splits(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if self.n_folds < 2 {
            return Err(SmeltError::InvalidParameter(format!(
                "GroupCV requires at least 2 folds, got {}",
                self.n_folds
            )));
        }
        if self.groups.len() != n_samples {
            return Err(SmeltError::DimensionMismatch {
                expected: n_samples,
                got: self.groups.len(),
            });
        }

        let mut unique_groups: Vec<usize> = {
            let mut g = self.groups.clone();
            g.sort_unstable();
            g.dedup();
            g
        };
        if unique_groups.len() < self.n_folds {
            return Err(SmeltError::InvalidParameter(format!(
                "GroupCV: only {} distinct groups, fewer than n_folds={}",
                unique_groups.len(),
                self.n_folds
            )));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        unique_groups.shuffle(&mut rng);
        let fold_of_group: HashMap<usize, usize> = unique_groups
            .iter()
            .enumerate()
            .map(|(k, &g)| (g, k % self.n_folds))
            .collect();

        Ok((0..self.n_folds)
            .map(|fold| {
                let test: Vec<usize> = (0..n_samples)
                    .filter(|&i| fold_of_group[&self.groups[i]] == fold)
                    .collect();
                let train: Vec<usize> = (0..n_samples)
                    .filter(|&i| fold_of_group[&self.groups[i]] != fold)
                    .collect();
                (train, test)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stratified_cv_preserves_class_balance_per_fold() {
        // 100 samples: 80 class 0, 20 class 1 (4:1 imbalance).
        let mut labels = vec![0usize; 80];
        labels.extend(vec![1usize; 20]);
        let cv = StratifiedCV::new(5, labels.clone());
        let splits = cv.splits(100).unwrap();

        assert_eq!(splits.len(), 5);
        for (train, test) in &splits {
            let test_pos = test.iter().filter(|&&i| labels[i] == 1).count();
            let test_neg = test.len() - test_pos;
            // Each fold should have ~4 positives and ~16 negatives (20/5, 80/5).
            assert!(
                (2..=6).contains(&test_pos),
                "fold should have ~4 positive-class samples, got {test_pos}"
            );
            assert!(test_neg > 0, "fold must not be missing the majority class");
            // No overlap between train and test.
            for &i in test {
                assert!(!train.contains(&i));
            }
        }
    }

    #[test]
    fn stratified_cv_rejects_class_smaller_than_n_folds() {
        let mut labels = vec![0usize; 50];
        labels.extend(vec![1usize; 2]); // only 2 samples of class 1
        let cv = StratifiedCV::new(5, labels);
        assert!(cv.splits(52).is_err());
    }

    #[test]
    fn group_cv_never_splits_a_group_across_train_and_test() {
        // 30 samples, 6 groups of 5 samples each.
        let groups: Vec<usize> = (0..30).map(|i| i / 5).collect();
        let cv = GroupCV::new(3, groups.clone());
        let splits = cv.splits(30).unwrap();

        assert_eq!(splits.len(), 3);
        for (train, test) in &splits {
            let train_groups: std::collections::HashSet<usize> =
                train.iter().map(|&i| groups[i]).collect();
            let test_groups: std::collections::HashSet<usize> =
                test.iter().map(|&i| groups[i]).collect();
            assert!(
                train_groups.is_disjoint(&test_groups),
                "a group must not appear in both train and test: train={train_groups:?} test={test_groups:?}"
            );
        }
    }

    #[test]
    fn group_cv_rejects_fewer_groups_than_folds() {
        let groups = vec![0usize, 0, 1, 1]; // only 2 distinct groups
        let cv = GroupCV::new(5, groups);
        assert!(cv.splits(4).is_err());
    }
}
