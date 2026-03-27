//! Resampling strategies: cross-validation, holdout, bootstrap.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Trait for resampling strategies.
pub trait Resample {
    /// Generate train/test index splits.
    fn splits(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)>;
}

/// K-fold cross-validation.
pub struct CrossValidation {
    pub folds: usize,
    pub seed: u64,
}

impl Default for CrossValidation {
    fn default() -> Self { Self { folds: 5, seed: 42 } }
}

impl CrossValidation {
    pub fn new(folds: usize) -> Self { Self { folds, seed: 42 } }
    pub fn with_seed(mut self, seed: u64) -> Self { self.seed = seed; self }
}

impl Resample for CrossValidation {
    fn splits(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = StdRng::seed_from_u64(self.seed);
        indices.shuffle(&mut rng);

        let fold_size = n_samples / self.folds;
        (0..self.folds).map(|fold| {
            let test_start = fold * fold_size;
            let test_end = if fold == self.folds - 1 { n_samples } else { test_start + fold_size };
            let test: Vec<usize> = indices[test_start..test_end].to_vec();
            let train: Vec<usize> = indices[..test_start].iter()
                .chain(indices[test_end..].iter())
                .copied()
                .collect();
            (train, test)
        }).collect()
    }
}

/// Simple train/test split.
pub struct Holdout {
    pub ratio: f64,
    pub seed: u64,
}

impl Default for Holdout {
    fn default() -> Self { Self { ratio: 0.8, seed: 42 } }
}

impl Holdout {
    pub fn new(ratio: f64) -> Self { Self { ratio, seed: 42 } }
    pub fn with_seed(mut self, seed: u64) -> Self { self.seed = seed; self }
}

impl Resample for Holdout {
    fn splits(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = StdRng::seed_from_u64(self.seed);
        indices.shuffle(&mut rng);

        let split = (n_samples as f64 * self.ratio) as usize;
        let train = indices[..split].to_vec();
        let test = indices[split..].to_vec();
        vec![(train, test)]
    }
}
