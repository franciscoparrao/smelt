//! SMOTE: Synthetic Minority Over-sampling Technique.
//!
//! Generates synthetic samples for minority classes to address class imbalance.

use crate::Result;
use crate::task::{ClassificationTask, Task};
use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// SMOTE oversampler for handling class imbalance.
///
/// Generates synthetic samples by interpolating between minority class
/// instances and their k-nearest neighbors.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],  // class 0 (majority: 3)
///     [1.0, 1.0],                             // class 1 (minority: 1)
/// ];
/// let target = vec![0, 0, 0, 1];
/// let task = ClassificationTask::new("imb", features, target).unwrap();
///
/// let smote = Smote::new().with_seed(42);
/// let balanced = smote.balance(&task).unwrap();
/// assert!(balanced.n_samples() >= 4); // at least as many as before
/// ```
pub struct Smote {
    k_neighbors: usize,
    seed: u64,
}

impl Default for Smote {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            seed: 42,
        }
    }
}

impl Smote {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Balance the task by oversampling minority classes.
    /// Returns a new ClassificationTask with synthetic samples added.
    pub fn balance(&self, task: &ClassificationTask) -> Result<ClassificationTask> {
        let features = task.features();
        let target = task.target();
        let n_classes = task.n_classes();
        let n_features = task.n_features();

        // Count samples per class
        let mut class_counts = vec![0usize; n_classes];
        for &t in target {
            class_counts[t] += 1;
        }
        let max_count = *class_counts.iter().max().unwrap();

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut new_features: Vec<Vec<f64>> =
            features.rows().into_iter().map(|r| r.to_vec()).collect();
        let mut new_target: Vec<usize> = target.to_vec();

        // For each minority class, generate synthetic samples
        for class in 0..n_classes {
            let n_to_generate = max_count - class_counts[class];
            if n_to_generate == 0 {
                continue;
            }

            // Collect indices of this class
            let class_indices: Vec<usize> = target
                .iter()
                .enumerate()
                .filter(|&(_, &t)| t == class)
                .map(|(i, _)| i)
                .collect();

            let n_class = class_indices.len();
            if n_class == 0 {
                continue;
            }

            let k = self.k_neighbors.min(n_class - 1).max(1);

            for _ in 0..n_to_generate {
                // Pick a random minority sample
                let idx = class_indices[rng.random_range(0..n_class)];
                let sample = features.row(idx);

                // Find k nearest neighbors within the same class
                let mut dists: Vec<(usize, f64)> = class_indices
                    .iter()
                    .filter(|&&i| i != idx)
                    .map(|&i| {
                        let d: f64 = features
                            .row(i)
                            .iter()
                            .zip(sample.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        (i, d)
                    })
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let synthetic: Vec<f64> = if dists.is_empty() {
                    // Only 1 sample in class: duplicate with small noise
                    sample
                        .iter()
                        .map(|&s| s + rng.random_range(-0.01..0.01))
                        .collect()
                } else {
                    // Pick a random neighbor from k nearest
                    let nn_idx = dists[rng.random_range(0..k.min(dists.len()))].0;
                    let neighbor = features.row(nn_idx);
                    let lambda: f64 = rng.random_range(0.0..1.0);
                    sample
                        .iter()
                        .zip(neighbor.iter())
                        .map(|(&s, &n)| s + lambda * (n - s))
                        .collect()
                };

                new_features.push(synthetic);
                new_target.push(class);
            }
        }

        // Build new feature matrix
        let n_total = new_features.len();
        let mut result = Array2::zeros((n_total, n_features));
        for (i, row) in new_features.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        ClassificationTask::new(task.id(), result, new_target)
    }
}
