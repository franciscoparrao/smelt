//! ADASYN: Adaptive Synthetic Sampling.
//!
//! Like SMOTE but generates more synthetic samples in regions where the
//! minority class is harder to learn (higher density of majority neighbors).
//!
//! Reference: He, H. et al. (2008). ADASYN. IJCNN, 1322-1328. (4,070 citations)

use crate::Result;
use crate::task::{ClassificationTask, Task};
use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// ADASYN adaptive oversampler.
///
/// Generates more synthetic samples near decision boundaries where
/// the minority class is harder to classify.
///
/// # Examples
///
/// ```
/// use smelt_ml::preprocess::Adasyn;
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![
///     [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2], [0.1, 0.0],
///     [1.0, 1.0],  // minority
/// ];
/// let target = vec![0, 0, 0, 0, 0, 1];
/// let task = ClassificationTask::new("ada", features, target).unwrap();
///
/// let adasyn = Adasyn::new().with_seed(42);
/// let balanced = adasyn.balance(&task).unwrap();
/// ```
pub struct Adasyn {
    k_neighbors: usize,
    seed: u64,
}

impl Default for Adasyn {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            seed: 42,
        }
    }
}

impl Adasyn {
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

    pub fn balance(&self, task: &ClassificationTask) -> Result<ClassificationTask> {
        let features = task.features();
        let target = task.target();
        let n_classes = task.n_classes();
        let n_features = task.n_features();
        let n_samples = task.n_samples();

        let mut class_counts = vec![0usize; n_classes];
        for &t in target {
            class_counts[t] += 1;
        }
        let max_count = *class_counts.iter().max().unwrap();

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut new_features: Vec<Vec<f64>> =
            features.rows().into_iter().map(|r| r.to_vec()).collect();
        let mut new_target: Vec<usize> = target.to_vec();

        for class in 0..n_classes {
            let n_to_generate = max_count - class_counts[class];
            if n_to_generate == 0 {
                continue;
            }

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
            let k = self.k_neighbors.min(n_samples - 1);

            // Compute density ratio r_i for each minority sample:
            // r_i = (# majority neighbors in k-NN) / k
            let mut ratios = Vec::with_capacity(n_class);
            for &idx in &class_indices {
                let mut dists: Vec<(usize, f64)> = (0..n_samples)
                    .filter(|&j| j != idx)
                    .map(|j| {
                        let d: f64 = features
                            .row(idx)
                            .iter()
                            .zip(features.row(j).iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        (j, d)
                    })
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let majority_neighbors = dists
                    .iter()
                    .take(k)
                    .filter(|(j, _)| target[*j] != class)
                    .count();
                ratios.push(majority_neighbors as f64 / k as f64);
            }

            // Normalize ratios
            let ratio_sum: f64 = ratios.iter().sum();
            if ratio_sum < 1e-10 {
                // All neighbors are same class: uniform distribution
                for r in &mut ratios {
                    *r = 1.0 / n_class as f64;
                }
            } else {
                for r in &mut ratios {
                    *r /= ratio_sum;
                }
            }

            // Generate samples proportional to r_i
            let samples_per_point: Vec<usize> = ratios
                .iter()
                .map(|&r| (r * n_to_generate as f64).round() as usize)
                .collect();

            for (ci, &idx) in class_indices.iter().enumerate() {
                let n_gen = samples_per_point[ci];
                let sample = features.row(idx);

                // Find same-class neighbors
                let same_class: Vec<usize> = class_indices
                    .iter()
                    .filter(|&&j| j != idx)
                    .copied()
                    .collect();

                for _ in 0..n_gen {
                    let synthetic = if same_class.is_empty() {
                        sample
                            .iter()
                            .map(|&s| s + rng.random_range(-0.01..0.01))
                            .collect()
                    } else {
                        let nn = same_class[rng.random_range(0..same_class.len())];
                        let neighbor = features.row(nn);
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
        }

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
