//! ADASYN: Adaptive Synthetic Sampling.
//!
//! Like SMOTE but generates more synthetic samples in regions where the
//! minority class is harder to learn (higher density of majority neighbors).
//!
//! Reference: He, H. et al. (2008). ADASYN. IJCNN, 1322-1328. (4,070 citations)

use super::Resampler;
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
    /// Create an ADASYN oversampler with default parameters.
    pub fn new() -> Self {
        Self::default()
    }
    /// Set the number of nearest neighbors used both to estimate density
    /// ratios and to synthesize new minority-class samples.
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k;
        self
    }
    /// Set the RNG seed for reproducible synthetic sample generation.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Rebalance `task` by adaptively generating synthetic minority-class
    /// samples, concentrated where the minority class is hardest to learn.
    pub fn balance(&self, task: &ClassificationTask) -> Result<ClassificationTask> {
        // k=0 would divide the density ratio r_i = majority_neighbors/k by
        // zero: every ratio becomes NaN, the NaN-poisoned normalization
        // yields per-sample counts of 0, and balance() "succeeds" while
        // generating nothing.
        if self.k_neighbors == 0 {
            return Err(crate::SmeltError::InvalidParameter(
                "Adasyn k_neighbors must be >= 1, got 0".into(),
            ));
        }
        // A weighted task cannot be resampled: a synthetic sample is an
        // interpolation of two real ones, and there is no principled weight
        // for it — any silent choice would corrupt the caller's weighting
        // scheme (same guard as Smote/SpatialSmote).
        if task.weights().is_some() {
            return Err(crate::SmeltError::InvalidParameter(
                "resampling a weighted task is not supported; the synthetic samples' \
                 weights are undefined — remove with_weights() before ADASYN"
                    .into(),
            ));
        }
        let features = task.features();
        let target = task.target();
        let n_classes = task.n_classes();
        let n_features = task.n_features();
        // NaN features poison the k-NN distances (arbitrary neighbour order
        // via partial_cmp -> Equal) and get interpolated INTO the synthetic
        // rows -- all silently (audit M-5). In a Pipeline the resampler runs
        // BEFORE any Imputer stage by design, so this must be an error, not
        // a footgun: impute before resampling.
        crate::validate::check_no_nan(features).map_err(|_| {
            crate::SmeltError::InvalidParameter(
                "resampling requires NaN-free features: SMOTE/ADASYN interpolate between \
                 k-nearest neighbours, and NaN corrupts both the distances and the synthetic \
                 samples -- impute missing values BEFORE resampling (note: a Pipeline's \
                 resampler stage runs before its transformers, so put the Imputer in its \
                 own earlier step)"
                    .into(),
            )
        })?;
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

            let interp_k = self.k_neighbors.min(n_class - 1).max(1);

            for (ci, &idx) in class_indices.iter().enumerate() {
                let n_gen = samples_per_point[ci];
                let sample = features.row(idx);

                // Find the k nearest same-class neighbors (not just any
                // same-class point, per He et al. 2008): interpolating
                // toward an arbitrary same-class point can cross into
                // majority-class regions when the minority class spans a
                // large or multi-modal area of feature space.
                let mut same_class_dists: Vec<(usize, f64)> = class_indices
                    .iter()
                    .filter(|&&j| j != idx)
                    .map(|&j| {
                        let d: f64 = features
                            .row(j)
                            .iter()
                            .zip(sample.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        (j, d)
                    })
                    .collect();
                same_class_dists
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                for _ in 0..n_gen {
                    let synthetic = if same_class_dists.is_empty() {
                        sample
                            .iter()
                            .map(|&s| s + rng.random_range(-0.01..0.01))
                            .collect()
                    } else {
                        let k_nn = interp_k.min(same_class_dists.len());
                        let nn = same_class_dists[rng.random_range(0..k_nn)].0;
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

        // Keep the input task's metadata (names/types/class width) -- same
        // propagation as Smote::balance (audit HIGH-4/M-4).
        ClassificationTask::new(task.id(), result, new_target)?
            .with_feature_names(task.feature_names().to_vec())?
            .with_feature_types(task.feature_types().to_vec())
            .map(|t| t.with_class_names(task.class_names().to_vec()))
    }
}

impl Resampler for Adasyn {
    fn id(&self) -> &str {
        "adasyn"
    }

    fn resample(&self, task: &ClassificationTask) -> Result<ClassificationTask> {
        self.balance(task)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4th-audit LOW: k_neighbors=0 used to divide the density ratio by
    /// zero -- NaN ratios, zero synthetic samples, and a successful return.
    #[test]
    fn rejects_zero_k_neighbors() {
        let features = ndarray::array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 0.9]];
        let target = vec![0, 0, 0, 1];
        let task = ClassificationTask::new("k0", features, target).unwrap();
        let err = Adasyn::new()
            .with_k_neighbors(0)
            .balance(&task)
            .unwrap_err();
        assert!(err.to_string().contains("k_neighbors"), "got: {err}");
    }

    /// Regression test for HIGH-12 in the audit: interpolation used to pick
    /// ANY same-class point, not one of the k nearest same-class neighbors
    /// (as He et al. 2008 requires). With a bimodal minority class (two
    /// tight, far-apart clusters) and majority samples occupying the region
    /// between them, the old code could pair a point in one cluster with a
    /// point in the other, generating a synthetic sample squarely in
    /// majority territory. Restricting to the k nearest neighbors keeps
    /// synthetics local to whichever cluster they came from.
    #[test]
    fn synthetic_samples_stay_within_local_cluster_not_across_the_gap() {
        let mut feats: Vec<Vec<f64>> = Vec::new();
        let mut target: Vec<usize> = Vec::new();
        // Minority cluster A, near (0, 0).
        for i in 0..6 {
            feats.push(vec![i as f64 * 0.01, i as f64 * 0.01]);
            target.push(1);
        }
        // Minority cluster B, near (10, 10) -- far from cluster A.
        for i in 0..6 {
            feats.push(vec![10.0 + i as f64 * 0.01, 10.0 + i as f64 * 0.01]);
            target.push(1);
        }
        // Majority samples fill the gap between the two minority clusters.
        for i in 0..30 {
            let x = 4.0 + (i as f64 * 0.13) % 3.0;
            let y = 4.0 + (i as f64 * 0.17) % 3.0;
            feats.push(vec![x, y]);
            target.push(0);
        }
        let n_original = feats.len();
        let flat: Vec<f64> = feats.into_iter().flatten().collect();
        let features = Array2::from_shape_vec((n_original, 2), flat).unwrap();
        let task = ClassificationTask::new("t", features, target).unwrap();

        let adasyn = Adasyn::new().with_k_neighbors(2).with_seed(0);
        let balanced = adasyn.balance(&task).unwrap();
        let feats_out = balanced.features();
        let target_out = balanced.target();

        let mut checked_any = false;
        for i in n_original..target_out.len() {
            if target_out[i] != 1 {
                continue;
            }
            checked_any = true;
            let x = feats_out[[i, 0]];
            let y = feats_out[[i, 1]];
            let dist_a = (x * x + y * y).sqrt();
            let dist_b = ((x - 10.0).powi(2) + (y - 10.0).powi(2)).sqrt();
            assert!(
                dist_a.min(dist_b) < 3.0,
                "synthetic minority sample ({x}, {y}) should stay near cluster A or B \
                 (dist_a={dist_a}, dist_b={dist_b}), not cross into the majority-occupied gap"
            );
        }
        assert!(
            checked_any,
            "test should have generated at least one synthetic minority sample"
        );
    }
}
