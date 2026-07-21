//! Isolation Forest: anomaly detection via random partitioning.
//!
//! Anomalies are isolated in fewer random splits than normal points.
//! Score s(x,n) = 2^{-E[h(x)]/c(n)} where h(x) = average path length.
//!
//! Reference: Liu, F., Ting, K., & Zhou, Z. (2008).
//! Isolation Forest. ICDM, 413-422.

use crate::Result;
use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Isolation Forest for anomaly/outlier detection.
///
/// Builds an ensemble of random isolation trees. Anomalies have
/// shorter average path lengths and higher anomaly scores.
///
/// # Examples
///
/// ```
/// use smelt_ml::cluster::IsolationForest;
/// use ndarray::array;
///
/// let data = array![
///     [1.0, 1.0], [1.1, 0.9], [0.9, 1.1], [1.0, 1.0],
///     [10.0, 10.0],  // outlier
/// ];
///
/// let iforest = IsolationForest::new()
///     .with_n_estimators(100)
///     .with_contamination(0.1);
/// let result = iforest.fit_predict(&data).unwrap();
///
/// // Outlier should have highest anomaly score
/// assert!(result.scores[4] > result.scores[0]);
/// ```
pub struct IsolationForest {
    n_estimators: usize,
    max_samples: Option<usize>,
    contamination: f64,
    seed: u64,
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_samples: None, // auto: min(256, n_samples)
            contamination: 0.1,
            seed: 42,
        }
    }
}

impl IsolationForest {
    /// Creates an isolation forest with default hyperparameters (100 trees,
    /// auto subsample size, 10% contamination, seed 42).
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the number of isolation trees in the ensemble.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    /// Sets the subsample size used to build each isolation tree.
    pub fn with_max_samples(mut self, n: usize) -> Self {
        self.max_samples = Some(n);
        self
    }
    /// Sets the expected fraction of anomalies, used to derive the score threshold.
    pub fn with_contamination(mut self, c: f64) -> Self {
        self.contamination = c;
        self
    }
    /// Sets the RNG seed used for subsampling and random splits.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Fit the forest and compute anomaly scores + labels.
    pub fn fit_predict(&self, features: &Array2<f64>) -> Result<AnomalyResult> {
        let n_samples = features.nrows();
        let n_features = features.ncols();
        let sub_size = self.max_samples.unwrap_or(n_samples.min(256));
        let mut rng = StdRng::seed_from_u64(self.seed);

        let max_depth = (sub_size as f64).log2().ceil() as usize + 2;

        // Build isolation trees
        let mut trees = Vec::with_capacity(self.n_estimators);
        for _ in 0..self.n_estimators {
            // Subsample
            let indices: Vec<usize> = (0..sub_size)
                .map(|_| rng.random_range(0..n_samples))
                .collect();
            let tree = build_itree(features, &indices, 0, max_depth, n_features, &mut rng);
            trees.push(tree);
        }

        // Compute anomaly scores
        let c_n = c_factor(sub_size);
        let mut scores = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let avg_path: f64 = trees
                .iter()
                .map(|tree| path_length(tree, features, i, 0))
                .sum::<f64>()
                / self.n_estimators as f64;

            // s(x,n) = 2^{-E[h(x)]/c(n)}
            let score = 2.0f64.powf(-avg_path / c_n);
            scores.push(score);
        }

        // Determine threshold from contamination
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold_idx = (n_samples as f64 * self.contamination).ceil() as usize;
        let threshold = if threshold_idx > 0 && threshold_idx <= n_samples {
            sorted_scores[threshold_idx - 1]
        } else {
            0.5
        };

        // Label: 1 = anomaly, 0 = normal
        let labels: Vec<i32> = scores
            .iter()
            .map(|&s| if s >= threshold { 1 } else { 0 })
            .collect();

        let n_anomalies = labels.iter().filter(|&&l| l == 1).count();

        Ok(AnomalyResult {
            scores,
            labels,
            threshold,
            n_anomalies,
        })
    }
}

/// Result of anomaly detection.
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Anomaly score per sample. Range [0, 1]. Higher = more anomalous.
    pub scores: Vec<f64>,
    /// Labels: 1 = anomaly, 0 = normal.
    pub labels: Vec<i32>,
    /// Score threshold used for labeling.
    pub threshold: f64,
    /// Number of detected anomalies.
    pub n_anomalies: usize,
}

// ── Isolation tree internals ────────────────────────────────────────

enum INode {
    Leaf {
        size: usize,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<INode>,
        right: Box<INode>,
    },
}

fn build_itree(
    features: &Array2<f64>,
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    n_features: usize,
    rng: &mut StdRng,
) -> INode {
    let n = indices.len();

    if n <= 1 || depth >= max_depth {
        return INode::Leaf { size: n };
    }

    // Random feature and random threshold
    let feat = rng.random_range(0..n_features);

    let min_val = indices
        .iter()
        .map(|&i| features[[i, feat]])
        .fold(f64::INFINITY, f64::min);
    let max_val = indices
        .iter()
        .map(|&i| features[[i, feat]])
        .fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < f64::EPSILON {
        return INode::Leaf { size: n };
    }

    let threshold = rng.random_range(min_val..max_val);

    let left_idx: Vec<usize> = indices
        .iter()
        .filter(|&&i| features[[i, feat]] < threshold)
        .copied()
        .collect();
    let right_idx: Vec<usize> = indices
        .iter()
        .filter(|&&i| features[[i, feat]] >= threshold)
        .copied()
        .collect();

    if left_idx.is_empty() || right_idx.is_empty() {
        return INode::Leaf { size: n };
    }

    let left = build_itree(features, &left_idx, depth + 1, max_depth, n_features, rng);
    let right = build_itree(features, &right_idx, depth + 1, max_depth, n_features, rng);

    INode::Split {
        feature: feat,
        threshold,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn path_length(node: &INode, features: &Array2<f64>, sample: usize, depth: usize) -> f64 {
    match node {
        INode::Leaf { size } => depth as f64 + c_factor(*size),
        INode::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            if features[[sample, *feature]] < *threshold {
                path_length(left, features, sample, depth + 1)
            } else {
                path_length(right, features, sample, depth + 1)
            }
        }
    }
}

/// Average path length of unsuccessful search in BST (normalization factor).
/// c(n) = 2*H(n-1) - 2*(n-1)/n where H(i) = ln(i) + 0.5772 (Euler constant).
/// Liu, Ting & Zhou (2008), Eq. 2. `n == 2` is special-cased to `1.0`
/// (matches scikit-learn's `_average_path_length`), since the general
/// formula's `ln(n-1) = ln(1) = 0` term underestimates the two-point case.
fn c_factor(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    if n == 2 {
        return 1.0;
    }
    let n = n as f64;
    2.0 * ((n - 1.0).ln() + 0.5772156649) - 2.0 * (n - 1.0) / n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the missing-parenthesis bug: `2*ln(n-1) + gamma`
    /// (wrong) vs `2*(ln(n-1) + gamma)` (correct) — applying gamma only once
    /// instead of twice made c(n) negative for small n and ~5% low at n=256.
    /// Reference values from scikit-learn's `_average_path_length`.
    #[test]
    fn c_factor_matches_reference_values() {
        assert_eq!(c_factor(0), 0.0);
        assert_eq!(c_factor(1), 0.0);
        assert_eq!(c_factor(2), 1.0);
        assert!(
            (c_factor(10) - 3.7488806).abs() < 1e-5,
            "c(10) = {}",
            c_factor(10)
        );
        assert!(
            (c_factor(256) - 10.244770).abs() < 1e-5,
            "c(256) = {}",
            c_factor(256)
        );
        // Must never be negative for any n >= 1 (the old bug gave -0.42 at n=2).
        for n in 1..1000 {
            assert!(
                c_factor(n) >= 0.0,
                "c_factor({n}) = {} is negative",
                c_factor(n)
            );
        }
    }

    #[test]
    fn outlier_gets_higher_score_than_inliers() {
        let mut data = Vec::new();
        for i in 0..50 {
            data.push((i as f64 * 0.01) % 1.0);
            data.push(((i as f64 * 0.017) % 1.0) + 1.0);
        }
        data.push(50.0);
        data.push(50.0); // clear outlier, far from the cluster
        let features = Array2::from_shape_vec((51, 2), data).unwrap();

        let iforest = IsolationForest::new().with_n_estimators(100).with_seed(42);
        let result = iforest.fit_predict(&features).unwrap();

        let outlier_score = result.scores[50];
        let mean_inlier_score: f64 = result.scores[..50].iter().sum::<f64>() / 50.0;
        assert!(
            outlier_score > mean_inlier_score,
            "outlier score ({outlier_score}) should exceed mean inlier score ({mean_inlier_score})"
        );
    }
}
