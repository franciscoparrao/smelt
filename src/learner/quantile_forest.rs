//! Quantile Regression Forest (QRF).
//!
//! Random Forest that stores all training targets in leaves to compute
//! any quantile at prediction time. Produces full conditional distributions.
//!
//! Reference: Meinshausen, N. (2006). Quantile Regression Forests. JMLR 7, 983-999.

use ndarray::Array2;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use crate::task::{ClassificationTask, RegressionTask, Task};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::{SmeltError, Result};

/// Quantile Regression Forest.
///
/// Unlike standard RF that predicts the mean, QRF stores all target values
/// in each leaf, enabling prediction of any quantile or prediction interval.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::QuantileForest;
/// use ndarray::array;
///
/// let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
/// let task = RegressionTask::new("qrf", features.clone(), target).unwrap();
///
/// let mut qrf = QuantileForest::new().with_n_estimators(50).with_seed(42);
/// let model = qrf.train_regress(&task).unwrap();
///
/// // Predict median
/// let pred = model.predict(&features).unwrap();
///
/// // Get prediction intervals (10th and 90th quantiles)
/// // Use the returned TrainedQuantileForest directly for quantile access
/// ```
pub struct QuantileForest {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    seed: u64,
}

impl Default for QuantileForest {
    fn default() -> Self {
        Self { n_estimators: 100, max_depth: None, min_samples_leaf: 5, seed: 42 }
    }
}

impl QuantileForest {
    pub fn new() -> Self { Self::default() }
    pub fn with_n_estimators(mut self, n: usize) -> Self { self.n_estimators = n; self }
    pub fn with_max_depth(mut self, d: usize) -> Self { self.max_depth = Some(d); self }
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self { self.min_samples_leaf = n; self }
    pub fn with_seed(mut self, s: u64) -> Self { self.seed = s; self }
}

// ── QRF Tree internals ─────────────────────────────────────────────

enum QRFNode {
    Leaf { values: Vec<f64> }, // all target values in this leaf
    Split {
        feature: usize,
        threshold: f64,
        left: Box<QRFNode>,
        right: Box<QRFNode>,
    },
}

impl QRFNode {
    fn find_leaf(&self, row: &[f64]) -> &[f64] {
        match self {
            QRFNode::Leaf { values } => values,
            QRFNode::Split { feature, threshold, left, right } => {
                if row[*feature] <= *threshold { left.find_leaf(row) }
                else { right.find_leaf(row) }
            }
        }
    }
}

fn build_qrf_tree(
    features: &Array2<f64>,
    target: &[f64],
    indices: &[usize],
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    n_features: usize,
    depth: usize,
    rng: &mut impl Rng,
) -> QRFNode {
    let n = indices.len();

    if n <= min_samples_leaf * 2
        || max_depth.is_some_and(|d| depth >= d)
    {
        let values: Vec<f64> = indices.iter().map(|&i| target[i]).collect();
        return QRFNode::Leaf { values };
    }

    // Random feature subset
    let n_try = ((n_features as f64).sqrt().ceil() as usize).max(1);
    let mut feat_indices: Vec<usize> = (0..n_features).collect();
    for i in 0..n_try.min(n_features) {
        let j = rng.random_range(i..n_features);
        feat_indices.swap(i, j);
    }

    let mut best_gain = 0.0;
    let mut best_split = None;

    // MSE-based splitting
    let parent_mse = mse_indices(target, indices);

    for &feat in &feat_indices[..n_try.min(n_features)] {
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| features[[a, feat]].partial_cmp(&features[[b, feat]])
            .unwrap_or(std::cmp::Ordering::Equal));

        for s in min_samples_leaf..(n.saturating_sub(min_samples_leaf)) {
            if (features[[sorted[s], feat]] - features[[sorted[s-1], feat]]).abs() < f64::EPSILON {
                continue;
            }
            let left = &sorted[..s];
            let right = &sorted[s..];
            let gain = parent_mse
                - (left.len() as f64 / n as f64) * mse_indices(target, left)
                - (right.len() as f64 / n as f64) * mse_indices(target, right);

            if gain > best_gain {
                best_gain = gain;
                let threshold = (features[[sorted[s-1], feat]] + features[[sorted[s], feat]]) / 2.0;
                best_split = Some((feat, threshold, left.to_vec(), right.to_vec()));
            }
        }
    }

    match best_split {
        Some((feat, threshold, left_idx, right_idx)) => {
            let left = build_qrf_tree(features, target, &left_idx, max_depth, min_samples_leaf, n_features, depth+1, rng);
            let right = build_qrf_tree(features, target, &right_idx, max_depth, min_samples_leaf, n_features, depth+1, rng);
            QRFNode::Split { feature: feat, threshold, left: Box::new(left), right: Box::new(right) }
        }
        None => {
            let values: Vec<f64> = indices.iter().map(|&i| target[i]).collect();
            QRFNode::Leaf { values }
        }
    }
}

fn mse_indices(target: &[f64], indices: &[usize]) -> f64 {
    let n = indices.len() as f64;
    let mean = indices.iter().map(|&i| target[i]).sum::<f64>() / n;
    indices.iter().map(|&i| (target[i] - mean).powi(2)).sum::<f64>() / n
}

// ── Trained QRF ─────────────────────────────────────────────────────

/// Trained Quantile Regression Forest with access to quantile predictions.
pub struct TrainedQuantileForest {
    trees: Vec<QRFNode>,
    n_features: usize,
}

impl TrainedQuantileForest {
    /// Predict a specific quantile for each sample.
    pub fn predict_quantile(&self, features: &Array2<f64>, quantile: f64) -> Result<Vec<f64>> {
        crate::validate::check_n_features(features, self.n_features)?;

        Ok(features.rows().into_iter()
            .map(|row| {
                let row_vec: Vec<f64> = row.to_vec();
                let mut all_values: Vec<f64> = Vec::new();
                for tree in &self.trees {
                    all_values.extend_from_slice(tree.find_leaf(&row_vec));
                }
                all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((all_values.len() as f64 * quantile).ceil() as usize)
                    .min(all_values.len()).max(1) - 1;
                all_values[idx]
            })
            .collect())
    }

    /// Predict interval [lower, upper] for each sample.
    pub fn predict_interval(&self, features: &Array2<f64>, alpha: f64) -> Result<Vec<(f64, f64)>> {
        let lower = self.predict_quantile(features, alpha / 2.0)?;
        let upper = self.predict_quantile(features, 1.0 - alpha / 2.0)?;
        Ok(lower.into_iter().zip(upper).collect())
    }
}

impl TrainedModel for TrainedQuantileForest {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        // Default: predict median (quantile 0.5)
        let predicted = self.predict_quantile(features, 0.5)?;
        Ok(Prediction::regression(predicted))
    }
}

// ── Learner impl ────────────────────────────────────────────────────

impl Learner for QuantileForest {
    fn id(&self) -> &str { "quantile_forest" }

    fn train_classif(&mut self, _: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Err(SmeltError::Other("QuantileForest only supports regression".into()))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();

        let trees: Vec<QRFNode> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64));
                // Bootstrap
                let indices: Vec<usize> = (0..n_samples)
                    .map(|_| rng.random_range(0..n_samples))
                    .collect();
                build_qrf_tree(features, target, &indices, self.max_depth,
                    self.min_samples_leaf, n_features, 0, &mut rng)
            })
            .collect();

        Ok(Box::new(TrainedQuantileForest { trees, n_features }))
    }
}
