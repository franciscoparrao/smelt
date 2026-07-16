//! Quantile Regression Forest (QRF).
//!
//! Random Forest that stores all training targets in leaves to compute
//! any quantile at prediction time. Produces full conditional distributions.
//!
//! Reference: Meinshausen, N. (2006). Quantile Regression Forests. JMLR 7, 983-999.

use crate::learner::tree::{MaxFeatures, mse_from_sums};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;
use crate::task::{RegressionTask, Task};
use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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
    max_features: MaxFeatures,
    seed: u64,
}

impl Default for QuantileForest {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            min_samples_leaf: 5,
            max_features: MaxFeatures::Auto,
            seed: 42,
        }
    }
}

impl QuantileForest {
    /// Creates a quantile regression forest with default hyperparameters --
    /// including, like [`RandomForest`](super::RandomForest)/
    /// [`ExtraTrees`](super::ExtraTrees), *all* features considered per
    /// split for this regression-only forest (see [`MaxFeatures`]), not the
    /// `sqrt(n_features)` this builder used to hardcode regardless of
    /// `max_features` setting -- QRF was the one regression forest left
    /// out when RF/ExtraTrees regression switched to an all-features
    /// default.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the number of trees in the forest.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    /// Sets the maximum depth of each tree.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }
    /// Sets the minimum number of training targets retained per leaf; leaves
    /// keep all targets that land in them so quantiles can be estimated
    /// empirically from that pooled sample at prediction time.
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }
    /// Forces the classic `sqrt(n_features)` candidate-feature heuristic,
    /// overriding the all-features default.
    pub fn with_max_features_sqrt(mut self) -> Self {
        self.max_features = MaxFeatures::Sqrt;
        self
    }
    /// Sets an explicit fraction of features considered at each split,
    /// overriding the all-features default.
    pub fn with_max_features_fraction(mut self, f: f64) -> Self {
        self.max_features = MaxFeatures::Fraction(f);
        self
    }
    /// Sets the RNG seed used for bootstrap sampling and feature subsetting.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

// ── QRF Tree internals ─────────────────────────────────────────────

#[derive(Clone, Serialize, Deserialize)]
enum QRFNode {
    Leaf {
        values: Vec<f64>,
    }, // all target values in this leaf
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
            QRFNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                if row[*feature] <= *threshold {
                    left.find_leaf(row)
                } else {
                    right.find_leaf(row)
                }
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
    max_features: Option<usize>,
    depth: usize,
    rng: &mut impl Rng,
) -> QRFNode {
    let n = indices.len();

    if n <= min_samples_leaf * 2 || max_depth.is_some_and(|d| depth >= d) {
        let values: Vec<f64> = indices.iter().map(|&i| target[i]).collect();
        return QRFNode::Leaf { values };
    }

    // Random feature subset: `max_features` (from `MaxFeatures::resolve`,
    // `None` meaning "all features") rather than a hardcoded
    // `sqrt(n_features)`, so this regression forest follows the same
    // task-appropriate default as RandomForest/ExtraTrees regression.
    let n_try = max_features.unwrap_or(n_features).clamp(1, n_features);
    let mut feat_indices: Vec<usize> = (0..n_features).collect();
    for i in 0..n_try {
        let j = rng.random_range(i..n_features);
        feat_indices.swap(i, j);
    }

    let mut best_gain = 0.0;
    let mut best_split = None;

    // MSE-based splitting
    let parent_mse = mse_indices(target, indices);
    // Center the running sums on the node mean — same catastrophic-
    // cancellation guard as TreeBuilder::best_split_regress: E[y²]−E[y]²
    // on raw targets with a large additive offset (UTM coordinates,
    // timestamps) turns split gains into rounding noise.
    let shift = indices.iter().map(|&i| target[i]).sum::<f64>() / n as f64;

    for &feat in &feat_indices[..n_try] {
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_by(|&a, &b| {
            features[[a, feat]]
                .partial_cmp(&features[[b, feat]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Sweep with running sum/sum-of-squares so mse(left)/mse(right) is
        // O(1) per candidate instead of an O(n) rescan (audit M-3, same fix
        // as TreeBuilder::best_split_regress — especially hot here since the
        // default considers *all* features per split). Candidates outside
        // the [min_samples_leaf, n - min_samples_leaf) window still feed the
        // running sums; they're just never evaluated, exactly like the old
        // loop bounds.
        let mut left_sum = 0.0;
        let mut left_sq = 0.0;
        let mut right_sum = 0.0;
        let mut right_sq = 0.0;
        for &idx in &sorted {
            let y = target[idx] - shift;
            right_sum += y;
            right_sq += y * y;
        }

        for s in 1..n {
            let y = target[sorted[s - 1]] - shift;
            left_sum += y;
            left_sq += y * y;
            right_sum -= y;
            right_sq -= y * y;

            if s < min_samples_leaf || s >= n.saturating_sub(min_samples_leaf) {
                continue;
            }
            if (features[[sorted[s], feat]] - features[[sorted[s - 1], feat]]).abs() < f64::EPSILON
            {
                continue;
            }

            let n_left = s as f64;
            let n_right = (n - s) as f64;
            let gain = parent_mse
                - (n_left / n as f64) * mse_from_sums(left_sum, left_sq, n_left)
                - (n_right / n as f64) * mse_from_sums(right_sum, right_sq, n_right);

            if gain > best_gain {
                best_gain = gain;
                let threshold =
                    (features[[sorted[s - 1], feat]] + features[[sorted[s], feat]]) / 2.0;
                best_split = Some((feat, threshold, sorted[..s].to_vec(), sorted[s..].to_vec()));
            }
        }
    }

    match best_split {
        Some((feat, threshold, left_idx, right_idx)) => {
            let left = build_qrf_tree(
                features,
                target,
                &left_idx,
                max_depth,
                min_samples_leaf,
                n_features,
                max_features,
                depth + 1,
                rng,
            );
            let right = build_qrf_tree(
                features,
                target,
                &right_idx,
                max_depth,
                min_samples_leaf,
                n_features,
                max_features,
                depth + 1,
                rng,
            );
            QRFNode::Split {
                feature: feat,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
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
    indices
        .iter()
        .map(|&i| (target[i] - mean).powi(2))
        .sum::<f64>()
        / n
}

// ── Trained QRF ─────────────────────────────────────────────────────

/// Trained Quantile Regression Forest with access to quantile predictions.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedQuantileForest {
    trees: Vec<QRFNode>,
    n_features: usize,
}

impl TrainedQuantileForest {
    /// Predict a specific quantile for each sample.
    pub fn predict_quantile(&self, features: &Array2<f64>, quantile: f64) -> Result<Vec<f64>> {
        crate::validate::check_n_features(features, self.n_features)?;

        Ok(features
            .rows()
            .into_iter()
            .map(|row| {
                let row_vec: Vec<f64> = row.to_vec();
                let mut all_values: Vec<f64> = Vec::new();
                for tree in &self.trees {
                    all_values.extend_from_slice(tree.find_leaf(&row_vec));
                }
                all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = ((all_values.len() as f64 * quantile).ceil() as usize)
                    .min(all_values.len())
                    .max(1)
                    - 1;
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

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::QuantileForest(
            self.clone(),
        ))
    }
}

// ── Learner impl ────────────────────────────────────────────────────

impl Learner for QuantileForest {
    fn id(&self) -> &str {
        "quantile_forest"
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();
        let max_features = self.max_features.resolve(n_features, false);

        let trees: Vec<QRFNode> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64));
                // Bootstrap
                let indices: Vec<usize> = (0..n_samples)
                    .map(|_| rng.random_range(0..n_samples))
                    .collect();
                build_qrf_tree(
                    features,
                    target,
                    &indices,
                    self.max_depth,
                    self.min_samples_leaf,
                    n_features,
                    max_features,
                    0,
                    &mut rng,
                )
            })
            .collect();

        Ok(Box::new(TrainedQuantileForest { trees, n_features }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: before this fix, QuantileForest hardcoded
    /// `sqrt(n_features)` candidates per split regardless of `max_features`
    /// (which didn't exist as a setting at all) -- the same failure mode
    /// that motivated RandomForest/ExtraTrees regression switching to an
    /// all-features default (`docs/auditoria_motor_2026-07-05.md` M-2).
    /// With most features pure noise, the all-features default (now QRF's
    /// default too) should out-predict the old hardcoded sqrt behavior.
    #[test]
    fn default_beats_sqrt_heuristic_when_few_features_are_informative() {
        let mut rng = StdRng::seed_from_u64(7);
        let n = 400;
        let p = 48;
        let mut feats = Vec::with_capacity(n * p);
        let mut target = Vec::with_capacity(n);
        for _ in 0..n {
            let mut row = Vec::with_capacity(p);
            for _ in 0..p {
                row.push(rng.random::<f64>());
            }
            let y = 5.0 * row[0] - 3.0 * row[1] + 2.0 * row[2] + rng.random::<f64>() * 0.1;
            feats.extend_from_slice(&row);
            target.push(y);
        }
        let features = Array2::from_shape_vec((n, p), feats).unwrap();
        let task = RegressionTask::new("qrf_sparse_signal", features.clone(), target.clone()).unwrap();

        let rmse = |predicted: &[f64]| -> f64 {
            (predicted.iter().zip(&target).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n as f64)
                .sqrt()
        };

        fn regression_values(pred: Prediction) -> Vec<f64> {
            let Prediction::Regression { predicted, .. } = pred else {
                panic!("expected regression");
            };
            predicted
        }

        let mut default_qrf = QuantileForest::new().with_n_estimators(50).with_seed(1);
        let default_model = default_qrf.train_regress(&task).unwrap();
        let default_rmse = rmse(&regression_values(default_model.predict(&features).unwrap()));

        let mut sqrt_qrf = QuantileForest::new()
            .with_n_estimators(50)
            .with_seed(1)
            .with_max_features_sqrt();
        let sqrt_model = sqrt_qrf.train_regress(&task).unwrap();
        let sqrt_rmse = rmse(&regression_values(sqrt_model.predict(&features).unwrap()));

        assert!(
            default_rmse < sqrt_rmse,
            "all-features default (RMSE={default_rmse}) should beat sqrt(p) \
             heuristic (RMSE={sqrt_rmse}) when only 3/{p} features carry signal"
        );
    }

    /// Regression test for the M-3 O(n²) fix (incremental sweep replacing
    /// the per-candidate `mse_indices` rescan) inheriting the HIGH-1 guard:
    /// the sums must be centered on the node mean, so the split found for a
    /// step signal must be independent of a large additive target offset
    /// (UTM northing ~7e6, timestamps ~1e9). `build_qrf_tree` only consumes
    /// RNG for the feature shuffle, so with a fixed seed the root split must
    /// be identical (and at the step) for every offset.
    #[test]
    fn qrf_root_split_is_invariant_to_large_target_offsets() {
        let n = 200;
        let features = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 / n as f64 * 10.0);
        let base: Vec<f64> = (0..n)
            .map(|i| {
                let x = features[[i, 0]];
                let step = if x < 5.0 { 0.0 } else { 4.0 };
                step + 0.3 * ((i as f64 * 12.9898).sin())
            })
            .collect();
        let indices: Vec<usize> = (0..n).collect();

        let mut thresholds = Vec::new();
        for offset in [0.0, 1e6, 1e8] {
            let target: Vec<f64> = base.iter().map(|y| y + offset).collect();
            let mut rng = StdRng::seed_from_u64(0);
            let root = build_qrf_tree(&features, &target, &indices, None, 1, 1, None, 0, &mut rng);
            let QRFNode::Split { threshold, .. } = root else {
                panic!("a step signal must produce a root split, got a leaf at offset {offset}");
            };
            thresholds.push(threshold);
        }
        assert!(
            (thresholds[0] - 5.0).abs() < 0.2,
            "root split should land at the step (~5.0), got {}",
            thresholds[0]
        );
        assert_eq!(
            thresholds[0], thresholds[1],
            "offset 1e6 changed the root split: {} vs {}",
            thresholds[0], thresholds[1]
        );
        assert_eq!(
            thresholds[0], thresholds[2],
            "offset 1e8 changed the root split: {} vs {}",
            thresholds[0], thresholds[2]
        );
    }

    #[test]
    fn max_features_fraction_is_plumbed_through() {
        let features = Array2::from_shape_fn((50, 10), |(i, j)| ((i + j) % 7) as f64);
        let target: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let task = RegressionTask::new("qrf_fraction", features.clone(), target).unwrap();

        let mut qrf = QuantileForest::new()
            .with_n_estimators(5)
            .with_seed(1)
            .with_max_features_fraction(0.3);
        let model = qrf.train_regress(&task).unwrap();
        let Prediction::Regression { predicted, .. } = model.predict(&features).unwrap() else {
            panic!("expected regression");
        };
        assert_eq!(predicted.len(), 50);
    }
}
