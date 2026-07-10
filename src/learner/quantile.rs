//! Quantile Regression via Gradient Boosting.
//!
//! Predicts conditional quantiles instead of the conditional mean.
//! Uses the pinball (quantile) loss function.

use crate::learner::tree::TreeBuilder;
use crate::learner::tree::{LeafValue, Node};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;
use crate::task::{RegressionTask, Task};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// Quantile Gradient Boosting regressor.
///
/// Predicts a specific quantile τ of the conditional distribution P(Y|X).
/// Use multiple QuantileGBs at different τ values to get prediction intervals
/// (e.g., τ=0.05 and τ=0.95 for a 90% interval).
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
/// let task = RegressionTask::new("q", features, target).unwrap();
///
/// // Predict the median (τ=0.5)
/// let mut qgb = QuantileGB::new(0.5).with_n_estimators(50);
/// let model = qgb.train_regress(&task).unwrap();
///
/// // For prediction intervals: train at τ=0.05 and τ=0.95
/// let mut lower = QuantileGB::new(0.05).with_n_estimators(50);
/// let mut upper = QuantileGB::new(0.95).with_n_estimators(50);
/// ```
pub struct QuantileGB {
    quantile: f64,
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    seed: u64,
}

impl QuantileGB {
    /// Creates a quantile gradient boosting learner targeting the given
    /// quantile `τ` (e.g. 0.5 for the median) of the conditional distribution.
    pub fn new(quantile: f64) -> Self {
        Self {
            quantile,
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            seed: 42,
        }
    }

    /// Sets the number of boosting rounds (trees).
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    /// Sets the shrinkage applied to each tree's contribution.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    /// Sets the maximum depth of each boosted tree.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }
    /// Sets the RNG seed used for tree building.
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

/// A trained Quantile Gradient Boosting regressor.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedQuantileGB {
    trees: Vec<Node>,
    initial: f64,
    learning_rate: f64,
    n_features: usize,
}

impl TrainedModel for TrainedQuantileGB {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.n_features)?;
        let predicted: Vec<f64> = features
            .rows()
            .into_iter()
            .map(|row| {
                let mut val = self.initial;
                for tree in &self.trees {
                    if let LeafValue::Value(v) = tree.predict_one(row) {
                        val += self.learning_rate * v;
                    }
                }
                val
            })
            .collect();
        Ok(Prediction::regression(predicted))
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::QuantileGB(
            self.clone(),
        ))
    }
}

impl Learner for QuantileGB {
    fn id(&self) -> &str {
        "quantile_gb"
    }


    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        if !(self.quantile > 0.0 && self.quantile < 1.0) {
            return Err(crate::SmeltError::InvalidParameter(format!(
                "quantile must be in (0, 1), got {}",
                self.quantile
            )));
        }
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();
        let tau = self.quantile;

        // Initial prediction: quantile of target
        let mut sorted_target: Vec<f64> = target.to_vec();
        sorted_target.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q_idx = ((n_samples as f64 * tau).ceil() as usize).clamp(1, n_samples) - 1;
        let initial = sorted_target[q_idx];

        let mut preds = vec![initial; n_samples];
        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _ in 0..self.n_estimators {
            // Pinball loss gradient: if y > ŷ → gradient = -τ, else → gradient = (1-τ)
            // We fit the tree to the negative gradient
            let neg_grads: Vec<f64> = (0..n_samples)
                .map(|i| {
                    let residual = target[i] - preds[i];
                    if residual > 0.0 { tau } else { -(1.0 - tau) }
                })
                .collect();

            let indices: Vec<usize> = (0..n_samples).collect();
            let mut builder = TreeBuilder::new(
                self.max_depth,
                self.min_samples_split,
                self.min_samples_leaf,
                None,
                n_features,
            );
            let root = builder.build_regressor(&features.view(), &neg_grads, &indices, 0, &mut rng);

            for i in 0..n_samples {
                if let LeafValue::Value(v) = root.predict_one(features.row(i)) {
                    preds[i] += self.learning_rate * v;
                }
            }
            trees.push(root);
        }

        Ok(Box::new(TrainedQuantileGB {
            trees,
            initial,
            learning_rate: self.learning_rate,
            n_features,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `QuantileGB::new(0.0)` (or `1.0`) used to panic with
    /// a `usize` subtraction overflow in `train_regress`'s initial-quantile
    /// index computation instead of returning a clean error.
    #[test]
    fn quantile_out_of_open_unit_interval_is_rejected() {
        let features = Array2::from_shape_vec((10, 1), (0..10).map(|i| i as f64).collect()).unwrap();
        let target: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let task = RegressionTask::new("q", features, target).unwrap();

        for tau in [0.0, 1.0, -0.1, 1.1] {
            let mut q = QuantileGB::new(tau).with_n_estimators(2);
            assert!(q.train_regress(&task).is_err(), "tau={tau} should be rejected");
        }
    }

    #[test]
    fn predict_rejects_wrong_feature_count() {
        let features = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let target: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let task = RegressionTask::new("q", features, target).unwrap();
        let mut q = QuantileGB::new(0.5).with_n_estimators(2);
        let model = q.train_regress(&task).unwrap();

        let wrong = Array2::from_shape_vec((3, 5), vec![0.0; 15]).unwrap();
        assert!(model.predict(&wrong).is_err());
    }
}
