//! Quantile Regression via Gradient Boosting.
//!
//! Predicts conditional quantiles instead of the conditional mean.
//! Uses the pinball (quantile) loss function.

use crate::learner::tree::TreeBuilder;
use crate::learner::tree::{LeafValue, Node};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use crate::{Result, SmeltError};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;

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

    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
}

struct TrainedQuantileGB {
    trees: Vec<Node>,
    initial: f64,
    learning_rate: f64,
}

impl TrainedModel for TrainedQuantileGB {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
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
}

impl Learner for QuantileGB {
    fn id(&self) -> &str {
        "quantile_gb"
    }

    fn train_classif(&mut self, _: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        Err(SmeltError::Other(
            "QuantileGB only supports regression".into(),
        ))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();
        let tau = self.quantile;

        // Initial prediction: quantile of target
        let mut sorted_target: Vec<f64> = target.to_vec();
        sorted_target.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q_idx = ((n_samples as f64 * tau).ceil() as usize).min(n_samples) - 1;
        let initial = sorted_target[q_idx.max(0)];

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
        }))
    }
}
