//! Explainable Boosting Machine (EBM): interpretable GAM via cyclic gradient boosting.
//!
//! Trains one feature at a time in round-robin fashion. The result is an additive model
//! f(x) = f₁(x₁) + f₂(x₂) + ... where each fᵢ is a shape function that can be visualized.

use crate::Result;
use crate::learner::math::sigmoid;
use crate::learner::tree::TreeBuilder;
use crate::learner::tree::{LeafValue, Node};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Explainable Boosting Machine.
///
/// A GAM (Generalized Additive Model) trained with cyclic gradient boosting.
/// As accurate as XGBoost but fully interpretable — each feature has a
/// shape function showing its contribution to the prediction.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
/// let target = vec![0, 0, 1, 1];
/// let task = ClassificationTask::new("ebm", features, target).unwrap();
///
/// let mut ebm = EBM::new().with_n_rounds(50);
/// let model = ebm.train_classif(&task).unwrap();
/// ```
pub struct EBM {
    n_rounds: usize, // number of cyclic boosting rounds
    learning_rate: f64,
    max_depth: Option<usize>,
    min_samples_leaf: usize,
    seed: u64,
}

impl Default for EBM {
    fn default() -> Self {
        Self {
            n_rounds: 100,
            learning_rate: 0.01, // very low LR is key to EBM
            max_depth: Some(3),
            min_samples_leaf: 2,
            seed: 42,
        }
    }
}

impl EBM {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_n_rounds(mut self, n: usize) -> Self {
        self.n_rounds = n;
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

/// Trained EBM. Each entry in `shape_trees` is a list of boosted stumps for that feature.
struct TrainedEBM {
    /// shape_trees[feature][round] = tree trained on that feature at that round
    shape_trees: Vec<Vec<Node>>,
    intercept: f64,
    learning_rate: f64,
    feature_names: Vec<String>,
    is_classifier: bool,
    n_classes: usize,
}

impl TrainedModel for TrainedEBM {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        let n_samples = features.nrows();
        let n_features = self.shape_trees.len();

        // Compute additive scores: intercept + Σ f_j(x_j)
        let mut scores = vec![self.intercept; n_samples];
        for j in 0..n_features {
            for tree in &self.shape_trees[j] {
                for i in 0..n_samples {
                    if let LeafValue::Value(v) = tree.predict_one(features.row(i)) {
                        scores[i] += self.learning_rate * v;
                    }
                }
            }
        }

        if self.is_classifier {
            let mut predicted = Vec::with_capacity(n_samples);
            let mut probabilities = Vec::with_capacity(n_samples);

            if self.n_classes == 2 {
                for &s in &scores {
                    let p = sigmoid(s);
                    predicted.push(if p >= 0.5 { 1 } else { 0 });
                    probabilities.push(vec![1.0 - p, p]);
                }
            } else {
                // For multiclass, we'd need per-class EBMs. Simplified: use binary for now.
                for &s in &scores {
                    let p = sigmoid(s);
                    predicted.push(if p >= 0.5 { 1 } else { 0 });
                    probabilities.push(vec![1.0 - p, p]);
                }
            }

            Ok(Prediction::Classification {
                predicted,
                truth: None,
                probabilities: Some(probabilities),
            })
        } else {
            Ok(Prediction::regression(scores))
        }
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        // Importance = sum of absolute leaf values across all rounds for each feature
        let mut importance: Vec<f64> = Vec::with_capacity(self.shape_trees.len());
        for trees in &self.shape_trees {
            let mut feat_imp = 0.0;
            for tree in trees {
                feat_imp += count_node_importance(tree);
            }
            importance.push(feat_imp);
        }

        let total: f64 = importance.iter().sum();
        if total == 0.0 {
            return None;
        }
        Some(
            self.feature_names
                .iter()
                .zip(&importance)
                .map(|(n, &i)| (n.clone(), i / total))
                .collect(),
        )
    }
}

fn count_node_importance(node: &Node) -> f64 {
    match node {
        Node::Leaf(LeafValue::Value(v)) => v.abs(),
        Node::Leaf(LeafValue::Class(_, _)) => 0.0,
        Node::Split { left, right, .. } => {
            count_node_importance(left) + count_node_importance(right)
        }
    }
}

impl Learner for EBM {
    fn id(&self) -> &str {
        "ebm"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();
        let n_classes = task.n_classes();

        // Binary classification: log-loss
        let p_pos = target.iter().filter(|&&t| t == 1).count() as f64 / n_samples as f64;
        let intercept = (p_pos / (1.0 - p_pos).max(1e-15)).ln();
        let mut f_vals = vec![intercept; n_samples];

        let mut shape_trees: Vec<Vec<Node>> = (0..n_features).map(|_| Vec::new()).collect();
        let mut rng = StdRng::seed_from_u64(self.seed);

        // Cyclic round-robin: each round trains ONE tree per feature
        for _ in 0..self.n_rounds {
            for j in 0..n_features {
                // Pseudo-residuals for log-loss
                let residuals: Vec<f64> = (0..n_samples)
                    .map(|i| target[i] as f64 - sigmoid(f_vals[i]))
                    .collect();

                // Build single-feature view: column j only
                let col = features.column(j);
                let single_feature = Array2::from_shape_fn((n_samples, 1), |(i, _)| col[i]);

                let indices: Vec<usize> = (0..n_samples).collect();
                let mut builder = TreeBuilder::new(
                    self.max_depth,
                    2,
                    self.min_samples_leaf,
                    None,
                    1, // only 1 feature
                );
                let tree = builder.build_regressor(
                    &single_feature.view(),
                    &residuals,
                    &indices,
                    0,
                    &mut rng,
                );

                // Update predictions
                for i in 0..n_samples {
                    if let LeafValue::Value(v) = tree.predict_one(single_feature.row(i)) {
                        f_vals[i] += self.learning_rate * v;
                    }
                }
                shape_trees[j].push(tree);
            }
        }

        Ok(Box::new(TrainedEBM {
            shape_trees,
            intercept,
            learning_rate: self.learning_rate,
            feature_names: task.feature_names().to_vec(),
            is_classifier: true,
            n_classes,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();

        let intercept = target.iter().sum::<f64>() / n_samples as f64;
        let mut preds = vec![intercept; n_samples];

        let mut shape_trees: Vec<Vec<Node>> = (0..n_features).map(|_| Vec::new()).collect();
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _ in 0..self.n_rounds {
            for j in 0..n_features {
                let residuals: Vec<f64> = (0..n_samples).map(|i| target[i] - preds[i]).collect();

                let col = features.column(j);
                let single_feature = Array2::from_shape_fn((n_samples, 1), |(i, _)| col[i]);

                let indices: Vec<usize> = (0..n_samples).collect();
                let mut builder =
                    TreeBuilder::new(self.max_depth, 2, self.min_samples_leaf, None, 1);
                let tree = builder.build_regressor(
                    &single_feature.view(),
                    &residuals,
                    &indices,
                    0,
                    &mut rng,
                );

                for i in 0..n_samples {
                    if let LeafValue::Value(v) = tree.predict_one(single_feature.row(i)) {
                        preds[i] += self.learning_rate * v;
                    }
                }
                shape_trees[j].push(tree);
            }
        }

        Ok(Box::new(TrainedEBM {
            shape_trees,
            intercept,
            learning_rate: self.learning_rate,
            feature_names: task.feature_names().to_vec(),
            is_classifier: false,
            n_classes: 0,
        }))
    }
}
