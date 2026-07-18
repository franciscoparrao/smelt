//! Decision Tree (CART) learner for classification and regression.
//!
//! Uses Gini impurity for classification and MSE for regression.

use super::{LeafValue, Node, TreeBuilder};
use crate::Result;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// CART Decision Tree learner.
///
/// Supports both classification (Gini impurity) and regression (MSE).
///
/// Supports per-sample weights ([`ClassificationTask::with_weights`]/
/// [`RegressionTask::with_weights`]): weights enter the impurity and the
/// leaf values (sklearn convention). An integer weight `k` is exactly
/// equivalent to duplicating the row `k` times, and a weight of `0.0`
/// exactly excludes the sample. `min_samples_split`/`min_samples_leaf`
/// count rows, not total weight.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::DecisionTree;
/// use ndarray::array;
///
/// let features = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
/// let target = vec![0, 1, 1, 0]; // XOR-like
///
/// let task = ClassificationTask::new("xor", features, target).unwrap();
/// let mut tree = DecisionTree::default();
/// let model = tree.train_classif(&task).unwrap();
/// let pred = model.predict(task.features()).unwrap();
/// ```
pub struct DecisionTree {
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

impl Default for DecisionTree {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }
}

impl DecisionTree {
    /// Creates a decision tree with default hyperparameters (unbounded depth, min split 2, min leaf 1).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum depth of the tree.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Sets the minimum number of samples required to split an internal node.
    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    /// Sets the minimum number of samples required in each leaf.
    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }
}

use serde::{Deserialize, Serialize};

/// A trained CART decision tree, ready to predict.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedDecisionTree {
    pub(crate) root: Node,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_importances: Vec<f64>,
    pub(crate) is_classifier: bool,
}

impl TrainedModel for TrainedDecisionTree {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        if self.is_classifier {
            let mut predicted = Vec::with_capacity(features.nrows());
            let mut probabilities = Vec::with_capacity(features.nrows());

            for row in features.rows() {
                match self.root.predict_one(row) {
                    LeafValue::Class(class, probs) => {
                        predicted.push(*class);
                        probabilities.push(probs.clone());
                    }
                    _ => unreachable!(),
                }
            }

            Ok(Prediction::Classification {
                predicted,
                truth: None,
                probabilities: Some(probabilities),
            })
        } else {
            let predicted: Vec<f64> = features
                .rows()
                .into_iter()
                .map(|row| match self.root.predict_one(row) {
                    LeafValue::Value(v) => *v,
                    _ => unreachable!(),
                })
                .collect();

            Ok(Prediction::regression(predicted))
        }
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        let total: f64 = self.feature_importances.iter().sum();
        if total == 0.0 {
            return None;
        }
        Some(
            self.feature_names
                .iter()
                .zip(&self.feature_importances)
                .map(|(name, &imp)| (name.clone(), imp / total))
                .collect(),
        )
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::DecisionTree(
            self.clone(),
        ))
    }
}

impl Learner for DecisionTree {
    fn id(&self) -> &str {
        "decision_tree"
    }

    /// `true`: weights enter the Gini/MSE impurity and the leaf values
    /// (sklearn convention); `min_samples_split`/`min_samples_leaf` still
    /// count rows, and a weight of 0.0 excludes the sample entirely.
    fn supports_weights(&self) -> bool {
        true
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_classes = task.n_classes();
        let n_features = task.n_features();
        let indices: Vec<usize> = (0..task.n_samples()).collect();

        let mut builder = TreeBuilder::new(
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            None,
            n_features,
        );
        // Seeded even though a single tree with max_features=None never
        // consumes it: an OS-seeded rng here would silently make DecisionTree
        // nondeterministic the day TreeBuilder starts drawing from it (e.g.
        // if a max_features knob is ever wired up).
        let mut rng = StdRng::seed_from_u64(0);
        let root = match task.weights() {
            None => builder.build_classifier(
                &features.view(),
                target,
                &indices,
                n_classes,
                0,
                &mut rng,
            ),
            Some(w) => {
                // Weight 0.0 = sample excluded: dropping the row here (not
                // just letting it contribute nothing) keeps candidate split
                // thresholds identical to actually deleting the row.
                let indices = super::retain_positive_weight(indices, w, task.n_samples());
                builder.build_classifier_weighted(
                    &features.view(),
                    target,
                    w,
                    &indices,
                    n_classes,
                    0,
                    &mut rng,
                )
            }
        };

        Ok(Box::new(TrainedDecisionTree {
            root,
            feature_names: task.feature_names().to_vec(),
            feature_importances: builder.feature_importances,
            is_classifier: true,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_features = task.n_features();
        let indices: Vec<usize> = (0..task.n_samples()).collect();

        let mut builder = TreeBuilder::new(
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
            None,
            n_features,
        );
        // Same rationale as train_classif: keep the never-consumed rng seeded.
        let mut rng = StdRng::seed_from_u64(0);
        let root = match task.weights() {
            None => builder.build_regressor(&features.view(), target, &indices, 0, &mut rng),
            Some(w) => {
                let indices = super::retain_positive_weight(indices, w, task.n_samples());
                builder.build_regressor_weighted(&features.view(), target, w, &indices, 0, &mut rng)
            }
        };

        Ok(Box::new(TrainedDecisionTree {
            root,
            feature_names: task.feature_names().to_vec(),
            feature_importances: builder.feature_importances,
            is_classifier: false,
        }))
    }
}
