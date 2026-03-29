//! Decision Tree (CART) learner for classification and regression.
//!
//! Uses Gini impurity for classification and MSE for regression.

use ndarray::Array2;
use crate::task::{ClassificationTask, RegressionTask, Task};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;
use super::{Node, LeafValue, TreeBuilder};

/// CART Decision Tree learner.
///
/// Supports both classification (Gini impurity) and regression (MSE).
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
    pub fn new() -> Self { Self::default() }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    pub fn with_min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = n;
        self
    }

    pub fn with_min_samples_leaf(mut self, n: usize) -> Self {
        self.min_samples_leaf = n;
        self
    }
}

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct TrainedDecisionTree {
    pub(crate) root: Node,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_importances: Vec<f64>,
    pub(crate) is_classifier: bool,
}

impl TrainedModel for TrainedDecisionTree {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
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
            let predicted: Vec<f64> = features.rows().into_iter()
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
            self.feature_names.iter()
                .zip(&self.feature_importances)
                .map(|(name, &imp)| (name.clone(), imp / total))
                .collect(),
        )
    }
}

impl Learner for DecisionTree {
    fn id(&self) -> &str { "decision_tree" }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_classes = task.n_classes();
        let n_features = task.n_features();
        let indices: Vec<usize> = (0..task.n_samples()).collect();

        let mut builder = TreeBuilder::new(
            self.max_depth, self.min_samples_split, self.min_samples_leaf,
            None, n_features,
        );
        let mut rng = rand::rng();
        let root = builder.build_classifier(&features.view(), target, &indices, n_classes, 0, &mut rng);

        Ok(Box::new(TrainedDecisionTree {
            root,
            feature_names: task.feature_names().to_vec(),
            feature_importances: builder.feature_importances,
            is_classifier: true,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_features = task.n_features();
        let indices: Vec<usize> = (0..task.n_samples()).collect();

        let mut builder = TreeBuilder::new(
            self.max_depth, self.min_samples_split, self.min_samples_leaf,
            None, n_features,
        );
        let mut rng = rand::rng();
        let root = builder.build_regressor(&features.view(), target, &indices, 0, &mut rng);

        Ok(Box::new(TrainedDecisionTree {
            root,
            feature_names: task.feature_names().to_vec(),
            feature_importances: builder.feature_importances,
            is_classifier: false,
        }))
    }
}
