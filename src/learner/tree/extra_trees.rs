//! Extra Trees (Extremely Randomized Trees): ensemble with random thresholds and no bootstrap.

use super::{LeafValue, MaxFeatures, Node, TreeBuilder};
use crate::Result;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Extremely Randomized Trees learner.
///
/// Like Random Forest but with two key differences:
/// - No bootstrap sampling (uses all training data)
/// - Random split thresholds instead of optimal ones
///
/// Often faster than RF and can achieve better generalization.
///
/// Supports per-sample weights: weights enter each tree's impurity and leaf
/// values (sklearn convention); a weight of `0.0` excludes the sample and
/// `min_samples_*` count rows, not total weight.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 0.9]];
/// let target = vec![0, 0, 1, 1];
/// let task = ClassificationTask::new("et_demo", features, target).unwrap();
///
/// let mut et = ExtraTrees::new().with_n_estimators(10).with_seed(42);
/// let model = et.train_classif(&task).unwrap();
/// ```
pub struct ExtraTrees {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_features: MaxFeatures,
    seed: u64,
}

impl Default for ExtraTrees {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::Auto,
            seed: 42,
        }
    }
}

impl ExtraTrees {
    /// Creates an Extra Trees ensemble with default hyperparameters (100
    /// trees; `sqrt(n_features)` per split for classification, all features
    /// for regression -- see [`MaxFeatures`]).
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the number of trees in the ensemble.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    /// Sets the maximum depth of each tree.
    pub fn with_max_depth(mut self, d: usize) -> Self {
        self.max_depth = Some(d);
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
    /// Forces the classic `sqrt(n_features)` candidate-feature heuristic for
    /// both classification and regression (the default already uses this for
    /// classification; use this to also apply it to regression, overriding
    /// the task-appropriate default of considering all features). Mirrors
    /// [`RandomForest::with_max_features_sqrt`](super::RandomForest::with_max_features_sqrt).
    pub fn with_max_features_sqrt(mut self) -> Self {
        self.max_features = MaxFeatures::Sqrt;
        self
    }
    /// Sets an explicit fraction of features considered at each split
    /// (applies to both classification and regression, overriding the
    /// task-appropriate default -- see [`MaxFeatures`]).
    pub fn with_max_features_fraction(mut self, f: f64) -> Self {
        self.max_features = MaxFeatures::Fraction(f);
        self
    }
    /// Sets the RNG seed used for bootstrap-free tree construction and random thresholds.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// A trained Extra Trees ensemble, ready to predict.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedExtraTrees {
    pub(crate) trees: Vec<Node>,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_importances: Vec<f64>,
    pub(crate) n_classes: Option<usize>,
    pub(crate) is_classifier: bool,
}

impl TrainedModel for TrainedExtraTrees {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        if self.is_classifier {
            let n_classes = self.n_classes.unwrap();
            let n_trees = self.trees.len() as f64;
            let mut predicted = Vec::with_capacity(features.nrows());
            let mut probabilities = Vec::with_capacity(features.nrows());

            for row in features.rows() {
                let mut avg_probs = vec![0.0; n_classes];
                for tree in &self.trees {
                    if let LeafValue::Class(_, probs) = tree.predict_one(row) {
                        for (j, p) in probs.iter().enumerate() {
                            avg_probs[j] += p;
                        }
                    }
                }
                for p in &mut avg_probs {
                    *p /= n_trees;
                }
                let pred_class = avg_probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
                    .0;
                predicted.push(pred_class);
                probabilities.push(avg_probs);
            }

            Ok(Prediction::Classification {
                predicted,
                truth: None,
                probabilities: Some(probabilities),
            })
        } else {
            let n_trees = self.trees.len() as f64;
            let predicted: Vec<f64> = features
                .rows()
                .into_iter()
                .map(|row| {
                    self.trees
                        .iter()
                        .map(|t| match t.predict_one(row) {
                            LeafValue::Value(v) => *v,
                            _ => unreachable!(),
                        })
                        .sum::<f64>()
                        / n_trees
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
        Some(crate::serialize::SerializableModel::ExtraTrees(
            self.clone(),
        ))
    }
}

impl Learner for ExtraTrees {
    fn id(&self) -> &str {
        "extra_trees"
    }

    /// `true`: weights enter each tree's impurity and leaf values (sklearn
    /// convention); a weight of 0.0 excludes the sample from every tree
    /// (Extra Trees has no bootstrap, so exclusion is exact and ensemble-wide).
    fn supports_weights(&self) -> bool {
        true
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let weights = task.weights();
        let n_samples = task.n_samples();
        let n_features = task.n_features();
        let n_classes = task.n_classes();
        let max_feat = self.max_features.resolve(n_features, true);
        // No bootstrap — use all samples (minus weight-0 exclusions).
        let indices: Vec<usize> = match weights {
            None => (0..n_samples).collect(),
            Some(w) => super::retain_positive_weight((0..n_samples).collect(), w, n_samples),
        };

        let results: Vec<(Node, Vec<f64>)> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64));
                let mut builder = TreeBuilder::new(
                    self.max_depth,
                    self.min_samples_split,
                    self.min_samples_leaf,
                    max_feat,
                    n_features,
                )
                .with_random_splits(true);
                let root = match weights {
                    None => builder.build_classifier(
                        &features.view(),
                        target,
                        &indices,
                        n_classes,
                        0,
                        &mut rng,
                    ),
                    Some(w) => builder.build_classifier_weighted(
                        &features.view(),
                        target,
                        w,
                        &indices,
                        n_classes,
                        0,
                        &mut rng,
                    ),
                };
                (root, builder.feature_importances)
            })
            .collect();

        let mut total_imp = vec![0.0; n_features];
        let mut trees = Vec::with_capacity(self.n_estimators);
        for (root, imp) in results {
            for (j, v) in imp.iter().enumerate() {
                total_imp[j] += v;
            }
            trees.push(root);
        }

        Ok(Box::new(TrainedExtraTrees {
            trees,
            feature_names: task.feature_names().to_vec(),
            feature_importances: total_imp,
            n_classes: Some(n_classes),
            is_classifier: true,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let weights = task.weights();
        let n_samples = task.n_samples();
        let n_features = task.n_features();
        let max_feat = self.max_features.resolve(n_features, false);
        let indices: Vec<usize> = match weights {
            None => (0..n_samples).collect(),
            Some(w) => super::retain_positive_weight((0..n_samples).collect(), w, n_samples),
        };

        let results: Vec<(Node, Vec<f64>)> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64));
                let mut builder = TreeBuilder::new(
                    self.max_depth,
                    self.min_samples_split,
                    self.min_samples_leaf,
                    max_feat,
                    n_features,
                )
                .with_random_splits(true);
                let root = match weights {
                    None => {
                        builder.build_regressor(&features.view(), target, &indices, 0, &mut rng)
                    }
                    Some(w) => builder.build_regressor_weighted(
                        &features.view(),
                        target,
                        w,
                        &indices,
                        0,
                        &mut rng,
                    ),
                };
                (root, builder.feature_importances)
            })
            .collect();

        let mut total_imp = vec![0.0; n_features];
        let mut trees = Vec::with_capacity(self.n_estimators);
        for (root, imp) in results {
            for (j, v) in imp.iter().enumerate() {
                total_imp[j] += v;
            }
            trees.push(root);
        }

        Ok(Box::new(TrainedExtraTrees {
            trees,
            feature_names: task.feature_names().to_vec(),
            feature_importances: total_imp,
            n_classes: None,
            is_classifier: false,
        }))
    }
}
