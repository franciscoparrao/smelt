//! Random Forest: ensemble of decision trees with bootstrap sampling and feature subsampling.

use super::{LeafValue, Node, TreeBuilder};
use crate::Result;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask, Task};
use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

/// Random Forest learner.
///
/// Trains multiple decision trees on bootstrap samples with random feature
/// subsets at each split. Predictions are aggregated by probability averaging
/// (classification) or mean (regression).
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 0.9]];
/// let target = vec![0, 0, 1, 1];
/// let task = ClassificationTask::new("rf_demo", features, target).unwrap();
///
/// let mut rf = RandomForest::new().with_n_estimators(10).with_seed(42);
/// let model = rf.train_classif(&task).unwrap();
/// ```
pub struct RandomForest {
    n_estimators: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    /// Fraction of features to consider at each split. 0.0 = sqrt(n_features) heuristic.
    max_features_fraction: f64,
    seed: u64,
}

impl Default for RandomForest {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features_fraction: 0.0, // sqrt heuristic
            seed: 42,
        }
    }
}

impl RandomForest {
    /// Creates a Random Forest learner with default hyperparameters (100 trees, sqrt feature heuristic).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of trees in the forest.
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    /// Sets the maximum depth of each tree.
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

    /// Uses the classic RF default: `sqrt(n_features)` candidate features per split.
    pub fn with_max_features_sqrt(mut self) -> Self {
        self.max_features_fraction = 0.0;
        self
    }

    /// Sets the fraction of features considered at each split; `0.0` uses the sqrt(n_features) heuristic.
    pub fn with_max_features_fraction(mut self, f: f64) -> Self {
        self.max_features_fraction = f;
        self
    }

    /// Sets the RNG seed used for bootstrap sampling and feature subsampling.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn max_features_count(&self, n_features: usize) -> usize {
        if self.max_features_fraction <= 0.0 {
            // sqrt heuristic
            (n_features as f64).sqrt().ceil() as usize
        } else {
            (n_features as f64 * self.max_features_fraction)
                .ceil()
                .max(1.0) as usize
        }
    }
}

use serde::{Deserialize, Serialize};

/// A trained Random Forest ensemble, ready to predict.
#[derive(Serialize, Deserialize)]
pub struct TrainedRandomForest {
    pub(crate) trees: Vec<Node>,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_importances: Vec<f64>,
    pub(crate) n_classes: Option<usize>,
    pub(crate) is_classifier: bool,
}

impl TrainedModel for TrainedRandomForest {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        if self.is_classifier {
            let n_classes = self.n_classes.unwrap();
            let n_trees = self.trees.len() as f64;

            let results: Vec<(usize, Vec<f64>)> = (0..features.nrows())
                .into_par_iter()
                .map(|i| {
                    let row = features.row(i);
                    let mut avg_probs = vec![0.0; n_classes];
                    for tree in &self.trees {
                        match tree.predict_one(row) {
                            LeafValue::Class(_, probs) => {
                                for (j, p) in probs.iter().enumerate() {
                                    avg_probs[j] += p;
                                }
                            }
                            _ => unreachable!(),
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
                    (pred_class, avg_probs)
                })
                .collect();

            let mut predicted = Vec::with_capacity(results.len());
            let mut probabilities = Vec::with_capacity(results.len());
            for (pred, prob) in results {
                predicted.push(pred);
                probabilities.push(prob);
            }

            Ok(Prediction::Classification {
                predicted,
                truth: None,
                probabilities: Some(probabilities),
            })
        } else {
            let n_trees = self.trees.len() as f64;
            let predicted: Vec<f64> = (0..features.nrows())
                .into_par_iter()
                .map(|i| {
                    let row = features.row(i);
                    let sum: f64 = self
                        .trees
                        .iter()
                        .map(|tree| match tree.predict_one(row) {
                            LeafValue::Value(v) => *v,
                            _ => unreachable!(),
                        })
                        .sum();
                    sum / n_trees
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
}

impl Learner for RandomForest {
    fn id(&self) -> &str {
        "random_forest"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();
        let n_classes = task.n_classes();
        let max_feat = self.max_features_count(n_features);

        let results: Vec<(Node, Vec<f64>)> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64));
                // Bootstrap sample (with replacement)
                let indices: Vec<usize> = (0..n_samples)
                    .map(|_| rng.random_range(0..n_samples))
                    .collect();

                let mut builder = TreeBuilder::new(
                    self.max_depth,
                    self.min_samples_split,
                    self.min_samples_leaf,
                    Some(max_feat),
                    n_features,
                );
                let root = builder.build_classifier(
                    &features.view(),
                    target,
                    &indices,
                    n_classes,
                    0,
                    &mut rng,
                );
                (root, builder.feature_importances)
            })
            .collect();

        let mut total_importances = vec![0.0; n_features];
        let mut trees = Vec::with_capacity(self.n_estimators);
        for (root, importances) in results {
            for (j, imp) in importances.iter().enumerate() {
                total_importances[j] += imp;
            }
            trees.push(root);
        }

        Ok(Box::new(TrainedRandomForest {
            trees,
            feature_names: task.feature_names().to_vec(),
            feature_importances: total_importances,
            n_classes: Some(n_classes),
            is_classifier: true,
        }))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();
        let max_feat = self.max_features_count(n_features);

        let results: Vec<(Node, Vec<f64>)> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64));
                let indices: Vec<usize> = (0..n_samples)
                    .map(|_| rng.random_range(0..n_samples))
                    .collect();

                let mut builder = TreeBuilder::new(
                    self.max_depth,
                    self.min_samples_split,
                    self.min_samples_leaf,
                    Some(max_feat),
                    n_features,
                );
                let root = builder.build_regressor(&features.view(), target, &indices, 0, &mut rng);
                (root, builder.feature_importances)
            })
            .collect();

        let mut total_importances = vec![0.0; n_features];
        let mut trees = Vec::with_capacity(self.n_estimators);
        for (root, importances) in results {
            for (j, imp) in importances.iter().enumerate() {
                total_importances[j] += imp;
            }
            trees.push(root);
        }

        Ok(Box::new(TrainedRandomForest {
            trees,
            feature_names: task.feature_names().to_vec(),
            feature_importances: total_importances,
            n_classes: None,
            is_classifier: false,
        }))
    }
}
