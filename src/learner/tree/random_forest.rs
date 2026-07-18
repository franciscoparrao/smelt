//! Random Forest: ensemble of decision trees with bootstrap sampling and feature subsampling.

use super::{LeafValue, MaxFeatures, Node, TreeBuilder};
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
/// Supports per-sample weights: weights enter each tree's impurity and leaf
/// values (sklearn convention), while the bootstrap remains uniform — the
/// weights do not change which rows are sampled. A weight of `0.0` excludes
/// the sample; `min_samples_*` count rows, not total weight.
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
    max_features: MaxFeatures,
    seed: u64,
}

impl Default for RandomForest {
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

impl RandomForest {
    /// Creates a Random Forest learner with default hyperparameters (100
    /// trees; `sqrt(n_features)` per split for classification, all features
    /// for regression -- see [`MaxFeatures`]).
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

    /// Forces the classic `sqrt(n_features)` candidate-feature heuristic for
    /// both classification and regression (the default already uses this for
    /// classification; use this to also apply it to regression, overriding
    /// the task-appropriate default of considering all features).
    pub fn with_max_features_sqrt(mut self) -> Self {
        self.max_features = MaxFeatures::Sqrt;
        self
    }

    /// Sets an explicit fraction of features considered at each split
    /// (applies to both classification and regression, overriding the
    /// task-appropriate default).
    pub fn with_max_features_fraction(mut self, f: f64) -> Self {
        self.max_features = MaxFeatures::Fraction(f);
        self
    }

    /// Sets the RNG seed used for bootstrap sampling and feature subsampling.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

use serde::{Deserialize, Serialize};

/// A trained Random Forest ensemble, ready to predict.
#[derive(Clone, Serialize, Deserialize)]
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

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::RandomForest(
            self.clone(),
        ))
    }
}

impl Learner for RandomForest {
    fn id(&self) -> &str {
        "random_forest"
    }

    /// `true`: weights enter each tree's impurity and leaf values (sklearn
    /// convention). The bootstrap stays **uniform** — weights do NOT change
    /// the resampling probabilities (also sklearn's behavior); a weight of
    /// 0.0 excludes the sample from every tree it is drawn into.
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

        let results: Vec<(Node, Vec<f64>)> = (0..self.n_estimators)
            .into_par_iter()
            .map(|i| {
                let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(i as u64));
                // Bootstrap sample (with replacement). Deliberately uniform
                // even when the task is weighted: weights change the
                // impurity/leaf math, never the sampling distribution.
                let indices: Vec<usize> = (0..n_samples)
                    .map(|_| rng.random_range(0..n_samples))
                    .collect();

                let mut builder = TreeBuilder::new(
                    self.max_depth,
                    self.min_samples_split,
                    self.min_samples_leaf,
                    max_feat,
                    n_features,
                );
                let root = match weights {
                    None => builder.build_classifier(
                        &features.view(),
                        target,
                        &indices,
                        n_classes,
                        0,
                        &mut rng,
                    ),
                    Some(w) => {
                        let indices = super::retain_positive_weight(indices, w, n_samples);
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
        let weights = task.weights();
        let n_samples = task.n_samples();
        let n_features = task.n_features();
        let max_feat = self.max_features.resolve(n_features, false);

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
                    max_feat,
                    n_features,
                );
                let root = match weights {
                    None => {
                        builder.build_regressor(&features.view(), target, &indices, 0, &mut rng)
                    }
                    Some(w) => {
                        let indices = super::retain_positive_weight(indices, w, n_samples);
                        builder.build_regressor_weighted(
                            &features.view(),
                            target,
                            w,
                            &indices,
                            0,
                            &mut rng,
                        )
                    }
                };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prediction::Prediction;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Regression test for a real gap found by an empirical benchmark
    /// against scikit-learn (docs/... benchmark, 2026-07-05): with 48
    /// features where only 3 actually carry signal (like the OpenML `pol`
    /// dataset), forcing the classification-style `sqrt(p)` candidate-
    /// feature heuristic onto regression means many splits never see an
    /// informative feature at all, degrading every tree in the ensemble.
    /// `RandomForest`'s regression default must use all features instead
    /// (matching scikit-learn's `RandomForestRegressor`), confirmed here by
    /// checking it clearly beats the old sqrt(p)-forced behavior on
    /// exactly this failure mode.
    #[test]
    fn regression_default_beats_sqrt_heuristic_when_few_features_are_informative() {
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
            // Only features 0, 1, 2 carry signal; the other 45 are pure noise.
            let y = 5.0 * row[0] - 3.0 * row[1] + 2.0 * row[2] + rng.random::<f64>() * 0.1;
            feats.extend_from_slice(&row);
            target.push(y);
        }
        let features = Array2::from_shape_vec((n, p), feats).unwrap();
        let task = RegressionTask::new("sparse_signal", features.clone(), target.clone()).unwrap();

        let rmse = |predicted: &[f64]| -> f64 {
            (predicted
                .iter()
                .zip(&target)
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / n as f64)
                .sqrt()
        };

        let mut default_rf = RandomForest::new().with_n_estimators(50).with_seed(1);
        let default_model = default_rf.train_regress(&task).unwrap();
        let Prediction::Regression { predicted: default_pred, .. } =
            default_model.predict(&features).unwrap()
        else {
            panic!("expected regression");
        };
        let default_rmse = rmse(&default_pred);

        let mut sqrt_rf = RandomForest::new()
            .with_n_estimators(50)
            .with_seed(1)
            .with_max_features_sqrt();
        let sqrt_model = sqrt_rf.train_regress(&task).unwrap();
        let Prediction::Regression { predicted: sqrt_pred, .. } = sqrt_model.predict(&features).unwrap()
        else {
            panic!("expected regression");
        };
        let sqrt_rmse = rmse(&sqrt_pred);

        assert!(
            default_rmse < sqrt_rmse * 0.8,
            "default (all-features) RMSE {default_rmse} should be clearly better than \
             sqrt(p)-forced RMSE {sqrt_rmse} when only 3 of {p} features are informative"
        );
    }

    #[test]
    fn classification_default_still_uses_sqrt_heuristic() {
        // Sanity check the Auto/is_classif=true branch didn't regress:
        // resolve() should still shrink the candidate set for classification.
        let rf = RandomForest::new();
        let resolved = rf.max_features.resolve(48, true);
        assert_eq!(resolved, Some(7)); // ceil(sqrt(48)) = 7

        let resolved_regress = rf.max_features.resolve(48, false);
        assert_eq!(resolved_regress, None); // all features
    }
}
