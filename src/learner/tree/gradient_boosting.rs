//! Gradient Boosting: sequential ensemble of shallow regression trees.
//!
//! Regression uses MSE loss (residual fitting). Classification uses log-loss
//! with sigmoid (binary) or softmax (multiclass).

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use crate::task::{ClassificationTask, RegressionTask, Task};
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;
use super::{Node, LeafValue, TreeBuilder};

/// Gradient Boosting learner.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[1.0], [2.0], [3.0], [4.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0];
/// let task = RegressionTask::new("gb_demo", features, target).unwrap();
///
/// let mut gb = GradientBoosting::new()
///     .with_n_estimators(50)
///     .with_learning_rate(0.1);
/// let model = gb.train_regress(&task).unwrap();
/// ```
pub struct GradientBoosting {
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    subsample: f64,
    seed: u64,
}

impl Default for GradientBoosting {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            seed: 42,
        }
    }
}

impl GradientBoosting {
    pub fn new() -> Self { Self::default() }

    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

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

    pub fn with_subsample(mut self, ratio: f64) -> Self {
        self.subsample = ratio;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn subsample_indices(&self, n_samples: usize, rng: &mut StdRng) -> Vec<usize> {
        if self.subsample >= 1.0 {
            return (0..n_samples).collect();
        }
        let k = (n_samples as f64 * self.subsample).ceil() as usize;
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(rng);
        indices.truncate(k);
        indices
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// --- Internal mode ---

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub(crate) enum GBMode {
    Regression,
    BinaryClassif,
    MultiClassif { n_classes: usize },
}

#[derive(Serialize, Deserialize)]
pub struct TrainedGradientBoosting {
    pub(crate) initial: Vec<f64>,
    pub(crate) trees: Vec<Node>,
    pub(crate) learning_rate: f64,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_importances: Vec<f64>,
    pub(crate) mode: GBMode,
}

impl TrainedModel for TrainedGradientBoosting {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        match &self.mode {
            GBMode::Regression => {
                let predicted: Vec<f64> = features.rows().into_iter()
                    .map(|row| {
                        let mut val = self.initial[0];
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
            GBMode::BinaryClassif => {
                let mut predicted = Vec::with_capacity(features.nrows());
                let mut probabilities = Vec::with_capacity(features.nrows());

                for row in features.rows() {
                    let mut f = self.initial[0];
                    for tree in &self.trees {
                        if let LeafValue::Value(v) = tree.predict_one(row) {
                            f += self.learning_rate * v;
                        }
                    }
                    let prob = sigmoid(f);
                    predicted.push(if prob >= 0.5 { 1 } else { 0 });
                    probabilities.push(vec![1.0 - prob, prob]);
                }

                Ok(Prediction::Classification {
                    predicted,
                    truth: None,
                    probabilities: Some(probabilities),
                })
            }
            GBMode::MultiClassif { n_classes } => {
                let k = *n_classes;
                let mut predicted = Vec::with_capacity(features.nrows());
                let mut probabilities = Vec::with_capacity(features.nrows());

                for row in features.rows() {
                    let mut scores = self.initial.clone();
                    // trees are stored flat: iteration i, class c => trees[i * k + c]
                    let n_iters = self.trees.len() / k;
                    for iter in 0..n_iters {
                        for c in 0..k {
                            if let LeafValue::Value(v) = self.trees[iter * k + c].predict_one(row) {
                                scores[c] += self.learning_rate * v;
                            }
                        }
                    }
                    let probs = softmax(&scores);
                    let pred_class = probs.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap().0;
                    predicted.push(pred_class);
                    probabilities.push(probs);
                }

                Ok(Prediction::Classification {
                    predicted,
                    truth: None,
                    probabilities: Some(probabilities),
                })
            }
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

impl Learner for GradientBoosting {
    fn id(&self) -> &str { "gradient_boosting" }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let n_classes = task.n_classes();
        if n_classes == 2 {
            self.train_binary(task)
        } else {
            self.train_multiclass(task)
        }
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();

        let initial = target.iter().sum::<f64>() / n_samples as f64;
        let mut current_preds = vec![initial; n_samples];
        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut total_importances = vec![0.0; n_features];
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _ in 0..self.n_estimators {
            let residuals: Vec<f64> = target.iter()
                .zip(&current_preds)
                .map(|(y, p)| y - p)
                .collect();

            let indices = self.subsample_indices(n_samples, &mut rng);

            let mut builder = TreeBuilder::new(
                self.max_depth, self.min_samples_split, self.min_samples_leaf,
                None, n_features,
            );
            let root = builder.build_regressor(
                &features.view(), &residuals, &indices, 0, &mut rng,
            );

            // Update predictions
            for i in 0..n_samples {
                if let LeafValue::Value(v) = root.predict_one(features.row(i)) {
                    current_preds[i] += self.learning_rate * v;
                }
            }

            for (j, imp) in builder.feature_importances.iter().enumerate() {
                total_importances[j] += imp;
            }
            trees.push(root);
        }

        Ok(Box::new(TrainedGradientBoosting {
            initial: vec![initial],
            trees,
            learning_rate: self.learning_rate,
            feature_names: task.feature_names().to_vec(),
            feature_importances: total_importances,
            mode: GBMode::Regression,
        }))
    }
}

impl GradientBoosting {
    fn train_binary(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();

        // Initial log-odds
        let p_pos = target.iter().filter(|&&t| t == 1).count() as f64 / n_samples as f64;
        let initial = (p_pos / (1.0 - p_pos).max(1e-15)).ln();
        let mut current_f = vec![initial; n_samples];
        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut total_importances = vec![0.0; n_features];
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _ in 0..self.n_estimators {
            // Pseudo-residuals: y - sigmoid(F)
            let residuals: Vec<f64> = target.iter()
                .zip(&current_f)
                .map(|(&y, &f)| y as f64 - sigmoid(f))
                .collect();

            let indices = self.subsample_indices(n_samples, &mut rng);

            let mut builder = TreeBuilder::new(
                self.max_depth, self.min_samples_split, self.min_samples_leaf,
                None, n_features,
            );
            let root = builder.build_regressor(
                &features.view(), &residuals, &indices, 0, &mut rng,
            );

            for i in 0..n_samples {
                if let LeafValue::Value(v) = root.predict_one(features.row(i)) {
                    current_f[i] += self.learning_rate * v;
                }
            }

            for (j, imp) in builder.feature_importances.iter().enumerate() {
                total_importances[j] += imp;
            }
            trees.push(root);
        }

        Ok(Box::new(TrainedGradientBoosting {
            initial: vec![initial],
            trees,
            learning_rate: self.learning_rate,
            feature_names: task.feature_names().to_vec(),
            feature_importances: total_importances,
            mode: GBMode::BinaryClassif,
        }))
    }

    fn train_multiclass(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n_samples = task.n_samples();
        let n_features = task.n_features();
        let n_classes = task.n_classes();

        // Initial log-proportions per class
        let mut class_counts = vec![0usize; n_classes];
        for &t in target {
            class_counts[t] += 1;
        }
        let initial: Vec<f64> = class_counts.iter()
            .map(|&c| ((c as f64 / n_samples as f64).max(1e-15)).ln())
            .collect();

        // Current raw scores: [sample][class]
        let mut current_f: Vec<Vec<f64>> = (0..n_samples)
            .map(|_| initial.clone())
            .collect();

        let mut trees = Vec::with_capacity(self.n_estimators * n_classes);
        let mut total_importances = vec![0.0; n_features];
        let mut rng = StdRng::seed_from_u64(self.seed);

        for _ in 0..self.n_estimators {
            // Compute softmax probabilities for all samples
            let probs: Vec<Vec<f64>> = current_f.iter().map(|f| softmax(f)).collect();

            let indices = self.subsample_indices(n_samples, &mut rng);

            // One tree per class
            for c in 0..n_classes {
                // Pseudo-residuals: y_ic - p_ic
                let residuals: Vec<f64> = (0..n_samples)
                    .map(|i| {
                        let y_ic = if target[i] == c { 1.0 } else { 0.0 };
                        y_ic - probs[i][c]
                    })
                    .collect();

                let mut builder = TreeBuilder::new(
                    self.max_depth, self.min_samples_split, self.min_samples_leaf,
                    None, n_features,
                );
                let root = builder.build_regressor(
                    &features.view(), &residuals, &indices, 0, &mut rng,
                );

                // Update scores for this class
                for i in 0..n_samples {
                    if let LeafValue::Value(v) = root.predict_one(features.row(i)) {
                        current_f[i][c] += self.learning_rate * v;
                    }
                }

                for (j, imp) in builder.feature_importances.iter().enumerate() {
                    total_importances[j] += imp;
                }
                trees.push(root);
            }
        }

        Ok(Box::new(TrainedGradientBoosting {
            initial,
            trees,
            learning_rate: self.learning_rate,
            feature_names: task.feature_names().to_vec(),
            feature_importances: total_importances,
            mode: GBMode::MultiClassif { n_classes },
        }))
    }
}
