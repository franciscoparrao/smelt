//! AdaBoost (Adaptive Boosting) classifier via SAMME algorithm.

use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;
use crate::task::{ClassificationTask, Task};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// AdaBoost classifier using decision stumps (depth-1 trees).
///
/// Iteratively trains weak learners on weighted data, focusing on
/// previously misclassified samples. Uses the SAMME algorithm for
/// multiclass support.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5]];
/// let target = vec![0, 0, 0, 0, 1, 1, 1, 1];
/// let task = ClassificationTask::new("ada", features, target).unwrap();
///
/// let mut ada = AdaBoost::new().with_n_estimators(50);
/// let model = ada.train_classif(&task).unwrap();
/// ```
pub struct AdaBoost {
    n_estimators: usize,
    learning_rate: f64,
}

impl Default for AdaBoost {
    fn default() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
        }
    }
}

impl AdaBoost {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
}

#[derive(Serialize, Deserialize)]
pub struct TrainedAdaBoost {
    pub(crate) stumps: Vec<TrainedStump>,
    pub(crate) alphas: Vec<f64>,
    pub(crate) n_classes: usize,
    pub(crate) feature_names: Vec<String>,
}

/// A trained decision stump (depth-1 tree) stored as a simple split.
#[derive(Serialize, Deserialize)]
pub struct TrainedStump {
    feature: usize,
    threshold: f64,
    left_class: usize,
    right_class: usize,
}

impl TrainedStump {
    fn predict_one(&self, row: &[f64]) -> usize {
        if row[self.feature] <= self.threshold {
            self.left_class
        } else {
            self.right_class
        }
    }
}

/// Train a weighted decision stump — find the single best split.
fn train_stump(
    features: &Array2<f64>,
    target: &[usize],
    weights: &[f64],
    n_classes: usize,
) -> (TrainedStump, f64) {
    let n = features.nrows();
    let p = features.ncols();
    let mut best_err = f64::INFINITY;
    let mut best_stump = TrainedStump {
        feature: 0,
        threshold: 0.0,
        left_class: 0,
        right_class: 0,
    };

    for feat in 0..p {
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_by(|&a, &b| {
            features[[a, feat]]
                .partial_cmp(&features[[b, feat]])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for i in 1..n {
            if (features[[sorted[i], feat]] - features[[sorted[i - 1], feat]]).abs() < f64::EPSILON
            {
                continue;
            }

            let threshold = (features[[sorted[i - 1], feat]] + features[[sorted[i], feat]]) / 2.0;

            // Weighted majority class for left and right
            let mut left_counts = vec![0.0; n_classes];
            let mut right_counts = vec![0.0; n_classes];
            for &idx in &sorted[..i] {
                left_counts[target[idx]] += weights[idx];
            }
            for &idx in &sorted[i..] {
                right_counts[target[idx]] += weights[idx];
            }

            let left_class = left_counts
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            let right_class = right_counts
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;

            // Weighted error
            let mut err = 0.0;
            for idx in 0..n {
                let pred = if features[[idx, feat]] <= threshold {
                    left_class
                } else {
                    right_class
                };
                if pred != target[idx] {
                    err += weights[idx];
                }
            }

            if err < best_err {
                best_err = err;
                best_stump = TrainedStump {
                    feature: feat,
                    threshold,
                    left_class,
                    right_class,
                };
            }
        }
    }

    (best_stump, best_err)
}

impl TrainedModel for TrainedAdaBoost {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        let mut predicted = Vec::with_capacity(features.nrows());
        let mut probabilities = Vec::with_capacity(features.nrows());

        for row in features.rows() {
            let row_slice: Vec<f64> = row.to_vec();
            let mut class_scores = vec![0.0; self.n_classes];
            for (stump, &alpha) in self.stumps.iter().zip(&self.alphas) {
                let pred = stump.predict_one(&row_slice);
                class_scores[pred] += alpha;
            }

            // Softmax for probabilities
            let max_s = class_scores
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = class_scores.iter().map(|&s| (s - max_s).exp()).sum();
            let probs: Vec<f64> = class_scores
                .iter()
                .map(|&s| (s - max_s).exp() / exp_sum)
                .collect();

            let pred_class = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
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

impl Learner for AdaBoost {
    fn id(&self) -> &str {
        "adaboost"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        let features = task.features();
        let target = task.target();
        let n = task.n_samples();
        let n_classes = task.n_classes();

        let mut weights = vec![1.0 / n as f64; n];
        let mut stumps = Vec::with_capacity(self.n_estimators);
        let mut alphas = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            let (stump, err) = train_stump(features, target, &weights, n_classes);

            if err >= 1.0 - 1e-10 / n_classes as f64 {
                break;
            } // can't improve
            let err = err.max(1e-10); // avoid log(0)

            // SAMME alpha
            let alpha =
                self.learning_rate * ((1.0 - err) / err).ln() + (n_classes as f64 - 1.0).ln();

            if alpha <= 0.0 {
                break;
            }

            // Update weights
            for i in 0..n {
                let row: Vec<f64> = features.row(i).to_vec();
                let pred = stump.predict_one(&row);
                if pred != target[i] {
                    weights[i] *= (alpha).exp();
                }
            }
            // Normalize
            let w_sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= w_sum;
            }

            stumps.push(stump);
            alphas.push(alpha);
        }

        Ok(Box::new(TrainedAdaBoost {
            stumps,
            alphas,
            n_classes,
            feature_names: task.feature_names().to_vec(),
        }))
    }
}
