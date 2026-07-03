//! Logistic Regression via gradient descent.
//!
//! Binary classification with sigmoid. Multiclass via one-vs-rest.

use crate::learner::math::sigmoid;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;
use crate::task::{ClassificationTask, Task};
use ndarray::{Array1, Array2};

/// Logistic Regression learner.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::LogisticRegression;
/// use ndarray::array;
///
/// let features = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
/// let target = vec![0, 0, 1, 1]; // class depends on x0
/// let task = ClassificationTask::new("lr_demo", features, target).unwrap();
///
/// let mut lr = LogisticRegression::default();
/// let model = lr.train_classif(&task).unwrap();
/// ```
pub struct LogisticRegression {
    learning_rate: f64,
    max_iter: usize,
    tol: f64,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            max_iter: 1000,
            tol: 1e-6,
        }
    }
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }
}

/// Train a single binary logistic regression (positive class vs rest).
fn train_binary(
    x: &Array2<f64>,
    y_binary: &[f64], // 1.0 for positive, 0.0 for negative
    lr: f64,
    max_iter: usize,
    tol: f64,
) -> Array1<f64> {
    let n = x.nrows();
    let p = x.ncols();
    // weights = [w1, ..., wp, bias]
    let mut w = Array1::zeros(p + 1);

    for _ in 0..max_iter {
        let mut grad: Array1<f64> = Array1::zeros(p + 1);
        let mut max_grad = 0.0f64;

        for i in 0..n {
            let mut z = w[p]; // bias
            for j in 0..p {
                z += x[[i, j]] * w[j];
            }
            let pred = sigmoid(z);
            let err = pred - y_binary[i];
            for j in 0..p {
                grad[j] += err * x[[i, j]];
            }
            grad[p] += err; // bias gradient
        }

        // Average and update
        for j in 0..=p {
            grad[j] /= n as f64;
            max_grad = max_grad.max(grad[j].abs());
            w[j] -= lr * grad[j];
        }

        if max_grad < tol {
            break;
        }
    }

    w
}

// --- Trained model ---

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct TrainedLogisticRegression {
    pub(crate) classifiers: Vec<Array1<f64>>,
    pub(crate) n_classes: usize,
    pub(crate) feature_names: Vec<String>,
    /// Internal scaling parameters (mean, std per feature). Applied automatically.
    pub(crate) scale_means: Vec<f64>,
    pub(crate) scale_stds: Vec<f64>,
}

impl TrainedLogisticRegression {
    /// Apply internal scaling to a row.
    #[inline]
    fn scale_value(&self, j: usize, val: f64) -> f64 {
        (val - self.scale_means[j]) / self.scale_stds[j]
    }
}

impl TrainedModel for TrainedLogisticRegression {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        let p = features.ncols();
        let mut predicted = Vec::with_capacity(features.nrows());
        let mut probabilities = Vec::with_capacity(features.nrows());

        for row in features.rows() {
            if self.n_classes == 2 {
                let w = &self.classifiers[0];
                let mut z = w[p]; // bias
                for j in 0..p {
                    z += self.scale_value(j, row[j]) * w[j];
                }
                let prob = sigmoid(z);
                predicted.push(if prob >= 0.5 { 1 } else { 0 });
                probabilities.push(vec![1.0 - prob, prob]);
            } else {
                // Multiclass OVR: pick class with highest score
                let scores: Vec<f64> = self
                    .classifiers
                    .iter()
                    .map(|w| {
                        let mut z = w[p];
                        for j in 0..p {
                            z += self.scale_value(j, row[j]) * w[j];
                        }
                        sigmoid(z)
                    })
                    .collect();

                let pred_class = scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
                    .0;
                predicted.push(pred_class);

                // Normalize scores to probabilities
                let total: f64 = scores.iter().sum();
                let probs = if total > 0.0 {
                    scores.iter().map(|&s| s / total).collect()
                } else {
                    vec![1.0 / self.n_classes as f64; self.n_classes]
                };
                probabilities.push(probs);
            }
        }

        Ok(Prediction::Classification {
            predicted,
            truth: None,
            probabilities: Some(probabilities),
        })
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        let p = self.feature_names.len();
        // Average absolute weights across all classifiers
        let mut importance = vec![0.0f64; p];
        for w in &self.classifiers {
            for j in 0..p {
                importance[j] += w[j].abs();
            }
        }
        let n_classifiers = self.classifiers.len() as f64;
        let total: f64 = importance.iter().sum();
        if total == 0.0 {
            return None;
        }
        Some(
            self.feature_names
                .iter()
                .zip(&importance)
                .map(|(name, &imp)| (name.clone(), imp / n_classifiers / total * n_classifiers))
                .collect(),
        )
    }
}

// --- Learner impl ---

impl Learner for LogisticRegression {
    fn id(&self) -> &str {
        "logistic_regression"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_nan(task.features())?;
        let x = task.features();
        let target = task.target();
        let n_classes = task.n_classes();
        let n_features = task.n_features();
        let n_samples = task.n_samples() as f64;

        // Auto-scale features (standardization)
        let mut means = vec![0.0; n_features];
        let mut stds = vec![0.0; n_features];
        for j in 0..n_features {
            let col = x.column(j);
            let mean = col.sum() / n_samples;
            let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_samples;
            means[j] = mean;
            stds[j] = if var > 0.0 { var.sqrt() } else { 1.0 };
        }

        // Create scaled feature matrix
        let mut x_scaled = x.clone();
        for i in 0..x.nrows() {
            for j in 0..n_features {
                x_scaled[[i, j]] = (x[[i, j]] - means[j]) / stds[j];
            }
        }

        let classifiers = if n_classes == 2 {
            let y_binary: Vec<f64> = target
                .iter()
                .map(|&t| if t == 1 { 1.0 } else { 0.0 })
                .collect();
            vec![train_binary(
                &x_scaled,
                &y_binary,
                self.learning_rate,
                self.max_iter,
                self.tol,
            )]
        } else {
            (0..n_classes)
                .map(|c| {
                    let y_binary: Vec<f64> = target
                        .iter()
                        .map(|&t| if t == c { 1.0 } else { 0.0 })
                        .collect();
                    train_binary(
                        &x_scaled,
                        &y_binary,
                        self.learning_rate,
                        self.max_iter,
                        self.tol,
                    )
                })
                .collect()
        };

        Ok(Box::new(TrainedLogisticRegression {
            classifiers,
            n_classes,
            feature_names: task.feature_names().to_vec(),
            scale_means: means,
            scale_stds: stds,
        }))
    }
}
