//! Linear Support Vector Machine via Stochastic Gradient Descent.

use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;
use crate::task::{ClassificationTask, Task};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

/// Linear SVM classifier via SGD with hinge loss.
///
/// For multiclass, uses one-vs-rest strategy.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[0.0, 0.0], [0.1, 0.1], [1.0, 1.0], [1.1, 0.9]];
/// let target = vec![0, 0, 1, 1];
/// let task = ClassificationTask::new("svm", features, target).unwrap();
///
/// let mut svm = LinearSVM::new();
/// let model = svm.train_classif(&task).unwrap();
/// ```
pub struct LinearSVM {
    c: f64, // regularization (higher = less regularization)
    max_iter: usize,
    learning_rate: f64,
    seed: u64,
}

impl Default for LinearSVM {
    fn default() -> Self {
        Self {
            c: 1.0,
            max_iter: 1000,
            learning_rate: 0.01,
            seed: 42,
        }
    }
}

impl LinearSVM {
    /// Creates a Linear SVM with C=1.0, 1000 SGD iterations, and learning rate 0.01.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the regularization strength C (higher = less regularization).
    pub fn with_c(mut self, c: f64) -> Self {
        self.c = c;
        self
    }
    /// Sets the maximum number of SGD epochs.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }
    /// Sets the initial SGD learning rate (decayed each epoch).
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    /// Sets the RNG seed used to shuffle samples during SGD.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Train a binary SVM: y_i in {-1, +1}
fn train_binary_svm(
    x: &Array2<f64>,
    y: &[f64],
    c: f64,
    lr: f64,
    max_iter: usize,
    seed: u64,
) -> Array1<f64> {
    let n = x.nrows();
    let p = x.ncols();
    let mut w: Array1<f64> = Array1::zeros(p + 1); // [w0..wp, bias]
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();

    // Per-sample regularization strength. The SVM objective is
    // ½‖w‖² + C·Σᵢ hinge(i); with per-sample SGD steps the weight decay
    // must be λ = 1/(n·C) per step (Pegasos / sklearn's LinearSVC-through-
    // SGDClassifier convention, alpha = 1/(n·C)). Using 1/C per step — the
    // pre-4th-audit behaviour — applied n times more regularization than
    // the objective asks for: with the defaults, ‖w‖ stayed pinned near
    // zero and training accuracy sat at chance level even on trivially
    // separable data.
    let lambda = 1.0 / (c * n as f64);

    for epoch in 0..max_iter {
        indices.shuffle(&mut rng);
        let eta = lr / (1.0 + epoch as f64 * 0.01); // learning rate decay

        for &i in &indices {
            let mut score = w[p]; // bias
            for j in 0..p {
                score += w[j] * x[[i, j]];
            }

            if y[i] * score < 1.0 {
                // Misclassified or within margin: hinge loss gradient
                for j in 0..p {
                    w[j] = w[j] * (1.0 - eta * lambda) + eta * y[i] * x[[i, j]];
                }
                w[p] += eta * y[i]; // bias update (no regularization)
            } else {
                // Correctly classified: only regularization
                for j in 0..p {
                    w[j] *= 1.0 - eta * lambda;
                }
            }
        }
    }

    w
}

/// A trained Linear SVM (one binary or one-vs-rest classifier per class), ready to predict.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedLinearSVM {
    pub(crate) classifiers: Vec<Array1<f64>>, // one per class (OVR)
    pub(crate) n_classes: usize,
    pub(crate) feature_names: Vec<String>,
    /// Internal scaling parameters (mean, std per feature), applied
    /// automatically — same standardization LogisticRegression and ELM
    /// do, without which hinge-SGD stalls on real-scale features (UTM
    /// coordinates, raw counts). `serde(default)`: models saved before
    /// these fields existed load with empty vecs, which `scale_value`
    /// treats as identity (those models were trained unscaled).
    #[serde(default)]
    pub(crate) scale_means: Vec<f64>,
    #[serde(default)]
    pub(crate) scale_stds: Vec<f64>,
}

impl TrainedLinearSVM {
    /// Apply internal scaling to one value (identity for legacy models
    /// deserialized without scaling parameters).
    #[inline]
    fn scale_value(&self, j: usize, val: f64) -> f64 {
        if self.scale_stds.is_empty() {
            val
        } else {
            (val - self.scale_means[j]) / self.scale_stds[j]
        }
    }
}

impl TrainedModel for TrainedLinearSVM {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        let p = features.ncols();
        let mut predicted = Vec::with_capacity(features.nrows());
        let mut probabilities = Vec::with_capacity(features.nrows());

        for row in features.rows() {
            if self.n_classes == 2 {
                let w = &self.classifiers[0];
                let mut score = w[p];
                for j in 0..p {
                    score += self.scale_value(j, row[j]) * w[j];
                }
                let pred = if score >= 0.0 { 1 } else { 0 };
                // Approximate probability via sigmoid of score
                let prob = 1.0 / (1.0 + (-score).exp());
                predicted.push(pred);
                probabilities.push(vec![1.0 - prob, prob]);
            } else {
                let scores: Vec<f64> = self
                    .classifiers
                    .iter()
                    .map(|w| {
                        let mut s = w[p];
                        for j in 0..p {
                            s += self.scale_value(j, row[j]) * w[j];
                        }
                        s
                    })
                    .collect();

                let pred_class = scores
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
                    .0;
                predicted.push(pred_class);

                // Softmax for probabilities
                let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = scores.iter().map(|&s| (s - max_s).exp()).sum();
                let probs: Vec<f64> = scores
                    .iter()
                    .map(|&s| (s - max_s).exp() / exp_sum)
                    .collect();
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
        let mut importance = vec![0.0; p];
        for w in &self.classifiers {
            for j in 0..p {
                importance[j] += w[j].abs();
            }
        }
        let total: f64 = importance.iter().sum();
        if total == 0.0 {
            return None;
        }
        Some(
            self.feature_names
                .iter()
                .zip(&importance)
                .map(|(name, &imp)| (name.clone(), imp / total))
                .collect(),
        )
    }

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::LinearSVM(
            self.clone(),
        ))
    }
}

impl Learner for LinearSVM {
    fn id(&self) -> &str {
        "linear_svm"
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_nan(task.features())?;
        let x = task.features();
        let target = task.target();
        let n_classes = task.n_classes();
        let n_features = task.n_features();
        let n_samples = task.n_samples() as f64;

        // Auto-scale features (standardization) — same as LogisticRegression
        // and ELM: hinge-SGD with a fixed learning rate stalls (or diverges)
        // on unnormalized magnitudes like UTM coordinates.
        let mut means = vec![0.0; n_features];
        let mut stds = vec![0.0; n_features];
        for j in 0..n_features {
            let col = x.column(j);
            let mean = col.sum() / n_samples;
            let var = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_samples;
            means[j] = mean;
            stds[j] = if var > 0.0 { var.sqrt() } else { 1.0 };
        }
        let mut x_scaled = x.clone();
        for i in 0..x.nrows() {
            for j in 0..n_features {
                x_scaled[[i, j]] = (x[[i, j]] - means[j]) / stds[j];
            }
        }

        let classifiers = if n_classes == 2 {
            let y: Vec<f64> = target
                .iter()
                .map(|&t| if t == 1 { 1.0 } else { -1.0 })
                .collect();
            vec![train_binary_svm(
                &x_scaled,
                &y,
                self.c,
                self.learning_rate,
                self.max_iter,
                self.seed,
            )]
        } else {
            (0..n_classes)
                .map(|c| {
                    let y: Vec<f64> = target
                        .iter()
                        .map(|&t| if t == c { 1.0 } else { -1.0 })
                        .collect();
                    train_binary_svm(
                        &x_scaled,
                        &y,
                        self.c,
                        self.learning_rate,
                        self.max_iter,
                        self.seed.wrapping_add(c as u64),
                    )
                })
                .collect()
        };

        Ok(Box::new(TrainedLinearSVM {
            classifiers,
            n_classes,
            feature_names: task.feature_names().to_vec(),
            scale_means: means,
            scale_stds: stds,
        }))
    }
}
