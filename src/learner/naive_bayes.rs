//! Gaussian Naive Bayes classifier.

use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::prediction::Prediction;
use crate::Result;
use crate::task::{ClassificationTask, Task};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Gaussian Naive Bayes classifier.
///
/// Assumes features are normally distributed within each class.
/// Fast training and prediction, works well as a baseline.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [6.0, 9.0]];
/// let target = vec![0, 0, 1, 1];
/// let task = ClassificationTask::new("nb", features, target).unwrap();
///
/// let mut nb = GaussianNB::new();
/// let model = nb.train_classif(&task).unwrap();
/// ```
pub struct GaussianNB;

impl GaussianNB {
    /// Creates a Gaussian Naive Bayes classifier.
    pub fn new() -> Self {
        Self
    }
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self
    }
}

/// A trained Gaussian Naive Bayes classifier, ready to predict.
#[derive(Clone, Serialize, Deserialize)]
pub struct TrainedGaussianNB {
    pub(crate) means: Vec<Vec<f64>>,     // [class][feature]
    pub(crate) variances: Vec<Vec<f64>>, // [class][feature]
    pub(crate) class_priors: Vec<f64>,   // P(class)
    pub(crate) n_classes: usize,
    pub(crate) feature_names: Vec<String>,
}

fn gaussian_log_pdf(x: f64, mean: f64, var: f64) -> f64 {
    let var = var.max(1e-9); // avoid division by zero
    -0.5 * ((x - mean).powi(2) / var + var.ln() + std::f64::consts::TAU.ln())
}

impl TrainedModel for TrainedGaussianNB {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        crate::validate::check_n_features(features, self.feature_names.len())?;
        let mut predicted = Vec::with_capacity(features.nrows());
        let mut probabilities = Vec::with_capacity(features.nrows());

        for row in features.rows() {
            let mut log_posteriors = vec![0.0; self.n_classes];
            for c in 0..self.n_classes {
                log_posteriors[c] = self.class_priors[c].ln();
                for (j, &x) in row.iter().enumerate() {
                    log_posteriors[c] +=
                        gaussian_log_pdf(x, self.means[c][j], self.variances[c][j]);
                }
            }

            // Convert to probabilities via log-sum-exp
            let max_lp = log_posteriors
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = log_posteriors.iter().map(|&lp| (lp - max_lp).exp()).sum();
            let probs: Vec<f64> = log_posteriors
                .iter()
                .map(|&lp| (lp - max_lp).exp() / exp_sum)
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

    fn to_serializable(&self) -> Option<crate::serialize::SerializableModel> {
        Some(crate::serialize::SerializableModel::GaussianNB(
            self.clone(),
        ))
    }
}

impl Learner for GaussianNB {
    fn id(&self) -> &str {
        "gaussian_nb"
    }

    fn properties(&self) -> LearnerProperties {
        LearnerProperties::classifier()
            .with_proba()
            .with_serializable()
    }

    fn train_classif(&mut self, task: &ClassificationTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "GaussianNB")?;
        crate::validate::check_no_nan(task.features())?;
        let features = task.features();
        let target = task.target();
        let n_classes = task.n_classes();
        let n_features = task.n_features();
        let n_samples = task.n_samples() as f64;

        let mut means = vec![vec![0.0; n_features]; n_classes];
        let mut variances = vec![vec![0.0; n_features]; n_classes];
        let mut counts = vec![0usize; n_classes];

        // Compute per-class means
        for (i, &c) in target.iter().enumerate() {
            counts[c] += 1;
            for j in 0..n_features {
                means[c][j] += features[[i, j]];
            }
        }
        for c in 0..n_classes {
            if counts[c] > 0 {
                for j in 0..n_features {
                    means[c][j] /= counts[c] as f64;
                }
            }
        }

        // Compute per-class variances
        for (i, &c) in target.iter().enumerate() {
            for j in 0..n_features {
                variances[c][j] += (features[[i, j]] - means[c][j]).powi(2);
            }
        }
        for c in 0..n_classes {
            if counts[c] > 0 {
                for j in 0..n_features {
                    variances[c][j] /= counts[c] as f64;
                }
            }
        }

        let class_priors: Vec<f64> = counts.iter().map(|&c| c as f64 / n_samples).collect();

        Ok(Box::new(TrainedGaussianNB {
            means,
            variances,
            class_priors,
            n_classes,
            feature_names: task.feature_names().to_vec(),
        }))
    }
}
