//! Conformal Prediction: distribution-free prediction intervals and sets.
//!
//! Model-agnostic wrapper that produces predictions with guaranteed coverage.
//! Works with any trained model without retraining.

pub mod cqr;

use crate::Result;
use crate::learner::TrainedModel;
use crate::prediction::Prediction;
use ndarray::Array2;

/// Conformal prediction result for regression.
#[derive(Debug, Clone)]
pub struct ConformalInterval {
    /// Point prediction.
    pub prediction: f64,
    /// Lower bound of the prediction interval.
    pub lower: f64,
    /// Upper bound of the prediction interval.
    pub upper: f64,
}

/// Conformal prediction result for classification.
#[derive(Debug, Clone)]
pub struct ConformalSet {
    /// Most likely class.
    pub prediction: usize,
    /// Set of classes included in the prediction set (guaranteed coverage).
    pub prediction_set: Vec<usize>,
    /// Probabilities per class.
    pub probabilities: Vec<f64>,
}

/// Calibrated conformal predictor for regression.
///
/// Wraps a trained model and provides prediction intervals with
/// guaranteed 1-α coverage using split conformal prediction.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::conformal::ConformalRegressor;
/// use ndarray::array;
///
/// let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
/// let target = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
/// let task = RegressionTask::new("cf", features.clone(), target.clone()).unwrap();
///
/// let mut dt = DecisionTree::default();
/// let model = dt.train_regress(&task).unwrap();
///
/// // Calibrate on a held-out set (use last 2 samples)
/// let cal_features = array![[5.0], [6.0]];
/// let cal_targets = vec![10.0, 12.0];
///
/// let cf = ConformalRegressor::calibrate(&*model, &cal_features, &cal_targets, 0.1).unwrap();
/// let intervals = cf.predict(&array![[3.5]]).unwrap();
/// ```
pub struct ConformalRegressor<'a> {
    model: &'a dyn TrainedModel,
    quantile_residual: f64, // calibration quantile of |y - ŷ|
}

impl<'a> ConformalRegressor<'a> {
    /// Calibrate the conformal predictor on a held-out calibration set.
    ///
    /// `alpha` is the miscoverage level (e.g., 0.1 for 90% coverage).
    pub fn calibrate(
        model: &'a dyn TrainedModel,
        cal_features: &Array2<f64>,
        cal_targets: &[f64],
        alpha: f64,
    ) -> Result<Self> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(crate::SmeltError::InvalidParameter(format!(
                "conformal alpha must be in (0, 1), got {alpha}"
            )));
        }
        if cal_targets.is_empty() {
            return Err(crate::SmeltError::EmptyDataset);
        }

        let pred = model.predict(cal_features)?;
        let predicted = match &pred {
            Prediction::Regression { predicted, .. } => predicted,
            _ => {
                return Err(crate::SmeltError::Other(
                    "Expected regression prediction".into(),
                ));
            }
        };

        // Compute absolute residuals
        let mut residuals: Vec<f64> = predicted
            .iter()
            .zip(cal_targets)
            .map(|(p, t)| (p - t).abs())
            .collect();
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Quantile rank: ceil((n+1)(1-alpha)). If this exceeds n, the
        // calibration set is too small to guarantee 1-alpha coverage at any
        // finite width — the only choice consistent with the coverage
        // guarantee (Vovk et al.) is an infinite interval, not silently
        // clamping to the largest observed residual.
        let n = residuals.len();
        let q_rank = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
        let quantile_residual = if q_rank > n {
            f64::INFINITY
        } else {
            residuals[q_rank - 1]
        };

        Ok(Self {
            model,
            quantile_residual,
        })
    }

    /// Predict with conformal intervals.
    pub fn predict(&self, features: &Array2<f64>) -> Result<Vec<ConformalInterval>> {
        let pred = self.model.predict(features)?;
        let predicted = match &pred {
            Prediction::Regression { predicted, .. } => predicted,
            _ => {
                return Err(crate::SmeltError::Other(
                    "Expected regression prediction".into(),
                ));
            }
        };

        Ok(predicted
            .iter()
            .map(|&p| ConformalInterval {
                prediction: p,
                lower: p - self.quantile_residual,
                upper: p + self.quantile_residual,
            })
            .collect())
    }

    /// The calibrated interval width (±).
    pub fn interval_width(&self) -> f64 {
        self.quantile_residual
    }
}

/// Calibrated conformal predictor for classification.
///
/// Produces prediction sets with guaranteed 1-α coverage.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::conformal::ConformalClassifier;
/// use ndarray::array;
///
/// let features = array![[0.0], [0.5], [1.0], [1.5], [2.0], [2.5]];
/// let target = vec![0, 0, 0, 1, 1, 1];
/// let task = ClassificationTask::new("cf", features, target).unwrap();
///
/// let mut dt = DecisionTree::default();
/// let model = dt.train_classif(&task).unwrap();
///
/// let cal_features = array![[1.0], [2.0]];
/// let cal_targets = vec![0, 1];
///
/// let cf = ConformalClassifier::calibrate(&*model, &cal_features, &cal_targets, 0.1).unwrap();
/// let sets = cf.predict(&array![[0.3], [1.8]]).unwrap();
/// ```
pub struct ConformalClassifier<'a> {
    model: &'a dyn TrainedModel,
    quantile_score: f64,
}

impl<'a> ConformalClassifier<'a> {
    /// Calibrate on a held-out set. Uses 1 - P(true class) as nonconformity score.
    pub fn calibrate(
        model: &'a dyn TrainedModel,
        cal_features: &Array2<f64>,
        cal_targets: &[usize],
        alpha: f64,
    ) -> Result<Self> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(crate::SmeltError::InvalidParameter(format!(
                "conformal alpha must be in (0, 1), got {alpha}"
            )));
        }
        if cal_targets.is_empty() {
            return Err(crate::SmeltError::EmptyDataset);
        }

        let pred = model.predict(cal_features)?;
        let probabilities = match &pred {
            Prediction::Classification {
                probabilities: Some(p),
                ..
            } => p,
            _ => {
                return Err(crate::SmeltError::Other(
                    "Conformal classification requires model with probabilities".into(),
                ));
            }
        };

        // Nonconformity score: 1 - P(true class)
        let mut scores: Vec<f64> = Vec::with_capacity(cal_targets.len());
        for (probs, &t) in probabilities.iter().zip(cal_targets) {
            let p_true = probs.get(t).ok_or_else(|| {
                crate::SmeltError::InvalidParameter(format!(
                    "calibration label {t} is out of range for a model with {} classes",
                    probs.len()
                ))
            })?;
            scores.push(1.0 - p_true);
        }
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Quantile rank: ceil((n+1)(1-alpha)). If this exceeds n, the
        // calibration set is too small to guarantee 1-alpha coverage — fall
        // back to the most conservative score (include every class) rather
        // than an out-of-bounds index.
        let n = scores.len();
        let q_rank = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
        let quantile_score = if q_rank > n { 1.0 } else { scores[q_rank - 1] };

        Ok(Self {
            model,
            quantile_score,
        })
    }

    /// Predict with conformal prediction sets.
    pub fn predict(&self, features: &Array2<f64>) -> Result<Vec<ConformalSet>> {
        let pred = self.model.predict(features)?;
        let (predicted, probabilities) = match &pred {
            Prediction::Classification {
                predicted,
                probabilities: Some(p),
                ..
            } => (predicted, p),
            _ => {
                return Err(crate::SmeltError::Other(
                    "Conformal classification requires model with probabilities".into(),
                ));
            }
        };

        Ok(predicted
            .iter()
            .zip(probabilities)
            .map(|(&pred, probs)| {
                // Include class in set if 1 - P(class) <= quantile
                let prediction_set: Vec<usize> = probs
                    .iter()
                    .enumerate()
                    .filter(|&(_, &p)| 1.0 - p <= self.quantile_score)
                    .map(|(c, _)| c)
                    .collect();

                ConformalSet {
                    prediction: pred,
                    prediction_set,
                    probabilities: probs.clone(),
                }
            })
            .collect())
    }
}
