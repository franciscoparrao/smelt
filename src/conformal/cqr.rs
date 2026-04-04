//! Conformalized Quantile Regression (CQR).
//!
//! Combines quantile regression with conformal prediction for
//! adaptive prediction intervals that are wider where uncertainty is high
//! and narrower where the model is confident.
//!
//! Reference: Romano, Y., Patterson, E., & Candès, E. (2019).
//! Conformalized Quantile Regression. NeurIPS.

use crate::learner::TrainedModel;
use crate::prediction::Prediction;
use crate::{Result, SmeltError};
use ndarray::Array2;

/// CQR prediction interval.
#[derive(Debug, Clone)]
pub struct CQRInterval {
    pub prediction: f64,
    pub lower: f64,
    pub upper: f64,
}

/// Conformalized Quantile Regression.
///
/// Uses two quantile models (lower and upper) calibrated with conformal
/// correction. Produces adaptive intervals — wider where uncertainty is high.
///
/// # Examples
///
/// ```no_run
/// use smelt_ml::conformal::cqr::CQR;
/// use ndarray::array;
///
/// // Train two quantile models (e.g., QuantileGB at 0.05 and 0.95)
/// // then calibrate CQR on held-out data
/// // let cqr = CQR::calibrate(&*lower_model, &*upper_model, &cal_features, &cal_targets, 0.1);
/// ```
pub struct CQR<'a> {
    lower_model: &'a dyn TrainedModel,
    upper_model: &'a dyn TrainedModel,
    correction: f64,
}

impl<'a> CQR<'a> {
    /// Calibrate CQR on a held-out calibration set.
    ///
    /// `lower_model`: trained to predict quantile α/2 (e.g., 0.05)
    /// `upper_model`: trained to predict quantile 1-α/2 (e.g., 0.95)
    /// `alpha`: miscoverage level (e.g., 0.1 for 90% coverage)
    pub fn calibrate(
        lower_model: &'a dyn TrainedModel,
        upper_model: &'a dyn TrainedModel,
        cal_features: &Array2<f64>,
        cal_targets: &[f64],
        alpha: f64,
    ) -> Result<Self> {
        let lower_pred = lower_model.predict(cal_features)?;
        let upper_pred = upper_model.predict(cal_features)?;

        let lower_vals = match &lower_pred {
            Prediction::Regression { predicted, .. } => predicted,
            _ => return Err(SmeltError::Other("Expected regression prediction".into())),
        };
        let upper_vals = match &upper_pred {
            Prediction::Regression { predicted, .. } => predicted,
            _ => return Err(SmeltError::Other("Expected regression prediction".into())),
        };

        // Conformity scores: max(lower - y, y - upper)
        // This captures how much the true value exceeds the predicted interval
        let mut scores: Vec<f64> = cal_targets
            .iter()
            .zip(lower_vals.iter().zip(upper_vals))
            .map(|(&y, (&lo, &hi))| (lo - y).max(y - hi))
            .collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Quantile of conformity scores
        let n = scores.len();
        let q_idx = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
        let correction = scores[q_idx.min(n) - 1];

        Ok(Self {
            lower_model,
            upper_model,
            correction,
        })
    }

    /// Predict with conformalized intervals.
    pub fn predict(&self, features: &Array2<f64>) -> Result<Vec<CQRInterval>> {
        let lower_pred = self.lower_model.predict(features)?;
        let upper_pred = self.upper_model.predict(features)?;

        let lower_vals = match &lower_pred {
            Prediction::Regression { predicted, .. } => predicted,
            _ => return Err(SmeltError::Other("Expected regression prediction".into())),
        };
        let upper_vals = match &upper_pred {
            Prediction::Regression { predicted, .. } => predicted,
            _ => return Err(SmeltError::Other("Expected regression prediction".into())),
        };

        Ok(lower_vals
            .iter()
            .zip(upper_vals)
            .map(|(&lo, &hi)| CQRInterval {
                prediction: (lo + hi) / 2.0,
                lower: lo - self.correction,
                upper: hi + self.correction,
            })
            .collect())
    }

    /// The conformal correction applied to the quantile intervals.
    pub fn correction(&self) -> f64 {
        self.correction
    }
}
