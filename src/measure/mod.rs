//! Measures: evaluate prediction quality.

use crate::prediction::Prediction;
use crate::{SmeltError, Result};

/// Trait for evaluation metrics.
pub trait Measure {
    /// Metric identifier (e.g., "classif.accuracy").
    fn id(&self) -> &str;
    /// Compute score. Higher is better for maximize=true, lower for maximize=false.
    fn score(&self, prediction: &Prediction) -> Result<f64>;
    /// Whether higher values are better.
    fn maximize(&self) -> bool;
}

/// Classification accuracy.
pub struct Accuracy;

impl Measure for Accuracy {
    fn id(&self) -> &str { "classif.accuracy" }
    fn maximize(&self) -> bool { true }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Classification { predicted, truth: Some(truth), .. } => {
                let correct = predicted.iter().zip(truth).filter(|(p, t)| p == t).count();
                Ok(correct as f64 / predicted.len() as f64)
            }
            _ => Err(SmeltError::Other("Accuracy requires classification prediction with truth".into())),
        }
    }
}

/// Root Mean Squared Error.
pub struct Rmse;

impl Measure for Rmse {
    fn id(&self) -> &str { "regr.rmse" }
    fn maximize(&self) -> bool { false }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Regression { predicted, truth: Some(truth) } => {
                let mse: f64 = predicted.iter().zip(truth)
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>() / predicted.len() as f64;
                Ok(mse.sqrt())
            }
            _ => Err(SmeltError::Other("RMSE requires regression prediction with truth".into())),
        }
    }
}

/// Mean Absolute Error.
pub struct Mae;

impl Measure for Mae {
    fn id(&self) -> &str { "regr.mae" }
    fn maximize(&self) -> bool { false }

    fn score(&self, prediction: &Prediction) -> Result<f64> {
        match prediction {
            Prediction::Regression { predicted, truth: Some(truth) } => {
                let mae: f64 = predicted.iter().zip(truth)
                    .map(|(p, t)| (p - t).abs())
                    .sum::<f64>() / predicted.len() as f64;
                Ok(mae)
            }
            _ => Err(SmeltError::Other("MAE requires regression prediction with truth".into())),
        }
    }
}
