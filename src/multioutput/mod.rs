//! Multi-output regression: predict multiple continuous targets per instance.
//!
//! Implements Regressor Chains — trains a sequence of regressors where
//! each uses previous predictions as additional features.

use ndarray::Array2;
use crate::task::RegressionTask;
use crate::learner::{Learner, TrainedModel};
use crate::prediction::Prediction;
use crate::{SmeltError, Result};

/// Multi-output prediction result.
#[derive(Debug, Clone)]
pub struct MultiOutputPrediction {
    /// Predictions per target: `values[sample][target]`.
    pub values: Vec<Vec<f64>>,
    pub n_samples: usize,
    pub n_targets: usize,
}

/// Regressor Chain for multi-output regression.
///
/// Trains T regressors sequentially. Each regressor j receives the original
/// features augmented with predictions from regressors 1..j-1.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::multioutput::RegressorChain;
/// use ndarray::array;
///
/// let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
/// // 2 targets: y0 = 2*x, y1 = x^2 (approximately)
/// let targets = vec![
///     vec![2.0, 1.0], vec![4.0, 4.0], vec![6.0, 9.0],
///     vec![8.0, 16.0], vec![10.0, 25.0], vec![12.0, 36.0],
/// ];
///
/// let rc = RegressorChain::new(|| Box::new(DecisionTree::default()));
/// let model = rc.fit(&features, &targets).unwrap();
/// let pred = model.predict(&features).unwrap();
/// assert_eq!(pred.n_targets, 2);
/// ```
pub struct RegressorChain {
    factory: Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>,
}

impl RegressorChain {
    pub fn new(factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static) -> Self {
        Self { factory: Box::new(factory) }
    }

    pub fn fit(
        &self,
        features: &Array2<f64>,
        targets: &[Vec<f64>],
    ) -> Result<TrainedRegressorChain> {
        let n_samples = features.nrows();
        let n_features = features.ncols();
        let n_targets = targets[0].len();

        if targets.len() != n_samples {
            return Err(SmeltError::DimensionMismatch {
                expected: n_samples, got: targets.len(),
            });
        }

        let mut models: Vec<Box<dyn TrainedModel>> = Vec::with_capacity(n_targets);

        for j in 0..n_targets {
            let n_aug = n_features + j;
            let mut aug_features = Array2::zeros((n_samples, n_aug));

            for i in 0..n_samples {
                for f in 0..n_features {
                    aug_features[[i, f]] = features[[i, f]];
                }
                for prev in 0..j {
                    aug_features[[i, n_features + prev]] = targets[i][prev];
                }
            }

            let target: Vec<f64> = targets.iter().map(|t| t[j]).collect();
            let task = RegressionTask::new(&format!("target_{j}"), aug_features, target)?;

            let mut learner = (self.factory)();
            let model = learner.train_regress(&task)?;
            models.push(model);
        }

        Ok(TrainedRegressorChain { models, n_features, n_targets })
    }
}

/// Trained regressor chain.
pub struct TrainedRegressorChain {
    models: Vec<Box<dyn TrainedModel>>,
    n_features: usize,
    n_targets: usize,
}

impl TrainedRegressorChain {
    pub fn predict(&self, features: &Array2<f64>) -> Result<MultiOutputPrediction> {
        let n_samples = features.nrows();
        let mut all_preds: Vec<Vec<f64>> = vec![vec![0.0; self.n_targets]; n_samples];

        for j in 0..self.n_targets {
            let n_aug = self.n_features + j;
            let mut aug_features = Array2::zeros((n_samples, n_aug));

            for i in 0..n_samples {
                for f in 0..self.n_features {
                    aug_features[[i, f]] = features[[i, f]];
                }
                for prev in 0..j {
                    aug_features[[i, self.n_features + prev]] = all_preds[i][prev];
                }
            }

            let pred = self.models[j].predict(&aug_features)?;
            if let Prediction::Regression { predicted, .. } = &pred {
                for (i, &p) in predicted.iter().enumerate() {
                    all_preds[i][j] = p;
                }
            }
        }

        Ok(MultiOutputPrediction {
            values: all_preds, n_samples, n_targets: self.n_targets,
        })
    }

    /// Mean RMSE across all targets.
    pub fn mean_rmse(&self, predictions: &MultiOutputPrediction, truth: &[Vec<f64>]) -> f64 {
        let n = predictions.n_samples as f64;
        let mut total_rmse = 0.0;
        for j in 0..predictions.n_targets {
            let mse: f64 = predictions.values.iter().zip(truth)
                .map(|(pred, true_vals)| (pred[j] - true_vals[j]).powi(2))
                .sum::<f64>() / n;
            total_rmse += mse.sqrt();
        }
        total_rmse / predictions.n_targets as f64
    }
}
