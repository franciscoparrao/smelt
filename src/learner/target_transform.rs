//! Target transformation for regression with automatic inverse at predict
//! time — the equivalent of mlr3's `po("targettrafo")` / sklearn's
//! `TransformedTargetRegressor`.
//!
//! Log-transforming a skewed target (concentrations, grades, precipitation —
//! routine in geochemistry) and remembering to invert the predictions by hand
//! is a classic source of bugs: forget the `exp`, or apply it to the truth
//! instead of the prediction, and every downstream metric is silently wrong.
//! [`TargetTransformRegressor`] wraps any regression [`Learner`] (same
//! `factory: Fn() -> Box<dyn Learner>` pattern as
//! [`crate::learner::Bagging`]/[`crate::learner::CostSensitiveClassifier`]),
//! trains it on the transformed target, and applies the inverse
//! transformation inside `predict`, so predictions always come back in the
//! original scale.
//!
//! # Retransformation bias (Log/Log1p)
//!
//! The naive inverse `exp(E[log y])` does **not** estimate `E[y]`: if the
//! base model's errors are symmetric in log-scale, it estimates the *median*
//! of `y` in the original scale, which under-estimates the mean of a
//! right-skewed distribution (Jensen's inequality). This wrapper applies
//! exactly that naive inverse — a median-type prediction is often what you
//! want for skewed targets, and it is what sklearn's
//! `TransformedTargetRegressor` does too. A bias correction such as Duan's
//! smearing estimator (Duan, 1983) is a possible future opt-in, deliberately
//! not implemented here.
//!
//! # Composability
//!
//! Because `predict` returns predictions already in the original scale, this
//! wrapper composes transparently with everything that consumes a
//! [`TrainedModel`]: [`crate::conformal::SplitConformal`] calibrates
//! original-scale residuals, measures score original-scale errors, and
//! [`crate::benchmark::resample_regress`] needs no special handling.
//!
//! # Registry
//!
//! Like `Bagging`/`Stacking`/`CostSensitiveClassifier`, this wrapper needs a
//! base-learner factory with no sensible default, so it is **not**
//! constructible via [`crate::learner::registry::learner_from_id`].

use crate::Result;
use crate::SmeltError;
use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{RegressionTask, Task};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Target transformation applied before training the base learner; the
/// inverse is applied automatically to the base model's predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetTransform {
    /// `ln(y)` / inverse `exp(x)`. Requires every target value `y > 0`.
    Log,
    /// `ln(1 + y)` / inverse `exp(x) - 1`. Requires every target value
    /// `y > -1`; keeps exact precision near zero via `ln_1p`/`exp_m1`.
    Log1p,
    /// `sqrt(y)` / inverse `x²`. Requires every target value `y >= 0`.
    Sqrt,
    /// `(y - mean) / std` with `mean`/`std` fitted on the *training* target
    /// / inverse `x * std + mean`. Accepts any finite target. When the
    /// training target is constant (zero variance), `std` falls back to
    /// `1.0` — the same convention as
    /// [`crate::preprocess::StandardScaler`], so the transform degenerates
    /// to a pure mean-shift instead of dividing by zero.
    Standardize,
}

impl TargetTransform {
    fn name(&self) -> &'static str {
        match self {
            TargetTransform::Log => "log",
            TargetTransform::Log1p => "log1p",
            TargetTransform::Sqrt => "sqrt",
            TargetTransform::Standardize => "standardize",
        }
    }

    /// Validate the training target's domain BEFORE transforming, naming the
    /// index of the first invalid value. Every transform requires finite
    /// values (a NaN/±inf target poisons the fit silently otherwise); Log/
    /// Log1p/Sqrt additionally restrict the domain so the transform can't
    /// produce NaN/-inf.
    fn validate_domain(&self, target: &[f64]) -> Result<()> {
        for (i, &y) in target.iter().enumerate() {
            if !y.is_finite() {
                return Err(SmeltError::InvalidParameter(format!(
                    "target_transform({}): target contains non-finite value {y} at index {i}; \
                     targets must be finite",
                    self.name()
                )));
            }
            let ok = match self {
                TargetTransform::Log => y > 0.0,
                TargetTransform::Log1p => y > -1.0,
                TargetTransform::Sqrt => y >= 0.0,
                TargetTransform::Standardize => true,
            };
            if !ok {
                let requirement = match self {
                    TargetTransform::Log => "y > 0",
                    TargetTransform::Log1p => "y > -1",
                    TargetTransform::Sqrt => "y >= 0",
                    TargetTransform::Standardize => unreachable!(),
                };
                return Err(SmeltError::InvalidParameter(format!(
                    "target_transform({}): target value {y} at index {i} is outside the \
                     transform's domain (requires {requirement})",
                    self.name()
                )));
            }
        }
        Ok(())
    }
}

/// The state fitted at train time and needed to invert predictions:
/// Log/Log1p/Sqrt are stateless; Standardize carries the training target's
/// mean/std.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
enum FittedTransform {
    Log,
    Log1p,
    Sqrt,
    Standardize {
        /// Training-target mean.
        mean: f64,
        /// Training-target (population) std; `1.0` if the target was
        /// constant, matching `StandardScaler`'s convention.
        std: f64,
    },
}

impl FittedTransform {
    fn inverse(&self, x: f64) -> f64 {
        match self {
            FittedTransform::Log => x.exp(),
            FittedTransform::Log1p => x.exp_m1(),
            FittedTransform::Sqrt => x * x,
            FittedTransform::Standardize { mean, std } => x * std + mean,
        }
    }
}

/// Regression wrapper that trains its base learner on a transformed target
/// and automatically applies the inverse transformation at predict time.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::learner::{TargetTransform, TargetTransformRegressor};
/// use ndarray::array;
///
/// // A right-skewed (log-normal-ish) target.
/// let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
/// let target = vec![1.1, 2.9, 8.2, 22.0, 60.5, 163.0];
/// let task = RegressionTask::new("skewed", features.clone(), target).unwrap();
///
/// let mut ttr = TargetTransformRegressor::new(
///     || Box::new(LinearRegression),
///     TargetTransform::Log,
/// );
/// let model = ttr.train_regress(&task).unwrap();
/// // Predictions come back in the ORIGINAL scale — no manual exp() needed.
/// let pred = model.predict(&features).unwrap();
/// ```
pub struct TargetTransformRegressor {
    factory: Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>,
    transform: TargetTransform,
}

impl TargetTransformRegressor {
    /// Creates a target-transforming wrapper from a base-learner factory and
    /// the transform to apply. The domain of the training target is
    /// validated against the transform at `train_regress` time.
    pub fn new(
        factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        transform: TargetTransform,
    ) -> Self {
        Self {
            factory: Box::new(factory),
            transform,
        }
    }
}

impl Learner for TargetTransformRegressor {
    fn id(&self) -> &str {
        "target_transform"
    }

    fn properties(&self) -> LearnerProperties {
        LearnerProperties::regressor().with_feature_importance()
    }

    fn train_classif(
        &mut self,
        _task: &crate::task::ClassificationTask,
    ) -> Result<Box<dyn TrainedModel>> {
        Err(SmeltError::InvalidParameter(
            "TargetTransformRegressor is a regression-only wrapper (target transforms like \
             log/sqrt have no meaning for class labels); use train_regress"
                .into(),
        ))
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        crate::validate::check_no_weights(task.weights(), "TargetTransformRegressor")?;
        let target = task.target();
        // (1) Validate the domain BEFORE transforming, naming the first
        // offending index.
        self.transform.validate_domain(target)?;

        // (2) Transform the target; Standardize fits mean/std on the
        // TRAINING target only (population variance, std -> 1.0 when the
        // target is constant — StandardScaler's convention).
        let (transformed, fitted): (Vec<f64>, FittedTransform) = match self.transform {
            TargetTransform::Log => (target.iter().map(|&y| y.ln()).collect(), FittedTransform::Log),
            TargetTransform::Log1p => (
                target.iter().map(|&y| y.ln_1p()).collect(),
                FittedTransform::Log1p,
            ),
            TargetTransform::Sqrt => (
                target.iter().map(|&y| y.sqrt()).collect(),
                FittedTransform::Sqrt,
            ),
            TargetTransform::Standardize => {
                let n = target.len() as f64;
                let mean = target.iter().sum::<f64>() / n;
                let variance = target.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / n;
                let std = if variance > 0.0 { variance.sqrt() } else { 1.0 };
                (
                    target.iter().map(|&y| (y - mean) / std).collect(),
                    FittedTransform::Standardize { mean, std },
                )
            }
        };

        // (3) Rebuild the task with the transformed target, PROPAGATING
        // feature_names and feature_types (dropping them silently disables
        // native categorical splits in the base learner — 5th audit M-3).
        let mut transformed_task =
            RegressionTask::new(task.id(), task.features().clone(), transformed)?
                .with_feature_names(task.feature_names().to_vec())?
                .with_feature_types(task.feature_types().to_vec())?;
        // Weights would propagate unchanged too (the transform is
        // row-preserving). Unreachable today — check_no_weights above
        // rejects weighted tasks — but kept in the rebuild so that removing
        // the guard (when this wrapper becomes weight-aware) cannot silently
        // drop them, the exact M-3 metadata-loss bug class.
        if let Some(w) = task.weights() {
            transformed_task = transformed_task.with_weights(w.to_vec());
        }

        // (4) Train the base learner from the factory.
        let mut base = (self.factory)();
        let model = base.train_regress(&transformed_task)?;

        Ok(Box::new(TrainedTargetTransformRegressor {
            base: model,
            fitted,
        }))
    }
}

/// Trained [`TargetTransformRegressor`]: the base model plus the fitted
/// inverse-transform state. `predict` returns original-scale predictions.
pub struct TrainedTargetTransformRegressor {
    base: Box<dyn TrainedModel>,
    fitted: FittedTransform,
}

impl TrainedModel for TrainedTargetTransformRegressor {
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        let pred = self.base.predict(features)?;
        match pred {
            Prediction::Regression { predicted, truth } => Ok(Prediction::Regression {
                predicted: predicted.iter().map(|&x| self.fitted.inverse(x)).collect(),
                truth,
            }),
            _ => Err(SmeltError::IncompatiblePrediction(
                "TargetTransformRegressor requires regression predictions from its base model"
                    .into(),
            )),
        }
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        self.base.feature_importance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::tree::decision_tree::DecisionTree;
    use crate::learner::{KNearestNeighbors, LinearRegression};
    use ndarray::array;

    fn simple_task() -> RegressionTask {
        let features = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let target = vec![1.5, 2.0, 4.5, 8.0, 20.0, 55.0];
        RegressionTask::new("t", features, target).unwrap()
    }

    fn predicted(pred: Prediction) -> Vec<f64> {
        match pred {
            Prediction::Regression { predicted, .. } => predicted,
            _ => panic!("expected regression prediction"),
        }
    }

    /// `unwrap_err` needs `Debug` on the Ok type, which `Box<dyn
    /// TrainedModel>` doesn't have — unwrap the error side by match instead.
    fn expect_err(res: Result<Box<dyn TrainedModel>>) -> SmeltError {
        match res {
            Err(e) => e,
            Ok(_) => panic!("expected an error, got a trained model"),
        }
    }

    /// For each transform: training the base learner BY HAND on the
    /// transformed target and inverting its predictions manually must be
    /// bit-identical to what the wrapper produces.
    #[test]
    fn wrapper_matches_manual_transform_bit_identically() {
        let task = simple_task();
        let query = array![[1.5], [3.5], [5.5]];

        let cases: Vec<(TargetTransform, Box<dyn Fn(f64) -> f64>, Box<dyn Fn(f64) -> f64>)> = {
            // Standardize's manual forward/inverse need the fitted mean/std.
            let target = task.target();
            let n = target.len() as f64;
            let mean = target.iter().sum::<f64>() / n;
            let variance = target.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt();
            vec![
                (
                    TargetTransform::Log,
                    Box::new(|y: f64| y.ln()),
                    Box::new(|x: f64| x.exp()),
                ),
                (
                    TargetTransform::Log1p,
                    Box::new(|y: f64| y.ln_1p()),
                    Box::new(|x: f64| x.exp_m1()),
                ),
                (
                    TargetTransform::Sqrt,
                    Box::new(|y: f64| y.sqrt()),
                    Box::new(|x: f64| x * x),
                ),
                (
                    TargetTransform::Standardize,
                    Box::new(move |y: f64| (y - mean) / std),
                    Box::new(move |x: f64| x * std + mean),
                ),
            ]
        };

        for (transform, forward, inverse) in cases {
            // Manual: transform target by hand, train the same deterministic
            // base learner, invert its predictions by hand.
            let manual_target: Vec<f64> = task.target().iter().map(|&y| forward(y)).collect();
            let manual_task =
                RegressionTask::new("manual", task.features().clone(), manual_target).unwrap();
            let manual_model = KNearestNeighbors::new(2).train_regress(&manual_task).unwrap();
            let manual: Vec<f64> = predicted(manual_model.predict(&query).unwrap())
                .into_iter()
                .map(&*inverse)
                .collect();

            // Wrapper.
            let mut ttr =
                TargetTransformRegressor::new(|| Box::new(KNearestNeighbors::new(2)), transform);
            let model = ttr.train_regress(&task).unwrap();
            let wrapped = predicted(model.predict(&query).unwrap());

            assert_eq!(
                manual, wrapped,
                "{transform:?}: wrapper predictions must be bit-identical to the manual \
                 transform-train-invert pipeline"
            );
        }
    }

    #[test]
    fn log_rejects_nonpositive_target_naming_index() {
        let features = array![[1.0], [2.0], [3.0]];
        let target = vec![1.0, 0.0, 2.0];
        let task = RegressionTask::new("t", features, target).unwrap();
        let mut ttr =
            TargetTransformRegressor::new(|| Box::new(LinearRegression), TargetTransform::Log);
        let err = expect_err(ttr.train_regress(&task));
        let msg = format!("{err}");
        assert!(
            msg.contains("index 1") && msg.contains("y > 0"),
            "error must name the offending index and the domain, got: {msg}"
        );
    }

    #[test]
    fn log1p_rejects_target_at_or_below_minus_one_naming_index() {
        let features = array![[1.0], [2.0], [3.0]];
        let target = vec![0.5, -0.5, -1.0];
        let task = RegressionTask::new("t", features, target).unwrap();
        let mut ttr =
            TargetTransformRegressor::new(|| Box::new(LinearRegression), TargetTransform::Log1p);
        let err = expect_err(ttr.train_regress(&task));
        let msg = format!("{err}");
        assert!(
            msg.contains("index 2") && msg.contains("y > -1"),
            "error must name the offending index and the domain, got: {msg}"
        );
    }

    #[test]
    fn sqrt_rejects_negative_target_naming_index() {
        let features = array![[1.0], [2.0], [3.0]];
        let target = vec![4.0, 9.0, -0.1];
        let task = RegressionTask::new("t", features, target).unwrap();
        let mut ttr =
            TargetTransformRegressor::new(|| Box::new(LinearRegression), TargetTransform::Sqrt);
        let err = expect_err(ttr.train_regress(&task));
        let msg = format!("{err}");
        assert!(
            msg.contains("index 2") && msg.contains("y >= 0"),
            "error must name the offending index and the domain, got: {msg}"
        );
    }

    #[test]
    fn standardize_rejects_non_finite_target_naming_index() {
        let features = array![[1.0], [2.0], [3.0]];
        let target = vec![1.0, f64::NAN, 2.0];
        let task = RegressionTask::new("t", features, target).unwrap();
        let mut ttr = TargetTransformRegressor::new(
            || Box::new(LinearRegression),
            TargetTransform::Standardize,
        );
        let err = expect_err(ttr.train_regress(&task));
        assert!(
            format!("{err}").contains("index 1"),
            "error must name the offending index, got: {err}"
        );
    }

    #[test]
    fn train_classif_rejected_with_clear_error() {
        let features = array![[0.0], [1.0], [2.0], [3.0]];
        let target = vec![0usize, 0, 1, 1];
        let task = crate::task::ClassificationTask::new("c", features, target).unwrap();
        let mut ttr =
            TargetTransformRegressor::new(|| Box::new(LinearRegression), TargetTransform::Log);
        let err = expect_err(ttr.train_classif(&task));
        assert!(
            format!("{err}").contains("regression-only"),
            "expected a clear regression-only error, got: {err}"
        );
    }

    /// Standardize must fit the training target's mean and (population) std:
    /// verified through a base learner that predicts the transformed-target
    /// mean (KNN over all samples), whose inverted prediction must therefore
    /// be exactly the original-scale mean.
    #[test]
    fn standardize_fits_training_mean_and_std() {
        let features = array![[1.0], [2.0], [3.0], [4.0]];
        let target = vec![10.0, 20.0, 30.0, 40.0];
        let task = RegressionTask::new("t", features, target.clone()).unwrap();

        // k = n: the KNN regressor predicts the mean of the (standardized)
        // training target = 0.0 for every query; inverting must give back
        // exactly mean(y) = 25.0.
        let mut ttr = TargetTransformRegressor::new(
            || Box::new(KNearestNeighbors::new(4)),
            TargetTransform::Standardize,
        );
        let model = ttr.train_regress(&task).unwrap();
        let preds = predicted(model.predict(&array![[2.5]]).unwrap());
        assert!(
            (preds[0] - 25.0).abs() < 1e-9,
            "inverse of standardized-mean prediction must be the original mean, got {}",
            preds[0]
        );
    }

    /// A constant target has zero variance; per StandardScaler's convention
    /// std falls back to 1.0, so the transform is a pure mean-shift and
    /// training must not fail or produce NaN.
    #[test]
    fn standardize_constant_target_follows_scaler_convention() {
        let features = array![[1.0], [2.0], [3.0]];
        let target = vec![7.0, 7.0, 7.0];
        let task = RegressionTask::new("t", features, target).unwrap();
        let mut ttr = TargetTransformRegressor::new(
            || Box::new(KNearestNeighbors::new(3)),
            TargetTransform::Standardize,
        );
        let model = ttr.train_regress(&task).unwrap();
        let preds = predicted(model.predict(&array![[2.0]]).unwrap());
        assert!(
            (preds[0] - 7.0).abs() < 1e-12,
            "constant target must round-trip exactly through the mean-shift, got {}",
            preds[0]
        );
    }

    #[test]
    fn feature_importance_delegates_to_base_model() {
        let features = array![[0.0, 5.0], [1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]];
        let target = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        let task = RegressionTask::new("imp", features, target).unwrap();
        let mut ttr =
            TargetTransformRegressor::new(|| Box::new(DecisionTree::default()), TargetTransform::Log);
        let model = ttr.train_regress(&task).unwrap();
        assert!(model.feature_importance().is_some());
    }

    /// The actual point of the wrapper: on a target generated as
    /// `y = exp(linear signal + gaussian noise)` (log-normal), the same base
    /// learner should predict markedly better with the Log transform than
    /// without it. Deterministic pseudo-noise (fixed "seed" via a hash-like
    /// recurrence) and a generous margin keep this non-flaky.
    #[test]
    fn log_transform_improves_rmse_on_lognormal_target() {
        let n = 200;
        let mut feats = Vec::with_capacity(n);
        let mut target = Vec::with_capacity(n);
        let mut state: u64 = 42;
        for i in 0..n {
            // xorshift for deterministic pseudo-uniform noise in [-0.5, 0.5).
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let noise = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            let x = i as f64 / n as f64 * 4.0; // x in [0, 4)
            feats.push(x);
            target.push((1.5 * x + 0.5 * noise).exp());
        }
        let features = Array2::from_shape_vec((n, 1), feats).unwrap();
        let task = RegressionTask::new("lognormal", features.clone(), target.clone()).unwrap();

        let rmse = |preds: &[f64]| -> f64 {
            (preds
                .iter()
                .zip(&target)
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / n as f64)
                .sqrt()
        };

        let plain_model = LinearRegression.train_regress(&task).unwrap();
        let plain_rmse = rmse(&predicted(plain_model.predict(&features).unwrap()));

        let mut ttr =
            TargetTransformRegressor::new(|| Box::new(LinearRegression), TargetTransform::Log);
        let log_model = ttr.train_regress(&task).unwrap();
        let log_rmse = rmse(&predicted(log_model.predict(&features).unwrap()));

        assert!(
            log_rmse < plain_rmse * 0.8,
            "log-transformed fit should beat the plain fit by a wide margin on a log-normal \
             target: log_rmse={log_rmse:.3} vs plain_rmse={plain_rmse:.3}"
        );
    }
}
