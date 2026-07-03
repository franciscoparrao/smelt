//! X-learner: imputed-effect regression combined via a propensity model.

use super::{LearnerFactory, MetaLearnerResult, validate_causal_inputs};
use crate::learner::Learner;
use crate::prediction::Prediction;
use crate::task::{ClassificationTask, RegressionTask};
use crate::{Result, SmeltError};
use ndarray::{Array2, Axis};

/// X-learner (Künzel et al. 2019, §2.3): builds on the T-learner idea but
/// makes better use of the data, especially when the two arms are very
/// unbalanced in size.
///
/// 1. Fit `μ̂0(x)`/`μ̂1(x)` as in [`super::TLearner`].
/// 2. Impute per-unit effects: treated units get `D1_i = Y_i - μ̂0(X_i)`,
///    control units get `D0_i = μ̂1(X_i) - Y_i`.
/// 3. Fit `τ̂1(x)` regressing `D1` on `X` over the treated subset, `τ̂0(x)`
///    regressing `D0` on `X` over the control subset.
/// 4. Fit a propensity model `ĝ(x) = P(T=1|X)` and combine:
///    `τ̂(x) = ĝ(x)·τ̂0(x) + (1-ĝ(x))·τ̂1(x)`.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::causal::meta_learners::XLearner;
///
/// let x_learner = XLearner::new(
///     || Box::new(RandomForest::new()),
///     || Box::new(RandomForest::new()),
///     || Box::new(RandomForest::new()),
///     || Box::new(RandomForest::new()),
///     || Box::new(LogisticRegression::new()),
/// );
/// ```
pub struct XLearner {
    control_factory: LearnerFactory,
    treated_factory: LearnerFactory,
    tau_control_factory: LearnerFactory,
    tau_treated_factory: LearnerFactory,
    propensity_factory: LearnerFactory,
    propensity_clip: f64,
}

impl XLearner {
    /// `control_factory`/`treated_factory` build `μ̂0`/`μ̂1` (the outcome
    /// models per arm). `tau_control_factory`/`tau_treated_factory` build
    /// `τ̂0`/`τ̂1` (regressing the imputed effects). `propensity_factory`
    /// builds `ĝ(x)`, any classifier that produces class probabilities.
    pub fn new(
        control_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        treated_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        tau_control_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        tau_treated_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        propensity_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
    ) -> Self {
        Self {
            control_factory: Box::new(control_factory),
            treated_factory: Box::new(treated_factory),
            tau_control_factory: Box::new(tau_control_factory),
            tau_treated_factory: Box::new(tau_treated_factory),
            propensity_factory: Box::new(propensity_factory),
            propensity_clip: 1e-3,
        }
    }

    /// Clip `ĝ(x)` to `[clip, 1-clip]` before combining `τ̂0`/`τ̂1`, to avoid
    /// a near-zero or near-one propensity collapsing the blend onto a
    /// single arm's estimate. Default `1e-3`.
    pub fn with_propensity_clip(mut self, clip: f64) -> Self {
        self.propensity_clip = clip;
        self
    }

    /// Estimate per-unit CATE. `treatment` must be binary (0/1) with at
    /// least one unit in each arm.
    pub fn estimate(
        &self,
        features: &Array2<f64>,
        treatment: &[usize],
        outcome: &[f64],
    ) -> Result<MetaLearnerResult> {
        validate_causal_inputs(features, treatment, outcome)?;

        let control_idx: Vec<usize> = (0..treatment.len()).filter(|&i| treatment[i] == 0).collect();
        let treated_idx: Vec<usize> = (0..treatment.len()).filter(|&i| treatment[i] == 1).collect();

        // Stage 1: mu0/mu1, same as T-learner.
        let control_features = features.select(Axis(0), &control_idx);
        let control_target: Vec<f64> = control_idx.iter().map(|&i| outcome[i]).collect();
        let control_task = RegressionTask::new("x_learner_control", control_features, control_target)?;
        let mu0 = (self.control_factory)().train_regress(&control_task)?;

        let treated_features = features.select(Axis(0), &treated_idx);
        let treated_target: Vec<f64> = treated_idx.iter().map(|&i| outcome[i]).collect();
        let treated_task = RegressionTask::new("x_learner_treated", treated_features, treated_target)?;
        let mu1 = (self.treated_factory)().train_regress(&treated_task)?;

        // Stage 2: impute per-unit effects.
        let mu0_on_treated = extract_regression(&mu0.predict(&features.select(Axis(0), &treated_idx))?)?;
        let d1: Vec<f64> = treated_idx
            .iter()
            .zip(&mu0_on_treated)
            .map(|(&i, &m0)| outcome[i] - m0)
            .collect();

        let mu1_on_control = extract_regression(&mu1.predict(&features.select(Axis(0), &control_idx))?)?;
        let d0: Vec<f64> = control_idx
            .iter()
            .zip(&mu1_on_control)
            .map(|(&i, &m1)| m1 - outcome[i])
            .collect();

        // Stage 3: tau0/tau1 regress the imputed effects.
        let tau_treated_features = features.select(Axis(0), &treated_idx);
        let tau_treated_task = RegressionTask::new("x_learner_tau1", tau_treated_features, d1)?;
        let tau1 = (self.tau_treated_factory)().train_regress(&tau_treated_task)?;

        let tau_control_features = features.select(Axis(0), &control_idx);
        let tau_control_task = RegressionTask::new("x_learner_tau0", tau_control_features, d0)?;
        let tau0 = (self.tau_control_factory)().train_regress(&tau_control_task)?;

        // Stage 4: propensity g(x).
        let propensity_task = ClassificationTask::new("x_learner_propensity", features.clone(), treatment.to_vec())?;
        let g_model = (self.propensity_factory)().train_classif(&propensity_task)?;
        let g_pred = g_model.predict(features)?;
        let Prediction::Classification {
            probabilities: Some(probs),
            ..
        } = g_pred
        else {
            return Err(SmeltError::InvalidParameter(
                "XLearner's propensity learner must produce class probabilities \
                 (e.g. logistic_regression, gaussian_nb, random_forest)"
                    .into(),
            ));
        };
        let g_hat: Vec<f64> = probs
            .iter()
            .map(|p| {
                p.get(1)
                    .copied()
                    .unwrap_or(0.5)
                    .clamp(self.propensity_clip, 1.0 - self.propensity_clip)
            })
            .collect();

        // Stage 5: combine.
        let tau0_pred = extract_regression(&tau0.predict(features)?)?;
        let tau1_pred = extract_regression(&tau1.predict(features)?)?;
        let cate: Vec<f64> = (0..features.nrows())
            .map(|i| g_hat[i] * tau0_pred[i] + (1.0 - g_hat[i]) * tau1_pred[i])
            .collect();

        Ok(MetaLearnerResult::new(cate))
    }
}

fn extract_regression(pred: &Prediction) -> Result<Vec<f64>> {
    match pred {
        Prediction::Regression { predicted, .. } => Ok(predicted.clone()),
        _ => Err(SmeltError::InvalidParameter(
            "XLearner's outcome/tau learners must produce regression predictions".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{synthetic_confounded_nonlinear_cate, synthetic_linear_cate};
    use super::*;
    use crate::learner::{LogisticRegression, RandomForest};
    use crate::measure::{AteBias, Measure, Pehe};
    use crate::prediction::Prediction;

    fn x_learner_rf() -> XLearner {
        XLearner::new(
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(1)),
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(2)),
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(3)),
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(4)),
            || Box::new(LogisticRegression::new()),
        )
    }

    #[test]
    fn recovers_linear_heterogeneous_effect() {
        let (features, treatment, outcome, true_cate) = synthetic_linear_cate(300, 5, 0.1);
        let result = x_learner_rf().estimate(&features, &treatment, &outcome).unwrap();
        let pred = Prediction::causal_effect_with_truth(result.cate, true_cate);
        assert!(Pehe.score(&pred).unwrap() < 1.5, "PEHE too high: {}", Pehe.score(&pred).unwrap());
        assert!(AteBias.score(&pred).unwrap() < 0.5);
    }

    #[test]
    fn handles_confounded_nonlinear_effect() {
        let (features, treatment, outcome, true_cate) =
            synthetic_confounded_nonlinear_cate(400, 6, 0.1);
        let result = x_learner_rf().estimate(&features, &treatment, &outcome).unwrap();
        let pred = Prediction::causal_effect_with_truth(result.cate, true_cate);
        assert!(Pehe.score(&pred).unwrap().is_finite());
        assert!(Pehe.score(&pred).unwrap() < 4.0, "PEHE too high: {}", Pehe.score(&pred).unwrap());
    }

    #[test]
    fn rejects_non_binary_treatment() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let err = x_learner_rf().estimate(&features, &[0, 1, 2], &[1.0, 2.0, 3.0]);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_single_arm() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let err = x_learner_rf().estimate(&features, &[0, 0, 0], &[1.0, 2.0, 3.0]);
        assert!(err.is_err());
    }
}
