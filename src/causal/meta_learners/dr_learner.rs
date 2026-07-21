//! DR-learner: doubly-robust pseudo-outcome regression via cross-fitting.

use super::{LearnerFactory, MetaLearnerResult, cross_fit, validate_causal_inputs};
use crate::learner::Learner;
use crate::prediction::Prediction;
use crate::task::RegressionTask;
use crate::{Result, SmeltError};
use ndarray::Array2;

/// DR-learner (doubly-robust pseudo-outcome, Kennedy 2020): cross-fits
/// per-arm outcome models and a propensity model (reusing the same
/// [`cross_fit`] machinery as [`super::RLearner`]), then regresses the
/// doubly-robust pseudo-outcome
///
/// ```text
/// φ_i = μ̂1(X_i) - μ̂0(X_i)
///     + T_i·(Y_i - μ̂1(X_i)) / ê(X_i)
///     - (1-T_i)·(Y_i - μ̂0(X_i)) / (1-ê(X_i))
/// ```
///
/// on `X`. Unlike [`super::RLearner`], this final regression is
/// **ordinary and unweighted** -- the doubly-robust construction folds the
/// propensity/outcome-model correction directly into the pseudo-outcome
/// itself, so there's no analog of R-learner's weighted-least-squares
/// requirement to work around. That makes DR-learner arguably *more*
/// faithful to its own literature in this crate than R-learner is forced
/// to be, at the cost of needing per-arm cross-fitting (`oof_regression_by_arm`)
/// rather than a single pooled outcome model.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::causal::meta_learners::DrLearner;
///
/// let dr_learner = DrLearner::new(
///     || Box::new(RandomForest::new()),
///     || Box::new(RandomForest::new()),
///     || Box::new(LogisticRegression::new()),
///     || Box::new(RandomForest::new()),
/// );
/// ```
pub struct DrLearner {
    control_factory: LearnerFactory,
    treated_factory: LearnerFactory,
    propensity_factory: LearnerFactory,
    effect_factory: LearnerFactory,
    cv_folds: usize,
    cv_seed: u64,
    propensity_clip: f64,
}

impl DrLearner {
    /// `control_factory`/`treated_factory` build the cross-fitted per-arm
    /// outcome models `μ̂0`/`μ̂1`, `propensity_factory` builds `ê(x)`,
    /// `effect_factory` builds the final regression of the pseudo-outcome
    /// on `X`.
    pub fn new(
        control_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        treated_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        propensity_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        effect_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
    ) -> Self {
        Self {
            control_factory: Box::new(control_factory),
            treated_factory: Box::new(treated_factory),
            propensity_factory: Box::new(propensity_factory),
            effect_factory: Box::new(effect_factory),
            cv_folds: 5,
            cv_seed: 42,
            propensity_clip: 1e-3,
        }
    }

    /// Sets the number of folds used for cross-fitting the outcome and
    /// propensity models. Default `5`.
    pub fn with_cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        self
    }
    /// Sets the RNG seed for the cross-fitting fold assignment. Default `42`.
    pub fn with_cv_seed(mut self, seed: u64) -> Self {
        self.cv_seed = seed;
        self
    }
    /// Clip `ê(x)` to `[clip, 1-clip]` before dividing by it/its complement
    /// in the pseudo-outcome formula. Default `1e-3`.
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

        let (mu0_hat, mu1_hat) = cross_fit::oof_regression_by_arm(
            features,
            treatment,
            outcome,
            &self.control_factory,
            &self.treated_factory,
            self.cv_folds,
            self.cv_seed,
        )?;
        let e_hat = cross_fit::oof_propensity(
            features,
            treatment,
            &self.propensity_factory,
            self.cv_folds,
            self.cv_seed,
        )?;

        let n = features.nrows();
        let mut phi = Vec::with_capacity(n);
        for i in 0..n {
            let e = e_hat[i].clamp(self.propensity_clip, 1.0 - self.propensity_clip);
            let t = treatment[i] as f64;
            let value = mu1_hat[i] - mu0_hat[i] + t * (outcome[i] - mu1_hat[i]) / e
                - (1.0 - t) * (outcome[i] - mu0_hat[i]) / (1.0 - e);
            phi.push(value);
        }

        let final_task = RegressionTask::new("dr_learner_effect", features.clone(), phi)?;
        let tau_model = (self.effect_factory)().train_regress(&final_task)?;

        let pred = tau_model.predict(features)?;
        let Prediction::Regression {
            predicted: cate, ..
        } = pred
        else {
            return Err(SmeltError::InvalidParameter(
                "DrLearner's effect learner must produce regression predictions".into(),
            ));
        };

        Ok(MetaLearnerResult::new(cate))
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{synthetic_confounded_nonlinear_cate, synthetic_linear_cate};
    use super::*;
    use crate::learner::{LogisticRegression, RandomForest};
    use crate::measure::{AteBias, Measure, Pehe};
    use crate::prediction::Prediction;

    fn dr_learner_rf() -> DrLearner {
        DrLearner::new(
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(1)),
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(2)),
            || Box::new(LogisticRegression::new()),
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(3)),
        )
    }

    #[test]
    fn recovers_linear_heterogeneous_effect() {
        let (features, treatment, outcome, true_cate) = synthetic_linear_cate(400, 9, 0.1);
        let result = dr_learner_rf()
            .estimate(&features, &treatment, &outcome)
            .unwrap();
        let pred = Prediction::causal_effect_with_truth(result.cate, true_cate);
        assert!(
            Pehe.score(&pred).unwrap() < 2.0,
            "PEHE too high: {}",
            Pehe.score(&pred).unwrap()
        );
        assert!(AteBias.score(&pred).unwrap() < 0.7);
    }

    #[test]
    fn handles_confounded_nonlinear_effect() {
        let (features, treatment, outcome, true_cate) =
            synthetic_confounded_nonlinear_cate(400, 10, 0.1);
        let result = dr_learner_rf()
            .estimate(&features, &treatment, &outcome)
            .unwrap();
        let pred = Prediction::causal_effect_with_truth(result.cate, true_cate);
        assert!(Pehe.score(&pred).unwrap().is_finite());
    }

    #[test]
    fn rejects_non_binary_treatment() {
        let features = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let err = dr_learner_rf().estimate(
            &features,
            &[0, 1, 2, 0, 1, 0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        assert!(err.is_err());
    }

    #[test]
    fn rejects_single_arm() {
        let features = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let err = dr_learner_rf().estimate(
            &features,
            &[1, 1, 1, 1, 1, 1],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        assert!(err.is_err());
    }
}
