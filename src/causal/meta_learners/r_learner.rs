//! R-learner: residual-on-residual regression via cross-fitting.

use super::{LearnerFactory, MetaLearnerResult, cross_fit, validate_causal_inputs};
use crate::learner::Learner;
use crate::prediction::Prediction;
use crate::task::RegressionTask;
use crate::{Result, SmeltError};
use ndarray::{Array2, Axis};

/// R-learner (Nie & Wager 2021, "Quasi-oracle estimation of heterogeneous
/// treatment effects"):
///
/// 1. Cross-fit `m̂(x) = E[Y|X]` and `ê(x) = E[T|X]` via K-fold
///    out-of-fold prediction (never predicting a unit with a model that
///    saw it during training -- see [`cross_fit`]).
/// 2. Residuals: `Ỹ_i = Y_i - m̂(X_i)`, `T̃_i = T_i - ê(X_i)`.
/// 3. Fit `τ̂(x)` on the pseudo-target `Ỹ_i / T̃_i`.
///
/// # Known simplification vs. the paper
///
/// Step 3 in Nie & Wager's paper is a **weighted** least-squares fit with
/// weight `T̃_i²` (the "R-loss") -- this is what makes the estimator
/// efficient, not just unbiased. This crate has no generic per-sample-weight
/// support outside `XGBoost`/`GeoXGBoost` (both bespoke builder methods
/// outside the `Learner` trait), so restricting the effect-stage learner to
/// one specific type would break the uniform `LearnerFactory` abstraction
/// this whole module (and `Bagging`/`Stacking`) is built on. Instead, this
/// implementation drops rows where `|T̃_i|` is small (division blow-up
/// guard, `residual_clip`) and fits an **unweighted** regression on the
/// clipped pseudo-target. Cross-fitting still gives this its core
/// "quasi-oracle" bias-removal property; what's lost is the efficiency
/// bound from weighting by how informative each unit's treatment residual
/// is. Adding generic sample-weight support to `Learner`/`RegressionTask`
/// (a cross-cutting change touching every regression learner) would let
/// this match the paper exactly, but is out of scope here.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use smelt_ml::causal::meta_learners::RLearner;
///
/// let r_learner = RLearner::new(
///     || Box::new(RandomForest::new()),
///     || Box::new(LogisticRegression::new()),
///     || Box::new(RandomForest::new()),
/// );
/// ```
pub struct RLearner {
    outcome_factory: LearnerFactory,
    propensity_factory: LearnerFactory,
    effect_factory: LearnerFactory,
    cv_folds: usize,
    cv_seed: u64,
    residual_clip: f64,
}

impl RLearner {
    /// `outcome_factory` builds `m̂(x) = E[Y|X]`, `propensity_factory`
    /// builds `ê(x) = E[T|X]` (any classifier producing probabilities),
    /// `effect_factory` builds the final `τ̂(x)` model.
    pub fn new(
        outcome_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        propensity_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        effect_factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
    ) -> Self {
        Self {
            outcome_factory: Box::new(outcome_factory),
            propensity_factory: Box::new(propensity_factory),
            effect_factory: Box::new(effect_factory),
            cv_folds: 5,
            cv_seed: 42,
            residual_clip: 1e-3,
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
    /// Rows with `|T̃_i| < clip` are dropped from the final regression (the
    /// pseudo-target `Ỹ_i/T̃_i` blows up as `T̃_i -> 0`). Default `1e-3`.
    pub fn with_residual_clip(mut self, clip: f64) -> Self {
        self.residual_clip = clip;
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

        let m_hat = cross_fit::oof_regression(
            features,
            outcome,
            &self.outcome_factory,
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
        let mut kept_idx = Vec::with_capacity(n);
        let mut pseudo_target = Vec::with_capacity(n);
        for i in 0..n {
            let y_resid = outcome[i] - m_hat[i];
            let t_resid = treatment[i] as f64 - e_hat[i];
            if t_resid.abs() >= self.residual_clip {
                kept_idx.push(i);
                pseudo_target.push(y_resid / t_resid);
            }
        }
        if kept_idx.is_empty() {
            return Err(SmeltError::InvalidParameter(
                "RLearner: every unit's treatment residual was below residual_clip -- the \
                 propensity model may be too close to deterministic; try a lower clip or a \
                 different propensity learner"
                    .into(),
            ));
        }

        let final_features = features.select(Axis(0), &kept_idx);
        let final_task = RegressionTask::new("r_learner_effect", final_features, pseudo_target)?;
        let tau_model = (self.effect_factory)().train_regress(&final_task)?;

        let pred = tau_model.predict(features)?;
        let Prediction::Regression { predicted: cate, .. } = pred else {
            return Err(SmeltError::InvalidParameter(
                "RLearner's effect learner must produce regression predictions".into(),
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

    fn r_learner_rf() -> RLearner {
        RLearner::new(
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(1)),
            || Box::new(LogisticRegression::new()),
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(2)),
        )
    }

    #[test]
    fn recovers_linear_heterogeneous_effect() {
        let (features, treatment, outcome, true_cate) = synthetic_linear_cate(400, 7, 0.1);
        let result = r_learner_rf().estimate(&features, &treatment, &outcome).unwrap();
        let pred = Prediction::causal_effect_with_truth(result.cate, true_cate);
        assert!(Pehe.score(&pred).unwrap() < 2.0, "PEHE too high: {}", Pehe.score(&pred).unwrap());
        assert!(AteBias.score(&pred).unwrap() < 0.7);
    }

    #[test]
    fn handles_confounded_nonlinear_effect() {
        let (features, treatment, outcome, true_cate) =
            synthetic_confounded_nonlinear_cate(400, 8, 0.1);
        let result = r_learner_rf().estimate(&features, &treatment, &outcome).unwrap();
        let pred = Prediction::causal_effect_with_truth(result.cate, true_cate);
        assert!(Pehe.score(&pred).unwrap().is_finite());
    }

    #[test]
    fn rejects_non_binary_treatment() {
        let features = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let err = r_learner_rf().estimate(&features, &[0, 1, 2, 0, 1, 0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(err.is_err());
    }

    #[test]
    fn rejects_single_arm() {
        let features = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let err = r_learner_rf().estimate(&features, &[0, 0, 0, 0, 0, 0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(err.is_err());
    }
}
