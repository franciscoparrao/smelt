//! R-learner: residual-on-residual regression via cross-fitting.

use super::{LearnerFactory, MetaLearnerResult, cross_fit, validate_causal_inputs};
use crate::learner::Learner;
use crate::prediction::Prediction;
use crate::task::RegressionTask;
use crate::{Result, SmeltError};
use ndarray::{Array2, Axis};

/// Target mean replica count for the R-loss weighting approximation (see
/// [`RLearner`]'s struct docs): on average, a kept row appears this many
/// times in the replicated effect-stage training set.
const TARGET_MEAN_REPLICAS: f64 = 3.0;
/// Upper bound on any single row's replica count, so one extreme weight
/// can't blow up the effect-stage dataset size.
const MAX_REPLICAS_PER_ROW: usize = 20;

/// Replica counts approximating a `weights`-weighted regression via row
/// duplication: index `k`'s count is proportional to its mean-normalized
/// weight (rounded, clamped to `[1, max_replicas]`). Returns a flat list of
/// indices into `weights` with each `k` repeated its replica count times —
/// fitting an unweighted model on the rows selected by this list
/// approximates fitting a `weights`-weighted model on the original rows.
fn replicate_by_weight(weights: &[f64], target_mean_replicas: f64, max_replicas: usize) -> Vec<usize> {
    let mean_weight = weights.iter().sum::<f64>() / weights.len() as f64;
    let mut out = Vec::new();
    for (k, &w) in weights.iter().enumerate() {
        let reps = ((w / mean_weight) * target_mean_replicas)
            .round()
            .clamp(1.0, max_replicas as f64) as usize;
        out.extend(std::iter::repeat_n(k, reps));
    }
    out
}

/// R-learner (Nie & Wager 2021, "Quasi-oracle estimation of heterogeneous
/// treatment effects"):
///
/// 1. Cross-fit `m̂(x) = E[Y|X]` and `ê(x) = E[T|X]` via K-fold
///    out-of-fold prediction (never predicting a unit with a model that
///    saw it during training -- see [`cross_fit`]).
/// 2. Residuals: `Ỹ_i = Y_i - m̂(X_i)`, `T̃_i = T_i - ê(X_i)`.
/// 3. Fit `τ̂(x)` on the pseudo-target `Ỹ_i / T̃_i`, **weighted by `T̃_i²`**
///    (the "R-loss") -- this weighting is what gives the estimator its
///    quasi-oracle efficiency bound, not just unbiasedness; an unweighted
///    fit on the same pseudo-target is the (documented-inferior) U-learner
///    of Künzel et al. (2019).
///
/// # Weighting via row replication
///
/// This crate has no generic per-sample-weight support on `Learner`/
/// `RegressionTask` outside `XGBoost`/`GeoXGBoost` (both bespoke builder
/// methods outside the trait), so restricting the effect-stage learner to
/// one specific type would break the uniform `LearnerFactory` abstraction
/// this whole module (and `Bagging`/`Stacking`) is built on. Instead, the
/// `T̃_i²` weights are approximated by **row replication**: each kept row is
/// duplicated a number of times proportional to its (mean-normalized)
/// weight, and the effect learner is fit unweighted on the replicated
/// dataset -- an unweighted fit on k copies of a row is exactly a weight-k
/// contribution to squared-error loss. Replica counts are rounded and
/// capped (`MAX_REPLICAS_PER_ROW`) to bound the dataset blow-up; since
/// `T ∈ {0, 1}` and `ê(x) ∈ [0, 1]`, `T̃_i² ∈ (0, 1]`, so the ratio between
/// any two rows' weights is bounded regardless of `residual_clip`. This is
/// an approximation (integer replication can't match a continuous weight
/// exactly, and duplicated rows interact with any internal bootstrapping the
/// effect learner does, e.g. `RandomForest`), not the exact weighted
/// R-loss -- but it captures the intended emphasis on units with
/// well-identified treatment residuals, unlike the plain U-learner.
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
            residual_clip: 0.05,
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
    /// pseudo-target `Ỹ_i/T̃_i` blows up as `T̃_i -> 0`). Default `0.05`,
    /// consistent with the 0.01-0.1 trimming range standard in the
    /// propensity-trimming literature (the earlier default of `1e-3` let a
    /// single near-deterministic propensity blow the pseudo-target up by a
    /// factor of ~1000).
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
        let mut r_loss_weight = Vec::with_capacity(n); // T̃_i², the R-loss weight
        for i in 0..n {
            let y_resid = outcome[i] - m_hat[i];
            let t_resid = treatment[i] as f64 - e_hat[i];
            if t_resid.abs() >= self.residual_clip {
                kept_idx.push(i);
                pseudo_target.push(y_resid / t_resid);
                r_loss_weight.push(t_resid * t_resid);
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

        // Approximate the T̃²-weighted R-loss by row replication (see the
        // struct-level docs): duplicate each kept row proportionally to its
        // mean-normalized weight, then fit unweighted on the replicated set.
        let replicated_idx =
            replicate_by_weight(&r_loss_weight, TARGET_MEAN_REPLICAS, MAX_REPLICAS_PER_ROW);

        let final_row_idx: Vec<usize> = replicated_idx.iter().map(|&k| kept_idx[k]).collect();
        let final_target: Vec<f64> = replicated_idx.iter().map(|&k| pseudo_target[k]).collect();
        let final_features = features.select(Axis(0), &final_row_idx);
        let final_task = RegressionTask::new("r_learner_effect", final_features, final_target)?;
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

    #[test]
    fn replicate_by_weight_is_proportional_to_normalized_weight() {
        // Equal weights -> every row gets (approximately) the target mean.
        let equal = vec![1.0, 1.0, 1.0, 1.0];
        let reps = replicate_by_weight(&equal, 3.0, 20);
        for k in 0..equal.len() {
            let count = reps.iter().filter(|&&i| i == k).count();
            assert_eq!(count, 3, "equal weights should each replicate to the target mean");
        }

        // A row with 10x the others' weight should replicate proportionally
        // more often (capped, not literally 10x if that would exceed max).
        let skewed = vec![1.0, 1.0, 1.0, 10.0];
        let reps = replicate_by_weight(&skewed, 3.0, 20);
        let count_small = reps.iter().filter(|&&i| i == 0).count();
        let count_big = reps.iter().filter(|&&i| i == 3).count();
        assert!(
            count_big > count_small,
            "higher-weight row should replicate more often: big={count_big} small={count_small}"
        );
    }

    #[test]
    fn replicate_by_weight_never_drops_a_row_and_respects_the_cap() {
        // Near-zero weight still gets at least 1 replica (never dropped),
        // and an extreme weight is capped rather than blowing up the count.
        let weights = vec![1e-9, 1.0, 1e9];
        let reps = replicate_by_weight(&weights, 3.0, 20);
        assert!(reps.contains(&0), "near-zero-weight row must still appear at least once");
        let count_extreme = reps.iter().filter(|&&i| i == 2).count();
        assert!(count_extreme <= 20, "replica count must respect the cap, got {count_extreme}");
    }
}
