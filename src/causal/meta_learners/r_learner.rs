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
/// Absolute floor on `|T̃_i|` below which a row is always excluded from the
/// effect-stage regression, regardless of how low `residual_clip` is set.
/// Rationale: the pseudo-target `Ỹ_i/T̃_i` overflows toward ±inf as
/// `T̃_i -> 0` (and is NaN at exactly 0), which would fail
/// `RegressionTask`'s finite-target validation even though the row's R-loss
/// weight `T̃_i² <= 1e-16` makes its contribution numerically negligible
/// anyway -- so dropping it changes the fit by at most that same negligible
/// amount while keeping the task constructible. `1e-8` (so `T̃² <= 1e-16`,
/// around `f64` epsilon relative to the O(1) weights of well-identified
/// rows) is far below any statistically sensible `residual_clip`, so the
/// floor only ever bites on misconfigured clips (e.g. `0.0`).
const T_RESID_EPS: f64 = 1e-8;

/// Replica counts approximating a `weights`-weighted regression via row
/// duplication: index `k`'s count is proportional to its mean-normalized
/// weight (rounded, clamped to `[1, max_replicas]`). Returns a flat list of
/// indices into `weights` with each `k` repeated its replica count times —
/// fitting an unweighted model on the rows selected by this list
/// approximates fitting a `weights`-weighted model on the original rows.
fn replicate_by_weight(
    weights: &[f64],
    target_mean_replicas: f64,
    max_replicas: usize,
) -> Vec<usize> {
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
/// # Weighting: exact R-loss vs. row-replication fallback
///
/// The effect-stage regression should minimize the weighted R-loss
/// `Σ T̃_i² · (Ỹ_i/T̃_i − τ(x_i))²` (algebraically identical to
/// `Σ (Ỹ_i − τ(x_i)·T̃_i)²`, the paper's Eq. 4 empirical loss). Since the
/// crate now has generic per-sample-weight support
/// ([`RegressionTask::with_weights`] + [`Learner::supports_weights`]),
/// which path runs is decided **by introspection** at [`Self::estimate`]
/// time, asking a fresh instance from `effect_factory` whether it consumes
/// weights:
///
/// - **Exact weighted R-loss** (`supports_weights() == true`): the
///   pseudo-target `Ỹ_i/T̃_i` is regressed on a task carrying `T̃_i²` as
///   per-sample weights -- Nie & Wager's actual R-loss, no approximation.
///   Built-in effect bases on this path: decision tree, random forest,
///   extra trees, gradient boosting, XGBoost, LightGBM, CatBoost,
///   linear/ridge/lasso/elastic-net regression, ELM.
/// - **Row-replication fallback** (`supports_weights() == false`, e.g.
///   KNN): the pre-introspection approximation is kept, unchanged: each
///   kept row is duplicated a number of times proportional to its
///   (mean-normalized) weight, and the effect learner is fit unweighted on
///   the replicated dataset -- an unweighted fit on k copies of a row is
///   exactly a weight-k contribution to squared-error loss. Replica counts
///   are rounded and capped (`MAX_REPLICAS_PER_ROW`) to bound the dataset
///   blow-up; since `T ∈ {0, 1}` and `ê(x) ∈ [0, 1]`, `T̃_i² ∈ (0, 1]`, so
///   the ratio between any two rows' weights is bounded regardless of
///   `residual_clip`. This is an approximation (integer replication can't
///   match a continuous weight exactly, and duplicated rows interact with
///   any internal bootstrapping the effect learner does), but it captures
///   the intended emphasis on units with well-identified treatment
///   residuals, unlike the plain U-learner (Künzel et al. 2019).
///
/// In both paths, rows with `|T̃_i| < max(residual_clip, 1e-8)` are excluded
/// *before* the effect-stage task is built (see [`Self::with_residual_clip`]
/// and the `T_RESID_EPS` constant for the rationale on each bound); if every
/// row is excluded, `estimate` returns a clear error rather than fitting on
/// nothing.
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
    ///
    /// Regardless of how low this is set, an absolute floor of `1e-8`
    /// (`T_RESID_EPS`) always applies: a `T̃_i` at exactly (or numerically
    /// indistinguishable from) zero would make the pseudo-target ±inf/NaN
    /// and fail task validation outright, even though such a row's R-loss
    /// weight `T̃_i² <= 1e-16` is negligible either way.
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
        // Effective exclusion threshold: the user-facing clip, floored at
        // T_RESID_EPS so a T̃ at (numerically) zero can never reach the
        // pseudo-target division below (see the constant's docs).
        let clip = self.residual_clip.max(T_RESID_EPS);
        let mut kept_idx = Vec::with_capacity(n);
        let mut pseudo_target = Vec::with_capacity(n);
        let mut r_loss_weight = Vec::with_capacity(n); // T̃_i², the R-loss weight
        for i in 0..n {
            let y_resid = outcome[i] - m_hat[i];
            let t_resid = treatment[i] as f64 - e_hat[i];
            if t_resid.abs() >= clip {
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

        // Introspection decides the weighting path (see the struct-level
        // docs): a weight-consuming effect learner gets the exact T̃²-weighted
        // R-loss; a weight-blind one keeps the row-replication approximation.
        //
        // `with_weights` panics (by contract) on invalid weights, but every
        // weight here is provably valid: `T̃_i²` is finite and `>= clip² > 0`
        // for every kept row (a NaN `T̃` fails the `>= clip` keep-check, so
        // NaN can't slip through), one weight per kept row by construction,
        // and "all zero" is impossible since all are strictly positive.
        let final_task = if (self.effect_factory)().supports_weights() {
            let final_features = features.select(Axis(0), &kept_idx);
            RegressionTask::new("r_learner_effect", final_features, pseudo_target)?
                .with_weights(r_loss_weight)
        } else {
            let replicated_idx =
                replicate_by_weight(&r_loss_weight, TARGET_MEAN_REPLICAS, MAX_REPLICAS_PER_ROW);
            let final_row_idx: Vec<usize> = replicated_idx.iter().map(|&k| kept_idx[k]).collect();
            let final_target: Vec<f64> = replicated_idx.iter().map(|&k| pseudo_target[k]).collect();
            let final_features = features.select(Axis(0), &final_row_idx);
            RegressionTask::new("r_learner_effect", final_features, final_target)?
        };
        let tau_model = (self.effect_factory)().train_regress(&final_task)?;

        let pred = tau_model.predict(features)?;
        let Prediction::Regression {
            predicted: cate, ..
        } = pred
        else {
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
    use crate::learner::{
        KNearestNeighbors, LinearRegression, LogisticRegression, RandomForest, TrainedModel,
    };
    use crate::measure::{AteBias, Measure, Pehe};
    use crate::prediction::Prediction;
    use crate::task::ClassificationTask;

    /// OLS with its weight support hidden: identical math to
    /// [`LinearRegression`] but `supports_weights()` stays `false`, so it
    /// deterministically forces the row-replication fallback path -- the
    /// only way to compare exact vs. fallback with the *same* effect-stage
    /// math on both sides.
    struct UnweightedLinear;
    impl Learner for UnweightedLinear {
        fn id(&self) -> &str {
            "test.unweighted_linear"
        }
        fn train_regress(&mut self, task: &RegressionTask) -> crate::Result<Box<dyn TrainedModel>> {
            LinearRegression::new().train_regress(task)
        }
    }

    /// Propensity stub predicting exactly `P(T=1|x) = 0.5` for every unit,
    /// so `T̃_i = ±0.5` exactly and the R-loss weights are all `0.25`.
    struct ConstantHalfPropensity;
    struct TrainedConstantHalf;
    impl Learner for ConstantHalfPropensity {
        fn id(&self) -> &str {
            "test.constant_half_propensity"
        }
        fn train_classif(
            &mut self,
            _task: &ClassificationTask,
        ) -> crate::Result<Box<dyn TrainedModel>> {
            Ok(Box::new(TrainedConstantHalf))
        }
    }
    impl TrainedModel for TrainedConstantHalf {
        fn predict(&self, features: &Array2<f64>) -> crate::Result<Prediction> {
            let n = features.nrows();
            Ok(Prediction::Classification {
                predicted: vec![0; n],
                truth: None,
                probabilities: Some(vec![vec![0.5, 0.5]; n]),
            })
        }
    }

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
        let result = r_learner_rf()
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
            synthetic_confounded_nonlinear_cate(400, 8, 0.1);
        let result = r_learner_rf()
            .estimate(&features, &treatment, &outcome)
            .unwrap();
        let pred = Prediction::causal_effect_with_truth(result.cate, true_cate);
        assert!(Pehe.score(&pred).unwrap().is_finite());
    }

    #[test]
    fn rejects_non_binary_treatment() {
        let features = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let err = r_learner_rf().estimate(
            &features,
            &[0, 1, 2, 0, 1, 0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        assert!(err.is_err());
    }

    #[test]
    fn rejects_single_arm() {
        let features = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let err = r_learner_rf().estimate(
            &features,
            &[0, 0, 0, 0, 0, 0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        );
        assert!(err.is_err());
    }

    #[test]
    fn replicate_by_weight_is_proportional_to_normalized_weight() {
        // Equal weights -> every row gets (approximately) the target mean.
        let equal = vec![1.0, 1.0, 1.0, 1.0];
        let reps = replicate_by_weight(&equal, 3.0, 20);
        for k in 0..equal.len() {
            let count = reps.iter().filter(|&&i| i == k).count();
            assert_eq!(
                count, 3,
                "equal weights should each replicate to the target mean"
            );
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

    /// Heteroscedastic-treatment-residual DGP: propensity varies sharply
    /// with `x0` (near-deterministic at the extremes), so `|T̃_i|` -- and
    /// with it the pseudo-target's noise variance `Var(noise)/T̃_i²` --
    /// varies over orders of magnitude across units. CATE `tau = 1 + 2*x1`
    /// depends only on `x1`, so an effect-stage fit corrupted by the noisy
    /// low-`|T̃|` units shows up directly as PEHE.
    fn synthetic_heteroscedastic_propensity(
        n: usize,
        seed: u64,
    ) -> (Array2<f64>, Vec<usize>, Vec<f64>, Vec<f64>) {
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut features = Array2::zeros((n, 2));
        let mut treatment = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        let mut true_cate = Vec::with_capacity(n);
        for i in 0..n {
            let x0 = rng.random::<f64>() * 4.0 - 2.0;
            let x1 = rng.random::<f64>() * 4.0 - 2.0;
            features[[i, 0]] = x0;
            features[[i, 1]] = x1;
            let e = 0.03 + 0.94 / (1.0 + (-4.0 * x0).exp());
            let t = if rng.random::<f64>() < e { 1 } else { 0 };
            let tau = 1.0 + 2.0 * x1;
            let noise = (rng.random::<f64>() - 0.5) * 2.0;
            let y = x1 + tau * t as f64 + noise;
            treatment.push(t);
            outcome.push(y);
            true_cate.push(tau);
        }
        (features, treatment, outcome, true_cate)
    }

    #[test]
    fn exact_weighted_r_loss_beats_replication_under_heteroscedastic_residuals() {
        let (features, treatment, outcome, true_cate) =
            synthetic_heteroscedastic_propensity(600, 11);

        // Identical outcome/propensity models (same seeds -> identical
        // m_hat/e_hat), identical effect-stage math (OLS both sides); the
        // ONLY difference is exact T̃² weighting vs. row replication.
        let exact = RLearner::new(
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(1)),
            || Box::new(LogisticRegression::new()),
            || Box::new(LinearRegression::new()),
        )
        .estimate(&features, &treatment, &outcome)
        .unwrap();
        let fallback = RLearner::new(
            || Box::new(RandomForest::new().with_n_estimators(50).with_seed(1)),
            || Box::new(LogisticRegression::new()),
            || Box::new(UnweightedLinear),
        )
        .estimate(&features, &treatment, &outcome)
        .unwrap();

        let max_diff = exact
            .cate
            .iter()
            .zip(&fallback.cate)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff > 1e-6,
            "exact and replication paths should produce different tau-hat, max diff {max_diff}"
        );

        let pehe_exact = Pehe
            .score(&Prediction::causal_effect_with_truth(
                exact.cate,
                true_cate.clone(),
            ))
            .unwrap();
        let pehe_fallback = Pehe
            .score(&Prediction::causal_effect_with_truth(
                fallback.cate,
                true_cate,
            ))
            .unwrap();
        assert!(
            pehe_exact < pehe_fallback,
            "exact R-loss should beat the replication approximation under \
             heteroscedastic treatment residuals: exact {pehe_exact} vs fallback {pehe_fallback}"
        );
    }

    /// Guards the introspection fallback against regressions: with a
    /// weight-blind effect base (KNN), `estimate` must reproduce the
    /// pre-exact-R-loss row-replication output **bit for bit**. The
    /// expected bit patterns were captured on the commit immediately before
    /// the exact-weighted path landed (same fixture, same seeds).
    #[test]
    fn weight_blind_effect_base_keeps_replication_path_bit_identical() {
        const EXPECTED_CATE_BITS: [u64; 10] = [
            13840701222709684699,
            4591294917977811371,
            13822378950606734926,
            4610853263513562214,
            13836221849996537715,
            4616934202024969749,
            4603534635365457444,
            13820250705773657597,
            13831746304029155948,
            4603555024545072614,
        ];
        const EXPECTED_ATE_BITS: u64 = 13821715289238223333;

        let (features, treatment, outcome, _) = synthetic_linear_cate(120, 7, 0.1);
        let result = RLearner::new(
            || Box::new(RandomForest::new().with_n_estimators(30).with_seed(1)),
            || Box::new(LogisticRegression::new()),
            || Box::new(KNearestNeighbors::new(5)),
        )
        .estimate(&features, &treatment, &outcome)
        .unwrap();

        for (i, (&v, &bits)) in result
            .cate
            .iter()
            .zip(EXPECTED_CATE_BITS.iter())
            .enumerate()
        {
            assert_eq!(
                v.to_bits(),
                bits,
                "cate[{i}] = {v} diverged from the pre-change reference"
            );
        }
        assert_eq!(
            result.ate.to_bits(),
            EXPECTED_ATE_BITS,
            "ATE diverged from the pre-change reference"
        );
    }

    #[test]
    fn zero_residual_clip_is_floored_and_never_produces_nan() {
        // clip = 0.0 would previously let |T̃| ~ 0 reach the Ỹ/T̃ division;
        // the T_RESID_EPS floor must keep every estimate finite (no NaN,
        // no panic, no task-validation error from an inf pseudo-target).
        let (features, treatment, outcome, _) = synthetic_confounded_nonlinear_cate(300, 8, 0.1);
        let result = RLearner::new(
            || Box::new(RandomForest::new().with_n_estimators(30).with_seed(1)),
            || Box::new(LogisticRegression::new()),
            || Box::new(LinearRegression::new()),
        )
        .with_residual_clip(0.0)
        .estimate(&features, &treatment, &outcome)
        .unwrap();
        assert!(
            result.cate.iter().all(|v| v.is_finite()),
            "all CATE estimates must be finite with residual_clip = 0"
        );
    }

    #[test]
    fn all_residuals_below_clip_errors_cleanly() {
        // |T̃| <= 1 always (T in {0,1}, e in [0,1]), so a clip above 1
        // deterministically excludes every unit -- degenerate-propensity
        // path must be a clean Err naming residual_clip, never a panic.
        let (features, treatment, outcome, _) = synthetic_linear_cate(100, 7, 0.1);
        let err = RLearner::new(
            || Box::new(RandomForest::new().with_n_estimators(10).with_seed(1)),
            || Box::new(LogisticRegression::new()),
            || Box::new(LinearRegression::new()),
        )
        .with_residual_clip(1.5)
        .estimate(&features, &treatment, &outcome)
        .unwrap_err();
        assert!(
            err.to_string().contains("residual_clip"),
            "error should name residual_clip, got: {err}"
        );
    }

    #[test]
    fn constant_treatment_residuals_make_exact_and_fallback_agree() {
        // Perfectly balanced RCT + a propensity stub predicting exactly 0.5
        // -> T̃_i = ±0.5 for every unit, so all R-loss weights equal 0.25.
        // A constant-weighted OLS is the same normal-equations solution as
        // unweighted OLS, and uniform replication (3 copies each) is too:
        // the exact and fallback paths must agree to solver tolerance.
        let (features, treatment, outcome, _) = synthetic_linear_cate(200, 7, 0.1);
        let exact = RLearner::new(
            || Box::new(RandomForest::new().with_n_estimators(30).with_seed(1)),
            || Box::new(ConstantHalfPropensity),
            || Box::new(LinearRegression::new()),
        )
        .estimate(&features, &treatment, &outcome)
        .unwrap();
        let fallback = RLearner::new(
            || Box::new(RandomForest::new().with_n_estimators(30).with_seed(1)),
            || Box::new(ConstantHalfPropensity),
            || Box::new(UnweightedLinear),
        )
        .estimate(&features, &treatment, &outcome)
        .unwrap();

        for (i, (a, b)) in exact.cate.iter().zip(&fallback.cate).enumerate() {
            assert!(
                (a - b).abs() < 1e-8,
                "constant T̃ should make paths equivalent; cate[{i}]: exact {a} vs fallback {b}"
            );
        }
    }

    #[test]
    fn replicate_by_weight_never_drops_a_row_and_respects_the_cap() {
        // Near-zero weight still gets at least 1 replica (never dropped),
        // and an extreme weight is capped rather than blowing up the count.
        let weights = vec![1e-9, 1.0, 1e9];
        let reps = replicate_by_weight(&weights, 3.0, 20);
        assert!(
            reps.contains(&0),
            "near-zero-weight row must still appear at least once"
        );
        let count_extreme = reps.iter().filter(|&&i| i == 2).count();
        assert!(
            count_extreme <= 20,
            "replica count must respect the cap, got {count_extreme}"
        );
    }
}
