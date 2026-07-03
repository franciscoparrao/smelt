//! Causal meta-learners: T/S/X/R/DR-learner for estimating heterogeneous
//! treatment effects (CATE) by composing ordinary regression/classification
//! learners, rather than a purpose-built tree ensemble like [`super::CausalForest`].
//!
//! References:
//! - Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019).
//!   Metalearners for estimating heterogeneous treatment effects using
//!   machine learning. PNAS. (T/S/X-learner)
//! - Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous
//!   treatment effects. Biometrika. (R-learner)
//! - Kennedy, E. H. (2020). Optimal doubly robust estimation of
//!   heterogeneous causal effects. (DR-learner)
//!
//! # Why these aren't `Learner` implementors
//!
//! Every meta-learner here needs three aligned inputs — `features`,
//! `treatment`, `outcome` — not the `(X, y)` pair `Learner::train_regress`
//! expects. Smuggling treatment in as a feature column would also break
//! `TrainedModel::predict(&Array2<f64>)` for CATE: predicting a treatment
//! effect means evaluating *both* potential-outcome arms for the same `X`,
//! not "predict target from features+whatever value happens to be in the
//! treatment column." [`super::CausalForest`] hit this exact issue and
//! settled on a standalone `estimate(features, treatment, outcome, ...)`
//! entry point — these meta-learners follow that same precedent rather
//! than inventing a new convention.

pub mod cross_fit;
pub mod dr_learner;
pub mod r_learner;
pub mod s_learner;
pub mod t_learner;
pub mod x_learner;

pub use dr_learner::DrLearner;
pub use r_learner::RLearner;
pub use s_learner::SLearner;
pub use t_learner::TLearner;
pub use x_learner::XLearner;

use crate::learner::Learner;
use crate::{Result, SmeltError};
use ndarray::Array2;

/// Factory for a base learner used internally by a causal meta-learner (an
/// outcome model, a propensity model, an effect model, ...). Mirrors
/// `Bagging`/`Stacking`'s `Fn() -> Box<dyn Learner> + Send + Sync` factory:
/// called once per arm/fold to get a fresh, untrained learner instance.
pub type LearnerFactory = Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>;

/// Per-unit CATE point estimates from a T/S/X/R/DR-learner.
///
/// Unlike `CausalForestResult`, this has no `std_error`/`ci_*`: none of
/// these estimators has `CausalForest`'s honest-splitting + infinitesimal-
/// jackknife machinery for a built-in variance estimate. A bootstrap-based
/// SE (refit the whole meta-learner B times over resampled data, take the
/// spread) is a natural follow-up, deliberately out of scope here.
#[derive(Debug, Clone)]
pub struct MetaLearnerResult {
    /// Estimated treatment effect for each unit, same order as the input rows.
    pub cate: Vec<f64>,
    /// Average treatment effect: the simple mean of `cate`.
    pub ate: f64,
}

impl MetaLearnerResult {
    pub(crate) fn new(cate: Vec<f64>) -> Self {
        let ate = cate.iter().sum::<f64>() / cate.len() as f64;
        Self { cate, ate }
    }
}

/// Shape/content checks shared by every meta-learner's `estimate()`.
/// `CausalForest` doesn't validate `treatment` is binary (it just tests
/// `treatment[i] == 1`), but the S-learner uses `treatment` as a raw
/// numeric feature column, where a non-binary value would silently change
/// semantics rather than erroring loudly -- so this helper adds that check
/// for the new learners even though the existing `CausalForest` doesn't
/// have it.
pub(crate) fn validate_causal_inputs(
    features: &Array2<f64>,
    treatment: &[usize],
    outcome: &[f64],
) -> Result<()> {
    let n = features.nrows();
    if treatment.len() != n {
        return Err(SmeltError::DimensionMismatch {
            expected: n,
            got: treatment.len(),
        });
    }
    if outcome.len() != n {
        return Err(SmeltError::DimensionMismatch {
            expected: n,
            got: outcome.len(),
        });
    }
    if treatment.iter().any(|&t| t > 1) {
        return Err(SmeltError::InvalidParameter(
            "treatment must be binary (0 = control, 1 = treated)".into(),
        ));
    }
    if !treatment.contains(&0) || !treatment.contains(&1) {
        return Err(SmeltError::InvalidParameter(
            "need at least one control (treatment=0) and one treated (treatment=1) unit".into(),
        ));
    }
    Ok(())
}

/// Synthetic data-generating processes shared by every meta-learner's test
/// module, so PEHE/ATE-bias assertions are comparable across T/S/X/R/DR.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use ndarray::Array2;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// `tau(x) = 2*x0` (linear, heterogeneous), RCT propensity 0.5 (treatment
    /// independent of X), baseline `m0(x) = x1`, Gaussian-ish noise.
    /// Returns `(features [n x 2], treatment, outcome, true_cate)`.
    pub(crate) fn synthetic_linear_cate(
        n: usize,
        seed: u64,
        noise_sd: f64,
    ) -> (Array2<f64>, Vec<usize>, Vec<f64>, Vec<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut features = Array2::zeros((n, 2));
        let mut treatment = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        let mut true_cate = Vec::with_capacity(n);
        for i in 0..n {
            let x0 = rng.random::<f64>() * 4.0 - 2.0; // [-2, 2]
            let x1 = rng.random::<f64>() * 4.0 - 2.0;
            features[[i, 0]] = x0;
            features[[i, 1]] = x1;
            let t = (i % 2) as f64; // balanced RCT assignment
            let tau = 2.0 * x0;
            let noise = (rng.random::<f64>() - 0.5) * 2.0 * noise_sd;
            let y = x1 + tau * t + noise;
            treatment.push(t as usize);
            outcome.push(y);
            true_cate.push(tau);
        }
        (features, treatment, outcome, true_cate)
    }

    /// `tau(x) = x0*x1` (nonlinear interaction), confounded propensity
    /// `e(x) = sigmoid(x0)` (NOT a coin flip -- treatment correlates with
    /// X0), baseline `m0(x) = x1`. T/S-learner are known in the literature
    /// to degrade specifically in this confounded/heterogeneous regime;
    /// this fixture is what differentiates X/R/DR-learner from them.
    /// Returns `(features [n x 2], treatment, outcome, true_cate)`.
    pub(crate) fn synthetic_confounded_nonlinear_cate(
        n: usize,
        seed: u64,
        noise_sd: f64,
    ) -> (Array2<f64>, Vec<usize>, Vec<f64>, Vec<f64>) {
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
            let propensity = 1.0 / (1.0 + (-x0).exp()); // sigmoid(x0)
            let t = if rng.random::<f64>() < propensity { 1 } else { 0 };
            let tau = x0 * x1;
            let noise = (rng.random::<f64>() - 0.5) * 2.0 * noise_sd;
            let y = x1 + tau * t as f64 + noise;
            treatment.push(t);
            outcome.push(y);
            true_cate.push(tau);
        }
        (features, treatment, outcome, true_cate)
    }
}
