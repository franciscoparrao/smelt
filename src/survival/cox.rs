//! Cox Proportional Hazards regression.
//!
//! Semi-parametric survival model: the hazard for an individual with
//! covariates `x` is `h(t | x) = h_0(t) · exp(βᵀx)`, where the baseline
//! hazard `h_0(t)` is left unspecified and the coefficients `β` are estimated
//! by maximizing the **partial** likelihood (Cox 1972), which cancels
//! `h_0(t)` out entirely. Optimization is Newton–Raphson on the partial
//! log-likelihood (globally concave, so it converges reliably from `β = 0`);
//! tied event times use the **Breslow** approximation. The baseline
//! cumulative hazard is recovered afterward with the Breslow estimator, giving
//! full survival curves per individual — matching [`SurvivalPrediction`], the
//! same output type [`super::RandomSurvivalForest`] produces.
//!
//! Reference: Cox, D. R. (1972). Regression models and life-tables. JRSS-B.

use super::{SurvivalEvent, SurvivalPrediction};
use crate::{Result, SmeltError};
use ndarray::{Array2, ArrayView1};

/// Cox Proportional Hazards regression model.
///
/// Features are mean-centered internally (so the recovered baseline hazard is
/// that of the *average* individual, matching R `survival::basehaz(centered =
/// TRUE)`); centering does not change `β`, since the partial likelihood
/// depends only on covariate differences within each risk set. An optional
/// `l2` ridge penalty (`0` by default) stabilizes the fit under collinear or
/// high-dimensional covariates — the penalized partial log-likelihood is
/// `ℓ(β) − (l2/2)‖β‖²`.
///
/// Ties are handled with the Breslow approximation, not Efron's — Breslow is
/// the simpler, widely-used default; Efron (more accurate when many events
/// share a time) is a possible future addition.
///
/// # Examples
///
/// ```
/// use smelt_ml::survival::{CoxPH, SurvivalEvent};
/// use ndarray::array;
///
/// // Higher covariate broadly means an earlier event (positive association
/// // with risk); one inversion keeps the MLE finite (perfect separation
/// // would send it to infinity).
/// let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
/// let events = vec![
///     SurvivalEvent { time: 6.0, event: true },
///     SurvivalEvent { time: 4.0, event: true }, // inversion vs the next row
///     SurvivalEvent { time: 5.0, event: true },
///     SurvivalEvent { time: 3.0, event: true },
///     SurvivalEvent { time: 2.0, event: true },
///     SurvivalEvent { time: 1.0, event: true },
/// ];
/// let model = CoxPH::new().fit(&x, &events).unwrap();
/// assert!(model.coefficients()[0] > 0.0); // risk rises with the covariate
/// ```
pub struct CoxPH {
    max_iter: usize,
    tol: f64,
    l2: f64,
}

impl Default for CoxPH {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-9,
            l2: 0.0,
        }
    }
}

impl CoxPH {
    /// Create a Cox model with default settings (100 Newton iterations,
    /// convergence tolerance `1e-9`, no ridge penalty).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of Newton–Raphson iterations.
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set the convergence tolerance (max absolute coefficient update).
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the L2 (ridge) penalty on the coefficients. `0` (the default)
    /// reproduces an unpenalized `coxph` fit; a positive value shrinks
    /// coefficients toward zero and keeps the information matrix invertible
    /// under collinear covariates.
    pub fn with_l2(mut self, l2: f64) -> Self {
        self.l2 = l2;
        self
    }

    /// Fit the model to `features` (one row per subject) and `events`
    /// (time + event/censoring indicator, aligned with the rows).
    ///
    /// Fails if the shapes disagree, any value is non-finite, or every
    /// observation is censored (the partial likelihood has no event terms to
    /// maximize). May also fail if the information matrix is singular and no
    /// `l2` penalty was set (perfectly collinear covariates).
    pub fn fit(&self, features: &Array2<f64>, events: &[SurvivalEvent]) -> Result<TrainedCoxPH> {
        let n = features.nrows();
        let p = features.ncols();
        if events.len() != n {
            return Err(SmeltError::DimensionMismatch {
                expected: n,
                got: events.len(),
            });
        }
        if n == 0 || p == 0 {
            return Err(SmeltError::InvalidParameter(
                "CoxPH requires a non-empty feature matrix".into(),
            ));
        }
        if !features.iter().all(|v| v.is_finite()) {
            return Err(SmeltError::InvalidParameter(
                "CoxPH features contain non-finite values; impute or drop them first".into(),
            ));
        }
        if !events.iter().all(|e| e.time.is_finite()) {
            return Err(SmeltError::InvalidParameter(
                "CoxPH event times must be finite".into(),
            ));
        }
        if !events.iter().any(|e| e.event) {
            return Err(SmeltError::InvalidParameter(
                "CoxPH requires at least one event; all observations are censored".into(),
            ));
        }

        // Center features: β is unchanged, but the Breslow baseline then
        // describes the mean individual (η = 0 at the mean).
        let means: Vec<f64> = (0..p)
            .map(|j| features.column(j).mean().unwrap_or(0.0))
            .collect();
        let z: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..p).map(|j| features[[i, j]] - means[j]).collect())
            .collect();

        // Sample indices sorted by time ascending; the risk-set accumulation
        // below walks them from the largest time downward.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            events[a]
                .time
                .partial_cmp(&events[b].time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Newton–Raphson on the (penalized) partial log-likelihood.
        let mut beta = vec![0.0; p];
        for _ in 0..self.max_iter {
            let (mut grad, mut info, loglik) = compute_derivatives(&z, events, &order, &beta, p);
            // Ridge: gradient − l2·β, information + l2·I.
            for a in 0..p {
                grad[a] -= self.l2 * beta[a];
                info[a][a] += self.l2;
            }
            let delta = solve_symmetric(info, grad).ok_or_else(|| {
                SmeltError::InvalidParameter(
                    "CoxPH information matrix is singular; set an l2 penalty or drop collinear \
                     covariates"
                        .into(),
                )
            })?;

            // Step-halving safeguard: the full Newton step almost always
            // increases the concave objective, but halve it if it ever
            // doesn't, so a pathological start can't overshoot.
            let mut step = 1.0;
            let mut candidate = beta.clone();
            for _ in 0..HALVING_LIMIT {
                for a in 0..p {
                    candidate[a] = beta[a] + step * delta[a];
                }
                let new_ll = penalized_loglik(&z, events, &order, &candidate, self.l2);
                if new_ll >= loglik - 1e-12 {
                    break;
                }
                step *= 0.5;
            }

            let max_update = (0..p)
                .map(|a| (candidate[a] - beta[a]).abs())
                .fold(0.0, f64::max);
            beta = candidate;
            if max_update < self.tol {
                break;
            }
        }

        // Breslow baseline cumulative hazard at the fitted β, on the grid of
        // unique event times (ascending).
        let (baseline_times, baseline_cumhaz) = breslow_baseline(&z, events, &order, &beta);
        let log_partial_likelihood = penalized_loglik(&z, events, &order, &beta, 0.0);

        Ok(TrainedCoxPH {
            coefficients: beta,
            means,
            baseline_times,
            baseline_cumhaz,
            log_partial_likelihood,
        })
    }
}

/// Number of step-halving attempts before accepting a Newton step regardless.
const HALVING_LIMIT: usize = 20;

/// Compute the gradient (score), observed information matrix, and partial
/// log-likelihood at `beta`, using the Breslow approximation for tied event
/// times. A single descending pass over event times accumulates the risk-set
/// sums `S0 = Σ w_j`, `S1 = Σ w_j z_j`, `S2 = Σ w_j z_j z_jᵀ`, so the whole
/// thing is `O(n·p²)` rather than re-scanning the risk set at every event.
fn compute_derivatives(
    z: &[Vec<f64>],
    events: &[SurvivalEvent],
    order: &[usize],
    beta: &[f64],
    p: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, f64) {
    let n = order.len();
    let w: Vec<f64> = z.iter().map(|row| dot(beta, row).exp()).collect();

    let mut s0 = 0.0;
    let mut s1 = vec![0.0; p];
    let mut s2 = vec![vec![0.0; p]; p];
    let mut grad = vec![0.0; p];
    let mut info = vec![vec![0.0; p]; p];
    let mut loglik = 0.0;

    // Walk unique times from largest to smallest; every sample at the current
    // time enters the risk set (risk set at t = {j : t_j ≥ t}).
    let mut k = n;
    while k > 0 {
        let t = events[order[k - 1]].time;
        let mut d = 0usize;
        let mut sum_event_z = vec![0.0; p];
        let mut sum_event_eta = 0.0;
        while k > 0 && events[order[k - 1]].time == t {
            let i = order[k - 1];
            s0 += w[i];
            for a in 0..p {
                s1[a] += w[i] * z[i][a];
                for b in 0..p {
                    s2[a][b] += w[i] * z[i][a] * z[i][b];
                }
            }
            if events[i].event {
                d += 1;
                for a in 0..p {
                    sum_event_z[a] += z[i][a];
                }
                sum_event_eta += dot(beta, &z[i]);
            }
            k -= 1;
        }
        if d > 0 {
            let df = d as f64;
            loglik += sum_event_eta - df * s0.ln();
            let inv = 1.0 / s0;
            for a in 0..p {
                let mean_a = s1[a] * inv;
                grad[a] += sum_event_z[a] - df * mean_a;
                for b in 0..p {
                    info[a][b] += df * (s2[a][b] * inv - mean_a * s1[b] * inv);
                }
            }
        }
    }
    (grad, info, loglik)
}

/// Penalized partial log-likelihood `ℓ(β) − (l2/2)‖β‖²` at `beta` — the
/// lightweight objective the step-halving loop compares (no gradient/Hessian).
fn penalized_loglik(
    z: &[Vec<f64>],
    events: &[SurvivalEvent],
    order: &[usize],
    beta: &[f64],
    l2: f64,
) -> f64 {
    let n = order.len();
    let mut s0 = 0.0;
    let mut loglik = 0.0;
    let mut k = n;
    while k > 0 {
        let t = events[order[k - 1]].time;
        let mut d = 0usize;
        let mut sum_event_eta = 0.0;
        while k > 0 && events[order[k - 1]].time == t {
            let i = order[k - 1];
            s0 += dot(beta, &z[i]).exp();
            if events[i].event {
                d += 1;
                sum_event_eta += dot(beta, &z[i]);
            }
            k -= 1;
        }
        if d > 0 {
            loglik += sum_event_eta - (d as f64) * s0.ln();
        }
    }
    let penalty = 0.5 * l2 * beta.iter().map(|b| b * b).sum::<f64>();
    loglik - penalty
}

/// Breslow baseline cumulative hazard at `beta`, evaluated on the unique
/// event times (ascending). `dH_0(t) = d_t / Σ_{j: t_j ≥ t} exp(βᵀz_j)`,
/// accumulated into a cumulative sum.
fn breslow_baseline(
    z: &[Vec<f64>],
    events: &[SurvivalEvent],
    order: &[usize],
    beta: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n = order.len();
    let w: Vec<f64> = z.iter().map(|row| dot(beta, row).exp()).collect();
    let mut s0 = 0.0;
    // Increments come out in descending time order; reverse before cumsum.
    let mut incr: Vec<(f64, f64)> = Vec::new();
    let mut k = n;
    while k > 0 {
        let t = events[order[k - 1]].time;
        let mut d = 0usize;
        while k > 0 && events[order[k - 1]].time == t {
            let i = order[k - 1];
            s0 += w[i];
            if events[i].event {
                d += 1;
            }
            k -= 1;
        }
        if d > 0 {
            incr.push((t, d as f64 / s0));
        }
    }
    incr.reverse();
    let mut times = Vec::with_capacity(incr.len());
    let mut cumhaz = Vec::with_capacity(incr.len());
    let mut running = 0.0;
    for (t, dh) in incr {
        running += dh;
        times.push(t);
        cumhaz.push(running);
    }
    (times, cumhaz)
}

/// Dot product of a coefficient vector and a covariate row.
fn dot(beta: &[f64], z: &[f64]) -> f64 {
    beta.iter().zip(z).map(|(b, x)| b * x).sum()
}

/// Solve `A x = b` for a small square system via Gaussian elimination with
/// partial pivoting. Returns `None` if `A` is (numerically) singular. Used
/// for the Newton step; kept local per this crate's per-module hand-rolled
/// numeric-routine convention (cf. `regularized.rs`, `kriging_hybrid.rs`).
fn solve_symmetric(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut pivot = col;
        let mut max_abs = a[col][col].abs();
        for r in (col + 1)..n {
            if a[r][col].abs() > max_abs {
                max_abs = a[r][col].abs();
                pivot = r;
            }
        }
        if max_abs < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let diag = a[col][col];
        for r in (col + 1)..n {
            let factor = a[r][col] / diag;
            if factor != 0.0 {
                for c in col..n {
                    a[r][c] -= factor * a[col][c];
                }
                b[r] -= factor * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for c in (i + 1)..n {
            sum -= a[i][c] * x[c];
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

/// A fitted [`CoxPH`] model: coefficients plus the Breslow baseline cumulative
/// hazard, retained so it can predict survival curves on new subjects.
pub struct TrainedCoxPH {
    coefficients: Vec<f64>,
    means: Vec<f64>,
    baseline_times: Vec<f64>,
    baseline_cumhaz: Vec<f64>,
    log_partial_likelihood: f64,
}

impl TrainedCoxPH {
    /// The estimated coefficients `β`, one per feature.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Hazard ratios `exp(β)`: the multiplicative change in hazard per unit
    /// increase of each covariate.
    pub fn hazard_ratios(&self) -> Vec<f64> {
        self.coefficients.iter().map(|b| b.exp()).collect()
    }

    /// The maximized partial log-likelihood at the fitted coefficients
    /// (unpenalized, so it is comparable to `survival::coxph`'s `loglik`).
    pub fn log_partial_likelihood(&self) -> f64 {
        self.log_partial_likelihood
    }

    /// The linear risk score `βᵀ(x − x̄)` for one subject — higher means
    /// higher hazard. Order-equivalent to what [`super::concordance_index`]
    /// consumes.
    pub fn risk_score(&self, row: ArrayView1<f64>) -> f64 {
        row.iter()
            .zip(&self.coefficients)
            .zip(&self.means)
            .map(|((x, b), m)| b * (x - m))
            .sum()
    }

    /// Predict a full [`SurvivalPrediction`] for each row of `features`:
    /// `S(t | x) = exp(−H_0(t)·exp(βᵀ(x−x̄)))` on the training event-time grid.
    pub fn predict(&self, features: &Array2<f64>) -> Vec<SurvivalPrediction> {
        features
            .rows()
            .into_iter()
            .map(|row| {
                let risk = self.risk_score(row);
                let hr = risk.exp();
                let cumulative_hazard: Vec<f64> =
                    self.baseline_cumhaz.iter().map(|h| h * hr).collect();
                let survival: Vec<f64> = cumulative_hazard.iter().map(|h| (-h).exp()).collect();
                SurvivalPrediction {
                    times: self.baseline_times.clone(),
                    survival,
                    cumulative_hazard,
                    risk_score: risk,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::survival::concordance_index;
    use ndarray::array;

    /// The 20-sample, 2-covariate fixture used for the R golden below.
    fn golden_data() -> (Array2<f64>, Vec<SurvivalEvent>) {
        let time = [
            5.0, 6.0, 6.0, 2.0, 4.0, 4.0, 3.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let status = [
            true, true, false, true, true, false, true, true, false, true, false, true, true, true,
            false, true, false, true, true, false,
        ];
        let x1 = [
            1.2, 0.5, 2.1, -0.3, 1.1, 0.2, -1.0, 1.5, 0.7, -0.6, 2.0, 0.1, -1.2, 0.9, 1.3, -0.4,
            0.6, 1.8, -0.9, 0.3,
        ];
        let x2 = [
            0.3, -1.1, 0.8, 1.2, -0.5, 0.6, 1.0, -0.2, 0.4, 1.1, -0.7, 0.9, 0.2, -1.3, 0.5, 0.7,
            -0.8, 0.1, 1.4, -0.6,
        ];
        let features = Array2::from_shape_fn((20, 2), |(i, j)| if j == 0 { x1[i] } else { x2[i] });
        let events: Vec<SurvivalEvent> = (0..20)
            .map(|i| SurvivalEvent {
                time: time[i],
                event: status[i],
            })
            .collect();
        (features, events)
    }

    /// Golden test against R `survival::coxph(..., ties="breslow")` 3.5.8 on
    /// the fixed dataset above: coefficients, partial log-likelihood, and the
    /// Breslow baseline cumulative hazard (centered) must all match.
    #[test]
    fn matches_r_coxph_breslow_golden() {
        let (features, events) = golden_data();
        let model = CoxPH::new().fit(&features, &events).unwrap();

        // R: coef = 0.0086888145, 0.1267919026
        let coef = model.coefficients();
        assert!((coef[0] - 0.0086888145).abs() < 1e-6, "coef0 = {}", coef[0]);
        assert!((coef[1] - 0.1267919026).abs() < 1e-6, "coef1 = {}", coef[1]);

        // R: loglik[2] = -28.0783180331
        assert!(
            (model.log_partial_likelihood() - (-28.0783180331)).abs() < 1e-6,
            "loglik = {}",
            model.log_partial_likelihood()
        );

        // R basehaz(centered=TRUE): cumulative hazard, reported here on the
        // event-time grid only (t=10 is censored — a flat step in R's fuller
        // grid — so it is absent here, matching RandomSurvivalForest's
        // event-times-only convention for SurvivalPrediction).
        let expected_times = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0];
        let expected_cumhaz = [
            0.0497735194,
            0.1552385149,
            0.2139189785,
            0.3565096769,
            0.4401918092,
            0.6593461333,
            0.7988464215,
            0.9992221199,
            1.4812779899,
            2.4961145150,
        ];
        assert_eq!(model.baseline_times, expected_times);
        for (got, exp) in model.baseline_cumhaz.iter().zip(&expected_cumhaz) {
            assert!((got - exp).abs() < 1e-6, "baseline {got} vs {exp}");
        }
    }

    /// The score (gradient) at the fitted coefficients must be ~0 — the
    /// defining first-order condition of the MLE, checked independently of the
    /// convergence loop's own stopping rule.
    #[test]
    fn score_vanishes_at_the_optimum() {
        let (features, events) = golden_data();
        let model = CoxPH::new()
            .with_tol(1e-12)
            .fit(&features, &events)
            .unwrap();
        // Rebuild centered z and recompute the gradient at the fitted β.
        let means = &model.means;
        let z: Vec<Vec<f64>> = (0..features.nrows())
            .map(|i| (0..2).map(|j| features[[i, j]] - means[j]).collect())
            .collect();
        let mut order: Vec<usize> = (0..events.len()).collect();
        order.sort_by(|&a, &b| events[a].time.partial_cmp(&events[b].time).unwrap());
        let (grad, _, _) = compute_derivatives(&z, &events, &order, model.coefficients(), 2);
        for g in grad {
            assert!(g.abs() < 1e-6, "score component {g} should vanish");
        }
    }

    /// On data where a single covariate orders event times, the coefficient
    /// is strongly positive and the in-sample C-index is 1. A small `l2`
    /// keeps `β` finite — perfect separation otherwise drives the Cox MLE to
    /// infinity (the classic monotone-likelihood degeneracy), which is a
    /// genuine property of the model, not a bug.
    #[test]
    fn recovers_direction_and_ranks_perfectly() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]];
        let events: Vec<SurvivalEvent> = (0..8)
            .map(|i| SurvivalEvent {
                time: (8 - i) as f64, // higher covariate → earlier event
                event: true,
            })
            .collect();
        let model = CoxPH::new().with_l2(0.1).fit(&x, &events).unwrap();
        assert!(model.coefficients()[0] > 0.5);
        let preds = model.predict(&x);
        assert!((concordance_index(&preds, &events) - 1.0).abs() < 1e-9);
    }

    /// Higher risk score → lower survival at every time: the predicted curves
    /// must respect the proportional-hazards ordering.
    #[test]
    fn survival_curves_ordered_by_risk() {
        let (features, events) = golden_data();
        let model = CoxPH::new().fit(&features, &events).unwrap();
        let preds = model.predict(&features);
        // Find the highest- and lowest-risk subjects.
        let hi = preds
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.risk_score.partial_cmp(&b.1.risk_score).unwrap())
            .unwrap()
            .0;
        let lo = preds
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.risk_score.partial_cmp(&b.1.risk_score).unwrap())
            .unwrap()
            .0;
        for k in 0..preds[hi].survival.len() {
            assert!(
                preds[hi].survival[k] <= preds[lo].survival[k] + 1e-12,
                "higher risk must not have higher survival at time index {k}"
            );
        }
    }

    #[test]
    fn rejects_all_censored_and_shape_mismatch() {
        let x = array![[0.0], [1.0], [2.0]];
        let all_censored: Vec<SurvivalEvent> = (0..3)
            .map(|i| SurvivalEvent {
                time: (i + 1) as f64,
                event: false,
            })
            .collect();
        assert!(CoxPH::new().fit(&x, &all_censored).is_err());

        let wrong_len = vec![SurvivalEvent {
            time: 1.0,
            event: true,
        }];
        assert!(CoxPH::new().fit(&x, &wrong_len).is_err());
    }

    /// A positive `l2` penalty shrinks coefficients toward zero relative to
    /// the unpenalized fit.
    #[test]
    fn l2_penalty_shrinks_coefficients() {
        let (features, events) = golden_data();
        let unpen = CoxPH::new().fit(&features, &events).unwrap();
        let pen = CoxPH::new().with_l2(5.0).fit(&features, &events).unwrap();
        let norm = |c: &[f64]| c.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            norm(pen.coefficients()) < norm(unpen.coefficients()),
            "l2 should shrink: {:?} vs {:?}",
            pen.coefficients(),
            unpen.coefficients()
        );
    }
}
