//! Kriging-ML Hybrid (regression-kriging): a base `Learner` plus ordinary
//! kriging of its residuals.
//!
//! Trains an arbitrary base learner (e.g. `XGBoost`, `RandomForest`) on the
//! features, then fits a variogram to the base model's residuals as a
//! function of spatial distance and krige-interpolates them at prediction
//! time. The final prediction is `base_model(x) + kriged_residual(coords)` --
//! the classic "regression-kriging" / "kriging with external drift" approach
//! from geostatistics, letting the base model capture the feature-driven
//! signal and kriging mop up spatially structured residual error the base
//! model can't see.

use crate::learner::{Learner, LearnerProperties, TrainedModel};
use crate::prediction::Prediction;
use crate::task::{RegressionTask, Task};
use crate::{Result, SmeltError};
use ndarray::Array2;

/// Semivariogram model family used to describe how residual dissimilarity
/// grows with spatial distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariogramModel {
    /// Reaches the sill exactly at `h = range` (linear-then-flat shape).
    Spherical,
    /// Approaches the sill asymptotically (`1 - e^{-h/range}`).
    Exponential,
    /// Approaches the sill asymptotically with a smooth (parabolic) origin
    /// (`1 - e^{-(h/range)^2}`), giving very smooth short-range behavior.
    Gaussian,
    /// Matérn with smoothness ν = 3/2:
    /// `1 - (1 + √3·h/range)·e^{-√3·h/range}` (sklearn/GP length-scale
    /// convention). Between `Exponential` and `Gaussian` in short-range
    /// smoothness: once-differentiable realizations.
    Matern32,
    /// Matérn with smoothness ν = 5/2:
    /// `1 - (1 + √5·h/range + 5h²/(3·range²))·e^{-√5·h/range}`. Smoother
    /// than `Matern32` (twice-differentiable realizations), still rougher
    /// than `Gaussian` (ν → ∞).
    ///
    /// Only the closed-form smoothness values are provided (ν = 1/2 is
    /// exactly `Exponential`, ν → ∞ is `Gaussian`): continuous ν requires
    /// the modified Bessel function K_ν, out of scope for this crate's
    /// hand-rolled numerics (use an external geostatistics crate for that).
    Matern52,
}

impl VariogramModel {
    /// Semivariance `γ(h)` for the given lag distance `h` and parameters.
    ///
    /// By convention `γ(0) = 0` (a point is perfectly correlated with
    /// itself); the nugget is the limit of `γ(h)` as `h → 0+`, i.e. the
    /// discontinuity at the origin caused by micro-scale variability or
    /// measurement error.
    fn value(&self, h: f64, nugget: f64, sill: f64, range: f64) -> f64 {
        if h <= 0.0 {
            return 0.0;
        }
        let partial = (sill - nugget).max(0.0);
        match self {
            VariogramModel::Spherical => {
                if h >= range {
                    sill
                } else {
                    let r = h / range;
                    nugget + partial * (1.5 * r - 0.5 * r.powi(3))
                }
            }
            VariogramModel::Exponential => nugget + partial * (1.0 - (-h / range).exp()),
            VariogramModel::Gaussian => nugget + partial * (1.0 - (-(h / range).powi(2)).exp()),
            VariogramModel::Matern32 => {
                let s = 3.0_f64.sqrt() * h / range;
                nugget + partial * (1.0 - (1.0 + s) * (-s).exp())
            }
            VariogramModel::Matern52 => {
                let s = 5.0_f64.sqrt() * h / range;
                nugget + partial * (1.0 - (1.0 + s + s * s / 3.0) * (-s).exp())
            }
        }
    }
}

/// Fitted variogram parameters: nugget (discontinuity at the origin), sill
/// (plateau semivariance at long range), and range (lag distance at which
/// the plateau is effectively reached).
#[derive(Debug, Clone)]
pub struct VariogramFit {
    /// Semivariance discontinuity at the origin (micro-scale/measurement noise).
    pub nugget: f64,
    /// Plateau semivariance at long range (total residual variance).
    pub sill: f64,
    /// Lag distance at which the plateau is effectively reached.
    pub range: f64,
    /// The model family these parameters belong to.
    pub model: VariogramModel,
}

/// Euclidean distance between two coordinates.
#[inline]
fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// Largest pairwise distance among `coords` (0 if fewer than 2 points).
fn max_pairwise_distance(coords: &[(f64, f64)]) -> f64 {
    let n = coords.len();
    let mut max_d = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            max_d = max_d.max(dist(coords[i], coords[j]));
        }
    }
    max_d
}

/// Binned empirical semivariogram: mean semivariance and pair-count per lag
/// bin, over `[0, max_lag]`, dropping empty bins. Returns `(lag_centers,
/// semivariances, weights)`.
fn empirical_variogram(
    coords: &[(f64, f64)],
    residuals: &[f64],
    n_lags: usize,
    max_lag: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = coords.len();
    let bin_width = (max_lag / n_lags as f64).max(1e-10);

    let mut sum_sv = vec![0.0; n_lags];
    let mut count = vec![0usize; n_lags];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist(coords[i], coords[j]);
            if d > max_lag {
                continue;
            }
            let bin = ((d / bin_width) as usize).min(n_lags - 1);
            sum_sv[bin] += 0.5 * (residuals[i] - residuals[j]).powi(2);
            count[bin] += 1;
        }
    }

    let mut lags = Vec::new();
    let mut semivariances = Vec::new();
    let mut weights = Vec::new();
    for b in 0..n_lags {
        if count[b] == 0 {
            continue;
        }
        lags.push((b as f64 + 0.5) * bin_width);
        semivariances.push(sum_sv[b] / count[b] as f64);
        weights.push(count[b] as f64);
    }
    (lags, semivariances, weights)
}

/// Fit `(nugget, sill, range)` by minimizing Cressie's (1985) weighted
/// least-squares objective against the empirical variogram:
/// `Σ_j N_j · (γ̂_j − γ(h_j; θ))² / γ(h_j; θ)²` -- pair counts over the
/// *squared model value*, so each bin contributes by its relative (not
/// absolute) misfit. This is the standard WLS family of the geostatistics
/// literature (Cressie 1985; gstat's `fit.method = 2` -- note gstat's own
/// *default* is method 7, which weights by `N_j / h_j²` instead): plain
/// `N_j`-weighted SSE lets the large-semivariance long-range bins dominate
/// and fits the short-range structure -- the part kriging actually uses --
/// worst.
///
/// There is no nonlinear least-squares solver in this crate (deliberately
/// -- see `src/sparse.rs` for the same "hand-roll the small numeric routine
/// instead of adding a dependency" precedent). A grid search evaluates the
/// exact WLS objective per candidate (no IRLS needed), in two stages: a
/// coarse pass over the full bounds (≤3000 combinations) and a finer local
/// pass (729) inside the ±1-coarse-step box around the winner. One-time
/// cost per `train_regress_geo` call.
fn fit_variogram(
    lags: &[f64],
    semivariances: &[f64],
    weights: &[f64],
    model: VariogramModel,
    max_lag: f64,
) -> VariogramFit {
    let max_sv = semivariances.iter().cloned().fold(0.0, f64::max).max(1e-10);
    let max_lag = max_lag.max(1e-10);

    if lags.is_empty() {
        // No pairs within range (e.g. a single distinct location): fall back
        // to a flat, no-op variogram rather than panicking on empty grids.
        return VariogramFit {
            nugget: 0.0,
            sill: max_sv,
            range: max_lag,
            model,
        };
    }

    // Parameterized as (nugget, partial = sill - nugget, range) so the
    // `nugget <= sill` invariant (which `ordinary_kriging` relies on) holds
    // by construction in both stages.
    let objective = |nugget: f64, partial: f64, range: f64| -> f64 {
        let sill = nugget + partial;
        let mut score = 0.0;
        for k in 0..lags.len() {
            // Clamp only to dodge a literal 0/0 for degenerate all-zero
            // candidates; genuine near-zero model values SHOULD be heavily
            // penalized when the empirical bin is positive.
            let pred = model.value(lags[k], nugget, sill, range).max(1e-12);
            let diff = pred - semivariances[k];
            score += weights[k] * diff * diff / (pred * pred);
        }
        score
    };

    const N_NUGGET: usize = 10;
    const N_PARTIAL: usize = 15;
    const N_RANGE: usize = 20;
    let nugget_step = max_sv / (N_NUGGET - 1) as f64;
    let partial_step = 1.5 * max_sv / (N_PARTIAL - 1) as f64;
    let range_step = max_lag / N_RANGE as f64;

    let mut best = (0.0, max_sv, max_lag);
    let mut best_score = f64::INFINITY;
    for ni in 0..N_NUGGET {
        let nugget = ni as f64 * nugget_step;
        for pi in 0..N_PARTIAL {
            let partial = pi as f64 * partial_step;
            for ri in 0..N_RANGE {
                let range = (ri as f64 + 1.0) * range_step;
                let score = objective(nugget, partial, range);
                if score < best_score {
                    best_score = score;
                    best = (nugget, partial, range);
                }
            }
        }
    }

    // Local refinement: same objective on a finer grid spanning one coarse
    // step to each side of the coarse winner.
    const N_FINE: usize = 9;
    let (coarse_nugget, coarse_partial, coarse_range) = best;
    for ni in 0..N_FINE {
        let frac_n = 2.0 * ni as f64 / (N_FINE - 1) as f64 - 1.0;
        let nugget = (coarse_nugget + frac_n * nugget_step).max(0.0);
        for pi in 0..N_FINE {
            let frac_p = 2.0 * pi as f64 / (N_FINE - 1) as f64 - 1.0;
            let partial = (coarse_partial + frac_p * partial_step).max(0.0);
            for ri in 0..N_FINE {
                let frac_r = 2.0 * ri as f64 / (N_FINE - 1) as f64 - 1.0;
                let range = (coarse_range + frac_r * range_step).max(range_step * 0.1);
                let score = objective(nugget, partial, range);
                if score < best_score {
                    best_score = score;
                    best = (nugget, partial, range);
                }
            }
        }
    }

    VariogramFit {
        nugget: best.0,
        sill: best.0 + best.1,
        range: best.2,
        model,
    }
}

/// Solves `matrix * x = rhs` via Gaussian elimination with partial
/// pivoting. `matrix` is square, `rhs.len() == matrix.len()`.
fn gaussian_elimination_solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    // Scaled partial pivoting (Golub & Van Loan §3.4.10): each row's pivot
    // candidate is judged relative to *that row's own* largest entry, not a
    // single absolute or matrix-global threshold. This matters because the
    // ordinary-kriging system built by `ordinary_kriging` is a bordered
    // system: a k×k semivariogram sub-block (whose scale is the fitted
    // sill -- tiny for small-magnitude targets, e.g. geochemistry
    // concentrations ~1e-5 giving sill ~1e-10) bordered by an unrelated,
    // always-O(1) Lagrange-multiplier row/column of ones. A single
    // matrix-global-max-relative threshold judges the semivariogram block's
    // pivots against that unrelated O(1) border scale and incorrectly
    // calls a perfectly well-conditioned small-scale system singular; a raw
    // fixed absolute threshold has the same problem in the other direction
    // for large-scale targets. Scaling each row by its own max entry (fixed
    // at the start, swapped alongside the row -- the standard textbook
    // algorithm) makes the check scale-invariant per row instead.
    let mut row_scale: Vec<f64> = matrix
        .iter()
        .map(|row| row.iter().fold(0.0f64, |acc, &v| acc.max(v.abs())).max(f64::MIN_POSITIVE))
        .collect();

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_score = matrix[col][col].abs() / row_scale[col];
        for r in (col + 1)..n {
            let score = matrix[r][col].abs() / row_scale[r];
            if score > pivot_score {
                pivot_score = score;
                pivot_row = r;
            }
        }
        if pivot_score < 1e-12 {
            return Err(SmeltError::NumericalError(
                "singular kriging system (duplicate or degenerate coordinates?)".into(),
            ));
        }
        matrix.swap(col, pivot_row);
        rhs.swap(col, pivot_row);
        row_scale.swap(col, pivot_row);

        for r in (col + 1)..n {
            let factor = matrix[r][col] / matrix[col][col];
            if factor == 0.0 {
                continue;
            }
            for c in col..n {
                matrix[r][c] -= factor * matrix[col][c];
            }
            rhs[r] -= factor * rhs[col];
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= matrix[i][j] * x[j];
        }
        x[i] = sum / matrix[i][i];
    }
    Ok(x)
}

/// Training locations closer to each other than this are treated as
/// coincident (e.g. repeat samples from the same borehole): kriging them as
/// distinct points produces identical matrix rows and a singular system.
const COINCIDENT_COORD_EPS: f64 = 1e-8;

/// Local ordinary kriging of the residual at one query point, given its
/// distances to the `k` nearest training points (already sorted/truncated).
///
/// Two failure modes that used to make this return `Err` on every query are
/// handled before building the system: a degenerate (near-zero-variance)
/// fitted variogram, and coincident/near-coincident training coordinates
/// among the neighbors.
fn ordinary_kriging(
    neighbors: &[(usize, f64)],
    coords: &[(f64, f64)],
    residuals: &[f64],
    fit: &VariogramFit,
) -> Result<f64> {
    if neighbors.is_empty() {
        return Ok(0.0);
    }

    // Degenerate variogram: the base learner's residuals have ~zero variance
    // relative to their own scale (e.g. it already interpolates the training
    // data), so γ(h) ≈ 0 for every lag — there's no spatially-structured
    // signal to krige, and the system built from an all-zero variogram is
    // singular. `nugget <= sill` by construction (see `fit_variogram`), so
    // checking the sill suffices. The threshold is scaled by the residuals'
    // own variance rather than a fixed absolute `1e-9`: with small-magnitude
    // targets (e.g. geochemistry concentrations ~1e-5), a real
    // spatially-structured sill can itself be ~1e-10 -- a fixed cutoff
    // disabled the kriging correction unconditionally for exactly the
    // datasets where it's most needed, silently degrading to base-model-only
    // predictions regardless of how good the base model actually is.
    let residual_variance = {
        let n = residuals.len().max(1) as f64;
        let mean = residuals.iter().sum::<f64>() / n;
        residuals.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n
    };
    let degenerate_sill_threshold = (residual_variance * 1e-9).max(f64::MIN_POSITIVE);
    if fit.sill < degenerate_sill_threshold {
        return Ok(0.0);
    }

    // Merge neighbors at (near-)identical training locations into groups
    // before solving, representing each group by its mean residual. This is
    // O(k²) in the neighborhood size (bounded by `n_neighbors`, small).
    let mut groups: Vec<Vec<usize>> = Vec::new(); // indices into `neighbors`
    'outer: for (ni, &(idx, _)) in neighbors.iter().enumerate() {
        for g in groups.iter_mut() {
            if dist(coords[neighbors[g[0]].0], coords[idx]) < COINCIDENT_COORD_EPS {
                g.push(ni);
                continue 'outer;
            }
        }
        groups.push(vec![ni]);
    }

    let group_residual = |g: &[usize]| -> f64 {
        g.iter().map(|&ni| residuals[neighbors[ni].0]).sum::<f64>() / g.len() as f64
    };

    let k = groups.len();
    if k == 1 {
        return Ok(group_residual(&groups[0]));
    }

    let group_coord = |g: &[usize]| coords[neighbors[g[0]].0];
    let group_query_dist = |g: &[usize]| neighbors[g[0]].1;

    let n = k + 1;
    let mut matrix = vec![vec![0.0; n]; n];
    let mut rhs = vec![0.0; n];

    for a in 0..k {
        for b in 0..k {
            let h = dist(group_coord(&groups[a]), group_coord(&groups[b]));
            matrix[a][b] = fit.model.value(h, fit.nugget, fit.sill, fit.range);
        }
        matrix[a][k] = 1.0;
        matrix[k][a] = 1.0;
        rhs[a] = fit.model.value(group_query_dist(&groups[a]), fit.nugget, fit.sill, fit.range);
    }
    rhs[k] = 1.0;

    let weights = gaussian_elimination_solve(matrix, rhs)?;
    Ok((0..k).map(|a| weights[a] * group_residual(&groups[a])).sum())
}

/// Regression-kriging: a base `Learner` plus ordinary kriging of its
/// residuals, combined as `base_model(x) + kriged_residual(coords)`.
///
/// # Examples
///
/// ```
/// use smelt_ml::prelude::*;
/// use ndarray::array;
///
/// let features = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
/// let target = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
/// let coords = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0), (5.0, 0.0)];
/// let task = RegressionTask::new("kr", features, target).unwrap();
///
/// let mut kh = KrigingHybrid::new(|| Box::new(LinearRegression::new()), coords)
///     .with_n_neighbors(3);
/// let model = kh.train_regress_geo(&task).unwrap();
/// ```
pub struct KrigingHybrid {
    factory: Box<dyn Fn() -> Box<dyn Learner> + Send + Sync>,
    coords: Vec<(f64, f64)>,
    model: VariogramModel,
    n_lags: usize,
    n_neighbors: usize,
}

impl KrigingHybrid {
    /// Creates a `KrigingHybrid` from a base-learner factory and training
    /// coordinates, defaulting to a spherical variogram, 15 lag bins, and a
    /// local kriging neighborhood of 20 points.
    pub fn new(
        factory: impl Fn() -> Box<dyn Learner> + Send + Sync + 'static,
        coords: Vec<(f64, f64)>,
    ) -> Self {
        Self {
            factory: Box::new(factory),
            coords,
            model: VariogramModel::Spherical,
            n_lags: 15,
            n_neighbors: 20,
        }
    }

    /// Sets the semivariogram model family.
    pub fn with_variogram_model(mut self, model: VariogramModel) -> Self {
        self.model = model;
        self
    }

    /// Sets the number of lag bins used to build the empirical variogram.
    pub fn with_n_lags(mut self, n: usize) -> Self {
        self.n_lags = n.max(1);
        self
    }

    /// Sets the number of nearest training points used in the local
    /// ordinary-kriging system at prediction time.
    pub fn with_n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n.max(1);
        self
    }

    /// Trains the base learner and fits the residual variogram, returning a
    /// concrete `TrainedKrigingHybrid` (use this instead of `Learner::train_regress`
    /// when you need [`TrainedKrigingHybrid::predict_spatial`]).
    pub fn train_regress_geo(&mut self, task: &RegressionTask) -> Result<TrainedKrigingHybrid> {
        // Guard here (not in `Learner::train_regress`, which just boxes this)
        // so BOTH public entry points reject weighted tasks.
        crate::validate::check_no_weights(task.weights(), "KrigingHybrid")?;
        let n_samples = task.n_samples();
        if self.coords.len() != n_samples {
            return Err(SmeltError::DimensionMismatch {
                expected: n_samples,
                got: self.coords.len(),
            });
        }
        crate::validate::check_coords_finite(&self.coords)?;

        let mut base_learner = (self.factory)();
        let base_model = base_learner.train_regress(task)?;
        let base_pred = base_model.predict(task.features())?;
        let base_vals = match &base_pred {
            Prediction::Regression { predicted, .. } => predicted.clone(),
            _ => return Err(SmeltError::IncompatiblePrediction("expected regression prediction".into())),
        };

        let target = task.target();
        let residuals: Vec<f64> = target.iter().zip(&base_vals).map(|(y, p)| y - p).collect();

        let max_lag = (2.0 / 3.0) * max_pairwise_distance(&self.coords);
        let (lags, semivariances, weights) =
            empirical_variogram(&self.coords, &residuals, self.n_lags, max_lag);
        let fit = fit_variogram(&lags, &semivariances, &weights, self.model, max_lag);

        Ok(TrainedKrigingHybrid {
            base_model,
            coords: self.coords.clone(),
            residuals,
            fit,
            n_neighbors: self.n_neighbors,
        })
    }
}

impl Learner for KrigingHybrid {
    fn id(&self) -> &str {
        "kriging_hybrid"
    }

    fn properties(&self) -> LearnerProperties {
        LearnerProperties::regressor().with_feature_importance()
    }

    fn train_regress(&mut self, task: &RegressionTask) -> Result<Box<dyn TrainedModel>> {
        Ok(Box::new(self.train_regress_geo(task)?))
    }
}

/// Trained Kriging-ML Hybrid model: a base trained model plus the fitted
/// residual variogram needed for spatial kriging correction.
pub struct TrainedKrigingHybrid {
    base_model: Box<dyn TrainedModel>,
    coords: Vec<(f64, f64)>,
    residuals: Vec<f64>,
    fit: VariogramFit,
    n_neighbors: usize,
}

impl TrainedKrigingHybrid {
    /// The fitted residual variogram (nugget, sill, range, model family).
    pub fn variogram_fit(&self) -> &VariogramFit {
        &self.fit
    }

    /// Predicts `base_model(features) + kriged_residual(new_coords)`.
    ///
    /// For each query point, the residual is estimated by local ordinary
    /// kriging over its `n_neighbors` nearest training points using the
    /// fitted variogram. Passing the training coordinates back with the
    /// training features reproduces the training residuals exactly (kriging
    /// is an exact interpolator at data locations).
    pub fn predict_spatial(
        &self,
        features: &Array2<f64>,
        new_coords: &[(f64, f64)],
    ) -> Result<Prediction> {
        let n_samples = features.nrows();
        if new_coords.len() != n_samples {
            return Err(SmeltError::DimensionMismatch {
                expected: n_samples,
                got: new_coords.len(),
            });
        }
        crate::validate::check_coords_finite(new_coords)?;

        let base_pred = self.base_model.predict(features)?;
        let base_vals = match &base_pred {
            Prediction::Regression { predicted, .. } => predicted.clone(),
            _ => return Err(SmeltError::IncompatiblePrediction("expected regression prediction".into())),
        };

        let k = self.n_neighbors.min(self.coords.len());
        let mut predicted = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut neighbor_dists: Vec<(usize, f64)> = self
                .coords
                .iter()
                .enumerate()
                .map(|(j, &c)| (j, dist(new_coords[i], c)))
                .collect();
            neighbor_dists
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            neighbor_dists.truncate(k);

            let kriged_residual =
                ordinary_kriging(&neighbor_dists, &self.coords, &self.residuals, &self.fit)?;
            predicted.push(base_vals[i] + kriged_residual);
        }

        Ok(Prediction::regression(predicted))
    }
}

impl TrainedModel for TrainedKrigingHybrid {
    /// Base-model-only prediction.
    ///
    /// The `TrainedModel` trait has no notion of spatial coordinates, so
    /// this method cannot apply the kriging correction (which needs new
    /// coordinates). Use [`TrainedKrigingHybrid::predict_spatial`] to get
    /// the spatially-corrected prediction -- matching how
    /// [`crate::learner::TrainedGeoXGBoost::predict`] is also base/global-only
    /// for the same reason.
    fn predict(&self, features: &Array2<f64>) -> Result<Prediction> {
        self.base_model.predict(features)
    }

    fn feature_importance(&self) -> Option<Vec<(String, f64)>> {
        self.base_model.feature_importance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::LinearRegression;
    use crate::learner::tree::decision_tree::DecisionTree;
    use ndarray::Array2;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Regression test (4th audit, HIGH-3): a single NaN coordinate used to
    /// be accepted silently — it entered every query's kriging system (no
    /// comparison can filter a NaN distance) and `predict_spatial` returned
    /// `Ok` with ALL predictions NaN. Both train and predict must reject
    /// non-finite coordinates with a clean error naming the index.
    #[test]
    fn non_finite_coordinates_are_rejected_at_train_and_predict() {
        let n = 20;
        let features = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let target: Vec<f64> = (0..n).map(|i| i as f64 * 2.0).collect();
        let task = RegressionTask::new("nan-coords", features.clone(), target).unwrap();

        let mut coords: Vec<(f64, f64)> = (0..n).map(|i| (i as f64, 0.0)).collect();
        coords[7] = (f64::NAN, 0.0);
        let mut kh = KrigingHybrid::new(|| Box::new(LinearRegression::new()), coords);
        let Err(err) = kh.train_regress_geo(&task) else {
            panic!("NaN training coordinate must be rejected")
        };
        assert!(
            err.to_string().contains("index 7"),
            "error should name the offending coordinate: {err}"
        );

        // Clean train, NaN in the *query* coordinates.
        let good_coords: Vec<(f64, f64)> = (0..n).map(|i| (i as f64, 0.0)).collect();
        let mut kh = KrigingHybrid::new(|| Box::new(LinearRegression::new()), good_coords);
        let trained = kh.train_regress_geo(&task).unwrap();
        let mut query = vec![(1.5, 0.0), (2.5, 0.0)];
        query[1] = (0.0, f64::INFINITY);
        let Err(err) =
            trained.predict_spatial(&features.slice(ndarray::s![0..2, ..]).to_owned(), &query)
        else {
            panic!("NaN query coordinate must be rejected")
        };
        assert!(
            err.to_string().contains("index 1"),
            "error should name the offending query coordinate: {err}"
        );
    }

    #[test]
    fn gaussian_elimination_solves_a_known_system() {
        // [2 1; 1 3] x = [5; 10] -> x = [1, 3]
        let matrix = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let rhs = vec![5.0, 10.0];
        let x = gaussian_elimination_solve(matrix, rhs).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-9);
        assert!((x[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn gaussian_elimination_rejects_singular_system() {
        let matrix = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let rhs = vec![1.0, 2.0];
        assert!(gaussian_elimination_solve(matrix, rhs).is_err());
    }

    #[test]
    fn variogram_value_boundary_conditions() {
        let nugget = 1.0;
        let sill = 5.0;
        let range = 10.0;

        for model in [
            VariogramModel::Spherical,
            VariogramModel::Exponential,
            VariogramModel::Gaussian,
        ] {
            assert_eq!(model.value(0.0, nugget, sill, range), 0.0);
            // As h -> 0+, gamma(h) -> nugget (the discontinuity at the origin).
            let near_zero = model.value(1e-6, nugget, sill, range);
            assert!(
                (near_zero - nugget).abs() < 0.05,
                "{model:?}: expected near-zero lag to approach nugget, got {near_zero}"
            );
        }

        // Spherical reaches the sill exactly at h = range.
        let sph = VariogramModel::Spherical.value(range, nugget, sill, range);
        assert!((sph - sill).abs() < 1e-9);

        // Exponential/Gaussian approach the sill asymptotically: at h = range
        // they should be close to, but strictly below, the sill.
        for model in [VariogramModel::Exponential, VariogramModel::Gaussian] {
            let v = model.value(range, nugget, sill, range);
            assert!(v < sill);
            assert!(v > nugget);
        }
    }

    #[test]
    fn empirical_variogram_bins_known_pairs() {
        // 3 colinear points 1 unit apart: pairwise distances are 1, 1, 2.
        let coords = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)];
        let residuals = vec![0.0, 1.0, 0.0];
        let max_lag = 2.0;
        let (lags, semivariances, weights) = empirical_variogram(&coords, &residuals, 2, max_lag);
        // bin 0 covers [0, 1.0): the two h=1 pairs fall in bin 1 ([1.0, 2.0)),
        // so with 2 bins of width 1.0, only 1 non-empty bin is expected
        // (h=1 pairs) plus possibly the h=2 pair depending on truncation.
        assert!(!lags.is_empty());
        assert_eq!(lags.len(), semivariances.len());
        assert_eq!(lags.len(), weights.len());
        // Every reported semivariance is non-negative.
        assert!(semivariances.iter().all(|&s| s >= 0.0));
    }

    /// Matérn closed forms against independently computed golden values
    /// (nugget=0, sill=1): γ(h) = 1 − (1+s)e^{-s} with s = √3·h/r for
    /// ν=3/2, and 1 − (1+s+s²/3)e^{-s} with s = √5·h/r for ν=5/2.
    #[test]
    fn matern_closed_forms_match_golden_values() {
        let cases = [
            (VariogramModel::Matern32, 1.0, 0.5166422754034923),
            (VariogramModel::Matern32, 0.5, 0.21511234604254936),
            (VariogramModel::Matern52, 1.0, 0.4760058911681797),
            (VariogramModel::Matern52, 0.5, 0.17135085758187452),
        ];
        for (model, h, expected) in cases {
            let got = model.value(h, 0.0, 1.0, 1.0);
            assert!(
                (got - expected).abs() < 1e-12,
                "{model:?} at h={h}: got {got}, expected {expected}"
            );
        }
    }

    /// Structural properties of the Matérn variants: γ(0)=0, monotone
    /// nondecreasing, sill-approaching, and the smoothness ordering near
    /// the origin (Matérn is O(h²) there -- with ν=5/2 flattest -- while
    /// Exponential rises O(h)).
    #[test]
    fn matern_variants_are_smooth_monotone_and_sill_bounded() {
        for model in [VariogramModel::Matern32, VariogramModel::Matern52] {
            assert_eq!(model.value(0.0, 0.3, 1.0, 1.0), 0.0, "{model:?}: γ(0) must be 0");
            let mut prev = 0.0;
            for i in 1..=100 {
                let h = i as f64 * 0.1;
                let v = model.value(h, 0.0, 1.0, 1.0);
                assert!(v >= prev - 1e-12, "{model:?} must be nondecreasing");
                assert!(v <= 1.0 + 1e-12, "{model:?} must stay below the sill");
                prev = v;
            }
            assert!(prev > 0.99, "{model:?} must approach the sill at long range");
        }
        let h = 0.2;
        let exp = VariogramModel::Exponential.value(h, 0.0, 1.0, 1.0);
        let m32 = VariogramModel::Matern32.value(h, 0.0, 1.0, 1.0);
        let m52 = VariogramModel::Matern52.value(h, 0.0, 1.0, 1.0);
        assert!(
            m52 < m32 && m32 < exp,
            "short-lag smoothness ordering must hold: m52 ({m52}) < m32 ({m32}) < exp ({exp})"
        );
    }

    /// The two-stage WLS fit must recover known parameters from an exact
    /// model-generated empirical variogram to well within one coarse grid
    /// step -- the local refinement pass is what buys the tight tolerance.
    #[test]
    fn wls_fit_recovers_known_variogram_parameters() {
        let true_nugget = 0.2;
        let true_sill = 1.0;
        let true_range = 3.0;
        let max_lag = 10.0;
        for model in [
            VariogramModel::Spherical,
            VariogramModel::Exponential,
            VariogramModel::Matern32,
            VariogramModel::Matern52,
        ] {
            let lags: Vec<f64> = (1..=20).map(|i| i as f64 * 0.5).collect();
            let semivariances: Vec<f64> = lags
                .iter()
                .map(|&h| model.value(h, true_nugget, true_sill, true_range))
                .collect();
            let weights = vec![30.0; lags.len()];
            let fit = fit_variogram(&lags, &semivariances, &weights, model, max_lag);

            // Coarse steps: nugget ~0.11, partial ~0.107, range 0.5. The
            // refinement pass must land clearly inside one coarse cell.
            assert!(
                (fit.nugget - true_nugget).abs() < 0.08,
                "{model:?}: nugget {} vs true {true_nugget}",
                fit.nugget
            );
            assert!(
                (fit.sill - true_sill).abs() < 0.08,
                "{model:?}: sill {} vs true {true_sill}",
                fit.sill
            );
            assert!(
                (fit.range - true_range).abs() < 0.5,
                "{model:?}: range {} vs true {true_range}",
                fit.range
            );
        }
    }

    #[test]
    fn fit_variogram_beats_flat_baseline_on_autocorrelated_residuals() {
        // Smooth spatial trend -> nearby points have similar residuals,
        // distant points don't: a real (non-flat) variogram should fit
        // better than a degenerate flat one (sill == nugget everywhere).
        let side = 10;
        let mut coords = Vec::new();
        let mut residuals = Vec::new();
        for r in 0..side {
            for c in 0..side {
                let x = c as f64;
                let y = r as f64;
                coords.push((x, y));
                residuals.push((x * 0.3).sin() + (y * 0.3).cos());
            }
        }

        let max_lag = (2.0 / 3.0) * max_pairwise_distance(&coords);
        let (lags, semivariances, weights) = empirical_variogram(&coords, &residuals, 15, max_lag);
        let fitted = fit_variogram(&lags, &semivariances, &weights, VariogramModel::Spherical, max_lag);

        let fitted_sse: f64 = (0..lags.len())
            .map(|k| {
                let pred = fitted.model.value(lags[k], fitted.nugget, fitted.sill, fitted.range);
                weights[k] * (pred - semivariances[k]).powi(2)
            })
            .sum();

        let mean_sv: f64 = semivariances.iter().sum::<f64>() / semivariances.len() as f64;
        let flat_sse: f64 = (0..lags.len())
            .map(|k| weights[k] * (mean_sv - semivariances[k]).powi(2))
            .sum();

        assert!(
            fitted_sse <= flat_sse + 1e-9,
            "fitted variogram (SSE {fitted_sse}) should fit at least as well as a flat baseline (SSE {flat_sse})"
        );
        assert!(fitted.sill >= fitted.nugget);
        assert!(fitted.range > 0.0);
    }

    #[test]
    fn train_regress_geo_rejects_coord_mismatch() {
        let features = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let task = RegressionTask::new("t", features, vec![0.0, 1.0, 2.0]).unwrap();
        let coords = vec![(0.0, 0.0), (1.0, 0.0)]; // wrong length
        let mut kh = KrigingHybrid::new(|| Box::new(LinearRegression::new()), coords);
        assert!(kh.train_regress_geo(&task).is_err());
    }

    #[test]
    fn predict_is_base_model_only() {
        let features = Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let target = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let coords: Vec<(f64, f64)> = (0..6).map(|i| (i as f64, 0.0)).collect();
        let task = RegressionTask::new("t", features.clone(), target).unwrap();

        let mut kh = KrigingHybrid::new(|| Box::new(LinearRegression::new()), coords);
        let model = kh.train_regress_geo(&task).unwrap();

        let via_trait = TrainedModel::predict(&model, &features).unwrap();
        let base_only = model.base_model.predict(&features).unwrap();
        let (Prediction::Regression { predicted: a, .. }, Prediction::Regression { predicted: b, .. }) =
            (via_trait, base_only)
        else {
            panic!("expected regression");
        };
        assert_eq!(a, b);
    }

    #[test]
    fn predict_spatial_reproduces_training_residuals_at_training_locations() {
        let side = 6;
        let mut feats = Vec::new();
        let mut target = Vec::new();
        let mut coords = Vec::new();
        for r in 0..side {
            for c in 0..side {
                let x = c as f64;
                let y = r as f64;
                coords.push((x, y));
                feats.push(x + y);
                // Base model (linear in x+y) can't capture this extra
                // spatially-smooth term -> spatially structured residuals.
                target.push((x + y) + (x * 0.4).sin() * 2.0);
            }
        }
        let features = Array2::from_shape_vec((side * side, 1), feats).unwrap();
        let task = RegressionTask::new("t", features.clone(), target).unwrap();

        let mut kh = KrigingHybrid::new(|| Box::new(LinearRegression::new()), coords.clone())
            .with_n_neighbors(8);
        let model = kh.train_regress_geo(&task).unwrap();

        let pred = model.predict_spatial(&features, &coords).unwrap();
        let Prediction::Regression { predicted, .. } = pred else {
            panic!("expected regression");
        };
        for (i, &p) in predicted.iter().enumerate() {
            let expected = task.target()[i];
            assert!(
                (p - expected).abs() < 1e-6,
                "kriging should exactly reproduce training targets at training locations (point {i}: got {p}, expected {expected})"
            );
        }
    }

    #[test]
    fn kriging_correction_reduces_error_on_spatially_structured_residuals() {
        // Base learner (decision tree, shallow) can't fully capture a smooth
        // spatial trend added on top of the feature signal; the kriging
        // correction should reduce held-out error versus the base model alone.
        let side = 10;
        let mut rng = StdRng::seed_from_u64(11);
        let mut feats = Vec::new();
        let mut target = Vec::new();
        let mut coords = Vec::new();
        for r in 0..side {
            for c in 0..side {
                let x = c as f64;
                let y = r as f64;
                coords.push((x, y));
                feats.push(x - y); // weak feature signal
                let spatial_trend = (x * 0.5).sin() * 3.0 + (y * 0.5).cos() * 3.0;
                target.push((x - y) * 0.1 + spatial_trend + rng.random::<f64>() * 0.05);
            }
        }
        let features = Array2::from_shape_vec((side * side, 1), feats).unwrap();
        let train_task = RegressionTask::new("t", features.clone(), target.clone()).unwrap();

        let mut kh = KrigingHybrid::new(|| Box::new(DecisionTree::new()), coords.clone())
            .with_n_neighbors(10);
        let model = kh.train_regress_geo(&train_task).unwrap();

        // Held-out points: offset grid locations (new coords, same domain).
        let mut held_features = Vec::new();
        let mut held_target = Vec::new();
        let mut held_coords = Vec::new();
        for r in 0..side {
            for c in 0..side {
                let x = c as f64 + 0.5;
                let y = r as f64 + 0.5;
                held_coords.push((x, y));
                held_features.push(x - y);
                let spatial_trend = (x * 0.5).sin() * 3.0 + (y * 0.5).cos() * 3.0;
                held_target.push((x - y) * 0.1 + spatial_trend);
            }
        }
        let held = Array2::from_shape_vec((side * side, 1), held_features).unwrap();

        let base_only = model.predict(&held).unwrap();
        let Prediction::Regression { predicted: base_pred, .. } = base_only else {
            panic!("expected regression");
        };
        let spatial = model.predict_spatial(&held, &held_coords).unwrap();
        let Prediction::Regression { predicted: spatial_pred, .. } = spatial else {
            panic!("expected regression");
        };

        let mse = |pred: &[f64]| -> f64 {
            pred.iter()
                .zip(&held_target)
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / pred.len() as f64
        };
        let base_mse = mse(&base_pred);
        let spatial_mse = mse(&spatial_pred);
        assert!(
            spatial_mse < base_mse,
            "kriging-corrected MSE ({spatial_mse}) should be lower than base-model-only MSE ({base_mse})"
        );
    }

    /// Same end-to-end setup as above, but through each Matérn variant:
    /// the correction must still beat the base model, confirming the new
    /// models are wired into the full fit → kriging path (not just
    /// `VariogramModel::value`).
    #[test]
    fn matern_variants_work_end_to_end_in_kriging_correction() {
        let side = 10;
        let mut rng = StdRng::seed_from_u64(11);
        let mut feats = Vec::new();
        let mut target = Vec::new();
        let mut coords = Vec::new();
        for r in 0..side {
            for c in 0..side {
                let x = c as f64;
                let y = r as f64;
                coords.push((x, y));
                feats.push(x - y);
                let spatial_trend = (x * 0.5).sin() * 3.0 + (y * 0.5).cos() * 3.0;
                target.push((x - y) * 0.1 + spatial_trend + rng.random::<f64>() * 0.05);
            }
        }
        let features = Array2::from_shape_vec((side * side, 1), feats).unwrap();
        let train_task = RegressionTask::new("t", features, target).unwrap();

        let mut held_features = Vec::new();
        let mut held_target = Vec::new();
        let mut held_coords = Vec::new();
        for r in 0..side {
            for c in 0..side {
                let x = c as f64 + 0.5;
                let y = r as f64 + 0.5;
                held_coords.push((x, y));
                held_features.push(x - y);
                let spatial_trend = (x * 0.5).sin() * 3.0 + (y * 0.5).cos() * 3.0;
                held_target.push((x - y) * 0.1 + spatial_trend);
            }
        }
        let held = Array2::from_shape_vec((side * side, 1), held_features).unwrap();

        for variant in [VariogramModel::Matern32, VariogramModel::Matern52] {
            let mut kh = KrigingHybrid::new(|| Box::new(DecisionTree::new()), coords.clone())
                .with_variogram_model(variant)
                .with_n_neighbors(10);
            let model = kh.train_regress_geo(&train_task).unwrap();

            let Prediction::Regression { predicted: base_pred, .. } =
                model.predict(&held).unwrap()
            else {
                panic!("expected regression");
            };
            let Prediction::Regression { predicted: spatial_pred, .. } =
                model.predict_spatial(&held, &held_coords).unwrap()
            else {
                panic!("expected regression");
            };

            let mse = |pred: &[f64]| -> f64 {
                pred.iter()
                    .zip(&held_target)
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / pred.len() as f64
            };
            assert!(
                mse(&spatial_pred) < mse(&base_pred),
                "{variant:?}: kriging correction should reduce held-out MSE"
            );
        }
    }

    /// Regression test: when the base learner already fits the target
    /// exactly (residuals ~0), the fitted variogram degenerates to
    /// nugget=sill=0 and used to make `predict_spatial` return `Err` for
    /// every query (an all-zero-variogram kriging system is singular). The
    /// honest correction in that case is simply 0.
    #[test]
    fn predict_spatial_handles_degenerate_zero_variance_residuals() {
        let n = 12;
        let feats: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let target: Vec<f64> = feats.iter().map(|&x| 2.0 * x + 1.0).collect(); // exactly linear
        let coords: Vec<(f64, f64)> = (0..n).map(|i| (i as f64, 0.0)).collect();
        let features = Array2::from_shape_vec((n, 1), feats.clone()).unwrap();
        let task = RegressionTask::new("t", features.clone(), target.clone()).unwrap();

        let mut kh = KrigingHybrid::new(|| Box::new(LinearRegression::new()), coords.clone())
            .with_n_neighbors(5);
        let model = kh.train_regress_geo(&task).unwrap();

        let pred = model
            .predict_spatial(&features, &coords)
            .expect("degenerate (zero-variance) variogram must not fail predict_spatial");
        let Prediction::Regression { predicted, .. } = pred else {
            panic!("expected regression");
        };
        for (i, &p) in predicted.iter().enumerate() {
            assert!(
                (p - target[i]).abs() < 1e-6,
                "base model already fits exactly, correction should be ~0: point {i} got {p}, expected {}",
                target[i]
            );
        }
    }

    /// Regression test for the M-8 finding (`docs/auditoria_motor_2026-07-05.md`):
    /// with small-magnitude targets (e.g. geochemistry concentrations
    /// ~1e-5), a genuinely spatially-structured sill can itself be tiny
    /// (~1e-10) -- the fixed absolute `1e-9` threshold this test replaces
    /// treated that as "degenerate" and silently returned the uncorrected
    /// base prediction for every query, regardless of how much real
    /// spatial signal there was to krige. The base learner here only sees
    /// a feature uncorrelated with position, so the entire small-scale
    /// spatial pattern in the target ends up in the residuals; the kriging
    /// correction should recover a meaningful chunk of it.
    #[test]
    fn predict_spatial_corrects_small_magnitude_spatially_structured_residuals() {
        let n = 30;
        // Feature scrambled relative to position: uncorrelated with the
        // spatial pattern below, so the base model can't explain it away.
        let feats: Vec<f64> = (0..n).map(|i| ((i * 37) % 11) as f64).collect();
        let coords: Vec<(f64, f64)> = (0..n).map(|i| (i as f64, 0.0)).collect();
        // Smooth, small-magnitude (~1e-5) spatial pattern, independent of `feats`.
        let target: Vec<f64> = (0..n).map(|i| 1e-5 * (i as f64 * 0.4).sin()).collect();

        let features = Array2::from_shape_vec((n, 1), feats).unwrap();
        let task = RegressionTask::new("t", features.clone(), target.clone()).unwrap();

        let mut kh = KrigingHybrid::new(|| Box::new(LinearRegression::new()), coords.clone())
            .with_n_neighbors(8);
        let model = kh.train_regress_geo(&task).unwrap();

        let mse = |predicted: &[f64]| -> f64 {
            predicted.iter().zip(&target).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n as f64
        };

        let Prediction::Regression { predicted: base_pred, .. } = model.predict(&features).unwrap()
        else {
            panic!("expected regression");
        };
        let Prediction::Regression { predicted: spatial_pred, .. } = model
            .predict_spatial(&features, &coords)
            .expect("a small-scale but genuinely non-degenerate variogram must not fail predict_spatial")
        else {
            panic!("expected regression");
        };

        let base_mse = mse(&base_pred);
        let spatial_mse = mse(&spatial_pred);
        assert!(
            spatial_mse < base_mse * 0.5,
            "kriging correction should meaningfully reduce MSE on this spatially \
             structured small-scale signal: base_mse={base_mse:e}, spatial_mse={spatial_mse:e} \
             (under the old fixed 1e-9 threshold these would be identical -- the \
             correction never activates)"
        );
    }

    /// Regression test: duplicate training coordinates (e.g. repeated
    /// samples from the same borehole) used to make the local kriging
    /// system singular whenever both copies fell in the same query's
    /// neighborhood, failing `predict_spatial` for any nearby query.
    #[test]
    fn predict_spatial_handles_duplicate_training_coordinates() {
        let side = 5;
        let mut feats = Vec::new();
        let mut target = Vec::new();
        let mut coords = Vec::new();
        for r in 0..side {
            for c in 0..side {
                let x = c as f64;
                let y = r as f64;
                coords.push((x, y));
                feats.push(x - y);
                let spatial_trend = (x * 0.5).sin() * 2.0;
                target.push((x - y) * 0.1 + spatial_trend);
            }
        }
        // Duplicate one location exactly (same coords, slightly different
        // residual-inducing target) — the classic "two analyses of the same
        // borehole" case.
        coords.push(coords[0]);
        feats.push(feats[0]);
        target.push(target[0] + 0.05);

        let features = Array2::from_shape_vec((coords.len(), 1), feats).unwrap();
        let task = RegressionTask::new("t", features.clone(), target).unwrap();

        let mut kh = KrigingHybrid::new(|| Box::new(DecisionTree::new()), coords.clone())
            .with_n_neighbors(6);
        let model = kh.train_regress_geo(&task).unwrap();

        // Query right at the duplicated location: its neighborhood is
        // guaranteed to contain both coincident training points.
        let (qx, qy) = coords[0];
        let query_features = Array2::from_shape_vec((1, 1), vec![qx - qy]).unwrap();
        let result = model.predict_spatial(&query_features, &[coords[0]]);
        assert!(
            result.is_ok(),
            "duplicate training coordinates should not make predict_spatial fail: {:?}",
            result.err()
        );
    }
}
