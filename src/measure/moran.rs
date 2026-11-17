//! Moran's I spatial-autocorrelation diagnostic for model residuals.
//!
//! A model whose residuals are spatially autocorrelated has left spatial
//! structure on the table: nearby sites are over- or under-predicted
//! together. Testing the residuals with Moran's I (Moran 1950; inference per
//! Cliff & Ord 1981) is the standard diagnostic for deciding whether a
//! spatially-aware learner ([`GeoXGBoost`](crate::learner::GeoXGBoost),
//! [`KrigingHybrid`](crate::learner::KrigingHybrid)) or spatial
//! cross-validation ([`SpatialBlockCV`](crate::resample::SpatialBlockCV)) is
//! warranted, and for checking afterwards whether it actually removed the
//! structure.
//!
//! Two entry points:
//!
//! - [`morans_i`] / [`morans_i_permutation`]: the statistic on any vector of
//!   values at known coordinates (residuals, raw target, a covariate), with
//!   full inference ([`MoranResult`]).
//! - [`MoranResiduals`]: a [`Measure`] wrapper that scores a regression
//!   [`Prediction`] by the Moran's I of `truth - predicted`, so it slots into
//!   any `&[&dyn Measure]` list — with the caveat that the prediction's rows
//!   must be aligned with the coordinates it was built with (see its docs).
//!
//! Coordinates are passed alongside the values rather than stored in
//! `Task`, the same idiom as `SpatialBlockCV`/`SpatialBufferCV`/`GeoXGBoost`.
//! Weights are always row-standardised (each row of *W* sums to 1), the
//! convention under which Moran's I is bounded in roughly `[-1, 1]` and the
//! one PySAL's `esda.Moran`/spdep's `moran.test` use by default.
//!
//! The golden tests at the bottom of this module reproduce
//! `esda 2.10.0` + `libpysal 4.15.1` to 1e-9 on the statistic and both
//! analytic variances.

use crate::measure::Measure;
use crate::prediction::Prediction;
use crate::{Result, SmeltError};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

/// How the spatial weight matrix *W* is built from coordinates.
///
/// Every variant is row-standardised after construction.
#[derive(Debug, Clone, PartialEq)]
pub enum SpatialWeights {
    /// Each site's `k` nearest neighbours (Euclidean) get equal weight.
    /// Ties at the k-th distance are broken by index order. Requires
    /// `1 <= k < n`. This is the variant that always yields a connected,
    /// island-free *W*, so it's the default for [`MoranResiduals`].
    KNearest {
        /// Number of neighbours per site.
        k: usize,
    },
    /// Every other site within `bandwidth` gets weight `1 / d^power`; sites
    /// with no neighbour inside the bandwidth become islands (an all-zero
    /// row, which simply doesn't contribute — same handling as libpysal).
    /// `bandwidth: None` means all pairs are neighbours. Coincident points
    /// (`d = 0`) are rejected with an error rather than silently given
    /// infinite weight; use [`SpatialWeights::KNearest`] or jitter them.
    InverseDistance {
        /// Maximum neighbour distance; `None` for no cutoff.
        bandwidth: Option<f64>,
        /// Exponent on the distance (1.0 = plain inverse distance).
        power: f64,
    },
    /// Binary contiguity by distance: every site within `bandwidth` gets
    /// weight 1, everything else 0 (then row-standardised). Islands are
    /// possible, as for `InverseDistance`.
    DistanceBand {
        /// Maximum neighbour distance.
        bandwidth: f64,
    },
}

impl Default for SpatialWeights {
    /// `KNearest { k: 8 }`.
    fn default() -> Self {
        SpatialWeights::KNearest { k: 8 }
    }
}

/// Result of a Moran's I test.
///
/// The headline fields (`variance`, `z_score`, `p_value`) use the
/// **randomisation** assumption (Cliff & Ord 1981, eq. 1.42), which does not
/// require the values to be normally distributed — the right default for
/// model residuals, which rarely are. The `*_normal` fields give the
/// normality-assumption inference for comparison with software that reports
/// it; `p_value_permutation` is only set by [`morans_i_permutation`]. All
/// p-values are two-sided.
#[derive(Debug, Clone, PartialEq)]
pub struct MoranResult {
    /// Moran's I statistic. Positive: nearby values are alike (clustered
    /// residuals); negative: nearby values alternate; near [`expected`]
    /// (`-1/(n-1)`, which is ≈ 0 for large `n`): no spatial autocorrelation.
    ///
    /// [`expected`]: MoranResult::expected
    pub i: f64,
    /// Expected value of I under the null hypothesis, `-1/(n-1)`.
    pub expected: f64,
    /// Variance of I under the randomisation assumption.
    pub variance: f64,
    /// Standardised statistic `(i - expected) / sqrt(variance)`.
    pub z_score: f64,
    /// Two-sided p-value for `z_score` (normal approximation).
    pub p_value: f64,
    /// Variance of I under the normality assumption.
    pub variance_normal: f64,
    /// Standardised statistic under the normality assumption.
    pub z_score_normal: f64,
    /// Two-sided p-value under the normality assumption.
    pub p_value_normal: f64,
    /// Two-sided pseudo p-value from random permutations of the values over
    /// the sites, `2 * min(tail) `, with the `+1` correction
    /// (`(count + 1) / (n_permutations + 1)`). `None` unless computed via
    /// [`morans_i_permutation`].
    pub p_value_permutation: Option<f64>,
    /// Number of sites.
    pub n: usize,
}

impl MoranResult {
    /// Whether the randomisation-assumption `p_value` is below `alpha`.
    pub fn significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

/// Sparse, row-standardised spatial weight matrix: `rows[i]` holds
/// `(j, w_ij)` sorted by `j`.
struct WeightMatrix {
    rows: Vec<Vec<(usize, f64)>>,
}

impl WeightMatrix {
    fn build(coords: &[(f64, f64)], scheme: &SpatialWeights) -> Result<Self> {
        let n = coords.len();
        for (i, &(x, y)) in coords.iter().enumerate() {
            if !x.is_finite() || !y.is_finite() {
                return Err(SmeltError::InvalidParameter(format!(
                    "coords[{i}] = ({x}, {y}) must be finite"
                )));
            }
        }
        match scheme {
            SpatialWeights::KNearest { k } => {
                if *k == 0 || *k >= n {
                    return Err(SmeltError::InvalidParameter(format!(
                        "KNearest k must satisfy 1 <= k < n (got k = {k}, n = {n})"
                    )));
                }
            }
            SpatialWeights::InverseDistance { bandwidth, power } => {
                if let Some(bw) = bandwidth.filter(|bw| !bw.is_finite() || *bw <= 0.0) {
                    return Err(SmeltError::InvalidParameter(format!(
                        "InverseDistance bandwidth must be a finite number > 0 (got {bw})"
                    )));
                }
                if !power.is_finite() || *power <= 0.0 {
                    return Err(SmeltError::InvalidParameter(format!(
                        "InverseDistance power must be a finite number > 0 (got {power})"
                    )));
                }
            }
            SpatialWeights::DistanceBand { bandwidth } => {
                if !bandwidth.is_finite() || *bandwidth <= 0.0 {
                    return Err(SmeltError::InvalidParameter(format!(
                        "DistanceBand bandwidth must be a finite number > 0 (got {bandwidth})"
                    )));
                }
            }
        }

        let rows: Vec<Result<Vec<(usize, f64)>>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (xi, yi) = coords[i];
                let dist = |j: usize| {
                    let (xj, yj) = coords[j];
                    ((xi - xj).powi(2) + (yi - yj).powi(2)).sqrt()
                };
                let mut row: Vec<(usize, f64)> = match scheme {
                    SpatialWeights::KNearest { k } => {
                        let mut others: Vec<(usize, f64)> =
                            (0..n).filter(|&j| j != i).map(|j| (j, dist(j))).collect();
                        // Partial sort: the k smallest distances to the front,
                        // ties broken by index so the result is deterministic.
                        others.select_nth_unstable_by(k - 1, |a, b| {
                            a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0))
                        });
                        others.truncate(*k);
                        others.into_iter().map(|(j, _)| (j, 1.0)).collect()
                    }
                    SpatialWeights::InverseDistance { bandwidth, power } => {
                        let mut row = Vec::new();
                        for j in (0..n).filter(|&j| j != i) {
                            let d = dist(j);
                            if bandwidth.is_some_and(|bw| d > bw) {
                                continue;
                            }
                            if d == 0.0 {
                                return Err(SmeltError::InvalidParameter(format!(
                                    "coords[{i}] and coords[{j}] coincide: inverse-distance \
                                     weight is undefined (use KNearest or jitter the points)"
                                )));
                            }
                            row.push((j, d.powf(-*power)));
                        }
                        row
                    }
                    SpatialWeights::DistanceBand { bandwidth } => (0..n)
                        .filter(|&j| j != i && dist(j) <= *bandwidth)
                        .map(|j| (j, 1.0))
                        .collect(),
                };
                row.sort_unstable_by_key(|&(j, _)| j);
                let sum: f64 = row.iter().map(|&(_, w)| w).sum();
                if sum > 0.0 {
                    for entry in &mut row {
                        entry.1 /= sum;
                    }
                }
                Ok(row)
            })
            .collect();
        let rows = rows.into_iter().collect::<Result<Vec<_>>>()?;
        Ok(WeightMatrix { rows })
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        let row = &self.rows[i];
        match row.binary_search_by_key(&j, |&(jj, _)| jj) {
            Ok(pos) => row[pos].1,
            Err(_) => 0.0,
        }
    }

    /// `S0 = Σ w_ij`, `S1 = ½ Σ (w_ij + w_ji)²`, `S2 = Σ_i (w_i· + w_·i)²`.
    fn sums(&self) -> (f64, f64, f64) {
        let n = self.rows.len();
        let mut col_sums = vec![0.0; n];
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        for (i, row) in self.rows.iter().enumerate() {
            for &(j, w) in row {
                s0 += w;
                col_sums[j] += w;
                // S1 runs over *ordered* pairs. When w_ji is also non-zero the
                // (j, i) visit contributes the other half; when it is zero
                // (asymmetric kNN links) that visit never happens, so this
                // one must account for both halves.
                let w_ji = self.get(j, i);
                let sym = w + w_ji;
                s1 += if w_ji != 0.0 {
                    0.5 * sym * sym
                } else {
                    sym * sym
                };
            }
        }
        let s2 = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let row_sum: f64 = row.iter().map(|&(_, w)| w).sum();
                (row_sum + col_sums[i]).powi(2)
            })
            .sum();
        (s0, s1, s2)
    }

    /// `Σ_ij w_ij z_i z_j` for centred values `z`.
    fn cross_product(&self, z: &[f64]) -> f64 {
        self.rows
            .iter()
            .enumerate()
            .map(|(i, row)| z[i] * row.iter().map(|&(j, w)| w * z[j]).sum::<f64>())
            .sum()
    }
}

/// Validated inputs shared by [`morans_i`] and [`morans_i_permutation`]:
/// the weight matrix, the centred values, and the constant factors.
struct Prepared {
    w: WeightMatrix,
    z: Vec<f64>,
    /// `n / (S0 · Σ z²)`, so `I = scale · Σ w_ij z_i z_j`.
    scale: f64,
    result: MoranResult,
}

fn prepare(values: &[f64], coords: &[(f64, f64)], weights: &SpatialWeights) -> Result<Prepared> {
    let n = values.len();
    if n == 0 {
        return Err(SmeltError::EmptyDataset);
    }
    if coords.len() != n {
        return Err(SmeltError::DimensionMismatch {
            expected: n,
            got: coords.len(),
        });
    }
    if n < 4 {
        return Err(SmeltError::InvalidParameter(format!(
            "Moran's I inference needs at least 4 sites (got {n})"
        )));
    }
    if let Some(v) = values.iter().find(|v| !v.is_finite()) {
        return Err(SmeltError::InvalidParameter(format!(
            "values must be finite (found {v})"
        )));
    }
    let w = WeightMatrix::build(coords, weights)?;

    let mean = values.iter().sum::<f64>() / n as f64;
    let z: Vec<f64> = values.iter().map(|v| v - mean).collect();
    let m2: f64 = z.iter().map(|v| v * v).sum();
    if m2 <= 0.0 {
        return Err(SmeltError::NumericalError(
            "Moran's I is undefined for constant values (zero variance)".into(),
        ));
    }
    let m4: f64 = z.iter().map(|v| v.powi(4)).sum();
    let (s0, s1, s2) = w.sums();
    if s0 <= 0.0 {
        return Err(SmeltError::InvalidParameter(
            "spatial weights are all zero (every site is an island); widen the bandwidth".into(),
        ));
    }

    let nf = n as f64;
    let scale = nf / (s0 * m2);
    let i = scale * w.cross_product(&z);
    let expected = -1.0 / (nf - 1.0);

    // Cliff & Ord (1981): normality assumption.
    let variance_normal = (nf * nf * s1 - nf * s2 + 3.0 * s0 * s0) / ((nf * nf - 1.0) * s0 * s0)
        - expected * expected;
    // Randomisation assumption, with sample kurtosis b2.
    let b2 = nf * m4 / (m2 * m2);
    let num = nf * ((nf * nf - 3.0 * nf + 3.0) * s1 - nf * s2 + 3.0 * s0 * s0)
        - b2 * ((nf * nf - nf) * s1 - 2.0 * nf * s2 + 6.0 * s0 * s0);
    let den = (nf - 1.0) * (nf - 2.0) * (nf - 3.0) * s0 * s0;
    let variance = num / den - expected * expected;

    let z_of = |var: f64| {
        if var > 0.0 {
            (i - expected) / var.sqrt()
        } else {
            0.0
        }
    };
    let z_score = z_of(variance);
    let z_score_normal = z_of(variance_normal);

    let result = MoranResult {
        i,
        expected,
        variance,
        z_score,
        p_value: two_sided_p(z_score),
        variance_normal,
        z_score_normal,
        p_value_normal: two_sided_p(z_score_normal),
        p_value_permutation: None,
        n,
    };
    Ok(Prepared {
        w,
        z,
        scale,
        result,
    })
}

/// Moran's I of `values` observed at `coords`, with analytic inference.
///
/// # Errors
///
/// Empty input, `values`/`coords` length mismatch, fewer than 4 sites,
/// non-finite values or coordinates, constant values, invalid weight
/// parameters (see [`SpatialWeights`]), or a weight matrix that is entirely
/// zero.
///
/// # Examples
///
/// ```
/// use smelt_ml::measure::{morans_i, SpatialWeights};
///
/// // A smooth north–south gradient: strongly positive autocorrelation.
/// let coords: Vec<(f64, f64)> = (0..5)
///     .flat_map(|r| (0..5).map(move |c| (c as f64, r as f64)))
///     .collect();
/// let values: Vec<f64> = coords.iter().map(|&(_, y)| y).collect();
/// let res = morans_i(&values, &coords, &SpatialWeights::KNearest { k: 4 }).unwrap();
/// assert!(res.i > 0.5);
/// assert!(res.p_value < 0.001);
/// ```
pub fn morans_i(
    values: &[f64],
    coords: &[(f64, f64)],
    weights: &SpatialWeights,
) -> Result<MoranResult> {
    Ok(prepare(values, coords, weights)?.result)
}

/// [`morans_i`] plus a two-sided permutation pseudo p-value from
/// `n_permutations` random reassignments of the values to the sites
/// (deterministic for a given `seed`; permutations run in parallel).
///
/// Prefer this over the analytic p-value when `n` is small (≲ 30) or the
/// weights are very uneven, where the normal approximation is poor.
///
/// # Errors
///
/// As [`morans_i`], plus `n_permutations == 0`.
pub fn morans_i_permutation(
    values: &[f64],
    coords: &[(f64, f64)],
    weights: &SpatialWeights,
    n_permutations: usize,
    seed: u64,
) -> Result<MoranResult> {
    if n_permutations == 0 {
        return Err(SmeltError::InvalidParameter(
            "n_permutations must be > 0".into(),
        ));
    }
    let prep = prepare(values, coords, weights)?;
    let observed = prep.result.i;
    let sims: Vec<f64> = (0..n_permutations as u64)
        .into_par_iter()
        .map(|p| {
            // Independent stream per permutation so the result doesn't depend
            // on rayon's scheduling.
            let mut rng =
                StdRng::seed_from_u64(seed.wrapping_add(p.wrapping_mul(0x9E37_79B9_7F4A_7C15)));
            let mut z = prep.z.clone();
            z.shuffle(&mut rng);
            prep.scale * prep.w.cross_product(&z)
        })
        .collect();
    let above = sims.iter().filter(|&&s| s >= observed).count();
    let below = sims.iter().filter(|&&s| s <= observed).count();
    let tail = above.min(below);
    let p = (2.0 * (tail as f64 + 1.0) / (n_permutations as f64 + 1.0)).min(1.0);
    let mut result = prep.result;
    result.p_value_permutation = Some(p);
    Ok(result)
}

fn two_sided_p(z: f64) -> f64 {
    2.0 * (1.0 - standard_normal_cdf(z.abs()))
}

/// Standard normal CDF via the Abramowitz & Stegun 7.1.26 `erf`
/// approximation (|error| < 1.5e-7). Same hand-rolled routine as
/// `learner/hoeffding.rs` — no `f64::erf` in stable Rust and no numerics
/// crate in this workspace.
fn standard_normal_cdf(x: f64) -> f64 {
    let z = x / std::f64::consts::SQRT_2;
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let z = z.abs();
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let poly =
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592;
    let erf = 1.0 - poly * t * (-z * z).exp();
    0.5 * (1.0 + sign * erf)
}

/// Moran's I of regression residuals (`truth - predicted`) as a [`Measure`].
///
/// `score` returns the **signed** statistic `I`. It is a diagnostic, not an
/// objective: the ideal value is `≈ 0` (no residual structure), while a
/// clearly positive `I` means the model misses spatial structure and a
/// clearly negative one means it over-smooths. `maximize()` is `false` only
/// so that benchmark tables sort the "least clustered residuals" first for
/// the usual positive case — do **not** tune hyperparameters on it. For
/// significance use [`MoranResiduals::test`], which returns the full
/// [`MoranResult`].
///
/// # Where the coordinates come from
///
/// Two constructors:
///
/// - [`MoranResiduals::new`]`(coords)`: the measure owns the coordinates and
///   `score` requires the prediction's rows to correspond one-to-one, in
///   order, with them (it errors when the lengths differ). Natural for a
///   hold-out set whose coordinates you hold.
/// - [`MoranResiduals::coords_from_task`]`()`: no coordinates of its own.
///   `score` errors; the measure only works through
///   [`Measure::score_spatial`], which `benchmark::resample_*` calls with
///   each test fold's slice of the task's coordinates
///   ([`crate::task::RegressionTask::with_coords`]). This is how you get a
///   per-fold Moran's I of the residuals inside an ordinary (spatial) CV
///   benchmark, next to RMSE.
///
/// When both are available (`new(coords)` scored via `score_spatial`), the
/// coordinates passed in win: they are the ones aligned with the rows.
///
/// # Examples
///
/// ```
/// use smelt_ml::measure::{Measure, MoranResiduals, SpatialWeights};
/// use smelt_ml::prediction::Prediction;
///
/// let coords: Vec<(f64, f64)> = (0..6)
///     .flat_map(|r| (0..6).map(move |c| (c as f64, r as f64)))
///     .collect();
/// // Residuals that follow a smooth east–west gradient.
/// let truth: Vec<f64> = coords.iter().map(|&(x, _)| x).collect();
/// let predicted = vec![2.5; 36];
/// let pred = Prediction::regression_with_truth(predicted, truth);
///
/// let m = MoranResiduals::new(coords).with_weights(SpatialWeights::KNearest { k: 4 });
/// assert!(m.score(&pred).unwrap() > 0.5);
/// assert!(m.test(&pred).unwrap().p_value < 0.001);
/// ```
pub struct MoranResiduals {
    coords: Option<Vec<(f64, f64)>>,
    weights: SpatialWeights,
}

impl MoranResiduals {
    /// Measure over sites at `coords` with the default weights
    /// (`SpatialWeights::KNearest { k: 8 }`).
    pub fn new(coords: Vec<(f64, f64)>) -> Self {
        Self {
            coords: Some(coords),
            weights: SpatialWeights::default(),
        }
    }

    /// Measure with no coordinates of its own: they must arrive through
    /// [`Measure::score_spatial`] (as `benchmark::resample_*` does for a
    /// task built with `with_coords`). Default weights
    /// (`SpatialWeights::KNearest { k: 8 }`); plain `score` is an error.
    pub fn coords_from_task() -> Self {
        Self {
            coords: None,
            weights: SpatialWeights::default(),
        }
    }

    /// Replace the weight scheme.
    pub fn with_weights(mut self, weights: SpatialWeights) -> Self {
        self.weights = weights;
        self
    }

    /// The coordinates this measure owns (`None` for
    /// [`Self::coords_from_task`]).
    pub fn coords(&self) -> Option<&[(f64, f64)]> {
        self.coords.as_deref()
    }

    fn own_coords(&self) -> Result<&[(f64, f64)]> {
        self.coords.as_deref().ok_or_else(|| {
            SmeltError::InvalidParameter(
                "MoranResiduals::coords_from_task() has no coordinates of its own: score it \
                 through Measure::score_spatial (benchmark::resample_* does this for a task \
                 built with with_coords), or construct it with MoranResiduals::new(coords)"
                    .into(),
            )
        })
    }

    /// The weight scheme in use.
    pub fn weights(&self) -> &SpatialWeights {
        &self.weights
    }

    fn residuals(&self, prediction: &Prediction, n_coords: usize) -> Result<Vec<f64>> {
        match prediction {
            Prediction::Regression {
                predicted,
                truth: Some(truth),
            } => {
                if predicted.len() != truth.len() {
                    return Err(SmeltError::DimensionMismatch {
                        expected: truth.len(),
                        got: predicted.len(),
                    });
                }
                if predicted.len() != n_coords {
                    return Err(SmeltError::DimensionMismatch {
                        expected: n_coords,
                        got: predicted.len(),
                    });
                }
                Ok(truth.iter().zip(predicted).map(|(t, p)| t - p).collect())
            }
            _ => Err(SmeltError::IncompatiblePrediction(
                "Moran's I of residuals requires a regression prediction with truth".into(),
            )),
        }
    }

    /// Full Moran's I test (statistic, z-score, p-values) on the residuals,
    /// against the measure's own coordinates.
    pub fn test(&self, prediction: &Prediction) -> Result<MoranResult> {
        self.test_with_coords(prediction, self.own_coords()?)
    }

    /// Full Moran's I test on the residuals, with the prediction's rows
    /// georeferenced by `coords` (used by [`Measure::score_spatial`]).
    pub fn test_with_coords(
        &self,
        prediction: &Prediction,
        coords: &[(f64, f64)],
    ) -> Result<MoranResult> {
        morans_i(
            &self.residuals(prediction, coords.len())?,
            coords,
            &self.weights,
        )
    }

    /// As [`MoranResiduals::test`], adding a permutation p-value.
    pub fn test_permutation(
        &self,
        prediction: &Prediction,
        n_permutations: usize,
        seed: u64,
    ) -> Result<MoranResult> {
        let coords = self.own_coords()?;
        morans_i_permutation(
            &self.residuals(prediction, coords.len())?,
            coords,
            &self.weights,
            n_permutations,
            seed,
        )
    }
}

impl Measure for MoranResiduals {
    fn id(&self) -> &str {
        "regr.moran_residuals"
    }
    fn maximize(&self) -> bool {
        false
    }
    fn score(&self, prediction: &Prediction) -> Result<f64> {
        Ok(self.test(prediction)?.i)
    }
    fn score_spatial(&self, prediction: &Prediction, coords: &[(f64, f64)]) -> Result<f64> {
        Ok(self.test_with_coords(prediction, coords)?.i)
    }
}

#[cfg(test)]
mod tests {
    //! Golden tests against `esda 2.10.0` / `libpysal 4.15.1`
    //! (`esda.moran.Moran(values, w, two_tailed=True)` with `w.transform =
    //! "r"`), on a 20-site fixture generated by a fixed LCG. Both analytic
    //! variances and z-scores are checked, so the S0/S1/S2 bookkeeping is
    //! verified independently of the statistic itself.
    use super::*;

    fn fixture() -> (Vec<(f64, f64)>, Vec<f64>) {
        let coords = vec![
            (6.552, 3.048),
            (6.75, 1.068),
            (5.166, 4.897),
            (6.025, 3.7),
            (2.567, 3.742),
            (8.256, 1.727),
            (2.978, 6.435),
            (7.897, 9.878),
            (8.006, 4.643),
            (5.39, 6.255),
            (2.5, 7.039),
            (7.163, 9.795),
            (3.29, 4.455),
            (7.084, 7.399),
            (1.728, 0.157),
            (7.826, 0.412),
            (5.974, 2.457),
            (5.565, 5.152),
            (3.975, 1.826),
            (6.46, 7.165),
        ];
        let values = vec![
            1.5213, 1.1865, 2.3132, 1.7978, 1.752, 0.6992, 2.4786, 2.9718, 1.6762, 2.6595, 2.469,
            3.233, 2.039, 2.4081, 0.5464, 0.4939, 1.5452, 2.2089, 1.3214, 2.7174,
        ];
        (coords, values)
    }

    /// Values drawn independently of the coordinates (same LCG stream).
    #[allow(clippy::approx_constant)] // 0.4343 is a fixture value, not log10(e)
    fn noise_fixture() -> Vec<f64> {
        vec![
            0.0722, 0.0857, 0.197, 0.8984, 0.5492, 0.4889, 0.3762, 0.0999, 0.4534, 0.3085, 0.3771,
            0.1227, 0.9454, 0.7919, 0.4343, 0.5654, 0.0211, 0.9603, 0.1596, 0.349,
        ]
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    fn assert_matches_esda(
        r: &MoranResult,
        i: f64,
        var_rand: f64,
        z_rand: f64,
        p_rand: f64,
        var_norm: f64,
        z_norm: f64,
    ) {
        assert_eq!(r.n, 20);
        assert!(close(r.expected, -1.0 / 19.0, 1e-12));
        assert!(close(r.i, i, 1e-9), "I {} vs esda {}", r.i, i);
        assert!(
            close(r.variance, var_rand, 1e-9),
            "VI_rand {} vs esda {}",
            r.variance,
            var_rand
        );
        assert!(
            close(r.z_score, z_rand, 1e-8),
            "z_rand {} vs esda {}",
            r.z_score,
            z_rand
        );
        // p-value goes through the A&S erf approximation (|err| < 1.5e-7).
        assert!(
            (r.p_value - p_rand).abs() < 1e-6,
            "p_rand {} vs esda {}",
            r.p_value,
            p_rand
        );
        assert!(
            close(r.variance_normal, var_norm, 1e-9),
            "VI_norm {} vs esda {}",
            r.variance_normal,
            var_norm
        );
        assert!(
            close(r.z_score_normal, z_norm, 1e-8),
            "z_norm {} vs esda {}",
            r.z_score_normal,
            z_norm
        );
        assert!(r.p_value_permutation.is_none());
    }

    #[test]
    fn knn4_matches_esda() {
        let (coords, values) = fixture();
        let r = morans_i(&values, &coords, &SpatialWeights::KNearest { k: 4 }).unwrap();
        assert_matches_esda(
            &r,
            0.604394436932,
            1.528639917934e-02,
            5.3141030054,
            1.0718397947e-07,
            1.482076902783e-02,
            5.3969351160,
        );
    }

    #[test]
    fn knn1_matches_esda() {
        let (coords, values) = fixture();
        let r = morans_i(&values, &coords, &SpatialWeights::KNearest { k: 1 }).unwrap();
        assert_matches_esda(
            &r,
            0.835134811667,
            8.101692201512e-02,
            3.1189673577,
            1.8148606608e-03,
            7.843292441630e-02,
            3.1699287095,
        );
    }

    #[test]
    fn inverse_distance_with_bandwidth_matches_esda() {
        // libpysal: DistanceBand(threshold=4.0, binary=False, alpha=-1.0)
        let (coords, values) = fixture();
        let w = SpatialWeights::InverseDistance {
            bandwidth: Some(4.0),
            power: 1.0,
        };
        let r = morans_i(&values, &coords, &w).unwrap();
        assert_matches_esda(
            &r,
            0.615783669862,
            1.308097171474e-02,
            5.8442171687,
            5.0895595879e-09,
            1.266408972919e-02,
            5.9396295547,
        );
    }

    #[test]
    fn inverse_distance_all_pairs_matches_esda() {
        // libpysal: DistanceBand(threshold=1e9, binary=False, alpha=-1.0)
        let (coords, values) = fixture();
        let w = SpatialWeights::InverseDistance {
            bandwidth: None,
            power: 1.0,
        };
        let r = morans_i(&values, &coords, &w).unwrap();
        assert_matches_esda(
            &r,
            0.257322140413,
            3.056681061801e-03,
            5.6062413334,
            2.0676760710e-08,
            2.960556201000e-03,
            5.6965274862,
        );
    }

    #[test]
    fn distance_band_matches_esda() {
        // libpysal: DistanceBand(threshold=3.0, binary=True)
        let (coords, values) = fixture();
        let r = morans_i(
            &values,
            &coords,
            &SpatialWeights::DistanceBand { bandwidth: 3.0 },
        )
        .unwrap();
        assert_matches_esda(
            &r,
            0.603942988882,
            1.446826914294e-02,
            5.4585308225,
            4.8009053091e-08,
            1.401196378059e-02,
            5.5466984200,
        );
    }

    #[test]
    fn spatially_independent_noise_is_not_significant() {
        // Same fixture coordinates, values drawn independently of space
        // (esda: I = -0.0874, z_rand = -0.281, p_rand = 0.779).
        let (coords, _) = fixture();
        let noise = noise_fixture();
        let r = morans_i(&noise, &coords, &SpatialWeights::KNearest { k: 4 }).unwrap();
        assert!(close(r.i, -0.087358370831, 1e-9));
        assert!(close(r.z_score, -0.2809665176, 1e-8));
        assert!((r.p_value - 0.77873608088).abs() < 1e-6);
        assert!(!r.significant(0.05));
    }

    #[test]
    fn permutation_p_value_agrees_with_analytic_for_clustered_and_noise() {
        let (coords, values) = fixture();
        let w = SpatialWeights::KNearest { k: 4 };
        // Clustered: esda's 9,999-permutation pseudo p is 1e-4 (one-sided);
        // ours is two-sided, so the floor is 2/(n+1).
        let r = morans_i_permutation(&values, &coords, &w, 1999, 7).unwrap();
        let p = r.p_value_permutation.unwrap();
        assert!(p <= 0.005, "permutation p = {p}");
        assert!(close(r.i, 0.604394436932, 1e-9)); // analytic part unchanged
        // Noise: should be comfortably non-significant.
        let noise = noise_fixture();
        let r = morans_i_permutation(&noise, &coords, &w, 1999, 7).unwrap();
        assert!(r.p_value_permutation.unwrap() > 0.2);
    }

    #[test]
    fn permutation_is_deterministic_in_seed_and_independent_of_thread_count() {
        let (coords, values) = fixture();
        let w = SpatialWeights::KNearest { k: 3 };
        let a = morans_i_permutation(&values, &coords, &w, 500, 42).unwrap();
        let b = morans_i_permutation(&values, &coords, &w, 500, 42).unwrap();
        assert_eq!(a.p_value_permutation, b.p_value_permutation);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let c = pool.install(|| morans_i_permutation(&values, &coords, &w, 500, 42).unwrap());
        assert_eq!(a.p_value_permutation, c.p_value_permutation);
    }

    #[test]
    fn rejects_invalid_inputs() {
        let (coords, values) = fixture();
        let k4 = SpatialWeights::KNearest { k: 4 };
        assert!(matches!(
            morans_i(&[], &[], &k4),
            Err(SmeltError::EmptyDataset)
        ));
        assert!(matches!(
            morans_i(&values[..5], &coords, &k4),
            Err(SmeltError::DimensionMismatch {
                expected: 5,
                got: 20
            })
        ));
        assert!(matches!(
            morans_i(
                &values[..3],
                &coords[..3],
                &SpatialWeights::KNearest { k: 1 }
            ),
            Err(SmeltError::InvalidParameter(_))
        ));
        assert!(matches!(
            morans_i(&[1.0; 20], &coords, &k4),
            Err(SmeltError::NumericalError(_))
        ));
        assert!(matches!(
            morans_i(&values, &coords, &SpatialWeights::KNearest { k: 0 }),
            Err(SmeltError::InvalidParameter(_))
        ));
        assert!(matches!(
            morans_i(&values, &coords, &SpatialWeights::KNearest { k: 20 }),
            Err(SmeltError::InvalidParameter(_))
        ));
        assert!(matches!(
            morans_i_permutation(&values, &coords, &k4, 0, 1),
            Err(SmeltError::InvalidParameter(_))
        ));
        let mut nan_vals = values.clone();
        nan_vals[3] = f64::NAN;
        assert!(matches!(
            morans_i(&nan_vals, &coords, &k4),
            Err(SmeltError::InvalidParameter(_))
        ));
        // Coincident points under inverse distance.
        let mut dup = coords.clone();
        dup[1] = dup[0];
        let idw = SpatialWeights::InverseDistance {
            bandwidth: None,
            power: 1.0,
        };
        assert!(matches!(
            morans_i(&values, &dup, &idw),
            Err(SmeltError::InvalidParameter(_))
        ));
        // ...but fine under kNN.
        assert!(morans_i(&values, &dup, &k4).is_ok());
        // Bandwidth so small that every site is an island.
        assert!(matches!(
            morans_i(
                &values,
                &coords,
                &SpatialWeights::DistanceBand { bandwidth: 0.01 }
            ),
            Err(SmeltError::InvalidParameter(_))
        ));
    }

    #[test]
    fn islands_are_skipped_not_fatal() {
        // Bandwidth 1.5 leaves some sites without neighbours; the statistic
        // must still be computable over the connected ones.
        let (coords, values) = fixture();
        let r = morans_i(
            &values,
            &coords,
            &SpatialWeights::DistanceBand { bandwidth: 1.5 },
        )
        .unwrap();
        assert!(r.i.is_finite());
        assert!(r.variance.is_finite() && r.variance > 0.0);
    }

    #[test]
    fn measure_wrapper_scores_residuals_and_checks_alignment() {
        let (coords, values) = fixture();
        let m = MoranResiduals::new(coords.clone()).with_weights(SpatialWeights::KNearest { k: 4 });
        assert_eq!(m.id(), "regr.moran_residuals");
        assert!(!m.maximize());

        // Constant prediction → residuals are the (centred) values themselves,
        // so the score must equal the golden I for the raw values.
        let pred = Prediction::regression_with_truth(vec![1.0; 20], values.clone());
        assert!(close(m.score(&pred).unwrap(), 0.604394436932, 1e-9));
        let t = m.test(&pred).unwrap();
        assert!(close(t.z_score, 5.3141030054, 1e-8));
        let tp = m.test_permutation(&pred, 999, 1).unwrap();
        assert!(tp.p_value_permutation.unwrap() < 0.01);

        // A perfect fit on a spatially smooth field leaves tiny, but not
        // identical, residuals: no structure left.
        let near_perfect: Vec<f64> = values
            .iter()
            .enumerate()
            .map(|(i, v)| v + 1e-3 * ((i * 7919) % 13) as f64)
            .collect();
        let pred = Prediction::regression_with_truth(near_perfect, values.clone());
        assert!(!m.test(&pred).unwrap().significant(0.05));

        // Misaligned lengths and wrong prediction type are errors.
        let short = Prediction::regression_with_truth(vec![1.0; 19], values[..19].to_vec());
        assert!(matches!(
            m.score(&short),
            Err(SmeltError::DimensionMismatch {
                expected: 20,
                got: 19
            })
        ));
        let no_truth = Prediction::regression(vec![1.0; 20]);
        assert!(matches!(
            m.score(&no_truth),
            Err(SmeltError::IncompatiblePrediction(_))
        ));
        let classif = Prediction::classification_with_truth(vec![0; 20], vec![0; 20]);
        assert!(matches!(
            m.score(&classif),
            Err(SmeltError::IncompatiblePrediction(_))
        ));
    }

    #[test]
    fn coords_from_task_scores_only_through_score_spatial() {
        let (coords, values) = fixture();
        let m = MoranResiduals::coords_from_task().with_weights(SpatialWeights::KNearest { k: 4 });
        assert!(m.coords().is_none());
        let pred = Prediction::regression_with_truth(vec![1.0; 20], values.clone());
        assert!(matches!(
            m.score(&pred),
            Err(SmeltError::InvalidParameter(_))
        ));
        assert!(matches!(
            m.test(&pred),
            Err(SmeltError::InvalidParameter(_))
        ));
        assert!(close(
            m.score_spatial(&pred, &coords).unwrap(),
            0.604394436932,
            1e-9
        ));
        // Supplied coordinates win over owned ones (they are the row-aligned ones).
        let owned = MoranResiduals::new(vec![(0.0, 0.0); 20])
            .with_weights(SpatialWeights::KNearest { k: 4 });
        assert!(close(
            owned.score_spatial(&pred, &coords).unwrap(),
            0.604394436932,
            1e-9
        ));
        // Misaligned supplied coordinates are an error, not a silent truncation.
        assert!(matches!(
            m.score_spatial(&pred, &coords[..19]),
            Err(SmeltError::DimensionMismatch {
                expected: 19,
                got: 20
            })
        ));
    }

    #[test]
    fn standard_normal_cdf_matches_known_values() {
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((standard_normal_cdf(1.959964) - 0.975).abs() < 1e-6);
        assert!((standard_normal_cdf(-1.959964) - 0.025).abs() < 1e-6);
        assert!((two_sided_p(1.959964) - 0.05).abs() < 1e-6);
    }
}
