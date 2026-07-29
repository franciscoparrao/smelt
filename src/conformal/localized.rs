//! Spatially-localized (weighted) split conformal prediction.
//!
//! Standard split conformal ([`super::SplitConformal`]) uses a single global
//! quantile of the calibration residuals, so its interval width is constant
//! across space. Under **spatial heteroscedasticity** — a residual scale that
//! varies smoothly across location, which is pervasive in geoscience — that
//! global width over-covers calm regions and under-covers noisy ones: marginal
//! coverage can look fine while *conditional* coverage fails exactly where a
//! hazard/prospectivity map must be trustworthy.
//!
//! `LocalizedConformal` weights each calibration residual by spatial proximity
//! to the test location, so the quantile — and hence the interval width —
//! adapts to the local residual scale. This is weighted / nonexchangeable
//! conformal prediction (Barber, Candès, Ramdas & Tibshirani, 2023, *Conformal
//! prediction beyond exchangeability*) with a spatial kernel supplying the
//! weights; the coverage guarantee carries over with a gap term controlled by
//! the weights.
//!
//! Model-free, like [`super::SplitConformal`]: the caller supplies calibration
//! predictions, targets, and coordinates (the `TrainedModel` trait carries no
//! coordinates, so the spatial learners' real predictors —
//! `TrainedKrigingHybrid::predict_spatial`, `TrainedGeoXGBoost::predict_spatial`
//! — must be conformalized through this precomputed path).

use super::ConformalInterval;
use crate::{Result, SmeltError};

/// Spatial weighting kernel for [`LocalizedConformal`]. Both kernels equal `1`
/// at distance `0`, so a test point's own (unknown) residual carries unit
/// weight in the weighted-quantile construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpatialKernel {
    /// Gaussian: `exp(-d² / (2·h²))`. Smooth, light tails — localizes sharply.
    Gaussian,
    /// Exponential: `exp(-d / h)`. Heavier tails — retains more distant mass.
    Exponential,
}

impl SpatialKernel {
    /// Kernel value at distance `d` with bandwidth `h`.
    fn eval(&self, d: f64, h: f64) -> f64 {
        match self {
            SpatialKernel::Gaussian => (-(d * d) / (2.0 * h * h)).exp(),
            SpatialKernel::Exponential => (-d / h).exp(),
        }
    }
}

fn euclidean(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// Spatially-localized split conformal predictor.
///
/// Calibrated from precomputed calibration predictions, targets, and their
/// coordinates. Each test location gets an interval whose half-width is the
/// spatially-weighted `(1-α)` quantile of the calibration residuals.
///
/// # Examples
///
/// ```
/// use smelt_ml::conformal::{LocalizedConformal, SpatialKernel};
///
/// // Calibration: residuals are small on the left, large on the right.
/// // (alpha=0.5 — a 50% interval — keeps this tiny illustrative example finite;
/// // higher coverage needs more calibration points *per region*.)
/// let cal_pred = vec![0.0; 6];
/// let cal_tgt  = vec![0.1, -0.1, 0.15, 2.0, -1.8, 2.2];
/// let cal_xy   = vec![(0.0, 0.0), (0.1, 0.2), (0.2, 0.1),
///                     (1.0, 0.0), (1.1, 0.2), (0.9, 0.1)];
///
/// let lc = LocalizedConformal::calibrate_from_predictions(
///     &cal_pred, &cal_tgt, &cal_xy, 0.5, 0.3, SpatialKernel::Gaussian,
/// ).unwrap();
///
/// // The interval is wider in the noisy (right) region than the calm (left) one.
/// let calm = lc.halfwidth_at((0.1, 0.1));
/// let noisy = lc.halfwidth_at((1.0, 0.1));
/// assert!(noisy > calm);
/// ```
pub struct LocalizedConformal {
    cal_coords: Vec<(f64, f64)>,
    cal_abs_residuals: Vec<f64>,
    /// Residual indices sorted ascending by `cal_abs_residuals` (precomputed so
    /// each `halfwidth_at` is a single weighted sweep, not a re-sort).
    sorted_idx: Vec<usize>,
    alpha: f64,
    bandwidth: f64,
    kernel: SpatialKernel,
}

impl LocalizedConformal {
    /// Calibrate from precomputed calibration predictions, their true targets,
    /// and their coordinates. `alpha` is the miscoverage level (0.1 → 90%),
    /// `bandwidth` the spatial kernel scale (in the units of `coords`).
    pub fn calibrate_from_predictions(
        cal_predictions: &[f64],
        cal_targets: &[f64],
        cal_coords: &[(f64, f64)],
        alpha: f64,
        bandwidth: f64,
        kernel: SpatialKernel,
    ) -> Result<Self> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(SmeltError::InvalidParameter(format!(
                "conformal alpha must be in (0, 1), got {alpha}"
            )));
        }
        if !(bandwidth > 0.0 && bandwidth.is_finite()) {
            return Err(SmeltError::InvalidParameter(format!(
                "LocalizedConformal bandwidth must be positive and finite, got {bandwidth}"
            )));
        }
        if cal_targets.is_empty() {
            return Err(SmeltError::EmptyDataset);
        }
        if cal_predictions.len() != cal_targets.len() || cal_coords.len() != cal_targets.len() {
            return Err(SmeltError::DimensionMismatch {
                expected: cal_targets.len(),
                got: cal_predictions.len().min(cal_coords.len()),
            });
        }
        if !cal_coords
            .iter()
            .all(|&(x, y)| x.is_finite() && y.is_finite())
        {
            return Err(SmeltError::InvalidParameter(
                "LocalizedConformal coordinates must be finite".into(),
            ));
        }

        let cal_abs_residuals: Vec<f64> = cal_predictions
            .iter()
            .zip(cal_targets)
            .map(|(p, t)| (p - t).abs())
            .collect();
        let mut sorted_idx: Vec<usize> = (0..cal_abs_residuals.len()).collect();
        sorted_idx.sort_by(|&a, &b| {
            cal_abs_residuals[a]
                .partial_cmp(&cal_abs_residuals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(Self {
            cal_coords: cal_coords.to_vec(),
            cal_abs_residuals,
            sorted_idx,
            alpha,
            bandwidth,
            kernel,
        })
    }

    /// The spatially-weighted `(1-α)` quantile of the calibration residuals at
    /// `coord` — the interval half-width for a test point there.
    ///
    /// Weights are `kernel(d(coord, sᵢ))`; the test point's own unknown
    /// residual carries weight `kernel(0)=1` placed at `+∞`. If the calibration
    /// weight cannot reach `1-α` (the test point is far from all calibration
    /// data), the half-width is `+∞` — an honest abstention rather than a
    /// falsely-confident finite interval.
    pub fn halfwidth_at(&self, coord: (f64, f64)) -> f64 {
        let w_star = 1.0; // kernel at distance 0
        let weights: Vec<f64> = self
            .cal_coords
            .iter()
            .map(|&c| self.kernel.eval(euclidean(coord, c), self.bandwidth))
            .collect();
        let total: f64 = weights.iter().sum::<f64>() + w_star;
        let target = 1.0 - self.alpha;

        let mut cum = 0.0;
        for &i in &self.sorted_idx {
            cum += weights[i] / total;
            if cum >= target {
                return self.cal_abs_residuals[i];
            }
        }
        f64::INFINITY
    }

    /// Interval for a single test point.
    pub fn interval_at(&self, prediction: f64, coord: (f64, f64)) -> ConformalInterval {
        let hw = self.halfwidth_at(coord);
        ConformalInterval {
            prediction,
            lower: prediction - hw,
            upper: prediction + hw,
        }
    }

    /// Intervals for a batch of test points (predictions aligned with coords).
    pub fn intervals_for(
        &self,
        predictions: &[f64],
        coords: &[(f64, f64)],
    ) -> Result<Vec<ConformalInterval>> {
        if predictions.len() != coords.len() {
            return Err(SmeltError::DimensionMismatch {
                expected: predictions.len(),
                got: coords.len(),
            });
        }
        Ok(predictions
            .iter()
            .zip(coords)
            .map(|(&p, &c)| self.interval_at(p, c))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformal::SplitConformal;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// With an enormous bandwidth every calibration point gets weight ≈ 1, so
    /// the weighted quantile collapses to the SAME finite-sample order
    /// statistic `⌈(n+1)(1-α)⌉` that `SplitConformal` uses — the localized
    /// predictor must reduce EXACTLY to standard split conformal in the
    /// no-localization limit. (Both place unit mass on the test point at +∞,
    /// so the rank arithmetic matches to the bit.)
    #[test]
    fn reduces_to_split_conformal_at_infinite_bandwidth() {
        let cal_pred = vec![0.0; 7];
        let cal_tgt = vec![0.4, -1.2, 0.9, -0.3, 2.1, -0.7, 1.5];
        let coords: Vec<(f64, f64)> = (0..7).map(|i| (i as f64, -(i as f64))).collect();
        let alpha = 0.2;

        let std = SplitConformal::calibrate_from_predictions(&cal_pred, &cal_tgt, alpha).unwrap();
        let loc = LocalizedConformal::calibrate_from_predictions(
            &cal_pred,
            &cal_tgt,
            &coords,
            alpha,
            1e9,
            SpatialKernel::Gaussian,
        )
        .unwrap();

        // At any location the localized half-width equals the global width.
        for &c in &[(0.0, 0.0), (3.5, -2.0), (100.0, 100.0)] {
            assert_eq!(loc.halfwidth_at(c), std.interval_width(), "at {c:?}");
        }
    }

    /// Statistical validity: on exchangeable (iid) data the localized intervals
    /// achieve approximately nominal MARGINAL coverage. Conformal coverage is
    /// distribution-free, so iid Uniform(-1,1) noise at random locations
    /// suffices — no spatial structure, so the method must not break the
    /// guarantee it is built on.
    #[test]
    fn marginal_coverage_is_approximately_nominal_on_iid_data() {
        let mut rng = StdRng::seed_from_u64(7);
        let alpha = 0.1;
        let noise = |r: &mut StdRng| r.random_range(-1.0..1.0);

        let n_cal = 500;
        let cal_pred = vec![0.0; n_cal];
        let cal_tgt: Vec<f64> = (0..n_cal).map(|_| noise(&mut rng)).collect();
        let cal_xy: Vec<(f64, f64)> = (0..n_cal)
            .map(|_| (rng_unit(&mut rng), rng_unit(&mut rng)))
            .collect();

        let lc = LocalizedConformal::calibrate_from_predictions(
            &cal_pred,
            &cal_tgt,
            &cal_xy,
            alpha,
            0.25,
            SpatialKernel::Gaussian,
        )
        .unwrap();

        let n_test = 4000;
        let mut covered = 0;
        for _ in 0..n_test {
            let c = (rng_unit(&mut rng), rng_unit(&mut rng));
            let y = noise(&mut rng);
            if y.abs() <= lc.halfwidth_at(c) {
                covered += 1;
            }
        }
        let coverage = covered as f64 / n_test as f64;
        assert!(
            (0.87..=0.97).contains(&coverage),
            "marginal coverage {coverage} should be near nominal 0.90"
        );
    }

    /// Adaptivity — the whole point: where local calibration residuals are
    /// large the half-width must be larger than where they are small. Each
    /// region needs enough points for a finite-sample (1-α) quantile to exist
    /// locally, so we seed dense left (calm) and right (noisy) clusters.
    #[test]
    fn halfwidth_adapts_to_local_residual_scale() {
        let mut rng = StdRng::seed_from_u64(11);
        let per = 60usize;
        let mut cal_pred = Vec::new();
        let mut cal_tgt = Vec::new();
        let mut cal_xy = Vec::new();
        // calm cluster around x≈0: residual scale 0.1
        for _ in 0..per {
            cal_pred.push(0.0);
            cal_tgt.push(0.1 * rng.random_range(-1.0..1.0));
            cal_xy.push((rng.random_range(0.0..0.2), rng.random_range(0.0..0.2)));
        }
        // noisy cluster around x≈1: residual scale 2.0
        for _ in 0..per {
            cal_pred.push(0.0);
            cal_tgt.push(2.0 * rng.random_range(-1.0..1.0));
            cal_xy.push((rng.random_range(0.9..1.1), rng.random_range(0.0..0.2)));
        }
        let lc = LocalizedConformal::calibrate_from_predictions(
            &cal_pred,
            &cal_tgt,
            &cal_xy,
            0.1,
            0.15,
            SpatialKernel::Gaussian,
        )
        .unwrap();
        let calm = lc.halfwidth_at((0.1, 0.1));
        let noisy = lc.halfwidth_at((1.0, 0.1));
        assert!(calm.is_finite() && noisy.is_finite());
        assert!(
            noisy > 3.0 * calm,
            "noisy region half-width {noisy} should dwarf calm {calm}"
        );
    }

    /// Far from every calibration point the kernel mass vanishes and the
    /// weighted quantile cannot reach 1-α → the interval is +∞ (abstention),
    /// not a spuriously tight finite interval.
    #[test]
    fn abstains_far_from_data() {
        let cal_pred = vec![0.0; 5];
        let cal_tgt = vec![0.3, -0.4, 0.5, -0.2, 0.6];
        let cal_xy = vec![(0.0, 0.0), (0.1, 0.0), (0.0, 0.1), (0.1, 0.1), (0.05, 0.05)];
        let lc = LocalizedConformal::calibrate_from_predictions(
            &cal_pred,
            &cal_tgt,
            &cal_xy,
            0.1,
            0.05,
            SpatialKernel::Gaussian,
        )
        .unwrap();
        assert!(lc.halfwidth_at((100.0, 100.0)).is_infinite());
    }

    #[test]
    fn rejects_bad_arguments() {
        let p = vec![0.0; 3];
        let t = vec![1.0, 2.0, 3.0];
        let xy = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
        let k = SpatialKernel::Gaussian;
        // bad alpha
        assert!(LocalizedConformal::calibrate_from_predictions(&p, &t, &xy, 0.0, 0.2, k).is_err());
        // bad bandwidth
        assert!(LocalizedConformal::calibrate_from_predictions(&p, &t, &xy, 0.1, 0.0, k).is_err());
        // length mismatch
        assert!(
            LocalizedConformal::calibrate_from_predictions(&p, &[1.0, 2.0], &xy, 0.1, 0.2, k)
                .is_err()
        );
        // empty
        assert!(
            LocalizedConformal::calibrate_from_predictions(&[], &[], &[], 0.1, 0.2, k).is_err()
        );
    }

    fn rng_unit(rng: &mut StdRng) -> f64 {
        use rand::Rng;
        rng.random_range(0.0..1.0)
    }
}
