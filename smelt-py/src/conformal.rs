//! Model-free split-conformal calibration (`smelt_ml::conformal::SplitConformal`).
//!
//! The learner-attached `conformal_predict` methods drive `&dyn TrainedModel`
//! internally, so they can only conformalize predictions computable from a
//! feature matrix alone. This class instead calibrates from predictions the
//! caller already made -- the path for `GeoXGBoost`/`KrigingHybrid`'s
//! `predict(x, coords)` spatial output, which the trait can't reach.

use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::common::{parse_coords, smelt_err};

/// Model-free split conformal prediction for regression.
///
/// Calibrates on precomputed `(prediction, truth)` pairs from a held-out
/// calibration set, then wraps any future point predictions in symmetric
/// intervals with guaranteed `1 - alpha` marginal coverage (Vovk et al.).
/// Because it never calls the model itself, it works with predictions from
/// ANY source -- in particular the spatially-corrected output of
/// `GeoXGBoost.predict(x, coords)` / `KrigingHybrid.predict(x, coords)`,
/// which the learner-attached `conformal_predict` methods cannot
/// conformalize (they only see the global, coordinate-free model). This is
/// exactly the flow the PM2.5 replication uses on the Rust side.
///
/// Example (spatial use case):
///     sc = smelt.SplitConformal(alpha=0.1)
///     cal_pred = model.predict(x_cal, coords=coords_cal)   # spatial predictions
///     sc.calibrate_from_predictions(cal_pred, y_cal)
///     test_pred = model.predict(x_test, coords=coords_test)
///     lower, upper = sc.predict_interval(test_pred)        # ~90% coverage
///
/// Notes:
///     - The calibration quantile uses rank `ceil((n+1)(1-alpha))`. If the
///       calibration set is too small for that rank (e.g. n=4 with
///       alpha=0.1), the interval width is `inf` -- the only width
///       consistent with the coverage guarantee -- rather than silently
///       clamping to the largest observed residual.
///     - Calibration data must be held out from training; calibrating on
///       training residuals underestimates the interval width.
#[pyclass]
pub(crate) struct SplitConformal {
    alpha: f64,
    calibrated: Option<smelt_ml::conformal::SplitConformal>,
}

#[pymethods]
impl SplitConformal {
    /// `alpha`: miscoverage level in (0, 1); 0.1 gives 90% coverage.
    #[new]
    #[pyo3(signature = (alpha=0.1))]
    fn new(alpha: f64) -> PyResult<Self> {
        // Eager validation, matching the crate's convention (KrigingHybrid's
        // variogram_model, XGBoost's objective): a bad alpha fails here, not
        // at calibration time.
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(PyValueError::new_err(format!(
                "alpha must be in (0, 1), got {alpha}"
            )));
        }
        Ok(Self {
            alpha,
            calibrated: None,
        })
    }

    /// Calibrate from precomputed calibration-set predictions and their true
    /// targets (1D array-likes of equal, non-zero length). Raises ValueError
    /// on mismatched lengths or an empty calibration set. Returns None;
    /// after this, `predict_interval` is available.
    fn calibrate_from_predictions(
        &mut self,
        cal_pred: Vec<f64>,
        cal_truth: Vec<f64>,
    ) -> PyResult<()> {
        if cal_pred.is_empty() || cal_truth.is_empty() {
            return Err(PyValueError::new_err(
                "calibration set is empty; conformal calibration needs at least \
                 ceil((n+1)(1-alpha)) <= n held-out samples for a finite interval",
            ));
        }
        if cal_pred.len() != cal_truth.len() {
            return Err(PyValueError::new_err(format!(
                "length mismatch: cal_pred has {} elements but cal_truth has {}",
                cal_pred.len(),
                cal_truth.len()
            )));
        }
        let sc = smelt_ml::conformal::SplitConformal::calibrate_from_predictions(
            &cal_pred, &cal_truth, self.alpha,
        )
        .map_err(smelt_err)?;
        self.calibrated = Some(sc);
        Ok(())
    }

    /// Wrap point predictions in the calibrated ± interval. Returns a
    /// `(lower, upper)` tuple of numpy arrays aligned with `test_pred`.
    fn predict_interval<'py>(
        &self,
        py: Python<'py>,
        test_pred: Vec<f64>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let sc = self.calibrated.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "not calibrated. Call calibrate_from_predictions(cal_pred, cal_truth) first.",
            )
        })?;
        let intervals = sc.intervals_for(&test_pred);
        let lower: Vec<f64> = intervals.iter().map(|iv| iv.lower).collect();
        let upper: Vec<f64> = intervals.iter().map(|iv| iv.upper).collect();
        Ok((PyArray1::from_vec(py, lower), PyArray1::from_vec(py, upper)))
    }

    /// The calibrated half-width (±) of the intervals; `inf` if the
    /// calibration set was too small for a finite `1 - alpha` guarantee.
    #[getter]
    fn interval_width(&self) -> PyResult<f64> {
        let sc = self.calibrated.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "not calibrated. Call calibrate_from_predictions(cal_pred, cal_truth) first.",
            )
        })?;
        Ok(sc.interval_width())
    }

    /// The miscoverage level this predictor was constructed with.
    #[getter]
    fn alpha(&self) -> f64 {
        self.alpha
    }
}

/// Resolve the Python-facing kernel string to `smelt_ml::conformal::SpatialKernel`.
fn resolve_kernel(kernel: &str) -> PyResult<smelt_ml::conformal::SpatialKernel> {
    use smelt_ml::conformal::SpatialKernel;
    match kernel {
        "gaussian" => Ok(SpatialKernel::Gaussian),
        "exponential" => Ok(SpatialKernel::Exponential),
        other => Err(PyValueError::new_err(format!(
            "unknown kernel '{other}'; use 'gaussian' or 'exponential'"
        ))),
    }
}

/// Spatially-localized (weighted) split conformal prediction for regression.
///
/// Unlike `SplitConformal`'s single global interval width, this weights each
/// calibration residual by spatial proximity to the test location, so the
/// interval width ADAPTS to the local residual scale — the fix for spatial
/// heteroscedasticity, where a global quantile over-covers calm regions and
/// under-covers noisy ones (Barber, Candès, Ramdas & Tibshirani 2023, weighted
/// conformal, with a spatial kernel supplying the weights).
///
/// Example:
///     lc = smelt.LocalizedConformal(alpha=0.1, bandwidth=0.2, kernel="gaussian")
///     cal_pred = model.predict(x_cal, coords=coords_cal)
///     lc.calibrate_from_predictions(cal_pred, y_cal, coords_cal)
///     test_pred = model.predict(x_test, coords=coords_test)
///     lower, upper = lc.predict_interval(test_pred, coords_test)  # adaptive width
///
/// Far from all calibration data the interval is `inf` (honest abstention).
#[pyclass]
pub(crate) struct LocalizedConformal {
    alpha: f64,
    bandwidth: f64,
    kernel: String,
    calibrated: Option<smelt_ml::conformal::LocalizedConformal>,
}

#[pymethods]
impl LocalizedConformal {
    /// `alpha`: miscoverage (0.1 → 90%). `bandwidth`: spatial kernel scale in
    /// coordinate units. `kernel`: "gaussian" (default) or "exponential".
    #[new]
    #[pyo3(signature = (alpha=0.1, bandwidth=0.2, kernel="gaussian".to_string()))]
    fn new(alpha: f64, bandwidth: f64, kernel: String) -> PyResult<Self> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(PyValueError::new_err(format!(
                "alpha must be in (0, 1), got {alpha}"
            )));
        }
        if !(bandwidth > 0.0 && bandwidth.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "bandwidth must be positive and finite, got {bandwidth}"
            )));
        }
        resolve_kernel(&kernel)?; // eager validation
        Ok(Self {
            alpha,
            bandwidth,
            kernel,
            calibrated: None,
        })
    }

    /// Calibrate from held-out calibration predictions, their true targets, and
    /// their coordinates (an (N, 2) array-like of (x, y)).
    fn calibrate_from_predictions(
        &mut self,
        cal_pred: Vec<f64>,
        cal_truth: Vec<f64>,
        coords: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let xy = parse_coords(coords)?;
        let kernel = resolve_kernel(&self.kernel)?;
        let lc = smelt_ml::conformal::LocalizedConformal::calibrate_from_predictions(
            &cal_pred,
            &cal_truth,
            &xy,
            self.alpha,
            self.bandwidth,
            kernel,
        )
        .map_err(smelt_err)?;
        self.calibrated = Some(lc);
        Ok(())
    }

    /// Wrap point predictions in the spatially-adaptive interval at each test
    /// coordinate. Returns `(lower, upper)` numpy arrays aligned with `test_pred`.
    fn predict_interval<'py>(
        &self,
        py: Python<'py>,
        test_pred: Vec<f64>,
        coords: &Bound<'_, PyAny>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let lc = self.not_calibrated()?;
        let xy = parse_coords(coords)?;
        let intervals = lc.intervals_for(&test_pred, &xy).map_err(smelt_err)?;
        let lower: Vec<f64> = intervals.iter().map(|iv| iv.lower).collect();
        let upper: Vec<f64> = intervals.iter().map(|iv| iv.upper).collect();
        Ok((PyArray1::from_vec(py, lower), PyArray1::from_vec(py, upper)))
    }

    /// The per-location interval half-widths at `coords` (diagnostic): shows
    /// how the width adapts across space. `inf` where the method abstains.
    fn halfwidths<'py>(
        &self,
        py: Python<'py>,
        coords: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let lc = self.not_calibrated()?;
        let xy = parse_coords(coords)?;
        let hw: Vec<f64> = xy.iter().map(|&c| lc.halfwidth_at(c)).collect();
        Ok(PyArray1::from_vec(py, hw))
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.alpha
    }
    #[getter]
    fn bandwidth(&self) -> f64 {
        self.bandwidth
    }
}

impl LocalizedConformal {
    fn not_calibrated(&self) -> PyResult<&smelt_ml::conformal::LocalizedConformal> {
        self.calibrated.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err(
                "not calibrated. Call calibrate_from_predictions(cal_pred, cal_truth, coords) first.",
            )
        })
    }
}
