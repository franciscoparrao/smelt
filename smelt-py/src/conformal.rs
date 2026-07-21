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

use crate::common::smelt_err;

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
