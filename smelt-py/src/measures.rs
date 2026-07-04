//! Standalone measure/scoring functions exposed to Python.

use crate::common::smelt_err;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use smelt_ml::measure::Measure;
use smelt_ml::prediction::Prediction;

// ── Measures ───────────────────────────────────────────────────────────

/// Converts predicted class labels from `f64` (as returned by `predict()`)
/// to `usize`, rejecting negative or non-integer values instead of letting
/// Rust's saturating float-to-int cast silently turn e.g. `-1.0` into class
/// `0` — which would then count as a correct prediction against a true `0`.
fn to_class_labels(y_pred: &[f64]) -> PyResult<Vec<usize>> {
    y_pred
        .iter()
        .map(|&v| {
            if v.is_finite() && v >= 0.0 && v.fract() == 0.0 {
                Ok(v as usize)
            } else {
                Err(PyRuntimeError::new_err(format!(
                    "y_pred must contain non-negative integer class labels, got {v}"
                )))
            }
        })
        .collect()
}

#[pyfunction]
pub(crate) fn accuracy_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::Accuracy.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn rmse_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred = Prediction::regression_with_truth(y_pred, y_true);
    smelt_ml::prelude::Rmse.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn r2_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred = Prediction::regression_with_truth(y_pred, y_true);
    smelt_ml::prelude::RSquared.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn mae_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred = Prediction::regression_with_truth(y_pred, y_true);
    smelt_ml::prelude::Mae.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn f1_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::F1Score.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn precision_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::Precision.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn recall_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::Recall.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn balanced_accuracy_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::BalancedAccuracy
        .score(&pred)
        .map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn cohens_kappa_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::CohensKappa.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn mcc_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::Mcc.score(&pred).map_err(smelt_err)
}

/// Brier score. `y_proba` is 2D: `[[p0, p1, ...], ...]` (per-class probabilities).
#[pyfunction]
pub(crate) fn brier_score(y_true: Vec<usize>, y_proba: Vec<Vec<f64>>) -> PyResult<f64> {
    let pred_class: Vec<usize> = y_proba
        .iter()
        .map(|p| {
            p.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or_else(|| PyRuntimeError::new_err("y_proba contains an empty row"))
        })
        .collect::<PyResult<Vec<usize>>>()?;
    let pred = Prediction::Classification {
        predicted: pred_class,
        truth: Some(y_true),
        probabilities: Some(y_proba),
    };
    smelt_ml::prelude::Brier.score(&pred).map_err(smelt_err)
}

/// AUC-ROC score. Accepts y_proba as either:
/// - 2D: [[p0, p1], ...] (per-class probabilities)
/// - 1D: [p1, ...] (probability of positive class, sklearn-compatible)
#[pyfunction]
pub(crate) fn auc_roc_score(y_true: Vec<usize>, y_proba: &Bound<'_, PyAny>) -> PyResult<f64> {
    // Try 2D first, then 1D
    let proba_2d: Vec<Vec<f64>> = if let Ok(v2d) = y_proba.extract::<Vec<Vec<f64>>>() {
        v2d
    } else if let Ok(v1d) = y_proba.extract::<Vec<f64>>() {
        // Convert 1D (sklearn format) to 2D: p1 → [1-p1, p1]
        v1d.iter().map(|&p| vec![1.0 - p, p]).collect()
    } else {
        return Err(PyRuntimeError::new_err(
            "y_proba must be 1D (sklearn format: [p_positive, ...]) or 2D ([[p0, p1], ...])",
        ));
    };

    let pred_class: Vec<usize> = proba_2d
        .iter()
        .map(|p| {
            p.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or_else(|| PyRuntimeError::new_err("y_proba contains an empty row"))
        })
        .collect::<PyResult<Vec<usize>>>()?;
    let pred = Prediction::Classification {
        predicted: pred_class,
        truth: Some(y_true),
        probabilities: Some(proba_2d),
    };
    smelt_ml::prelude::AucRoc.score(&pred).map_err(smelt_err)
}

