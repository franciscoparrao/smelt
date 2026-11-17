//! Standalone measure/scoring functions exposed to Python.

use crate::common::smelt_err;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use smelt_ml::measure::Measure;
use smelt_ml::prediction::Prediction;

// ── Measures ───────────────────────────────────────────────────────────

/// Rejects mismatched `y_true`/`y_pred` lengths with a `ValueError` naming
/// both lengths. Before this, every two-array measure zip-truncated to the
/// common prefix and returned a "perfect-looking" score (5th audit LOW-D3:
/// `rmse_score([1,2,3],[1.0])` → 0.0, `accuracy_score` → 1.0) -- the same
/// check `wilcoxon_signed_rank` already gets from the Rust stats module.
fn check_same_len(len_true: usize, name_pred: &str, len_pred: usize) -> PyResult<()> {
    if len_true != len_pred {
        return Err(PyValueError::new_err(format!(
            "length mismatch: y_true has {len_true} elements but {name_pred} has {len_pred}"
        )));
    }
    Ok(())
}

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
                Err(PyValueError::new_err(format!(
                    "y_pred must contain non-negative integer class labels, got {v}"
                )))
            }
        })
        .collect()
}

#[pyfunction]
pub(crate) fn accuracy_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    check_same_len(y_true.len(), "y_pred", y_pred.len())?;
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::Accuracy.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn rmse_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    check_same_len(y_true.len(), "y_pred", y_pred.len())?;
    let pred = Prediction::regression_with_truth(y_pred, y_true);
    smelt_ml::prelude::Rmse.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn r2_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    check_same_len(y_true.len(), "y_pred", y_pred.len())?;
    let pred = Prediction::regression_with_truth(y_pred, y_true);
    smelt_ml::prelude::RSquared.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn mae_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    check_same_len(y_true.len(), "y_pred", y_pred.len())?;
    let pred = Prediction::regression_with_truth(y_pred, y_true);
    smelt_ml::prelude::Mae.score(&pred).map_err(smelt_err)
}

/// Mean Absolute Percentage Error (lower is better). Zero-valued actuals
/// are skipped, matching the Rust `Mape` measure. Listed in
/// `tuning._MINIMIZE_METRIC_NAMES` since 0.4.x but only actually bound now.
#[pyfunction]
pub(crate) fn mape_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    check_same_len(y_true.len(), "y_pred", y_pred.len())?;
    let pred = Prediction::regression_with_truth(y_pred, y_true);
    smelt_ml::measure::Mape.score(&pred).map_err(smelt_err)
}

/// Logarithmic loss (lower is better) from per-class probabilities
/// (`[[p0, p1, ...], ...]`). Probabilities are clamped to [1e-15, 1-1e-15]
/// before the log, matching the Rust `LogLoss` measure. Listed in
/// `tuning._MINIMIZE_METRIC_NAMES` since 0.4.x but only actually bound now.
#[pyfunction]
pub(crate) fn logloss_score(y_true: Vec<usize>, y_proba: Vec<Vec<f64>>) -> PyResult<f64> {
    check_same_len(y_true.len(), "y_proba", y_proba.len())?;
    for (i, (&t, row)) in y_true.iter().zip(&y_proba).enumerate() {
        if t >= row.len() {
            return Err(PyValueError::new_err(format!(
                "y_true[{i}] = {t} but y_proba rows only cover {} classes",
                row.len()
            )));
        }
    }
    let pred_class: Vec<usize> = argmax_rows(&y_proba)?;
    let pred = Prediction::Classification {
        predicted: pred_class,
        truth: Some(y_true),
        probabilities: Some(y_proba),
    };
    smelt_ml::measure::LogLoss.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn f1_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    check_same_len(y_true.len(), "y_pred", y_pred.len())?;
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::F1Score.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn precision_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    check_same_len(y_true.len(), "y_pred", y_pred.len())?;
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::Precision.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn recall_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    check_same_len(y_true.len(), "y_pred", y_pred.len())?;
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::Recall.score(&pred).map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn balanced_accuracy_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    check_same_len(y_true.len(), "y_pred", y_pred.len())?;
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::BalancedAccuracy
        .score(&pred)
        .map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn cohens_kappa_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    check_same_len(y_true.len(), "y_pred", y_pred.len())?;
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::CohensKappa
        .score(&pred)
        .map_err(smelt_err)
}

#[pyfunction]
pub(crate) fn mcc_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    check_same_len(y_true.len(), "y_pred", y_pred.len())?;
    let pred_u = to_class_labels(&y_pred)?;
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::Mcc.score(&pred).map_err(smelt_err)
}

/// Per-row argmax over a probability matrix; a `ValueError` on empty rows.
fn argmax_rows(proba: &[Vec<f64>]) -> PyResult<Vec<usize>> {
    proba
        .iter()
        .map(|p| {
            p.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or_else(|| PyValueError::new_err("y_proba contains an empty row"))
        })
        .collect()
}

/// Extract a per-class probability matrix from a Python value that may be
/// 2D (`[[p0, p1, ...], ...]`) or 1D sklearn-style (`[p_positive, ...]`,
/// expanded to `[[1-p, p], ...]`). Anything else raises a clear
/// `ValueError` instead of PyO3's raw extraction `TypeError` (5th audit
/// LOW-D5) -- same handling `auc_roc_score` already had.
fn extract_proba_matrix(y_proba: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(v2d) = y_proba.extract::<Vec<Vec<f64>>>() {
        Ok(v2d)
    } else if let Ok(v1d) = y_proba.extract::<Vec<f64>>() {
        Ok(v1d.iter().map(|&p| vec![1.0 - p, p]).collect())
    } else {
        Err(PyValueError::new_err(
            "y_proba must be 1D (sklearn format: [p_positive, ...]) or 2D ([[p0, p1], ...])",
        ))
    }
}

/// Brier score. `y_proba` is 2D: `[[p0, p1, ...], ...]` (per-class
/// probabilities), or 1D sklearn-style `[p_positive, ...]` for the binary
/// case.
#[pyfunction]
pub(crate) fn brier_score(y_true: Vec<usize>, y_proba: &Bound<'_, PyAny>) -> PyResult<f64> {
    let proba_2d = extract_proba_matrix(y_proba)?;
    check_same_len(y_true.len(), "y_proba", proba_2d.len())?;
    let pred_class: Vec<usize> = argmax_rows(&proba_2d)?;
    let pred = Prediction::Classification {
        predicted: pred_class,
        truth: Some(y_true),
        probabilities: Some(proba_2d),
    };
    smelt_ml::prelude::Brier.score(&pred).map_err(smelt_err)
}

/// AUC-ROC score. Accepts y_proba as either:
/// - 2D: [[p0, p1], ...] (per-class probabilities)
/// - 1D: [p1, ...] (probability of positive class, sklearn-compatible)
#[pyfunction]
pub(crate) fn auc_roc_score(y_true: Vec<usize>, y_proba: &Bound<'_, PyAny>) -> PyResult<f64> {
    let proba_2d = extract_proba_matrix(y_proba)?;
    check_same_len(y_true.len(), "y_proba", proba_2d.len())?;
    let pred_class: Vec<usize> = argmax_rows(&proba_2d)?;
    let pred = Prediction::Classification {
        predicted: pred_class,
        truth: Some(y_true),
        probabilities: Some(proba_2d),
    };
    smelt_ml::prelude::AucRoc.score(&pred).map_err(smelt_err)
}

// ── Moran's I (spatial autocorrelation of residuals) ───────────────────

/// Result of a Moran's I test (see `smelt_ml::measure::MoranResult`).
///
/// `variance`/`z_score`/`p_value` use the randomisation assumption; the
/// `*_normal` fields the normality assumption; `p_value_permutation` is
/// `None` unless `n_permutations > 0`. All p-values are two-sided.
#[pyclass(name = "MoranResult")]
#[derive(Clone)]
pub(crate) struct PyMoranResult {
    #[pyo3(get)]
    i: f64,
    #[pyo3(get)]
    expected: f64,
    #[pyo3(get)]
    variance: f64,
    #[pyo3(get)]
    z_score: f64,
    #[pyo3(get)]
    p_value: f64,
    #[pyo3(get)]
    variance_normal: f64,
    #[pyo3(get)]
    z_score_normal: f64,
    #[pyo3(get)]
    p_value_normal: f64,
    #[pyo3(get)]
    p_value_permutation: Option<f64>,
    #[pyo3(get)]
    n: usize,
}

#[pymethods]
impl PyMoranResult {
    fn __repr__(&self) -> String {
        match self.p_value_permutation {
            Some(pp) => format!(
                "MoranResult(I={:.4}, expected={:.4}, z={:.3}, p={:.3e}, p_perm={:.3e}, n={})",
                self.i, self.expected, self.z_score, self.p_value, pp, self.n
            ),
            None => format!(
                "MoranResult(I={:.4}, expected={:.4}, z={:.3}, p={:.3e}, n={})",
                self.i, self.expected, self.z_score, self.p_value, self.n
            ),
        }
    }

    /// Whether the randomisation-assumption p-value is below `alpha`.
    #[pyo3(signature = (alpha=0.05))]
    fn significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

impl From<smelt_ml::measure::MoranResult> for PyMoranResult {
    fn from(r: smelt_ml::measure::MoranResult) -> Self {
        Self {
            i: r.i,
            expected: r.expected,
            variance: r.variance,
            z_score: r.z_score,
            p_value: r.p_value,
            variance_normal: r.variance_normal,
            z_score_normal: r.z_score_normal,
            p_value_normal: r.p_value_normal,
            p_value_permutation: r.p_value_permutation,
            n: r.n,
        }
    }
}

/// Resolves the Python `weights`/`k`/`bandwidth`/`power` arguments into a
/// `SpatialWeights`, rejecting unknown scheme names eagerly (same
/// `resolve_*` pattern as XGBoost's `objective`).
fn resolve_spatial_weights(
    weights: &str,
    k: usize,
    bandwidth: Option<f64>,
    power: f64,
) -> PyResult<smelt_ml::measure::SpatialWeights> {
    use smelt_ml::measure::SpatialWeights;
    match weights {
        "knn" => Ok(SpatialWeights::KNearest { k }),
        "inverse_distance" => Ok(SpatialWeights::InverseDistance { bandwidth, power }),
        "distance_band" => match bandwidth {
            Some(bandwidth) => Ok(SpatialWeights::DistanceBand { bandwidth }),
            None => Err(PyValueError::new_err(
                "weights='distance_band' requires a bandwidth",
            )),
        },
        other => Err(PyValueError::new_err(format!(
            "unknown weights scheme {other:?}; expected 'knn', 'inverse_distance' or 'distance_band'"
        ))),
    }
}

fn run_morans_i(
    values: &[f64],
    coords: &Bound<'_, PyAny>,
    weights: &str,
    k: usize,
    bandwidth: Option<f64>,
    power: f64,
    n_permutations: usize,
    seed: u64,
) -> PyResult<PyMoranResult> {
    let coords = crate::common::parse_coords(coords)?;
    let scheme = resolve_spatial_weights(weights, k, bandwidth, power)?;
    let r = if n_permutations == 0 {
        smelt_ml::measure::morans_i(values, &coords, &scheme)
    } else {
        smelt_ml::measure::morans_i_permutation(values, &coords, &scheme, n_permutations, seed)
    }
    .map_err(smelt_err)?;
    Ok(r.into())
}

/// Moran's I of `values` observed at `coords` (n×2 array or list of (x, y)).
///
/// `weights`: 'knn' (k nearest neighbours, default k=8), 'inverse_distance'
/// (1/d^power within `bandwidth`, or all pairs when None) or 'distance_band'
/// (binary within `bandwidth`); all row-standardised. `n_permutations > 0`
/// adds a two-sided permutation pseudo p-value.
#[pyfunction]
#[pyo3(signature = (values, coords, weights="knn", k=8, bandwidth=None, power=1.0, n_permutations=0, seed=42))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn morans_i(
    values: Vec<f64>,
    coords: &Bound<'_, PyAny>,
    weights: &str,
    k: usize,
    bandwidth: Option<f64>,
    power: f64,
    n_permutations: usize,
    seed: u64,
) -> PyResult<PyMoranResult> {
    run_morans_i(
        &values,
        coords,
        weights,
        k,
        bandwidth,
        power,
        n_permutations,
        seed,
    )
}

/// Moran's I of the regression residuals `y_true - y_pred` at `coords`:
/// a clearly positive, significant I means the model leaves spatial
/// structure unexplained. Same arguments as `morans_i`.
#[pyfunction]
#[pyo3(signature = (y_true, y_pred, coords, weights="knn", k=8, bandwidth=None, power=1.0, n_permutations=0, seed=42))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn moran_residuals(
    y_true: Vec<f64>,
    y_pred: Vec<f64>,
    coords: &Bound<'_, PyAny>,
    weights: &str,
    k: usize,
    bandwidth: Option<f64>,
    power: f64,
    n_permutations: usize,
    seed: u64,
) -> PyResult<PyMoranResult> {
    check_same_len(y_true.len(), "y_pred", y_pred.len())?;
    let residuals: Vec<f64> = y_true.iter().zip(&y_pred).map(|(t, p)| t - p).collect();
    run_morans_i(
        &residuals,
        coords,
        weights,
        k,
        bandwidth,
        power,
        n_permutations,
        seed,
    )
}
