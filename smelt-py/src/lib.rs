//! Python bindings for smelt-ml via PyO3.

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use smelt_ml::learner::{Learner, TrainedModel};
use smelt_ml::measure::Measure;
use smelt_ml::prediction::Prediction;

// ── Helpers ────────────────────────────────────────────────────────────

fn to_array2(x: PyReadonlyArray2<'_, f64>) -> Array2<f64> {
    x.as_array().to_owned()
}

fn smelt_err(e: smelt_ml::SmeltError) -> PyErr {
    PyRuntimeError::new_err(format!("{}", e))
}

fn is_integer(y: &Bound<'_, PyAny>) -> bool {
    if let Ok(arr) = y.getattr("dtype") {
        if let Ok(kind) = arr.getattr("kind") {
            if let Ok(k) = kind.extract::<String>() {
                return k == "i" || k == "u" || k == "b";
            }
        }
    }
    y.extract::<Vec<i64>>().is_ok()
}

/// Convert Python integer labels to non-negative `usize` class indices.
/// Rejects negative labels explicitly instead of letting `v as usize` wrap
/// (e.g. the SVM convention `y = [-1, 1]` would otherwise become
/// `18446744073709551615`, silently corrupting the class count downstream).
fn extract_class_labels(y: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let target: Vec<i64> = y.extract()?;
    target
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            usize::try_from(v).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "negative class label {v} at index {i}; class labels must be non-negative integers"
                ))
            })
        })
        .collect()
}

/// Train a learner, releasing the GIL for the (rayon-parallel, potentially
/// long-running) training call. `y` is extracted into an owned Rust value
/// first since that requires the GIL; the actual `train_classif`/
/// `train_regress` call runs under `py.allow_threads` so it doesn't block
/// other Python threads (e.g. a Jupyter kernel) or Ctrl+C for the duration.
fn fit_learner(
    py: Python<'_>,
    learner: &mut dyn Learner,
    features: Array2<f64>,
    y: &Bound<'_, PyAny>,
) -> PyResult<(Box<dyn TrainedModel>, bool)> {
    if is_integer(y) {
        let target = extract_class_labels(y)?;
        let task = smelt_ml::task::ClassificationTask::new("py", features, target)
            .map_err(smelt_err)?;
        let model = py
            .allow_threads(|| learner.train_classif(&task))
            .map_err(smelt_err)?;
        Ok((model, true))
    } else {
        let target: Vec<f64> = y.extract()?;
        let task =
            smelt_ml::task::RegressionTask::new("py", features, target).map_err(smelt_err)?;
        let model = py
            .allow_threads(|| learner.train_regress(&task))
            .map_err(smelt_err)?;
        Ok((model, false))
    }
}

fn predict_values<'py>(
    model: &dyn TrainedModel,
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let pred = model.predict(&to_array2(x)).map_err(smelt_err)?;
    let values: Vec<f64> = match &pred {
        Prediction::Classification { predicted, .. } => {
            predicted.iter().map(|&p| p as f64).collect()
        }
        Prediction::Regression { predicted, .. } => predicted.clone(),
    };
    Ok(PyArray1::from_vec(py, values))
}

fn predict_proba_values<'py>(
    model: &dyn TrainedModel,
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let pred = model.predict(&to_array2(x)).map_err(smelt_err)?;
    match &pred {
        Prediction::Classification {
            probabilities: Some(probs),
            ..
        } => {
            let n = probs.len();
            let k = probs.first().map_or(0, |row| row.len());
            let flat: Vec<f64> = probs.iter().flatten().copied().collect();
            let arr = Array2::from_shape_vec((n, k), flat)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyArray2::from_owned_array(py, arr))
        }
        _ => Err(PyRuntimeError::new_err("No probabilities available")),
    }
}

fn not_fitted() -> PyErr {
    PyRuntimeError::new_err("Model not fitted. Call fit() first.")
}

// ── XGBoost ────────────────────────────────────────────────────────────

#[pyclass]
struct XGBoost {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    n_estimators: usize,
    max_depth: usize,
    learning_rate: f64,
    lambda: f64,
    alpha: f64,
    gamma: f64,
    subsample: f64,
    colsample_bytree: f64,
    seed: u64,
}

#[pymethods]
impl XGBoost {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=6, learning_rate=0.3, lambda_=1.0, alpha=0.0, gamma=0.0, subsample=1.0, colsample_bytree=1.0, seed=42))]
    fn new(
        n_estimators: usize,
        max_depth: usize,
        learning_rate: f64,
        lambda_: f64,
        alpha: f64,
        gamma: f64,
        subsample: f64,
        colsample_bytree: f64,
        seed: u64,
    ) -> Self {
        Self {
            trained: None,
            is_classif: false,
            n_estimators,
            max_depth,
            learning_rate,
            lambda: lambda_,
            alpha,
            gamma,
            subsample,
            colsample_bytree,
            seed,
        }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::XGBoost::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_learning_rate(self.learning_rate)
            .with_lambda(self.lambda)
            .with_alpha(self.alpha)
            .with_gamma(self.gamma)
            .with_subsample(self.subsample)
            .with_colsample_bytree(self.colsample_bytree)
            .with_seed(self.seed);
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        self.is_classif = is_classif;
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        predict_proba_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        Ok(self
            .trained
            .as_ref()
            .ok_or_else(not_fitted)?
            .feature_importance())
    }
}

// ── CatBoost ───────────────────────────────────────────────────────────

#[pyclass]
struct CatBoost {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    n_estimators: usize,
    depth: usize,
    learning_rate: f64,
    lambda: f64,
    seed: u64,
}

#[pymethods]
impl CatBoost {
    #[new]
    #[pyo3(signature = (n_estimators=100, depth=6, learning_rate=0.3, lambda_=1.0, seed=42))]
    fn new(n_estimators: usize, depth: usize, learning_rate: f64, lambda_: f64, seed: u64) -> Self {
        Self {
            trained: None,
            is_classif: false,
            n_estimators,
            depth,
            learning_rate,
            lambda: lambda_,
            seed,
        }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::CatBoost::new()
            .with_n_estimators(self.n_estimators)
            .with_depth(self.depth)
            .with_learning_rate(self.learning_rate)
            .with_lambda(self.lambda)
            .with_seed(self.seed);
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        self.is_classif = is_classif;
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        predict_proba_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }
}

// ── RandomForest ───────────────────────────────────────────────────────

#[pyclass]
struct RandomForest {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    n_estimators: usize,
    max_depth: usize,
    seed: u64,
}

#[pymethods]
impl RandomForest {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=10, seed=42))]
    fn new(n_estimators: usize, max_depth: usize, seed: u64) -> Self {
        Self {
            trained: None,
            is_classif: false,
            n_estimators,
            max_depth,
            seed,
        }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::RandomForest::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_seed(self.seed);
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        self.is_classif = is_classif;
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        predict_proba_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        Ok(self.trained.as_ref().ok_or_else(not_fitted)?.feature_importance())
    }
}

// ── DecisionTree ───────────────────────────────────────────────────────

#[pyclass]
struct DecisionTree {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    max_depth: usize,
}

#[pymethods]
impl DecisionTree {
    #[new]
    #[pyo3(signature = (max_depth=10))]
    fn new(max_depth: usize) -> Self {
        Self {
            trained: None,
            is_classif: false,
            max_depth,
        }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner =
            smelt_ml::prelude::DecisionTree::default().with_max_depth(self.max_depth);
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        self.is_classif = is_classif;
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }
}

// ── LogisticRegression ─────────────────────────────────────────────────

#[pyclass]
struct LogisticRegression {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
}

#[pymethods]
impl LogisticRegression {
    #[new]
    fn new() -> Self {
        Self { trained: None, is_classif: false }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::LogisticRegression::new();
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        self.is_classif = is_classif;
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        predict_proba_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }
}

// ── StandardScaler ─────────────────────────────────────────────────────

#[pyclass]
struct StandardScaler {
    inner: smelt_ml::prelude::StandardScaler,
}

#[pymethods]
impl StandardScaler {
    #[new]
    fn new() -> Self {
        Self {
            inner: smelt_ml::prelude::StandardScaler::new(),
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
        use smelt_ml::prelude::Transformer;
        self.inner.fit(&to_array2(x)).map_err(smelt_err)
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        use smelt_ml::prelude::Transformer;
        let r = self.inner.transform(&to_array2(x)).map_err(smelt_err)?;
        Ok(PyArray2::from_owned_array(py, r))
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        use smelt_ml::prelude::Transformer;
        let r = self.inner.fit_transform(&to_array2(x)).map_err(smelt_err)?;
        Ok(PyArray2::from_owned_array(py, r))
    }
}

// ── CrossValidation ────────────────────────────────────────────────────

#[pyclass]
struct CrossValidation {
    inner: smelt_ml::prelude::CrossValidation,
}

#[pymethods]
impl CrossValidation {
    #[new]
    #[pyo3(signature = (n_folds=5, seed=42))]
    fn new(n_folds: usize, seed: u64) -> Self {
        Self {
            inner: smelt_ml::prelude::CrossValidation::new(n_folds).with_seed(seed),
        }
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}

// ── SpatialBlockCV ─────────────────────────────────────────────────────

#[pyclass]
struct SpatialBlockCV {
    inner: smelt_ml::prelude::SpatialBlockCV,
}

/// Parse coords from numpy array (Nx2), list of tuples, or list of lists.
fn parse_coords(coords: &Bound<'_, PyAny>) -> PyResult<Vec<(f64, f64)>> {
    // Try numpy 2D array first (most common case)
    if let Ok(arr) = coords.extract::<PyReadonlyArray2<'_, f64>>() {
        let a = arr.as_array();
        if a.ncols() != 2 {
            return Err(PyRuntimeError::new_err(format!(
                "coords array must have 2 columns (X, Y), got {}",
                a.ncols()
            )));
        }
        return Ok(a.rows().into_iter().map(|r| (r[0], r[1])).collect());
    }
    // Fallback: list of (x, y) tuples
    if let Ok(tuples) = coords.extract::<Vec<(f64, f64)>>() {
        return Ok(tuples);
    }
    // Fallback: list of [x, y] lists
    if let Ok(lists) = coords.extract::<Vec<Vec<f64>>>() {
        let mut out = Vec::with_capacity(lists.len());
        for (i, v) in lists.iter().enumerate() {
            if v.len() != 2 {
                return Err(PyRuntimeError::new_err(format!(
                    "coords[{i}] must have 2 elements, got {}",
                    v.len()
                )));
            }
            out.push((v[0], v[1]));
        }
        return Ok(out);
    }
    Err(PyRuntimeError::new_err(
        "coords must be a numpy array (Nx2), list of (x, y) tuples, or list of [x, y] lists",
    ))
}

#[pymethods]
impl SpatialBlockCV {
    #[new]
    fn new(n_folds: usize, coords: &Bound<'_, PyAny>) -> PyResult<Self> {
        let parsed = parse_coords(coords)?;
        Ok(Self {
            inner: smelt_ml::prelude::SpatialBlockCV::new(n_folds, parsed),
        })
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}

// ── SpatialBufferCV ────────────────────────────────────────────────────

#[pyclass]
struct SpatialBufferCV {
    inner: smelt_ml::resample::SpatialBufferCV,
}

#[pymethods]
impl SpatialBufferCV {
    /// Buffered k-fold spatial CV. Training samples within `buffer_distance`
    /// of any test sample are excluded, reducing spatial autocorrelation leakage.
    ///
    /// For Spatial Leave-One-Out behaviour, set `n_folds = n_samples`.
    #[new]
    #[pyo3(signature = (n_folds, coords, buffer_distance, seed=42))]
    fn new(
        n_folds: usize,
        coords: &Bound<'_, PyAny>,
        buffer_distance: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let parsed = parse_coords(coords)?;
        Ok(Self {
            inner: smelt_ml::resample::SpatialBufferCV::new(n_folds, parsed, buffer_distance)
                .with_seed(seed),
        })
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}

// ── StratifiedCV ───────────────────────────────────────────────────────

#[pyclass]
struct StratifiedCV {
    inner: smelt_ml::prelude::StratifiedCV,
}

#[pymethods]
impl StratifiedCV {
    /// Stratified k-fold: each fold preserves the overall class proportions.
    /// `labels` are the classification target values (0-indexed class ids).
    #[new]
    #[pyo3(signature = (n_folds, labels, seed=42))]
    fn new(n_folds: usize, labels: Vec<usize>, seed: u64) -> Self {
        Self {
            inner: smelt_ml::prelude::StratifiedCV::new(n_folds, labels).with_seed(seed),
        }
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}

// ── GroupCV ────────────────────────────────────────────────────────────

#[pyclass]
struct GroupCV {
    inner: smelt_ml::prelude::GroupCV,
}

#[pymethods]
impl GroupCV {
    /// Group k-fold: every sample sharing a group id stays in the same
    /// fold, so a group never spans both train and test.
    #[new]
    #[pyo3(signature = (n_folds, groups, seed=42))]
    fn new(n_folds: usize, groups: Vec<usize>, seed: u64) -> Self {
        Self {
            inner: smelt_ml::prelude::GroupCV::new(n_folds, groups).with_seed(seed),
        }
    }

    fn splits(&self, n_samples: usize) -> PyResult<Vec<(Vec<usize>, Vec<usize>)>> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples).map_err(smelt_err)
    }
}

// ── Measures ───────────────────────────────────────────────────────────

#[pyfunction]
fn accuracy_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred_u: Vec<usize> = y_pred.iter().map(|&v| v as usize).collect();
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::Accuracy.score(&pred).map_err(smelt_err)
}

#[pyfunction]
fn rmse_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred = Prediction::regression_with_truth(y_pred, y_true);
    smelt_ml::prelude::Rmse.score(&pred).map_err(smelt_err)
}

#[pyfunction]
fn r2_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred = Prediction::regression_with_truth(y_pred, y_true);
    smelt_ml::prelude::RSquared.score(&pred).map_err(smelt_err)
}

#[pyfunction]
fn mae_score(y_true: Vec<f64>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred = Prediction::regression_with_truth(y_pred, y_true);
    smelt_ml::prelude::Mae.score(&pred).map_err(smelt_err)
}

#[pyfunction]
fn f1_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred_u: Vec<usize> = y_pred.iter().map(|&v| v as usize).collect();
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::F1Score.score(&pred).map_err(smelt_err)
}

#[pyfunction]
fn precision_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred_u: Vec<usize> = y_pred.iter().map(|&v| v as usize).collect();
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::Precision.score(&pred).map_err(smelt_err)
}

#[pyfunction]
fn recall_score(y_true: Vec<usize>, y_pred: Vec<f64>) -> PyResult<f64> {
    let pred_u: Vec<usize> = y_pred.iter().map(|&v| v as usize).collect();
    let pred = Prediction::classification_with_truth(pred_u, y_true);
    smelt_ml::prelude::Recall.score(&pred).map_err(smelt_err)
}

/// AUC-ROC score. Accepts y_proba as either:
/// - 2D: [[p0, p1], ...] (per-class probabilities)
/// - 1D: [p1, ...] (probability of positive class, sklearn-compatible)
#[pyfunction]
fn auc_roc_score(y_true: Vec<usize>, y_proba: &Bound<'_, PyAny>) -> PyResult<f64> {
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

// ── Stats ──────────────────────────────────────────────────────────────

#[pyclass]
#[derive(Clone)]
struct TestResult {
    #[pyo3(get)]
    test: String,
    #[pyo3(get)]
    statistic: f64,
    #[pyo3(get)]
    p_value: f64,
    #[pyo3(get)]
    significant: bool,
}

#[pymethods]
impl TestResult {
    fn __repr__(&self) -> String {
        format!(
            "TestResult(test='{}', statistic={:.4}, p_value={:.4}, significant={})",
            self.test, self.statistic, self.p_value, self.significant
        )
    }
}

#[pyfunction]
fn wilcoxon_signed_rank(a: Vec<f64>, b: Vec<f64>) -> PyResult<TestResult> {
    let r = smelt_ml::stats::wilcoxon_signed_rank(&a, &b).map_err(smelt_err)?;
    Ok(TestResult {
        test: r.test.to_string(),
        statistic: r.statistic,
        p_value: r.p_value,
        significant: r.significant,
    })
}

#[pyfunction]
#[pyo3(signature = (scores, confidence=0.95, n_bootstrap=10000, seed=42))]
fn bootstrap_ci(
    scores: Vec<f64>,
    confidence: f64,
    n_bootstrap: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let r = smelt_ml::stats::bootstrap_ci(&scores, confidence, n_bootstrap, seed);
    (r.estimate, r.lower, r.upper)
}

#[pyfunction]
fn sign_test(a: Vec<f64>, b: Vec<f64>) -> PyResult<TestResult> {
    let r = smelt_ml::stats::sign_test(&a, &b).map_err(smelt_err)?;
    Ok(TestResult {
        test: r.test.to_string(),
        statistic: r.statistic,
        p_value: r.p_value,
        significant: r.significant,
    })
}

// ── LightGBM ───────────────────────────────────────────────────────────

#[pyclass]
struct LightGBM {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    n_estimators: usize,
    num_leaves: usize,
    learning_rate: f64,
    max_depth: usize,
    seed: u64,
}

#[pymethods]
impl LightGBM {
    #[new]
    #[pyo3(signature = (n_estimators=100, num_leaves=31, learning_rate=0.1, max_depth=6, seed=42))]
    fn new(n_estimators: usize, num_leaves: usize, learning_rate: f64, max_depth: usize, seed: u64) -> Self {
        Self { trained: None, is_classif: false, n_estimators, num_leaves, learning_rate, max_depth, seed }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::LightGBM::new()
            .with_n_estimators(self.n_estimators)
            .with_num_leaves(self.num_leaves)
            .with_learning_rate(self.learning_rate)
            .with_max_depth(self.max_depth)
            .with_seed(self.seed);
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        self.is_classif = is_classif;
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }

    fn predict_proba<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        predict_proba_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        Ok(self.trained.as_ref().ok_or_else(not_fitted)?.feature_importance())
    }
}

// ── ExtraTrees ─────────────────────────────────────────────────────────

#[pyclass]
struct ExtraTrees {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    n_estimators: usize,
    max_depth: usize,
    seed: u64,
}

#[pymethods]
impl ExtraTrees {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=10, seed=42))]
    fn new(n_estimators: usize, max_depth: usize, seed: u64) -> Self {
        Self { trained: None, is_classif: false, n_estimators, max_depth, seed }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::ExtraTrees::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_seed(self.seed);
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        self.is_classif = is_classif;
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }
}

// ── KNearestNeighbors ──────────────────────────────────────────────────

#[pyclass]
struct KNearestNeighbors {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    k: usize,
}

#[pymethods]
impl KNearestNeighbors {
    #[new]
    #[pyo3(signature = (k=5))]
    fn new(k: usize) -> Self {
        Self { trained: None, is_classif: false, k }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::KNearestNeighbors::new(self.k);
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        self.is_classif = is_classif;
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }

    fn predict_proba<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        predict_proba_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }
}

// ── GaussianNB ─────────────────────────────────────────────────────────

#[pyclass]
struct GaussianNB {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
}

#[pymethods]
impl GaussianNB {
    #[new]
    fn new() -> Self {
        Self { trained: None, is_classif: false }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::GaussianNB::new();
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        self.is_classif = is_classif;
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }

    fn predict_proba<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        predict_proba_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }
}

// ── Ridge ──────────────────────────────────────────────────────────────

#[pyclass]
struct Ridge {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    alpha: f64,
}

#[pymethods]
impl Ridge {
    #[new]
    #[pyo3(signature = (alpha=1.0))]
    fn new(alpha: f64) -> Self {
        Self { trained: None, is_classif: false, alpha }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::Ridge::new(self.alpha);
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        self.is_classif = is_classif;
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }
}

// ── LinearRegression ───────────────────────────────────────────────────

#[pyclass]
struct LinearRegression {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
}

#[pymethods]
impl LinearRegression {
    #[new]
    fn new() -> Self {
        Self { trained: None, is_classif: false }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::LinearRegression::new();
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        self.is_classif = is_classif;
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }
}

// ── GeoXGBoost ─────────────────────────────────────────────────────────

#[pyclass]
struct GeoXGBoost {
    trained: Option<smelt_ml::prelude::TrainedGeoXGBoost>,
    bandwidth: usize,
    n_estimators: usize,
    max_depth: usize,
    learning_rate: f64,
    lambda: f64,
    alpha: Option<f64>,
    seed: u64,
}

#[pymethods]
impl GeoXGBoost {
    /// Geographical-XGBoost for spatially-local regression (Grekousis, 2025).
    ///
    /// Args:
    ///     bandwidth: number of nearest neighbours for adaptive bi-square kernel.
    ///     n_estimators, max_depth, learning_rate, lambda_: XGBoost hyperparameters
    ///         used for both global and local models.
    ///     alpha: None (default) for adaptive blending, 1.0 for pure local,
    ///         0.0 for pure global.
    ///     seed: random seed.
    #[new]
    #[pyo3(signature = (bandwidth=30, n_estimators=100, max_depth=6, learning_rate=0.3, lambda_=1.0, alpha=None, seed=42))]
    fn new(
        bandwidth: usize,
        n_estimators: usize,
        max_depth: usize,
        learning_rate: f64,
        lambda_: f64,
        alpha: Option<f64>,
        seed: u64,
    ) -> Self {
        Self {
            trained: None,
            bandwidth,
            n_estimators,
            max_depth,
            learning_rate,
            lambda: lambda_,
            alpha,
            seed,
        }
    }

    /// Train the model. `coords` is an (N, 2) array-like of (x, y) per sample.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: Vec<f64>,
        coords: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let parsed = parse_coords(coords)?;
        let features = to_array2(x);
        if parsed.len() != features.nrows() {
            return Err(PyRuntimeError::new_err(format!(
                "coords length ({}) must match number of samples ({})",
                parsed.len(),
                features.nrows()
            )));
        }
        let task = smelt_ml::task::RegressionTask::new("gxgb", features, y).map_err(smelt_err)?;
        let mut learner = smelt_ml::prelude::GeoXGBoost::new(parsed)
            .with_bandwidth(self.bandwidth)
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_learning_rate(self.learning_rate)
            .with_lambda(self.lambda)
            .with_seed(self.seed);
        if let Some(a) = self.alpha {
            learner = learner.with_alpha(a);
        }
        let trained = py
            .allow_threads(|| learner.train_geo(&task))
            .map_err(smelt_err)?;
        self.trained = Some(trained);
        Ok(())
    }

    /// Select the optimal bandwidth by minimising the leave-one-out CV criterion
    /// of Grekousis (2025, Eq. 11).
    ///
    /// For every candidate bandwidth, each location is predicted by a *local*
    /// model fit on its neighbours excluding itself; the criterion is the
    /// leave-one-out RMSE. The bandwidth minimising it is returned and stored on
    /// this estimator, so a subsequent `fit(...)` uses it automatically.
    ///
    /// This is a property of the local model only: the global model and `alpha`
    /// are not involved, because bandwidth is tuned before the ensemble step.
    ///
    /// Args:
    ///     x, y, coords: training data (same as `fit`).
    ///     candidates: list of bandwidths to try. Defaults to
    ///         [30, 50, 100, 150, 200, 300, 400] (filtered to values < n).
    ///
    /// Returns:
    ///     dict with keys ``best`` (int), ``bandwidths`` (list[int]) and
    ///     ``cv`` (list[float], the LOO criterion), aligned by index.
    #[pyo3(signature = (x, y, coords, candidates=None))]
    fn select_bandwidth(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: Vec<f64>,
        coords: &Bound<'_, PyAny>,
        candidates: Option<Vec<usize>>,
    ) -> PyResult<PyObject> {
        let parsed = parse_coords(coords)?;
        let features = to_array2(x);
        let n = features.nrows();
        if parsed.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "coords length ({}) must match number of samples ({})",
                parsed.len(),
                n
            )));
        }
        let grid: Vec<usize> = candidates
            .unwrap_or_else(|| vec![30, 50, 100, 150, 200, 300, 400])
            .into_iter()
            .filter(|&bw| bw > 0 && bw < n)
            .collect();
        if grid.is_empty() {
            return Err(PyRuntimeError::new_err(
                "no valid candidate bandwidths (all >= number of samples?)",
            ));
        }

        let task = smelt_ml::task::RegressionTask::new("gxgb", features, y).map_err(smelt_err)?;
        let learner = smelt_ml::prelude::GeoXGBoost::new(parsed)
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_learning_rate(self.learning_rate)
            .with_lambda(self.lambda)
            .with_seed(self.seed);
        let sel = py
            .allow_threads(|| learner.select_bandwidth(&task, &grid))
            .map_err(smelt_err)?;

        // Store the selected bandwidth so the next fit() uses it.
        self.bandwidth = sel.best;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("best", sel.best)?;
        let bws: Vec<usize> = sel.scores.iter().map(|&(bw, _)| bw).collect();
        let cv: Vec<f64> = sel.scores.iter().map(|&(_, r)| r).collect();
        dict.set_item("bandwidths", bws)?;
        dict.set_item("cv", cv)?;
        Ok(dict.into())
    }

    /// Predict. If `coords` is provided, uses per-sample nearest local model
    /// (spatial prediction); otherwise falls back to in-sample / global model.
    #[pyo3(signature = (x, coords=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        coords: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        use smelt_ml::learner::TrainedModel;
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let features = to_array2(x);
        let pred = if let Some(c) = coords {
            let new_coords = parse_coords(c)?;
            if new_coords.len() != features.nrows() {
                return Err(PyRuntimeError::new_err(format!(
                    "coords length ({}) must match number of samples ({})",
                    new_coords.len(),
                    features.nrows()
                )));
            }
            model.predict_spatial(&features, &new_coords).map_err(smelt_err)?
        } else {
            model.predict(&features).map_err(smelt_err)?
        };
        let values: Vec<f64> = match &pred {
            Prediction::Regression { predicted, .. } => predicted.clone(),
            _ => return Err(PyRuntimeError::new_err("Expected regression prediction")),
        };
        Ok(PyArray1::from_vec(py, values))
    }

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        use smelt_ml::learner::TrainedModel;
        Ok(self
            .trained
            .as_ref()
            .ok_or_else(not_fitted)?
            .feature_importance())
    }

    /// Per-location local-model feature importances, for mapping spatial
    /// non-stationarity (how each predictor's influence varies across space).
    ///
    /// Returns a dict with:
    ///   - ``coords``: (N, 2) float array of training (x, y) coordinates,
    ///   - ``importances``: list of length N; entry i is a dict
    ///     {feature_name: gain} for location i's local model, or None where the
    ///     neighbourhood was too small (global-model fallback).
    /// Feature names are the internal x0, x1, ... order.
    fn local_feature_importances<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let coords = model.coords();
        let imps = model.local_importances();

        let n = coords.len();
        let flat: Vec<f64> = coords.iter().flat_map(|&(x, y)| [x, y]).collect();
        let coord_arr = ndarray::Array2::from_shape_vec((n, 2), flat)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let imp_list = pyo3::types::PyList::empty(py);
        for entry in imps {
            match entry {
                Some(v) => {
                    let d = pyo3::types::PyDict::new(py);
                    for (name, gain) in v {
                        d.set_item(name, gain)?;
                    }
                    imp_list.append(d)?;
                }
                None => imp_list.append(py.None())?,
            }
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coords", PyArray2::from_owned_array(py, coord_arr))?;
        dict.set_item("importances", imp_list)?;
        Ok(dict.into())
    }

    #[getter]
    fn supports_classification(&self) -> bool { false }

    #[getter]
    fn supports_regression(&self) -> bool { true }

    /// SHAP values (interventional). Uses the global model internally.
    #[pyo3(signature = (x, y, n_background=50, feature_names=None))]
    fn shap_values<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        n_background: usize,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        shap_impl(py, model, false, x, y, n_background, feature_names, 0)
    }

    /// Permutation importance.
    #[pyo3(signature = (x, y, metric="rmse", n_repeats=5, seed=42, feature_names=None))]
    fn permutation_importance<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        metric: &str,
        n_repeats: usize,
        seed: u64,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        perm_importance_impl(py, model, false, x, y, metric, n_repeats, seed, feature_names)
    }

    /// Split conformal prediction intervals. Calibration data may include
    /// their own `coords` for spatially-aware predictions on the calibration set;
    /// if omitted, the global model is used for calibration.
    #[pyo3(signature = (x_cal, y_cal, x_test, alpha=0.1))]
    fn conformal_predict<'py>(
        &self,
        py: Python<'py>,
        x_cal: PyReadonlyArray2<'_, f64>,
        y_cal: Vec<f64>,
        x_test: PyReadonlyArray2<'_, f64>,
        alpha: f64,
    ) -> PyResult<PyObject> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        conformal_predict_impl(py, model, x_cal, y_cal, x_test, alpha)
    }
}

// ── SHAP + Permutation Importance (shared helpers) ────────────────────

fn resolve_measure(metric: &str) -> PyResult<Box<dyn Measure>> {
    use smelt_ml::measure::*;
    match metric {
        "rmse" => Ok(Box::new(Rmse)),
        "mae" => Ok(Box::new(Mae)),
        "r2" => Ok(Box::new(RSquared)),
        "mape" => Ok(Box::new(Mape)),
        "accuracy" => Ok(Box::new(Accuracy)),
        "f1" => Ok(Box::new(F1Score)),
        "precision" => Ok(Box::new(Precision)),
        "recall" => Ok(Box::new(Recall)),
        "logloss" => Ok(Box::new(LogLoss)),
        "auc" => Ok(Box::new(AucRoc)),
        _ => Err(PyRuntimeError::new_err(format!("Unknown metric: {metric}"))),
    }
}

fn shap_impl<'py>(
    py: Python<'py>,
    model: &dyn TrainedModel,
    is_classif: bool,
    x: PyReadonlyArray2<'_, f64>,
    y: &Bound<'_, PyAny>,
    n_background: usize,
    feature_names: Option<Vec<String>>,
    target_class: usize,
) -> PyResult<PyObject> {
    let features = to_array2(x);
    let n_feat = features.ncols();
    let names = feature_names.unwrap_or_else(|| (0..n_feat).map(|i| format!("f{i}")).collect());

    let result = if is_classif {
        let target = extract_class_labels(y)?;
        let mut task = smelt_ml::task::ClassificationTask::new("shap", features, target)
            .map_err(smelt_err)?;
        task = task.with_feature_names(names.clone()).map_err(smelt_err)?;
        py.allow_threads(|| {
            smelt_ml::importance::shap::tree_shap_classif(model, &task, n_background, target_class)
        })
        .map_err(smelt_err)?
    } else {
        let target: Vec<f64> = y.extract()?;
        let mut task = smelt_ml::task::RegressionTask::new("shap", features, target)
            .map_err(smelt_err)?;
        task = task.with_feature_names(names.clone()).map_err(smelt_err)?;
        py.allow_threads(|| smelt_ml::importance::shap::tree_shap_regress(model, &task, n_background))
            .map_err(smelt_err)?
    };

    // Convert to Python dict
    let dict = pyo3::types::PyDict::new(py);

    // SHAP values as 2D numpy array
    let n = result.explanations.len();
    let p = if n > 0 { result.explanations[0].values.len() } else { 0 };
    let flat: Vec<f64> = result.explanations.iter()
        .flat_map(|e| e.values.iter().copied()).collect();
    let arr = ndarray::Array2::from_shape_vec((n, p), flat)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    dict.set_item("values", PyArray2::from_owned_array(py, arr))?;

    dict.set_item("base_value",
        result.explanations.first().map(|e| e.base_value).unwrap_or(0.0))?;
    dict.set_item("feature_names", &names)?;

    let imp: Vec<(String, f64)> = result.global_importance;
    dict.set_item("global_importance", imp)?;

    Ok(dict.into_pyobject(py)?.into_any().unbind())
}

fn perm_importance_impl<'py>(
    py: Python<'py>,
    model: &dyn TrainedModel,
    is_classif: bool,
    x: PyReadonlyArray2<'_, f64>,
    y: &Bound<'_, PyAny>,
    metric: &str,
    n_repeats: usize,
    seed: u64,
    feature_names: Option<Vec<String>>,
) -> PyResult<PyObject> {
    let features = to_array2(x);
    let n_feat = features.ncols();
    let names = feature_names.unwrap_or_else(|| (0..n_feat).map(|i| format!("f{i}")).collect());
    let measure = resolve_measure(metric)?;

    let importances = if is_classif {
        let target = extract_class_labels(y)?;
        let mut task = smelt_ml::task::ClassificationTask::new("perm", features, target)
            .map_err(smelt_err)?;
        task = task.with_feature_names(names).map_err(smelt_err)?;
        py.allow_threads(|| {
            smelt_ml::importance::permutation_importance_classif(
                model, &task, &*measure, n_repeats, seed,
            )
        })
        .map_err(smelt_err)?
    } else {
        let target: Vec<f64> = y.extract()?;
        let mut task = smelt_ml::task::RegressionTask::new("perm", features, target)
            .map_err(smelt_err)?;
        task = task.with_feature_names(names).map_err(smelt_err)?;
        py.allow_threads(|| {
            smelt_ml::importance::permutation_importance_regress(
                model, &task, &*measure, n_repeats, seed,
            )
        })
        .map_err(smelt_err)?
    };

    // Return list of dicts
    let list = pyo3::types::PyList::empty(py);
    for fi in &importances {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("feature", &fi.feature)?;
        d.set_item("importance", fi.importance)?;
        d.set_item("std_dev", fi.std_dev)?;
        list.append(d)?;
    }
    Ok(list.into_pyobject(py)?.into_any().unbind())
}

// ── Conformal prediction helper ────────────────────────────────────────

fn conformal_predict_impl<'py>(
    py: Python<'py>,
    model: &dyn TrainedModel,
    x_cal: PyReadonlyArray2<'_, f64>,
    y_cal: Vec<f64>,
    x_test: PyReadonlyArray2<'_, f64>,
    alpha: f64,
) -> PyResult<PyObject> {
    let cal_features = to_array2(x_cal);
    let test_features = to_array2(x_test);

    let cf = smelt_ml::conformal::ConformalRegressor::calibrate(
        model, &cal_features, &y_cal, alpha,
    )
    .map_err(smelt_err)?;

    let intervals = cf.predict(&test_features).map_err(smelt_err)?;

    let n = intervals.len();
    let mut preds = Vec::with_capacity(n);
    let mut lower = Vec::with_capacity(n);
    let mut upper = Vec::with_capacity(n);
    for iv in &intervals {
        preds.push(iv.prediction);
        lower.push(iv.lower);
        upper.push(iv.upper);
    }

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("predictions", PyArray1::from_vec(py, preds))?;
    dict.set_item("lower", PyArray1::from_vec(py, lower))?;
    dict.set_item("upper", PyArray1::from_vec(py, upper))?;
    dict.set_item("interval_width", cf.interval_width())?;
    dict.set_item("alpha", alpha)?;
    Ok(dict.into_pyobject(py)?.into_any().unbind())
}

// ── Macro: add shap_values + permutation_importance to all learners ───

macro_rules! add_explain_methods {
    ($($name:ident),+ $(,)?) => {
        $(
            #[pymethods]
            impl $name {
                /// Compute SHAP values for each sample.
                #[pyo3(signature = (x, y, n_background=50, feature_names=None, target_class=1))]
                fn shap_values<'py>(
                    &self,
                    py: Python<'py>,
                    x: PyReadonlyArray2<'_, f64>,
                    y: &Bound<'_, PyAny>,
                    n_background: usize,
                    feature_names: Option<Vec<String>>,
                    target_class: usize,
                ) -> PyResult<PyObject> {
                    let model = self.trained.as_deref().ok_or_else(not_fitted)?;
                    shap_impl(py, model, self.is_classif, x, y, n_background, feature_names, target_class)
                }

                /// Compute permutation importance.
                #[pyo3(signature = (x, y, metric="rmse", n_repeats=5, seed=42, feature_names=None))]
                fn permutation_importance<'py>(
                    &self,
                    py: Python<'py>,
                    x: PyReadonlyArray2<'_, f64>,
                    y: &Bound<'_, PyAny>,
                    metric: &str,
                    n_repeats: usize,
                    seed: u64,
                    feature_names: Option<Vec<String>>,
                ) -> PyResult<PyObject> {
                    let model = self.trained.as_deref().ok_or_else(not_fitted)?;
                    perm_importance_impl(py, model, self.is_classif, x, y, metric, n_repeats, seed, feature_names)
                }

                /// Split conformal prediction intervals with guaranteed (1-alpha) coverage.
                ///
                /// Args:
                ///     x_cal, y_cal: calibration data (held out from training)
                ///     x_test: features to predict on
                ///     alpha: miscoverage level (default 0.1 → 90% coverage)
                ///
                /// Returns dict with: "predictions", "lower", "upper" (numpy arrays),
                /// "interval_width" (float), "alpha" (float).
                #[pyo3(signature = (x_cal, y_cal, x_test, alpha=0.1))]
                fn conformal_predict<'py>(
                    &self,
                    py: Python<'py>,
                    x_cal: PyReadonlyArray2<'_, f64>,
                    y_cal: Vec<f64>,
                    x_test: PyReadonlyArray2<'_, f64>,
                    alpha: f64,
                ) -> PyResult<PyObject> {
                    let model = self.trained.as_deref().ok_or_else(not_fitted)?;
                    if self.is_classif {
                        return Err(PyRuntimeError::new_err(
                            "conformal_predict is only available for regression models",
                        ));
                    }
                    conformal_predict_impl(py, model, x_cal, y_cal, x_test, alpha)
                }
            }
        )+
    };
}

add_explain_methods!(
    XGBoost, CatBoost, LightGBM, RandomForest, ExtraTrees,
    DecisionTree, LogisticRegression, LinearRegression, Ridge,
    KNearestNeighbors, GaussianNB,
);

// ── Macro: declare task-type support flags per learner ────────────────

macro_rules! declare_support {
    ($name:ident, classif = $c:expr, regress = $r:expr) => {
        #[pymethods]
        impl $name {
            /// Whether this learner can train on classification targets (integer y).
            #[getter]
            fn supports_classification(&self) -> bool { $c }

            /// Whether this learner can train on regression targets (continuous y).
            #[getter]
            fn supports_regression(&self) -> bool { $r }
        }
    };
}

declare_support!(XGBoost,            classif = true,  regress = true);
declare_support!(CatBoost,           classif = true,  regress = true);
declare_support!(LightGBM,           classif = true,  regress = true);
declare_support!(RandomForest,       classif = true,  regress = true);
declare_support!(ExtraTrees,         classif = true,  regress = true);
declare_support!(DecisionTree,       classif = true,  regress = true);
declare_support!(KNearestNeighbors,  classif = true,  regress = true);
declare_support!(LogisticRegression, classif = true,  regress = false);
declare_support!(GaussianNB,         classif = true,  regress = false);
declare_support!(LinearRegression,   classif = false, regress = true);
declare_support!(Ridge,              classif = false, regress = true);

// ── RFE ───────────────────────────────────────────────────────────────

// ── BayesianOptimizer ──────────────────────────────────────────────────

fn make_learner_factory(
    learner_type: &str,
) -> PyResult<Box<dyn Fn(&smelt_ml::tuning::ParamSet) -> Box<dyn smelt_ml::learner::Learner> + Send + Sync>>
{
    use smelt_ml::prelude::*;
    type L = Box<dyn smelt_ml::learner::Learner>;
    type PS = smelt_ml::tuning::ParamSet;

    fn get(p: &PS, k: &str, def: f64) -> f64 {
        p.get(k).copied().unwrap_or(def)
    }

    match learner_type {
        "xgboost" => Ok(Box::new(|p: &PS| -> L {
            Box::new(XGBoost::new()
                .with_n_estimators(get(p, "n_estimators", 100.0) as usize)
                .with_max_depth(get(p, "max_depth", 6.0) as usize)
                .with_learning_rate(get(p, "learning_rate", 0.3))
                .with_lambda(get(p, "lambda", 1.0))
                .with_alpha(get(p, "alpha", 0.0))
                .with_gamma(get(p, "gamma", 0.0))
                .with_subsample(get(p, "subsample", 1.0))
                .with_colsample_bytree(get(p, "colsample_bytree", 1.0)))
        })),
        "catboost" => Ok(Box::new(|p: &PS| -> L {
            Box::new(CatBoost::new()
                .with_n_estimators(get(p, "n_estimators", 100.0) as usize)
                .with_depth(get(p, "depth", 6.0) as usize)
                .with_learning_rate(get(p, "learning_rate", 0.3))
                .with_lambda(get(p, "lambda", 1.0)))
        })),
        "lightgbm" => Ok(Box::new(|p: &PS| -> L {
            Box::new(LightGBM::new()
                .with_n_estimators(get(p, "n_estimators", 100.0) as usize)
                .with_num_leaves(get(p, "num_leaves", 31.0) as usize)
                .with_learning_rate(get(p, "learning_rate", 0.1))
                .with_max_depth(get(p, "max_depth", 6.0) as usize))
        })),
        "random_forest" | "rf" => Ok(Box::new(|p: &PS| -> L {
            Box::new(RandomForest::new()
                .with_n_estimators(get(p, "n_estimators", 100.0) as usize)
                .with_max_depth(get(p, "max_depth", 10.0) as usize))
        })),
        "extra_trees" | "et" => Ok(Box::new(|p: &PS| -> L {
            Box::new(ExtraTrees::new()
                .with_n_estimators(get(p, "n_estimators", 100.0) as usize)
                .with_max_depth(get(p, "max_depth", 10.0) as usize))
        })),
        "decision_tree" | "dt" => Ok(Box::new(|p: &PS| -> L {
            Box::new(DecisionTree::new()
                .with_max_depth(get(p, "max_depth", 10.0) as usize))
        })),
        "ridge" => Ok(Box::new(|p: &PS| -> L {
            Box::new(Ridge::new(get(p, "alpha", 1.0)))
        })),
        "knn" => Ok(Box::new(|p: &PS| -> L {
            Box::new(KNearestNeighbors::new(get(p, "k", 5.0) as usize))
        })),
        _ => Err(PyRuntimeError::new_err(format!("Unknown learner type: {learner_type}"))),
    }
}

fn build_param_space(dict: &Bound<'_, PyAny>) -> PyResult<smelt_ml::tuning::ParamSpace> {
    use smelt_ml::tuning::{ParamDistribution, ParamSpace};

    let py_dict: &Bound<'_, pyo3::types::PyDict> = dict.downcast()
        .map_err(|_| PyRuntimeError::new_err("param_space must be a dict"))?;

    let mut space = ParamSpace::new();

    for (key, val) in py_dict.iter() {
        let name: String = key.extract()?;

        // Accept dict format: {"type": "uniform", "low": 0.1, "high": 1.0}
        // Or tuple format: (low, high) → uniform
        // Or list format: [1, 2, 3] → choice
        if let Ok(inner_dict) = val.downcast::<pyo3::types::PyDict>() {
            let dtype: String = inner_dict.get_item("type")?
                .ok_or_else(|| PyRuntimeError::new_err(format!("Missing 'type' for param '{name}'")))?
                .extract()?;
            let required = |field: &str| -> PyResult<Bound<'_, PyAny>> {
                inner_dict
                    .get_item(field)?
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "param '{name}' of type '{dtype}' requires '{field}'"
                        ))
                    })
            };
            match dtype.as_str() {
                "uniform" => {
                    let low: f64 = required("low")?.extract()?;
                    let high: f64 = required("high")?.extract()?;
                    space.insert(name, ParamDistribution::Uniform(low, high));
                }
                "log_uniform" | "loguniform" => {
                    let low: f64 = required("low")?.extract()?;
                    let high: f64 = required("high")?.extract()?;
                    space.insert(name, ParamDistribution::LogUniform(low, high));
                }
                "choice" => {
                    let choices: Vec<f64> = required("values")?.extract()?;
                    space.insert(name, ParamDistribution::Choice(choices));
                }
                _ => return Err(PyRuntimeError::new_err(format!("Unknown param type: {dtype}"))),
            }
        } else if let Ok(tup) = val.extract::<(f64, f64)>() {
            // Shorthand: (low, high) → uniform
            space.insert(name, ParamDistribution::Uniform(tup.0, tup.1));
        } else if let Ok(choices) = val.extract::<Vec<f64>>() {
            // Shorthand: [1, 2, 3] → choice
            space.insert(name, ParamDistribution::Choice(choices));
        } else {
            return Err(PyRuntimeError::new_err(
                format!("Invalid param spec for '{name}'. Use dict, tuple (low, high), or list [choices]"),
            ));
        }
    }

    Ok(space)
}

#[pyclass]
#[pyo3(name = "BayesianOptimizer")]
struct PyBayesianOptimizer {
    n_iter: usize,
    n_initial: usize,
    seed: u64,
}

#[pymethods]
impl PyBayesianOptimizer {
    #[new]
    #[pyo3(signature = (n_iter=30, n_initial=5, seed=42))]
    fn new(n_iter: usize, n_initial: usize, seed: u64) -> Self {
        Self { n_iter, n_initial, seed }
    }

    /// Optimize hyperparameters using Bayesian TPE.
    ///
    /// Args:
    ///     learner_type: "xgboost", "rf", "catboost", "lightgbm", "dt", "ridge", "knn"
    ///     param_space: dict of param → spec. Specs can be:
    ///         - (low, high) → uniform distribution
    ///         - [v1, v2, v3] → choice
    ///         - {"type": "uniform"/"log_uniform"/"choice", "low": ..., "high": ..., "values": [...]}
    ///     x, y: training data
    ///     metric: "rmse", "r2", "accuracy", etc.
    ///     n_folds: cross-validation folds
    ///     cv_seed: seed for CV splits
    #[pyo3(signature = (learner_type, param_space, x, y, metric="rmse", n_folds=5, cv_seed=42))]
    fn optimize<'py>(
        &self,
        py: Python<'py>,
        learner_type: &str,
        param_space: &Bound<'_, PyAny>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        metric: &str,
        n_folds: usize,
        cv_seed: u64,
    ) -> PyResult<PyObject> {
        let factory = make_learner_factory(learner_type)?;
        let space = build_param_space(param_space)?;
        let measure = resolve_measure(metric)?;
        let cv = smelt_ml::resample::CrossValidation::new(n_folds).with_seed(cv_seed);

        let bo = smelt_ml::tuning::BayesianOptimizer::new(
            move |params| factory(params),
            space,
        )
        .with_n_iter(self.n_iter)
        .with_n_initial(self.n_initial)
        .with_seed(self.seed);

        let features = to_array2(x);

        let result = if is_integer(y) {
            let target = extract_class_labels(y)?;
            let task = smelt_ml::task::ClassificationTask::new("bo", features, target)
                .map_err(smelt_err)?;
            py.allow_threads(|| bo.tune_classif(&task, &cv, &*measure))
                .map_err(smelt_err)?
        } else {
            let target: Vec<f64> = y.extract()?;
            let task = smelt_ml::task::RegressionTask::new("bo", features, target)
                .map_err(smelt_err)?;
            py.allow_threads(|| bo.tune_regress(&task, &cv, &*measure))
                .map_err(smelt_err)?
        };

        // Convert TuneResult to Python dict, casting integer params to int
        let dict = pyo3::types::PyDict::new(py);

        let bp = pyo3::types::PyDict::new(py);
        for (k, v) in &result.best_params {
            set_param(&bp, k, *v)?;
        }
        dict.set_item("best_params", bp)?;
        dict.set_item("best_score", result.best_score)?;
        dict.set_item("measure", &result.measure_id)?;

        let history = pyo3::types::PyList::empty(py);
        for (params, score) in &result.all_results {
            let pd = pyo3::types::PyDict::new(py);
            for (k, v) in params {
                set_param(&pd, k, *v)?;
            }
            let tup = pyo3::types::PyTuple::new(py, &[pd.as_any(), score.into_pyobject(py)?.as_any()])?;
            history.append(tup)?;
        }
        dict.set_item("all_results", history)?;

        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }
}

/// Param names that are conceptually integer-valued and should be rounded.
fn is_integer_param(name: &str) -> bool {
    matches!(
        name,
        "n_estimators"
            | "max_depth"
            | "depth"
            | "num_leaves"
            | "k"
            | "min_samples_split"
            | "min_samples_leaf"
            | "n_features"
            | "seed"
            | "random_state"
    )
}

fn set_param(dict: &Bound<'_, pyo3::types::PyDict>, name: &str, value: f64) -> PyResult<()> {
    if is_integer_param(name) {
        dict.set_item(name, value.round() as i64)
    } else {
        dict.set_item(name, value)
    }
}

fn make_rfe_factory(learner_type: &str) -> PyResult<Box<dyn Fn() -> Box<dyn smelt_ml::learner::Learner> + Send + Sync>> {
    match learner_type {
        "decision_tree" => Ok(Box::new(|| Box::new(smelt_ml::prelude::DecisionTree::default()) as Box<dyn smelt_ml::learner::Learner>)),
        "random_forest" => Ok(Box::new(|| Box::new(smelt_ml::prelude::RandomForest::new()) as Box<dyn smelt_ml::learner::Learner>)),
        "extra_trees" => Ok(Box::new(|| Box::new(smelt_ml::prelude::ExtraTrees::new()) as Box<dyn smelt_ml::learner::Learner>)),
        "xgboost" => Ok(Box::new(|| Box::new(smelt_ml::prelude::XGBoost::new()) as Box<dyn smelt_ml::learner::Learner>)),
        "ridge" => Ok(Box::new(|| Box::new(smelt_ml::prelude::Ridge::new(1.0)) as Box<dyn smelt_ml::learner::Learner>)),
        _ => Err(PyRuntimeError::new_err(format!("Unknown learner for RFE: {learner_type}"))),
    }
}

#[pyfunction]
#[pyo3(signature = (x, y, learner_type="decision_tree", n_features=5, feature_names=None))]
fn rfe<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    y: &Bound<'_, PyAny>,
    learner_type: &str,
    n_features: usize,
    feature_names: Option<Vec<String>>,
) -> PyResult<PyObject> {
    use smelt_ml::preprocess::Transformer;

    let factory = make_rfe_factory(learner_type)?;
    let features = to_array2(x);
    let n_feat = features.ncols();

    let is_classif = is_integer(y);
    let mut selector = if is_classif {
        smelt_ml::preprocess::RFE::classif(move || factory(), n_features)
    } else {
        smelt_ml::preprocess::RFE::regress(move || factory(), n_features)
    };

    let target_f64: Vec<f64> = if is_classif {
        let t: Vec<i64> = y.extract()?;
        t.into_iter().map(|v| v as f64).collect()
    } else {
        y.extract()?
    };

    py.allow_threads(|| selector.fit_supervised(&features, &target_f64))
        .map_err(smelt_err)?;
    let indices = selector
        .selected_indices()
        .ok_or_else(|| PyRuntimeError::new_err("RFE selector was not fitted"))?;

    let names: Vec<String> = feature_names.unwrap_or_else(|| (0..n_feat).map(|i| format!("f{i}")).collect());
    let result: Vec<(String, usize)> = indices.iter().map(|&i| (names[i].clone(), i)).collect();
    Ok(result.into_pyobject(py)?.into_any().unbind())
}

// ── Feature Selection Filters ──────────────────────────────────────────

fn run_filter(
    method: &str,
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: &Bound<'_, PyAny>,
    feature_names: Vec<String>,
    k: usize,
) -> PyResult<PyObject> {
    use smelt_ml::preprocess::filter::{
        AnovaFFilter, CmimFilter, CorrelationFilter, Filter, InformationGainFilter, JmiFilter,
        JmimFilter, MrmrFilter, MutualInfoFilter, ReliefFilter, VarianceFilter,
    };

    let features = to_array2(x);
    let n_feat = features.ncols();
    let k = k.min(n_feat);

    let target: Vec<f64> = y.extract()?;

    // Get raw per-feature scores (higher = better)
    let scores: Vec<f64> = match method {
        "variance" => VarianceFilter.score(&features, &target),
        "correlation" => CorrelationFilter.score(&features, &target),
        "anova_f" => AnovaFFilter.score(&features, &target),
        "information_gain" => InformationGainFilter.score(&features, &target),
        "mutual_information" => MutualInfoFilter.score(&features, &target),
        "mrmr" => MrmrFilter.score(&features, &target),
        "jmi" => JmiFilter.score(&features, &target),
        "jmim" => JmimFilter.score(&features, &target),
        "cmim" => CmimFilter.score(&features, &target),
        "relief" => ReliefFilter.score(&features, &target),
        _ => return Err(PyRuntimeError::new_err(format!("Unknown filter: {method}"))),
    };

    let names: Vec<String> = if feature_names.len() == n_feat {
        feature_names
    } else {
        (0..n_feat).map(|i| format!("f{i}")).collect()
    };

    // Sort by score descending (higher = more important), take top k
    let mut ranked: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let result: Vec<(String, f64)> = ranked
        .into_iter()
        .take(k)
        .map(|(i, s)| (names[i].clone(), s))
        .collect();
    Ok(result.into_pyobject(py)?.into_any().unbind())
}

macro_rules! filter_fn {
    ($name:ident, $method:expr) => {
        #[pyfunction]
        #[pyo3(signature = (x, y, feature_names, k=15))]
        fn $name<'py>(
            py: Python<'py>,
            x: PyReadonlyArray2<'_, f64>,
            y: &Bound<'_, PyAny>,
            feature_names: Vec<String>,
            k: usize,
        ) -> PyResult<PyObject> {
            run_filter($method, py, x, y, feature_names, k)
        }
    };
}

filter_fn!(filter_variance, "variance");
filter_fn!(filter_correlation, "correlation");
filter_fn!(filter_anova_f, "anova_f");
filter_fn!(filter_information_gain, "information_gain");
filter_fn!(filter_mutual_information, "mutual_information");
filter_fn!(filter_mrmr, "mrmr");
filter_fn!(filter_jmi, "jmi");
filter_fn!(filter_jmim, "jmim");
filter_fn!(filter_cmim, "cmim");
filter_fn!(filter_relief, "relief");

// ── Module ─────────────────────────────────────────────────────────────

#[pymodule]
fn _smelt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Learners
    m.add_class::<XGBoost>()?;
    m.add_class::<CatBoost>()?;
    m.add_class::<LightGBM>()?;
    m.add_class::<RandomForest>()?;
    m.add_class::<ExtraTrees>()?;
    m.add_class::<DecisionTree>()?;
    m.add_class::<LogisticRegression>()?;
    m.add_class::<LinearRegression>()?;
    m.add_class::<Ridge>()?;
    m.add_class::<KNearestNeighbors>()?;
    m.add_class::<GaussianNB>()?;
    m.add_class::<GeoXGBoost>()?;

    // Preprocessing
    m.add_class::<StandardScaler>()?;

    // Resampling
    m.add_class::<CrossValidation>()?;
    m.add_class::<SpatialBlockCV>()?;
    m.add_class::<SpatialBufferCV>()?;
    m.add_class::<StratifiedCV>()?;
    m.add_class::<GroupCV>()?;

    // Measures
    m.add_function(wrap_pyfunction!(accuracy_score, m)?)?;
    m.add_function(wrap_pyfunction!(rmse_score, m)?)?;
    m.add_function(wrap_pyfunction!(r2_score, m)?)?;
    m.add_function(wrap_pyfunction!(mae_score, m)?)?;
    m.add_function(wrap_pyfunction!(f1_score, m)?)?;
    m.add_function(wrap_pyfunction!(precision_score, m)?)?;
    m.add_function(wrap_pyfunction!(recall_score, m)?)?;
    m.add_function(wrap_pyfunction!(auc_roc_score, m)?)?;

    // Stats
    m.add_function(wrap_pyfunction!(wilcoxon_signed_rank, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_ci, m)?)?;
    m.add_function(wrap_pyfunction!(sign_test, m)?)?;

    // Filters
    m.add_function(wrap_pyfunction!(filter_variance, m)?)?;
    m.add_function(wrap_pyfunction!(filter_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(filter_anova_f, m)?)?;
    m.add_function(wrap_pyfunction!(filter_information_gain, m)?)?;
    m.add_function(wrap_pyfunction!(filter_mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(filter_mrmr, m)?)?;
    m.add_function(wrap_pyfunction!(filter_jmi, m)?)?;
    m.add_function(wrap_pyfunction!(filter_jmim, m)?)?;
    m.add_function(wrap_pyfunction!(filter_cmim, m)?)?;
    m.add_function(wrap_pyfunction!(filter_relief, m)?)?;

    // Tuning
    m.add_class::<PyBayesianOptimizer>()?;

    // RFE
    m.add_function(wrap_pyfunction!(rfe, m)?)?;

    Ok(())
}
