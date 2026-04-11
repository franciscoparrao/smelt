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

fn fit_learner(
    learner: &mut dyn Learner,
    features: Array2<f64>,
    y: &Bound<'_, PyAny>,
) -> PyResult<(Box<dyn TrainedModel>, bool)> {
    if is_integer(y) {
        let target: Vec<i64> = y.extract()?;
        let target: Vec<usize> = target.into_iter().map(|v| v as usize).collect();
        let task = smelt_ml::task::ClassificationTask::new("py", features, target)
            .map_err(smelt_err)?;
        let model = learner.train_classif(&task).map_err(smelt_err)?;
        Ok((model, true))
    } else {
        let target: Vec<f64> = y.extract()?;
        let task =
            smelt_ml::task::RegressionTask::new("py", features, target).map_err(smelt_err)?;
        let model = learner.train_regress(&task).map_err(smelt_err)?;
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
            let k = probs[0].len();
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

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
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
        let (model, _) = fit_learner(&mut learner, to_array2(x), y)?;
        self.trained = Some(model);
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
            n_estimators,
            depth,
            learning_rate,
            lambda: lambda_,
            seed,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::CatBoost::new()
            .with_n_estimators(self.n_estimators)
            .with_depth(self.depth)
            .with_learning_rate(self.learning_rate)
            .with_lambda(self.lambda)
            .with_seed(self.seed);
        let (model, _) = fit_learner(&mut learner, to_array2(x), y)?;
        self.trained = Some(model);
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
            n_estimators,
            max_depth,
            seed,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::RandomForest::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_seed(self.seed);
        let (model, _) = fit_learner(&mut learner, to_array2(x), y)?;
        self.trained = Some(model);
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
    max_depth: usize,
}

#[pymethods]
impl DecisionTree {
    #[new]
    #[pyo3(signature = (max_depth=10))]
    fn new(max_depth: usize) -> Self {
        Self {
            trained: None,
            max_depth,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut learner =
            smelt_ml::prelude::DecisionTree::default().with_max_depth(self.max_depth);
        let (model, _) = fit_learner(&mut learner, to_array2(x), y)?;
        self.trained = Some(model);
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
}

#[pymethods]
impl LogisticRegression {
    #[new]
    fn new() -> Self {
        Self { trained: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::LogisticRegression::new();
        let (model, _) = fit_learner(&mut learner, to_array2(x), y)?;
        self.trained = Some(model);
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

    fn splits(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples)
    }
}

// ── SpatialBlockCV ─────────────────────────────────────────────────────

#[pyclass]
struct SpatialBlockCV {
    inner: smelt_ml::prelude::SpatialBlockCV,
}

#[pymethods]
impl SpatialBlockCV {
    #[new]
    fn new(n_folds: usize, coords: Vec<(f64, f64)>) -> Self {
        Self {
            inner: smelt_ml::prelude::SpatialBlockCV::new(n_folds, coords),
        }
    }

    fn splits(&self, n_samples: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        use smelt_ml::prelude::Resample;
        self.inner.splits(n_samples)
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

#[pyfunction]
fn auc_roc_score(y_true: Vec<usize>, y_proba: Vec<Vec<f64>>) -> PyResult<f64> {
    let n = y_true.len();
    let pred_class: Vec<usize> = y_proba
        .iter()
        .map(|p| {
            p.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0
        })
        .collect();
    let pred = Prediction::Classification {
        predicted: pred_class,
        truth: Some(y_true),
        probabilities: Some(y_proba),
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
fn wilcoxon_signed_rank(a: Vec<f64>, b: Vec<f64>) -> TestResult {
    let r = smelt_ml::stats::wilcoxon_signed_rank(&a, &b);
    TestResult {
        test: r.test.to_string(),
        statistic: r.statistic,
        p_value: r.p_value,
        significant: r.significant,
    }
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
fn sign_test(a: Vec<f64>, b: Vec<f64>) -> TestResult {
    let r = smelt_ml::stats::sign_test(&a, &b);
    TestResult {
        test: r.test.to_string(),
        statistic: r.statistic,
        p_value: r.p_value,
        significant: r.significant,
    }
}

// ── LightGBM ───────────────────────────────────────────────────────────

#[pyclass]
struct LightGBM {
    trained: Option<Box<dyn TrainedModel>>,
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
        Self { trained: None, n_estimators, num_leaves, learning_rate, max_depth, seed }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::LightGBM::new()
            .with_n_estimators(self.n_estimators)
            .with_num_leaves(self.num_leaves)
            .with_learning_rate(self.learning_rate)
            .with_max_depth(self.max_depth)
            .with_seed(self.seed);
        let (model, _) = fit_learner(&mut learner, to_array2(x), y)?;
        self.trained = Some(model);
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
    n_estimators: usize,
    max_depth: usize,
    seed: u64,
}

#[pymethods]
impl ExtraTrees {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=10, seed=42))]
    fn new(n_estimators: usize, max_depth: usize, seed: u64) -> Self {
        Self { trained: None, n_estimators, max_depth, seed }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::ExtraTrees::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_seed(self.seed);
        let (model, _) = fit_learner(&mut learner, to_array2(x), y)?;
        self.trained = Some(model);
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
    k: usize,
}

#[pymethods]
impl KNearestNeighbors {
    #[new]
    #[pyo3(signature = (k=5))]
    fn new(k: usize) -> Self {
        Self { trained: None, k }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::KNearestNeighbors::new(self.k);
        let (model, _) = fit_learner(&mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }
}

// ── GaussianNB ─────────────────────────────────────────────────────────

#[pyclass]
struct GaussianNB {
    trained: Option<Box<dyn TrainedModel>>,
}

#[pymethods]
impl GaussianNB {
    #[new]
    fn new() -> Self {
        Self { trained: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::GaussianNB::new();
        let (model, _) = fit_learner(&mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }
}

// ── Ridge ──────────────────────────────────────────────────────────────

#[pyclass]
struct Ridge {
    trained: Option<Box<dyn TrainedModel>>,
    alpha: f64,
}

#[pymethods]
impl Ridge {
    #[new]
    #[pyo3(signature = (alpha=1.0))]
    fn new(alpha: f64) -> Self {
        Self { trained: None, alpha }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::Ridge::new(self.alpha);
        let (model, _) = fit_learner(&mut learner, to_array2(x), y)?;
        self.trained = Some(model);
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
}

#[pymethods]
impl LinearRegression {
    #[new]
    fn new() -> Self {
        Self { trained: None }
    }

    fn fit(&mut self, x: PyReadonlyArray2<'_, f64>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::LinearRegression::new();
        let (model, _) = fit_learner(&mut learner, to_array2(x), y)?;
        self.trained = Some(model);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        predict_values(self.trained.as_deref().ok_or_else(not_fitted)?, py, x)
    }
}

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

    // Preprocessing
    m.add_class::<StandardScaler>()?;

    // Resampling
    m.add_class::<CrossValidation>()?;
    m.add_class::<SpatialBlockCV>()?;

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

    Ok(())
}
