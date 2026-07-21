//! Miscellaneous learners: KNearestNeighbors, GaussianNB, AdaBoost, EBM,
//! QuantileForest, QuantileGB, ExtremeLearningMachine.

use crate::common::{
    add_explain_methods, add_persistence_methods, declare_params, declare_support,
    declare_weight_support, define_learner,
};
use crate::common::{fit_learner, not_fitted, predict_proba_values, predict_values, to_array2};
use crate::common::{load_model_checked, save_model};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};

use pyo3::prelude::*;
use smelt_ml::learner::TrainedModel;

/// Maps the Python-facing `activation` string to `smelt_ml::prelude::Activation`.
fn resolve_activation(activation: &str) -> PyResult<smelt_ml::prelude::Activation> {
    use smelt_ml::prelude::Activation;
    match activation {
        "sigmoid" => Ok(Activation::Sigmoid),
        "tanh" => Ok(Activation::Tanh),
        "relu" => Ok(Activation::Relu),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown activation '{other}'; expected one of: sigmoid, tanh, relu"
        ))),
    }
}

// ── KNearestNeighbors ──────────────────────────────────────────────────

#[pyclass]
pub(crate) struct KNearestNeighbors {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    k: usize,
}

#[pymethods]
impl KNearestNeighbors {
    #[new]
    #[pyo3(signature = (k=5))]
    fn new(k: usize) -> Self {
        Self {
            trained: None,
            is_classif: false,
            k,
        }
    }

    /// `sample_weight` (sklearn convention): optional per-sample weights,
    /// validated in the binding (length == n_samples, finite, >= 0, not all
    /// zero) before training; learners without weight support reject it
    /// with a clear ValueError.
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::KNearestNeighbors::new(self.k);
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y, sample_weight)?;
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

    /// Save the fitted model to a JSON file.
    fn save(&self, path: &str) -> PyResult<()> {
        save_model(&self.trained, path)
    }

    /// Load a previously saved model from a JSON file. Unlike other
    /// learners, `KNearestNeighbors` maps to one of two distinct
    /// `SerializableModel` variants (`KnnClassifier`/`KnnRegressor`)
    /// depending on `is_classif`, since classification and regression KNN
    /// are separate Rust types -- checked accordingly here instead of via
    /// `add_persistence_methods!`'s single fixed `serial_as`.
    #[staticmethod]
    #[pyo3(signature = (path, is_classif=false))]
    fn load(path: &str, is_classif: bool) -> PyResult<Self> {
        let expected = if is_classif {
            "KnnClassifier"
        } else {
            "KnnRegressor"
        };
        Ok(Self {
            trained: Some(load_model_checked(path, expected)?),
            is_classif,
            k: 5,
        })
    }
}

// ── GaussianNB ─────────────────────────────────────────────────────────

#[pyclass]
#[derive(Default)]
pub(crate) struct GaussianNB {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
}

#[pymethods]
impl GaussianNB {
    #[new]
    fn new() -> Self {
        Self {
            trained: None,
            is_classif: false,
        }
    }

    /// `sample_weight` (sklearn convention): optional per-sample weights,
    /// validated in the binding (length == n_samples, finite, >= 0, not all
    /// zero) before training; learners without weight support reject it
    /// with a clear ValueError.
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::GaussianNB::new();
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y, sample_weight)?;
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

define_learner! {
    name = AdaBoost,
    params = { n_estimators: usize = 50, learning_rate: f64 = 1.0 },
    ctor = |slf| smelt_ml::prelude::AdaBoost::default()
        .with_n_estimators(slf.n_estimators)
        .with_learning_rate(slf.learning_rate),
    proba = true,
    serial_as = "AdaBoost",
}

define_learner! {
    name = EBM,
    params = { n_rounds: usize = 100, learning_rate: f64 = 0.01, max_depth: usize = 3, seed: u64 = 42 },
    ctor = |slf| smelt_ml::prelude::EBM::default()
        .with_n_rounds(slf.n_rounds)
        .with_learning_rate(slf.learning_rate)
        .with_max_depth(slf.max_depth)
        .with_seed(slf.seed),
    proba = true,
    serial_as = "EBM",
}

// ── QuantileForest ──────────────────────────────────────────────────────
// Hand-written rather than via `define_learner!` (audit M-19): the macro
// stores `Box<dyn TrainedModel>`, which loses the concrete
// `TrainedQuantileForest` and with it `predict_quantile`/`predict_interval`
// — the whole reason to use a QRF over a plain RandomForest. Same
// concrete-storage shape as GeoXGBoost/KrigingHybrid in `boosting.rs`, but
// unlike those two this model IS in `SerializableModel`, so `save`/`load`
// are kept (hand-written, recovering the concrete type on load).
#[pyclass]
pub(crate) struct QuantileForest {
    trained: Option<smelt_ml::prelude::TrainedQuantileForest>,
    n_estimators: usize,
    max_depth: usize,
    min_samples_leaf: usize,
    seed: u64,
}

#[pymethods]
impl QuantileForest {
    /// Quantile Regression Forest (Meinshausen, 2006): a random forest whose
    /// leaves keep every training target that lands in them, so any quantile
    /// or prediction interval can be computed at prediction time.
    /// Regression-only.
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=10, min_samples_leaf=5, seed=42))]
    fn new(n_estimators: usize, max_depth: usize, min_samples_leaf: usize, seed: u64) -> Self {
        Self {
            trained: None,
            n_estimators,
            max_depth,
            min_samples_leaf,
            seed,
        }
    }

    /// Train on regression data.
    fn fit(&mut self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>, y: Vec<f64>) -> PyResult<()> {
        crate::common::check_finite_target(&y)?;
        let features = to_array2(x);
        let task = smelt_ml::task::RegressionTask::new("qrf", features, y)
            .map_err(crate::common::smelt_err)?;
        let mut learner = smelt_ml::prelude::QuantileForest::default()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_min_samples_leaf(self.min_samples_leaf)
            .with_seed(self.seed);
        let trained = py
            .allow_threads(|| learner.fit(&task))
            .map_err(crate::common::smelt_err)?;
        self.trained = Some(trained);
        Ok(())
    }

    /// Predict the median (quantile 0.5) for each sample — same as the
    /// generic `predict` any learner exposes.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_quantile(py, x, 0.5)
    }

    /// Predict an arbitrary quantile (0 <= quantile <= 1) for each sample.
    fn predict_quantile<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        quantile: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let features = to_array2(x);
        let values = py
            .allow_threads(|| model.predict_quantile(&features, quantile))
            .map_err(crate::common::smelt_err)?;
        Ok(PyArray1::from_vec(py, values))
    }

    /// Per-sample prediction interval spanning the `alpha/2` and
    /// `1 - alpha/2` quantiles (default alpha=0.1 → 90% interval).
    ///
    /// Returns dict with "predictions" (median), "lower", "upper" (numpy
    /// arrays) and "alpha" — same shape as `conformal_predict`, but from the
    /// forest's own conditional distribution instead of a calibration set.
    #[pyo3(signature = (x, alpha=0.1))]
    fn predict_interval<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        alpha: f64,
    ) -> PyResult<PyObject> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let features = to_array2(x);
        let (intervals, median) = py
            .allow_threads(|| {
                let intervals = model.predict_interval(&features, alpha)?;
                let median = model.predict_quantile(&features, 0.5)?;
                Ok::<_, smelt_ml::SmeltError>((intervals, median))
            })
            .map_err(crate::common::smelt_err)?;
        let (lower, upper): (Vec<f64>, Vec<f64>) = intervals.into_iter().unzip();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("predictions", PyArray1::from_vec(py, median))?;
        dict.set_item("lower", PyArray1::from_vec(py, lower))?;
        dict.set_item("upper", PyArray1::from_vec(py, upper))?;
        dict.set_item("alpha", alpha)?;
        Ok(dict.into())
    }

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        Ok(self
            .trained
            .as_ref()
            .ok_or_else(not_fitted)?
            .feature_importance())
    }

    /// Save the fitted model to a JSON file.
    fn save(&self, path: &str) -> PyResult<()> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let serializable = model
            .to_serializable()
            .expect("TrainedQuantileForest always has a SerializableModel variant");
        smelt_ml::serialize::save_json(&serializable, path).map_err(crate::common::smelt_err)
    }

    /// Load a previously saved model from a JSON file. `is_classif` is
    /// accepted for API compatibility with the other learners but must be
    /// False — QuantileForest is regression-only. Hyperparameters reset to
    /// the CONSTRUCTOR defaults (the file doesn't store them); call
    /// `set_params` first to restore yours before refitting.
    #[staticmethod]
    #[pyo3(signature = (path, is_classif=false))]
    fn load(path: &str, is_classif: bool) -> PyResult<Self> {
        if is_classif {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "QuantileForest is regression-only; is_classif must be False",
            ));
        }
        match smelt_ml::serialize::load_json(path).map_err(crate::common::smelt_err)? {
            smelt_ml::serialize::SerializableModel::QuantileForest(model) => {
                let mut inst = Self::new(100, 10, 5, 42);
                inst.trained = Some(model);
                Ok(inst)
            }
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected a 'QuantileForest' model in {path}, found '{}'",
                other.type_name()
            ))),
        }
    }

    /// Compute SHAP values for each sample.
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
        crate::common::shap_impl(py, model, false, x, y, n_background, feature_names, 0)
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
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        crate::common::perm_importance_impl(
            py,
            model,
            false,
            x,
            y,
            metric,
            n_repeats,
            seed,
            feature_names,
        )
    }

    /// Split conformal prediction intervals with guaranteed (1-alpha)
    /// coverage from a held-out calibration set — distribution-free, unlike
    /// `predict_interval`'s forest-native quantiles.
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
        crate::common::conformal_predict_impl(py, model, x_cal, y_cal, x_test, alpha)
    }
}

declare_params!(QuantileForest, {
    n_estimators => "n_estimators",
    max_depth => "max_depth",
    min_samples_leaf => "min_samples_leaf",
    seed => "seed",
});

// ── GaussianProcess ────────────────────────────────────────────────────
//
// Regression-only GP with an RBF kernel. Beyond the generic `predict`
// (posterior mean), it exposes `predict_std`/`predict_with_std` — the
// predictive standard error that is the whole point of a GP — the same
// "inherent method past the trait" shape as QuantileForest's
// predict_quantile/predict_interval above. Not serializable, so no
// save/load (matching its Rust side, which declares no SerializableModel
// variant).

#[pyclass]
pub(crate) struct GaussianProcess {
    trained: Option<smelt_ml::prelude::TrainedGaussianProcess>,
    length_scale: f64,
    signal_variance: f64,
    alpha: f64,
}

#[pymethods]
impl GaussianProcess {
    /// Gaussian Process regressor (RBF kernel). `alpha` is the observation
    /// noise variance added to the kernel diagonal; kernel hyperparameters are
    /// fixed (not optimized). Regression-only.
    #[new]
    #[pyo3(signature = (length_scale=1.0, signal_variance=1.0, alpha=1e-10))]
    fn new(length_scale: f64, signal_variance: f64, alpha: f64) -> Self {
        Self {
            trained: None,
            length_scale,
            signal_variance,
            alpha,
        }
    }

    /// Train on regression data. Raises ValueError for non-positive kernel
    /// hyperparameters or a kernel matrix that isn't positive definite.
    fn fit(&mut self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>, y: Vec<f64>) -> PyResult<()> {
        crate::common::check_finite_target(&y)?;
        let features = to_array2(x);
        let task = smelt_ml::task::RegressionTask::new("gp", features, y)
            .map_err(crate::common::smelt_err)?;
        let learner = smelt_ml::prelude::GaussianProcess::new()
            .with_length_scale(self.length_scale)
            .with_signal_variance(self.signal_variance)
            .with_alpha(self.alpha);
        let trained = py
            .allow_threads(|| learner.fit(&task))
            .map_err(crate::common::smelt_err)?;
        self.trained = Some(trained);
        Ok(())
    }

    /// Posterior mean prediction for each sample.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let features = to_array2(x);
        let (means, _) = py
            .allow_threads(|| model.predict_with_std(&features))
            .map_err(crate::common::smelt_err)?;
        Ok(PyArray1::from_vec(py, means))
    }

    /// Posterior predictive standard deviation (`se`) for each sample.
    fn predict_std<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let features = to_array2(x);
        let stds = py
            .allow_threads(|| model.predict_std(&features))
            .map_err(crate::common::smelt_err)?;
        Ok(PyArray1::from_vec(py, stds))
    }

    /// Posterior mean AND standard deviation together. Returns a dict with
    /// "mean" and "std" numpy arrays.
    fn predict_with_std<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<PyObject> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let features = to_array2(x);
        let (means, stds) = py
            .allow_threads(|| model.predict_with_std(&features))
            .map_err(crate::common::smelt_err)?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("mean", PyArray1::from_vec(py, means))?;
        dict.set_item("std", PyArray1::from_vec(py, stds))?;
        Ok(dict.into())
    }
}

declare_params!(GaussianProcess, {
    length_scale => "length_scale",
    signal_variance => "signal_variance",
    alpha => "alpha",
});

// ── KernelSVM ──────────────────────────────────────────────────────────
//
// Kernel C-SVC (SMO). Classification-only. Beyond `predict`, exposes
// `decision_function` (the raw signed margin / one-vs-rest scores) — the
// inherent method past the trait, like GeoXGBoost's predict_spatial.
// get_params/set_params are hand-written (not `declare_params!`) so the
// `kernel` string is re-validated eagerly on set_params, same rationale as
// KrigingHybrid/XGBoost.

/// Maps the Python-facing `kernel` string to `smelt_ml::prelude::Kernel`.
fn resolve_kernel(kernel: &str, degree: u32, coef0: f64) -> PyResult<smelt_ml::prelude::Kernel> {
    use smelt_ml::prelude::Kernel;
    match kernel {
        "linear" => Ok(Kernel::Linear),
        "rbf" => Ok(Kernel::Rbf),
        "poly" => Ok(Kernel::Poly { degree, coef0 }),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown kernel '{other}'; use 'linear', 'rbf', or 'poly'"
        ))),
    }
}

#[pyclass]
pub(crate) struct KernelSVM {
    trained: Option<smelt_ml::prelude::TrainedKernelSVM>,
    c: f64,
    kernel: String,
    gamma: Option<f64>,
    degree: u32,
    coef0: f64,
    tol: f64,
}

#[pymethods]
impl KernelSVM {
    /// Kernel Support Vector Classifier (C-SVC, SMO solver). `kernel` is
    /// "rbf" (default), "linear", or "poly"; `gamma=None` defaults to
    /// 1/n_features at fit time. Classification-only.
    #[new]
    #[pyo3(signature = (C=1.0, kernel="rbf".to_string(), gamma=None, degree=3, coef0=0.0, tol=1e-4))]
    #[allow(non_snake_case)]
    fn new(
        C: f64,
        kernel: String,
        gamma: Option<f64>,
        degree: u32,
        coef0: f64,
        tol: f64,
    ) -> PyResult<Self> {
        resolve_kernel(&kernel, degree, coef0)?; // eager validation
        Ok(Self {
            trained: None,
            c: C,
            kernel,
            gamma,
            degree,
            coef0,
            tol,
        })
    }

    /// Train on classification data.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let features = to_array2(x);
        let target = crate::common::extract_class_labels(y)?;
        let task = smelt_ml::task::ClassificationTask::new("kernel_svm", features, target)
            .map_err(crate::common::smelt_err)?;
        let kernel = resolve_kernel(&self.kernel, self.degree, self.coef0)?;
        let mut svm = smelt_ml::prelude::KernelSVM::new()
            .with_c(self.c)
            .with_kernel(kernel)
            .with_tol(self.tol);
        if let Some(g) = self.gamma {
            svm = svm.with_gamma(g);
        }
        let trained = py
            .allow_threads(|| svm.fit(&task))
            .map_err(crate::common::smelt_err)?;
        self.trained = Some(trained);
        Ok(())
    }

    /// Predict class labels for each sample.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let features = to_array2(x);
        let pred = py
            .allow_threads(|| model.predict(&features))
            .map_err(crate::common::smelt_err)?;
        match pred {
            smelt_ml::prelude::Prediction::Classification { predicted, .. } => Ok(
                PyArray1::from_vec(py, predicted.iter().map(|&p| p as f64).collect()),
            ),
            _ => unreachable!("KernelSVM always predicts classification"),
        }
    }

    /// Raw decision values: shape (n_samples,) for binary (signed margin,
    /// >0 → class 1), or (n_samples, n_classes) one-vs-rest scores otherwise.
    fn decision_function(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<PyObject> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let features = to_array2(x);
        let rows: Vec<Vec<f64>> = features
            .rows()
            .into_iter()
            .map(|r| model.decision_function(r))
            .collect();
        let n_out = rows.first().map(|r| r.len()).unwrap_or(0);
        if n_out == 1 {
            let flat: Vec<f64> = rows.iter().map(|r| r[0]).collect();
            Ok(PyArray1::from_vec(py, flat).into_any().unbind())
        } else {
            let arr = PyArray2::from_vec2(py, &rows).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("decision_function shape: {e}"))
            })?;
            Ok(arr.into_any().unbind())
        }
    }

    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("C", self.c)?;
        dict.set_item("kernel", self.kernel.clone())?;
        dict.set_item("gamma", self.gamma)?;
        dict.set_item("degree", self.degree)?;
        dict.set_item("coef0", self.coef0)?;
        dict.set_item("tol", self.tol)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "C" => self.c = v.extract()?,
                    "kernel" => {
                        let kernel: String = v.extract()?;
                        // Re-validate against current degree/coef0.
                        resolve_kernel(&kernel, self.degree, self.coef0)?;
                        self.kernel = kernel;
                    }
                    "gamma" => self.gamma = v.extract()?,
                    "degree" => self.degree = v.extract()?,
                    "coef0" => self.coef0 = v.extract()?,
                    "tol" => self.tol = v.extract()?,
                    other => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

define_learner! {
    name = QuantileGB,
    params = { quantile: f64 = 0.5, n_estimators: usize = 100, learning_rate: f64 = 0.1, max_depth: usize = 3, seed: u64 = 42 },
    ctor = |slf| smelt_ml::prelude::QuantileGB::new(slf.quantile)
        .with_n_estimators(slf.n_estimators)
        .with_learning_rate(slf.learning_rate)
        .with_max_depth(slf.max_depth)
        .with_seed(slf.seed),
    proba = false,
    serial_as = "QuantileGB",
}

#[pyclass]
#[derive(Default)]
pub(crate) struct ExtremeLearningMachine {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    n_hidden: usize,
    activation: String,
    regularization: f64,
    seed: u64,
}

#[pymethods]
impl ExtremeLearningMachine {
    #[new]
    #[pyo3(signature = (n_hidden=100, activation="sigmoid".to_string(), regularization=1e-3, seed=42))]
    fn new(n_hidden: usize, activation: String, regularization: f64, seed: u64) -> PyResult<Self> {
        resolve_activation(&activation)?;
        Ok(Self {
            trained: None,
            is_classif: false,
            n_hidden,
            activation,
            regularization,
            seed,
        })
    }

    /// `sample_weight` (sklearn convention): optional per-sample weights,
    /// validated in the binding (length == n_samples, finite, >= 0, not all
    /// zero) before training; learners without weight support reject it
    /// with a clear ValueError.
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::ExtremeLearningMachine::new()
            .with_n_hidden(self.n_hidden)
            .with_activation(resolve_activation(&self.activation)?)
            .with_regularization(self.regularization)
            .with_seed(self.seed);
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y, sample_weight)?;
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

    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("n_hidden", self.n_hidden)?;
        dict.set_item("activation", self.activation.clone())?;
        dict.set_item("regularization", self.regularization)?;
        dict.set_item("seed", self.seed)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "n_hidden" => self.n_hidden = v.extract()?,
                    "activation" => {
                        let activation: String = v.extract()?;
                        resolve_activation(&activation)?;
                        self.activation = activation;
                    }
                    "regularization" => self.regularization = v.extract()?,
                    "seed" => self.seed = v.extract()?,
                    other => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

// QuantileForest is excluded: it stores its model as a concrete
// `Option<TrainedQuantileForest>` (not `Option<Box<dyn TrainedModel>>`),
// so its explain methods are hand-written above.
add_explain_methods!(
    KNearestNeighbors,
    GaussianNB,
    AdaBoost,
    EBM,
    QuantileGB,
    ExtremeLearningMachine
);

declare_support!(KNearestNeighbors, classif = true, regress = true);
declare_support!(GaussianNB, classif = true, regress = false);
declare_support!(AdaBoost, classif = true, regress = false);
declare_support!(EBM, classif = true, regress = true);
declare_support!(QuantileForest, classif = false, regress = true);
declare_support!(QuantileGB, classif = false, regress = true);
declare_support!(ExtremeLearningMachine, classif = true, regress = true);

declare_weight_support!(
    KNearestNeighbors      => smelt_ml::prelude::KNearestNeighbors::new(5),
    GaussianNB             => smelt_ml::prelude::GaussianNB::new(),
    ExtremeLearningMachine => smelt_ml::prelude::ExtremeLearningMachine::new(),
    QuantileForest         => smelt_ml::prelude::QuantileForest::default(),
);

declare_params!(KNearestNeighbors, { k => "k" });
declare_params!(GaussianNB, {});

add_persistence_methods!(
    GaussianNB => "GaussianNB",
    ExtremeLearningMachine => "ExtremeLearningMachine",
);
