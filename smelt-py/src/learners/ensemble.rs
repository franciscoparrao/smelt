//! Meta-learners: Bagging, Stacking, DynamicEnsemble. These select base
//! learners by id string (see `validate_learner_id`) rather than accepting
//! an already-constructed Python learner object -- see module comment below.

use crate::common::{add_explain_methods, add_persistence_methods, declare_support};
use crate::common::{
    extract_class_labels, fit_learner, is_integer, not_fitted, predict_proba_values,
    predict_values, smelt_err, to_array2, validate_sample_weight,
};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use smelt_ml::learner::TrainedModel;
use smelt_ml::measure::Measure;

// ── Bagging / Stacking / DynamicEnsemble ────────────────────────────────
//
// These wrap *other* learners via a `Fn() -> Box<dyn Learner>` factory in
// Rust. There's no equivalent of "pass an already-constructed Python
// learner object as the base" here: bridging an arbitrary PyO3 pyclass
// back into that closure would mean re-acquiring the GIL on every
// bootstrap sample / CV fold just to call back into Python, which is both
// a much bigger design than the rest of this file and much slower than
// staying in Rust. Instead, base learners are selected by the same id
// strings as `learner_from_id` (validated eagerly in `new()`, not at fit
// time) -- e.g. `Bagging(base="decision_tree")`, not `Bagging(base=DecisionTree())`.

pub(crate) fn validate_learner_id(id: &str) -> PyResult<()> {
    smelt_ml::prelude::learner_from_id(id).map(|_| ()).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "unknown base learner id \"{id}\"; valid ids: {}",
            smelt_ml::prelude::registered_learner_ids().join(", ")
        ))
    })
}

/// Learner id strings accepted as `base`/`meta` by Bagging, Stacking, and
/// DynamicEnsemble (the same set `learner_from_id` supports in Rust).
#[pyfunction]
pub(crate) fn registered_learner_ids() -> Vec<&'static str> {
    smelt_ml::prelude::registered_learner_ids().to_vec()
}

/// Query the declared capability metadata of a registered learner by id,
/// returning a dict of boolean flags (`supports_classification`,
/// `supports_regression`, `supports_weights`, `supports_proba`,
/// `supports_nan`, `supports_categorical`, `provides_feature_importance`,
/// `serializable`). Mirrors the Rust `learner_properties(id)`. Raises
/// `ValueError` for an unknown id.
#[pyfunction]
pub(crate) fn learner_properties(py: Python<'_>, id: &str) -> PyResult<PyObject> {
    let props = smelt_ml::prelude::learner_properties(id).map_err(|_| {
        PyValueError::new_err(format!(
            "unknown learner id \"{id}\"; valid ids: {}",
            smelt_ml::prelude::registered_learner_ids().join(", ")
        ))
    })?;
    crate::common::properties_to_dict(py, props)
}

#[pyclass]
#[derive(Default)]
pub(crate) struct Bagging {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    base: String,
    n_estimators: usize,
    seed: u64,
}

#[pymethods]
impl Bagging {
    /// `base`: learner id string (see `smelt.registered_learner_ids()`).
    #[new]
    #[pyo3(signature = (base, n_estimators=10, seed=42))]
    fn new(base: String, n_estimators: usize, seed: u64) -> PyResult<Self> {
        validate_learner_id(&base)?;
        Ok(Self {
            trained: None,
            is_classif: false,
            base,
            n_estimators,
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
        let base = self.base.clone();
        let mut learner = smelt_ml::prelude::Bagging::new(move || {
            smelt_ml::prelude::learner_from_id(&base).expect("validated in Bagging::new")
        })
        .with_n_estimators(self.n_estimators)
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

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        Ok(self
            .trained
            .as_ref()
            .ok_or_else(not_fitted)?
            .feature_importance())
    }
}

#[pyclass]
#[derive(Default)]
pub(crate) struct CostSensitiveClassifier {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    base: String,
    cost_matrix: Vec<Vec<f64>>,
}

#[pymethods]
impl CostSensitiveClassifier {
    /// `base`: learner id string (see `smelt.registered_learner_ids()`);
    /// `cost_matrix[true][predicted]`, validated against the task's actual
    /// n_classes at `fit()` time (not eagerly here, unlike `base`).
    #[new]
    #[pyo3(signature = (base, cost_matrix))]
    fn new(base: String, cost_matrix: Vec<Vec<f64>>) -> PyResult<Self> {
        validate_learner_id(&base)?;
        Ok(Self {
            trained: None,
            is_classif: false,
            base,
            cost_matrix,
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
        let base = self.base.clone();
        let cost_matrix = self.cost_matrix.clone();
        let mut learner = smelt_ml::prelude::CostSensitiveClassifier::new(
            move || smelt_ml::prelude::learner_from_id(&base).expect("validated in CostSensitiveClassifier::new"),
            cost_matrix,
        );
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

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        Ok(self.trained.as_ref().ok_or_else(not_fitted)?.feature_importance())
    }
}

/// Resolve a Python-facing transform name to the core enum, validating
/// eagerly (same `resolve_*` pattern as ELM's `activation` / XGBoost's
/// `objective`) so both `__new__` and `set_params` reject bad values with a
/// clear ValueError listing the options instead of failing later at fit.
fn resolve_transform(transform: &str) -> PyResult<smelt_ml::prelude::TargetTransform> {
    use smelt_ml::prelude::TargetTransform;
    match transform {
        "log" => Ok(TargetTransform::Log),
        "log1p" => Ok(TargetTransform::Log1p),
        "sqrt" => Ok(TargetTransform::Sqrt),
        "standardize" => Ok(TargetTransform::Standardize),
        other => Err(PyValueError::new_err(format!(
            "unknown transform \"{other}\"; valid transforms: log, log1p, sqrt, standardize"
        ))),
    }
}

/// Regression wrapper that trains its base learner on a transformed target
/// (log/log1p/sqrt/standardize) and automatically applies the inverse
/// transformation at predict time, so predictions come back in the original
/// scale. Regression-only. Note the naive log inverse estimates the
/// *median* (not the mean) of a right-skewed target under symmetric
/// log-scale errors — same behavior as sklearn's TransformedTargetRegressor.
#[pyclass]
#[derive(Default)]
pub(crate) struct TargetTransformRegressor {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    base: String,
    transform: String,
}

#[pymethods]
impl TargetTransformRegressor {
    /// `base`: learner id string (see `smelt.registered_learner_ids()`);
    /// `transform`: one of "log", "log1p", "sqrt", "standardize". Both are
    /// validated eagerly here (and again in `set_params`).
    #[new]
    #[pyo3(signature = (base, transform="log".to_string()))]
    fn new(base: String, transform: String) -> PyResult<Self> {
        validate_learner_id(&base)?;
        resolve_transform(&transform)?;
        Ok(Self {
            trained: None,
            is_classif: false,
            base,
            transform,
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
        let base = self.base.clone();
        let transform = resolve_transform(&self.transform)?;
        let mut learner = smelt_ml::prelude::TargetTransformRegressor::new(
            move || {
                smelt_ml::prelude::learner_from_id(&base)
                    .expect("validated in TargetTransformRegressor::new")
            },
            transform,
        );
        // `fit_learner` routes an integer `y` to `train_classif`, which the
        // Rust wrapper rejects with a clear regression-only error; a float
        // `y` goes through `check_finite_target` (5th audit M-4) before the
        // wrapper's own domain validation.
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

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        Ok(self
            .trained
            .as_ref()
            .ok_or_else(not_fitted)?
            .feature_importance())
    }
}

// ── CalibratedClassifier / ThresholdedClassifier ────────────────────────

/// Resolve a Python-facing calibration-method name to the core enum,
/// validating eagerly (same pattern as `resolve_transform`) so both `__new__`
/// and `set_params` reject bad values with a clear ValueError.
fn resolve_calibration_method(method: &str) -> PyResult<smelt_ml::prelude::CalibrationMethod> {
    use smelt_ml::prelude::CalibrationMethod;
    match method {
        "platt" => Ok(CalibrationMethod::Platt),
        "isotonic" => Ok(CalibrationMethod::Isotonic),
        other => Err(PyValueError::new_err(format!(
            "unknown calibration method \"{other}\"; valid methods: platt, isotonic"
        ))),
    }
}

/// Resolve a threshold-tuning metric name to a boxed measure. Restricted to
/// the classification measures where "predict class 1 iff p1 >= t" tuning is
/// meaningful (higher-is-better scores); rejects regression / probability-only
/// measures eagerly so a bad choice fails at `__new__`/`set_params`, not fit.
fn resolve_threshold_metric(metric: &str) -> PyResult<Box<dyn Measure>> {
    use smelt_ml::measure::*;
    match metric {
        "f1" => Ok(Box::new(F1Score)),
        "balanced_accuracy" => Ok(Box::new(BalancedAccuracy)),
        "accuracy" => Ok(Box::new(Accuracy)),
        "precision" => Ok(Box::new(Precision)),
        "recall" => Ok(Box::new(Recall)),
        "mcc" => Ok(Box::new(Mcc)),
        "kappa" => Ok(Box::new(CohensKappa)),
        other => Err(PyValueError::new_err(format!(
            "unknown threshold metric \"{other}\"; valid metrics: f1, balanced_accuracy, \
             accuracy, precision, recall, mcc, kappa"
        ))),
    }
}

/// Probability-calibration wrapper: trains the base classifier, remaps its
/// probabilities (Platt sigmoid or isotonic regression) on a held-out split,
/// and refits the base on all the data (scikit-learn `ensemble=False` style).
/// Classification only. Calibration preserves the ranking (Platt is monotone,
/// so AUC is unchanged) while making the probability *values* trustworthy.
#[pyclass]
#[derive(Default)]
pub(crate) struct CalibratedClassifier {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    base: String,
    method: String,
    calib_fraction: f64,
    seed: u64,
}

#[pymethods]
impl CalibratedClassifier {
    /// `base`: learner id string (see `smelt.registered_learner_ids()`);
    /// `method`: "platt" or "isotonic"; `calib_fraction`: held-out fraction
    /// used to fit the calibrator; `seed`: RNG seed for that split. `base` and
    /// `method` are validated eagerly here (and again in `set_params`).
    #[new]
    #[pyo3(signature = (base, method="platt".to_string(), calib_fraction=0.3, seed=42))]
    fn new(base: String, method: String, calib_fraction: f64, seed: u64) -> PyResult<Self> {
        validate_learner_id(&base)?;
        resolve_calibration_method(&method)?;
        Ok(Self {
            trained: None,
            is_classif: false,
            base,
            method,
            calib_fraction,
            seed,
        })
    }

    /// `sample_weight` (sklearn convention): rejected with a clear ValueError
    /// (the calibration split does not propagate weights), same as the other
    /// factory-based composites.
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let base = self.base.clone();
        let method = resolve_calibration_method(&self.method)?;
        let mut learner = smelt_ml::prelude::CalibratedClassifier::new(
            move || {
                smelt_ml::prelude::learner_from_id(&base)
                    .expect("validated in CalibratedClassifier::new")
            },
            method,
        )
        .with_calib_fraction(self.calib_fraction)
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

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        Ok(self.trained.as_ref().ok_or_else(not_fitted)?.feature_importance())
    }

    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("base", self.base.clone())?;
        dict.set_item("method", self.method.clone())?;
        dict.set_item("calib_fraction", self.calib_fraction)?;
        dict.set_item("seed", self.seed)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "base" => {
                        let base: String = v.extract()?;
                        validate_learner_id(&base)?;
                        self.base = base;
                    }
                    "method" => {
                        let method: String = v.extract()?;
                        resolve_calibration_method(&method)?;
                        self.method = method;
                    }
                    "calib_fraction" => self.calib_fraction = v.extract()?,
                    "seed" => self.seed = v.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )))
                    }
                }
            }
        }
        Ok(())
    }
}

/// Binary decision-threshold wrapper: replaces the implicit 0.5 rule with
/// either a fixed threshold or one tuned on a holdout to maximize a metric
/// (F1/balanced_accuracy/...). Binary classification only. After `fit`, the
/// applied threshold is exposed as `best_threshold_` (sklearn
/// `TunedThresholdClassifierCV` style). Note: the cost-optimal threshold in
/// closed form is `CostSensitiveClassifier` instead — this one tunes it
/// empirically from data.
#[pyclass]
#[derive(Default)]
pub(crate) struct ThresholdedClassifier {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    base: String,
    threshold: Option<f64>,
    metric: String,
    calib_fraction: f64,
    seed: u64,
    best_threshold: Option<f64>,
}

#[pymethods]
impl ThresholdedClassifier {
    /// `base`: learner id string; `threshold`: a fixed threshold (skips
    /// tuning) or `None` to tune; `metric`: measure to tune for when
    /// `threshold is None` (f1/balanced_accuracy/accuracy/precision/recall/
    /// mcc/kappa); `calib_fraction`/`seed`: the tuning holdout. `base` and
    /// `metric` are validated eagerly (and again in `set_params`).
    #[new]
    #[pyo3(signature = (base, threshold=None, metric="f1".to_string(), calib_fraction=0.3, seed=42))]
    fn new(
        base: String,
        threshold: Option<f64>,
        metric: String,
        calib_fraction: f64,
        seed: u64,
    ) -> PyResult<Self> {
        validate_learner_id(&base)?;
        resolve_threshold_metric(&metric)?;
        Ok(Self {
            trained: None,
            is_classif: false,
            base,
            threshold,
            metric,
            calib_fraction,
            seed,
            best_threshold: None,
        })
    }

    /// `sample_weight` (sklearn convention): rejected with a clear ValueError
    /// (the tuning split does not propagate weights).
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        if !is_integer(y) {
            return Err(PyValueError::new_err(
                "ThresholdedClassifier is binary-classification only; pass integer 0/1 labels",
            ));
        }
        let features = to_array2(x);
        if let Some(w) = &sample_weight {
            validate_sample_weight(w, features.nrows())?;
        }
        let target = extract_class_labels(y)?;
        let mut task =
            smelt_ml::task::ClassificationTask::new("py", features, target).map_err(smelt_err)?;
        if let Some(w) = sample_weight {
            task = task.with_weights(w);
        }

        let base = self.base.clone();
        let mut wrapper = smelt_ml::prelude::ThresholdedClassifier::new(move || {
            smelt_ml::prelude::learner_from_id(&base)
                .expect("validated in ThresholdedClassifier::new")
        });
        wrapper = match self.threshold {
            Some(t) => wrapper.with_threshold(t),
            None => wrapper.with_metric(resolve_threshold_metric(&self.metric)?),
        };
        wrapper = wrapper.with_calib_fraction(self.calib_fraction).with_seed(self.seed);

        let trained = py
            .allow_threads(|| wrapper.fit_classif(&task))
            .map_err(smelt_err)?;
        self.best_threshold = Some(trained.best_threshold());
        self.trained = Some(Box::new(trained));
        self.is_classif = true;
        Ok(())
    }

    /// The decision threshold this fitted model applies (the fixed value, or
    /// the tuned winner). Raises if not fitted.
    #[getter]
    fn best_threshold_(&self) -> PyResult<f64> {
        self.best_threshold.ok_or_else(not_fitted)
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

    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("base", self.base.clone())?;
        dict.set_item("threshold", self.threshold)?;
        dict.set_item("metric", self.metric.clone())?;
        dict.set_item("calib_fraction", self.calib_fraction)?;
        dict.set_item("seed", self.seed)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "base" => {
                        let base: String = v.extract()?;
                        validate_learner_id(&base)?;
                        self.base = base;
                    }
                    "threshold" => self.threshold = v.extract()?,
                    "metric" => {
                        let metric: String = v.extract()?;
                        resolve_threshold_metric(&metric)?;
                        self.metric = metric;
                    }
                    "calib_fraction" => self.calib_fraction = v.extract()?,
                    "seed" => self.seed = v.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )))
                    }
                }
            }
        }
        Ok(())
    }
}

#[pyclass]
#[derive(Default)]
pub(crate) struct Stacking {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    base_learners: Vec<String>,
    meta: String,
    cv_folds: usize,
    cv_seed: u64,
}

#[pymethods]
impl Stacking {
    /// `base_learners`: list of learner id strings; `meta`: learner id string
    /// for the meta-learner trained on out-of-fold base predictions.
    #[new]
    #[pyo3(signature = (base_learners, meta="logistic_regression".to_string(), cv_folds=5, cv_seed=42))]
    fn new(
        base_learners: Vec<String>,
        meta: String,
        cv_folds: usize,
        cv_seed: u64,
    ) -> PyResult<Self> {
        if base_learners.is_empty() {
            return Err(PyRuntimeError::new_err(
                "Stacking requires at least 1 base learner",
            ));
        }
        for id in &base_learners {
            validate_learner_id(id)?;
        }
        validate_learner_id(&meta)?;
        Ok(Self {
            trained: None,
            is_classif: false,
            base_learners,
            meta,
            cv_folds,
            cv_seed,
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
        let base_factories: Vec<Box<dyn Fn() -> Box<dyn smelt_ml::learner::Learner> + Send + Sync>> =
            self.base_learners
                .iter()
                .cloned()
                .map(|id| -> Box<dyn Fn() -> Box<dyn smelt_ml::learner::Learner> + Send + Sync> {
                    Box::new(move || {
                        smelt_ml::prelude::learner_from_id(&id).expect("validated in Stacking::new")
                    })
                })
                .collect();
        let meta = self.meta.clone();
        let mut learner = smelt_ml::prelude::Stacking::new(base_factories, move || {
            smelt_ml::prelude::learner_from_id(&meta).expect("validated in Stacking::new")
        })
        .with_cv_folds(self.cv_folds)
        .with_cv_seed(self.cv_seed);
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

#[pyclass]
#[derive(Default)]
pub(crate) struct DynamicEnsemble {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    base_learners: Vec<String>,
    k_neighbors: usize,
    dsel_fraction: f64,
    seed: u64,
}

#[pymethods]
impl DynamicEnsemble {
    /// KNORA-E dynamic ensemble selection (classification only).
    /// `base_learners`: list of learner id strings. `dsel_fraction`: fraction
    /// of training data held out as the Dynamic Selection set used only to
    /// evaluate each base model's local competence, never to train them.
    #[new]
    #[pyo3(signature = (base_learners, k_neighbors=7, dsel_fraction=0.3, seed=42))]
    fn new(base_learners: Vec<String>, k_neighbors: usize, dsel_fraction: f64, seed: u64) -> PyResult<Self> {
        if base_learners.is_empty() {
            return Err(PyRuntimeError::new_err(
                "DynamicEnsemble requires at least 1 base learner",
            ));
        }
        for id in &base_learners {
            validate_learner_id(id)?;
        }
        Ok(Self {
            trained: None,
            is_classif: false,
            base_learners,
            k_neighbors,
            dsel_fraction,
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
        let base_factories: Vec<Box<dyn Fn() -> Box<dyn smelt_ml::learner::Learner> + Send + Sync>> =
            self.base_learners
                .iter()
                .cloned()
                .map(|id| -> Box<dyn Fn() -> Box<dyn smelt_ml::learner::Learner> + Send + Sync> {
                    Box::new(move || {
                        smelt_ml::prelude::learner_from_id(&id)
                            .expect("validated in DynamicEnsemble::new")
                    })
                })
                .collect();
        let mut learner = smelt_ml::prelude::DynamicEnsemble::new(base_factories)
            .with_k_neighbors(self.k_neighbors)
            .with_dsel_fraction(self.dsel_fraction)
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
}


add_explain_methods!(
    Bagging,
    Stacking,
    DynamicEnsemble,
    CostSensitiveClassifier,
    TargetTransformRegressor,
    CalibratedClassifier,
    ThresholdedClassifier
);

declare_support!(Bagging,                classif = true, regress = true);
declare_support!(Stacking,               classif = true, regress = true);
declare_support!(DynamicEnsemble,        classif = true, regress = false);
declare_support!(CostSensitiveClassifier, classif = true, regress = false);
declare_support!(TargetTransformRegressor, classif = false, regress = true);
declare_support!(CalibratedClassifier,   classif = true, regress = false);
declare_support!(ThresholdedClassifier,  classif = true, regress = false);

// All 5 hold their base learner(s)' `Box<dyn TrainedModel>` internally, so
// `SerializableModel` (`src/serialize.rs`) has no variant for any of
// them -- `save()` always fails with a clear "does not support
// serialization" error rather than being silently absent from the API.
add_persistence_methods!(
    Bagging => "Bagging",
    Stacking => "Stacking",
    DynamicEnsemble => "DynamicEnsemble",
    CostSensitiveClassifier => "CostSensitiveClassifier",
    TargetTransformRegressor => "TargetTransformRegressor",
    CalibratedClassifier => "CalibratedClassifier",
    ThresholdedClassifier => "ThresholdedClassifier",
);

// get_params/set_params are hand-written here (not via `declare_params!`)
// because `base`/`base_learners`/`meta` need the same eager id validation
// `new()` does -- `fit()` relies on that having already happened
// (`.expect("validated in ...")`), so letting `set_params` skip it would
// turn a bad id into a panic instead of a clean PyValueError.

#[pymethods]
impl Bagging {
    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("base", self.base.clone())?;
        dict.set_item("n_estimators", self.n_estimators)?;
        dict.set_item("seed", self.seed)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "base" => {
                        let base: String = v.extract()?;
                        validate_learner_id(&base)?;
                        self.base = base;
                    }
                    "n_estimators" => self.n_estimators = v.extract()?,
                    "seed" => self.seed = v.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )))
                    }
                }
            }
        }
        Ok(())
    }
}

#[pymethods]
impl CostSensitiveClassifier {
    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("base", self.base.clone())?;
        dict.set_item("cost_matrix", self.cost_matrix.clone())?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "base" => {
                        let base: String = v.extract()?;
                        validate_learner_id(&base)?;
                        self.base = base;
                    }
                    "cost_matrix" => self.cost_matrix = v.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )))
                    }
                }
            }
        }
        Ok(())
    }
}

#[pymethods]
impl TargetTransformRegressor {
    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("base", self.base.clone())?;
        dict.set_item("transform", self.transform.clone())?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "base" => {
                        let base: String = v.extract()?;
                        validate_learner_id(&base)?;
                        self.base = base;
                    }
                    "transform" => {
                        let transform: String = v.extract()?;
                        resolve_transform(&transform)?;
                        self.transform = transform;
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )))
                    }
                }
            }
        }
        Ok(())
    }
}

#[pymethods]
impl Stacking {
    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("base_learners", self.base_learners.clone())?;
        dict.set_item("meta", self.meta.clone())?;
        dict.set_item("cv_folds", self.cv_folds)?;
        dict.set_item("cv_seed", self.cv_seed)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "base_learners" => {
                        let ids: Vec<String> = v.extract()?;
                        if ids.is_empty() {
                            return Err(PyValueError::new_err(
                                "Stacking requires at least 1 base learner",
                            ));
                        }
                        for id in &ids {
                            validate_learner_id(id)?;
                        }
                        self.base_learners = ids;
                    }
                    "meta" => {
                        let meta: String = v.extract()?;
                        validate_learner_id(&meta)?;
                        self.meta = meta;
                    }
                    "cv_folds" => self.cv_folds = v.extract()?,
                    "cv_seed" => self.cv_seed = v.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )))
                    }
                }
            }
        }
        Ok(())
    }
}

#[pymethods]
impl DynamicEnsemble {
    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("base_learners", self.base_learners.clone())?;
        dict.set_item("k_neighbors", self.k_neighbors)?;
        dict.set_item("dsel_fraction", self.dsel_fraction)?;
        dict.set_item("seed", self.seed)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "base_learners" => {
                        let ids: Vec<String> = v.extract()?;
                        if ids.is_empty() {
                            return Err(PyValueError::new_err(
                                "DynamicEnsemble requires at least 1 base learner",
                            ));
                        }
                        for id in &ids {
                            validate_learner_id(id)?;
                        }
                        self.base_learners = ids;
                    }
                    "k_neighbors" => self.k_neighbors = v.extract()?,
                    "dsel_fraction" => self.dsel_fraction = v.extract()?,
                    "seed" => self.seed = v.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )))
                    }
                }
            }
        }
        Ok(())
    }
}
