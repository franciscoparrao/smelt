//! Meta-learners: Bagging, Stacking, DynamicEnsemble. These select base
//! learners by id string (see `validate_learner_id`) rather than accepting
//! an already-constructed Python learner object -- see module comment below.

use crate::common::{add_explain_methods, declare_support};
use crate::common::{fit_learner, not_fitted, predict_proba_values, predict_values, to_array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use smelt_ml::learner::TrainedModel;

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
        PyRuntimeError::new_err(format!(
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

#[pyclass]
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

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let base = self.base.clone();
        let mut learner = smelt_ml::prelude::Bagging::new(move || {
            smelt_ml::prelude::learner_from_id(&base).expect("validated in Bagging::new")
        })
        .with_n_estimators(self.n_estimators)
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

#[pyclass]
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

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let base = self.base.clone();
        let cost_matrix = self.cost_matrix.clone();
        let mut learner = smelt_ml::prelude::CostSensitiveClassifier::new(
            move || smelt_ml::prelude::learner_from_id(&base).expect("validated in CostSensitiveClassifier::new"),
            cost_matrix,
        );
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

#[pyclass]
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

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
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

#[pyclass]
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

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
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


add_explain_methods!(Bagging, Stacking, DynamicEnsemble, CostSensitiveClassifier);

declare_support!(Bagging,                classif = true, regress = true);
declare_support!(Stacking,               classif = true, regress = true);
declare_support!(DynamicEnsemble,        classif = true, regress = false);
declare_support!(CostSensitiveClassifier, classif = true, regress = false);

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
