//! Causal meta-learners: T/S/X/R/DR-learner. Like `Bagging`/`Stacking`/
//! `DynamicEnsemble` (`learners/ensemble.rs`), base learners are selected by
//! id string (validated eagerly in `new()`, revalidated in `set_params`)
//! rather than accepting a Python learner object -- see that module's
//! comment for why. `get_params`/`set_params` are hand-written here (not
//! via the `declare_params!` macro) for the same reason as `Bagging`:
//! the id-string fields need that same eager validation, which the generic
//! macro doesn't know how to do.

use crate::common::{smelt_err, to_array2};
use crate::learners::ensemble::validate_learner_id;
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use smelt_ml::causal::meta_learners::MetaLearnerResult;

fn meta_learner_result_to_dict(py: Python<'_>, result: MetaLearnerResult) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("cate", PyArray1::from_vec(py, result.cate))?;
    dict.set_item("ate", result.ate)?;
    Ok(dict.into_pyobject(py)?.into_any().unbind())
}

// ── T-learner ────────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct TLearner {
    control: String,
    treated: String,
}

#[pymethods]
impl TLearner {
    /// `control`/`treated`: learner id strings (see
    /// `smelt.registered_learner_ids()`) for the control-arm/treated-arm
    /// outcome models.
    #[new]
    fn new(control: String, treated: String) -> PyResult<Self> {
        validate_learner_id(&control)?;
        validate_learner_id(&treated)?;
        Ok(Self { control, treated })
    }

    /// Estimate per-unit CATE. Returns a dict with `cate` (1D array) and
    /// `ate` (float). `treatment` must be binary (0/1) with at least one
    /// unit in each arm.
    fn estimate(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        treatment: Vec<usize>,
        outcome: Vec<f64>,
    ) -> PyResult<PyObject> {
        let features = to_array2(x);
        let control = self.control.clone();
        let treated = self.treated.clone();
        let learner = smelt_ml::causal::meta_learners::TLearner::new(
            move || smelt_ml::prelude::learner_from_id(&control).expect("validated in TLearner::new"),
            move || smelt_ml::prelude::learner_from_id(&treated).expect("validated in TLearner::new"),
        );
        let result = py
            .allow_threads(|| learner.estimate(&features, &treatment, &outcome))
            .map_err(smelt_err)?;
        meta_learner_result_to_dict(py, result)
    }

    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("control", self.control.clone())?;
        dict.set_item("treated", self.treated.clone())?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "control" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.control = id;
                    }
                    "treated" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.treated = id;
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

// ── S-learner ────────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct SLearner {
    base: String,
}

#[pymethods]
impl SLearner {
    /// `base`: learner id string for the single outcome model trained on
    /// `(X, T)` jointly.
    #[new]
    fn new(base: String) -> PyResult<Self> {
        validate_learner_id(&base)?;
        Ok(Self { base })
    }

    fn estimate(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        treatment: Vec<usize>,
        outcome: Vec<f64>,
    ) -> PyResult<PyObject> {
        let features = to_array2(x);
        let base = self.base.clone();
        let learner = smelt_ml::causal::meta_learners::SLearner::new(move || {
            smelt_ml::prelude::learner_from_id(&base).expect("validated in SLearner::new")
        });
        let result = py
            .allow_threads(|| learner.estimate(&features, &treatment, &outcome))
            .map_err(smelt_err)?;
        meta_learner_result_to_dict(py, result)
    }

    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("base", self.base.clone())?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "base" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.base = id;
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

// ── X-learner ────────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct XLearner {
    control: String,
    treated: String,
    tau_control: String,
    tau_treated: String,
    propensity: String,
    propensity_clip: f64,
}

#[pymethods]
impl XLearner {
    /// `control`/`treated`: outcome models per arm. `tau_control`/
    /// `tau_treated`: models regressing the imputed effects.
    /// `propensity`: classifier for `P(T=1|X)`.
    #[new]
    #[pyo3(signature = (control, treated, tau_control, tau_treated, propensity, propensity_clip=1e-3))]
    fn new(
        control: String,
        treated: String,
        tau_control: String,
        tau_treated: String,
        propensity: String,
        propensity_clip: f64,
    ) -> PyResult<Self> {
        validate_learner_id(&control)?;
        validate_learner_id(&treated)?;
        validate_learner_id(&tau_control)?;
        validate_learner_id(&tau_treated)?;
        validate_learner_id(&propensity)?;
        Ok(Self {
            control,
            treated,
            tau_control,
            tau_treated,
            propensity,
            propensity_clip,
        })
    }

    fn estimate(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        treatment: Vec<usize>,
        outcome: Vec<f64>,
    ) -> PyResult<PyObject> {
        let features = to_array2(x);
        let control = self.control.clone();
        let treated = self.treated.clone();
        let tau_control = self.tau_control.clone();
        let tau_treated = self.tau_treated.clone();
        let propensity = self.propensity.clone();
        let learner = smelt_ml::causal::meta_learners::XLearner::new(
            move || smelt_ml::prelude::learner_from_id(&control).expect("validated in XLearner::new"),
            move || smelt_ml::prelude::learner_from_id(&treated).expect("validated in XLearner::new"),
            move || smelt_ml::prelude::learner_from_id(&tau_control).expect("validated in XLearner::new"),
            move || smelt_ml::prelude::learner_from_id(&tau_treated).expect("validated in XLearner::new"),
            move || smelt_ml::prelude::learner_from_id(&propensity).expect("validated in XLearner::new"),
        )
        .with_propensity_clip(self.propensity_clip);
        let result = py
            .allow_threads(|| learner.estimate(&features, &treatment, &outcome))
            .map_err(smelt_err)?;
        meta_learner_result_to_dict(py, result)
    }

    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("control", self.control.clone())?;
        dict.set_item("treated", self.treated.clone())?;
        dict.set_item("tau_control", self.tau_control.clone())?;
        dict.set_item("tau_treated", self.tau_treated.clone())?;
        dict.set_item("propensity", self.propensity.clone())?;
        dict.set_item("propensity_clip", self.propensity_clip)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "control" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.control = id;
                    }
                    "treated" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.treated = id;
                    }
                    "tau_control" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.tau_control = id;
                    }
                    "tau_treated" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.tau_treated = id;
                    }
                    "propensity" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.propensity = id;
                    }
                    "propensity_clip" => self.propensity_clip = v.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

// ── R-learner ────────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct RLearner {
    outcome: String,
    propensity: String,
    effect: String,
    cv_folds: usize,
    cv_seed: u64,
    residual_clip: f64,
}

#[pymethods]
impl RLearner {
    /// `outcome`: model for `m(x)=E[Y|X]`. `propensity`: classifier for
    /// `e(x)=E[T|X]`. `effect`: final model regressing the residual-ratio
    /// pseudo-target on `X`.
    #[new]
    #[pyo3(signature = (outcome, propensity, effect, cv_folds=5, cv_seed=42, residual_clip=1e-3))]
    fn new(
        outcome: String,
        propensity: String,
        effect: String,
        cv_folds: usize,
        cv_seed: u64,
        residual_clip: f64,
    ) -> PyResult<Self> {
        validate_learner_id(&outcome)?;
        validate_learner_id(&propensity)?;
        validate_learner_id(&effect)?;
        Ok(Self {
            outcome,
            propensity,
            effect,
            cv_folds,
            cv_seed,
            residual_clip,
        })
    }

    fn estimate(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        treatment: Vec<usize>,
        outcome: Vec<f64>,
    ) -> PyResult<PyObject> {
        let features = to_array2(x);
        let outcome_id = self.outcome.clone();
        let propensity_id = self.propensity.clone();
        let effect_id = self.effect.clone();
        let learner = smelt_ml::causal::meta_learners::RLearner::new(
            move || smelt_ml::prelude::learner_from_id(&outcome_id).expect("validated in RLearner::new"),
            move || smelt_ml::prelude::learner_from_id(&propensity_id).expect("validated in RLearner::new"),
            move || smelt_ml::prelude::learner_from_id(&effect_id).expect("validated in RLearner::new"),
        )
        .with_cv_folds(self.cv_folds)
        .with_cv_seed(self.cv_seed)
        .with_residual_clip(self.residual_clip);
        let result = py
            .allow_threads(|| learner.estimate(&features, &treatment, &outcome))
            .map_err(smelt_err)?;
        meta_learner_result_to_dict(py, result)
    }

    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("outcome", self.outcome.clone())?;
        dict.set_item("propensity", self.propensity.clone())?;
        dict.set_item("effect", self.effect.clone())?;
        dict.set_item("cv_folds", self.cv_folds)?;
        dict.set_item("cv_seed", self.cv_seed)?;
        dict.set_item("residual_clip", self.residual_clip)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "outcome" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.outcome = id;
                    }
                    "propensity" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.propensity = id;
                    }
                    "effect" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.effect = id;
                    }
                    "cv_folds" => self.cv_folds = v.extract()?,
                    "cv_seed" => self.cv_seed = v.extract()?,
                    "residual_clip" => self.residual_clip = v.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

// ── DR-learner ───────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct DrLearner {
    control: String,
    treated: String,
    propensity: String,
    effect: String,
    cv_folds: usize,
    cv_seed: u64,
    propensity_clip: f64,
}

#[pymethods]
impl DrLearner {
    /// `control`/`treated`: cross-fitted per-arm outcome models.
    /// `propensity`: classifier for `e(x)`. `effect`: final model
    /// regressing the doubly-robust pseudo-outcome on `X`.
    #[new]
    #[pyo3(signature = (control, treated, propensity, effect, cv_folds=5, cv_seed=42, propensity_clip=1e-3))]
    fn new(
        control: String,
        treated: String,
        propensity: String,
        effect: String,
        cv_folds: usize,
        cv_seed: u64,
        propensity_clip: f64,
    ) -> PyResult<Self> {
        validate_learner_id(&control)?;
        validate_learner_id(&treated)?;
        validate_learner_id(&propensity)?;
        validate_learner_id(&effect)?;
        Ok(Self {
            control,
            treated,
            propensity,
            effect,
            cv_folds,
            cv_seed,
            propensity_clip,
        })
    }

    fn estimate(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        treatment: Vec<usize>,
        outcome: Vec<f64>,
    ) -> PyResult<PyObject> {
        let features = to_array2(x);
        let control = self.control.clone();
        let treated = self.treated.clone();
        let propensity = self.propensity.clone();
        let effect = self.effect.clone();
        let learner = smelt_ml::causal::meta_learners::DrLearner::new(
            move || smelt_ml::prelude::learner_from_id(&control).expect("validated in DrLearner::new"),
            move || smelt_ml::prelude::learner_from_id(&treated).expect("validated in DrLearner::new"),
            move || smelt_ml::prelude::learner_from_id(&propensity).expect("validated in DrLearner::new"),
            move || smelt_ml::prelude::learner_from_id(&effect).expect("validated in DrLearner::new"),
        )
        .with_cv_folds(self.cv_folds)
        .with_cv_seed(self.cv_seed)
        .with_propensity_clip(self.propensity_clip);
        let result = py
            .allow_threads(|| learner.estimate(&features, &treatment, &outcome))
            .map_err(smelt_err)?;
        meta_learner_result_to_dict(py, result)
    }

    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("control", self.control.clone())?;
        dict.set_item("treated", self.treated.clone())?;
        dict.set_item("propensity", self.propensity.clone())?;
        dict.set_item("effect", self.effect.clone())?;
        dict.set_item("cv_folds", self.cv_folds)?;
        dict.set_item("cv_seed", self.cv_seed)?;
        dict.set_item("propensity_clip", self.propensity_clip)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "control" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.control = id;
                    }
                    "treated" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.treated = id;
                    }
                    "propensity" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.propensity = id;
                    }
                    "effect" => {
                        let id: String = v.extract()?;
                        validate_learner_id(&id)?;
                        self.effect = id;
                    }
                    "cv_folds" => self.cv_folds = v.extract()?,
                    "cv_seed" => self.cv_seed = v.extract()?,
                    "propensity_clip" => self.propensity_clip = v.extract()?,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "invalid parameter '{other}' for this estimator"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}
