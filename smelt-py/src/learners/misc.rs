//! Miscellaneous learners: KNearestNeighbors, GaussianNB, AdaBoost, EBM,
//! QuantileForest, QuantileGB, ExtremeLearningMachine.

use crate::common::{define_learner, add_explain_methods, declare_support, declare_params};
use crate::common::{fit_learner, not_fitted, predict_proba_values, predict_values, to_array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use smelt_ml::learner::TrainedModel;

/// Maps the Python-facing `activation` string to `smelt_ml::prelude::Activation`.
fn resolve_activation(activation: &str) -> PyResult<smelt_ml::prelude::Activation> {
    use smelt_ml::prelude::Activation;
    match activation {
        "sigmoid" => Ok(Activation::Sigmoid),
        "tanh" => Ok(Activation::Tanh),
        "relu" => Ok(Activation::Relu),
        other => Err(PyRuntimeError::new_err(format!(
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
pub(crate) struct GaussianNB {
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


define_learner! {
    name = AdaBoost,
    params = { n_estimators: usize = 50, learning_rate: f64 = 1.0 },
    ctor = |slf| smelt_ml::prelude::AdaBoost::default()
        .with_n_estimators(slf.n_estimators)
        .with_learning_rate(slf.learning_rate),
    proba = true,
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
}

define_learner! {
    name = QuantileForest,
    params = { n_estimators: usize = 100, max_depth: usize = 10, min_samples_leaf: usize = 5, seed: u64 = 42 },
    ctor = |slf| smelt_ml::prelude::QuantileForest::default()
        .with_n_estimators(slf.n_estimators)
        .with_max_depth(slf.max_depth)
        .with_min_samples_leaf(slf.min_samples_leaf)
        .with_seed(slf.seed),
    proba = false,
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
}

#[pyclass]
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

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::ExtremeLearningMachine::new()
            .with_n_hidden(self.n_hidden)
            .with_activation(resolve_activation(&self.activation)?)
            .with_regularization(self.regularization)
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
                        )))
                    }
                }
            }
        }
        Ok(())
    }
}

add_explain_methods!(KNearestNeighbors, GaussianNB, AdaBoost, EBM, QuantileForest, QuantileGB, ExtremeLearningMachine);

declare_support!(KNearestNeighbors, classif = true,  regress = true);
declare_support!(GaussianNB,        classif = true,  regress = false);
declare_support!(AdaBoost,          classif = true,  regress = false);
declare_support!(EBM,               classif = true,  regress = true);
declare_support!(QuantileForest,    classif = false, regress = true);
declare_support!(QuantileGB,        classif = false, regress = true);
declare_support!(ExtremeLearningMachine, classif = true, regress = true);

declare_params!(KNearestNeighbors, { k => "k" });
declare_params!(GaussianNB,        {});
