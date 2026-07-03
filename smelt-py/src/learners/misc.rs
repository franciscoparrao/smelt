//! Miscellaneous learners: KNearestNeighbors, GaussianNB, AdaBoost, EBM,
//! QuantileForest, QuantileGB.

use crate::common::{define_learner, add_explain_methods, declare_support, declare_params};
use crate::common::{fit_learner, not_fitted, predict_proba_values, predict_values, to_array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use smelt_ml::learner::TrainedModel;

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

add_explain_methods!(KNearestNeighbors, GaussianNB, AdaBoost, EBM, QuantileForest, QuantileGB);

declare_support!(KNearestNeighbors, classif = true,  regress = true);
declare_support!(GaussianNB,        classif = true,  regress = false);
declare_support!(AdaBoost,          classif = true,  regress = false);
declare_support!(EBM,               classif = true,  regress = true);
declare_support!(QuantileForest,    classif = false, regress = true);
declare_support!(QuantileGB,        classif = false, regress = true);

declare_params!(KNearestNeighbors, { k => "k" });
declare_params!(GaussianNB,        {});
