//! Linear learners: LogisticRegression, LinearRegression, Ridge, Lasso,
//! ElasticNet, LinearSVM.

use crate::common::{define_learner, add_explain_methods, declare_support, declare_params};
use crate::common::{fit_learner, not_fitted, predict_proba_values, predict_values, to_array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use smelt_ml::learner::TrainedModel;

// ── LogisticRegression ─────────────────────────────────────────────────

#[pyclass]
pub(crate) struct LogisticRegression {
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


// ── LinearRegression ───────────────────────────────────────────────────

#[pyclass]
pub(crate) struct LinearRegression {
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


// ── Ridge ──────────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct Ridge {
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


define_learner! {
    name = Lasso,
    params = { alpha: f64 = 1.0 },
    ctor = |slf| smelt_ml::prelude::Lasso::new(slf.alpha),
    proba = false,
}

define_learner! {
    name = ElasticNet,
    params = { alpha: f64 = 1.0, l1_ratio: f64 = 0.5 },
    ctor = |slf| smelt_ml::prelude::ElasticNet::new(slf.alpha, slf.l1_ratio),
    proba = false,
}

define_learner! {
    name = LinearSVM,
    params = { c: f64 = 1.0, max_iter: usize = 1000, learning_rate: f64 = 0.01, seed: u64 = 42 },
    ctor = |slf| smelt_ml::prelude::LinearSVM::default()
        .with_c(slf.c)
        .with_max_iter(slf.max_iter)
        .with_learning_rate(slf.learning_rate)
        .with_seed(slf.seed),
    proba = true,
}

add_explain_methods!(LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, LinearSVM);

declare_support!(LogisticRegression, classif = true,  regress = false);
declare_support!(LinearRegression,   classif = false, regress = true);
declare_support!(Ridge,              classif = false, regress = true);
declare_support!(Lasso,              classif = false, regress = true);
declare_support!(ElasticNet,         classif = false, regress = true);
declare_support!(LinearSVM,          classif = true,  regress = false);

declare_params!(LogisticRegression, {});
declare_params!(LinearRegression,   {});
declare_params!(Ridge,              { alpha => "alpha" });
