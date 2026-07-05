//! Tree-based learners: RandomForest, ExtraTrees, DecisionTree,
//! GradientBoosting, HoeffdingTree, AdaptiveRandomForest, ObliqueTree,
//! ObliqueForest.

use crate::common::{define_learner, add_explain_methods, declare_support, declare_params};
use crate::common::{fit_learner, not_fitted, predict_proba_values, predict_values, to_array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use smelt_ml::learner::TrainedModel;

// ── RandomForest ───────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct RandomForest {
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


// ── ExtraTrees ─────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct ExtraTrees {
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


// ── DecisionTree ───────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct DecisionTree {
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


define_learner! {
    name = GradientBoosting,
    params = { n_estimators: usize = 100, learning_rate: f64 = 0.1, max_depth: usize = 3, seed: u64 = 42 },
    ctor = |slf| smelt_ml::prelude::GradientBoosting::default()
        .with_n_estimators(slf.n_estimators)
        .with_learning_rate(slf.learning_rate)
        .with_max_depth(slf.max_depth)
        .with_seed(slf.seed),
    proba = true,
}

define_learner! {
    name = HoeffdingTree,
    params = { grace_period: usize = 200, max_depth: usize = 10 },
    ctor = |slf| smelt_ml::prelude::HoeffdingTree::default()
        .with_grace_period(slf.grace_period)
        .with_max_depth(slf.max_depth),
    proba = true,
}

define_learner! {
    name = AdaptiveRandomForest,
    // `lambda_` (not `lambda`): `lambda` is a Python keyword and can't be
    // used as a keyword-argument name, same reason XGBoost/GeoXGBoost/
    // CatBoost expose their L2 term as `lambda_`.
    params = {
        n_trees: usize = 10,
        lambda_: f64 = 6.0,
        delta_warning: f64 = 0.01,
        delta_drift: f64 = 0.001,
        split_confidence: f64 = 1e-7,
        grace_period: usize = 200,
        max_depth: usize = 10,
        seed: u64 = 42
    },
    ctor = |slf| smelt_ml::prelude::AdaptiveRandomForest::new()
        .with_n_trees(slf.n_trees)
        .with_lambda(slf.lambda_)
        .with_delta_warning(slf.delta_warning)
        .with_delta_drift(slf.delta_drift)
        .with_split_confidence(slf.split_confidence)
        .with_grace_period(slf.grace_period)
        .with_max_depth(slf.max_depth)
        .with_seed(slf.seed),
    proba = true,
}

define_learner! {
    name = MondrianForest,
    params = { n_trees: usize = 10, lifetime: f64 = f64::INFINITY, seed: u64 = 42 },
    ctor = |slf| smelt_ml::prelude::MondrianForest::new()
        .with_n_trees(slf.n_trees)
        .with_lifetime(slf.lifetime)
        .with_seed(slf.seed),
    proba = true,
}

define_learner! {
    name = ObliqueTree,
    params = { max_depth: usize = 10, n_projections: usize = 10, seed: u64 = 42 },
    ctor = |slf| smelt_ml::prelude::ObliqueTree::default()
        .with_max_depth(slf.max_depth)
        .with_n_projections(slf.n_projections)
        .with_seed(slf.seed),
    proba = true,
}

define_learner! {
    name = ObliqueForest,
    params = { n_estimators: usize = 100, max_depth: usize = 10, n_projections: usize = 10, seed: u64 = 42 },
    ctor = |slf| smelt_ml::prelude::ObliqueForest::default()
        .with_n_estimators(slf.n_estimators)
        .with_max_depth(slf.max_depth)
        .with_n_projections(slf.n_projections)
        .with_seed(slf.seed),
    proba = true,
}

add_explain_methods!(RandomForest, ExtraTrees, DecisionTree, GradientBoosting, HoeffdingTree, AdaptiveRandomForest, MondrianForest, ObliqueTree, ObliqueForest);

declare_support!(RandomForest,      classif = true,  regress = true);
declare_support!(ExtraTrees,        classif = true,  regress = true);
declare_support!(DecisionTree,      classif = true,  regress = true);
declare_support!(GradientBoosting,  classif = true,  regress = true);
declare_support!(HoeffdingTree,     classif = true,  regress = false);
declare_support!(AdaptiveRandomForest, classif = true, regress = false);
declare_support!(MondrianForest,    classif = true,  regress = true);
declare_support!(ObliqueTree,       classif = true,  regress = true);
declare_support!(ObliqueForest,     classif = true,  regress = true);

declare_params!(RandomForest, { n_estimators => "n_estimators", max_depth => "max_depth", seed => "seed" });
declare_params!(ExtraTrees,   { n_estimators => "n_estimators", max_depth => "max_depth", seed => "seed" });
declare_params!(DecisionTree, { max_depth => "max_depth" });
