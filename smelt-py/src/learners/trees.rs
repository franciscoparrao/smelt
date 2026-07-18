//! Tree-based learners: RandomForest, ExtraTrees, DecisionTree,
//! GradientBoosting, HoeffdingTree, AdaptiveRandomForest, ObliqueTree,
//! ObliqueForest.

use crate::common::{define_learner, add_explain_methods, add_persistence_methods, declare_support, declare_params, declare_weight_support};
use crate::common::{fit_learner, not_fitted, predict_proba_values, predict_values, to_array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use smelt_ml::learner::TrainedModel;

/// `max_depth` for RF/ET/DT: `None` means unlimited depth (the Rust
/// builders' own default), and `0` is rejected -- a depth-0 tree is a
/// single constant leaf, which used to train "successfully" and predict
/// at chance level with no hint of why.
fn apply_max_depth<L>(
    learner: L,
    max_depth: Option<usize>,
    with: impl FnOnce(L, usize) -> L,
) -> PyResult<L> {
    match max_depth {
        Some(0) => Err(pyo3::exceptions::PyValueError::new_err(
            "max_depth=0 would build a constant (root-only) tree; use max_depth=None for \
             unlimited depth or a positive integer",
        )),
        Some(d) => Ok(with(learner, d)),
        None => Ok(learner),
    }
}

// ── RandomForest ───────────────────────────────────────────────────────

#[pyclass]
#[derive(Default)]
pub(crate) struct RandomForest {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    n_estimators: usize,
    max_depth: Option<usize>,
    seed: u64,
}

#[pymethods]
impl RandomForest {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=10, seed=42))]
    fn new(n_estimators: usize, max_depth: Option<usize>, seed: u64) -> Self {
        Self {
            trained: None,
            is_classif: false,
            n_estimators,
            max_depth,
            seed,
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
        let mut learner = smelt_ml::prelude::RandomForest::new()
            .with_n_estimators(self.n_estimators)
            .with_seed(self.seed);
        learner = apply_max_depth(learner, self.max_depth, |l, d| l.with_max_depth(d))?;
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


// ── ExtraTrees ─────────────────────────────────────────────────────────

#[pyclass]
#[derive(Default)]
pub(crate) struct ExtraTrees {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    n_estimators: usize,
    max_depth: Option<usize>,
    seed: u64,
}

#[pymethods]
impl ExtraTrees {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=10, seed=42))]
    fn new(n_estimators: usize, max_depth: Option<usize>, seed: u64) -> Self {
        Self { trained: None, is_classif: false, n_estimators, max_depth, seed }
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
        let mut learner = smelt_ml::prelude::ExtraTrees::new()
            .with_n_estimators(self.n_estimators)
            .with_seed(self.seed);
        learner = apply_max_depth(learner, self.max_depth, |l, d| l.with_max_depth(d))?;
        let (model, is_classif) = fit_learner(py, &mut learner, to_array2(x), y, sample_weight)?;
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
#[derive(Default)]
pub(crate) struct DecisionTree {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    max_depth: Option<usize>,
}

#[pymethods]
impl DecisionTree {
    #[new]
    #[pyo3(signature = (max_depth=10))]
    fn new(max_depth: Option<usize>) -> Self {
        Self {
            trained: None,
            is_classif: false,
            max_depth,
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
        let mut learner = apply_max_depth(
            smelt_ml::prelude::DecisionTree::default(),
            self.max_depth,
            |l, d| l.with_max_depth(d),
        )?;
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
    serial_as = "GradientBoosting",
}

define_learner! {
    name = HoeffdingTree,
    params = { grace_period: usize = 200, max_depth: usize = 10 },
    ctor = |slf| smelt_ml::prelude::HoeffdingTree::default()
        .with_grace_period(slf.grace_period)
        .with_max_depth(slf.max_depth),
    proba = true,
    serial_as = "HoeffdingTree",
    note = "Streaming (incremental) decision tree, exposed here batch-only \
        (fit/predict, not partial_fit/predict_one). `grace_period` (default \
        200) is the minimum samples a leaf must see before a split is even \
        considered -- a streaming-oriented default. Fitting a single small \
        batch (n well under grace_period) means the tree may never split at \
        all, silently degrading to a majority-class predictor. For batch \
        datasets with n in the hundreds or low thousands, pass a much \
        smaller grace_period (e.g. max(10, n // 10)), or prefer \
        RandomForest/ExtraTrees for non-streaming use cases.",
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
    serial_as = "AdaptiveRandomForest",
    note = "Streaming ensemble of Hoeffding trees with ADWIN concept-drift \
        detection, exposed here batch-only (fit/predict). `grace_period` \
        (default 200, same streaming-oriented default as HoeffdingTree) is \
        the minimum samples a leaf must see before splitting -- on a small \
        batch dataset (n in the hundreds), trees may never split, degrading \
        to near-chance accuracy silently. For batch use with n well under a \
        few thousand, pass a much smaller grace_period (e.g. max(10, n // \
        10)), or prefer RandomForest/ExtraTrees.",
}

define_learner! {
    name = DeepForest,
    params = {
        n_forests_per_type: usize = 2,
        n_estimators_per_forest: usize = 100,
        max_depth: usize = 10,
        cv_folds: usize = 3,
        max_layers: usize = 10,
        early_stopping_rounds: usize = 2,
        seed: u64 = 42
    },
    ctor = |slf| smelt_ml::prelude::DeepForest::new()
        .with_n_forests_per_type(slf.n_forests_per_type)
        .with_n_estimators_per_forest(slf.n_estimators_per_forest)
        .with_max_depth(slf.max_depth)
        .with_cv_folds(slf.cv_folds)
        .with_max_layers(slf.max_layers)
        .with_early_stopping_rounds(slf.early_stopping_rounds)
        .with_seed(slf.seed),
    proba = true,
    // `DeepForest` is a `Box<dyn TrainedModel>`-holding cascade internally
    // (each layer's forests), so it has no `SerializableModel` variant --
    // `save()`/`load()` always fail cleanly, matching the exclusion
    // documented in `src/serialize.rs`. No literal string matches this on
    // load, so any file is correctly rejected.
    serial_as = "DeepForest",
}

define_learner! {
    name = MondrianForest,
    params = { n_trees: usize = 10, lifetime: f64 = f64::INFINITY, seed: u64 = 42 },
    ctor = |slf| smelt_ml::prelude::MondrianForest::new()
        .with_n_trees(slf.n_trees)
        .with_lifetime(slf.lifetime)
        .with_seed(slf.seed),
    proba = true,
    serial_as = "MondrianForest",
}

define_learner! {
    name = ObliqueTree,
    params = { max_depth: usize = 10, n_projections: usize = 10, seed: u64 = 42 },
    ctor = |slf| smelt_ml::prelude::ObliqueTree::default()
        .with_max_depth(slf.max_depth)
        .with_n_projections(slf.n_projections)
        .with_seed(slf.seed),
    proba = true,
    serial_as = "ObliqueTree",
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
    serial_as = "ObliqueForest",
}

add_explain_methods!(RandomForest, ExtraTrees, DecisionTree, GradientBoosting, HoeffdingTree, AdaptiveRandomForest, DeepForest, MondrianForest, ObliqueTree, ObliqueForest);

declare_support!(RandomForest,      classif = true,  regress = true);
declare_support!(ExtraTrees,        classif = true,  regress = true);
declare_support!(DecisionTree,      classif = true,  regress = true);
declare_support!(GradientBoosting,  classif = true,  regress = true);
declare_support!(HoeffdingTree,     classif = true,  regress = false);
declare_support!(AdaptiveRandomForest, classif = true, regress = false);
declare_support!(DeepForest,        classif = true,  regress = false);
declare_support!(MondrianForest,    classif = true,  regress = true);
declare_support!(ObliqueTree,       classif = true,  regress = true);
declare_support!(ObliqueForest,     classif = true,  regress = true);

declare_weight_support!(
    RandomForest => smelt_ml::prelude::RandomForest::new(),
    ExtraTrees   => smelt_ml::prelude::ExtraTrees::new(),
    DecisionTree => smelt_ml::prelude::DecisionTree::default(),
);

declare_params!(RandomForest, { n_estimators => "n_estimators", max_depth => "max_depth", seed => "seed" });
declare_params!(ExtraTrees,   { n_estimators => "n_estimators", max_depth => "max_depth", seed => "seed" });
declare_params!(DecisionTree, { max_depth => "max_depth" });

add_persistence_methods!(
    RandomForest => "RandomForest",
    ExtraTrees => "ExtraTrees",
    DecisionTree => "DecisionTree",
);
