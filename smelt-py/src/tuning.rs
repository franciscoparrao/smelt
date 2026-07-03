//! Hyperparameter tuning: Bayesian optimization over a param space.

use crate::common::{extract_class_labels, is_integer, resolve_measure, smelt_err, to_array2};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// ── BayesianOptimizer ──────────────────────────────────────────────────

pub(crate) fn make_learner_factory(
    learner_type: &str,
) -> PyResult<Box<dyn Fn(&smelt_ml::tuning::ParamSet) -> Box<dyn smelt_ml::learner::Learner> + Send + Sync>>
{
    use smelt_ml::prelude::*;
    type L = Box<dyn smelt_ml::learner::Learner>;
    type PS = smelt_ml::tuning::ParamSet;

    fn get(p: &PS, k: &str, def: f64) -> f64 {
        p.get(k).copied().unwrap_or(def)
    }

    match learner_type {
        "xgboost" => Ok(Box::new(|p: &PS| -> L {
            Box::new(XGBoost::new()
                .with_n_estimators(get(p, "n_estimators", 100.0) as usize)
                .with_max_depth(get(p, "max_depth", 6.0) as usize)
                .with_learning_rate(get(p, "learning_rate", 0.3))
                .with_lambda(get(p, "lambda", 1.0))
                .with_alpha(get(p, "alpha", 0.0))
                .with_gamma(get(p, "gamma", 0.0))
                .with_subsample(get(p, "subsample", 1.0))
                .with_colsample_bytree(get(p, "colsample_bytree", 1.0)))
        })),
        "catboost" => Ok(Box::new(|p: &PS| -> L {
            Box::new(CatBoost::new()
                .with_n_estimators(get(p, "n_estimators", 100.0) as usize)
                .with_depth(get(p, "depth", 6.0) as usize)
                .with_learning_rate(get(p, "learning_rate", 0.3))
                .with_lambda(get(p, "lambda", 1.0)))
        })),
        "lightgbm" => Ok(Box::new(|p: &PS| -> L {
            Box::new(LightGBM::new()
                .with_n_estimators(get(p, "n_estimators", 100.0) as usize)
                .with_num_leaves(get(p, "num_leaves", 31.0) as usize)
                .with_learning_rate(get(p, "learning_rate", 0.1))
                .with_max_depth(get(p, "max_depth", 6.0) as usize))
        })),
        "random_forest" | "rf" => Ok(Box::new(|p: &PS| -> L {
            Box::new(RandomForest::new()
                .with_n_estimators(get(p, "n_estimators", 100.0) as usize)
                .with_max_depth(get(p, "max_depth", 10.0) as usize))
        })),
        "extra_trees" | "et" => Ok(Box::new(|p: &PS| -> L {
            Box::new(ExtraTrees::new()
                .with_n_estimators(get(p, "n_estimators", 100.0) as usize)
                .with_max_depth(get(p, "max_depth", 10.0) as usize))
        })),
        "decision_tree" | "dt" => Ok(Box::new(|p: &PS| -> L {
            Box::new(DecisionTree::new()
                .with_max_depth(get(p, "max_depth", 10.0) as usize))
        })),
        "ridge" => Ok(Box::new(|p: &PS| -> L {
            Box::new(Ridge::new(get(p, "alpha", 1.0)))
        })),
        "knn" => Ok(Box::new(|p: &PS| -> L {
            Box::new(KNearestNeighbors::new(get(p, "k", 5.0) as usize))
        })),
        _ => Err(PyRuntimeError::new_err(format!("Unknown learner type: {learner_type}"))),
    }
}

pub(crate) fn build_param_space(dict: &Bound<'_, PyAny>) -> PyResult<smelt_ml::tuning::ParamSpace> {
    use smelt_ml::tuning::{ParamDistribution, ParamSpace};

    let py_dict: &Bound<'_, pyo3::types::PyDict> = dict.downcast()
        .map_err(|_| PyRuntimeError::new_err("param_space must be a dict"))?;

    let mut space = ParamSpace::new();

    for (key, val) in py_dict.iter() {
        let name: String = key.extract()?;

        // Accept dict format: {"type": "uniform", "low": 0.1, "high": 1.0}
        // Or tuple format: (low, high) → uniform
        // Or list format: [1, 2, 3] → choice
        if let Ok(inner_dict) = val.downcast::<pyo3::types::PyDict>() {
            let dtype: String = inner_dict.get_item("type")?
                .ok_or_else(|| PyRuntimeError::new_err(format!("Missing 'type' for param '{name}'")))?
                .extract()?;
            let required = |field: &str| -> PyResult<Bound<'_, PyAny>> {
                inner_dict
                    .get_item(field)?
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "param '{name}' of type '{dtype}' requires '{field}'"
                        ))
                    })
            };
            match dtype.as_str() {
                "uniform" => {
                    let low: f64 = required("low")?.extract()?;
                    let high: f64 = required("high")?.extract()?;
                    space.insert(name, ParamDistribution::Uniform(low, high));
                }
                "log_uniform" | "loguniform" => {
                    let low: f64 = required("low")?.extract()?;
                    let high: f64 = required("high")?.extract()?;
                    space.insert(name, ParamDistribution::LogUniform(low, high));
                }
                "choice" => {
                    let choices: Vec<f64> = required("values")?.extract()?;
                    space.insert(name, ParamDistribution::Choice(choices));
                }
                _ => return Err(PyRuntimeError::new_err(format!("Unknown param type: {dtype}"))),
            }
        } else if let Ok(tup) = val.extract::<(f64, f64)>() {
            // Shorthand: (low, high) → uniform
            space.insert(name, ParamDistribution::Uniform(tup.0, tup.1));
        } else if let Ok(choices) = val.extract::<Vec<f64>>() {
            // Shorthand: [1, 2, 3] → choice
            space.insert(name, ParamDistribution::Choice(choices));
        } else {
            return Err(PyRuntimeError::new_err(
                format!("Invalid param spec for '{name}'. Use dict, tuple (low, high), or list [choices]"),
            ));
        }
    }

    Ok(space)
}

#[pyclass]
#[pyo3(name = "BayesianOptimizer")]
pub(crate) struct PyBayesianOptimizer {
    n_iter: usize,
    n_initial: usize,
    seed: u64,
}

#[pymethods]
impl PyBayesianOptimizer {
    #[new]
    #[pyo3(signature = (n_iter=30, n_initial=5, seed=42))]
    fn new(n_iter: usize, n_initial: usize, seed: u64) -> Self {
        Self { n_iter, n_initial, seed }
    }

    /// Optimize hyperparameters using Bayesian TPE.
    ///
    /// Args:
    ///     learner_type: "xgboost", "rf", "catboost", "lightgbm", "dt", "ridge", "knn"
    ///     param_space: dict of param → spec. Specs can be:
    ///         - (low, high) → uniform distribution
    ///         - [v1, v2, v3] → choice
    ///         - {"type": "uniform"/"log_uniform"/"choice", "low": ..., "high": ..., "values": [...]}
    ///     x, y: training data
    ///     metric: "rmse", "r2", "accuracy", etc.
    ///     n_folds: cross-validation folds
    ///     cv_seed: seed for CV splits
    #[pyo3(signature = (learner_type, param_space, x, y, metric="rmse", n_folds=5, cv_seed=42))]
    fn optimize<'py>(
        &self,
        py: Python<'py>,
        learner_type: &str,
        param_space: &Bound<'_, PyAny>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        metric: &str,
        n_folds: usize,
        cv_seed: u64,
    ) -> PyResult<PyObject> {
        let factory = make_learner_factory(learner_type)?;
        let space = build_param_space(param_space)?;
        let measure = resolve_measure(metric)?;
        let cv = smelt_ml::resample::CrossValidation::new(n_folds).with_seed(cv_seed);

        let bo = smelt_ml::tuning::BayesianOptimizer::new(
            move |params| factory(params),
            space,
        )
        .with_n_iter(self.n_iter)
        .with_n_initial(self.n_initial)
        .with_seed(self.seed);

        let features = to_array2(x);

        let result = if is_integer(y) {
            let target = extract_class_labels(y)?;
            let task = smelt_ml::task::ClassificationTask::new("bo", features, target)
                .map_err(smelt_err)?;
            py.allow_threads(|| bo.tune_classif(&task, &cv, &*measure))
                .map_err(smelt_err)?
        } else {
            let target: Vec<f64> = y.extract()?;
            let task = smelt_ml::task::RegressionTask::new("bo", features, target)
                .map_err(smelt_err)?;
            py.allow_threads(|| bo.tune_regress(&task, &cv, &*measure))
                .map_err(smelt_err)?
        };

        // Convert TuneResult to Python dict, casting integer params to int
        let dict = pyo3::types::PyDict::new(py);

        let bp = pyo3::types::PyDict::new(py);
        for (k, v) in &result.best_params {
            set_param(&bp, k, *v)?;
        }
        dict.set_item("best_params", bp)?;
        dict.set_item("best_score", result.best_score)?;
        dict.set_item("measure", &result.measure_id)?;

        let history = pyo3::types::PyList::empty(py);
        for (params, score) in &result.all_results {
            let pd = pyo3::types::PyDict::new(py);
            for (k, v) in params {
                set_param(&pd, k, *v)?;
            }
            let tup = pyo3::types::PyTuple::new(py, &[pd.as_any(), score.into_pyobject(py)?.as_any()])?;
            history.append(tup)?;
        }
        dict.set_item("all_results", history)?;

        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }
}

/// Param names that are conceptually integer-valued and should be rounded.
pub(crate) fn is_integer_param(name: &str) -> bool {
    matches!(
        name,
        "n_estimators"
            | "max_depth"
            | "depth"
            | "num_leaves"
            | "k"
            | "min_samples_split"
            | "min_samples_leaf"
            | "n_features"
            | "seed"
            | "random_state"
    )
}

pub(crate) fn set_param(dict: &Bound<'_, pyo3::types::PyDict>, name: &str, value: f64) -> PyResult<()> {
    if is_integer_param(name) {
        dict.set_item(name, value.round() as i64)
    } else {
        dict.set_item(name, value)
    }
}

