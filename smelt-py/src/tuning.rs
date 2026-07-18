//! Hyperparameter tuning: Bayesian optimization over a param space.

use crate::common::{extract_class_labels, is_integer, resolve_measure, smelt_err, to_array2};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ── BayesianOptimizer ──────────────────────────────────────────────────

pub(crate) fn make_learner_factory(
    learner_type: &str,
) -> PyResult<Box<dyn Fn(&smelt_ml::tuning::ParamSet) -> Box<dyn smelt_ml::learner::Learner> + Send + Sync>>
{
    use smelt_ml::prelude::*;
    type L = Box<dyn smelt_ml::learner::Learner>;
    type PS = smelt_ml::tuning::ParamSet;

    // The tuning factory signature (`Fn(&ParamSet) -> Box<dyn Learner>`, no
    // `Result`) can't propagate a clean error for a misused param type, so a
    // genuine type mismatch (e.g. a string Choice value for a numeric
    // hyperparameter) panics with a clear message instead of silently
    // falling back to `def` -- PyO3 catches panics at the `#[pymethods]`
    // boundary (`optimize`, below) and surfaces them as a Python exception,
    // not a process crash.
    fn get(p: &PS, k: &str, def: f64) -> f64 {
        p.get(k)
            .map(|v| {
                v.as_f64()
                    .unwrap_or_else(|e| panic!("invalid value for parameter '{k}': {e}"))
            })
            .unwrap_or(def)
    }

    match learner_type {
        "xgboost" => Ok(Box::new(|p: &PS| -> L {
            let mut xgb = XGBoost::new()
                .with_n_estimators(get(p, "n_estimators", 100.0) as usize)
                .with_max_depth(get(p, "max_depth", 6.0) as usize)
                .with_learning_rate(get(p, "learning_rate", 0.3))
                .with_lambda(get(p, "lambda", 1.0))
                .with_alpha(get(p, "alpha", 0.0))
                .with_gamma(get(p, "gamma", 0.0))
                .with_subsample(get(p, "subsample", 1.0))
                .with_colsample_bytree(get(p, "colsample_bytree", 1.0));
            // The M10 string-choice use case, now actually wired (audit
            // M-13: tuning `objective` used to be a silent no-op). Choice
            // values are validated eagerly in `optimize`, so these panics
            // are unreachable from the Python entry point; they guard
            // direct Rust callers of the factory.
            if let Some(v) = p.get("objective") {
                let name = v
                    .as_str()
                    .unwrap_or_else(|e| panic!("invalid value for parameter 'objective': {e}"));
                let obj = crate::learners::boosting::resolve_objective(
                    name,
                    get(p, "huber_delta", 1.0),
                )
                .unwrap_or_else(|e| panic!("invalid value for parameter 'objective': {e}"));
                xgb = xgb.with_objective(obj);
            }
            Box::new(xgb)
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
        _ => Err(PyValueError::new_err(format!("Unknown learner type: {learner_type}"))),
    }
}

/// The exact parameter names each `make_learner_factory` closure reads.
/// `optimize` validates the user's `param_space` against this list, because
/// the factory looks params up by name with a default fallback: a name it
/// never reads (a typo, or a parameter this binding doesn't wire) would
/// otherwise be "tuned" silently — every trial trains the identical model
/// and `best_params` is meaningless (audit M-13). Keep in sync with the
/// factory arms above.
pub(crate) fn factory_param_names(learner_type: &str) -> &'static [&'static str] {
    match learner_type {
        "xgboost" => &[
            "n_estimators",
            "max_depth",
            "learning_rate",
            "lambda",
            "alpha",
            "gamma",
            "subsample",
            "colsample_bytree",
            "objective",
            "huber_delta",
        ],
        "catboost" => &["n_estimators", "depth", "learning_rate", "lambda"],
        "lightgbm" => &["n_estimators", "num_leaves", "learning_rate", "max_depth"],
        "random_forest" | "rf" | "extra_trees" | "et" => &["n_estimators", "max_depth"],
        "decision_tree" | "dt" => &["max_depth"],
        "ridge" => &["alpha"],
        "knn" => &["k"],
        _ => &[],
    }
}

/// Rejects param-space entries the factory would silently ignore, and
/// eagerly validates `objective` choice values (so a bad objective fails
/// with a `ValueError` before any training, instead of a panic mid-tune).
pub(crate) fn validate_param_space(
    learner_type: &str,
    space: &smelt_ml::tuning::ParamSpace,
) -> PyResult<()> {
    use pyo3::exceptions::PyValueError;
    use smelt_ml::tuning::ParamDistribution;

    let allowed = factory_param_names(learner_type);
    let mut names: Vec<&str> = space.keys().map(String::as_str).collect();
    names.sort_unstable();
    for name in names {
        if !allowed.contains(&name) {
            return Err(PyValueError::new_err(format!(
                "parameter '{name}' is not tunable for learner '{learner_type}'; tunable \
                 parameters are: {}",
                allowed.join(", ")
            )));
        }
    }

    if let Some(dist) = space.get("objective") {
        let ParamDistribution::Choice(values) = dist else {
            return Err(PyValueError::new_err(
                "'objective' must be a choice list of strings, e.g. [\"squared_error\", \"huber\"]",
            ));
        };
        for v in values {
            let name = v.as_str().map_err(smelt_err)?;
            crate::learners::boosting::resolve_objective(name, 1.0)?;
        }
    }

    // `huber_delta` is only ever read inside the factory's `if let` on
    // `objective` when the sampled value is "huber" -- tuning it without
    // "huber" among the objective choices is a silent no-op where every
    // trial trains the identical model (5th audit M-5, the same class as
    // M-13 surviving inside the allowlist).
    if space.get("huber_delta").is_some() {
        let has_huber = matches!(
            space.get("objective"),
            Some(ParamDistribution::Choice(values))
                if values.iter().any(|v| v.as_str().is_ok_and(|s| s == "huber"))
        );
        if !has_huber {
            return Err(PyValueError::new_err(
                "'huber_delta' only takes effect when 'objective' is tuned to include \
                 \"huber\"; add e.g. \"objective\": [\"huber\"] (or [\"squared_error\", \
                 \"huber\"]) to the param space, or remove 'huber_delta' -- otherwise \
                 every trial trains the identical model and its value is ignored",
            ));
        }
    }

    Ok(())
}

/// Convert one Python value to a `ParamValue`, preserving its actual type
/// (audit issue M10: a Python `Choice` list used to be coerced to `Vec<f64>`
/// unconditionally, so a string choice like `["squared_error", "huber"]`
/// failed extraction outright -- there was no way to tune a string-valued
/// hyperparameter at all). Checked in order bool -> int -> float -> str
/// since Python's `bool` is itself an `int` subclass (a bare `int` check
/// first would silently read `True`/`False` as `1`/`0`).
fn py_to_param_value(v: &Bound<'_, PyAny>) -> PyResult<smelt_ml::tuning::ParamValue> {
    use smelt_ml::tuning::ParamValue;
    if let Ok(b) = v.extract::<bool>() {
        Ok(ParamValue::Bool(b))
    } else if let Ok(i) = v.extract::<i64>() {
        Ok(ParamValue::Int(i))
    } else if let Ok(f) = v.extract::<f64>() {
        Ok(ParamValue::Float(f))
    } else if let Ok(s) = v.extract::<String>() {
        Ok(ParamValue::Str(s))
    } else {
        Err(PyValueError::new_err(
            "choice values must be bool, int, float, or str",
        ))
    }
}

pub(crate) fn build_param_space(dict: &Bound<'_, PyAny>) -> PyResult<smelt_ml::tuning::ParamSpace> {
    use smelt_ml::tuning::{ParamDistribution, ParamSpace};

    let py_dict: &Bound<'_, pyo3::types::PyDict> = dict.downcast()
        .map_err(|_| PyValueError::new_err("param_space must be a dict"))?;

    let mut space = ParamSpace::new();

    for (key, val) in py_dict.iter() {
        let name: String = key.extract()?;

        // Accept dict format: {"type": "uniform", "low": 0.1, "high": 1.0}
        // Or tuple format: (low, high) → uniform
        // Or list format: [1, 2, 3] → choice
        if let Ok(inner_dict) = val.downcast::<pyo3::types::PyDict>() {
            let dtype: String = inner_dict.get_item("type")?
                .ok_or_else(|| PyValueError::new_err(format!("Missing 'type' for param '{name}'")))?
                .extract()?;
            let required = |field: &str| -> PyResult<Bound<'_, PyAny>> {
                inner_dict
                    .get_item(field)?
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
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
                    let values = required("values")?;
                    let items: Vec<Bound<'_, PyAny>> = values.extract()?;
                    let choices = items
                        .iter()
                        .map(py_to_param_value)
                        .collect::<PyResult<Vec<_>>>()?;
                    space.insert(name, ParamDistribution::Choice(choices));
                }
                _ => return Err(PyValueError::new_err(format!("Unknown param type: {dtype}"))),
            }
        } else if let Ok(tup) = val.extract::<(f64, f64)>() {
            // Shorthand: (low, high) → uniform
            space.insert(name, ParamDistribution::Uniform(tup.0, tup.1));
        } else if let Ok(items) = val.extract::<Vec<Bound<'_, PyAny>>>() {
            // Shorthand: [1, 2, 3] or ["a", "b"] → choice
            let choices = items
                .iter()
                .map(py_to_param_value)
                .collect::<PyResult<Vec<_>>>()?;
            space.insert(name, ParamDistribution::Choice(choices));
        } else {
            return Err(PyValueError::new_err(
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
    ///         Param names are validated against the learner's tunable set;
    ///         an unknown name raises ValueError listing the valid ones.
    ///         For "xgboost", `objective` (["squared_error", "huber",
    ///         "poisson"]) and `huber_delta` are tunable too; `huber_delta`
    ///         requires "huber" among the `objective` choices (it is
    ///         ignored by every other objective).
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
        validate_param_space(learner_type, &space)?;
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
            crate::common::check_finite_target(&target)?;
            let task = smelt_ml::task::RegressionTask::new("bo", features, target)
                .map_err(smelt_err)?;
            py.allow_threads(|| bo.tune_regress(&task, &cv, &*measure))
                .map_err(smelt_err)?
        };

        // Convert TuneResult to Python dict, casting integer params to int
        let dict = pyo3::types::PyDict::new(py);

        let bp = pyo3::types::PyDict::new(py);
        for (k, v) in &result.best_params {
            set_param(&bp, k, v)?;
        }
        dict.set_item("best_params", bp)?;
        dict.set_item("best_score", result.best_score)?;
        dict.set_item("measure", &result.measure_id)?;

        let history = pyo3::types::PyList::empty(py);
        for (params, score) in &result.all_results {
            let pd = pyo3::types::PyDict::new(py);
            for (k, v) in params {
                set_param(&pd, k, v)?;
            }
            let tup = pyo3::types::PyTuple::new(py, [pd.as_any(), score.into_pyobject(py)?.as_any()])?;
            history.append(tup)?;
        }
        dict.set_item("all_results", history)?;

        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }
}

/// Param names that are conceptually integer-valued and should be rounded
/// when sampled as a continuous `Float` (Uniform/LogUniform have no way to
/// carry "this is really an integer" on their own). Not consulted for
/// `Int`/`Bool`/`Str` values below -- those already carry their real type
/// (e.g. from an integer- or string-valued `Choice` list), so no name-based
/// guess is needed for them at all; this allowlist only exists for the
/// continuous-range case.
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

pub(crate) fn set_param(
    dict: &Bound<'_, pyo3::types::PyDict>,
    name: &str,
    value: &smelt_ml::tuning::ParamValue,
) -> PyResult<()> {
    use smelt_ml::tuning::ParamValue;
    match value {
        ParamValue::Int(v) => dict.set_item(name, v),
        ParamValue::Bool(v) => dict.set_item(name, v),
        ParamValue::Str(v) => dict.set_item(name, v),
        ParamValue::Float(v) => {
            if is_integer_param(name) {
                // TRUNCATE, matching the factory's `get(...) as usize`: the
                // reported best_params must be the value the winning model
                // was actually trained and scored with. Rounding here while
                // the factory truncated meant a Uniform draw of e.g. 3.7
                // trained max_depth=3 but reported max_depth=4 (audit M-12).
                dict.set_item(name, v.trunc() as i64)
            } else {
                dict.set_item(name, v)
            }
        }
    }
}

