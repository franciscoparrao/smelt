//! `AutoTuner`: bundle a base learner + tuner + parameter space as a single
//! sklearn-`GridSearchCV`-style estimator with `fit`/`predict`/`get_params`/
//! `set_params` and post-fit `best_params_`/`best_score_` properties.
//!
//! Reuses the tuning infra in [`crate::tuning`] wholesale: the base learner is
//! selected by the same id strings as `BayesianOptimizer.optimize`
//! (`make_learner_factory`), the `param_space` is parsed by the same
//! `build_param_space`, validated by the same allowlist + `huber_delta` rule
//! (`validate_param_space`), and `best_params_` is rendered with the same
//! integer-truncating `set_param`. The only new surface is choosing among the
//! four tuners and wrapping the whole thing as a fittable estimator instead of
//! a one-shot `optimize()` call.

use crate::common::{
    check_finite_target, extract_class_labels, is_integer, not_fitted, predict_proba_values,
    predict_values, resolve_measure, save_model, smelt_err, to_array2, validate_sample_weight,
};
use crate::tuning::{build_param_space, make_learner_factory, set_param, validate_param_space};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use smelt_ml::learner::{AutoTuner as CoreAutoTuner, TrainedModel, TunerSpec};
use smelt_ml::tuning::{ParamDistribution, ParamGrid, ParamSet, ParamSpace};

/// Convert a distribution `ParamSpace` into a discrete `ParamGrid` for the
/// grid tuner: every entry must be a `Choice` list (a continuous
/// `Uniform`/`LogUniform` range cannot be exhaustively enumerated).
fn space_to_grid(space: &ParamSpace) -> PyResult<ParamGrid> {
    let mut grid = ParamGrid::new();
    for (name, dist) in space {
        match dist {
            ParamDistribution::Choice(values) => {
                grid.insert(name.clone(), values.clone());
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "tuner='grid' requires every parameter to be a discrete list of values; \
                     parameter '{name}' is a continuous range — pass it as a list \
                     (e.g. [2, 4, 8]) or switch to tuner='random'/'bayesian'"
                )));
            }
        }
    }
    Ok(grid)
}

/// A sklearn-`GridSearchCV`-style auto-tuning estimator: tunes a base learner's
/// hyperparameters with the chosen `tuner` over `param_space` via inner
/// cross-validation, then refits the winning configuration on the full data.
#[pyclass]
pub(crate) struct AutoTuner {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    learner: String,
    param_space: Py<PyAny>,
    tuner: String,
    n_iter: usize,
    n_initial: usize,
    eta: usize,
    cv: usize,
    metric: String,
    seed: u64,
    best_params: Option<ParamSet>,
    best_score: Option<f64>,
}

impl AutoTuner {
    /// Build the inner `ParamSpace` from the stored Python `param_space`,
    /// validated against the base learner's tunable set (+ `huber_delta` rule).
    fn build_space(&self, py: Python<'_>) -> PyResult<ParamSpace> {
        let bound = self.param_space.bind(py);
        let space = build_param_space(bound)?;
        validate_param_space(&self.learner, &space)?;
        Ok(space)
    }

    /// Assemble the Rust [`CoreAutoTuner`] from the current configuration.
    fn build_core(&self, py: Python<'_>) -> PyResult<CoreAutoTuner> {
        let factory = make_learner_factory(&self.learner)?;
        let space = self.build_space(py)?;

        let spec = match self.tuner.as_str() {
            "grid" => TunerSpec::Grid(space_to_grid(&space)?),
            "random" => TunerSpec::Random {
                space,
                n_iter: self.n_iter,
            },
            "bayesian" | "bo" => TunerSpec::Bayesian {
                space,
                n_iter: self.n_iter,
                n_initial: self.n_initial,
            },
            "hyperband" | "hb" => TunerSpec::Hyperband {
                space,
                max_folds: self.cv,
                eta: self.eta,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown tuner '{other}'; valid tuners: grid, random, bayesian, hyperband"
                )));
            }
        };

        let resampling = smelt_ml::resample::CrossValidation::new(self.cv).with_seed(self.seed);
        let measure = resolve_measure(&self.metric)?;
        Ok(
            CoreAutoTuner::new(move |p| factory(p), spec, Box::new(resampling), measure)
                .with_seed(self.seed),
        )
    }
}

#[pymethods]
impl AutoTuner {
    /// Args:
    ///     learner: base learner id ("xgboost", "rf", "catboost", "lightgbm",
    ///         "dt", "ridge", "knn", ...) — same set as `BayesianOptimizer`.
    ///     param_space: dict of param → spec, same format as
    ///         `BayesianOptimizer.optimize` ((low, high) → uniform, [v1, v2]
    ///         → choice, or a {"type": ...} dict). For tuner="grid" every
    ///         entry must be a discrete list.
    ///     tuner: "random" (default), "grid", "bayesian", or "hyperband".
    ///     n_iter: configurations to evaluate (random/bayesian; ignored by
    ///         grid/hyperband).
    ///     n_initial: initial random configs before TPE (bayesian only).
    ///     eta: successive-halving rate (hyperband only, >= 2).
    ///     cv: inner cross-validation folds (also hyperband's fold budget).
    ///     metric: measure to optimize ("rmse", "r2", "accuracy", "f1", ...).
    ///     seed: RNG seed (tuner + CV splits); reproducible best_params.
    #[new]
    #[pyo3(signature = (
        learner,
        param_space,
        tuner="random",
        n_iter=20,
        n_initial=5,
        eta=3,
        cv=5,
        metric="rmse",
        seed=42,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        learner: String,
        param_space: Py<PyAny>,
        tuner: &str,
        n_iter: usize,
        n_initial: usize,
        eta: usize,
        cv: usize,
        metric: &str,
        seed: u64,
    ) -> PyResult<Self> {
        // Validate eagerly (like Bagging's base id): a bad learner id or tuner
        // name must fail at construction, not deep inside fit().
        let _ = make_learner_factory(&learner)?;
        if !matches!(
            tuner,
            "random" | "grid" | "bayesian" | "bo" | "hyperband" | "hb"
        ) {
            return Err(PyValueError::new_err(format!(
                "unknown tuner '{tuner}'; valid tuners: grid, random, bayesian, hyperband"
            )));
        }
        Ok(Self {
            trained: None,
            is_classif: false,
            learner,
            param_space,
            tuner: tuner.to_string(),
            n_iter,
            n_initial,
            eta,
            cv,
            metric: metric.to_string(),
            seed,
            best_params: None,
            best_score: None,
        })
    }

    /// Tune, then refit the winning configuration on all of `(x, y)`.
    ///
    /// `sample_weight` (sklearn convention): optional per-sample weights,
    /// validated in the binding (length == n_samples, finite, >= 0, not all
    /// zero) and flowed through to the inner folds + final refit. The base
    /// learner must support weights (e.g. "ridge", "rf"); a weight-blind base
    /// ("knn") rejects them with a clear error.
    #[pyo3(signature = (x, y, sample_weight=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        sample_weight: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let features = to_array2(x);
        // Validate BEFORE any Task construction: with_weights would panic on
        // exactly these conditions (see validate_sample_weight's docs).
        if let Some(w) = &sample_weight {
            validate_sample_weight(w, features.nrows())?;
        }
        let core = self.build_core(py)?;

        // Use the AutoTuner's concrete `fit_classif`/`fit_regress` (they carry
        // best_params/best_score, unlike the boxed `Learner::train_*`), so the
        // task/weight plumbing is inlined here rather than routed through the
        // trait-erasing `fit_learner`.
        if is_integer(y) {
            let target = extract_class_labels(y)?;
            let mut task = smelt_ml::task::ClassificationTask::new("py", features, target)
                .map_err(smelt_err)?;
            if let Some(w) = sample_weight {
                task = task.with_weights(w);
            }
            let trained = py
                .allow_threads(|| core.fit_classif(&task))
                .map_err(smelt_err)?;
            self.best_params = Some(trained.best_params().clone());
            self.best_score = Some(trained.best_score());
            self.trained = Some(Box::new(trained));
            self.is_classif = true;
        } else {
            let target: Vec<f64> = y.extract()?;
            check_finite_target(&target)?;
            let mut task =
                smelt_ml::task::RegressionTask::new("py", features, target).map_err(smelt_err)?;
            if let Some(w) = sample_weight {
                task = task.with_weights(w);
            }
            let trained = py
                .allow_threads(|| core.fit_regress(&task))
                .map_err(smelt_err)?;
            self.best_params = Some(trained.best_params().clone());
            self.best_score = Some(trained.best_score());
            self.trained = Some(Box::new(trained));
            self.is_classif = false;
        }
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

    /// The hyperparameter configuration the inner tuner selected (the one the
    /// refit model was trained with). Raises if not fitted yet.
    #[getter]
    fn best_params_(&self, py: Python<'_>) -> PyResult<PyObject> {
        let bp = self.best_params.as_ref().ok_or_else(not_fitted)?;
        let dict = PyDict::new(py);
        for (k, v) in bp {
            set_param(&dict, k, v)?;
        }
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    /// The inner-CV score of `best_params_` (in the measure's own direction).
    #[getter]
    fn best_score_(&self) -> PyResult<f64> {
        self.best_score.ok_or_else(not_fitted)
    }

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        Ok(self
            .trained
            .as_ref()
            .ok_or_else(not_fitted)?
            .feature_importance())
    }

    /// Always raises `NotImplementedError`: an AutoTuner holds a factory-built
    /// base model internally (a composite), which smelt's serialization format
    /// has no variant for — re-fit instead of persisting.
    fn save(&self, path: &str) -> PyResult<()> {
        save_model(&self.trained, path)
    }

    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("learner", self.learner.clone())?;
        dict.set_item("param_space", self.param_space.clone_ref(py))?;
        dict.set_item("tuner", self.tuner.clone())?;
        dict.set_item("n_iter", self.n_iter)?;
        dict.set_item("n_initial", self.n_initial)?;
        dict.set_item("eta", self.eta)?;
        dict.set_item("cv", self.cv)?;
        dict.set_item("metric", self.metric.clone())?;
        dict.set_item("seed", self.seed)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let Some(kwargs) = kwargs else {
            return Ok(());
        };
        for (k, v) in kwargs.iter() {
            let key: String = k.extract()?;
            match key.as_str() {
                "learner" => {
                    let learner: String = v.extract()?;
                    // Re-validate eagerly, exactly as `new` does.
                    let _ = make_learner_factory(&learner)?;
                    self.learner = learner;
                }
                "param_space" => self.param_space = v.unbind(),
                "tuner" => {
                    let tuner: String = v.extract()?;
                    if !matches!(
                        tuner.as_str(),
                        "random" | "grid" | "bayesian" | "bo" | "hyperband" | "hb"
                    ) {
                        return Err(PyValueError::new_err(format!(
                            "unknown tuner '{tuner}'; valid tuners: grid, random, bayesian, hyperband"
                        )));
                    }
                    self.tuner = tuner;
                }
                "n_iter" => self.n_iter = v.extract()?,
                "n_initial" => self.n_initial = v.extract()?,
                "eta" => self.eta = v.extract()?,
                "cv" => self.cv = v.extract()?,
                "metric" => self.metric = v.extract()?,
                "seed" => self.seed = v.extract()?,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "invalid parameter '{other}' for this estimator"
                    )));
                }
            }
        }
        Ok(())
    }
}
