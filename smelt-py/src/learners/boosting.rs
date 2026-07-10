//! Gradient boosting learners: XGBoost, CatBoost, LightGBM, GeoXGBoost,
//! KrigingHybrid.

use crate::common::{
    fit_learner_cat, not_fitted, parse_coords, parse_eval_set, perm_importance_impl,
    predict_proba_values, predict_values, shap_impl, smelt_err, to_array2, conformal_predict_impl,
    EvalKind,
};
use crate::common::{add_explain_methods, add_persistence_methods, declare_support, declare_params};
use crate::learners::ensemble::validate_learner_id;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use smelt_ml::learner::TrainedModel;
use smelt_ml::prediction::Prediction;

/// Map the Python-facing `variogram_model` string to `smelt_ml::prelude::VariogramModel`.
fn resolve_variogram_model(name: &str) -> PyResult<smelt_ml::prelude::VariogramModel> {
    use smelt_ml::prelude::VariogramModel;
    match name {
        "spherical" => Ok(VariogramModel::Spherical),
        "exponential" => Ok(VariogramModel::Exponential),
        "gaussian" => Ok(VariogramModel::Gaussian),
        other => Err(PyRuntimeError::new_err(format!(
            "unknown variogram_model '{other}'; expected one of: spherical, exponential, gaussian"
        ))),
    }
}

/// Map the Python-facing `objective` string to `smelt_ml::prelude::Objective`.
/// `Objective::Custom` (an arbitrary Rust closure) isn't exposed here --
/// bridging a Python callback into it would mean re-acquiring the GIL on
/// every gradient/hessian evaluation, the same cost/complexity trade-off
/// that keeps `Bagging`/`Stacking` on learner-id strings instead of Python
/// learner objects (see `ensemble.rs`).
fn resolve_objective(objective: &str, huber_delta: f64) -> PyResult<smelt_ml::prelude::Objective> {
    use smelt_ml::prelude::Objective;
    match objective {
        "squared_error" => Ok(Objective::SquaredError),
        "huber" => Ok(Objective::Huber { delta: huber_delta }),
        "poisson" => Ok(Objective::Poisson),
        other => Err(PyRuntimeError::new_err(format!(
            "unknown objective '{other}'; expected one of: squared_error, huber, poisson"
        ))),
    }
}

// ── XGBoost ────────────────────────────────────────────────────────────

#[pyclass]
#[derive(Default)]
pub(crate) struct XGBoost {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    n_estimators: usize,
    max_depth: usize,
    learning_rate: f64,
    lambda: f64,
    alpha: f64,
    gamma: f64,
    subsample: f64,
    colsample_bytree: f64,
    seed: u64,
    monotone_constraints: Option<Vec<i8>>,
    objective: String,
    huber_delta: f64,
}

#[pymethods]
impl XGBoost {
    /// `monotone_constraints`: one of -1/0/+1 per feature (+1 = non-decreasing,
    /// -1 = non-increasing, 0 = unconstrained), checked against the number of
    /// features at fit time. `objective`: one of "squared_error" (default),
    /// "huber" (uses `huber_delta`), or "poisson"; only affects regression.
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=6, learning_rate=0.3, lambda_=1.0, alpha=0.0, gamma=0.0, subsample=1.0, colsample_bytree=1.0, seed=42, monotone_constraints=None, objective="squared_error".to_string(), huber_delta=1.0))]
    fn new(
        n_estimators: usize,
        max_depth: usize,
        learning_rate: f64,
        lambda_: f64,
        alpha: f64,
        gamma: f64,
        subsample: f64,
        colsample_bytree: f64,
        seed: u64,
        monotone_constraints: Option<Vec<i8>>,
        objective: String,
        huber_delta: f64,
    ) -> Self {
        Self {
            trained: None,
            is_classif: false,
            n_estimators,
            max_depth,
            learning_rate,
            lambda: lambda_,
            alpha,
            gamma,
            subsample,
            colsample_bytree,
            seed,
            monotone_constraints,
            objective,
            huber_delta,
        }
    }

    /// `cat_features`: column indices to treat as categorical (native Fisher
    /// splits instead of numeric thresholds). `eval_set`: `(x_val, y_val)`
    /// held out for `early_stopping_rounds` to monitor; without it, early
    /// stopping watches training loss, which rarely plateaus under boosting.
    #[pyo3(signature = (x, y, cat_features=None, eval_set=None, early_stopping_rounds=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        cat_features: Option<Vec<usize>>,
        eval_set: Option<(PyReadonlyArray2<'_, f64>, Bound<'_, PyAny>)>,
        early_stopping_rounds: Option<usize>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::XGBoost::new()
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_learning_rate(self.learning_rate)
            .with_lambda(self.lambda)
            .with_alpha(self.alpha)
            .with_gamma(self.gamma)
            .with_subsample(self.subsample)
            .with_colsample_bytree(self.colsample_bytree)
            .with_seed(self.seed)
            .with_objective(resolve_objective(&self.objective, self.huber_delta)?);
        if let Some(c) = self.monotone_constraints.clone() {
            learner = learner.with_monotone_constraints(c);
        }
        if let Some(rounds) = early_stopping_rounds {
            learner = learner.with_early_stopping_rounds(rounds);
        }
        if let Some((features, target)) = parse_eval_set(eval_set)? {
            learner = match target {
                EvalKind::Classification(t) => learner.with_eval_set_classif(features, t),
                EvalKind::Regression(t) => learner.with_eval_set_regress(features, t),
            };
        }
        let (model, is_classif) =
            fit_learner_cat(py, &mut learner, to_array2(x), y, cat_features)?;
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

// ── CatBoost ───────────────────────────────────────────────────────────

#[pyclass]
#[derive(Default)]
pub(crate) struct CatBoost {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    n_estimators: usize,
    depth: usize,
    learning_rate: f64,
    lambda: f64,
    seed: u64,
}

#[pymethods]
impl CatBoost {
    #[new]
    #[pyo3(signature = (n_estimators=100, depth=6, learning_rate=0.3, lambda_=1.0, seed=42))]
    fn new(n_estimators: usize, depth: usize, learning_rate: f64, lambda_: f64, seed: u64) -> Self {
        Self {
            trained: None,
            is_classif: false,
            n_estimators,
            depth,
            learning_rate,
            lambda: lambda_,
            seed,
        }
    }

    /// `cat_features`: column indices to treat as categorical (CatBoost's own
    /// target-statistic splits, falling back to whichever columns are passed
    /// here since this wrapper always trains against a plain `Task`).
    /// `eval_set`: `(x_val, y_val)` held out for `early_stopping_rounds`.
    #[pyo3(signature = (x, y, cat_features=None, eval_set=None, early_stopping_rounds=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        cat_features: Option<Vec<usize>>,
        eval_set: Option<(PyReadonlyArray2<'_, f64>, Bound<'_, PyAny>)>,
        early_stopping_rounds: Option<usize>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::CatBoost::new()
            .with_n_estimators(self.n_estimators)
            .with_depth(self.depth)
            .with_learning_rate(self.learning_rate)
            .with_lambda(self.lambda)
            .with_seed(self.seed);
        if let Some(rounds) = early_stopping_rounds {
            learner = learner.with_early_stopping_rounds(rounds);
        }
        if let Some((features, target)) = parse_eval_set(eval_set)? {
            learner = match target {
                EvalKind::Classification(t) => learner.with_eval_set_classif(features, t),
                EvalKind::Regression(t) => learner.with_eval_set_regress(features, t),
            };
        }
        let (model, is_classif) =
            fit_learner_cat(py, &mut learner, to_array2(x), y, cat_features)?;
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

// ── LightGBM ───────────────────────────────────────────────────────────

#[pyclass]
#[derive(Default)]
pub(crate) struct LightGBM {
    trained: Option<Box<dyn TrainedModel>>,
    is_classif: bool,
    n_estimators: usize,
    num_leaves: usize,
    learning_rate: f64,
    max_depth: usize,
    seed: u64,
}

#[pymethods]
impl LightGBM {
    #[new]
    #[pyo3(signature = (n_estimators=100, num_leaves=31, learning_rate=0.1, max_depth=6, seed=42))]
    fn new(n_estimators: usize, num_leaves: usize, learning_rate: f64, max_depth: usize, seed: u64) -> Self {
        Self { trained: None, is_classif: false, n_estimators, num_leaves, learning_rate, max_depth, seed }
    }

    /// `cat_features`: column indices to treat as categorical (native Fisher
    /// splits). `eval_set`: `(x_val, y_val)` held out for
    /// `early_stopping_rounds`.
    #[pyo3(signature = (x, y, cat_features=None, eval_set=None, early_stopping_rounds=None))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        cat_features: Option<Vec<usize>>,
        eval_set: Option<(PyReadonlyArray2<'_, f64>, Bound<'_, PyAny>)>,
        early_stopping_rounds: Option<usize>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::LightGBM::new()
            .with_n_estimators(self.n_estimators)
            .with_num_leaves(self.num_leaves)
            .with_learning_rate(self.learning_rate)
            .with_max_depth(self.max_depth)
            .with_seed(self.seed);
        if let Some(rounds) = early_stopping_rounds {
            learner = learner.with_early_stopping_rounds(rounds);
        }
        if let Some((features, target)) = parse_eval_set(eval_set)? {
            learner = match target {
                EvalKind::Classification(t) => learner.with_eval_set_classif(features, t),
                EvalKind::Regression(t) => learner.with_eval_set_regress(features, t),
            };
        }
        let (model, is_classif) =
            fit_learner_cat(py, &mut learner, to_array2(x), y, cat_features)?;
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

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        Ok(self.trained.as_ref().ok_or_else(not_fitted)?.feature_importance())
    }
}


// ── GeoXGBoost ─────────────────────────────────────────────────────────

#[pyclass]
pub(crate) struct GeoXGBoost {
    trained: Option<smelt_ml::prelude::TrainedGeoXGBoost>,
    bandwidth: usize,
    n_estimators: usize,
    max_depth: usize,
    learning_rate: f64,
    lambda: f64,
    alpha: Option<f64>,
    seed: u64,
}

#[pymethods]
impl GeoXGBoost {
    /// Geographical-XGBoost for spatially-local regression (Grekousis, 2025).
    ///
    /// Args:
    ///     bandwidth: number of nearest neighbours for adaptive bi-square kernel.
    ///     n_estimators, max_depth, learning_rate, lambda_: XGBoost hyperparameters
    ///         used for both global and local models.
    ///     alpha: None (default) for adaptive blending, 1.0 for pure local,
    ///         0.0 for pure global.
    ///     seed: random seed.
    #[new]
    #[pyo3(signature = (bandwidth=30, n_estimators=100, max_depth=6, learning_rate=0.3, lambda_=1.0, alpha=None, seed=42))]
    fn new(
        bandwidth: usize,
        n_estimators: usize,
        max_depth: usize,
        learning_rate: f64,
        lambda_: f64,
        alpha: Option<f64>,
        seed: u64,
    ) -> Self {
        Self {
            trained: None,
            bandwidth,
            n_estimators,
            max_depth,
            learning_rate,
            lambda: lambda_,
            alpha,
            seed,
        }
    }

    /// Train the model. `coords` is an (N, 2) array-like of (x, y) per sample.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: Vec<f64>,
        coords: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let parsed = parse_coords(coords)?;
        let features = to_array2(x);
        if parsed.len() != features.nrows() {
            return Err(PyRuntimeError::new_err(format!(
                "coords length ({}) must match number of samples ({})",
                parsed.len(),
                features.nrows()
            )));
        }
        let task = smelt_ml::task::RegressionTask::new("gxgb", features, y).map_err(smelt_err)?;
        let mut learner = smelt_ml::prelude::GeoXGBoost::new(parsed)
            .with_bandwidth(self.bandwidth)
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_learning_rate(self.learning_rate)
            .with_lambda(self.lambda)
            .with_seed(self.seed);
        if let Some(a) = self.alpha {
            learner = learner.with_alpha(a);
        }
        let trained = py
            .allow_threads(|| learner.train_geo(&task))
            .map_err(smelt_err)?;
        self.trained = Some(trained);
        Ok(())
    }

    /// Select the optimal bandwidth by minimising the leave-one-out CV criterion
    /// of Grekousis (2025, Eq. 11).
    ///
    /// For every candidate bandwidth, each location is predicted by a *local*
    /// model fit on its neighbours excluding itself; the criterion is the
    /// leave-one-out RMSE. The bandwidth minimising it is returned and stored on
    /// this estimator, so a subsequent `fit(...)` uses it automatically.
    ///
    /// This is a property of the local model only: the global model and `alpha`
    /// are not involved, because bandwidth is tuned before the ensemble step.
    ///
    /// Args:
    ///     x, y, coords: training data (same as `fit`).
    ///     candidates: list of bandwidths to try. Defaults to
    ///         [30, 50, 100, 150, 200, 300, 400] (filtered to values < n).
    ///
    /// Returns:
    ///     dict with keys ``best`` (int), ``bandwidths`` (list[int]) and
    ///     ``cv`` (list[float], the LOO criterion), aligned by index.
    #[pyo3(signature = (x, y, coords, candidates=None))]
    fn select_bandwidth(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: Vec<f64>,
        coords: &Bound<'_, PyAny>,
        candidates: Option<Vec<usize>>,
    ) -> PyResult<PyObject> {
        let parsed = parse_coords(coords)?;
        let features = to_array2(x);
        let n = features.nrows();
        if parsed.len() != n {
            return Err(PyRuntimeError::new_err(format!(
                "coords length ({}) must match number of samples ({})",
                parsed.len(),
                n
            )));
        }
        let grid: Vec<usize> = candidates
            .unwrap_or_else(|| vec![30, 50, 100, 150, 200, 300, 400])
            .into_iter()
            .filter(|&bw| bw > 0 && bw < n)
            .collect();
        if grid.is_empty() {
            return Err(PyRuntimeError::new_err(
                "no valid candidate bandwidths (all >= number of samples?)",
            ));
        }

        let task = smelt_ml::task::RegressionTask::new("gxgb", features, y).map_err(smelt_err)?;
        let learner = smelt_ml::prelude::GeoXGBoost::new(parsed)
            .with_n_estimators(self.n_estimators)
            .with_max_depth(self.max_depth)
            .with_learning_rate(self.learning_rate)
            .with_lambda(self.lambda)
            .with_seed(self.seed);
        let sel = py
            .allow_threads(|| learner.select_bandwidth(&task, &grid))
            .map_err(smelt_err)?;

        // Store the selected bandwidth so the next fit() uses it.
        self.bandwidth = sel.best;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("best", sel.best)?;
        let bws: Vec<usize> = sel.scores.iter().map(|&(bw, _)| bw).collect();
        let cv: Vec<f64> = sel.scores.iter().map(|&(_, r)| r).collect();
        dict.set_item("bandwidths", bws)?;
        dict.set_item("cv", cv)?;
        Ok(dict.into())
    }

    /// Predict. If `coords` is provided, uses per-sample nearest local model
    /// (spatial prediction); otherwise falls back to in-sample / global model.
    #[pyo3(signature = (x, coords=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        coords: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        use smelt_ml::learner::TrainedModel;
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let features = to_array2(x);
        let new_coords = coords.map(parse_coords).transpose()?;
        if let Some(c) = &new_coords {
            if c.len() != features.nrows() {
                return Err(PyRuntimeError::new_err(format!(
                    "coords length ({}) must match number of samples ({})",
                    c.len(),
                    features.nrows()
                )));
            }
        }
        let pred = py
            .allow_threads(|| match &new_coords {
                Some(c) => model.predict_spatial(&features, c),
                None => model.predict(&features),
            })
            .map_err(smelt_err)?;
        let values: Vec<f64> = match &pred {
            Prediction::Regression { predicted, .. } => predicted.clone(),
            _ => return Err(PyRuntimeError::new_err("Expected regression prediction")),
        };
        Ok(PyArray1::from_vec(py, values))
    }

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        use smelt_ml::learner::TrainedModel;
        Ok(self
            .trained
            .as_ref()
            .ok_or_else(not_fitted)?
            .feature_importance())
    }

    /// Per-location local-model feature importances, for mapping spatial
    /// non-stationarity (how each predictor's influence varies across space).
    ///
    /// Returns a dict with:
    ///   - ``coords``: (N, 2) float array of training (x, y) coordinates,
    ///   - ``importances``: list of length N; entry i is a dict
    ///     {feature_name: gain} for location i's local model, or None where the
    ///     neighbourhood was too small (global-model fallback).
    /// Feature names are the internal x0, x1, ... order.
    fn local_feature_importances<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let coords = model.coords();
        let imps = model.local_importances();

        let n = coords.len();
        let flat: Vec<f64> = coords.iter().flat_map(|&(x, y)| [x, y]).collect();
        let coord_arr = ndarray::Array2::from_shape_vec((n, 2), flat)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let imp_list = pyo3::types::PyList::empty(py);
        for entry in imps {
            match entry {
                Some(v) => {
                    let d = pyo3::types::PyDict::new(py);
                    for (name, gain) in v {
                        d.set_item(name, gain)?;
                    }
                    imp_list.append(d)?;
                }
                None => imp_list.append(py.None())?,
            }
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coords", PyArray2::from_owned_array(py, coord_arr))?;
        dict.set_item("importances", imp_list)?;
        Ok(dict.into())
    }

    #[getter]
    fn supports_classification(&self) -> bool { false }

    #[getter]
    fn supports_regression(&self) -> bool { true }

    /// SHAP values (interventional). Uses the global model internally.
    #[pyo3(signature = (x, y, n_background=50, feature_names=None))]
    fn shap_values<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        n_background: usize,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        shap_impl(py, model, false, x, y, n_background, feature_names, 0)
    }

    /// Permutation importance.
    #[pyo3(signature = (x, y, metric="rmse", n_repeats=5, seed=42, feature_names=None))]
    fn permutation_importance<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
        metric: &str,
        n_repeats: usize,
        seed: u64,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        perm_importance_impl(py, model, false, x, y, metric, n_repeats, seed, feature_names)
    }

    /// Split conformal prediction intervals. Calibration data may include
    /// their own `coords` for spatially-aware predictions on the calibration set;
    /// if omitted, the global model is used for calibration.
    #[pyo3(signature = (x_cal, y_cal, x_test, alpha=0.1))]
    fn conformal_predict<'py>(
        &self,
        py: Python<'py>,
        x_cal: PyReadonlyArray2<'_, f64>,
        y_cal: Vec<f64>,
        x_test: PyReadonlyArray2<'_, f64>,
        alpha: f64,
    ) -> PyResult<PyObject> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        conformal_predict_impl(py, model, x_cal, y_cal, x_test, alpha)
    }
}


// ── KrigingHybrid ──────────────────────────────────────────────────────
//
// Wraps a base learner (selected by id string, same rationale as
// `Bagging`/`Stacking` in `ensemble.rs`: bridging a Python object back
// into the `Fn() -> Box<dyn Learner>` factory the Rust struct needs would
// mean re-acquiring the GIL on internal calls) plus ordinary kriging of
// its residuals -- regression-only, so no `predict_proba`/`is_classif`.

#[pyclass]
pub(crate) struct KrigingHybrid {
    trained: Option<smelt_ml::prelude::TrainedKrigingHybrid>,
    base: String,
    variogram_model: String,
    n_lags: usize,
    n_neighbors: usize,
}

#[pymethods]
impl KrigingHybrid {
    /// Regression-kriging: trains `base` then krige-interpolates its
    /// residuals spatially, combining as `base(x) + kriged_residual(coords)`.
    ///
    /// Args:
    ///     base: base learner id string (see `smelt.registered_learner_ids()`).
    ///     variogram_model: "spherical" (default), "exponential", or "gaussian".
    ///     n_lags: number of lag bins used to build the empirical variogram.
    ///     n_neighbors: local kriging neighborhood size at predict time.
    #[new]
    #[pyo3(signature = (base, variogram_model="spherical".to_string(), n_lags=15, n_neighbors=20))]
    fn new(base: String, variogram_model: String, n_lags: usize, n_neighbors: usize) -> PyResult<Self> {
        validate_learner_id(&base)?;
        Ok(Self {
            trained: None,
            base,
            variogram_model,
            n_lags,
            n_neighbors,
        })
    }

    /// Train the model. `coords` is an (N, 2) array-like of (x, y) per sample.
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: Vec<f64>,
        coords: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let parsed = parse_coords(coords)?;
        let features = to_array2(x);
        if parsed.len() != features.nrows() {
            return Err(PyRuntimeError::new_err(format!(
                "coords length ({}) must match number of samples ({})",
                parsed.len(),
                features.nrows()
            )));
        }
        let model_kind = resolve_variogram_model(&self.variogram_model)?;
        let task =
            smelt_ml::task::RegressionTask::new("kriging_hybrid", features, y).map_err(smelt_err)?;
        let base = self.base.clone();
        let mut learner = smelt_ml::prelude::KrigingHybrid::new(
            move || smelt_ml::prelude::learner_from_id(&base).expect("validated in KrigingHybrid::new"),
            parsed,
        )
        .with_variogram_model(model_kind)
        .with_n_lags(self.n_lags)
        .with_n_neighbors(self.n_neighbors);
        let trained = py
            .allow_threads(|| learner.train_regress_geo(&task))
            .map_err(smelt_err)?;
        self.trained = Some(trained);
        Ok(())
    }

    /// Predict. If `coords` is provided, applies the kriging correction
    /// (`base(x) + kriged_residual(coords)`); otherwise returns the base
    /// model's prediction alone (no spatial correction).
    #[pyo3(signature = (x, coords=None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        coords: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        use smelt_ml::learner::TrainedModel;
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let features = to_array2(x);
        let new_coords = coords.map(parse_coords).transpose()?;
        if let Some(c) = &new_coords {
            if c.len() != features.nrows() {
                return Err(PyRuntimeError::new_err(format!(
                    "coords length ({}) must match number of samples ({})",
                    c.len(),
                    features.nrows()
                )));
            }
        }
        let pred = py
            .allow_threads(|| match &new_coords {
                Some(c) => model.predict_spatial(&features, c),
                None => model.predict(&features),
            })
            .map_err(smelt_err)?;
        let values: Vec<f64> = match &pred {
            Prediction::Regression { predicted, .. } => predicted.clone(),
            _ => return Err(PyRuntimeError::new_err("Expected regression prediction")),
        };
        Ok(PyArray1::from_vec(py, values))
    }

    /// Fitted variogram parameters: dict with `nugget`, `sill`, `range`
    /// (floats) and `model` (the model family string).
    #[getter]
    fn variogram_fit_(&self, py: Python<'_>) -> PyResult<PyObject> {
        let model = self.trained.as_ref().ok_or_else(not_fitted)?;
        let fit = model.variogram_fit();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("nugget", fit.nugget)?;
        dict.set_item("sill", fit.sill)?;
        dict.set_item("range", fit.range)?;
        dict.set_item("model", format!("{:?}", fit.model).to_lowercase())?;
        Ok(dict.into())
    }

    #[getter]
    fn feature_importances_(&self) -> PyResult<Option<Vec<(String, f64)>>> {
        use smelt_ml::learner::TrainedModel;
        Ok(self
            .trained
            .as_ref()
            .ok_or_else(not_fitted)?
            .feature_importance())
    }

    #[getter]
    fn supports_classification(&self) -> bool { false }

    #[getter]
    fn supports_regression(&self) -> bool { true }
}

add_explain_methods!(XGBoost, CatBoost, LightGBM);

declare_support!(XGBoost,  classif = true, regress = true);
declare_support!(CatBoost, classif = true, regress = true);
declare_support!(LightGBM, classif = true, regress = true);

declare_params!(XGBoost, {
    n_estimators => "n_estimators",
    max_depth => "max_depth",
    learning_rate => "learning_rate",
    lambda => "lambda_",
    alpha => "alpha",
    gamma => "gamma",
    subsample => "subsample",
    colsample_bytree => "colsample_bytree",
    seed => "seed",
    monotone_constraints => "monotone_constraints",
    objective => "objective",
    huber_delta => "huber_delta",
});

declare_params!(CatBoost, {
    n_estimators => "n_estimators",
    depth => "depth",
    learning_rate => "learning_rate",
    lambda => "lambda_",
    seed => "seed",
});

declare_params!(LightGBM, {
    n_estimators => "n_estimators",
    num_leaves => "num_leaves",
    learning_rate => "learning_rate",
    max_depth => "max_depth",
    seed => "seed",
});

// GeoXGBoost/KrigingHybrid are excluded here: they hold their trained model
// as a concrete `Option<TrainedGeoXGBoost>`/`Option<TrainedKrigingHybrid>`
// (not `Option<Box<dyn TrainedModel>>`, since both expose an inherent
// `predict_spatial` beyond the `TrainedModel` trait), and both are already
// excluded from `SerializableModel` (see `src/serialize.rs`) since they hold
// `Box<dyn TrainedModel>` internally.
add_persistence_methods!(
    XGBoost => "XGBoost",
    CatBoost => "CatBoost",
    LightGBM => "LightGBM",
);

declare_params!(GeoXGBoost, {
    bandwidth => "bandwidth",
    n_estimators => "n_estimators",
    max_depth => "max_depth",
    learning_rate => "learning_rate",
    lambda => "lambda_",
    alpha => "alpha",
    seed => "seed",
});

// `base` is a learner id string re-validated on `set_params` (same reason
// `Bagging`/`Stacking` in `ensemble.rs` hand-write `get_params`/`set_params`
// instead of using `declare_params!`: the macro can't express re-running
// `validate_learner_id`, and a bad id would otherwise only surface as the
// `.expect("validated in KrigingHybrid::new")` panic in `fit()`).
#[pymethods]
impl KrigingHybrid {
    fn get_params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("base", self.base.clone())?;
        dict.set_item("variogram_model", self.variogram_model.clone())?;
        dict.set_item("n_lags", self.n_lags)?;
        dict.set_item("n_neighbors", self.n_neighbors)?;
        Ok(dict.into_pyobject(py)?.into_any().unbind())
    }

    #[pyo3(signature = (**kwargs))]
    fn set_params(&mut self, kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            for (k, v) in kwargs.iter() {
                let key: String = k.extract()?;
                match key.as_str() {
                    "base" => {
                        let base: String = v.extract()?;
                        validate_learner_id(&base)?;
                        self.base = base;
                    }
                    "variogram_model" => {
                        let m: String = v.extract()?;
                        resolve_variogram_model(&m)?;
                        self.variogram_model = m;
                    }
                    "n_lags" => self.n_lags = v.extract()?,
                    "n_neighbors" => self.n_neighbors = v.extract()?,
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
