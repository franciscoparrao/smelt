//! Gradient boosting learners: XGBoost, CatBoost, LightGBM, GeoXGBoost.

use crate::common::{
    fit_learner, not_fitted, parse_coords, perm_importance_impl, predict_proba_values,
    predict_values, shap_impl, smelt_err, to_array2, conformal_predict_impl,
};
use crate::common::{add_explain_methods, declare_support};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use smelt_ml::learner::TrainedModel;
use smelt_ml::prediction::Prediction;

// ── XGBoost ────────────────────────────────────────────────────────────

#[pyclass]
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
}

#[pymethods]
impl XGBoost {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=6, learning_rate=0.3, lambda_=1.0, alpha=0.0, gamma=0.0, subsample=1.0, colsample_bytree=1.0, seed=42))]
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
        }
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
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
        Ok(self
            .trained
            .as_ref()
            .ok_or_else(not_fitted)?
            .feature_importance())
    }
}

// ── CatBoost ───────────────────────────────────────────────────────────

#[pyclass]
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

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::CatBoost::new()
            .with_n_estimators(self.n_estimators)
            .with_depth(self.depth)
            .with_learning_rate(self.learning_rate)
            .with_lambda(self.lambda)
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
}

// ── LightGBM ───────────────────────────────────────────────────────────

#[pyclass]
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

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<'_, f64>,
        y: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut learner = smelt_ml::prelude::LightGBM::new()
            .with_n_estimators(self.n_estimators)
            .with_num_leaves(self.num_leaves)
            .with_learning_rate(self.learning_rate)
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
        let pred = if let Some(c) = coords {
            let new_coords = parse_coords(c)?;
            if new_coords.len() != features.nrows() {
                return Err(PyRuntimeError::new_err(format!(
                    "coords length ({}) must match number of samples ({})",
                    new_coords.len(),
                    features.nrows()
                )));
            }
            model.predict_spatial(&features, &new_coords).map_err(smelt_err)?
        } else {
            model.predict(&features).map_err(smelt_err)?
        };
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


add_explain_methods!(XGBoost, CatBoost, LightGBM);

declare_support!(XGBoost,  classif = true, regress = true);
declare_support!(CatBoost, classif = true, regress = true);
declare_support!(LightGBM, classif = true, regress = true);
