//! Feature selection: RFE (wrapper-based) and univariate filter methods.

use crate::common::{is_integer, smelt_err, to_array2};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

pub(crate) fn make_rfe_factory(learner_type: &str) -> PyResult<Box<dyn Fn() -> Box<dyn smelt_ml::learner::Learner> + Send + Sync>> {
    match learner_type {
        "decision_tree" => Ok(Box::new(|| Box::new(smelt_ml::prelude::DecisionTree::default()) as Box<dyn smelt_ml::learner::Learner>)),
        "random_forest" => Ok(Box::new(|| Box::new(smelt_ml::prelude::RandomForest::new()) as Box<dyn smelt_ml::learner::Learner>)),
        "extra_trees" => Ok(Box::new(|| Box::new(smelt_ml::prelude::ExtraTrees::new()) as Box<dyn smelt_ml::learner::Learner>)),
        "xgboost" => Ok(Box::new(|| Box::new(smelt_ml::prelude::XGBoost::new()) as Box<dyn smelt_ml::learner::Learner>)),
        "ridge" => Ok(Box::new(|| Box::new(smelt_ml::prelude::Ridge::new(1.0)) as Box<dyn smelt_ml::learner::Learner>)),
        _ => Err(PyRuntimeError::new_err(format!("Unknown learner for RFE: {learner_type}"))),
    }
}

#[pyfunction]
#[pyo3(signature = (x, y, learner_type="decision_tree", n_features=5, feature_names=None))]
pub(crate) fn rfe<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'_, f64>,
    y: &Bound<'_, PyAny>,
    learner_type: &str,
    n_features: usize,
    feature_names: Option<Vec<String>>,
) -> PyResult<PyObject> {
    use smelt_ml::preprocess::Transformer;

    let factory = make_rfe_factory(learner_type)?;
    let features = to_array2(x);
    let n_feat = features.ncols();

    let is_classif = is_integer(y);
    let mut selector = if is_classif {
        smelt_ml::preprocess::RFE::classif(move || factory(), n_features)
    } else {
        smelt_ml::preprocess::RFE::regress(move || factory(), n_features)
    };

    let target_f64: Vec<f64> = if is_classif {
        let t: Vec<i64> = y.extract()?;
        t.into_iter().map(|v| v as f64).collect()
    } else {
        y.extract()?
    };

    py.allow_threads(|| selector.fit_supervised(&features, &target_f64))
        .map_err(smelt_err)?;
    let indices = selector
        .selected_indices()
        .ok_or_else(|| PyRuntimeError::new_err("RFE selector was not fitted"))?;

    let names: Vec<String> = feature_names.unwrap_or_else(|| (0..n_feat).map(|i| format!("f{i}")).collect());
    let result: Vec<(String, usize)> = indices.iter().map(|&i| (names[i].clone(), i)).collect();
    Ok(result.into_pyobject(py)?.into_any().unbind())
}


// ── Feature Selection Filters ──────────────────────────────────────────

fn run_filter(
    method: &str,
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: &Bound<'_, PyAny>,
    feature_names: Vec<String>,
    k: usize,
) -> PyResult<PyObject> {
    use smelt_ml::preprocess::filter::{
        AnovaFFilter, CmimFilter, CorrelationFilter, Filter, InformationGainFilter, JmiFilter,
        JmimFilter, MrmrFilter, MutualInfoFilter, ReliefFilter, VarianceFilter,
    };

    let features = to_array2(x);
    let n_feat = features.ncols();
    let k = k.min(n_feat);

    let target: Vec<f64> = y.extract()?;

    let known = matches!(
        method,
        "variance"
            | "correlation"
            | "anova_f"
            | "information_gain"
            | "mutual_information"
            | "mrmr"
            | "jmi"
            | "jmim"
            | "cmim"
            | "relief"
    );
    if !known {
        return Err(PyRuntimeError::new_err(format!("Unknown filter: {method}")));
    }

    // Get raw per-feature scores (higher = better). Relief is O(n^2) and the
    // rest scan every feature-target pair, so release the GIL for the
    // computation (validated `method` above, so the wildcard is unreachable).
    let scores: Vec<f64> = py.allow_threads(|| match method {
        "variance" => VarianceFilter.score(&features, &target),
        "correlation" => CorrelationFilter.score(&features, &target),
        "anova_f" => AnovaFFilter.score(&features, &target),
        "information_gain" => InformationGainFilter.score(&features, &target),
        "mutual_information" => MutualInfoFilter.score(&features, &target),
        "mrmr" => MrmrFilter.score(&features, &target),
        "jmi" => JmiFilter.score(&features, &target),
        "jmim" => JmimFilter.score(&features, &target),
        "cmim" => CmimFilter.score(&features, &target),
        "relief" => ReliefFilter.score(&features, &target),
        _ => unreachable!("validated above"),
    });

    let names: Vec<String> = if feature_names.len() == n_feat {
        feature_names
    } else {
        (0..n_feat).map(|i| format!("f{i}")).collect()
    };

    // Sort by score descending (higher = more important), take top k
    let mut ranked: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let result: Vec<(String, f64)> = ranked
        .into_iter()
        .take(k)
        .map(|(i, s)| (names[i].clone(), s))
        .collect();
    Ok(result.into_pyobject(py)?.into_any().unbind())
}

macro_rules! filter_fn {
    ($name:ident, $method:expr) => {
        #[pyfunction]
        #[pyo3(signature = (x, y, feature_names, k=15))]
        pub(crate) fn $name<'py>(
            py: Python<'py>,
            x: PyReadonlyArray2<'_, f64>,
            y: &Bound<'_, PyAny>,
            feature_names: Vec<String>,
            k: usize,
        ) -> PyResult<PyObject> {
            run_filter($method, py, x, y, feature_names, k)
        }
    };
}

filter_fn!(filter_variance, "variance");
filter_fn!(filter_correlation, "correlation");
filter_fn!(filter_anova_f, "anova_f");
filter_fn!(filter_information_gain, "information_gain");
filter_fn!(filter_mutual_information, "mutual_information");
filter_fn!(filter_mrmr, "mrmr");
filter_fn!(filter_jmi, "jmi");
filter_fn!(filter_jmim, "jmim");
filter_fn!(filter_cmim, "cmim");
filter_fn!(filter_relief, "relief");

