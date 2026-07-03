//! Python bindings for smelt-ml via PyO3.

mod causal;
mod common;
mod feature_selection;
mod learners;
mod measures;
mod preprocess;
mod py_stats;
mod resample;
mod tuning;

use pyo3::prelude::*;

use causal::{DrLearner, RLearner, SLearner, TLearner, XLearner};
use feature_selection::{
    filter_anova_f, filter_cmim, filter_correlation, filter_information_gain, filter_jmi,
    filter_jmim, filter_mrmr, filter_mutual_information, filter_relief, filter_variance, rfe,
};
use learners::boosting::{CatBoost, GeoXGBoost, LightGBM, XGBoost};
use learners::ensemble::{Bagging, DynamicEnsemble, Stacking, registered_learner_ids};
use learners::linear::{ElasticNet, Lasso, LinearRegression, LinearSVM, LogisticRegression, Ridge};
use learners::misc::{AdaBoost, EBM, GaussianNB, KNearestNeighbors, QuantileForest, QuantileGB};
use learners::trees::{
    DecisionTree, ExtraTrees, GradientBoosting, HoeffdingTree, ObliqueForest, ObliqueTree,
    RandomForest,
};
use measures::{
    accuracy_score, auc_roc_score, balanced_accuracy_score, brier_score, cohens_kappa_score,
    f1_score, mae_score, mcc_score, precision_score, r2_score, recall_score, rmse_score,
};
use preprocess::StandardScaler;
use py_stats::{bootstrap_ci, sign_test, wilcoxon_signed_rank};
use resample::{CrossValidation, GroupCV, SpatialBlockCV, SpatialBufferCV, StratifiedCV};
use tuning::PyBayesianOptimizer;

#[pymodule]
fn _smelt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Learners
    m.add_class::<XGBoost>()?;
    m.add_class::<CatBoost>()?;
    m.add_class::<LightGBM>()?;
    m.add_class::<RandomForest>()?;
    m.add_class::<ExtraTrees>()?;
    m.add_class::<DecisionTree>()?;
    m.add_class::<LogisticRegression>()?;
    m.add_class::<LinearRegression>()?;
    m.add_class::<Ridge>()?;
    m.add_class::<KNearestNeighbors>()?;
    m.add_class::<GaussianNB>()?;
    m.add_class::<GeoXGBoost>()?;
    m.add_class::<AdaBoost>()?;
    m.add_class::<EBM>()?;
    m.add_class::<Lasso>()?;
    m.add_class::<ElasticNet>()?;
    m.add_class::<GradientBoosting>()?;
    m.add_class::<HoeffdingTree>()?;
    m.add_class::<LinearSVM>()?;
    m.add_class::<ObliqueTree>()?;
    m.add_class::<ObliqueForest>()?;
    m.add_class::<QuantileForest>()?;
    m.add_class::<QuantileGB>()?;
    m.add_class::<Bagging>()?;
    m.add_class::<Stacking>()?;
    m.add_class::<DynamicEnsemble>()?;
    m.add_function(wrap_pyfunction!(registered_learner_ids, m)?)?;

    // Causal meta-learners
    m.add_class::<TLearner>()?;
    m.add_class::<SLearner>()?;
    m.add_class::<XLearner>()?;
    m.add_class::<RLearner>()?;
    m.add_class::<DrLearner>()?;

    // Preprocessing
    m.add_class::<StandardScaler>()?;

    // Resampling
    m.add_class::<CrossValidation>()?;
    m.add_class::<SpatialBlockCV>()?;
    m.add_class::<SpatialBufferCV>()?;
    m.add_class::<StratifiedCV>()?;
    m.add_class::<GroupCV>()?;

    // Measures
    m.add_function(wrap_pyfunction!(accuracy_score, m)?)?;
    m.add_function(wrap_pyfunction!(rmse_score, m)?)?;
    m.add_function(wrap_pyfunction!(r2_score, m)?)?;
    m.add_function(wrap_pyfunction!(mae_score, m)?)?;
    m.add_function(wrap_pyfunction!(f1_score, m)?)?;
    m.add_function(wrap_pyfunction!(precision_score, m)?)?;
    m.add_function(wrap_pyfunction!(recall_score, m)?)?;
    m.add_function(wrap_pyfunction!(auc_roc_score, m)?)?;
    m.add_function(wrap_pyfunction!(balanced_accuracy_score, m)?)?;
    m.add_function(wrap_pyfunction!(cohens_kappa_score, m)?)?;
    m.add_function(wrap_pyfunction!(mcc_score, m)?)?;
    m.add_function(wrap_pyfunction!(brier_score, m)?)?;

    // Stats
    m.add_function(wrap_pyfunction!(wilcoxon_signed_rank, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_ci, m)?)?;
    m.add_function(wrap_pyfunction!(sign_test, m)?)?;

    // Filters
    m.add_function(wrap_pyfunction!(filter_variance, m)?)?;
    m.add_function(wrap_pyfunction!(filter_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(filter_anova_f, m)?)?;
    m.add_function(wrap_pyfunction!(filter_information_gain, m)?)?;
    m.add_function(wrap_pyfunction!(filter_mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(filter_mrmr, m)?)?;
    m.add_function(wrap_pyfunction!(filter_jmi, m)?)?;
    m.add_function(wrap_pyfunction!(filter_jmim, m)?)?;
    m.add_function(wrap_pyfunction!(filter_cmim, m)?)?;
    m.add_function(wrap_pyfunction!(filter_relief, m)?)?;

    // Tuning
    m.add_class::<PyBayesianOptimizer>()?;

    // RFE
    m.add_function(wrap_pyfunction!(rfe, m)?)?;

    Ok(())
}
