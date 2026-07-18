"""smelt — Rust-powered ML framework with spatial modeling and conformal prediction.

Fast gradient boosting (3.7x faster prediction than scikit-learn),
spatial cross-validation, conformal prediction, and statistical testing
in a single package.

Example:
    >>> from smelt import XGBoost, accuracy_score
    >>> model = XGBoost(n_estimators=100)
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    >>> print(accuracy_score(y_test, preds))
"""

from smelt._smelt import (
    # Learners
    XGBoost,
    CatBoost,
    LightGBM,
    RandomForest,
    ExtraTrees,
    DecisionTree,
    LogisticRegression,
    LinearRegression,
    Ridge,
    KNearestNeighbors,
    GaussianNB,
    GeoXGBoost,
    KrigingHybrid,
    AdaBoost,
    EBM,
    Lasso,
    ElasticNet,
    GradientBoosting,
    HoeffdingTree,
    AdaptiveRandomForest,
    MondrianForest,
    DeepForest,
    ExtremeLearningMachine,
    LinearSVM,
    ObliqueTree,
    ObliqueForest,
    QuantileForest,
    QuantileGB,
    Bagging,
    Stacking,
    DynamicEnsemble,
    CostSensitiveClassifier,
    TargetTransformRegressor,
    AutoTuner,
    registered_learner_ids,
    learner_properties,
    # Causal meta-learners
    TLearner,
    SLearner,
    XLearner,
    RLearner,
    DrLearner,
    # Clustering / anomaly detection
    KMeans,
    DBSCAN,
    IsolationForest,
    # Data loaders
    CsvLoader,
    # Preprocessing
    StandardScaler,
    Smote,
    SpatialSmote,
    # Conformal prediction
    SplitConformal,
    # Resampling
    CrossValidation,
    SpatialBlockCV,
    SpatialBufferCV,
    StratifiedCV,
    GroupCV,
    TimeSeriesCV,
    # Measures
    accuracy_score,
    rmse_score,
    r2_score,
    mae_score,
    f1_score,
    precision_score,
    recall_score,
    auc_roc_score,
    balanced_accuracy_score,
    cohens_kappa_score,
    mcc_score,
    brier_score,
    mape_score,
    logloss_score,
    # Stats
    wilcoxon_signed_rank,
    bootstrap_ci,
    sign_test,
    # Filters
    filter_variance,
    filter_correlation,
    filter_anova_f,
    filter_information_gain,
    filter_mutual_information,
    filter_mrmr,
    filter_jmi,
    filter_jmim,
    filter_cmim,
    filter_relief,
    # RFE
    rfe,
    # Tuning
    BayesianOptimizer,
)

from smelt.filters import cumulative_ranking
from smelt.spatial import spatial_leave_one_out

try:
    # Only present when the extension was built with `--features parquet`
    # (pulls in polars, ~200 transitive crates) -- opt-in, so a plain
    # `pip install smelt` / `maturin develop` build doesn't need it.
    from smelt._smelt import ParquetLoader
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

# Single-sourced from smelt-py/Cargo.toml via the installed package
# metadata (maturin stamps it at build time; pyproject declares
# `dynamic = ["version"]`). The triple hand-synced copy this replaces
# drifted more than once (audit M20).
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("smelt-ml")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"
__all__ = [
    "XGBoost", "CatBoost", "LightGBM",
    "RandomForest", "ExtraTrees", "DecisionTree",
    "LogisticRegression", "LinearRegression", "Ridge",
    "KNearestNeighbors", "GaussianNB", "GeoXGBoost", "KrigingHybrid",
    "AdaBoost", "EBM", "Lasso", "ElasticNet", "GradientBoosting",
    "HoeffdingTree", "AdaptiveRandomForest", "MondrianForest", "DeepForest", "ExtremeLearningMachine", "LinearSVM", "ObliqueTree", "ObliqueForest", "QuantileForest", "QuantileGB",
    "Bagging", "Stacking", "DynamicEnsemble", "CostSensitiveClassifier", "TargetTransformRegressor", "AutoTuner", "registered_learner_ids", "learner_properties",
    "TLearner", "SLearner", "XLearner", "RLearner", "DrLearner",
    "KMeans", "DBSCAN", "IsolationForest",
    "CsvLoader",
    "StandardScaler", "Smote", "SpatialSmote",
    "SplitConformal",
    "CrossValidation", "SpatialBlockCV", "SpatialBufferCV", "StratifiedCV", "GroupCV", "TimeSeriesCV",
    "accuracy_score", "rmse_score", "r2_score", "mae_score",
    "f1_score", "precision_score", "recall_score", "auc_roc_score",
    "balanced_accuracy_score", "cohens_kappa_score", "mcc_score", "brier_score",
    "mape_score", "logloss_score",
    "wilcoxon_signed_rank", "bootstrap_ci", "sign_test",
    "filter_variance", "filter_correlation", "filter_anova_f",
    "filter_information_gain", "filter_mutual_information",
    "filter_mrmr", "filter_jmi", "filter_jmim", "filter_cmim", "filter_relief",
    "cumulative_ranking",
    "rfe",
    "BayesianOptimizer",
    "spatial_leave_one_out",
]
