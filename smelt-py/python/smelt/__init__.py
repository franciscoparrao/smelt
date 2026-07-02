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
    # Preprocessing
    StandardScaler,
    # Resampling
    CrossValidation,
    SpatialBlockCV,
    SpatialBufferCV,
    StratifiedCV,
    GroupCV,
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

__version__ = "0.4.6"
__all__ = [
    "XGBoost", "CatBoost", "LightGBM",
    "RandomForest", "ExtraTrees", "DecisionTree",
    "LogisticRegression", "LinearRegression", "Ridge",
    "KNearestNeighbors", "GaussianNB", "GeoXGBoost",
    "StandardScaler",
    "CrossValidation", "SpatialBlockCV", "SpatialBufferCV", "StratifiedCV", "GroupCV",
    "accuracy_score", "rmse_score", "r2_score", "mae_score",
    "f1_score", "precision_score", "recall_score", "auc_roc_score",
    "balanced_accuracy_score", "cohens_kappa_score", "mcc_score", "brier_score",
    "wilcoxon_signed_rank", "bootstrap_ci", "sign_test",
    "filter_variance", "filter_correlation", "filter_anova_f",
    "filter_information_gain", "filter_mutual_information",
    "filter_mrmr", "filter_jmi", "filter_jmim", "filter_cmim", "filter_relief",
    "cumulative_ranking",
    "rfe",
    "BayesianOptimizer",
    "spatial_leave_one_out",
]
