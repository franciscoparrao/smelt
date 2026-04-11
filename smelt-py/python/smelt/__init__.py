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
    # Preprocessing
    StandardScaler,
    # Resampling
    CrossValidation,
    SpatialBlockCV,
    # Measures
    accuracy_score,
    rmse_score,
    r2_score,
    mae_score,
    f1_score,
    precision_score,
    recall_score,
    auc_roc_score,
    # Stats
    wilcoxon_signed_rank,
    bootstrap_ci,
    sign_test,
)

__version__ = "0.2.0"
__all__ = [
    "XGBoost", "CatBoost", "LightGBM",
    "RandomForest", "ExtraTrees", "DecisionTree",
    "LogisticRegression", "LinearRegression", "Ridge",
    "KNearestNeighbors", "GaussianNB",
    "StandardScaler",
    "CrossValidation", "SpatialBlockCV",
    "accuracy_score", "rmse_score", "r2_score", "mae_score",
    "f1_score", "precision_score", "recall_score", "auc_roc_score",
    "wilcoxon_signed_rank", "bootstrap_ci", "sign_test",
]
