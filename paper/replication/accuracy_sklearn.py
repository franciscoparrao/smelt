#!/usr/bin/env python3
"""
Reference accuracy values from scikit-learn for comparison with smelt-ml.
Uses stratified 5-fold CV with seed 42 on the same datasets.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import (
    fetch_california_housing, load_digits, load_wine, load_breast_cancer
)

SEED = 42
N_FOLDS = 5


def cv_classif(X, y, model_fn, name):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], pred))
    mean = np.mean(scores)
    std = np.std(scores)
    print(f"  {name:25s} {mean:.3f} +/- {std:.3f}")
    return mean


def cv_regress(X, y, model_fn, name):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for train_idx, test_idx in kf.split(X):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
    mean = np.mean(scores)
    std = np.std(scores)
    print(f"  {name:25s} {mean:.4f} +/- {std:.4f}")
    return mean


print("=" * 60)
print("scikit-learn Reference Accuracy (5-fold CV, seed=42)")
print("=" * 60)

# ── Classification datasets ──

for ds_name, loader in [("Wine", load_wine), ("Breast Cancer", load_breast_cancer),
                         ("Digits", load_digits)]:
    data = loader()
    X, y = data.data, data.target
    print(f"\n{ds_name} ({X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes)")
    cv_classif(X, y, lambda: DecisionTreeClassifier(random_state=SEED), "Decision Tree")
    cv_classif(X, y, lambda: RandomForestClassifier(n_estimators=100, random_state=SEED), "Random Forest")
    cv_classif(X, y, lambda: LogisticRegression(max_iter=1000, random_state=SEED), "Logistic Regression")

# ── Regression datasets ──

for ds_name, loader in [("California Housing", fetch_california_housing)]:
    data = loader()
    X, y = data.data, data.target
    print(f"\n{ds_name} ({X.shape[0]} samples, {X.shape[1]} features)")
    cv_regress(X, y, lambda: DecisionTreeRegressor(random_state=SEED), "Decision Tree")
    cv_regress(X, y, lambda: RandomForestRegressor(n_estimators=100, random_state=SEED), "Random Forest")
    cv_regress(X, y, lambda: Ridge(alpha=1.0), "Ridge Regression")

print()
