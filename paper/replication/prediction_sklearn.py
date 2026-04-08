#!/usr/bin/env python3
"""Prediction time benchmark for scikit-learn (reference for smelt-ml comparison)."""

import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier

N_TRAIN = 10_000
N_TEST = 1_000
N_RUNS = 100

X_train, y_train = make_classification(
    n_samples=N_TRAIN, n_features=20, n_informative=10, random_state=42
)
X_test, _ = make_classification(
    n_samples=N_TEST, n_features=20, n_informative=10, random_state=99
)

print(f"Prediction Time Benchmark (sklearn)")
print(f"Train N={N_TRAIN}, Predict N={N_TEST}, {N_RUNS} runs\n")

for name, clf in [
    ("GBM (100 trees)", GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
]:
    clf.fit(X_train, y_train)
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        clf.predict(X_test)
        times.append((time.perf_counter() - t0) * 1_000_000)  # microseconds
    mean = np.mean(times)
    std = np.std(times)
    print(f"  {name:20s} {mean:>8.0f} +/- {std:>5.0f} us")
