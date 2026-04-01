"""Benchmark official CatBoost for comparison with smelt-ml."""
import time
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.datasets import make_classification, make_regression

np.random.seed(42)
sizes = [100, 500, 1000, 5000, 10000]

print("=== CatBoost Official — Classification Benchmark ===")
print(f"{'N':>7} {'Features':>8} {'Trees':>5} {'Time (ms)':>10} {'Acc':>6}")
print("-" * 45)

for n in sizes:
    X, y = make_classification(n_samples=n, n_features=20, n_informative=10, random_state=42)
    model = CatBoostClassifier(
        iterations=100, depth=6, learning_rate=0.1,
        l2_leaf_reg=3.0, thread_count=1, random_seed=42, verbose=0,
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = (time.perf_counter() - t0) * 1000
    acc = (model.predict(X).flatten() == y).mean()
    print(f"{n:>7} {20:>8} {100:>5} {elapsed:>10.1f} {acc:>6.4f}")

print()
print("=== CatBoost Official — Regression Benchmark ===")
print(f"{'N':>7} {'Features':>8} {'Trees':>5} {'Time (ms)':>10}")
print("-" * 38)

for n in sizes:
    X, y = make_regression(n_samples=n, n_features=20, n_informative=10, random_state=42)
    model = CatBoostRegressor(
        iterations=100, depth=6, learning_rate=0.1,
        l2_leaf_reg=3.0, thread_count=1, random_seed=42, verbose=0,
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"{n:>7} {20:>8} {100:>5} {elapsed:>10.1f}")
