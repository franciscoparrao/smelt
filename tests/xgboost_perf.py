"""Benchmark official XGBoost for comparison with smelt-ml."""
import time
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification, make_regression

np.random.seed(42)

sizes = [100, 500, 1000, 5000, 10000]

print("=== XGBoost Official — Classification Benchmark ===")
print(f"{'N':>7} {'Features':>8} {'Trees':>5} {'Time (ms)':>10} {'Acc':>6}")
print("-" * 45)

for n in sizes:
    X, y = make_classification(n_samples=n, n_features=20, n_informative=10, random_state=42)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.3,
        reg_lambda=1.0, eval_metric='logloss', random_state=42,
        n_jobs=1,  # single thread for fair comparison
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = (time.perf_counter() - t0) * 1000
    acc = model.score(X, y)
    print(f"{n:>7} {20:>8} {100:>5} {elapsed:>10.1f} {acc:>6.4f}")

print()
print("=== XGBoost Official — Regression Benchmark ===")
print(f"{'N':>7} {'Features':>8} {'Trees':>5} {'Time (ms)':>10}")
print("-" * 38)

for n in sizes:
    X, y = make_regression(n_samples=n, n_features=20, n_informative=10, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.3,
        reg_lambda=1.0, random_state=42, n_jobs=1,
    )
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"{n:>7} {20:>8} {100:>5} {elapsed:>10.1f}")

# Save datasets for Rust benchmark
for n in sizes:
    X, y = make_classification(n_samples=n, n_features=20, n_informative=10, random_state=42)
    np.savetxt(f"/tmp/bench_classif_{n}_X.csv", X, delimiter=",")
    np.savetxt(f"/tmp/bench_classif_{n}_y.csv", y, delimiter=",")

    X, y = make_regression(n_samples=n, n_features=20, n_informative=10, random_state=42)
    np.savetxt(f"/tmp/bench_regress_{n}_X.csv", X, delimiter=",")
    np.savetxt(f"/tmp/bench_regress_{n}_y.csv", y, delimiter=",")

print("\n✓ Datasets saved to /tmp/bench_*")
