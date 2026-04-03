#!/usr/bin/env python3
"""
Benchmark official C++ gradient boosting libraries (via Python API).

Generates synthetic datasets and measures training time for XGBoost, LightGBM,
and CatBoost at various dataset sizes. Each benchmark is repeated N_RUNS times
to report mean +/- std.

Output: benchmark_cpp_results.json
"""

import json
import time
import sys
import platform
import os
import numpy as np
from sklearn.datasets import make_classification, make_regression

# ── Configuration ──────────────────────────────────────────────────────────────

N_RUNS = 10
SIZES = [500, 1_000, 5_000, 10_000, 50_000, 100_000]
N_FEATURES = 20
N_INFORMATIVE = 10
N_TREES = 100
MAX_DEPTH = 6
SEED = 42

# ── Hardware info ──────────────────────────────────────────────────────────────

def get_hardware_info():
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    # CPU info on Linux
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":")[1].strip()
                    break
    except FileNotFoundError:
        pass
    # Cache sizes
    try:
        for level in [1, 2, 3]:
            path = f"/sys/devices/system/cpu/cpu0/cache/index{level}/size"
            if os.path.exists(path):
                with open(path) as f:
                    info[f"L{level}_cache"] = f.read().strip()
    except Exception:
        pass
    # Memory
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["memory_total"] = line.split(":")[1].strip()
                    break
    except FileNotFoundError:
        pass
    return info


# ── Benchmark functions ───────────────────────────────────────────────────────

def bench_xgboost_classif(X, y, n_runs):
    import xgboost as xgb
    times = []
    for _ in range(n_runs):
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "max_depth": MAX_DEPTH,
            "eta": 0.3,
            "objective": "binary:logistic",
            "nthread": 1,
            "verbosity": 0,
            "seed": SEED,
        }
        t0 = time.perf_counter()
        xgb.train(params, dtrain, num_boost_round=N_TREES)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_xgboost_regress(X, y, n_runs):
    import xgboost as xgb
    times = []
    for _ in range(n_runs):
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "max_depth": MAX_DEPTH,
            "eta": 0.3,
            "objective": "reg:squarederror",
            "nthread": 1,
            "verbosity": 0,
            "seed": SEED,
        }
        t0 = time.perf_counter()
        xgb.train(params, dtrain, num_boost_round=N_TREES)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_lightgbm_classif(X, y, n_runs):
    import lightgbm as lgb
    times = []
    for _ in range(n_runs):
        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        params = {
            "max_depth": MAX_DEPTH,
            "learning_rate": 0.1,
            "objective": "binary",
            "n_jobs": 1,
            "verbosity": -1,
            "seed": SEED,
            "num_leaves": 31,
        }
        t0 = time.perf_counter()
        lgb.train(params, dtrain, num_boost_round=N_TREES)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_lightgbm_regress(X, y, n_runs):
    import lightgbm as lgb
    times = []
    for _ in range(n_runs):
        dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
        params = {
            "max_depth": MAX_DEPTH,
            "learning_rate": 0.1,
            "objective": "regression",
            "n_jobs": 1,
            "verbosity": -1,
            "seed": SEED,
            "num_leaves": 31,
        }
        t0 = time.perf_counter()
        lgb.train(params, dtrain, num_boost_round=N_TREES)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_catboost_classif(X, y, n_runs):
    from catboost import CatBoostClassifier
    times = []
    for _ in range(n_runs):
        model = CatBoostClassifier(
            iterations=N_TREES,
            depth=MAX_DEPTH,
            learning_rate=0.3,
            thread_count=1,
            verbose=0,
            random_seed=SEED,
        )
        t0 = time.perf_counter()
        model.fit(X, y)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def bench_catboost_regress(X, y, n_runs):
    from catboost import CatBoostRegressor
    times = []
    for _ in range(n_runs):
        model = CatBoostRegressor(
            iterations=N_TREES,
            depth=MAX_DEPTH,
            learning_rate=0.3,
            thread_count=1,
            verbose=0,
            random_seed=SEED,
        )
        t0 = time.perf_counter()
        model.fit(X, y)
        times.append((time.perf_counter() - t0) * 1000)
    return times


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Benchmark: Official C++ Gradient Boosting Libraries")
    print(f"Configuration: {N_TREES} trees, max_depth={MAX_DEPTH}, {N_RUNS} runs each")
    print(f"Features: {N_FEATURES} ({N_INFORMATIVE} informative)")
    print("=" * 70)

    # Check library versions
    versions = {}
    try:
        import xgboost; versions["xgboost"] = xgboost.__version__
    except ImportError:
        print("ERROR: xgboost not installed"); sys.exit(1)
    try:
        import lightgbm; versions["lightgbm"] = lightgbm.__version__
    except ImportError:
        print("ERROR: lightgbm not installed"); sys.exit(1)
    try:
        import catboost; versions["catboost"] = catboost.__version__
    except ImportError:
        print("ERROR: catboost not installed"); sys.exit(1)
    try:
        import sklearn; versions["scikit-learn"] = sklearn.__version__
    except ImportError:
        print("ERROR: scikit-learn not installed"); sys.exit(1)

    print(f"Versions: {versions}")

    results = {
        "hardware": get_hardware_info(),
        "versions": versions,
        "config": {
            "n_runs": N_RUNS,
            "n_trees": N_TREES,
            "max_depth": MAX_DEPTH,
            "n_features": N_FEATURES,
            "n_informative": N_INFORMATIVE,
            "seed": SEED,
        },
        "classification": {},
        "regression": {},
    }

    for n in SIZES:
        print(f"\n{'─' * 50}")
        print(f"N = {n:,}")
        print(f"{'─' * 50}")

        # Generate data
        X_c, y_c = make_classification(
            n_samples=n, n_features=N_FEATURES, n_informative=N_INFORMATIVE,
            n_redundant=5, random_state=SEED
        )
        X_r, y_r = make_regression(
            n_samples=n, n_features=N_FEATURES, n_informative=N_INFORMATIVE,
            random_state=SEED
        )

        key = str(n)
        results["classification"][key] = {}
        results["regression"][key] = {}

        # Classification benchmarks
        for name, func in [
            ("xgboost", bench_xgboost_classif),
            ("lightgbm", bench_lightgbm_classif),
            ("catboost", bench_catboost_classif),
        ]:
            print(f"  {name} classif...", end=" ", flush=True)
            times = func(X_c, y_c, N_RUNS)
            mean = np.mean(times)
            std = np.std(times)
            print(f"{mean:.1f} +/- {std:.1f} ms")
            results["classification"][key][name] = {
                "times_ms": [round(t, 2) for t in times],
                "mean_ms": round(mean, 2),
                "std_ms": round(std, 2),
            }

        # Regression benchmarks
        for name, func in [
            ("xgboost", bench_xgboost_regress),
            ("lightgbm", bench_lightgbm_regress),
            ("catboost", bench_catboost_regress),
        ]:
            print(f"  {name} regress...", end=" ", flush=True)
            times = func(X_r, y_r, N_RUNS)
            mean = np.mean(times)
            std = np.std(times)
            print(f"{mean:.1f} +/- {std:.1f} ms")
            results["regression"][key][name] = {
                "times_ms": [round(t, 2) for t in times],
                "mean_ms": round(mean, 2),
                "std_ms": round(std, 2),
            }

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_cpp_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
