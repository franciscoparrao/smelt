#!/usr/bin/env python3
"""
Validate smelt-ml's GeoXGBoost against Grekousis' original geoxgboost package.

Runs both implementations on King County housing data (1K samples) with
matched hyperparameters, comparing RMSE and per-point predictions.

Output: validation table suitable for the JSS paper.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import time
import json
import sys
import os

# ── Load data ──────────────────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "king_county_1k.csv")
df = pd.read_csv(DATA_PATH)

target_col = "log_price"
coord_cols = ["lat", "long"]
feature_cols = [c for c in df.columns if c not in [target_col] + coord_cols]

X = df[feature_cols]
y = df[target_col]
coords = df[coord_cols]

# Fixed train/test split (80/20, seed=42)
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
coords_train, coords_test = coords.iloc[train_idx], coords.iloc[test_idx]

print(f"Data: {len(df)} samples, {len(feature_cols)} features")
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")
print(f"Features: {feature_cols}")
print()

# ── Matched hyperparameters ────────────────────────────────────────────

BANDWIDTH = 30
N_ESTIMATORS = 100
MAX_DEPTH = 6
LEARNING_RATE = 0.3
REG_LAMBDA = 1.0
SEED = 42

xgb_params = {
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "learning_rate": LEARNING_RATE,
    "reg_lambda": REG_LAMBDA,
    "random_state": SEED,
    "n_jobs": 1,
    "verbosity": 0,
}

# ── 1. Run Grekousis' geoxgboost ───────────────────────────────────────

print("=" * 60)
print("Running Grekousis geoxgboost (Python original)")
print("=" * 60)

from geoxgboost import gxgb, predict_gxgb

# geoxgboost expects DataFrames; params as dict for XGBoost
gxgb_params = {
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "learning_rate": LEARNING_RATE,
    "reg_lambda": REG_LAMBDA,
    "reg_alpha": 0,
    "random_state": SEED,
    "n_jobs": 1,
    "verbosity": 0,
}

t0 = time.time()
result = gxgb(
    X=X_train.reset_index(drop=True),
    y=y_train.reset_index(drop=True),
    Coords=coords_train.reset_index(drop=True),
    params=gxgb_params,
    bw=BANDWIDTH,
    Kernel="Adaptive",
    alpha_wt_type="varying",
    alpha_wt=1,
    test_size=0.2,
    seed=SEED,
    n_splits=5,
    path_save=False,
)
gxgb_train_time = time.time() - t0

# Extract training predictions
train_preds_df = result["Prediction"]
print(f"\nTrain time: {gxgb_train_time:.2f}s")
print(f"Result keys: {list(result.keys())}")
print(f"Stats:\n{result['Stats']}")

# Predict on test set
t0 = time.time()
test_result = predict_gxgb(
    DataPredict=X_test.reset_index(drop=True),
    CoordsPredict=coords_test.reset_index(drop=True),
    Coords=coords_train.reset_index(drop=True),
    Output_GXGB_LocalModel=result,
    alpha_wt_type="varying",
    path_save=False,
)
gxgb_predict_time = time.time() - t0

# Extract test predictions
if isinstance(test_result, pd.DataFrame):
    gxgb_test_preds = test_result.iloc[:, -1].values  # last column is usually prediction
    print(f"\nTest prediction columns: {list(test_result.columns)}")
elif isinstance(test_result, dict):
    print(f"\nTest prediction keys: {list(test_result.keys())}")
    if "Prediction" in test_result:
        pred_df = test_result["Prediction"]
        gxgb_test_preds = pred_df.iloc[:, -1].values
    else:
        gxgb_test_preds = np.array(list(test_result.values())[0])
else:
    gxgb_test_preds = np.array(test_result)

gxgb_test_rmse = np.sqrt(mean_squared_error(y_test, gxgb_test_preds))
gxgb_test_r2 = r2_score(y_test, gxgb_test_preds)

print(f"\ngeoxgboost Test RMSE: {gxgb_test_rmse:.4f}")
print(f"geoxgboost Test R²:   {gxgb_test_r2:.4f}")
print(f"Predict time: {gxgb_predict_time:.2f}s")

# ── 2. Run standard XGBoost (baseline) ─────────────────────────────────

print()
print("=" * 60)
print("Running standard XGBoost (baseline)")
print("=" * 60)

import xgboost as xgb

model_xgb = xgb.XGBRegressor(**xgb_params)
t0 = time.time()
model_xgb.fit(X_train, y_train)
xgb_train_time = time.time() - t0

xgb_test_preds = model_xgb.predict(X_test)
xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_preds))
xgb_test_r2 = r2_score(y_test, xgb_test_preds)

print(f"XGBoost Test RMSE: {xgb_test_rmse:.4f}")
print(f"XGBoost Test R²:   {xgb_test_r2:.4f}")
print(f"Train time: {xgb_train_time:.2f}s")

# ── 3. Summary ─────────────────────────────────────────────────────────

print()
print("=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"{'Method':<25} {'RMSE':>8} {'R²':>8} {'Train(s)':>10}")
print("-" * 55)
print(f"{'XGBoost (baseline)':<25} {xgb_test_rmse:>8.4f} {xgb_test_r2:>8.4f} {xgb_train_time:>10.2f}")
print(f"{'geoxgboost (Grekousis)':<25} {gxgb_test_rmse:>8.4f} {gxgb_test_r2:>8.4f} {gxgb_train_time:>10.2f}")
print()
print("Save train/test indices and geoxgboost predictions for Rust comparison:")

# Save for Rust comparison
np.savetxt(
    os.path.join(os.path.dirname(__file__), "geoxgb_train_idx.csv"),
    train_idx, fmt="%d"
)
np.savetxt(
    os.path.join(os.path.dirname(__file__), "geoxgb_test_idx.csv"),
    test_idx, fmt="%d"
)
np.savetxt(
    os.path.join(os.path.dirname(__file__), "geoxgb_test_preds.csv"),
    gxgb_test_preds, fmt="%.6f"
)
np.savetxt(
    os.path.join(os.path.dirname(__file__), "xgb_test_preds.csv"),
    xgb_test_preds, fmt="%.6f"
)

# Save full results as JSON
results = {
    "dataset": "king_county_1k",
    "n_train": len(train_idx),
    "n_test": len(test_idx),
    "n_features": len(feature_cols),
    "bandwidth": BANDWIDTH,
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "learning_rate": LEARNING_RATE,
    "seed": SEED,
    "xgboost": {
        "rmse": round(float(xgb_test_rmse), 4),
        "r2": round(float(xgb_test_r2), 4),
        "train_time_s": round(xgb_train_time, 2),
    },
    "geoxgboost_python": {
        "rmse": round(float(gxgb_test_rmse), 4),
        "r2": round(float(gxgb_test_r2), 4),
        "train_time_s": round(gxgb_train_time, 2),
    },
}

with open(os.path.join(os.path.dirname(__file__), "geoxgb_validation.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nFiles saved in paper/replication/:")
print("  geoxgb_train_idx.csv, geoxgb_test_idx.csv")
print("  geoxgb_test_preds.csv, xgb_test_preds.csv")
print("  geoxgb_validation.json")
print("\nNext: run the Rust GeoXGBoost example with the same indices to compare.")
