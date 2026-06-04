"""Geographical-XGBoost end-to-end demo on the King County housing dataset.

Demonstrates the full smelt-ml spatial ML pipeline:

  1. Data loading and preprocessing (King County housing, 1,000 samples)
  2. Train/test split with held-out coordinates
  3. GeoXGBoost training and out-of-sample prediction (`predict_spatial`)
  4. Baseline comparison against vanilla XGBoost (with and without coords)
  5. Spatial cross-validation (SpatialBufferCV)
  6. Split conformal prediction intervals + empirical coverage check
  7. Feature importance (averaged across local models)

Run with the project virtualenv:
  /home/<user>/proyectos/smelt/smelt-py/.venv/bin/python3 \
      paper/replication/demo_geoxgboost.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from smelt import XGBoost, GeoXGBoost
from smelt import rmse_score, r2_score
from smelt.spatial import SpatialBufferCV

SEED = 42
ALPHA = 0.10  # conformal: target 90% coverage


def _find_data():
    """Locate king_county_1k.csv next to this script, in ./data, or in the repo."""
    name = "king_county_1k.csv"
    here = Path(__file__).resolve().parent
    candidates = [
        here / name,                       # same folder as the script
        here / "data" / name,              # ./data subfolder
        here.parent.parent / "data" / name,  # repo layout (paper/replication -> data)
        Path.cwd() / name,                 # current working directory
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"Could not find {name}. Place it next to this script "
        f"(searched: {', '.join(str(c) for c in candidates)})."
    )


DATA = _find_data()


# ── 1. Load and preprocess ──────────────────────────────────────────────
df = pd.read_csv(DATA)
target_col = "log_price"
coord_cols = ["lat", "long"]
feature_cols = [c for c in df.columns if c not in coord_cols + [target_col]]

X = df[feature_cols].to_numpy(dtype=np.float64)
y = df[target_col].to_numpy(dtype=np.float64)
coords = df[coord_cols].to_numpy(dtype=np.float64)

print(f"Dataset: {len(df)} samples, {len(feature_cols)} features")
print(f"Features: {feature_cols}")
print(f"Target:   {target_col}  (range {y.min():.2f}..{y.max():.2f})")

# 80% train / 20% test
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(df))
split = int(0.8 * len(df))
tr, te = idx[:split], idx[split:]
Xtr, ytr, Ctr = X[tr], y[tr], coords[tr]
Xte, yte, Cte = X[te], y[te], coords[te]
print(f"Train: {len(tr)}   Test: {len(te)}")


# ── 2. Baseline: vanilla XGBoost (no spatial info) ──────────────────────
xgb = XGBoost(n_estimators=100, max_depth=6, learning_rate=0.3, seed=SEED)
xgb.fit(Xtr, ytr)
pred_xgb = xgb.predict(Xte)
rmse_xgb = rmse_score(yte.tolist(), pred_xgb.tolist())
r2_xgb   = r2_score(yte.tolist(), pred_xgb.tolist())

# Baseline + coords as features (matches paper Table 9 "XGBoost + coords")
Xtr_aug = np.column_stack([Xtr, Ctr])
Xte_aug = np.column_stack([Xte, Cte])
xgb_aug = XGBoost(n_estimators=100, max_depth=6, learning_rate=0.3, seed=SEED)
xgb_aug.fit(Xtr_aug, ytr)
pred_aug = xgb_aug.predict(Xte_aug)
rmse_aug = rmse_score(yte.tolist(), pred_aug.tolist())
r2_aug   = r2_score(yte.tolist(), pred_aug.tolist())


# ── 3. GeoXGBoost ───────────────────────────────────────────────────────
# Hyperparameters from paper Table 11 (validation against original Python pkg):
#   100 trees, depth 6, learning_rate 0.3, bandwidth 30 nearest neighbours,
#   adaptive alpha (None) — set to 1.0 for pure-local prediction.
geo = GeoXGBoost(
    bandwidth=30,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    alpha=1.0,           # pure local; use None for adaptive global/local blend
    seed=SEED,
)
geo.fit(Xtr, ytr, coords=Ctr)
# Spatial-aware prediction: pass new coords so each test point uses
# its nearest local model.
pred_geo = geo.predict(Xte, coords=Cte)
rmse_geo = rmse_score(yte.tolist(), pred_geo.tolist())
r2_geo   = r2_score(yte.tolist(), pred_geo.tolist())

print("\n── Holdout performance (test n = {:d}) ──".format(len(te)))
print(f"  XGBoost           RMSE={rmse_xgb:.3f}  R²={r2_xgb:.3f}")
print(f"  XGBoost + coords  RMSE={rmse_aug:.3f}  R²={r2_aug:.3f}")
print(f"  GeoXGBoost (α=1)  RMSE={rmse_geo:.3f}  R²={r2_geo:.3f}")


# ── 4. Spatial cross-validation ─────────────────────────────────────────
# SpatialBufferCV excludes training samples within `buffer_distance` of
# each test fold's centroid — prevents spatial leakage.
# Coordinates are lat/long in degrees; ~0.05° ≈ 5 km at this latitude.
cv = SpatialBufferCV(n_folds=5, coords=coords, buffer_distance=0.05, seed=SEED)
rmse_per_fold = []
for fold_idx, (train_idx, test_idx) in enumerate(cv.splits(len(df))):
    if len(train_idx) < 10 or len(test_idx) == 0:
        continue
    Xtr_f, ytr_f, Ctr_f = X[train_idx], y[train_idx], coords[train_idx]
    Xte_f, yte_f, Cte_f = X[test_idx],  y[test_idx],  coords[test_idx]
    m = GeoXGBoost(bandwidth=30, n_estimators=100, max_depth=6,
                   learning_rate=0.3, alpha=1.0, seed=SEED)
    m.fit(Xtr_f, ytr_f, coords=Ctr_f)
    pred = m.predict(Xte_f, coords=Cte_f)
    rmse_per_fold.append(rmse_score(yte_f.tolist(), pred.tolist()))

print("\n── SpatialBufferCV (5 folds, 0.05° ≈ 5 km buffer) ──")
print(f"  mean RMSE = {np.mean(rmse_per_fold):.3f}  "
      f"± {np.std(rmse_per_fold):.3f}  "
      f"(per-fold: {[round(v, 3) for v in rmse_per_fold]})")


# ── 5. Conformal prediction ─────────────────────────────────────────────
# Split conformal: hold out 20% of TRAINING as a calibration set, refit on
# the remaining 80%, then build distribution-free intervals.
n_tr = len(tr)
n_cal = int(0.20 * n_tr)
cal_perm = rng.permutation(n_tr)
cal_idx, fit_idx = cal_perm[:n_cal], cal_perm[n_cal:]

geo_cf = GeoXGBoost(bandwidth=30, n_estimators=100, max_depth=6,
                    learning_rate=0.3, alpha=1.0, seed=SEED)
geo_cf.fit(Xtr[fit_idx], ytr[fit_idx], coords=Ctr[fit_idx])
cf = geo_cf.conformal_predict(Xtr[cal_idx], ytr[cal_idx].tolist(),
                              Xte, alpha=ALPHA)
lower = np.asarray(cf["lower"])
upper = np.asarray(cf["upper"])
covered = ((yte >= lower) & (yte <= upper)).mean()
width = (upper - lower).mean()

print(f"\n── Conformal prediction (α = {ALPHA}, target {100*(1-ALPHA):.0f}% coverage) ──")
print(f"  empirical coverage = {100*covered:.1f}%")
print(f"  mean interval width = {width:.3f} log-dollars")


# ── 6. Feature importance (averaged across local models) ───────────────
imps = geo.feature_importances_  # list of (name, gain) tuples
if imps:
    # Internal names are "x0", "x1", ... — map back to real feature names.
    def _resolve(n):
        if (n.startswith("x") or n.startswith("f")) and n[1:].isdigit():
            j = int(n[1:])
            if 0 <= j < len(feature_cols):
                return feature_cols[j]
        return n
    imps_named = [(_resolve(n), v) for n, v in imps]
    top = sorted(imps_named, key=lambda kv: -kv[1])[:5]
    print("\n── Top-5 features (mean importance across local models) ──")
    for name, val in top:
        print(f"  {name:>14s}  {val:.4f}")
