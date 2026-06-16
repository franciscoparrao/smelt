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
    """Locate king_county_1k_utm.csv next to this script, in ./data, or in the repo."""
    name = "king_county_1k_utm.csv"
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

# Geographically-weighted models (GeoXGBoost, GRF, GWR) feed coordinates into a
# Euclidean distance / kernel, so coordinates MUST be in a projected planar CRS,
# not raw lat/long degrees (1° lon != 1° lat in distance). We use PROJECTED
# X, Y in metres (UTM Zone 10N, EPSG:32610 — appropriate for King County, WA).
ALPHA_GEO = 0.5      # blend global+local (knowledge transfer), per Grekousis
BUFFER_M = 5_000.0   # spatial-CV exclusion buffer, in metres


# ── 1. Load and preprocess ──────────────────────────────────────────────
df = pd.read_csv(DATA)
target_col = "log_price"
# lat/long kept for reference only; X, Y (projected, metres) are the coords used.
coord_cols = ["X", "Y"]
drop_cols = ["lat", "long", "X", "Y", target_col]
feature_cols = [c for c in df.columns if c not in drop_cols]

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


# ── 3. Bandwidth selection ──────────────────────────────────────────────
# The neighbourhood size (bandwidth = number of nearest neighbours) is the key
# hyperparameter of any geographically-weighted model and should be tuned, not
# guessed. `select_bandwidth` minimises the leave-one-out CV criterion of
# Grekousis (2025, Eq. 11): each location is predicted by a LOCAL model fit on
# its neighbours excluding itself, and the criterion is the LOO error. This is a
# property of the local model alone — the global model and alpha are not used,
# because bandwidth is tuned before the ensemble step. A too-small neighbourhood
# leaves each excluded point with too few neighbours, so the criterion spikes;
# the optimum is therefore typically large. The selected value is stored on the
# estimator, so the subsequent fit() uses it.
geo = GeoXGBoost(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    alpha=ALPHA_GEO,     # 0.5 = blend global+local (used at fit time, not tuning)
    seed=SEED,
)
bw_sel = geo.select_bandwidth(
    Xtr, ytr.tolist(), Ctr,
    candidates=[30, 50, 100, 150, 200, 300],
)
BW_GEO = bw_sel["best"]
print("\n── Bandwidth selection (leave-one-out CV criterion, training set) ──")
for bw, c in zip(bw_sel["bandwidths"], bw_sel["cv"]):
    print(f"  bandwidth={bw:<4d} CV-criterion={c:.4f}" + ("   <- selected" if bw == BW_GEO else ""))


# ── 4. GeoXGBoost ───────────────────────────────────────────────────────
# 100 trees, depth 6, learning_rate 0.3, CV-selected bandwidth (BW_GEO).
# alpha=0.5 blends the global and local models so the global model transfers
# knowledge to each local model (Grekousis) — this improves on pure-local
# (alpha=1) prediction. Coordinates are projected X, Y in metres.
# `geo` already carries the selected bandwidth from select_bandwidth above.
geo.fit(Xtr, ytr, coords=Ctr)
# Spatial-aware prediction: pass new coords so each test point uses
# its nearest local model.
pred_geo = geo.predict(Xte, coords=Cte)
rmse_geo = rmse_score(yte.tolist(), pred_geo.tolist())
r2_geo   = r2_score(yte.tolist(), pred_geo.tolist())

print("\n── Holdout performance (test n = {:d}, projected UTM coords) ──".format(len(te)))
print(f"  XGBoost              RMSE={rmse_xgb:.3f}  R²={r2_xgb:.3f}")
print(f"  XGBoost + X,Y        RMSE={rmse_aug:.3f}  R²={r2_aug:.3f}")
print(f"  GeoXGBoost (α={ALPHA_GEO}, bw={BW_GEO})  RMSE={rmse_geo:.3f}  R²={r2_geo:.3f}")


# ── 5. Spatial cross-validation ─────────────────────────────────────────
# SpatialBufferCV excludes training samples within `buffer_distance` of
# each test fold's centroid — prevents spatial leakage. With projected
# coordinates the buffer is in metres (5 km here).
cv = SpatialBufferCV(n_folds=5, coords=coords, buffer_distance=BUFFER_M, seed=SEED)
rmse_per_fold = []
for fold_idx, (train_idx, test_idx) in enumerate(cv.splits(len(df))):
    if len(train_idx) < 10 or len(test_idx) == 0:
        continue
    Xtr_f, ytr_f, Ctr_f = X[train_idx], y[train_idx], coords[train_idx]
    Xte_f, yte_f, Cte_f = X[test_idx],  y[test_idx],  coords[test_idx]
    m = GeoXGBoost(bandwidth=BW_GEO, n_estimators=100, max_depth=6,
                   learning_rate=0.3, alpha=ALPHA_GEO, seed=SEED)
    m.fit(Xtr_f, ytr_f, coords=Ctr_f)
    pred = m.predict(Xte_f, coords=Cte_f)
    rmse_per_fold.append(rmse_score(yte_f.tolist(), pred.tolist()))

print("\n── SpatialBufferCV (5 folds, 5 km buffer, projected coords) ──")
print(f"  mean RMSE = {np.mean(rmse_per_fold):.3f}  "
      f"± {np.std(rmse_per_fold):.3f}  "
      f"(per-fold: {[round(v, 3) for v in rmse_per_fold]})")


# ── 6. Conformal prediction ─────────────────────────────────────────────
# Split conformal: hold out 20% of TRAINING as a calibration set, refit on
# the remaining 80%, then build distribution-free intervals.
n_tr = len(tr)
n_cal = int(0.20 * n_tr)
cal_perm = rng.permutation(n_tr)
cal_idx, fit_idx = cal_perm[:n_cal], cal_perm[n_cal:]

geo_cf = GeoXGBoost(bandwidth=BW_GEO, n_estimators=100, max_depth=6,
                    learning_rate=0.3, alpha=ALPHA_GEO, seed=SEED)
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


# ── 7. Feature importance (averaged across local models) ───────────────
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
