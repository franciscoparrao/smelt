"""Calibrated spatial prediction intervals for G-XGBoost on King County.

Numbers for the paper's calibrated-uncertainty subsection (agreed with
G. Grekousis, 2026-07-11): split-conformal prediction intervals over
G-XGBoost's SPATIAL predictions, versus the same procedure over the
aspatial XGBoost baseline.

Method (split conformal, Vovk et al.): hold out a calibration set, compute
the absolute residuals |y - yhat| of the model's own predictions on it, and
take the ceil((n_cal + 1) * (1 - alpha))-th smallest residual as the
half-width q. The interval yhat +/- q then covers the truth with
probability >= 1 - alpha on exchangeable data, regardless of the model.
This mirrors `smelt_ml::conformal::SplitConformal` exactly (the Rust API
introduced for spatial learners, whose predict needs coordinates); it is
spelled out in numpy here so the paper's numbers are self-contained.

Setup mirrors demo_geoxgboost.py: king_county_1k_utm.csv, log_price target,
projected UTM X/Y in metres, alpha_geo = 0.5, seed 42, and the bandwidth
selected by the leave-one-out CV criterion (Grekousis 2025, Eq. 11) on the
fit split, over candidates >= 30 (the minimum neighbourhood size, per
author correspondence).

Run:  python conformal_intervals.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from smelt import GeoXGBoost, XGBoost, rmse_score

SEED = 42
ALPHA_GEO = 0.5  # global/local blend, per Grekousis
TARGETS = [0.20, 0.10, 0.05]  # miscoverage levels -> 80/90/95% intervals


def _find_data():
    name = "king_county_1k_utm.csv"
    here = Path(__file__).resolve().parent
    for c in [here / name, here / "data" / name, here.parent.parent / "data" / name]:
        if c.is_file():
            return c
    raise FileNotFoundError(name)


def conformal_halfwidth(cal_pred, cal_true, alpha):
    """Split-conformal residual quantile (same rule as smelt's SplitConformal)."""
    resid = np.sort(np.abs(np.asarray(cal_pred) - np.asarray(cal_true)))
    n = len(resid)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if rank > n else resid[rank - 1]


# ── Data: same columns/roles as demo_geoxgboost.py ─────────────────────
df = pd.read_csv(_find_data())
target_col = "log_price"
coord_cols = ["X", "Y"]
drop_cols = ["lat", "long", "X", "Y", target_col]
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].to_numpy(dtype=np.float64)
y = df[target_col].to_numpy(dtype=np.float64)
coords = df[coord_cols].to_numpy(dtype=np.float64)

# 60% fit / 20% calibration / 20% test. The calibration set must be
# disjoint from fitting (split conformal's only requirement) and
# exchangeable with the test set (both are random draws here).
rng = np.random.default_rng(SEED)
idx = rng.permutation(len(df))
n_fit, n_cal = int(0.6 * len(df)), int(0.2 * len(df))
fit, cal, te = idx[:n_fit], idx[n_fit : n_fit + n_cal], idx[n_fit + n_cal :]
print(f"Split: fit={len(fit)}  calibration={len(cal)}  test={len(te)}")

# ── Bandwidth on the fit split (LOO CV criterion, Eq. 11) ──────────────
geo = GeoXGBoost(n_estimators=100, max_depth=6, learning_rate=0.3,
                 alpha=ALPHA_GEO, seed=SEED)
bw_sel = geo.select_bandwidth(
    X[fit], y[fit].tolist(), coords[fit],
    candidates=[30, 50, 100, 150, 200, 300],
)
print(f"Selected bandwidth: {bw_sel['best']} "
      f"(LOO criterion over candidates {bw_sel['bandwidths']})")

# ── Models on the fit split only ────────────────────────────────────────
geo.fit(X[fit], y[fit], coords[fit])
xgb = XGBoost(n_estimators=100, max_depth=6, learning_rate=0.3, seed=SEED)
xgb.fit(X[fit], y[fit])

geo_cal, geo_te = geo.predict(X[cal], coords[cal]), geo.predict(X[te], coords[te])
xgb_cal, xgb_te = xgb.predict(X[cal]), xgb.predict(X[te])

print(f"\nPoint accuracy on test (RMSE, log price): "
      f"G-XGBoost {rmse_score(y[te].tolist(), geo_te.tolist()):.4f}  |  "
      f"XGBoost {rmse_score(y[te].tolist(), xgb_te.tolist()):.4f}")

# ── Conformal intervals ─────────────────────────────────────────────────
print(f"\n{'Nominal':>8} {'Model':>10} {'Coverage':>9} {'q (log)':>8} "
      f"{'as % of price':>14}")
print("-" * 56)
for alpha in TARGETS:
    for name, cal_pred, te_pred in [
        ("G-XGBoost", geo_cal, geo_te),
        ("XGBoost", xgb_cal, xgb_te),
    ]:
        q = conformal_halfwidth(cal_pred, y[cal], alpha)
        covered = np.mean(np.abs(te_pred - y[te]) <= q)
        # A +/- q interval in log price is a multiplicative band in price:
        # exp(q) - 1 is the half-width as a fraction of the predicted price.
        pct = (np.exp(q) - 1.0) * 100.0
        print(f"{1 - alpha:>7.0%} {name:>10} {covered:>8.1%} {q:>8.3f} "
              f"{'+/- ' + f'{pct:.1f}%':>14}")

print(
    "\nReading: coverage should match the nominal level (the conformal\n"
    "guarantee); the interesting comparison is the WIDTH -- the spatial\n"
    "model's better point predictions earn it tighter calibrated intervals\n"
    "at the same guaranteed coverage."
)
