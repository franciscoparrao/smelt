"""Map spatial non-stationarity from GeoXGBoost local models (King County).

For each training location, GeoXGBoost fits a local model; smelt exposes that
model's feature importances via `local_feature_importances()`. By mapping a
predictor's *local* importance share across space we can see where, e.g., grade
or living area drives price more — the spatially-varying relationships that a
single global model cannot reveal (Grekousis, 2025).

Output:
  paper/replication/figs/local_importance_map.png   (4 most spatially-variable predictors)
  paper/replication/figs/local_importances.csv      (per-location importance shares)

Run with the project virtualenv:
  smelt-py/.venv/bin/python3 paper/replication/local_importance_map.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from smelt import GeoXGBoost

SEED = 42
ALPHA_GEO = 0.5

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "king_county_1k_utm.csv"
OUTDIR = HERE / "figs"
OUTDIR.mkdir(exist_ok=True)

# ── Load ────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
target_col = "log_price"
drop_cols = ["lat", "long", "X", "Y", target_col]
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].to_numpy(float)
y = df[target_col].to_numpy(float)
coords = df[["X", "Y"]].to_numpy(float)
print(f"Loaded {len(df)} samples, {len(feature_cols)} features")

# ── Bandwidth selection (LOO criterion) then fit on all data ─────────────
geo = GeoXGBoost(n_estimators=100, max_depth=6, learning_rate=0.3,
                 alpha=ALPHA_GEO, seed=SEED)
bw_sel = geo.select_bandwidth(X, y.tolist(), coords,
                              candidates=[30, 50, 100, 150, 200])
BW = bw_sel["best"]
print(f"Selected bandwidth (LOO): {BW}")
geo.fit(X, y, coords=coords)

# ── Pull per-location local importances ──────────────────────────────────
lfi = geo.local_feature_importances()
xy = np.asarray(lfi["coords"], float)           # (N, 2) UTM metres
records = lfi["importances"]                    # list of dict{x0:..} or None


def resolve(name):
    if (name.startswith("x") or name.startswith("f")) and name[1:].isdigit():
        j = int(name[1:])
        if 0 <= j < len(feature_cols):
            return feature_cols[j]
    return name


# Build (N, F) matrix of per-location importance SHARES (each row sums to 1).
shares = np.full((len(records), len(feature_cols)), np.nan)
for i, rec in enumerate(records):
    if rec is None:
        continue
    vals = {resolve(k): v for k, v in rec.items()}
    total = sum(vals.values())
    if total <= 0:
        continue
    for j, fc in enumerate(feature_cols):
        shares[i, j] = vals.get(fc, 0.0) / total

share_df = pd.DataFrame(shares, columns=feature_cols)
share_df.insert(0, "X", xy[:, 0])
share_df.insert(1, "Y", xy[:, 1])
share_df.to_csv(OUTDIR / "local_importances.csv", index=False)

# Pick the 4 predictors whose local share varies MOST across space
# (i.e. the strongest spatial non-stationarity).
variability = np.nanstd(shares, axis=0)
order = np.argsort(variability)[::-1]
top4 = [feature_cols[j] for j in order[:4]]
print("Most spatially-variable predictors:", top4)

# ── Plot ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130,
})
fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)
xkm = (xy[:, 0] - xy[:, 0].min()) / 1000.0
ykm = (xy[:, 1] - xy[:, 1].min()) / 1000.0

for ax, feat in zip(axes.ravel(), top4):
    j = feature_cols.index(feat)
    c = shares[:, j]
    m = np.isfinite(c)
    vmax = np.nanpercentile(c[m], 97)  # clip extreme tail for contrast
    sc = ax.scatter(xkm[m], ykm[m], c=c[m], s=16, cmap="magma",
                    vmin=0, vmax=vmax, edgecolors="none", alpha=0.9)
    ax.set_title(f"{feat}", fontsize=11, pad=6)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("local importance share", fontsize=8)
    cb.ax.tick_params(labelsize=7)

fig.suptitle("GeoXGBoost local feature-importance shares across King County\n"
             f"(spatial non-stationarity; adaptive bandwidth = {BW} neighbours, "
             f"alpha = {ALPHA_GEO})",
             fontsize=12)
out = OUTDIR / "local_importance_map.png"
fig.savefig(out, bbox_inches="tight")
print(f"Wrote {out}")
print(f"Wrote {OUTDIR / 'local_importances.csv'}")
