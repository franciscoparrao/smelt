"""Arctic lakes: Geo-XGBoost analysis of major cations and DOC.

Stage A (smelt-ml GeoXGBoost, fast, Rust):
  A1 bandwidth k by the LOO criterion (Grekousis 2025, Eq. 11), coarse then fine.
  A2 cross-validation under four schemes (random, site-grouped, 55 km blocks,
     250 km blocks) for RF, XGBoost and Geo-XGBoost (alpha 0.5), plus an alpha
     sensitivity sweep under the 250 km block scheme.
  A3 full-data fit and per-location local importances.
Stage B (official geoxgboost 1.1.1, slow, Python):
  B1 full-data gxgb at the selected k: OOB stats and local importances.
  B2 250 km block CV with gxgb + predict_gxgb (alpha 0.5 fixed).

Run:  SMOKE=1 for a 200-sample dry run.  Outputs in artic/results/.
"""
import contextlib
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "artic")
OUT = os.path.join(DATA, "results")
os.makedirs(OUT, exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"
SEED = 42
ALPHA = 0.5
XGB = dict(n_estimators=200, max_depth=4, learning_rate=0.1, reg_lambda=1.0)

TARGETS = {
    "cation": dict(
        file="Original dataset_20261010_v01_XGboost analysis_Cation_1731 samples _updated.xlsx",
        y="cation_meq/L", geo="Geology_glim_wgs"),
    "doc": dict(
        file="Original dataset_20261010_v01_XGboost analysis_DOC_1118 samples_updated.xlsx",
        y="DOC_mg/L", geo=None),
}
NUMERIC = {
    "Elevation_m": "Elev",
    "Permafrost extent (CCI) (1997-2023)": "PermExt",
    "ALT mean (CCI)_cm (1997-2023)": "ALT",
    "GIMMS_NDVI_annual_peak_mean_1982_2022": "NDVI",
    "Temp_Yearly_CRU (1980-2024)_°C": "Temp",
    "Precip_Yearly_GPCC (1980-2024)_mm": "Pre",
    "Month.1": "Month",
}
GEO_ABBR = {
    "Acid Plutonic Rocks": "AP", "Acid Volcanic Rocks": "AV", "Basic Plutonic Rocks": "BP",
    "Basic Volcanic Rocks": "BV", "Carbonate Sedimentary Rocks": "CS",
    "Intermediate Plutonic Rocks": "IP", "Intermediate Volcanic Rocks": "IV",
    "Metamorphic Rocks": "ME", "Mixed Sedimentary Rocks": "MS", "Pyroclastics": "PY",
    "Siliciclastic Sedimentary Rocks": "SC", "Unconsolidated Sediments": "US",
    "Ice and Glaciers": "IG", "Water Bodies": "WB",
}

LOG = open(os.path.join(OUT, "log.txt"), "a")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def load(name):
    spec = TARGETS[name]
    df = pd.read_excel(os.path.join(DATA, spec["file"]))
    y = np.log(df[spec["y"]].values)
    X = df[list(NUMERIC)].rename(columns=NUMERIC).astype(float)
    if spec["geo"]:
        X["SOCC"] = np.nan  # placeholder removed below; keeps column order explicit
        X = X.drop(columns="SOCC")
        d = pd.get_dummies(df[spec["geo"]].map(lambda g: "Geo_" + GEO_ABBR.get(g, g[:2])), dtype=float)
        X = pd.concat([X, d], axis=1)
    else:
        X.insert(1, "SOCC", df["SOCC_calculated_kg/m2/cm_NCSCDv2 "].astype(float).values)
    coords_km = df[["X_EPSG3995_m", "Y_EPSG3995_m"]].values / 1000.0
    site = pd.factorize(pd.Series([tuple(r) for r in coords_km]))[0]
    if SMOKE:
        rng = np.random.RandomState(SEED)
        keep = np.sort(rng.choice(len(df), 200, replace=False))
        X, y, coords_km, site = X.iloc[keep].reset_index(drop=True), y[keep], coords_km[keep], site[keep]
    return X, y, coords_km, site


def block_ids(coords_km, size_km):
    return pd.factorize(pd.Series(list(zip(np.floor(coords_km[:, 0] / size_km), np.floor(coords_km[:, 1] / size_km)))))[0]


def schemes(coords_km, site, n_folds=5):
    from sklearn.model_selection import GroupKFold, KFold
    out = {}
    out["random"] = list(KFold(n_folds, shuffle=True, random_state=SEED).split(coords_km))
    out["site"] = list(GroupKFold(n_folds).split(coords_km, groups=site))
    for s in (55, 250):
        out[f"block{s}"] = list(GroupKFold(n_folds).split(coords_km, groups=block_ids(coords_km, s)))
    return out


def metrics(y, p):
    from sklearn.metrics import mean_squared_error, r2_score
    return dict(rmse=float(np.sqrt(mean_squared_error(y, p))), r2=float(r2_score(y, p)),
                mae=float(np.mean(np.abs(y - p))))


def smelt_geo(k, alpha=ALPHA):
    import smelt
    return smelt.GeoXGBoost(bandwidth=int(k), n_estimators=XGB["n_estimators"], max_depth=XGB["max_depth"],
                            learning_rate=XGB["learning_rate"], lambda_=XGB["reg_lambda"], alpha=alpha, seed=SEED)


# ---------------------------------------------------------------- Stage A1
def select_k(name, X, y, coords):
    t0 = time.time()
    coarse = list(range(60, 151, 10)) if not SMOKE else [40, 60, 80]
    coarse = [c for c in coarse if c < len(y)]
    r = smelt_geo(coarse[0]).select_bandwidth(X.values, y, coords, candidates=coarse)
    best = int(r["best"])
    fine = [k for k in range(best - 10, best + 11, 2) if 30 <= k < len(y) and k not in coarse]
    r2 = smelt_geo(best).select_bandwidth(X.values, y, coords, candidates=fine) if fine else None
    curve = list(zip(r["bandwidths"], r["cv"])) + (list(zip(r2["bandwidths"], r2["cv"])) if r2 else [])
    curve.sort()
    best = int(min(curve, key=lambda t: t[1])[0])
    pd.DataFrame(curve, columns=["k", "loo_rmse"]).to_csv(os.path.join(OUT, f"{name}_k_curve.csv"), index=False)
    log(f"{name}: selected k={best} ({len(curve)} candidates, {time.time()-t0:.0f}s)")
    return best, curve


# ---------------------------------------------------------------- Stage A2
def cv_stage(name, X, y, coords, site, k):
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    rows, oof = [], {}
    for scheme, folds in schemes(coords, site).items():
        preds = {"rf": np.full(len(y), np.nan), "xgb": np.full(len(y), np.nan), "geo": np.full(len(y), np.nan)}
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0] if scheme == "block250" else []
        for a in alphas:
            preds[f"geo_a{a}"] = np.full(len(y), np.nan)
        t0 = time.time()
        for tr, te in folds:
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
            preds["rf"][te] = rf.fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            m = xgb.XGBRegressor(**XGB, random_state=SEED, n_jobs=8, verbosity=0)
            preds["xgb"][te] = m.fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            g = smelt_geo(k)
            g.fit(X.values[tr], y[tr], coords[tr])
            preds["geo"][te] = np.asarray(g.predict(X.values[te], coords=coords[te])).ravel()
            for a in alphas:
                ga = smelt_geo(k, alpha=a)
                ga.fit(X.values[tr], y[tr], coords[tr])
                preds[f"geo_a{a}"][te] = np.asarray(ga.predict(X.values[te], coords=coords[te])).ravel()
        for model, p in preds.items():
            rows.append(dict(target=name, scheme=scheme, model=model, k=k, **metrics(y, p)))
        oof[scheme] = preds
        log(f"{name}/{scheme}: " + ", ".join(f"{m}={metrics(y, p)['r2']:.3f}" for m, p in preds.items())
            + f" ({time.time()-t0:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, f"{name}_cv_results.csv"), index=False)
    df = pd.DataFrame({"y_log": y, "x_km": coords[:, 0], "y_km": coords[:, 1], "site": site})
    for scheme, preds in oof.items():
        for model, p in preds.items():
            df[f"{scheme}__{model}"] = p
    df.to_csv(os.path.join(OUT, f"{name}_oof_predictions.csv"), index=False)
    return res


# ---------------------------------------------------------------- Stage A3
def full_fit_smelt(name, X, y, coords, k):
    g = smelt_geo(k)
    g.fit(X.values, y, coords)
    li = g.local_feature_importances()
    rows = []
    for i, imp in enumerate(li["importances"]):
        row = dict(x_km=coords[i, 0], y_km=coords[i, 1])
        if imp is not None:
            for j, col in enumerate(X.columns):
                row[col] = imp.get(f"x{j}", 0.0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{name}_local_importance_smelt.csv"), index=False)
    gi = g.feature_importances_
    if gi:
        pd.DataFrame([(X.columns[int(n[1:])] if n.startswith("x") else n, v) for n, v in gi],
                     columns=["feature", "gain"]).to_csv(os.path.join(OUT, f"{name}_global_importance_smelt.csv"), index=False)
    log(f"{name}: smelt full fit done, local importances for {len(rows)} locations")


# ---------------------------------------------------------------- Stage B
def official_gxgb(name, X, y, coords, site, k):
    from geoxgboost import gxgb, predict_gxgb, get_local_feature_importance
    params = dict(**XGB, reg_alpha=0, random_state=SEED, n_jobs=1, verbosity=0)
    Xd = X.reset_index(drop=True)
    Cd = pd.DataFrame(coords, columns=["x_km", "y_km"])
    yd = pd.Series(y)
    # B1 full data
    t0 = time.time()
    pdir = os.path.join(OUT, f"{name}_gxgb_full") + os.sep
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, "gxgb_stdout.txt"), "w") as f, contextlib.redirect_stdout(f):
        res = gxgb(X=Xd, y=yd, Coords=Cd, params=params, bw=k, Kernel="Adaptive", spatial_weights=True, alpha_wt_type="fixed",
                   alpha_wt=ALPHA, test_size=0.2, seed=SEED, n_splits=5 if not SMOKE else 2, path_save=pdir)
    res["Stats"].to_csv(os.path.join(OUT, f"{name}_gxgb_full_stats.csv"))
    res["Prediction"].to_csv(os.path.join(OUT, f"{name}_gxgb_full_prediction.csv"))
    imp = get_local_feature_importance(res)
    imp = pd.concat([Cd, imp.reset_index(drop=True)], axis=1)
    imp.to_csv(os.path.join(OUT, f"{name}_local_importance_gxgb.csv"), index=False)
    log(f"{name}: official gxgb full fit done ({time.time()-t0:.0f}s); stats: " +
        res["Stats"].iloc[0].to_dict().__repr__()[:300])
    # B2 block CV
    folds = schemes(coords, site)["block250"]
    oof = np.full(len(y), np.nan)
    for fi, (tr, te) in enumerate(folds):
        t0 = time.time()
        fdir = os.path.join(OUT, f"{name}_gxgb_fold{fi}") + os.sep
        os.makedirs(fdir, exist_ok=True)
        with open(os.path.join(fdir, "gxgb_stdout.txt"), "w") as f, contextlib.redirect_stdout(f):
            r = gxgb(X=Xd.iloc[tr].reset_index(drop=True), y=yd.iloc[tr].reset_index(drop=True),
                     Coords=Cd.iloc[tr].reset_index(drop=True), params=params, bw=k, Kernel="Adaptive", spatial_weights=True,
                     alpha_wt_type="fixed", alpha_wt=ALPHA, test_size=0.2, seed=SEED,
                     n_splits=5 if not SMOKE else 2, path_save=fdir)
            p = predict_gxgb(DataPredict=Xd.iloc[te].reset_index(drop=True),
                             CoordsPredict=Cd.iloc[te].reset_index(drop=True),
                             Coords=Cd.iloc[tr].reset_index(drop=True), Output_GXGB_LocalModel=r,
                             alpha_wt=ALPHA, alpha_wt_type="fixed", path_save=fdir)
        oof[te] = p.iloc[:, -1].values
        log(f"{name}: official gxgb fold {fi} done ({time.time()-t0:.0f}s), fold R2={metrics(y[te], oof[te])['r2']:.3f}")
    m = metrics(y, oof)
    pd.DataFrame([dict(target=name, scheme="block250", model="gxgb_official", k=k, **m)]).to_csv(
        os.path.join(OUT, f"{name}_cv_results_gxgb_official.csv"), index=False)
    pd.DataFrame({"y_log": y, "x_km": coords[:, 0], "y_km": coords[:, 1], "gxgb_official": oof}).to_csv(
        os.path.join(OUT, f"{name}_oof_gxgb_official.csv"), index=False)
    log(f"{name}: official gxgb block250 CV: R2={m['r2']:.3f} RMSE={m['rmse']:.3f}")


def main():
    log(f"=== start (smoke={SMOKE}) ===")
    data = {n: load(n) for n in TARGETS}
    chosen = {}
    kfile = os.path.join(OUT, "selected_k.json")
    if os.environ.get("STAGE") == "B" and os.path.exists(kfile):
        chosen = json.load(open(kfile))
        log(f"stage A skipped, using k={chosen}")
        data_items = []
    else:
        data_items = list(data.items())
    for name, (X, y, coords, site) in data_items:
        log(f"{name}: n={len(y)} features={list(X.columns)}")
        k, _ = select_k(name, X, y, coords)
        chosen[name] = k
        cv_stage(name, X, y, coords, site, k)
        full_fit_smelt(name, X, y, coords, k)
    json.dump(chosen, open(os.path.join(OUT, "selected_k.json"), "w"))
    log("=== stage A done; starting official geoxgboost ===")
    for name, (X, y, coords, site) in data.items():
        official_gxgb(name, X, y, coords, site, chosen[name])
    log("=== all done ===")


if __name__ == "__main__":
    main()
