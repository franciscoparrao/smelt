"""Second-round check: do SurtGIS lake/catchment variables add skill under spatial CV
with Geo-XGBoost (smelt)? Variants: base, +log lake area, +log catchment/lake ratio,
+local/connected ratio, +all three. NaN where no accepted delineation (XGBoost
handles missing natively; if smelt rejects NaN we fall back to median + flag)."""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_arctic as ra
from sklearn.model_selection import GroupKFold

V = pd.read_csv("/home/franciscoparrao/proyectos/smelt/artic/results/lake_catchments_full/site_catchment_vars.csv")
V["log_lake_area"] = np.log10(V.lake_area_km2)
V["log_catch_lake_ratio"] = np.log10(V.catchment_lake_ratio)
V["kx"] = np.round(V.x, 3); V["ky"] = np.round(V.y, 3)
V = V.drop_duplicates(["kx", "ky"]).set_index(["kx", "ky"])
K = json.load(open(os.path.join(ra.OUT, "selected_k.json"))); K["doc"] = 290
VARIANTS = {"base": [], "+lake_area": ["log_lake_area"], "+catch_ratio": ["log_catch_lake_ratio"],
            "+local_conn": ["local_connected_ratio"], "+all3": ["log_lake_area", "log_catch_lake_ratio", "local_connected_ratio"]}
rows = []
for name in ("cation", "doc"):
    X, y, coords, site = ra.load(name)
    key = pd.MultiIndex.from_arrays([np.round(coords[:, 0] * 1000, 3), np.round(coords[:, 1] * 1000, 3)])
    extra = V.reindex(key)[["log_lake_area", "log_catch_lake_ratio", "local_connected_ratio"]].reset_index(drop=True)
    ra.log(f"catchment test {name}: matched {int(extra.notna().any(axis=1).sum())}/{len(extra)} samples with catchment vars")
    nan_ok = True
    for scheme, size in (("block250", 250), ("block55", 55)):
        folds = list(GroupKFold(5).split(coords, groups=ra.block_ids(coords, size)))
        for vname, cols in VARIANTS.items():
            Xv = pd.concat([X, extra[cols]], axis=1) if cols else X
            if cols and not nan_ok:
                for ccol in cols:
                    Xv[ccol + "_miss"] = Xv[ccol].isna().astype(float); Xv[ccol] = Xv[ccol].fillna(Xv[ccol].median())
            t0 = time.time(); oof = np.full(len(y), np.nan)
            try:
                for tr, te in folds:
                    g = ra.smelt_geo(K[name]); g.fit(Xv.values[tr], y[tr], coords[tr])
                    oof[te] = np.asarray(g.predict(Xv.values[te], coords=coords[te])).ravel()
            except Exception as e:
                if nan_ok and cols:
                    ra.log(f"NaN rejected ({str(e)[:80]}); switching to median+flag"); nan_ok = False
                    for ccol in cols:
                        Xv[ccol + "_miss"] = Xv[ccol].isna().astype(float); Xv[ccol] = Xv[ccol].fillna(Xv[ccol].median())
                    for tr, te in folds:
                        g = ra.smelt_geo(K[name]); g.fit(Xv.values[tr], y[tr], coords[tr])
                        oof[te] = np.asarray(g.predict(Xv.values[te], coords=coords[te])).ravel()
                else:
                    raise
            m = ra.metrics(y, oof); rows.append(dict(target=name, scheme=scheme, variant=vname, k=K[name], nan_native=nan_ok, **m))
            ra.log(f"catchment test {name}/{scheme}/{vname}: R2={m['r2']:.3f} RMSE={m['rmse']:.3f} ({time.time()-t0:.0f}s)")
res = pd.DataFrame(rows); res.to_csv(os.path.join(ra.OUT, "catchment_geoxgb_test.csv"), index=False)
base = res[res.variant == "base"].set_index(["target", "scheme"]).r2
res["delta_r2"] = [r.r2 - base[(r.target, r.scheme)] for r in res.itertuples()]
print(res.pivot_table(index=["target", "scheme"], columns="variant", values="delta_r2").round(3).to_string())
