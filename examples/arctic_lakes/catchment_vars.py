"""Turn SurtGIS lake-catchment output into per-site predictors.

Rules (agreed with the SurtGIS session, 2026-11-06):
- prefer the FABDEM row when ok, else GLO-30; keep dem_spread as uncertainty
- status ok: accept; review: accept only if compactness >= threshold (p10 of ok rows)
  and lake_area < 500 km2; point_only / edge / error: NA
- tiny-lake rule: lake_area < 0.01 km2 and a HydroLAKES lake within 500 m -> use the
  HydroLAKES lake area (source hydrolakes-near), catchment stays from the DEM row
Outputs: artic/results/lake_catchments_full/site_catchment_vars.csv and a summary.
"""
import os, numpy as np, pandas as pd
D = "/home/franciscoparrao/proyectos/smelt/artic/results/lake_catchments_full/"
c = pd.read_csv(D + "catchments.csv")
s = pd.read_csv(D + "sites.csv")
near = pd.read_csv("/home/franciscoparrao/proyectos/smelt/artic/results/lake_catchments_surtgis/near.csv").drop_duplicates(["x", "y"])
print("filas", c.shape, "| columnas:", list(c.columns))
print("\nstatus por DEM:\n", pd.crosstab(c.dem, c.status))
if "lake_source" in c:
    print("\nlake_source (GLO-30):\n", c[c.dem == "cop-dem-glo-30"].lake_source.value_counts(dropna=False).to_dict())
err = c[c.status == "error"].message.astype(str).str[:60].value_counts().head(5)
print("\nmensajes de error (top):\n", err.to_string())
ok = c[c.status == "ok"]
thr = float(np.nanpercentile(ok.lake_compactness, 10)) if "lake_compactness" in c else np.nan
print(f"\numbral compacidad (p10 de ok): {thr:.3f}")

def pick(g):
    g = g.copy()
    g["rank"] = g.dem.map({"fabdem": 0, "cop-dem-glo-30": 1, "alos-dem": 2})
    acc = g[(g.status == "ok") | ((g.status == "review") & (g.lake_compactness >= thr) & (g.lake_area_km2 < 500))]
    if len(acc) == 0:
        r = g.sort_values("rank").iloc[0].copy(); r["accepted"] = False; return r
    r = acc.sort_values("rank").iloc[0].copy(); r["accepted"] = True; return r

best = pd.DataFrame([pick(g) for _, g in c.groupby("lake_id")]).reset_index(drop=True)
best = best.rename(columns={"lake_id": "site_id"}).merge(s, on="site_id", how="right")
assert len(best) == len(s), (len(best), len(s))
# tiny-lake rule with HydroLAKES near (near.csv has dmin_m per x,y)
best = best.merge(near, on=["x", "y"], how="left")
tiny = best.accepted & (best.lake_area_km2 < 0.01) & (best.dmin_m < 500) & best.hl_lake_area.notna()
best.loc[tiny, "lake_area_km2"] = best.loc[tiny, "hl_lake_area"]
best.loc[tiny, "lake_source"] = "hydrolakes-near"
best["catchment_lake_ratio"] = best.catchment_area_km2 / best.lake_area_km2
best["local_connected_ratio"] = best.local_catchment_area_km2 / best.catchment_area_km2
for col in ["lake_area_km2", "catchment_area_km2", "local_catchment_area_km2", "catchment_lake_ratio",
            "local_connected_ratio", "catchment_mean_elev_m", "lake_elev_m", "dem_spread"]:
    best.loc[~best.accepted.fillna(False), col] = np.nan
print(f"\nsitios: {len(best)} | aceptados: {int(best.accepted.sum())} | tiny->hydrolakes: {int(tiny.sum())}")
print("status del elegido:", best.status.value_counts(dropna=False).to_dict())
print("dem del elegido:", best.dem.value_counts(dropna=False).to_dict())
hl = best[best.hl_lake_area.notna() & best.accepted]
r_lake = (best.lake_area_km2 / best.hl_lake_area)[best.hl_lake_area.notna() & best.accepted]
r_wshd = (best.catchment_area_km2 / best.hl_wshd_area)[best.hl_wshd_area.notna() & best.accepted]
print(f"\nvs HydroLAKES (n={len(hl)}): area lago ratio mediana {r_lake.median():.2f}, dentro de ±20% {(r_lake.between(0.8,1.25)).mean():.0%}, ±50% {(r_lake.between(0.5,2)).mean():.0%}")
print(f"  cuenca conectada ratio mediana {r_wshd.median():.2f}, dentro de ±50% {(r_wshd.between(0.5,2)).mean():.0%}")
print("\nresumen variables:\n", best[["lake_area_km2", "catchment_area_km2", "catchment_lake_ratio", "local_connected_ratio"]].describe(percentiles=[.1,.5,.9]).round(3).to_string())
best.to_csv(D + "site_catchment_vars.csv", index=False)
print("\nguardado", D + "site_catchment_vars.csv")
