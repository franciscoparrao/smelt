"""Second-round check: does Evapo's water balance add skill under spatial CV
with Geo-XGBoost (smelt), not only with a global RF? Cations and DOC,
block250 and block55 folds, k from selected_k.json, alpha 0.5.
Variants: base, +P-PET(HS), +P-PET(PM), +T_JJAS, +P-PET(HS)+T_JJAS."""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_arctic as ra
from sklearn.model_selection import GroupKFold

W = pd.read_csv('/home/franciscoparrao/proyectos/evapo/data/arctic/lakes_water_balance.csv').set_index('id')
K = json.load(open(os.path.join(ra.OUT, 'selected_k.json')))
VARIANTS = {"base": [], "+PmPET_HS": ["p_minus_pet_hs_ann"], "+PmPET_PM": ["p_minus_pet_pm_ann"],
            "+T_JJAS": ["t_era5l_jjas"], "+PmPET_HS+T_JJAS": ["p_minus_pet_hs_ann", "t_era5l_jjas"]}
rows = []
for name in ("cation", "doc"):
    spec = ra.TARGETS[name]
    df = pd.read_excel(os.path.join(ra.DATA, spec["file"]))
    ids = [f"{name}_{i}" for i in df["consecutive numbering"]]
    X, y, coords, site = ra.load(name)
    extra = W.loc[ids, ["p_minus_pet_hs_ann", "p_minus_pet_pm_ann", "t_era5l_jjas"]].reset_index(drop=True)
    k = K[name]
    for scheme, size in (("block250", 250), ("block55", 55)):
        folds = list(GroupKFold(5).split(coords, groups=ra.block_ids(coords, size)))
        for vname, cols in VARIANTS.items():
            Xv = pd.concat([X, extra[cols]], axis=1) if cols else X
            t0 = time.time(); oof = np.full(len(y), np.nan)
            for tr, te in folds:
                g = ra.smelt_geo(k); g.fit(Xv.values[tr], y[tr], coords[tr])
                oof[te] = np.asarray(g.predict(Xv.values[te], coords=coords[te])).ravel()
            m = ra.metrics(y, oof)
            rows.append(dict(target=name, scheme=scheme, variant=vname, k=k, **m))
            ra.log(f"water-balance test {name}/{scheme}/{vname}: R2={m['r2']:.3f} RMSE={m['rmse']:.3f} ({time.time()-t0:.0f}s)")
res = pd.DataFrame(rows)
res.to_csv(os.path.join(ra.OUT, "water_balance_geoxgb_test.csv"), index=False)
base = res[res.variant == "base"].set_index(["target", "scheme"]).r2
res["delta_r2"] = [r.r2 - base[(r.target, r.scheme)] for r in res.itertuples()]
print(res.pivot_table(index=["target", "scheme"], columns="variant", values="delta_r2").round(3).to_string())
