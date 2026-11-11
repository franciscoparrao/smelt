"""Does the extended-search k change spatial-CV skill? smelt Geo-XGBoost,
alpha 0.5, block250 and block55, selected k vs extended best k."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_arctic as ra
from sklearn.model_selection import GroupKFold
PAIRS = {"cation": (160, 170), "doc": (154, 290)}
rows = []
for name, ks in PAIRS.items():
    X, y, coords, site = ra.load(name)
    for scheme, size in (("block250", 250), ("block55", 55)):
        folds = list(GroupKFold(5).split(coords, groups=ra.block_ids(coords, size)))
        for k in ks:
            t0 = time.time(); oof = np.full(len(y), np.nan)
            for tr, te in folds:
                g = ra.smelt_geo(k); g.fit(X.values[tr], y[tr], coords[tr])
                oof[te] = np.asarray(g.predict(X.values[te], coords=coords[te])).ravel()
            m = ra.metrics(y, oof); rows.append(dict(target=name, scheme=scheme, k=k, **m))
            ra.log(f"k check {name}/{scheme}/k={k}: R2={m['r2']:.3f} RMSE={m['rmse']:.3f} ({time.time()-t0:.0f}s)")
pd.DataFrame(rows).to_csv(os.path.join(ra.OUT, "k_alt_check.csv"), index=False)
print(pd.DataFrame(rows).pivot_table(index=["target", "scheme"], columns="k", values="r2").round(3).to_string())
