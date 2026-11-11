"""Extend the bandwidth search to k = 170..300 (step 10) with smelt's LOO criterion
and append to <target>_k_curve.csv. Run after run_arctic.py."""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_arctic as ra

for name in (sys.argv[1:] or ("cation", "doc")):
    X, y, coords, site = ra.load(name)
    cand = [k for k in range(170, 301, 10) if k < len(y)]
    t0 = time.time()
    r = ra.smelt_geo(cand[0]).select_bandwidth(X.values, y, coords, candidates=cand)
    new = pd.DataFrame({"k": r["bandwidths"], "loo_rmse": r["cv"]})
    p = os.path.join(ra.OUT, f"{name}_k_curve.csv")
    cur = pd.read_csv(p)
    allc = pd.concat([cur, new]).drop_duplicates("k").sort_values("k").reset_index(drop=True)
    allc.to_csv(p, index=False)
    best = int(allc.loc[allc.loo_rmse.idxmin(), "k"])
    ra.log(f"{name}: extended k search 170-300 ({time.time()-t0:.0f}s); best overall k={best}; "
           + ", ".join(f"{int(a)}:{b:.3f}" for a, b in zip(new.k, new.loo_rmse)))
