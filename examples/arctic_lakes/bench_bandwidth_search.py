"""Benchmark: exhaustive LOO grid vs coarse-to-fine search_bandwidth on the Arctic data.

For each target, times (a) the grid already run in run_arctic.py/extend_k.py
(60..300 step 10 plus the +-10 step-2 refinement: 33 candidates for cations) and
(b) GeoXGBoost.search_bandwidth with defaults (30..300, step 10, tol 2, early stop)
and with early_stop=False. Reports best k, LOO score, local fits and wall time.
Output: artic/results/bandwidth_search_benchmark.csv
"""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_arctic as ra

rows = []
for name in (sys.argv[1:] or ["cation", "doc"]):
    X, y, coords, site = ra.load(name)
    n = len(y)
    curve = pd.read_csv(os.path.join(ra.OUT, f"{name}_k_curve.csv"))
    grid_best = int(curve.loc[curve.loo_rmse.idxmin(), "k"])
    grid_fits = len(curve) * n
    # reference grid timing from the logs: cation 1199 s (19 cand) + extension; recompute a subset
    # to have a measured per-candidate cost on this machine today.
    t0 = time.time()
    r_sub = ra.smelt_geo(60).select_bandwidth(X.values, y, coords, candidates=[60, 160, 300])
    t_sub = time.time() - t0
    per_cand = t_sub / 3
    rows.append(dict(target=name, method="grid (33 candidates, measured per-candidate cost x33)", best=grid_best,
                     score=float(curve.loo_rmse.min()), local_fits=grid_fits, seconds=per_cand * len(curve)))
    ra.log(f"bench {name}: grid best={grid_best}, est {per_cand*len(curve):.0f}s for {len(curve)} candidates")
    for es in (True, False):
        g = ra.smelt_geo(60)
        t0 = time.time()
        r = g.search_bandwidth(X.values, y, coords, min_bw=30, max_bw=300, step=10, tol=2, early_stop=es, seed=42)
        dt = time.time() - t0
        rows.append(dict(target=name, method=f"search (early_stop={es})", best=r["best"], score=r["best_score"],
                         local_fits=r["local_fits"], seconds=dt, n_full=len(r["bandwidths"]), n_pruned=len(r["pruned"])))
        ra.log(f"bench {name}: search es={es} best={r['best']} score={r['best_score']:.4f} full={len(r['bandwidths'])} "
               f"pruned={len(r['pruned'])} fits={r['local_fits']}/{r['exhaustive_local_fits']} {dt:.0f}s; "
               f"curve={[(b, round(c,3)) for b, c in zip(r['bandwidths'], r['cv'])]}")
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ra.OUT, "bandwidth_search_benchmark.csv"), index=False)
print(df.to_string())
