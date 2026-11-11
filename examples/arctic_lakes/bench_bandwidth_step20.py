"""Coarser first pass: search_bandwidth with step=20 (and 30) vs the step-10 run."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_arctic as ra
for name in (sys.argv[1:] or ["cation", "doc"]):
    X, y, coords, site = ra.load(name)
    for step in (20, 30):
        g = ra.smelt_geo(60); t0 = time.time()
        r = g.search_bandwidth(X.values, y, coords, min_bw=30, max_bw=300, step=step, tol=2, early_stop=True, seed=42)
        ra.log(f"bench step={step} {name}: best={r['best']} score={r['best_score']:.4f} full={len(r['bandwidths'])} pruned={len(r['pruned'])} "
               f"fits={r['local_fits']} {time.time()-t0:.0f}s; curve={[(b, round(c,3)) for b, c in zip(r['bandwidths'], r['cv'])]}")
    t0 = time.time(); ra.smelt_geo(170).select_bandwidth(X.values, y, coords, candidates=[170]); t1 = time.time() - t0
    t0 = time.time(); ra.smelt_geo(170).select_bandwidth(X.values, y, coords, candidates=[170]); t2 = time.time() - t0
    ra.log(f"bench timing {name}: single candidate k=170 twice: {t1:.1f}s, {t2:.1f}s (load {os.getloadavg()[0]:.1f})")
