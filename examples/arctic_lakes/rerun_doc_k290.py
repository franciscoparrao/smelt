"""Official geoxgboost stage (full fit + block250 CV) for DOC at k=290, the best k from
the extended LOO search (smelt block CV: R2 0.321 vs 0.288 at k=154). Writes to
artic/results/doc_k290/ so the k=154 outputs stay intact."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_arctic as ra
ra.OUT = os.path.join(ra.OUT, "doc_k290"); os.makedirs(ra.OUT, exist_ok=True)
ra.LOG = open(os.path.join(ra.OUT, "log.txt"), "a")
X, y, coords, site = ra.load("doc")
ra.log("doc k=290: official geoxgboost rerun start")
ra.official_gxgb("doc", X, y, coords, site, 290)
ra.log("doc k=290: done")
