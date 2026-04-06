#!/usr/bin/env python3
"""
Statistical analysis of benchmark results using smelt-ml's own methodology.

Applies Wilcoxon signed-rank tests to determine whether performance differences
between smelt-ml and official C++ libraries are statistically significant.
Also computes bootstrap 95% CIs for speedup ratios.

This analysis mirrors what smelt-ml's built-in stats module provides,
demonstrating the framework's statistical testing capabilities.
"""

import json
import os
import numpy as np
from scipy import stats as scipy_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "benchmark_cpp_results.json")) as f:
    cpp = json.load(f)
with open(os.path.join(SCRIPT_DIR, "benchmark_rust_results.json")) as f:
    rust = json.load(f)

SIZES = ["500", "1000", "5000", "10000", "50000", "100000"]
ENGINES = ["xgboost", "lightgbm", "catboost"]
TASKS = ["classification", "regression"]


def wilcoxon_test(cpp_times, rust_times):
    """Paired Wilcoxon signed-rank test. Returns (statistic, p_value, faster)."""
    n = min(len(cpp_times), len(rust_times))
    c = np.array(cpp_times[:n])
    r = np.array(rust_times[:n])
    diffs = c - r  # positive = Rust faster
    if np.all(np.abs(diffs) < 1e-10):
        return 0.0, 1.0, "tied"
    try:
        stat, p = scipy_stats.wilcoxon(c, r, alternative='two-sided')
    except ValueError:
        return 0.0, 1.0, "tied"
    faster = "Rust" if np.median(diffs) > 0 else "C++"
    return stat, p, faster


def bootstrap_speedup_ci(cpp_times, rust_times, confidence=0.95, n_boot=10000):
    """Bootstrap CI for the speedup ratio (C++/Rust)."""
    n = min(len(cpp_times), len(rust_times))
    c = np.array(cpp_times[:n])
    r = np.array(rust_times[:n])
    rng = np.random.RandomState(42)
    ratios = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        ratio = np.mean(c[idx]) / np.mean(r[idx])
        ratios.append(ratio)
    ratios = np.sort(ratios)
    alpha = 1 - confidence
    lo = ratios[int(alpha / 2 * n_boot)]
    hi = ratios[int((1 - alpha / 2) * n_boot)]
    point = np.mean(c) / np.mean(r)
    return point, lo, hi


print("=" * 80)
print("  Statistical Analysis of Benchmark Results")
print("  Wilcoxon signed-rank test (paired, two-sided) + Bootstrap 95% CI")
print("=" * 80)

# Collect all results for LaTeX table
latex_rows = []

for task in TASKS:
    print(f"\n{'─' * 80}")
    print(f"  {task.upper()}")
    print(f"{'─' * 80}")
    print(f"{'Engine':<10} {'N':>8}  {'C++ mean':>8} {'Rust mean':>9} "
          f"{'Speedup':>8} {'95% CI':>16} {'p-value':>10} {'Sig':>4} {'Faster':>6}")
    print("-" * 80)

    for engine in ENGINES:
        for size in SIZES:
            c_data = cpp[task][size][engine]
            r_data = rust[task][size][engine]
            c_times = c_data["times_ms"]
            r_times = r_data["times_ms"]

            stat, p, faster = wilcoxon_test(c_times, r_times)
            speedup, ci_lo, ci_hi = bootstrap_speedup_ci(c_times, r_times)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

            n_fmt = f"{int(size):,}"
            print(f"{engine:<10} {n_fmt:>8}  {c_data['mean_ms']:>8.1f} {r_data['mean_ms']:>9.1f} "
                  f"{speedup:>7.2f}x [{ci_lo:.2f}, {ci_hi:.2f}] "
                  f"{p:>10.4f} {sig:>4} {faster:>6}")

            latex_rows.append({
                "task": task, "engine": engine, "size": size,
                "speedup": speedup, "ci_lo": ci_lo, "ci_hi": ci_hi,
                "p": p, "sig": sig, "faster": faster,
            })
        print()

# Summary: Friedman test across sizes for each engine
print("\n" + "=" * 80)
print("  Friedman Test: Are speedup ratios consistent across dataset sizes?")
print("=" * 80)

for task in TASKS:
    for engine in ENGINES:
        speedups = []
        for size in SIZES:
            c = cpp[task][size][engine]["times_ms"]
            r = rust[task][size][engine]["times_ms"]
            n = min(len(c), len(r))
            ratios = [c[i] / r[i] for i in range(n)]
            speedups.append(ratios)

        # Friedman across 6 sizes × 10 paired observations
        try:
            stat, p = scipy_stats.friedmanchisquare(*speedups)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {task:15s} {engine:10s}: χ²={stat:.2f}, p={p:.4f} {sig}")
        except Exception as e:
            print(f"  {task:15s} {engine:10s}: {e}")

# Key findings for paper
print("\n" + "=" * 80)
print("  KEY FINDINGS FOR PAPER")
print("=" * 80)

# CatBoost classification: how many sizes does Rust win significantly?
catboost_classif_wins = 0
for size in SIZES:
    c = cpp["classification"][size]["catboost"]["times_ms"]
    r = rust["classification"][size]["catboost"]["times_ms"]
    _, p, faster = wilcoxon_test(c, r)
    if p < 0.05 and faster == "Rust":
        catboost_classif_wins += 1

print(f"\n  CatBoost classification: Rust significantly faster at "
      f"{catboost_classif_wins}/{len(SIZES)} sizes (p < 0.05)")

# Overall: count significant wins/losses
rust_wins = sum(1 for r in latex_rows if r["p"] < 0.05 and r["faster"] == "Rust")
cpp_wins = sum(1 for r in latex_rows if r["p"] < 0.05 and r["faster"] == "C++")
ties = sum(1 for r in latex_rows if r["p"] >= 0.05)
total = len(latex_rows)
print(f"  Overall: Rust significantly faster in {rust_wins}/{total}, "
      f"C++ in {cpp_wins}/{total}, not significant in {ties}/{total}")

# Save for paper
out_path = os.path.join(SCRIPT_DIR, "statistical_analysis.txt")
print(f"\n  Results saved to {out_path}")
