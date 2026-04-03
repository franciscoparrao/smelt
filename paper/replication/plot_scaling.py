#!/usr/bin/env python3
"""Generate scaling comparison figure for the paper."""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(SCRIPT_DIR, "benchmark_cpp_results.json")) as f:
    cpp = json.load(f)
with open(os.path.join(SCRIPT_DIR, "benchmark_rust_results.json")) as f:
    rust = json.load(f)

sizes = [500, 1000, 5000, 10000, 50000, 100000]
engines = [("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("catboost", "CatBoost")]
tasks = [("classification", "Classification"), ("regression", "Regression")]

fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)

for row, (task_key, task_label) in enumerate(tasks):
    for col, (eng_key, eng_label) in enumerate(engines):
        ax = axes[row][col]

        cpp_means = [cpp[task_key][str(n)][eng_key]["mean_ms"] for n in sizes]
        cpp_stds = [cpp[task_key][str(n)][eng_key]["std_ms"] for n in sizes]
        rust_means = [rust[task_key][str(n)][eng_key]["mean_ms"] for n in sizes]
        rust_stds = [rust[task_key][str(n)][eng_key]["std_ms"] for n in sizes]

        ax.errorbar(sizes, cpp_means, yerr=cpp_stds, marker='s', label='C++ official',
                    color='#d62728', capsize=3, linewidth=1.5, markersize=5)
        ax.errorbar(sizes, rust_means, yerr=rust_stds, marker='o', label='smelt-ml (Rust)',
                    color='#1f77b4', capsize=3, linewidth=1.5, markersize=5)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{eng_label} — {task_label}', fontsize=10)
        ax.grid(True, alpha=0.3)

        if row == 1:
            ax.set_xlabel('Samples (n)')
        if col == 0:
            ax.set_ylabel('Training time (ms)')
        if row == 0 and col == 2:
            ax.legend(fontsize=8, loc='upper left')

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "..", "scaling.pdf")
plt.savefig(out_path, bbox_inches='tight', dpi=150)
print(f"Figure saved to {out_path}")

# Also save PNG for quick preview
out_png = os.path.join(SCRIPT_DIR, "..", "scaling.png")
plt.savefig(out_png, bbox_inches='tight', dpi=150)
print(f"Preview saved to {out_png}")
