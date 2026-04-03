#!/usr/bin/env python3
"""
Compare Rust vs C++ benchmark results and generate formatted tables.

Reads benchmark_cpp_results.json and benchmark_rust_results.json,
computes speedup ratios, and outputs LaTeX-formatted tables matching
the paper format.
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_results():
    cpp_path = os.path.join(SCRIPT_DIR, "benchmark_cpp_results.json")
    rust_path = os.path.join(SCRIPT_DIR, "benchmark_rust_results.json")

    with open(cpp_path) as f:
        cpp = json.load(f)
    with open(rust_path) as f:
        rust = json.load(f)
    return cpp, rust


def format_table(task_type, cpp, rust):
    sizes = ["500", "1000", "5000", "10000", "50000", "100000"]
    engines = ["xgboost", "lightgbm", "catboost"]

    print(f"\n{'='*80}")
    print(f"  {task_type.upper()} - Training Time (ms, mean +/- std, 10 runs)")
    print(f"{'='*80}")

    # Header
    print(f"{'N':>8s}", end="")
    for eng in engines:
        print(f"  | {'C++':>10s} {'Rust':>10s} {'Speed':>6s}", end="")
    print()
    print("-" * 80)

    for n in sizes:
        if n not in cpp[task_type] or n not in rust[task_type]:
            continue

        n_fmt = f"{int(n):,}"
        print(f"{n_fmt:>8s}", end="")

        for eng in engines:
            c = cpp[task_type][n][eng]
            r = rust[task_type][n][eng]

            c_str = f"{c['mean_ms']:.0f}±{c['std_ms']:.0f}"
            r_str = f"{r['mean_ms']:.0f}±{r['std_ms']:.0f}"
            speedup = c["mean_ms"] / r["mean_ms"]

            print(f"  | {c_str:>10s} {r_str:>10s} {speedup:>5.1f}x", end="")

        print()

    print()


def main():
    cpp, rust = load_results()

    print("Hardware:", cpp.get("hardware", {}).get("cpu_model", "N/A"))
    print("C++ versions:", cpp.get("versions", {}))

    format_table("classification", cpp, rust)
    format_table("regression", cpp, rust)

    # Summary statistics
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    for task in ["classification", "regression"]:
        for eng in ["xgboost", "lightgbm", "catboost"]:
            speedups = []
            for n in ["500", "1000", "5000", "10000", "50000", "100000"]:
                if n in cpp[task] and n in rust[task]:
                    c = cpp[task][n][eng]["mean_ms"]
                    r = rust[task][n][eng]["mean_ms"]
                    speedups.append((int(n), c / r))

            faster = [(n, s) for n, s in speedups if s > 1.0]
            slower = [(n, s) for n, s in speedups if s <= 1.0]

            print(f"\n  {task} - {eng}:")
            if faster:
                print(f"    Rust faster at: {', '.join(f'N={n:,} ({s:.1f}x)' for n, s in faster)}")
            if slower:
                print(f"    C++ faster at:  {', '.join(f'N={n:,} ({1/s:.1f}x)' for n, s in slower)}")

    # Save to file
    out_path = os.path.join(SCRIPT_DIR, "comparison_tables.txt")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("Hardware:", cpp.get("hardware", {}).get("cpu_model", "N/A"))
        print("C++ versions:", cpp.get("versions", {}))
        format_table("classification", cpp, rust)
        format_table("regression", cpp, rust)
    with open(out_path, "w") as f:
        f.write(buf.getvalue())
    print(f"\nTables saved to {out_path}")


if __name__ == "__main__":
    main()
