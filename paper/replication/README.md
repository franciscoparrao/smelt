# Replication Package

This directory contains scripts to reproduce all benchmark tables in the paper.

## Requirements

### Rust
- Rust 1.89+ (via rustup)
- smelt-ml is compiled from source (parent directory)

### Python
```bash
pip install xgboost lightgbm catboost scikit-learn numpy
```

Tested versions: xgboost 3.1.1, lightgbm 4.6.0, catboost 1.2.8, scikit-learn 1.8.0

## Hardware

Results in the paper were obtained on:
- CPU: Intel Core i7-1270P (12th Gen, 12 cores)
- RAM: 40 GB
- Cache: L1 32K, L2 1280K, L3 18432K
- OS: Linux 6.17.0-19-generic (Ubuntu)

## Profile-Guided Optimization (PGO)

For maximum performance, build with PGO:

```bash
# 1. Install llvm-tools
rustup component add llvm-tools

# 2. Build with profiling
RUSTFLAGS="-C target-cpu=native -Cprofile-generate=/tmp/pgo-data" \
    cargo build --release --example benchmark_large

# 3. Generate profile data (run typical workloads)
./target/release/examples/benchmark_large

# 4. Merge profiles
LLVM_PROFDATA=$(find ~/.rustup -name llvm-profdata | head -1)
$LLVM_PROFDATA merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data/*.profraw

# 5. Rebuild with PGO
cargo clean
RUSTFLAGS="-C target-cpu=native -Cprofile-use=/tmp/pgo-data/merged.profdata" \
    cargo build --release --example benchmark_large
```

PGO typically provides 1.3-2.1x additional speedup over LTO alone.

## Reproducing tables and figures

Run the master script from this directory:

```bash
bash replicate.sh          # all parts, about one hour
bash replicate.sh A B      # selected parts
```

| Part | What it reproduces | Scripts | Time |
|---|---|---|---|
| A | Training-time benchmarks (Tables 9-10), scaling figure (Figure 1), Wilcoxon and bootstrap significance (Table 11) | `benchmark_cpp.py`, `cargo run --example benchmark_large`, `compare_results.py`, `plot_scaling.py`, `statistical_analysis.py` | ~30 min |
| B | Prediction time (Table 12), accuracy validation against scikit-learn (Table 13) | `cargo run --example benchmark_prediction`, `prediction_sklearn.py`, `cargo run --example accuracy_validation`, `accuracy_sklearn.py` | ~5 min |
| C | Histogram-kernel ablation (Table 7) | `cargo run --example ablation_study` | ~2 min |
| D | King County case study (Table 2), Geo-XGBoost validation against geoxgboost 1.1.1 (Table 4), split-conformal intervals (Table 3) | `cargo run --example case_study_king_county`, `cargo run --example validate_geoxgboost_blend` (with `GXGB_ALPHA=0.5`), `validate_geoxgboost_blend.py`, `conformal_intervals.py` | ~15 min |

Part D needs `pip install geoxgboost==1.1.1`. Table numbers refer to the
submitted manuscript. The remaining scripts (`demo_geoxgboost.py`,
`local_importance_map.py`, `validate_geoxgboost.py`, `huasco_validation.py`)
are exploratory and not needed for any table.

## Output files

- `benchmark_cpp_results.json`, `benchmark_rust_results.json`: raw timings (10 runs per configuration)
- `comparison_tables.txt`: Tables 9-10
- `statistical_analysis.txt`: Table 11
- `../scaling.pdf`: Figure 1
- `geoxgb_validation_blend.json`, `smelt_geoxgb_test_preds_a05.csv`, `geoxgb_test_preds_a05.csv`: Table 4
- Console output of the Rust examples and Python scripts for Tables 2, 3, 7, 12 and 13

The versions of these files shipped with the submission were produced on the
hardware listed above.

## Docker (exact reproducibility)

```bash
# From project root:
docker build -t smelt-replication .
docker run --rm smelt-replication

# Interactive shell:
docker run --rm -it smelt-replication bash
```

The Dockerfile pins all dependency versions (Rust stable, scikit-learn 1.8.0,
xgboost 3.1.1, lightgbm 4.6.0, catboost 1.2.8) for exact reproducibility.
