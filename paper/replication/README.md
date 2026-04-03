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

## Reproducing Tables

Run the master script:

```bash
cd paper/replication
bash replicate.sh
```

This will:
1. Run C++ library benchmarks via Python (`benchmark_cpp.py`)
2. Compile and run Rust benchmarks (`benchmark_large` example)
3. Generate comparison tables (`compare_results.py`)

Expected total runtime: ~30 minutes on similar hardware.

### Individual steps

```bash
# Step 1: C++ benchmarks (~15 min)
python3 benchmark_cpp.py

# Step 2: Rust benchmarks (~15 min)
cd ../..
RUSTFLAGS="-C target-cpu=native" cargo run --release --example benchmark_large
cd paper/replication

# Step 3: Generate comparison tables
python3 compare_results.py
```

## Output Files

- `benchmark_cpp_results.json` — Raw C++ timing data (10 runs per config)
- `benchmark_rust_results.json` — Raw Rust timing data (10 runs per config)
- `comparison_tables.txt` — Formatted tables matching paper Tables 2-3
