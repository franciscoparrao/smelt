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
