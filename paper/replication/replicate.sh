#!/bin/bash
# Replication script for smelt-ml paper benchmarks.
# Reproduces Tables 2-3 (training time comparison).
#
# Usage: bash replicate.sh
# Expected runtime: ~30 minutes

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================================"
echo "  smelt-ml Paper Replication"
echo "  Project: $PROJECT_DIR"
echo "============================================================"

# Step 1: C++ benchmarks
echo ""
echo "[1/3] Running C++ library benchmarks (via Python)..."
echo "      This takes ~15 minutes."
python3 "$SCRIPT_DIR/benchmark_cpp.py"

# Step 2: Rust benchmarks
echo ""
echo "[2/3] Running Rust benchmarks..."
echo "      This takes ~15 minutes."
cd "$PROJECT_DIR"
RUSTFLAGS="-C target-cpu=native" cargo run --release --example benchmark_large

# Step 3: Comparison
echo ""
echo "[3/3] Generating comparison tables..."
python3 "$SCRIPT_DIR/compare_results.py"

echo ""
echo "============================================================"
echo "  Replication complete!"
echo "  Results: $SCRIPT_DIR/comparison_tables.txt"
echo "============================================================"
