#!/bin/bash
# Run Rust benchmarks with cooling pauses between sizes.
# This produces reliable results on thermally-constrained laptops.
#
# Usage: bash benchmark_rust_cooled.sh
# Prerequisite: cargo build --release --example benchmark_large

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$PROJECT_DIR/target/release/examples/benchmark_large"

if [ ! -f "$BIN" ]; then
    echo "Building benchmark binary..."
    cd "$PROJECT_DIR"
    RUSTFLAGS="-C target-cpu=native" cargo build --release --example benchmark_large
    echo "Cooling down after compilation (60s)..."
    sleep 60
fi

echo "Running full benchmark with cooling pauses..."
echo "CPU freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 'N/A')"

"$BIN"

echo ""
echo "CPU freq after: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo 'N/A')"
echo "Results saved to paper/replication/benchmark_rust_results.json"
