#!/bin/bash
# Replication script for the smelt-ml JSS paper.
#
# Reproduces every table and figure of the manuscript:
#   Part A  training-time benchmarks (Tables 9-10), scaling figure (Figure 1),
#           statistical significance (Table 11)                     ~30 min
#   Part B  prediction time (Table 12) and accuracy validation (Table 13) ~5 min
#   Part C  histogram-kernel ablation (Table 7)                      ~2 min
#   Part D  spatial pipeline: King County case study (Table 2),
#           Geo-XGBoost validation against geoxgboost 1.1.1 (Table 4),
#           split-conformal intervals (Table 3)                     ~15 min
#
# Usage:  bash replicate.sh            # everything (about one hour)
#         bash replicate.sh A B        # selected parts only
#
# Requirements: Rust 1.89+, Python 3.12 with numpy, pandas, scikit-learn,
# xgboost, lightgbm, catboost, matplotlib, and geoxgboost==1.1.1 (Part D).
# Outputs are written next to this script; the files shipped with the paper
# were produced on an Intel i7-1270P (see README.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARTS="${*:-A B C D}"
export RUSTFLAGS="-C target-cpu=native"

run_part() { case " $PARTS " in *" $1 "*) return 0;; *) return 1;; esac; }

echo "============================================================"
echo "  smelt-ml paper replication   parts: $PARTS"
echo "  Project: $PROJECT_DIR"
echo "============================================================"

if run_part A; then
  echo; echo "[A1] C++ library benchmarks via Python (~15 min)"
  python3 "$SCRIPT_DIR/benchmark_cpp.py"
  echo; echo "[A2] Rust benchmarks (~15 min)"
  (cd "$PROJECT_DIR" && cargo run --release --example benchmark_large)
  echo; echo "[A3] Comparison tables (Tables 9-10) -> comparison_tables.txt"
  python3 "$SCRIPT_DIR/compare_results.py"
  echo; echo "[A4] Scaling figure (Figure 1) -> ../scaling.pdf"
  python3 "$SCRIPT_DIR/plot_scaling.py"
  echo; echo "[A5] Wilcoxon / bootstrap significance (Table 11) -> statistical_analysis.txt"
  python3 "$SCRIPT_DIR/statistical_analysis.py"
fi

if run_part B; then
  echo; echo "[B1] Prediction time, smelt-ml side (Table 12)"
  (cd "$PROJECT_DIR" && cargo run --release --example benchmark_prediction)
  echo; echo "[B2] Prediction time, scikit-learn side (Table 12)"
  python3 "$SCRIPT_DIR/prediction_sklearn.py"
  echo; echo "[B3] Accuracy validation, smelt-ml side (Table 13)"
  (cd "$PROJECT_DIR" && cargo run --release --example accuracy_validation)
  echo; echo "[B4] Accuracy validation, scikit-learn side (Table 13)"
  python3 "$SCRIPT_DIR/accuracy_sklearn.py"
fi

if run_part C; then
  echo; echo "[C1] Histogram-kernel ablation (Table 7)"
  (cd "$PROJECT_DIR" && cargo run --release --example ablation_study)
fi

if run_part D; then
  echo; echo "[D1] King County case study (Table 2)"
  (cd "$PROJECT_DIR" && cargo run --release --example case_study_king_county)
  echo; echo "[D2] Geo-XGBoost validation, smelt-ml side at alpha_geo = 0.5 (Table 4)"
  (cd "$PROJECT_DIR" && GXGB_ALPHA=0.5 cargo run --release --example validate_geoxgboost_blend)
  echo; echo "[D3] Geo-XGBoost validation, geoxgboost 1.1.1 side (Table 4) -> geoxgb_validation_blend.json"
  python3 "$SCRIPT_DIR/validate_geoxgboost_blend.py"
  echo; echo "[D4] Split-conformal intervals over the spatial predictions (Table 3)"
  python3 "$SCRIPT_DIR/conformal_intervals.py"
fi

echo
echo "============================================================"
echo "  Replication complete. Outputs in $SCRIPT_DIR"
echo "============================================================"
