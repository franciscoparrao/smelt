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
#         PYTHON=.venv/bin/python bash replicate.sh   # choose the interpreter
#
# This is the single entry point. README.md ("Map from manuscript to
# scripts") lists, for every table and figure, the step below that produces
# it and the output file it writes.
#
# Requirements: Rust 1.89+ (rustup), git, Python 3.12 with the packages in
# requirements.txt (pip install -r requirements.txt). Works both inside a
# checkout of the smelt repository and on its own: if the crate is not found
# two levels up, the tagged source is cloned into ./smelt-src.
# Outputs are written next to this script; the files shipped with the paper
# were produced on an Intel i7-1270P (see README.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARTS="${*:-A B C D}"
export RUSTFLAGS="-C target-cpu=native"
PYTHON="${PYTHON:-python3}"

# The Rust examples live in the smelt-ml source tree (examples/ and data/).
# When this directory is run inside a checkout of the repository the tree is
# two levels up; when it is run on its own (e.g. the replication archive
# extracted into an empty folder) the tagged source is cloned next to it.
SMELT_REPO="${SMELT_REPO:-https://github.com/franciscoparrao/smelt}"
SMELT_REF="${SMELT_REF:-paper-jss-submission}"
if [ -f "$SCRIPT_DIR/../../Cargo.toml" ] && grep -q '^name = "smelt-ml"' "$SCRIPT_DIR/../../Cargo.toml"; then
  PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
  PROJECT_DIR="$SCRIPT_DIR/smelt-src"
  if [ ! -f "$PROJECT_DIR/Cargo.toml" ]; then
    echo "smelt-ml source not found two levels up; cloning $SMELT_REPO at $SMELT_REF into $PROJECT_DIR"
    if ! git clone --quiet --depth 1 --branch "$SMELT_REF" "$SMELT_REPO" "$PROJECT_DIR" 2>/dev/null; then
      echo "ref $SMELT_REF not found on the remote; cloning the default branch instead (set SMELT_REF to pin a release)"
      rm -rf "$PROJECT_DIR"
      git clone --quiet --depth 1 "$SMELT_REPO" "$PROJECT_DIR"
    fi
  fi
fi

for tool in cargo "$PYTHON"; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing: $tool (see README.md, Requirements)"; exit 1; }
done
"$PYTHON" - <<'PYCHECK' || { echo "Python dependencies missing: pip install -r $SCRIPT_DIR/requirements.txt"; exit 1; }
import numpy, pandas, sklearn, xgboost, lightgbm, catboost, matplotlib  # noqa: F401
PYCHECK

run_part() { case " $PARTS " in *" $1 "*) return 0;; *) return 1;; esac; }

echo "============================================================"
echo "  smelt-ml paper replication   parts: $PARTS"
echo "  Project: $PROJECT_DIR"
echo "============================================================"

if run_part A; then
  echo; echo "[A1] C++ library benchmarks via Python (~15 min)"
  "$PYTHON" "$SCRIPT_DIR/benchmark_cpp.py"
  echo; echo "[A2] Rust benchmarks (~15 min)"
  (cd "$PROJECT_DIR" && cargo run --release --example benchmark_large)
  echo; echo "[A3] Comparison tables (Tables 9-10) -> comparison_tables.txt"
  "$PYTHON" "$SCRIPT_DIR/compare_results.py"
  echo; echo "[A4] Scaling figure (Figure 1) -> ../scaling.pdf"
  "$PYTHON" "$SCRIPT_DIR/plot_scaling.py"
  echo; echo "[A5] Wilcoxon / bootstrap significance (Table 11) -> statistical_analysis.txt"
  "$PYTHON" "$SCRIPT_DIR/statistical_analysis.py"
fi

if run_part B; then
  echo; echo "[B1] Prediction time, smelt-ml side (Table 12)"
  (cd "$PROJECT_DIR" && cargo run --release --example benchmark_prediction)
  echo; echo "[B2] Prediction time, scikit-learn side (Table 12)"
  "$PYTHON" "$SCRIPT_DIR/prediction_sklearn.py"
  echo; echo "[B3] Accuracy validation, smelt-ml side (Table 13)"
  (cd "$PROJECT_DIR" && cargo run --release --example accuracy_validation)
  echo; echo "[B4] Accuracy validation, scikit-learn side (Table 13)"
  "$PYTHON" "$SCRIPT_DIR/accuracy_sklearn.py"
fi

if run_part C; then
  echo; echo "[C1] Histogram-kernel ablation (Table 7)"
  (cd "$PROJECT_DIR" && cargo run --release --example ablation_study)
fi

if run_part D; then
  echo; echo "[D1] King County case study walkthrough (Table 2, listing in the case-study section) -> case_study_walkthrough_output.txt"
  (cd "$PROJECT_DIR" && cargo run --release --example case_study_walkthrough) | tee "$SCRIPT_DIR/case_study_walkthrough_output.txt"
  echo; echo "[D1b] King County case study, extended comparison (RF, XGBoost with coordinates)"
  (cd "$PROJECT_DIR" && cargo run --release --example case_study_king_county)
  echo; echo "[D2] Geo-XGBoost validation, smelt-ml side at alpha_geo = 0.5 (Table 4)"
  (cd "$PROJECT_DIR" && GXGB_ALPHA=0.5 cargo run --release --example validate_geoxgboost_blend)
  echo; echo "[D3] Geo-XGBoost validation, geoxgboost 1.1.1 side (Table 4) -> geoxgb_validation_blend.json"
  "$PYTHON" "$SCRIPT_DIR/validate_geoxgboost_blend.py"
  echo; echo "[D4] Split-conformal intervals over the spatial predictions (Table 3)"
  "$PYTHON" "$SCRIPT_DIR/conformal_intervals.py"
fi

echo
echo "============================================================"
echo "  Replication complete. Outputs in $SCRIPT_DIR"
echo "============================================================"
