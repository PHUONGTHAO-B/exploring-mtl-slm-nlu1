#!/usr/bin/env bash
# Run all 22 experiments sequentially.
# Usage: bash scripts/run_all.sh [OUTPUT_ROOT]
set -e

OUT_ROOT=${1:-outputs}
mkdir -p "$OUT_ROOT"

echo "=========================================="
echo "Running 22 experiments → $OUT_ROOT"
echo "=========================================="

python -m scripts.run_all --output_root "$OUT_ROOT"

echo ""
echo "Consolidated CSV: $OUT_ROOT/all_summary.csv"
echo "Consolidated JSON: $OUT_ROOT/all_summary.json"
