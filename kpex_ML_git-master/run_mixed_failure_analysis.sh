#!/usr/bin/env bash

set -e

ANALYSIS_TAG="${ANALYSIS_TAG:-08sep2026}"

REPORT_ROOT="restructured_pipeline/outputs/reports"

ENRICHED_FAILURES="${REPORT_ROOT}/failure_analysis_summary_scheme1_scheme3_${ANALYSIS_TAG}.xlsx"

STRAT_Y_OUTPUT="${REPORT_ROOT}/mixed_failure_case_combinations_xgb_strat_y_${ANALYSIS_TAG}.xlsx"

STRAT_YK_OUTPUT="${REPORT_ROOT}/mixed_failure_case_combinations_xgb_strat_y_K_${ANALYSIS_TAG}.xlsx"


# ------------------------------------------------------------
# Python
# ------------------------------------------------------------

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python.exe)"
else
  PYTHON_BIN="python"
fi


echo "============================================================"
echo "XGB Mixed Failure Analysis"
echo "Scheme 1 vs Scheme 3"
echo "============================================================"
echo "Python: $PYTHON_BIN"
echo "Tag:    $ANALYSIS_TAG"
echo


if [ ! -f "$ENRICHED_FAILURES" ]; then
  echo "ERROR: Enriched failure workbook not found:"
  echo "  $ENRICHED_FAILURES"
  exit 1
fi


# ============================================================
# 1. XGB strat_y
# ============================================================

echo
echo "============================================================"
echo "1. XGB strat_y"
echo "============================================================"

"$PYTHON_BIN" restructured_pipeline/src/mixed_failure_case_combinations_xgb_only.py \
  --failure-workbook "$ENRICHED_FAILURES" \
  --split-mode strat_y \
  --output-path "$STRAT_Y_OUTPUT"


# ============================================================
# 2. XGB strat_y_K
# ============================================================

echo
echo "============================================================"
echo "2. XGB strat_y_K"
echo "============================================================"

"$PYTHON_BIN" restructured_pipeline/src/mixed_failure_case_combinations_xgb_only.py \
  --failure-workbook "$ENRICHED_FAILURES" \
  --split-mode strat_y_K \
  --output-path "$STRAT_YK_OUTPUT"


# ============================================================
# Verify
# ============================================================

echo
echo "============================================================"
echo "3. OUTPUT CHECK"
echo "============================================================"

for file in \
  "$STRAT_Y_OUTPUT" \
  "$STRAT_YK_OUTPUT"
do
  if [ ! -f "$file" ]; then
    echo "ERROR: Missing output:"
    echo "  $file"
    exit 1
  fi

  echo "FOUND:"
  echo "  $file"
done


echo
echo "============================================================"
echo "Mixed failure analysis completed."
echo "============================================================"