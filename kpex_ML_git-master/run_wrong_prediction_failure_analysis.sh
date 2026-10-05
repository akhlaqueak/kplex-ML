#!/usr/bin/env bash

set -e

RUN_TAG="${RUN_TAG:-08sep2026}"

REPORT_ROOT="restructured_pipeline/outputs/reports"
WRONG_CASES_ROOT="restructured_pipeline/outputs/wrong_cases"

TEST_REPORT_DIR="${REPORT_ROOT}/test_data"
ENTIRE_REPORT_DIR="${REPORT_ROOT}/entire_data"

TEST_IMPORTANT="${TEST_REPORT_DIR}/compare_wrong_predictions_tolerance_important_failures_test_data.xlsx"
ENTIRE_IMPORTANT="${ENTIRE_REPORT_DIR}/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"

ENRICHED_FAILURES="${REPORT_ROOT}/failure_analysis_summary_scheme1_scheme3_${RUN_TAG}.xlsx"

SCOPE_HIGHLIGHTED="${REPORT_ROOT}/failure_analysis_summary_scheme1_scheme3_${RUN_TAG}_scope_overlap_highlighted.xlsx"

FINAL_COUNTS="${REPORT_ROOT}/failure_train_test_split_count_summary_${RUN_TAG}.xlsx"

MIXED_WORKBOOK="${REPORT_ROOT}/mixed_failure_case_combinations_xgb_only_corrected.xlsx"


# ============================================================
# Python
# ============================================================

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python.exe)"
else
  PYTHON_BIN="python"
fi


echo "============================================================"
echo "Wrong-prediction + failure-analysis pipeline"
echo "============================================================"
echo "Python:  $PYTHON_BIN"
echo "Run tag: $RUN_TAG"
echo

"$PYTHON_BIN" --version


# ============================================================
# 1. TEST-DATA WRONG-PREDICTION PIPELINE
# ============================================================

echo
echo "============================================================"
echo "1. TEST-DATA WRONG-PREDICTION PIPELINE"
echo "   compare -> tolerance -> important failures"
echo "============================================================"

"$PYTHON_BIN" restructured_pipeline/src/compare_wrong_predictions.py \
  --run-test-data-pipeline


# ============================================================
# 2. ENTIRE-DATA WRONG-PREDICTION PIPELINE
# ============================================================

echo
echo "============================================================"
echo "2. ENTIRE-DATA WRONG-PREDICTION PIPELINE"
echo "   compare -> tolerance -> important failures"
echo "============================================================"

"$PYTHON_BIN" restructured_pipeline/src/compare_wrong_predictions.py \
  --run-entire-data-pipeline


# ============================================================
# Verify important-failure inputs
# ============================================================

echo
echo "Checking important-failure workbooks..."

if [ ! -f "$TEST_IMPORTANT" ]; then
  echo "ERROR: Test important-failure workbook not found:"
  echo "$TEST_IMPORTANT"
  exit 1
fi

if [ ! -f "$ENTIRE_IMPORTANT" ]; then
  echo "ERROR: Entire-data important-failure workbook not found:"
  echo "$ENTIRE_IMPORTANT"
  exit 1
fi

echo "Found test-data important failures:"
echo "  $TEST_IMPORTANT"

echo "Found entire-data important failures:"
echo "  $ENTIRE_IMPORTANT"


# ============================================================
# 3. FAILURE ANALYSIS ENRICHMENT
#
# Creates:
#   scheme_1_RF_strat_y
#   scheme_1_RF_strat_y_K
#   scheme_1_XGB_strat_y
#   scheme_1_XGB_strat_y_K
#   scheme_3_RF_strat_y
#   scheme_3_RF_strat_y_K
#   scheme_3_XGB_strat_y
#   scheme_3_XGB_strat_y_K
#
# plus failure_scope_summary.
# ============================================================

echo
echo "============================================================"
echo "3. FAILURE ANALYSIS ENRICHMENT"
echo "============================================================"

"$PYTHON_BIN" restructured_pipeline/src/failure_analysis_summary.py \
  --mode enrich \
  --failure-workbook "$ENTIRE_IMPORTANT" \
  --test-failure-workbook "$TEST_IMPORTANT" \
  --enriched-failure-workbook-output "$ENRICHED_FAILURES"


# ============================================================
# Verify enriched workbook
# ============================================================

if [ ! -f "$ENRICHED_FAILURES" ]; then
  echo "ERROR: Enriched failure workbook was not generated:"
  echo "$ENRICHED_FAILURES"
  exit 1
fi

echo
echo "Enriched failure workbook:"
echo "  $ENRICHED_FAILURES"


# ============================================================
# 4. UNIFIED FAILURE COUNTS + TRAIN/TEST ANALYSIS
#
# Replaces:
#   failure_counts_by_split_scheme_model.py
#   failure_train_test_split_count_summary.py
#
# Also creates:
#   - scope-overlap highlighted copy
#   - duplicate audit
#   - important failure counts
#   - all wrong prediction counts
#   - train/test partition counts
#   - scheme/model summaries
# ============================================================

echo
echo "============================================================"
echo "4. FAILURE COUNTS + TRAIN/TEST + DUPLICATE/SCOPE ANALYSIS"
echo "============================================================"

if [ -f "$MIXED_WORKBOOK" ]; then

  echo "Using mixed-combination workbook:"
  echo "  $MIXED_WORKBOOK"

  "$PYTHON_BIN" restructured_pipeline/src/failure_analysis_counts_and_train_test_summary.py \
    --important-workbook "$ENRICHED_FAILURES" \
    --highlighted-input-copy "$SCOPE_HIGHLIGHTED" \
    --mixed-workbook "$MIXED_WORKBOOK" \
    --entire-raw-glob "${WRONG_CASES_ROOT}/*/*/wrong_predictions_*${RUN_TAG}*_predict_inference_entire_data.xlsx" \
    --test-raw-glob "${WRONG_CASES_ROOT}/*/*/wrong_predictions_*${RUN_TAG}*_predict_inference_test_data.xlsx" \
    --output-path "$FINAL_COUNTS"

else

  echo "WARNING: Current mixed-combination workbook not found."
  echo "Mixed analysis will be skipped."
  echo "Expected:"
  echo "  $MIXED_WORKBOOK"

  "$PYTHON_BIN" restructured_pipeline/src/failure_analysis_counts_and_train_test_summary.py \
    --important-workbook "$ENRICHED_FAILURES" \
    --highlighted-input-copy "$SCOPE_HIGHLIGHTED" \
    --entire-raw-glob "${WRONG_CASES_ROOT}/*/*/wrong_predictions_*${RUN_TAG}*_predict_inference_entire_data.xlsx" \
    --test-raw-glob "${WRONG_CASES_ROOT}/*/*/wrong_predictions_*${RUN_TAG}*_predict_inference_test_data.xlsx" \
    --output-path "$FINAL_COUNTS" \
    --skip-mixed

fi


# ============================================================
# 5. FINAL OUTPUT CHECK
# ============================================================

echo
echo "============================================================"
echo "5. OUTPUT CHECK"
echo "============================================================"

if [ ! -f "$SCOPE_HIGHLIGHTED" ]; then
  echo "ERROR: Scope-overlap workbook missing:"
  echo "$SCOPE_HIGHLIGHTED"
  exit 1
fi

if [ ! -f "$FINAL_COUNTS" ]; then
  echo "ERROR: Final failure-count workbook missing:"
  echo "$FINAL_COUNTS"
  exit 1
fi

echo
echo "Generated:"
echo
echo "Enriched failure analysis:"
echo "  $ENRICHED_FAILURES"
echo
echo "Scope-overlap highlighted copy:"
echo "  $SCOPE_HIGHLIGHTED"
echo
echo "Final train/test failure summary:"
echo "  $FINAL_COUNTS"


echo
echo "============================================================"
echo "Wrong-prediction and failure-case pipeline completed."
echo "============================================================"