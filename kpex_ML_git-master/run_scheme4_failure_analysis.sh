# #!/usr/bin/env bash
# set -euo pipefail

# # TRAIN_TAG identifies the trained Scheme 4 run in the filename.
# # PREDICTION_TAG identifies the prediction generation to analyze.
# TRAIN_TAG="${TRAIN_TAG:-14sep2026}"
# PREDICTION_TAG="${PREDICTION_TAG:-15sep2026}"
# OUTPUT_TAG="${OUTPUT_TAG:-${TRAIN_TAG}_${PREDICTION_TAG}}"

# PYTHON_BIN="python"
# if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
#   PYTHON_BIN="${CONDA_PREFIX}/python.exe"
# elif command -v python.exe >/dev/null 2>&1; then
#   PYTHON_BIN="$(command -v python.exe)"
# fi

# SRC="restructured_pipeline/src"
# REPORT="restructured_pipeline/outputs/reports"
# WRONG="restructured_pipeline/outputs/wrong_cases/scheme4"

# echo "Python:         $PYTHON_BIN"
# echo "Train tag:      $TRAIN_TAG"
# echo "Prediction tag: $PREDICTION_TAG"
# echo "Output tag:     $OUTPUT_TAG"

# for FAMILY in oriented forward; do
#   echo "=== Scheme 4 ${FAMILY}: compare -> tolerance -> important failures ==="

#   "$PYTHON_BIN" "$SRC/compare_wrong_predictions.py" \
#     --run-test-data-pipeline \
#     --scheme4-family "$FAMILY" \
#     --analysis-tag "$TRAIN_TAG" \
#     --prediction-tag "$PREDICTION_TAG" \
#     --wrong-cases-root "$WRONG"

#   "$PYTHON_BIN" "$SRC/compare_wrong_predictions.py" \
#     --run-entire-data-pipeline \
#     --scheme4-family "$FAMILY" \
#     --analysis-tag "$TRAIN_TAG" \
#     --prediction-tag "$PREDICTION_TAG" \
#     --wrong-cases-root "$WRONG"

#   TEST_IMPORTANT="$REPORT/test_data/scheme4/$FAMILY/compare_wrong_predictions_tolerance_important_failures_test_data.xlsx"
#   ENTIRE_IMPORTANT="$REPORT/entire_data/scheme4/$FAMILY/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"
#   ENRICHED="$REPORT/failure_analysis_scheme4_${FAMILY}_${OUTPUT_TAG}.xlsx"

#   test -f "$TEST_IMPORTANT"
#   test -f "$ENTIRE_IMPORTANT"

#   echo "=== Scheme 4 ${FAMILY}: enrich Top-K failures ==="
#   "$PYTHON_BIN" "$SRC/failure_analysis_scheme4.py" \
#     --family "$FAMILY" \
#     --entire-important-workbook "$ENTIRE_IMPORTANT" \
#     --test-important-workbook "$TEST_IMPORTANT" \
#     --output-path "$ENRICHED"
# done

# ORIENTED="$REPORT/failure_analysis_scheme4_oriented_${OUTPUT_TAG}.xlsx"
# FORWARD="$REPORT/failure_analysis_scheme4_forward_${OUTPUT_TAG}.xlsx"

# test -f "$ORIENTED"
# test -f "$FORWARD"

# for SPLIT in strat_y strat_y_K; do
#   OUT="$REPORT/scheme4_mixed_topk_${SPLIT}_${OUTPUT_TAG}.xlsx"

#   echo "=== Scheme 4 mixed Top-K: ${SPLIT} ==="
#   "$PYTHON_BIN" "$SRC/scheme4_mixed_topk.py" \
#     --oriented-workbook "$ORIENTED" \
#     --forward-workbook "$FORWARD" \
#     --split-mode "$SPLIT" \
#     --output-path "$OUT"

#   test -f "$OUT"
# done

# echo "=== Scheme 4 failure analysis complete ==="
# echo "Oriented: $ORIENTED"
# echo "Forward:  $FORWARD"
# echo "Mixed y:  $REPORT/scheme4_mixed_topk_strat_y_${OUTPUT_TAG}.xlsx"
# echo "Mixed yK: $REPORT/scheme4_mixed_topk_strat_y_K_${OUTPUT_TAG}.xlsx"

#!/usr/bin/env bash
set -euo pipefail

TRAIN_TAG="${TRAIN_TAG:-14sep2026}"
PREDICTION_TAG="${PREDICTION_TAG:-15sep2026}"
OUTPUT_TAG="${OUTPUT_TAG:-${TRAIN_TAG}_${PREDICTION_TAG}}"

PYTHON_BIN="python"

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python.exe)"
fi

SRC="restructured_pipeline/src"
REPORT="restructured_pipeline/outputs/reports"
PRED="restructured_pipeline/outputs/predictions/scheme4"

echo "============================================================"
echo "Scheme 4 failure analysis only"
echo "============================================================"
echo "Python:         $PYTHON_BIN"
echo "Train tag:      $TRAIN_TAG"
echo "Prediction tag: $PREDICTION_TAG"
echo "Output tag:     $OUTPUT_TAG"
echo


# ============================================================
# 1. FAMILY-SPECIFIC FAILURE ANALYSIS
# ============================================================

for FAMILY in oriented forward; do

  TEST_IMPORTANT="$REPORT/test_data/scheme4/$FAMILY/compare_wrong_predictions_tolerance_important_failures_test_data.xlsx"

  ENTIRE_IMPORTANT="$REPORT/entire_data/scheme4/$FAMILY/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"

  ENRICHED="$REPORT/failure_analysis_scheme4_${FAMILY}_${OUTPUT_TAG}.xlsx"

  echo "============================================================"
  echo "Scheme 4 ${FAMILY}: failure enrichment"
  echo "============================================================"

  if [ ! -f "$TEST_IMPORTANT" ]; then
    echo "ERROR: Missing test important-failure workbook:"
    echo "  $TEST_IMPORTANT"
    exit 1
  fi

  if [ ! -f "$ENTIRE_IMPORTANT" ]; then
    echo "ERROR: Missing entire-data important-failure workbook:"
    echo "  $ENTIRE_IMPORTANT"
    exit 1
  fi

  "$PYTHON_BIN" "$SRC/failure_analysis_scheme4.py" \
    --family "$FAMILY" \
    --entire-important-workbook "$ENTIRE_IMPORTANT" \
    --test-important-workbook "$TEST_IMPORTANT" \
    --output-path "$ENRICHED"

  if [ ! -f "$ENRICHED" ]; then
    echo "ERROR: Failure-analysis workbook was not generated:"
    echo "  $ENRICHED"
    exit 1
  fi

  echo "Generated:"
  echo "  $ENRICHED"
  echo
done


# ============================================================
# 2. ENRICHED FAMILY WORKBOOKS
# ============================================================

ORIENTED="$REPORT/failure_analysis_scheme4_oriented_${OUTPUT_TAG}.xlsx"
FORWARD="$REPORT/failure_analysis_scheme4_forward_${OUTPUT_TAG}.xlsx"

test -f "$ORIENTED"
test -f "$FORWARD"


# ============================================================
# 3. MIXED TOP-K ANALYSIS
#
# Uses:
#   important failure UNION:
#       Oriented failures U Forward failures
#
# and retrieves the other family's ranking from the complete
# entire-data prediction CSV.
# ============================================================

for SPLIT in strat_y strat_y_K; do

  ORIENTED_PRED="$PRED/oriented/xgboost/all_predictions_scheme4_oriented_xgboost_xgb_${SPLIT}_${TRAIN_TAG}_${PREDICTION_TAG}_predict_inference_entire_data.csv"

  FORWARD_PRED="$PRED/forward/xgboost/all_predictions_scheme4_forward_xgboost_xgb_${SPLIT}_${TRAIN_TAG}_${PREDICTION_TAG}_predict_inference_entire_data.csv"

  OUT="$REPORT/scheme4_mixed_topk_${SPLIT}_${OUTPUT_TAG}.xlsx"

  echo "============================================================"
  echo "Scheme 4 mixed Top-K: $SPLIT"
  echo "============================================================"

  if [ ! -f "$ORIENTED_PRED" ]; then
    echo "ERROR: Missing Oriented prediction CSV:"
    echo "  $ORIENTED_PRED"
    exit 1
  fi

  if [ ! -f "$FORWARD_PRED" ]; then
    echo "ERROR: Missing Forward prediction CSV:"
    echo "  $FORWARD_PRED"
    exit 1
  fi

  echo "Oriented predictions:"
  echo "  $ORIENTED_PRED"

  echo "Forward predictions:"
  echo "  $FORWARD_PRED"

  "$PYTHON_BIN" "$SRC/scheme4_mixed_topk.py" \
    --oriented-workbook "$ORIENTED" \
    --forward-workbook "$FORWARD" \
    --oriented-predictions "$ORIENTED_PRED" \
    --forward-predictions "$FORWARD_PRED" \
    --split-mode "$SPLIT" \
    --output-path "$OUT"

  if [ ! -f "$OUT" ]; then
    echo "ERROR: Mixed Top-K workbook was not generated:"
    echo "  $OUT"
    exit 1
  fi

  echo "Generated:"
  echo "  $OUT"
  echo
done


# ============================================================
# COMPLETE
# ============================================================

echo "============================================================"
echo "Scheme 4 failure analysis complete"
echo "============================================================"

echo "Oriented:"
echo "  $ORIENTED"

echo "Forward:"
echo "  $FORWARD"

echo "Mixed strat_y:"
echo "  $REPORT/scheme4_mixed_topk_strat_y_${OUTPUT_TAG}.xlsx"

echo "Mixed strat_y_K:"
echo "  $REPORT/scheme4_mixed_topk_strat_y_K_${OUTPUT_TAG}.xlsx"