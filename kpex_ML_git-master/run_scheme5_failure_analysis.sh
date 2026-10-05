#!/usr/bin/env bash
set -euo pipefail

TRAIN_TAG="${TRAIN_TAG:-17sep2026}"
PREDICTION_TAG="${PREDICTION_TAG:-17sep2026}"
OUTPUT_TAG="${OUTPUT_TAG:-${TRAIN_TAG}_${PREDICTION_TAG}}"

# Prefer the Windows Anaconda environment used by this project.
# You can override it explicitly with PYTHON_BIN=/path/to/python.
if [ -n "${PYTHON_BIN:-}" ]; then
  :
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif [ -x "/mnt/c/Users/n1a9m/Anaconda3_/python.exe" ]; then
  PYTHON_BIN="/mnt/c/Users/n1a9m/Anaconda3_/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python.exe)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "ERROR: No usable Python interpreter found."
  exit 1
fi

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

if ! "$PYTHON_BIN" -c 'import pandas, openpyxl, yaml; print("Python dependencies: OK")'; then
  echo "ERROR: Selected Python lacks project dependencies: $PYTHON_BIN"
  echo "Try:"
  echo 'PYTHON_BIN=/mnt/c/Users/n1a9m/Anaconda3_/python.exe bash run_scheme5_failure_analysis.sh'
  exit 1
fi

SRC="restructured_pipeline/src"
REPORT="restructured_pipeline/outputs/reports"
WRONG="restructured_pipeline/outputs/wrong_cases/scheme5"
PRED="restructured_pipeline/outputs/predictions/scheme5"

for FAMILY in b other; do
  echo "=== Scheme 5 ${FAMILY}: compare -> tolerance -> important failures ==="

  "$PYTHON_BIN" "$SRC/compare_wrong_predictions.py" \
    --run-test-data-pipeline \
    --scheme5-family "$FAMILY" \
    --model-family xgboost \
    --analysis-tag "$TRAIN_TAG" \
    --prediction-tag "$PREDICTION_TAG" \
    --wrong-cases-root "$WRONG"

  "$PYTHON_BIN" "$SRC/compare_wrong_predictions.py" \
    --run-entire-data-pipeline \
    --scheme5-family "$FAMILY" \
    --model-family xgboost \
    --analysis-tag "$TRAIN_TAG" \
    --prediction-tag "$PREDICTION_TAG" \
    --wrong-cases-root "$WRONG"

  TEST_IMPORTANT="$REPORT/test_data/scheme5/$FAMILY/compare_wrong_predictions_tolerance_important_failures_test_data.xlsx"
  ENTIRE_IMPORTANT="$REPORT/entire_data/scheme5/$FAMILY/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"
  ENRICHED="$REPORT/failure_analysis_scheme5_${FAMILY}_${OUTPUT_TAG}.xlsx"

  test -f "$TEST_IMPORTANT"
  test -f "$ENTIRE_IMPORTANT"

  "$PYTHON_BIN" "$SRC/failure_analysis_scheme4.py" \
    --scheme scheme5 \
    --family "$FAMILY" \
    --entire-important-workbook "$ENTIRE_IMPORTANT" \
    --test-important-workbook "$TEST_IMPORTANT" \
    --output-path "$ENRICHED"
done

BBOOK="$REPORT/failure_analysis_scheme5_b_${OUTPUT_TAG}.xlsx"
OBOOK="$REPORT/failure_analysis_scheme5_other_${OUTPUT_TAG}.xlsx"
test -f "$BBOOK"
test -f "$OBOOK"

for SPLIT in strat_y strat_y_K; do
  BPRED="$PRED/b/xgboost/all_predictions_scheme5_b_xgboost_xgb_${SPLIT}_${TRAIN_TAG}_${PREDICTION_TAG}_predict_inference_entire_data.csv"
  OPRED="$PRED/other/xgboost/all_predictions_scheme5_other_xgboost_xgb_${SPLIT}_${TRAIN_TAG}_${PREDICTION_TAG}_predict_inference_entire_data.csv"
  OUT="$REPORT/scheme5_mixed_topk_${SPLIT}_${OUTPUT_TAG}.xlsx"

  test -f "$BPRED"
  test -f "$OPRED"

  "$PYTHON_BIN" "$SRC/scheme4_mixed_topk.py" \
    --scheme scheme5 \
    --b-workbook "$BBOOK" \
    --other-workbook "$OBOOK" \
    --b-predictions "$BPRED" \
    --other-predictions "$OPRED" \
    --split-mode "$SPLIT" \
    --output-path "$OUT"

  test -f "$OUT"
done

echo "=== Scheme 5 failure analysis complete ==="
echo "B:       $BBOOK"
echo "Other:   $OBOOK"
echo "Mixed y:  $REPORT/scheme5_mixed_topk_strat_y_${OUTPUT_TAG}.xlsx"
echo "Mixed yK: $REPORT/scheme5_mixed_topk_strat_y_K_${OUTPUT_TAG}.xlsx"
