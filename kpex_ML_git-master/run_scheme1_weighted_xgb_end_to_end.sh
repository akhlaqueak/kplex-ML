#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${RUN_TAG:-23sep2026}"
ANALYSIS_TAG="${ANALYSIS_TAG:-$RUN_TAG}"
[ "$RUN_TAG" = "$ANALYSIS_TAG" ] || { echo "ERROR: RUN_TAG and ANALYSIS_TAG must match."; exit 1; }

if [ -n "${PYTHON_BIN:-}" ]; then :
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif [ -x "/mnt/c/Users/n1a9m/Anaconda3_/python.exe" ]; then PYTHON_BIN="/mnt/c/Users/n1a9m/Anaconda3_/python.exe"
elif command -v python.exe >/dev/null 2>&1; then PYTHON_BIN="$(command -v python.exe)"
elif command -v python >/dev/null 2>&1; then PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3)"
else echo "ERROR: No Python found."; exit 1; fi

SRC="restructured_pipeline/src"
OUT="restructured_pipeline/outputs"
CHECKPOINT_ROOT="$OUT/checkpoints/scheme1_weighting/$RUN_TAG"
mkdir -p "$CHECKPOINT_ROOT"

WEIGHTS=(
  "2min_x_log_2min_ratio"
  "3min_x_log_3min_ratio"
  "4min_x_log_4min_ratio"
  "5min_x_log_5min_ratio"
  "6min_x_log_6min_ratio"
)
SPLITS=("strat_y" "strat_y_K")

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c 'import pandas, openpyxl, yaml, sklearn, xgboost; print("Dependencies: OK")'

for WEIGHT in "${WEIGHTS[@]}"; do
  echo "============================================================"
  echo "Weight experiment: $WEIGHT"
  echo "============================================================"

  PRED_DIR="$OUT/predictions/scheme1/xgboost/$WEIGHT"
  WRONG_DIR="$OUT/wrong_cases/scheme1/xgboost/$WEIGHT"
  MODEL_DIR="$OUT/models/scheme1/xgboost/$WEIGHT"
  REPORT_DIR="$OUT/reports/scheme1/xgboost/$WEIGHT"
  REPORT_SUBDIR="scheme1/xgboost/$WEIGHT"
  mkdir -p "$REPORT_DIR"

  for SPLIT in "${SPLITS[@]}"; do
    VARIANT="xgb_${SPLIT}"
    RUN_NAME="scheme1_xgboost_${VARIANT}_weight_${WEIGHT}_${RUN_TAG}"
    TEST_NAME="${RUN_NAME}_${ANALYSIS_TAG}_predict_inference_test_data"
    ENTIRE_NAME="${RUN_NAME}_${ANALYSIS_TAG}_predict_inference_entire_data"

    MODEL="$MODEL_DIR/model_${RUN_NAME}.pkl"
    TEST_PRED="$PRED_DIR/all_predictions_${TEST_NAME}.csv"
    ENTIRE_PRED="$PRED_DIR/all_predictions_${ENTIRE_NAME}.csv"
    TEST_WRONG="$WRONG_DIR/wrong_predictions_${TEST_NAME}.xlsx"
    ENTIRE_WRONG="$WRONG_DIR/wrong_predictions_${ENTIRE_NAME}.xlsx"

    echo "--- $WEIGHT / $SPLIT ---"

    if [ ! -f "$MODEL" ]; then
      "$PYTHON_BIN" "$SRC/train.py" \
        --scheme scheme1 --model-family xgboost --variant "$VARIANT" \
        --weight-scheme "$WEIGHT" --file-suffix "$RUN_NAME" --save-files
    else echo "[SKIP train]"; fi

    if [ ! -f "$TEST_PRED" ] || [ ! -f "$TEST_WRONG" ]; then
      "$PYTHON_BIN" "$SRC/predict.py" \
        --scheme scheme1 --model-family xgboost --variant "$VARIANT" \
        --experiment-subdir "$WEIGHT" --run-name "$RUN_NAME" \
        --inference-mode inference_test_set --file-suffix "$TEST_NAME"
    else echo "[SKIP predict test]"; fi

    if [ ! -f "$ENTIRE_PRED" ] || [ ! -f "$ENTIRE_WRONG" ]; then
      "$PYTHON_BIN" "$SRC/predict.py" \
        --scheme scheme1 --model-family xgboost --variant "$VARIANT" \
        --experiment-subdir "$WEIGHT" --run-name "$RUN_NAME" \
        --inference-mode inference_entire_data --file-suffix "$ENTIRE_NAME"
    else echo "[SKIP predict entire]"; fi

    EVAL_MARK="$CHECKPOINT_ROOT/${WEIGHT}_${SPLIT}_evaluate.done"
    if [ ! -f "$EVAL_MARK" ]; then
      "$PYTHON_BIN" "$SRC/evaluate.py" \
        --scheme scheme1 --model-family xgboost --variant "$VARIANT" \
        --experiment-subdir "$WEIGHT" --run-name "$RUN_NAME" \
        --predictions-path "$TEST_PRED" --inference-mode inference_test_set \
        --file-suffix "${RUN_NAME}_${ANALYSIS_TAG}_eval_inference_test_data"
      "$PYTHON_BIN" "$SRC/evaluate.py" \
        --scheme scheme1 --model-family xgboost --variant "$VARIANT" \
        --experiment-subdir "$WEIGHT" --run-name "$RUN_NAME" \
        --predictions-path "$ENTIRE_PRED" --inference-mode inference_entire_data \
        --file-suffix "${RUN_NAME}_${ANALYSIS_TAG}_eval_inference_entire_data"
      touch "$EVAL_MARK"
    else echo "[SKIP evaluate]"; fi

    TEST_IMPORTANT="$OUT/reports/test_data/$REPORT_SUBDIR/${SPLIT}_compare_wrong_predictions_tolerance_important_failures_test_data.xlsx"
    ENTIRE_IMPORTANT="$OUT/reports/entire_data/$REPORT_SUBDIR/${SPLIT}_compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"
    EXP_TOKEN="${VARIANT}_weight_${WEIGHT}_${RUN_TAG}"

    if [ ! -f "$TEST_IMPORTANT" ]; then
      "$PYTHON_BIN" "$SRC/compare_wrong_predictions.py" \
        --run-test-data-pipeline --model-family xgboost \
        --analysis-tag "$RUN_TAG" --prediction-tag "$ANALYSIS_TAG" \
        --experiment-token "$EXP_TOKEN" --report-subdir "$REPORT_SUBDIR" \
        --report-tag "$SPLIT" --wrong-cases-root "$WRONG_DIR"
    else echo "[SKIP important test]"; fi

    if [ ! -f "$ENTIRE_IMPORTANT" ]; then
      "$PYTHON_BIN" "$SRC/compare_wrong_predictions.py" \
        --run-entire-data-pipeline --model-family xgboost \
        --analysis-tag "$RUN_TAG" --prediction-tag "$ANALYSIS_TAG" \
        --experiment-token "$EXP_TOKEN" --report-subdir "$REPORT_SUBDIR" \
        --report-tag "$SPLIT" --wrong-cases-root "$WRONG_DIR"
    else echo "[SKIP important entire]"; fi
  done

  # Both split strategies are now combined into ONE tolerance workbook.
  Y_RUN="scheme1_xgboost_xgb_strat_y_weight_${WEIGHT}_${RUN_TAG}"
  YK_RUN="scheme1_xgboost_xgb_strat_y_K_weight_${WEIGHT}_${RUN_TAG}"
  Y_TEST="$PRED_DIR/all_predictions_${Y_RUN}_${ANALYSIS_TAG}_predict_inference_test_data.csv"
  Y_ENTIRE="$PRED_DIR/all_predictions_${Y_RUN}_${ANALYSIS_TAG}_predict_inference_entire_data.csv"
  YK_TEST="$PRED_DIR/all_predictions_${YK_RUN}_${ANALYSIS_TAG}_predict_inference_test_data.csv"
  YK_ENTIRE="$PRED_DIR/all_predictions_${YK_RUN}_${ANALYSIS_TAG}_predict_inference_entire_data.csv"

  TOLERANCE="$REPORT_DIR/tolerance_metrics_${RUN_TAG}.xlsx"
  if [ ! -f "$TOLERANCE" ]; then
    "$PYTHON_BIN" "$SRC/scheme1_weighted_tolerance.py" \
      --strat-y-test-predictions "$Y_TEST" --strat-y-entire-predictions "$Y_ENTIRE" \
      --strat-y-k-test-predictions "$YK_TEST" --strat-y-k-entire-predictions "$YK_ENTIRE" \
      --weight-scheme "$WEIGHT" --output-path "$TOLERANCE"
  else echo "[SKIP combined tolerance]"; fi

  Y_TEST_WRONG="$WRONG_DIR/wrong_predictions_${Y_RUN}_${ANALYSIS_TAG}_predict_inference_test_data.xlsx"
  Y_ENTIRE_WRONG="$WRONG_DIR/wrong_predictions_${Y_RUN}_${ANALYSIS_TAG}_predict_inference_entire_data.xlsx"
  YK_TEST_WRONG="$WRONG_DIR/wrong_predictions_${YK_RUN}_${ANALYSIS_TAG}_predict_inference_test_data.xlsx"
  YK_ENTIRE_WRONG="$WRONG_DIR/wrong_predictions_${YK_RUN}_${ANALYSIS_TAG}_predict_inference_entire_data.xlsx"

  Y_TEST_IMP="$OUT/reports/test_data/$REPORT_SUBDIR/strat_y_compare_wrong_predictions_tolerance_important_failures_test_data.xlsx"
  Y_ENTIRE_IMP="$OUT/reports/entire_data/$REPORT_SUBDIR/strat_y_compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"
  YK_TEST_IMP="$OUT/reports/test_data/$REPORT_SUBDIR/strat_y_K_compare_wrong_predictions_tolerance_important_failures_test_data.xlsx"
  YK_ENTIRE_IMP="$OUT/reports/entire_data/$REPORT_SUBDIR/strat_y_K_compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"

  FAILURE="$REPORT_DIR/failure_analysis_${RUN_TAG}.xlsx"
  if [ ! -f "$FAILURE" ]; then
    "$PYTHON_BIN" "$SRC/scheme1_weighted_failure_analysis.py" \
      --strat-y-test-important-workbook "$Y_TEST_IMP" \
      --strat-y-entire-important-workbook "$Y_ENTIRE_IMP" \
      --strat-y-k-test-important-workbook "$YK_TEST_IMP" \
      --strat-y-k-entire-important-workbook "$YK_ENTIRE_IMP" \
      --strat-y-test-predictions "$Y_TEST" --strat-y-entire-predictions "$Y_ENTIRE" \
      --strat-y-k-test-predictions "$YK_TEST" --strat-y-k-entire-predictions "$YK_ENTIRE" \
      --strat-y-test-wrong "$Y_TEST_WRONG" --strat-y-entire-wrong "$Y_ENTIRE_WRONG" \
      --strat-y-k-test-wrong "$YK_TEST_WRONG" --strat-y-k-entire-wrong "$YK_ENTIRE_WRONG" \
      --weight-scheme "$WEIGHT" --output-path "$FAILURE"
  else echo "[SKIP combined failure analysis]"; fi

done

echo "All weighted Scheme 1 XGB experiments complete."
