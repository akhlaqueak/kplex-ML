#!/usr/bin/env bash
# Clean downstream regeneration after the Scheme 1 split correction.
#
# Model provenance:
#   Scheme 1 models: 08sep2026 (corrected split)
#   Scheme 3 models: 03sep2026 (existing valid models)
#
# New downstream analysis generation:
#   ANALYSIS_TAG=09sep2026
#
# This script does NOT retrain either scheme.
# It regenerates evaluate/predict outputs, tolerance reports, wrong/failure
# reports, enriched failure analysis, mixed XGB workbooks, and final counts.

set -e

S1_MODEL_TAG="${S1_MODEL_TAG:-08sep2026}"
S3_MODEL_TAG="${S3_MODEL_TAG:-03sep2026}"
ANALYSIS_TAG="${ANALYSIS_TAG:-09sep2026}"

PROJECT_CONFIG="restructured_pipeline/configs/project.yaml"
SCHEME1_CONFIG="restructured_pipeline/configs/models_scheme1.yaml"
SCHEME3_CONFIG="restructured_pipeline/configs/models_scheme3.yaml"

REPORT_ROOT="restructured_pipeline/outputs/reports"
WRONG_ROOT="restructured_pipeline/outputs/wrong_cases"

TEST_IMPORTANT="${REPORT_ROOT}/test_data/compare_wrong_predictions_tolerance_important_failures_test_data.xlsx"
ENTIRE_IMPORTANT="${REPORT_ROOT}/entire_data/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"
ENRICHED="${REPORT_ROOT}/failure_analysis_summary_scheme1_scheme3_${ANALYSIS_TAG}.xlsx"
MIXED_Y="${REPORT_ROOT}/mixed_failure_case_combinations_xgb_strat_y_${ANALYSIS_TAG}.xlsx"
MIXED_YK="${REPORT_ROOT}/mixed_failure_case_combinations_xgb_strat_y_K_${ANALYSIS_TAG}.xlsx"
COUNTS="${REPORT_ROOT}/failure_train_test_split_count_summary_${ANALYSIS_TAG}.xlsx"
HIGHLIGHTED="${REPORT_ROOT}/failure_analysis_summary_scheme1_scheme3_${ANALYSIS_TAG}_scope_overlap_highlighted.xlsx"

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python.exe)"
else
  PYTHON_BIN="python"
fi

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Scheme 1 model tag: $S1_MODEL_TAG"
echo "Scheme 3 model tag: $S3_MODEL_TAG"
echo "Analysis tag:       $ANALYSIS_TAG"

# eval_s1() {
#   family="$1"; variant="$2"
#   run="scheme1_${family}_${variant}_${S1_MODEL_TAG}"

#   "$PYTHON_BIN" restructured_pipeline/src/evaluate.py \
#     --scheme scheme1 --model-family "$family" --variant "$variant" \
#     --run-name "$run" --inference-mode inference_test_set \
#     --file-suffix "${run}_${ANALYSIS_TAG}_evaluate_test_data" \
#     --project-config "$PROJECT_CONFIG" --model-config "$SCHEME1_CONFIG"

#   "$PYTHON_BIN" restructured_pipeline/src/evaluate.py \
#     --scheme scheme1 --model-family "$family" --variant "$variant" \
#     --run-name "$run" --inference-mode inference_entire_data \
#     --file-suffix "${run}_${ANALYSIS_TAG}_evaluate_entire_data" \
#     --project-config "$PROJECT_CONFIG" --model-config "$SCHEME1_CONFIG"
# }

# eval_s3() {
#   family="$1"; split="$2"
#   run="scheme3_${family}_${split}_${S3_MODEL_TAG}"

#   "$PYTHON_BIN" restructured_pipeline/src/evaluate.py \
#     --scheme scheme3 --model-family "$family" --split-mode "$split" \
#     --run-name "$run" --inference-mode inference_test_set \
#     --file-suffix "${run}_${ANALYSIS_TAG}_evaluate_test_data" \
#     --project-config "$PROJECT_CONFIG" --model-config "$SCHEME3_CONFIG"

#   "$PYTHON_BIN" restructured_pipeline/src/evaluate.py \
#     --scheme scheme3 --model-family "$family" --split-mode "$split" \
#     --run-name "$run" --inference-mode inference_entire_data \
#     --file-suffix "${run}_${ANALYSIS_TAG}_evaluate_entire_data" \
#     --project-config "$PROJECT_CONFIG" --model-config "$SCHEME3_CONFIG"
# }

# predict_s1() {
#   family="$1"; variant="$2"
#   run="scheme1_${family}_${variant}_${S1_MODEL_TAG}"

#   "$PYTHON_BIN" restructured_pipeline/src/predict.py \
#     --scheme scheme1 --model-family "$family" --variant "$variant" \
#     --run-name "$run" --inference-mode inference_test_set \
#     --file-suffix "${run}_${ANALYSIS_TAG}_predict_inference_test_data" \
#     --project-config "$PROJECT_CONFIG" --model-config "$SCHEME1_CONFIG"

#   "$PYTHON_BIN" restructured_pipeline/src/predict.py \
#     --scheme scheme1 --model-family "$family" --variant "$variant" \
#     --run-name "$run" --inference-mode inference_entire_data \
#     --file-suffix "${run}_${ANALYSIS_TAG}_predict_inference_entire_data" \
#     --project-config "$PROJECT_CONFIG" --model-config "$SCHEME1_CONFIG"
# }

# predict_s3() {
#   family="$1"; split="$2"
#   run="scheme3_${family}_${split}_${S3_MODEL_TAG}"

#   "$PYTHON_BIN" restructured_pipeline/src/predict.py \
#     --scheme scheme3 --model-family "$family" --split-mode "$split" \
#     --run-name "$run" --inference-mode inference_test_set \
#     --file-suffix "${run}_${ANALYSIS_TAG}_predict_inference_test_data" \
#     --project-config "$PROJECT_CONFIG" --model-config "$SCHEME3_CONFIG"

#   "$PYTHON_BIN" restructured_pipeline/src/predict.py \
#     --scheme scheme3 --model-family "$family" --split-mode "$split" \
#     --run-name "$run" --inference-mode inference_entire_data \
#     --file-suffix "${run}_${ANALYSIS_TAG}_predict_inference_entire_data" \
#     --project-config "$PROJECT_CONFIG" --model-config "$SCHEME3_CONFIG"
# }

# echo "=== 1. EVALUATE SAVED MODELS ==="
# eval_s1 random_forest rf_strat_y
# eval_s1 random_forest rf_strat_y_K
# eval_s1 xgboost xgb_strat_y
# eval_s1 xgboost xgb_strat_y_K
# eval_s3 random_forest strat_y
# eval_s3 random_forest strat_y_K
# eval_s3 xgboost strat_y
# eval_s3 xgboost strat_y_K

# echo "=== 2. REGENERATE PREDICTION / WRONG-CASE EXPORTS ==="
# predict_s1 random_forest rf_strat_y
# predict_s1 random_forest rf_strat_y_K
# predict_s1 xgboost xgb_strat_y
# predict_s1 xgboost xgb_strat_y_K
# predict_s3 random_forest strat_y
# predict_s3 random_forest strat_y_K
# predict_s3 xgboost strat_y
# predict_s3 xgboost strat_y_K

# echo "=== 3. SAVED-RUN TOLERANCE REPORTS ==="
# "$PYTHON_BIN" restructured_pipeline/src/generate_saved_run_tolerance_reports.py \
#   --run-tag "$ANALYSIS_TAG"
# ============================================================
# 3. SAVED-RUN TOLERANCE REPORTS
# ============================================================

echo
echo "============================================================"
echo "3. SAVED-RUN TOLERANCE REPORTS"
echo "============================================================"

echo "Skipped in this workflow."
echo
echo "Scheme 1 model tag: $S1_MODEL_TAG"
echo "Scheme 3 model tag: $S3_MODEL_TAG"
echo "Analysis tag:       $ANALYSIS_TAG"
echo
echo "The failure-analysis tolerance workbooks are generated by"
echo "compare_wrong_predictions.py below."


# ============================================================
# 4. WRONG-PREDICTION -> TOLERANCE -> IMPORTANT FAILURES
# ============================================================

echo
echo "============================================================"
echo "4. WRONG-PREDICTION -> TOLERANCE -> IMPORTANT FAILURES"
echo "============================================================"

"$PYTHON_BIN" restructured_pipeline/src/compare_wrong_predictions.py \
  --run-test-data-pipeline \
  --analysis-tag "$ANALYSIS_TAG"

"$PYTHON_BIN" restructured_pipeline/src/compare_wrong_predictions.py \
  --run-entire-data-pipeline \
  --analysis-tag "$ANALYSIS_TAG"


# ============================================================
# Verify important-failure workbooks
# ============================================================

if [ ! -f "$TEST_IMPORTANT" ]; then
  echo "ERROR: Missing:"
  echo "  $TEST_IMPORTANT"
  exit 1
fi

if [ ! -f "$ENTIRE_IMPORTANT" ]; then
  echo "ERROR: Missing:"
  echo "  $ENTIRE_IMPORTANT"
  exit 1
fi


# ============================================================
# 5. ENRICHED FAILURE ANALYSIS
# ============================================================

echo
echo "============================================================"
echo "5. FAILURE ANALYSIS ENRICHMENT"
echo "============================================================"

"$PYTHON_BIN" restructured_pipeline/src/failure_analysis_summary.py \
  --mode enrich \
  --failure-workbook "$ENTIRE_IMPORTANT" \
  --test-failure-workbook "$TEST_IMPORTANT" \
  --enriched-failure-workbook-output "$ENRICHED" \
  --project-config "$PROJECT_CONFIG"

if [ ! -f "$ENRICHED" ]; then
  echo "ERROR: Missing enriched workbook:"
  echo "  $ENRICHED"
  exit 1
fi


# ============================================================
# 6. MIXED XGB FAILURE ANALYSIS
# ============================================================

echo
echo "============================================================"
echo "6. MIXED XGB FAILURE ANALYSIS"
echo "============================================================"

"$PYTHON_BIN" restructured_pipeline/src/mixed_failure_case_combinations_xgb_only.py \
  --failure-workbook "$ENRICHED" \
  --split-mode strat_y \
  --output-path "$MIXED_Y"

"$PYTHON_BIN" restructured_pipeline/src/mixed_failure_case_combinations_xgb_only.py \
  --failure-workbook "$ENRICHED" \
  --split-mode strat_y_K \
  --output-path "$MIXED_YK"


# ============================================================
# 7. FINAL FAILURE COUNTS
# ============================================================

echo
echo "============================================================"
echo "7. FAILURE COUNTS / TRAIN-TEST ANALYSIS"
echo "============================================================"

"$PYTHON_BIN" restructured_pipeline/src/failure_analysis_counts_and_train_test_summary.py \
  --important-workbook "$ENRICHED" \
  --highlighted-input-copy "$HIGHLIGHTED" \
  --scheme1-entire-raw-glob "${WRONG_ROOT}/scheme1/*/wrong_predictions_*${ANALYSIS_TAG}*_predict_inference_entire_data.xlsx" \
  --scheme1-test-raw-glob "${WRONG_ROOT}/scheme1/*/wrong_predictions_*${ANALYSIS_TAG}*_predict_inference_test_data.xlsx" \
  --scheme3-entire-raw-glob "${WRONG_ROOT}/scheme3/*/wrong_predictions_*${ANALYSIS_TAG}*_predict_inference_entire_data.xlsx" \
  --scheme3-test-raw-glob "${WRONG_ROOT}/scheme3/*/wrong_predictions_*${ANALYSIS_TAG}*_predict_inference_test_data.xlsx" \
  --skip-mixed \
  --output-path "$COUNTS"


# ============================================================
# 8. FINAL CHECK
# ============================================================

echo
echo "============================================================"
echo "8. OUTPUT CHECK"
echo "============================================================"

for file in \
  "$ENRICHED" \
  "$MIXED_Y" \
  "$MIXED_YK" \
  "$HIGHLIGHTED" \
  "$COUNTS"
do
  if [ ! -f "$file" ]; then
    echo "ERROR: Missing output:"
    echo "  $file"
    exit 1
  fi

  echo "FOUND: $file"
done

echo
echo "============================================================"
echo "CLEAN FAILURE ANALYSIS COMPLETED"
echo "============================================================"