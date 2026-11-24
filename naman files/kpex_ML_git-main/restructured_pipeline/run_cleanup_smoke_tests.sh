#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "/opt/miniconda3/envs/kpex/bin/python" ]]; then
  PYTHON_BIN="/opt/miniconda3/envs/kpex/bin/python"
else
  PYTHON_BIN="python3"
fi

STAMP="${SMOKE_TEST_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TMP_ROOT="${SMOKE_TEST_TMP_ROOT:-/private/tmp}"
SMOKE_ROOT="${TMP_ROOT}/restructured_pipeline_cleanup_smoke_${STAMP}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${SMOKE_ROOT}/mplconfig}"

mkdir -p "${SMOKE_ROOT}" "${MPLCONFIGDIR}"

PROJECT_CONFIG="restructured_pipeline/configs/project.yaml"
MODEL_CONFIG="restructured_pipeline/configs/models_scheme1.yaml"
SCHEME="scheme1"
MODEL_FAMILY="random_forest"
VARIANT="rf_strat_y"
SPLIT_MODE="strat_y"
TOL="0.0"
THRESHOLD="0.5"

RUN_NAME="scheme1_random_forest_rf_strat_y_cleanup_smoke_${STAMP}"
EVAL_RUN_NAME="${RUN_NAME}_eval_inference_test_data"
PREDICT_TEST_RUN_NAME="${RUN_NAME}_predict_inference_test_data"
PREDICT_FULL_RUN_NAME="${RUN_NAME}_predict_inference_entire_data"

TRAIN_WRONG_XLSX="restructured_pipeline/outputs/wrong_cases/${SCHEME}/${MODEL_FAMILY}/wrong_predictions_${RUN_NAME}.xlsx"
PREDICT_TEST_WRONG_XLSX="restructured_pipeline/outputs/wrong_cases/${SCHEME}/${MODEL_FAMILY}/wrong_predictions_${PREDICT_TEST_RUN_NAME}.xlsx"
PREDICT_FULL_WRONG_XLSX="restructured_pipeline/outputs/wrong_cases/${SCHEME}/${MODEL_FAMILY}/wrong_predictions_${PREDICT_FULL_RUN_NAME}.xlsx"

COMPARE_OUTPUT="${SMOKE_ROOT}/compare_wrong_predictions_smoke.xlsx"
TOLERANCE_OUTPUT="${SMOKE_ROOT}/compare_wrong_predictions_tolerance_smoke.xlsx"
IMPORTANT_OUTPUT="${SMOKE_ROOT}/compare_wrong_predictions_important_failures_smoke.xlsx"
FILTERED_OUTPUT="${SMOKE_ROOT}/failure_case_analysis_important_failures_smoke.xlsx"
ENRICH_OUTPUT="${SMOKE_ROOT}/failure_analysis_summary_smoke.xlsx"
UNFILTERED_OUTPUT="${SMOKE_ROOT}/failure_analysis_summary_unfiltered_smoke.xlsx"
MIXED_OUTPUT="${SMOKE_ROOT}/mixed_failure_case_combinations_smoke.xlsx"
MIXED_MASTER_INPUT=""
MIXED_FILTERED_INPUT=""

declare -a COMMAND_LOG=()
declare -a OUTPUT_LOG=()

log() {
  printf '%s\n' "$*"
}

record_command() {
  COMMAND_LOG+=("$1")
}

record_output() {
  OUTPUT_LOG+=("$1")
}

run_step() {
  local label="$1"
  shift
  local rendered
  rendered="$(printf '%q ' "$@")"
  rendered="${rendered% }"
  record_command "${label}: ${rendered}"
  log ""
  log "==> ${label}"
  log "CMD: ${rendered}"
  if "$@"; then
    log "PASS: ${label}"
  else
    log "FAIL: ${label}"
    exit 1
  fi
}

expect_file() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    record_output "${path}"
    log "PASS: ${label}: ${path}"
  else
    log "FAIL: ${label}: missing ${path}"
    exit 1
  fi
}

resolve_mixed_inputs() {
  local phase9_master="restructured_pipeline/outputs/reports/phase9_validation/failure_analysis_summary_unfiltered_phase9.xlsx"
  local phase9_filtered="restructured_pipeline/outputs/reports/phase9_validation/failure_analysis_summary_phase9.xlsx"
  local canonical_master="restructured_pipeline/outputs/reports/failure_analysis_summary_unfiltered_all_scheme_entire_data.xlsx"
  local canonical_filtered="restructured_pipeline/outputs/reports/failure_analysis_summary_all_scheme_entire_data.xlsx"

  if [[ -f "${phase9_master}" && -f "${phase9_filtered}" ]]; then
    MIXED_MASTER_INPUT="${phase9_master}"
    MIXED_FILTERED_INPUT="${phase9_filtered}"
    return 0
  fi

  if [[ -f "${canonical_master}" && -f "${canonical_filtered}" ]]; then
    MIXED_MASTER_INPUT="${canonical_master}"
    MIXED_FILTERED_INPUT="${canonical_filtered}"
    return 0
  fi

  log "FAIL: Mixed failure-case smoke test requires an existing multi-scheme"
  log "filtered/unfiltered workbook pair."
  log "Looked for:"
  log "  - ${phase9_master}"
  log "  - ${phase9_filtered}"
  log "  - ${canonical_master}"
  log "  - ${canonical_filtered}"
  exit 1
}

log "Cleanup smoke tests will write temporary outputs under:"
log "  ${SMOKE_ROOT}"
log "Using Python interpreter:"
log "  ${PYTHON_BIN}"

run_step "Python syntax check" \
  "${PYTHON_BIN}" -m py_compile $(rg --files restructured_pipeline/src -g '*.py')

run_step "Bash syntax check" \
  bash -n restructured_pipeline/run_full_pipeline.sh

run_step "Import checks" \
  env MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" - <<'PY'
from restructured_pipeline.src import train, evaluate, predict
from restructured_pipeline.src import compare_wrong_predictions
from restructured_pipeline.src import failure_case_filter
from restructured_pipeline.src import failure_analysis_summary
from restructured_pipeline.src import failure_analysis_unfiltered_summary
from restructured_pipeline.src import mixed_failure_case_combinations
print("imports_ok")
PY

run_step "Scheme 1 RF training smoke test" \
  env MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" \
  restructured_pipeline/src/train.py \
  --scheme "${SCHEME}" \
  --model-family "${MODEL_FAMILY}" \
  --variant "${VARIANT}" \
  --split-mode "${SPLIT_MODE}" \
  --tol "${TOL}" \
  --save-files \
  --file-suffix "${RUN_NAME}" \
  --project-config "${PROJECT_CONFIG}" \
  --model-config "${MODEL_CONFIG}"

expect_file "Training wrong-case workbook" "${TRAIN_WRONG_XLSX}"

run_step "Held-out evaluation smoke test" \
  env MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" \
  restructured_pipeline/src/evaluate.py \
  --scheme "${SCHEME}" \
  --model-family "${MODEL_FAMILY}" \
  --variant "${VARIANT}" \
  --split-mode "${SPLIT_MODE}" \
  --run-name "${RUN_NAME}" \
  --inference-mode inference_test_set \
  --tol "${TOL}" \
  --file-suffix "${EVAL_RUN_NAME}" \
  --project-config "${PROJECT_CONFIG}" \
  --model-config "${MODEL_CONFIG}"

run_step "Held-out prediction smoke test" \
  env MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" \
  restructured_pipeline/src/predict.py \
  --scheme "${SCHEME}" \
  --model-family "${MODEL_FAMILY}" \
  --variant "${VARIANT}" \
  --split-mode "${SPLIT_MODE}" \
  --run-name "${RUN_NAME}" \
  --inference-mode inference_test_set \
  --tol "${TOL}" \
  --file-suffix "${PREDICT_TEST_RUN_NAME}" \
  --project-config "${PROJECT_CONFIG}" \
  --model-config "${MODEL_CONFIG}"

expect_file "Held-out wrong-case workbook" "${PREDICT_TEST_WRONG_XLSX}"

run_step "Entire-data prediction smoke test" \
  env MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" \
  restructured_pipeline/src/predict.py \
  --scheme "${SCHEME}" \
  --model-family "${MODEL_FAMILY}" \
  --variant "${VARIANT}" \
  --split-mode "${SPLIT_MODE}" \
  --run-name "${RUN_NAME}" \
  --inference-mode inference_entire_data \
  --tol "${TOL}" \
  --file-suffix "${PREDICT_FULL_RUN_NAME}" \
  --project-config "${PROJECT_CONFIG}" \
  --model-config "${MODEL_CONFIG}"

expect_file "Entire-data predict wrong-case workbook or alias" "${PREDICT_FULL_WRONG_XLSX}"

run_step "Direct compare smoke test" \
  env MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" \
  restructured_pipeline/src/compare_wrong_predictions.py \
  --wrong-predictions-file "wrong_predictions_inference_entire_data_without_holdout_${SPLIT_MODE}_${SCHEME}_${MODEL_FAMILY}=${TRAIN_WRONG_XLSX}" \
  --wrong-predictions-file "wrong_predictions_predict_inference_test_data_with_holdout_${SPLIT_MODE}_${SCHEME}_${MODEL_FAMILY}=${PREDICT_TEST_WRONG_XLSX}" \
  --threshold "${THRESHOLD}" \
  --output-path "${COMPARE_OUTPUT}" \
  --generate-tolerance-report \
  --tolerance-output-path "${TOLERANCE_OUTPUT}" \
  --generate-important-failure-report \
  --important-failure-output-path "${IMPORTANT_OUTPUT}" \
  --source-model-label "${SCHEME}_${MODEL_FAMILY}_${SPLIT_MODE}"

expect_file "Compare workbook" "${COMPARE_OUTPUT}"
expect_file "Tolerance workbook" "${TOLERANCE_OUTPUT}"
expect_file "Important-failure workbook" "${IMPORTANT_OUTPUT}"

run_step "Failure-case filter smoke test" \
  env MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" \
  restructured_pipeline/src/failure_case_filter.py \
  --input-workbook "${COMPARE_OUTPUT}" \
  --output-path "${FILTERED_OUTPUT}"

expect_file "Filtered important-failure workbook" "${FILTERED_OUTPUT}"

run_step "Failure-analysis summary enrich smoke test" \
  env MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" \
  restructured_pipeline/src/failure_analysis_summary.py \
  --mode enrich \
  --failure-workbook "${FILTERED_OUTPUT}" \
  --enriched-failure-workbook-output "${ENRICH_OUTPUT}" \
  --project-config "${PROJECT_CONFIG}"

expect_file "Filtered failure-analysis summary workbook" "${ENRICH_OUTPUT}"

run_step "Unfiltered summary smoke test" \
  env MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" \
  restructured_pipeline/src/failure_analysis_unfiltered_summary.py \
  --raw-glob "restructured_pipeline/outputs/wrong_cases/*/*/wrong_predictions_*${STAMP}*.xlsx" \
  --project-config-path "${PROJECT_CONFIG}" \
  --filtered-reference-path "${ENRICH_OUTPUT}" \
  --output-path "${UNFILTERED_OUTPUT}"

expect_file "Unfiltered failure-analysis summary workbook" "${UNFILTERED_OUTPUT}"

resolve_mixed_inputs
log "Using mixed-failure smoke inputs:"
log "  master: ${MIXED_MASTER_INPUT}"
log "  filtered: ${MIXED_FILTERED_INPUT}"

run_step "Mixed failure-case combinations smoke test" \
  env MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" \
  restructured_pipeline/src/mixed_failure_case_combinations.py \
  --master-workbook "${MIXED_MASTER_INPUT}" \
  --filtered-workbook "${MIXED_FILTERED_INPUT}" \
  --output-path "${MIXED_OUTPUT}"

expect_file "Mixed failure-case combinations workbook" "${MIXED_OUTPUT}"

log ""
log "Cleanup smoke tests completed successfully."
log ""
log "Commands run:"
for entry in "${COMMAND_LOG[@]}"; do
  log "  - ${entry}"
done
log ""
log "Outputs created or verified:"
for entry in "${OUTPUT_LOG[@]}"; do
  log "  - ${entry}"
done
