#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
# Core selection variables for the end-to-end pipeline run.
SCHEME="${SCHEME:-scheme2}"
MODEL_FAMILY="${MODEL_FAMILY:-random_forest}"
SPLIT_MODE="${SPLIT_MODE:-strat_y}"
RUN_NAME="${RUN_NAME:-}"
FULL_EXPERIMENT_MATRIX="${FULL_EXPERIMENT_MATRIX:-0}"
MATRIX_RUN_NAME="${MATRIX_RUN_NAME:-}"

# Execution and output variables.
INFERENCE_MODE="${INFERENCE_MODE:-}"
EVAL_INFERENCE_MODE="${EVAL_INFERENCE_MODE:-${INFERENCE_MODE:-inference_test_set}}"
PREDICT_INFERENCE_MODE_TEST="${PREDICT_INFERENCE_MODE_TEST:-inference_test_set}"
PREDICT_INFERENCE_MODE_FULL="${PREDICT_INFERENCE_MODE_FULL:-${PREDICT_INFERENCE_MODE:-inference_entire_data}}"
TIME_GAP_THRESHOLD="${TIME_GAP_THRESHOLD:-0.5}"
PROJECT_CONFIG="${PROJECT_CONFIG:-${REPO_ROOT}/restructured_pipeline/configs/project.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-}"
MODEL_CONFIG_SCHEME1="${MODEL_CONFIG_SCHEME1:-}"
MODEL_CONFIG_SCHEME2="${MODEL_CONFIG_SCHEME2:-}"
MODEL_CONFIG_SCHEME3="${MODEL_CONFIG_SCHEME3:-}"
RUN_TAG="${RUN_TAG:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
COMPARE_INPUT_ROOT="${COMPARE_INPUT_ROOT:-}"
COMPARE_INPUT_MODE="${COMPARE_INPUT_MODE:-direct}"
REPORT_MODEL_LABEL="${REPORT_MODEL_LABEL:-}"
REPORT_OUTPUT_PATH="${REPORT_OUTPUT_PATH:-}"
TOLERANCE_OUTPUT_PATH="${TOLERANCE_OUTPUT_PATH:-}"
IMPORTANT_FAILURE_OUTPUT_PATH="${IMPORTANT_FAILURE_OUTPUT_PATH:-}"
VARIANT="${VARIANT:-}"
TOL="${TOL:-0.0}"
RUN_FULL_PREDICT="${RUN_FULL_PREDICT:-1}"

output_mode_tag() {
  case "$1" in
    inference_test_set) echo "inference_test_data" ;;
    inference_entire_data) echo "inference_entire_data" ;;
    *)
      echo "Unsupported inference mode for output naming: $1" >&2
      exit 1
      ;;
  esac
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --full-experiment-matrix
  --matrix-run-name <name>
  --scheme <scheme1|scheme2|scheme3>
  --model-family <rf|random_forest|xgb|xgboost>
  --split-mode <strat_y|strat_y_K>
  --inference-mode <inference_test_set|inference_entire_data>
  --eval-inference-mode <inference_test_set|inference_entire_data>
  --predict-inference-mode <inference_test_set|inference_entire_data>
  --predict-inference-mode-test <inference_test_set|inference_entire_data>
  --predict-inference-mode-full <inference_test_set|inference_entire_data>
  --time-gap-threshold <float>
  --project-config <path>
  --model-config <path>
  --model-config-scheme1 <path>
  --model-config-scheme2 <path>
  --model-config-scheme3 <path>
  --variant <scheme1_variant>
  --tol <float>
  --run-name <name>
  --output-root <path>
  --compare-input-mode <direct|staging>
  --compare-input-root <path>
  --skip-full-predict
  --report-output-path <path>
  --tolerance-output-path <path>
  --important-failure-output-path <path>
  --python-bin <python-executable>
  --help

Examples:
  $(basename "$0") --full-experiment-matrix --matrix-run-name all_schemes_apr15
  $(basename "$0") --scheme scheme2 --model-family random_forest --split-mode strat_y
  $(basename "$0") --scheme scheme2 --model-family xgboost --split-mode strat_y_K
  $(basename "$0") --scheme scheme1 --model-family xgboost --split-mode strat_y_K
  $(basename "$0") --scheme scheme3 --model-family random_forest --split-mode strat_y
  $(basename "$0") --scheme scheme3 --model-family xgboost --split-mode strat_y_K
  $(basename "$0") --scheme scheme2 --model-family xgboost --eval-inference-mode inference_test_set --predict-inference-mode-full inference_entire_data
  $(basename "$0") --python-bin /opt/miniconda3/envs/kpex/bin/python --scheme scheme2

Notes:
  --python-bin must be a direct Python executable path or command name.
  Use /path/to/python or python, not a multi-word wrapper like:
    conda run -n kpex python
  The wrapper always writes both prediction exports in one run:
    all_predictions_<RUN_NAME>_predict_inference_test_data.csv
    all_predictions_<RUN_NAME>_predict_inference_entire_data.csv
  Use --skip-full-predict to skip the full-dataset predict.py export when
  you only need the direct compare flow and do not need the compatibility
  alias outputs from predict.py.
  Test-set outputs use the token "test_data" in filenames and
  full-dataset outputs use the token "entire_data".
  --predict-inference-mode is kept as a compatibility alias for the full-dataset
  prediction export mode.
  --compare-input-mode defaults to direct and only uses staged compare-input
  copies when explicitly set to staging.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-experiment-matrix) FULL_EXPERIMENT_MATRIX="1"; shift 1 ;;
    --matrix-run-name) MATRIX_RUN_NAME="$2"; shift 2 ;;
    --scheme) SCHEME="$2"; shift 2 ;;
    --model-family) MODEL_FAMILY="$2"; shift 2 ;;
    --split-mode) SPLIT_MODE="$2"; shift 2 ;;
    --inference-mode) INFERENCE_MODE="$2"; EVAL_INFERENCE_MODE="$2"; shift 2 ;;
    --eval-inference-mode) EVAL_INFERENCE_MODE="$2"; shift 2 ;;
    --predict-inference-mode) PREDICT_INFERENCE_MODE_FULL="$2"; shift 2 ;;
    --predict-inference-mode-test) PREDICT_INFERENCE_MODE_TEST="$2"; shift 2 ;;
    --predict-inference-mode-full) PREDICT_INFERENCE_MODE_FULL="$2"; shift 2 ;;
    --time-gap-threshold) TIME_GAP_THRESHOLD="$2"; shift 2 ;;
    --project-config) PROJECT_CONFIG="$2"; shift 2 ;;
    --model-config) MODEL_CONFIG="$2"; shift 2 ;;
    --model-config-scheme1) MODEL_CONFIG_SCHEME1="$2"; shift 2 ;;
    --model-config-scheme2) MODEL_CONFIG_SCHEME2="$2"; shift 2 ;;
    --model-config-scheme3) MODEL_CONFIG_SCHEME3="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --tol) TOL="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --compare-input-mode) COMPARE_INPUT_MODE="$2"; shift 2 ;;
    --compare-input-root) COMPARE_INPUT_ROOT="$2"; shift 2 ;;
    --skip-full-predict) RUN_FULL_PREDICT="0"; shift 1 ;;
    --report-output-path) REPORT_OUTPUT_PATH="$2"; shift 2 ;;
    --tolerance-output-path) TOLERANCE_OUTPUT_PATH="$2"; shift 2 ;;
    --important-failure-output-path) IMPORTANT_FAILURE_OUTPUT_PATH="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

case "$MODEL_FAMILY" in
  rf) MODEL_FAMILY="random_forest" ;;
  xgb) MODEL_FAMILY="xgboost" ;;
esac

case "$EVAL_INFERENCE_MODE" in
  test|test_set) EVAL_INFERENCE_MODE="inference_test_set" ;;
  entire|entire_data|full) EVAL_INFERENCE_MODE="inference_entire_data" ;;
esac

case "$PREDICT_INFERENCE_MODE_TEST" in
  test|test_set) PREDICT_INFERENCE_MODE_TEST="inference_test_set" ;;
  entire|entire_data|full) PREDICT_INFERENCE_MODE_TEST="inference_entire_data" ;;
esac

case "$PREDICT_INFERENCE_MODE_FULL" in
  test|test_set) PREDICT_INFERENCE_MODE_FULL="inference_test_set" ;;
  entire|entire_data|full) PREDICT_INFERENCE_MODE_FULL="inference_entire_data" ;;
esac

if [[ "$PYTHON_BIN" =~ [[:space:]] ]]; then
  echo "--python-bin must be a single Python executable path or command name." >&2
  echo "Received: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -z "$RUN_NAME" ]]; then
  if [[ -z "$RUN_TAG" ]]; then
    RUN_TAG="$(date +%Y%m%d_%H%M%S)"
  fi
  RUN_NAME="${SCHEME}_${MODEL_FAMILY}_${SPLIT_MODE}_${RUN_TAG}"
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="${REPO_ROOT}/restructured_pipeline/outputs"
fi

if [[ -z "$COMPARE_INPUT_ROOT" ]]; then
  COMPARE_INPUT_ROOT="${OUTPUT_ROOT}/reports/compare_inputs/${RUN_NAME}"
fi

if [[ -z "$REPORT_MODEL_LABEL" ]]; then
  REPORT_MODEL_LABEL="${SCHEME}_${MODEL_FAMILY}_${SPLIT_MODE}"
fi

if [[ -z "$REPORT_OUTPUT_PATH" ]]; then
  REPORT_OUTPUT_PATH="${OUTPUT_ROOT}/reports/compare_wrong_predictions_${RUN_NAME}.xlsx"
fi

case "$COMPARE_INPUT_MODE" in
  direct|staging) ;;
  *)
    echo "Unsupported compare input mode: ${COMPARE_INPUT_MODE}" >&2
    exit 1
    ;;
esac

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ "${FULL_EXPERIMENT_MATRIX}" == "1" ]]; then
  MATRIX_ARGS=(
    "${PYTHON_BIN}" "${REPO_ROOT}/restructured_pipeline/src/experiment_matrix.py"
    --project-config "${PROJECT_CONFIG}"
    --tol "${TOL}"
  )

  if [[ -n "${MATRIX_RUN_NAME}" ]]; then
    MATRIX_ARGS+=(--matrix-run-name "${MATRIX_RUN_NAME}")
  fi

  if [[ -n "${MODEL_CONFIG_SCHEME1}" ]]; then
    MATRIX_ARGS+=(--model-config-scheme1 "${MODEL_CONFIG_SCHEME1}")
  fi

  if [[ -n "${MODEL_CONFIG_SCHEME2}" ]]; then
    MATRIX_ARGS+=(--model-config-scheme2 "${MODEL_CONFIG_SCHEME2}")
  fi

  if [[ -n "${MODEL_CONFIG_SCHEME3}" ]]; then
    MATRIX_ARGS+=(--model-config-scheme3 "${MODEL_CONFIG_SCHEME3}")
  fi

  echo "Running full experiment matrix..."
  "${MATRIX_ARGS[@]}"
  exit 0
fi

if [[ -z "$MODEL_CONFIG" ]]; then
  case "$SCHEME" in
    scheme1)
      MODEL_CONFIG="${REPO_ROOT}/restructured_pipeline/configs/models_scheme1.yaml"
      ;;
    scheme2)
      MODEL_CONFIG="${REPO_ROOT}/restructured_pipeline/configs/models_scheme2.yaml"
      ;;
    scheme3)
      MODEL_CONFIG="${REPO_ROOT}/restructured_pipeline/configs/models_scheme3.yaml"
      ;;
    *)
      echo "Unsupported scheme: ${SCHEME}" >&2
      exit 1
      ;;
  esac
fi

resolve_scheme1_variant() {
  case "${MODEL_FAMILY}:${SPLIT_MODE}" in
    random_forest:strat_y) echo "rf_strat_y" ;;
    random_forest:strat_y_K) echo "rf_strat_y_K" ;;
    xgboost:strat_y) echo "xgb_strat_y" ;;
    xgboost:strat_y_K) echo "xgb_strat_y_K" ;;
    *)
      echo "Unsupported scheme1 model/split combination: ${MODEL_FAMILY} ${SPLIT_MODE}" >&2
      exit 1
      ;;
  esac
}

if [[ "$SCHEME" == "scheme1" && -z "$VARIANT" ]]; then
  VARIANT="$(resolve_scheme1_variant)"
fi

mkdir -p \
  "${OUTPUT_ROOT}/reports"

if [[ "${COMPARE_INPUT_MODE}" == "staging" ]]; then
  mkdir -p \
    "${COMPARE_INPUT_ROOT}/expand_and_split" \
    "${COMPARE_INPUT_ROOT}/split_and_expand"
fi

build_compare_metadata_label() {
  local inference_scope="$1"
  local holdout_scope="$2"
  local scheme="$3"
  local model_family="$4"
  local split_mode="$5"

  if [[ "${inference_scope}" == "test_data" ]]; then
    echo "wrong_predictions_predict_inference_test_data_${holdout_scope}_${split_mode}_${scheme}_${model_family}"
    return
  fi

  echo "wrong_predictions_inference_entire_data_${holdout_scope}_${split_mode}_${scheme}_${model_family}"
}

TRAIN_WRONG_XLSX="${OUTPUT_ROOT}/wrong_cases/${SCHEME}/${MODEL_FAMILY}/wrong_predictions_${RUN_NAME}.xlsx"
EVAL_OUTPUT_MODE_TAG="$(output_mode_tag "${EVAL_INFERENCE_MODE}")"
PREDICT_TEST_OUTPUT_MODE_TAG="$(output_mode_tag "${PREDICT_INFERENCE_MODE_TEST}")"
PREDICT_FULL_OUTPUT_MODE_TAG="$(output_mode_tag "${PREDICT_INFERENCE_MODE_FULL}")"
EVAL_RUN_NAME="${RUN_NAME}_eval_${EVAL_OUTPUT_MODE_TAG}"
PREDICT_TEST_RUN_NAME="${RUN_NAME}_predict_${PREDICT_TEST_OUTPUT_MODE_TAG}"
PREDICT_FULL_RUN_NAME="${RUN_NAME}_predict_${PREDICT_FULL_OUTPUT_MODE_TAG}"
PREDICT_TEST_WRONG_XLSX="${OUTPUT_ROOT}/wrong_cases/${SCHEME}/${MODEL_FAMILY}/wrong_predictions_${PREDICT_TEST_RUN_NAME}.xlsx"
PREDICT_FULL_WRONG_XLSX="${OUTPUT_ROOT}/wrong_cases/${SCHEME}/${MODEL_FAMILY}/wrong_predictions_${PREDICT_FULL_RUN_NAME}.xlsx"

TRAIN_ARGS=(
  "${PYTHON_BIN}" "${REPO_ROOT}/restructured_pipeline/src/train.py"
  --scheme "${SCHEME}"
  --model-family "${MODEL_FAMILY}"
  --split-mode "${SPLIT_MODE}"
  --tol "${TOL}"
  --save-files
  --file-suffix "${RUN_NAME}"
  --project-config "${PROJECT_CONFIG}"
  --model-config "${MODEL_CONFIG}"
)

if [[ "$SCHEME" == "scheme1" ]]; then
  TRAIN_ARGS+=(--variant "${VARIANT}")
fi

EVAL_ARGS=(
  "${PYTHON_BIN}" "${REPO_ROOT}/restructured_pipeline/src/evaluate.py"
  --scheme "${SCHEME}"
  --model-family "${MODEL_FAMILY}"
  --split-mode "${SPLIT_MODE}"
  --inference-mode "${EVAL_INFERENCE_MODE}"
  --tol "${TOL}"
  --run-name "${RUN_NAME}"
  --file-suffix "${EVAL_RUN_NAME}"
  --project-config "${PROJECT_CONFIG}"
  --model-config "${MODEL_CONFIG}"
)

if [[ "$SCHEME" == "scheme1" ]]; then
  EVAL_ARGS+=(--variant "${VARIANT}")
fi

PREDICT_TEST_ARGS=(
  "${PYTHON_BIN}" "${REPO_ROOT}/restructured_pipeline/src/predict.py"
  --scheme "${SCHEME}"
  --model-family "${MODEL_FAMILY}"
  --split-mode "${SPLIT_MODE}"
  --inference-mode "${PREDICT_INFERENCE_MODE_TEST}"
  --tol "${TOL}"
  --run-name "${RUN_NAME}"
  --file-suffix "${PREDICT_TEST_RUN_NAME}"
  --project-config "${PROJECT_CONFIG}"
  --model-config "${MODEL_CONFIG}"
)

if [[ "$SCHEME" == "scheme1" ]]; then
  PREDICT_TEST_ARGS+=(--variant "${VARIANT}")
fi

PREDICT_FULL_ARGS=(
  "${PYTHON_BIN}" "${REPO_ROOT}/restructured_pipeline/src/predict.py"
  --scheme "${SCHEME}"
  --model-family "${MODEL_FAMILY}"
  --split-mode "${SPLIT_MODE}"
  --inference-mode "${PREDICT_INFERENCE_MODE_FULL}"
  --tol "${TOL}"
  --run-name "${RUN_NAME}"
  --file-suffix "${PREDICT_FULL_RUN_NAME}"
  --project-config "${PROJECT_CONFIG}"
  --model-config "${MODEL_CONFIG}"
)

if [[ "$SCHEME" == "scheme1" ]]; then
  PREDICT_FULL_ARGS+=(--variant "${VARIANT}")
fi

echo "Running training..."
"${TRAIN_ARGS[@]}"

echo "Running evaluation..."
"${EVAL_ARGS[@]}"

echo "Running test-set prediction export..."
"${PREDICT_TEST_ARGS[@]}"

if [[ "${RUN_FULL_PREDICT}" == "1" ]]; then
  echo "Running full-dataset prediction export..."
  "${PREDICT_FULL_ARGS[@]}"
else
  echo "Skipping full-dataset prediction export (--skip-full-predict)."
fi

if [[ ! -f "${TRAIN_WRONG_XLSX}" ]]; then
  echo "Missing training wrong-predictions file: ${TRAIN_WRONG_XLSX}" >&2
  exit 1
fi

if [[ ! -f "${PREDICT_TEST_WRONG_XLSX}" ]]; then
  echo "Missing test-set prediction wrong-predictions file: ${PREDICT_TEST_WRONG_XLSX}" >&2
  exit 1
fi

if [[ "${RUN_FULL_PREDICT}" == "1" && ! -f "${PREDICT_FULL_WRONG_XLSX}" ]]; then
  echo "Missing full-dataset prediction wrong-predictions file: ${PREDICT_FULL_WRONG_XLSX}" >&2
  exit 1
fi

COMPARE_ARGS=(
  "${PYTHON_BIN}" "${REPO_ROOT}/restructured_pipeline/src/compare_wrong_predictions.py"
  --threshold "${TIME_GAP_THRESHOLD}"
  --output-path "${REPORT_OUTPUT_PATH}"
  --generate-tolerance-report
  --generate-important-failure-report
)

if [[ -n "${TOLERANCE_OUTPUT_PATH}" ]]; then
  COMPARE_ARGS+=(--tolerance-output-path "${TOLERANCE_OUTPUT_PATH}")
fi

if [[ -n "${IMPORTANT_FAILURE_OUTPUT_PATH}" ]]; then
  COMPARE_ARGS+=(--important-failure-output-path "${IMPORTANT_FAILURE_OUTPUT_PATH}")
fi

if [[ "${COMPARE_INPUT_MODE}" == "staging" ]]; then
  # Preserve strategy semantics for failure-case analysis staging:
  # - training wrong cases are staged under expand_and_split / without_holdout
  # - held-out prediction wrong cases are staged under split_and_expand / with_holdout
  TRAIN_COMPARE_XLSX="${COMPARE_INPUT_ROOT}/expand_and_split/wrong_predictions_inference_entire_data_without_holdout_${SPLIT_MODE}_exp_00.xlsx"
  PREDICT_COMPARE_XLSX="${COMPARE_INPUT_ROOT}/split_and_expand/wrong_predictions_${PREDICT_TEST_OUTPUT_MODE_TAG}_with_holdout_${SPLIT_MODE}_exp_01.xlsx"

  cp "${TRAIN_WRONG_XLSX}" "${TRAIN_COMPARE_XLSX}"
  cp "${PREDICT_TEST_WRONG_XLSX}" "${PREDICT_COMPARE_XLSX}"
  COMPARE_ARGS+=(--model-root "${REPORT_MODEL_LABEL}=${COMPARE_INPUT_ROOT}")
else
  TRAIN_COMPARE_LABEL="$(build_compare_metadata_label "entire_data" "without_holdout" "${SCHEME}" "${MODEL_FAMILY}" "${SPLIT_MODE}")"
  PREDICT_COMPARE_LABEL="$(build_compare_metadata_label "test_data" "with_holdout" "${SCHEME}" "${MODEL_FAMILY}" "${SPLIT_MODE}")"
  COMPARE_ARGS+=(
    --wrong-predictions-file "${TRAIN_COMPARE_LABEL}=${TRAIN_WRONG_XLSX}"
    --wrong-predictions-file "${PREDICT_COMPARE_LABEL}=${PREDICT_TEST_WRONG_XLSX}"
    --source-model-label "${REPORT_MODEL_LABEL}"
  )
fi

echo "Running failure-case analysis..."
"${COMPARE_ARGS[@]}"

echo "Pipeline complete."
echo "Run name: ${RUN_NAME}"
echo "Evaluation inference mode: ${EVAL_INFERENCE_MODE}"
echo "Evaluation run: ${EVAL_RUN_NAME}"
echo "Test prediction inference mode: ${PREDICT_INFERENCE_MODE_TEST}"
echo "Test prediction run: ${PREDICT_TEST_RUN_NAME}"
if [[ "${RUN_FULL_PREDICT}" == "1" ]]; then
  echo "Full prediction inference mode: ${PREDICT_INFERENCE_MODE_FULL}"
  echo "Full prediction run: ${PREDICT_FULL_RUN_NAME}"
else
  echo "Full prediction export: skipped"
fi
echo "Compare input mode: ${COMPARE_INPUT_MODE}"
echo "Failure-case report: ${REPORT_OUTPUT_PATH}"
if [[ -n "${TOLERANCE_OUTPUT_PATH}" ]]; then
  echo "Tolerance report: ${TOLERANCE_OUTPUT_PATH}"
fi
if [[ -n "${IMPORTANT_FAILURE_OUTPUT_PATH}" ]]; then
  echo "Important-failure report: ${IMPORTANT_FAILURE_OUTPUT_PATH}"
fi
