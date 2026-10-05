#!/usr/bin/env bash

set -e

RUN_TAG="${RUN_TAG:-03sep2026}"

PROJECT_CONFIG="restructured_pipeline/configs/project.yaml"
SCHEME1_CONFIG="restructured_pipeline/configs/models_scheme1.yaml"
SCHEME3_CONFIG="restructured_pipeline/configs/models_scheme3.yaml"

# Use Python from the currently active Conda environment when available.
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python.exe)"
else
  PYTHON_BIN="python"
fi

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Run tag: $RUN_TAG"
echo


# ============================================================
# Scheme 1
# Scheme 1 determines split_mode from the model variant.
# ============================================================

eval_scheme1() {
  family="$1"
  variant="$2"
  run="$3"

  echo "============================================================"
  echo "Scheme 1 | $family | $variant"
  echo "Run: $run"
  echo "============================================================"

  echo "=== TEST DATA ==="

  "$PYTHON_BIN" restructured_pipeline/src/evaluate.py \
    --scheme scheme1 \
    --model-family "$family" \
    --variant "$variant" \
    --run-name "$run" \
    --inference-mode inference_test_set \
    --project-config "$PROJECT_CONFIG" \
    --model-config "$SCHEME1_CONFIG"

  echo "=== ENTIRE DATA ==="

  "$PYTHON_BIN" restructured_pipeline/src/evaluate.py \
    --scheme scheme1 \
    --model-family "$family" \
    --variant "$variant" \
    --run-name "$run" \
    --inference-mode inference_entire_data \
    --project-config "$PROJECT_CONFIG" \
    --model-config "$SCHEME1_CONFIG"

  echo "Completed: $run"
  echo
}


# ============================================================
# Scheme 3
# Scheme 3 receives split_mode explicitly.
# ============================================================

eval_scheme3() {
  family="$1"
  split_mode="$2"
  run="$3"

  echo "============================================================"
  echo "Scheme 3 | $family | $split_mode"
  echo "Run: $run"
  echo "============================================================"

  echo "=== TEST DATA ==="

  "$PYTHON_BIN" restructured_pipeline/src/evaluate.py \
    --scheme scheme3 \
    --model-family "$family" \
    --split-mode "$split_mode" \
    --run-name "$run" \
    --inference-mode inference_test_set \
    --project-config "$PROJECT_CONFIG" \
    --model-config "$SCHEME3_CONFIG"

  echo "=== ENTIRE DATA ==="

  "$PYTHON_BIN" restructured_pipeline/src/evaluate.py \
    --scheme scheme3 \
    --model-family "$family" \
    --split-mode "$split_mode" \
    --run-name "$run" \
    --inference-mode inference_entire_data \
    --project-config "$PROJECT_CONFIG" \
    --model-config "$SCHEME3_CONFIG"

  echo "Completed: $run"
  echo
}


# ============================================================
# Scheme 1 evaluations
# ============================================================

echo "Starting Scheme 1 evaluations..."

eval_scheme1 \
  random_forest \
  rf_strat_y \
  "scheme1_random_forest_rf_strat_y_${RUN_TAG}"

eval_scheme1 \
  random_forest \
  rf_strat_y_K \
  "scheme1_random_forest_rf_strat_y_K_${RUN_TAG}"

eval_scheme1 \
  xgboost \
  xgb_strat_y \
  "scheme1_xgboost_xgb_strat_y_${RUN_TAG}"

eval_scheme1 \
  xgboost \
  xgb_strat_y_K \
  "scheme1_xgboost_xgb_strat_y_K_${RUN_TAG}"


# ============================================================
# Scheme 3 evaluations
# ============================================================

echo "Starting Scheme 3 evaluations..."

eval_scheme3 \
  random_forest \
  strat_y \
  "scheme3_random_forest_strat_y_${RUN_TAG}"

eval_scheme3 \
  random_forest \
  strat_y_K \
  "scheme3_random_forest_strat_y_K_${RUN_TAG}"

eval_scheme3 \
  xgboost \
  strat_y \
  "scheme3_xgboost_strat_y_${RUN_TAG}"

eval_scheme3 \
  xgboost \
  strat_y_K \
  "scheme3_xgboost_strat_y_K_${RUN_TAG}"


echo "============================================================"
echo "All Scheme 1 / Scheme 3 evaluations completed."
echo "============================================================"