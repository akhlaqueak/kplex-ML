#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${RUN_TAG:-17sep2026}"
PROJECT_CONFIG="restructured_pipeline/configs/project.yaml"
SCHEME5_CONFIG="restructured_pipeline/configs/models_scheme5.yaml"

PYTHON_BIN="python"
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python.exe)"
fi

echo "Python:  $PYTHON_BIN"
echo "Run tag: $RUN_TAG"
"$PYTHON_BIN" --version

train_one() {
  family="$1"
  model="$2"
  variant="$3"
  run="scheme5_${family}_${model}_${variant}_${RUN_TAG}"

  echo "=== TRAIN Scheme 5 ${family} / ${model} / ${variant} ==="
  "$PYTHON_BIN" restructured_pipeline/src/train.py \
    --scheme scheme5 \
    --scheme5-family "$family" \
    --model-family "$model" \
    --variant "$variant" \
    --project-config "$PROJECT_CONFIG" \
    --model-config "$SCHEME5_CONFIG" \
    --save-files \
    --file-suffix "$run"
}

for FAMILY in b other; do
  train_one "$FAMILY" random_forest rf_strat_y
  train_one "$FAMILY" random_forest rf_strat_y_K
  train_one "$FAMILY" xgboost xgb_strat_y
  train_one "$FAMILY" xgboost xgb_strat_y_K
done

echo "=== Scheme 5 RF + XGB training complete: $RUN_TAG ==="
