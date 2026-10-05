#!/usr/bin/env bash
set -euo pipefail
RUN_TAG="${RUN_TAG:-17sep2026}"
PROJECT_CONFIG="restructured_pipeline/configs/project.yaml"
SCHEME5_CONFIG="restructured_pipeline/configs/models_scheme5.yaml"
PYTHON_BIN="python"
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then PYTHON_BIN="${CONDA_PREFIX}/python.exe"; elif command -v python.exe >/dev/null 2>&1; then PYTHON_BIN="$(command -v python.exe)"; fi
train_one(){ family="$1"; variant="$2"; run="scheme5_${family}_xgboost_${variant}_${RUN_TAG}"; echo "=== TRAIN Scheme 5 ${family} / ${variant} ==="; "$PYTHON_BIN" restructured_pipeline/src/train.py --scheme scheme5 --scheme5-family "$family" --model-family xgboost --variant "$variant" --project-config "$PROJECT_CONFIG" --model-config "$SCHEME5_CONFIG" --save-files --file-suffix "$run"; }
train_one b xgb_strat_y
train_one b xgb_strat_y_K
train_one other xgb_strat_y
train_one other xgb_strat_y_K
echo "=== Scheme 5 training complete: $RUN_TAG ==="
