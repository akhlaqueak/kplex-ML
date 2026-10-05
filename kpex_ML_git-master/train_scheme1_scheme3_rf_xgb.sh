#!/usr/bin/env bash

# Scheme 1 + Scheme 3 training matrix.
# Designed to run from Git Bash / bash on Windows while using the active
# Windows Python environment (including Conda base).
#
# Run from the repository root:
#   bash train_scheme1_scheme3_rf_xgb.sh
#
# Optional:
#   RUN_TAG=03sep2026 bash train_scheme1_scheme3_rf_xgb.sh

set -e

PROJECT_CONFIG="restructured_pipeline/configs/project.yaml"
SCHEME1_CONFIG="restructured_pipeline/configs/models_scheme1.yaml"
SCHEME3_CONFIG="restructured_pipeline/configs/models_scheme3.yaml"
RUN_TAG="${RUN_TAG:-03sep2026}"

# Prefer the Python executable belonging to the active Conda environment.
# Under Git Bash, `python` may resolve to the Microsoft Store shim, so use
# CONDA_PREFIX/python.exe when available.
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python.exe)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "ERROR: Python executable not found."
    exit 1
fi

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "Run tag: $RUN_TAG"
echo

run_scheme1() {
    model_family="$1"
    variant="$2"
    run_name="scheme1_${model_family}_${variant}_${RUN_TAG}"

    echo "============================================================"
    echo "Training: $run_name"
    echo "============================================================"

    "$PYTHON_BIN" restructured_pipeline/src/train.py \
        --scheme scheme1 \
        --model-family "$model_family" \
        --variant "$variant" \
        --file-suffix "$run_name" \
        --project-config "$PROJECT_CONFIG" \
        --model-config "$SCHEME1_CONFIG" \
        --save-files

    echo "Completed: $run_name"
    echo
}

run_scheme3() {
    model_family="$1"
    split_mode="$2"
    run_name="scheme3_${model_family}_${split_mode}_${RUN_TAG}"

    echo "============================================================"
    echo "Training: $run_name"
    echo "============================================================"

    "$PYTHON_BIN" restructured_pipeline/src/train.py \
        --scheme scheme3 \
        --model-family "$model_family" \
        --split-mode "$split_mode" \
        --file-suffix "$run_name" \
        --project-config "$PROJECT_CONFIG" \
        --model-config "$SCHEME3_CONFIG" \
        --save-files

    echo "Completed: $run_name"
    echo
}

echo "Starting Scheme 1 training..."
run_scheme1 random_forest rf_strat_y
run_scheme1 random_forest rf_strat_y_K
run_scheme1 xgboost xgb_strat_y
run_scheme1 xgboost xgb_strat_y_K

echo "Starting Scheme 3 training..."
run_scheme3 random_forest strat_y
run_scheme3 random_forest strat_y_K
run_scheme3 xgboost strat_y
run_scheme3 xgboost strat_y_K

echo "============================================================"
echo "All Scheme 1 and Scheme 3 training runs completed."
echo "============================================================"