#!/usr/bin/env bash

set -e

RUN_TAG="${RUN_TAG:-14sep2026}"

PROJECT_CONFIG="restructured_pipeline/configs/project.yaml"
SCHEME4_CONFIG="restructured_pipeline/configs/models_scheme4.yaml"

# Use Python from the currently active Conda environment.
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python.exe)"
else
  PYTHON_BIN="python"
fi

echo "============================================================"
echo "Scheme 4 XGB training"
echo "============================================================"
echo "Python:  $PYTHON_BIN"
echo "Run tag: $RUN_TAG"
echo

"$PYTHON_BIN" --version


train_scheme4() {
  family="$1"
  variant="$2"

  run_name="scheme4_${family}_xgboost_${variant}_${RUN_TAG}"

  echo
  echo "============================================================"
  echo "Scheme 4"
  echo "Family:  $family"
  echo "Model:   XGBoost"
  echo "Variant: $variant"
  echo "Run:     $run_name"
  echo "============================================================"

  "$PYTHON_BIN" restructured_pipeline/src/train.py \
    --scheme scheme4 \
    --scheme4-family "$family" \
    --model-family xgboost \
    --variant "$variant" \
    --project-config "$PROJECT_CONFIG" \
    --model-config "$SCHEME4_CONFIG" \
    --save-files \
    --file-suffix "$run_name"

  echo
  echo "Completed:"
  echo "  $run_name"
}


# ============================================================
# 1. Oriented / strat_y
# ============================================================

train_scheme4 \
  oriented \
  xgb_strat_y


# ============================================================
# 2. Oriented / strat_y_K
# ============================================================

train_scheme4 \
  oriented \
  xgb_strat_y_K


# ============================================================
# 3. Forward / strat_y
# ============================================================

train_scheme4 \
  forward \
  xgb_strat_y


# ============================================================
# 4. Forward / strat_y_K
# ============================================================

train_scheme4 \
  forward \
  xgb_strat_y_K


echo
echo "============================================================"
echo "ALL SCHEME 4 TRAINING COMPLETED"
echo "============================================================"
echo
echo "Completed experiments:"
echo "  oriented / xgb / strat_y"
echo "  oriented / xgb / strat_y_K"
echo "  forward  / xgb / strat_y"
echo "  forward  / xgb / strat_y_K"
echo
echo "Run tag:"
echo "  $RUN_TAG"
echo "============================================================"