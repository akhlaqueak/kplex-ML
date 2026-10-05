#!/usr/bin/env bash
set -e

TRAIN_TAG="${TRAIN_TAG:-14sep2026}"
ANALYSIS_TAG="${ANALYSIS_TAG:-14sep2026}"
PROJECT_CONFIG="restructured_pipeline/configs/project.yaml"
SCHEME4_CONFIG="restructured_pipeline/configs/models_scheme4.yaml"
PRED_ROOT="restructured_pipeline/outputs/predictions/scheme4"

if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python.exe)"
else
  PYTHON_BIN="python"
fi

echo "============================================================"
echo "SCHEME 4: PREDICT -> EVALUATE -> TOLERANCE"
echo "Python:       $PYTHON_BIN"
echo "Training tag: $TRAIN_TAG"
echo "Analysis tag: $ANALYSIS_TAG"
echo "============================================================"

run_one() {
  subfamily="$1"
  variant="$2"
  split_mode="$3"
  run="scheme4_${subfamily}_xgboost_${variant}_${TRAIN_TAG}"

  for scope in test_data entire_data; do
    if [ "$scope" = "test_data" ]; then
      inference_mode="inference_test_set"
    else
      inference_mode="inference_entire_data"
    fi

    suffix="${run}_${ANALYSIS_TAG}_predict_inference_${scope}"
    prediction_path="${PRED_ROOT}/${subfamily}/xgboost/all_predictions_${suffix}.csv"

    echo
    echo "============================================================"
    echo "PREDICT: $subfamily / $split_mode / $scope"
    echo "============================================================"

    "$PYTHON_BIN" restructured_pipeline/src/predict.py \
      --scheme scheme4 \
      --scheme4-family "$subfamily" \
      --model-family xgboost \
      --variant "$variant" \
      --run-name "$run" \
      --inference-mode "$inference_mode" \
      --file-suffix "$suffix" \
      --project-config "$PROJECT_CONFIG" \
      --model-config "$SCHEME4_CONFIG"

    if [ ! -f "$prediction_path" ]; then
      echo "ERROR: expected prediction CSV not found:"
      echo "  $prediction_path"
      exit 1
    fi

    echo
    echo "============================================================"
    echo "EVALUATE: $subfamily / $split_mode / $scope"
    echo "============================================================"

    "$PYTHON_BIN" restructured_pipeline/src/evaluate.py \
      --scheme scheme4 \
      --scheme4-family "$subfamily" \
      --model-family xgboost \
      --variant "$variant" \
      --split-mode "$split_mode" \
      --run-name "$run" \
      --predictions-path "$prediction_path" \
      --inference-mode "$inference_mode" \
      --file-suffix "${run}_${ANALYSIS_TAG}_eval_${scope}" \
      --project-config "$PROJECT_CONFIG" \
      --model-config "$SCHEME4_CONFIG"
  done
}

run_one oriented xgb_strat_y strat_y
run_one oriented xgb_strat_y_K strat_y_K
run_one forward xgb_strat_y strat_y
run_one forward xgb_strat_y_K strat_y_K

echo
echo "============================================================"
echo "GENERATE TOLERANCE REPORTS FROM EXISTING PREDICTIONS"
echo "============================================================"

"$PYTHON_BIN" restructured_pipeline/src/generate_saved_run_tolerance_reports.py \
  --scheme scheme4 \
  --scheme4-model-tag "$TRAIN_TAG" \
  --analysis-tag "$ANALYSIS_TAG" \
  --use-existing-predictions

echo
echo "============================================================"
echo "SCHEME 4 PIPELINE COMPLETE"
echo "============================================================"
echo "Predictions: scheme4/oriented + scheme4/forward"
echo "Evaluation:  test_data + entire_data for all four models"
echo "Tolerance:   0%, 10%, 20%, 50% for both families/splits/scopes"
echo "============================================================"
