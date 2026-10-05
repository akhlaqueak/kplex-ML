#!/usr/bin/env bash
set -euo pipefail

TRAIN_TAG="${TRAIN_TAG:-17sep2026}"
ANALYSIS_TAG="${ANALYSIS_TAG:-17sep2026}"
PROJECT_CONFIG="restructured_pipeline/configs/project.yaml"
SCHEME5_CONFIG="restructured_pipeline/configs/models_scheme5.yaml"
PRED_ROOT="restructured_pipeline/outputs/predictions/scheme5"

PYTHON_BIN="python"
if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/python.exe" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/python.exe"
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python.exe)"
fi

echo "Python:       $PYTHON_BIN"
echo "Train tag:    $TRAIN_TAG"
echo "Analysis tag: $ANALYSIS_TAG"
"$PYTHON_BIN" --version

run_one() {
  family="$1"
  model="$2"
  variant="$3"
  split="$4"
  run="scheme5_${family}_${model}_${variant}_${TRAIN_TAG}"

  for scope in test_data entire_data; do
    if [ "$scope" = test_data ]; then
      mode=inference_test_set
    else
      mode=inference_entire_data
    fi

    suffix="${run}_${ANALYSIS_TAG}_predict_inference_${scope}"
    pred="${PRED_ROOT}/${family}/${model}/all_predictions_${suffix}.csv"

    echo "=== PREDICT ${family}/${model}/${split}/${scope} ==="
    "$PYTHON_BIN" restructured_pipeline/src/predict.py \
      --scheme scheme5 \
      --scheme5-family "$family" \
      --model-family "$model" \
      --variant "$variant" \
      --run-name "$run" \
      --inference-mode "$mode" \
      --file-suffix "$suffix" \
      --project-config "$PROJECT_CONFIG" \
      --model-config "$SCHEME5_CONFIG"

    test -f "$pred"

    echo "=== EVALUATE ${family}/${model}/${split}/${scope} ==="
    "$PYTHON_BIN" restructured_pipeline/src/evaluate.py \
      --scheme scheme5 \
      --scheme5-family "$family" \
      --model-family "$model" \
      --variant "$variant" \
      --split-mode "$split" \
      --run-name "$run" \
      --predictions-path "$pred" \
      --inference-mode "$mode" \
      --file-suffix "${run}_${ANALYSIS_TAG}_eval_${scope}" \
      --project-config "$PROJECT_CONFIG" \
      --model-config "$SCHEME5_CONFIG"
  done
}

for FAMILY in b other; do
  run_one "$FAMILY" random_forest rf_strat_y strat_y
  run_one "$FAMILY" random_forest rf_strat_y_K strat_y_K
  run_one "$FAMILY" xgboost xgb_strat_y strat_y
  run_one "$FAMILY" xgboost xgb_strat_y_K strat_y_K
done

echo "=== TOLERANCE + SCHEME 5 ALGORITHM-LEVEL METRICS ==="
"$PYTHON_BIN" restructured_pipeline/src/generate_saved_run_tolerance_reports.py \
  --scheme scheme5 \
  --scheme5-model-tag "$TRAIN_TAG" \
  --analysis-tag "$ANALYSIS_TAG" \
  --use-existing-predictions

echo "=== Scheme 5 predict/evaluate/tolerance/metrics complete ==="
echo "Algorithm-level metrics: restructured_pipeline/outputs/reports/scheme5/scheme5_algorithm_level_metrics_${TRAIN_TAG}_${ANALYSIS_TAG}.xlsx"
