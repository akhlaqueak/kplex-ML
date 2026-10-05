# Restructured Pipeline

This folder contains the active config-driven ML pipeline that replaces the
legacy notebook-only workflow. All runnable training, evaluation, prediction,
and reporting code lives under `restructured_pipeline/`.

## What The New Architecture Is

- `configs/project.yaml`: shared dataset path, feature columns, target labels,
  split rules, tolerances, and output locations
- `configs/models_scheme1.yaml`: Scheme 1 model variants and hyperparameters
- `configs/models_scheme2.yaml`: Scheme 2 staged model settings
- `configs/models_scheme3.yaml`: Scheme 3 staged model settings
- `src/features.py`, `src/data_splits.py`, `src/metrics.py`,
  `src/write_utils.py`: shared preprocessing, split, metric, and reporting code
- `src/models/flat.py`: Scheme 1 model wrappers
- `src/models/hierarchical.py`: Scheme 2 model wrappers
- `src/models/hierarchical_scheme3.py`: Scheme 3 model wrappers
- `src/train.py`: training entry point
- `src/evaluate.py`: evaluation entry point
- `src/predict.py`: prediction export entry point
- `src/experiment_matrix.py`: full scheme/model/split/inference matrix runner
- `src/tolerance_reports.py`: tolerance-workbook export helpers
- `run_full_pipeline.sh`: end-to-end wrapper for training, evaluation,
  prediction export, wrong-case reporting, and the full experiment matrix

## Data And Targets

The active input file is:

- `input_data/ml-data.csv`

The dataset now contains both BnB time columns and total-time columns. The
pipeline uses only the BnB columns for:

- fastest-algorithm target construction
- sample weights
- split stratification targets
- evaluation metrics
- wrong-prediction and failure-case runtime comparisons

The total-time columns are kept in the dataframe when present, but they are not
used as supervised targets.

The active algorithm labels are:

- `kPEX-S-bnb`
- `kPEX-R-bnb`
- `kPEX-P-bnb`
- `kPEX-B-bnb`
- `UMKP-S-bnb`
- `UMKP-R-bnb`
- `UMKP-P-bnb`
- `UMKP+-S-bnb`
- `UMKP+-R-bnb`
- `UMKP+-P-bnb`

## Scheme 1, Scheme 2, And Scheme 3

- Scheme 1: flat 10-class classification across the BnB labels above
- Scheme 2: hierarchical classification with `Sparse` vs `Dense`, then
  `AltRB` vs `No-AltRB`, then branch-local leaf prediction
- Scheme 3: 3-stage hierarchical classification with one shared 4-class router

Current implementation status:

- Scheme 1 RF and XGB are runnable
- Scheme 2 RF and XGB are runnable
- Scheme 3 RF and XGB are runnable

## Scheme 3 Routing

Scheme 3 uses three stages:

1. Stage 1: `Sparse` vs `Dense`
2. Stage 2: if `Sparse`, `AltRB` vs `No-AltRB`
3. Stage 3: one router across `Class1`, `Class2`, `Class3`, `Class4`

Stage 3 class mapping:

- `Class1`: `UMKP-S-bnb`, `UMKP+-S-bnb`, `kPEX-S-bnb`
- `Class2`: `UMKP-R-bnb`, `UMKP+-R-bnb`, `kPEX-R-bnb`
- `Class3`: `UMKP-P-bnb`, `UMKP+-P-bnb`, `kPEX-P-bnb`
- `Class4`: `kPEX-B-bnb`

Final-output rule:

- if Stage 3 predicts `Class4`, the final output is `kPEX-B-bnb` directly
- otherwise the final algorithm is resolved from the router class plus the
  Stage 1 / Stage 2 path

## Run From Repo Root

Use `kpex_ML_git/` as the working directory.

If you are using the shared conda environment for experiments:

```bash
conda activate kpex
export MPLCONFIGDIR=/tmp/matplotlib
```

## Workflow At A Glance

The active saved-run workflow is:

1. Train once with `src/train.py --save-files`
2. Evaluate the held-out test split with
   `src/evaluate.py --inference-mode inference_test_set`
3. Evaluate the entire dataset with
   `src/evaluate.py --inference-mode inference_entire_data`
4. Export held-out test predictions and wrong cases with
   `src/predict.py --inference-mode inference_test_set`
5. Export full-dataset predictions and wrong cases with
   `src/predict.py --inference-mode inference_entire_data`
6. Compare wrong predictions either:
   - per run through `run_full_pipeline.sh`, or
   - directly from saved wrong-case workbooks through
     `src/compare_wrong_predictions.py`

Primary doc pointers:

- `restructured_pipeline/README.md`
- `restructured_pipeline/docs/comparison_wrong_predictions.md`
- `restructured_pipeline/agents.md`

## Output Locations

The active output directories come from `configs/project.yaml`:

- models:
  `restructured_pipeline/outputs/models/<scheme>/<model_family>/`
- predictions:
  `restructured_pipeline/outputs/predictions/<scheme>/<model_family>/`
- wrong cases:
  `restructured_pipeline/outputs/wrong_cases/<scheme>/<model_family>/`
- metrics and confusion matrices:
  `restructured_pipeline/outputs/metrics/<scheme>/<model_family>/`
- logs:
  `restructured_pipeline/outputs/logs/<scheme>/<model_family>/`
- split indices:
  `restructured_pipeline/data/splits/<scheme>/<model_family>/`
- comparison reports:
  `restructured_pipeline/outputs/reports/`

Important saved artifact names:

- training model artifact:
  `model_<RUN_NAME>.pkl`
- training predictions CSV:
  `all_predictions_<RUN_NAME>.csv`
- training wrong cases workbook:
  `wrong_predictions_<RUN_NAME>.xlsx`
- prediction CSV exports:
  `all_predictions_<RUN_NAME>_predict_inference_test_data.csv`
  and
  `all_predictions_<RUN_NAME>_predict_inference_entire_data.csv`
- prediction wrong-case exports:
  `wrong_predictions_<RUN_NAME>_predict_inference_test_data.xlsx`
  and
  `wrong_predictions_<RUN_NAME>_predict_inference_entire_data.xlsx`
- shared metrics workbook:
  `model_performance_for_all_exp.xlsx`

Scheme 2 / Scheme 3 training runs also write:

- `hierarchical_summary_report_<RUN_NAME>.xlsx`
- `hierarchical_per_class_report_<RUN_NAME>.xlsx`

## Naming Conventions

- If you pass `--file-suffix`, that value becomes the run name exactly.
- If you omit `--file-suffix`:
  - Scheme 1 default run names include the variant, for example
    `scheme1_random_forest_rf_strat_y`
  - Scheme 2 and Scheme 3 default run names include the split mode, for example
    `scheme2_random_forest_strat_y`
- Evaluation output names are:
  `<BASE_RUN_NAME>_eval_inference_test_data`
  or
  `<BASE_RUN_NAME>_eval_inference_entire_data`
- Prediction output names are:
  `<BASE_RUN_NAME>_predict_inference_test_data`
  or
  `<BASE_RUN_NAME>_predict_inference_entire_data`

Scheme 1 variant mapping used by the code and the wrapper:

- `random_forest + strat_y -> rf_strat_y`
- `random_forest + strat_y_K -> rf_strat_y_K`
- `xgboost + strat_y -> xgb_strat_y`
- `xgboost + strat_y_K -> xgb_strat_y_K`

## Test-Set Versus Entire-Dataset Outputs

- `inference_test_set`:
  uses the saved test indices from training and evaluates or exports only the
  held-out test subset.
- `inference_entire_data`:
  evaluates or exports every row in the current preprocessed dataset.
- `evaluate.py` and `predict.py` prefer the saved model artifact from training.
  If the model artifact is missing, they fall back to the saved
  `all_predictions_<RUN_NAME>.csv` file when possible.

## Training

Training reads `project.yaml` plus the scheme-specific model config, trains the
selected model, and optionally writes the saved model artifact, the base
full-dataset predictions CSV used for saved-run reuse, the training wrong-case
Excel file, the confusion matrix, the metrics workbook, saved split indices,
and for Scheme 2 / Scheme 3 the hierarchical stage-summary and per-class
report workbooks.

Example: Scheme 1 random forest

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme1 \
  --model-family random_forest \
  --variant rf_strat_y \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme1.yaml \
  --save-files
```

Example: Scheme 2 random forest

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme2 \
  --model-family random_forest \
  --split-mode strat_y \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme2.yaml \
  --save-files
```

Example: Scheme 2 XGB

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme2 \
  --model-family xgboost \
  --split-mode strat_y \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme2.yaml \
  --save-files
```

Example: Scheme 3 random forest

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme3 \
  --model-family random_forest \
  --split-mode strat_y \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme3.yaml \
  --save-files
```

Example: Scheme 3 XGB

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme3 \
  --model-family xgboost \
  --split-mode strat_y \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme3.yaml \
  --save-files
```

Notes:

- `--save-files` is required if you want to reuse the trained run from
  `evaluate.py` or `predict.py`
- use `--file-suffix` when you want an explicit reusable run name for the
  later evaluation and prediction commands
- with `--save-files`, training writes `model_<run_name>.pkl` plus
  `all_predictions_<run_name>.csv`
- the training-time `all_predictions_<run_name>.csv` file remains the
  full-dataset training export used for saved-run reuse
- training wrong-case exports are written as `wrong_predictions_<run_name>.xlsx`
  and the redundant `.csv` copy is no longer written
- training scores the fitted model on the full preprocessed dataset using BnB
  runtimes; use `evaluate.py --inference-mode inference_test_set` for held-out
  test evaluation
- For Scheme 1, use `--variant` to select a model variant from
  `models_scheme1.yaml`
- For Scheme 2 and Scheme 3, use `--split-mode` if you want to override the
  default split mode
- see `## Exact Command Matrix` below for all Scheme 1 / 2 / 3
  model-family and split-mode combinations

## Evaluation

`evaluate.py` evaluates a saved run on either the held-out test split or the
entire dataset. When a saved model artifact is available, it reloads the
trained model and recomputes predictions on the current preprocessed dataset.
Those predictions use the BnB runtime columns for the target and metric logic.
Saved evaluation outputs are limited to the confusion matrix plus the shared
metrics workbook entry for that run.

Use `--run-name <TRAIN_RUN_NAME>` to evaluate a previously saved run exactly.

Example: evaluate a saved Scheme 1 run on the held-out test set

```bash
python restructured_pipeline/src/evaluate.py \
  --scheme scheme1 \
  --model-family random_forest \
  --run-name scheme1_random_forest_rf_strat_y \
  --inference-mode inference_test_set \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme1.yaml
```

Example: evaluate a saved Scheme 2 run on the full dataset

```bash
python restructured_pipeline/src/evaluate.py \
  --scheme scheme2 \
  --model-family random_forest \
  --run-name scheme2_random_forest_strat_y \
  --inference-mode inference_entire_data \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme2.yaml
```

## Prediction Exports

`predict.py` can export either the held-out test subset or the full dataset for
a saved run. It prefers the saved model artifact from training and falls back
to the saved `all_predictions_<run_name>.csv` export when needed.

Current behavior:

- `predict.py --inference-mode inference_test_set` exports the held-out test
  subset for a saved run
- `predict.py --inference-mode inference_entire_data` exports predictions for
  the full dataset
- both exports keep the active BnB algorithm labels in `predicted`,
  `fastest_algorithms`, `prob_<algorithm>`, and wrong-case outputs
- wrong-case exports are written as Excel files only:
  `wrong_predictions_<RUN_NAME>.xlsx`
- helper-generated export names use `test_data` for held-out outputs and
  `entire_data` for full-dataset outputs
- direct prediction exports now default to:
  `all_predictions_<RUN_NAME>_predict_inference_test_data.csv` and
  `all_predictions_<RUN_NAME>_predict_inference_entire_data.csv`

Example: export full-dataset predictions from a saved Scheme 2 run

```bash
python restructured_pipeline/src/predict.py \
  --scheme scheme2 \
  --model-family random_forest \
  --run-name scheme2_random_forest_strat_y \
  --inference-mode inference_entire_data \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme2.yaml
```

Use `--run-name <TRAIN_RUN_NAME>` to export predictions from a previously
saved run exactly.

## Exact Command Matrix

The commands below use explicit `--file-suffix` values at training time and
reuse the same value through `--run-name` for evaluation and prediction.

Each combination includes:

- one training command
- one held-out test evaluation command
- one entire-dataset evaluation command
- one held-out test prediction command
- one entire-dataset prediction command

### Scheme 1

RF + `strat_y`

```bash
python restructured_pipeline/src/train.py --scheme scheme1 --model-family random_forest --variant rf_strat_y --file-suffix scheme1_random_forest_rf_strat_y_demo --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml --save-files
python restructured_pipeline/src/evaluate.py --scheme scheme1 --model-family random_forest --run-name scheme1_random_forest_rf_strat_y_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
python restructured_pipeline/src/evaluate.py --scheme scheme1 --model-family random_forest --run-name scheme1_random_forest_rf_strat_y_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
python restructured_pipeline/src/predict.py --scheme scheme1 --model-family random_forest --run-name scheme1_random_forest_rf_strat_y_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
python restructured_pipeline/src/predict.py --scheme scheme1 --model-family random_forest --run-name scheme1_random_forest_rf_strat_y_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
```

RF + `strat_y_K`

```bash
python restructured_pipeline/src/train.py --scheme scheme1 --model-family random_forest --variant rf_strat_y_K --file-suffix scheme1_random_forest_rf_strat_y_K_demo --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml --save-files
python restructured_pipeline/src/evaluate.py --scheme scheme1 --model-family random_forest --run-name scheme1_random_forest_rf_strat_y_K_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
python restructured_pipeline/src/evaluate.py --scheme scheme1 --model-family random_forest --run-name scheme1_random_forest_rf_strat_y_K_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
python restructured_pipeline/src/predict.py --scheme scheme1 --model-family random_forest --run-name scheme1_random_forest_rf_strat_y_K_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
python restructured_pipeline/src/predict.py --scheme scheme1 --model-family random_forest --run-name scheme1_random_forest_rf_strat_y_K_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
```

XGB + `strat_y`

```bash
python restructured_pipeline/src/train.py --scheme scheme1 --model-family xgboost --variant xgb_strat_y --file-suffix scheme1_xgboost_xgb_strat_y_demo --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml --save-files
python restructured_pipeline/src/evaluate.py --scheme scheme1 --model-family xgboost --run-name scheme1_xgboost_xgb_strat_y_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
python restructured_pipeline/src/evaluate.py --scheme scheme1 --model-family xgboost --run-name scheme1_xgboost_xgb_strat_y_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
python restructured_pipeline/src/predict.py --scheme scheme1 --model-family xgboost --run-name scheme1_xgboost_xgb_strat_y_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
python restructured_pipeline/src/predict.py --scheme scheme1 --model-family xgboost --run-name scheme1_xgboost_xgb_strat_y_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
```

XGB + `strat_y_K`

```bash
python restructured_pipeline/src/train.py --scheme scheme1 --model-family xgboost --variant xgb_strat_y_K --file-suffix scheme1_xgboost_xgb_strat_y_K_demo --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml --save-files
python restructured_pipeline/src/evaluate.py --scheme scheme1 --model-family xgboost --run-name scheme1_xgboost_xgb_strat_y_K_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
python restructured_pipeline/src/evaluate.py --scheme scheme1 --model-family xgboost --run-name scheme1_xgboost_xgb_strat_y_K_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
python restructured_pipeline/src/predict.py --scheme scheme1 --model-family xgboost --run-name scheme1_xgboost_xgb_strat_y_K_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
python restructured_pipeline/src/predict.py --scheme scheme1 --model-family xgboost --run-name scheme1_xgboost_xgb_strat_y_K_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme1.yaml
```

### Scheme 2

RF + `strat_y`

```bash
python restructured_pipeline/src/train.py --scheme scheme2 --model-family random_forest --split-mode strat_y --file-suffix scheme2_random_forest_strat_y_demo --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml --save-files
python restructured_pipeline/src/evaluate.py --scheme scheme2 --model-family random_forest --run-name scheme2_random_forest_strat_y_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
python restructured_pipeline/src/evaluate.py --scheme scheme2 --model-family random_forest --run-name scheme2_random_forest_strat_y_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
python restructured_pipeline/src/predict.py --scheme scheme2 --model-family random_forest --run-name scheme2_random_forest_strat_y_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
python restructured_pipeline/src/predict.py --scheme scheme2 --model-family random_forest --run-name scheme2_random_forest_strat_y_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
```

RF + `strat_y_K`

```bash
python restructured_pipeline/src/train.py --scheme scheme2 --model-family random_forest --split-mode strat_y_K --file-suffix scheme2_random_forest_strat_y_K_demo --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml --save-files
python restructured_pipeline/src/evaluate.py --scheme scheme2 --model-family random_forest --run-name scheme2_random_forest_strat_y_K_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
python restructured_pipeline/src/evaluate.py --scheme scheme2 --model-family random_forest --run-name scheme2_random_forest_strat_y_K_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
python restructured_pipeline/src/predict.py --scheme scheme2 --model-family random_forest --run-name scheme2_random_forest_strat_y_K_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
python restructured_pipeline/src/predict.py --scheme scheme2 --model-family random_forest --run-name scheme2_random_forest_strat_y_K_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
```

XGB + `strat_y`

```bash
python restructured_pipeline/src/train.py --scheme scheme2 --model-family xgboost --split-mode strat_y --file-suffix scheme2_xgboost_strat_y_demo --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml --save-files
python restructured_pipeline/src/evaluate.py --scheme scheme2 --model-family xgboost --run-name scheme2_xgboost_strat_y_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
python restructured_pipeline/src/evaluate.py --scheme scheme2 --model-family xgboost --run-name scheme2_xgboost_strat_y_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
python restructured_pipeline/src/predict.py --scheme scheme2 --model-family xgboost --run-name scheme2_xgboost_strat_y_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
python restructured_pipeline/src/predict.py --scheme scheme2 --model-family xgboost --run-name scheme2_xgboost_strat_y_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
```

XGB + `strat_y_K`

```bash
python restructured_pipeline/src/train.py --scheme scheme2 --model-family xgboost --split-mode strat_y_K --file-suffix scheme2_xgboost_strat_y_K_demo --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml --save-files
python restructured_pipeline/src/evaluate.py --scheme scheme2 --model-family xgboost --run-name scheme2_xgboost_strat_y_K_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
python restructured_pipeline/src/evaluate.py --scheme scheme2 --model-family xgboost --run-name scheme2_xgboost_strat_y_K_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
python restructured_pipeline/src/predict.py --scheme scheme2 --model-family xgboost --run-name scheme2_xgboost_strat_y_K_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
python restructured_pipeline/src/predict.py --scheme scheme2 --model-family xgboost --run-name scheme2_xgboost_strat_y_K_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme2.yaml
```

### Scheme 3

RF + `strat_y`

```bash
python restructured_pipeline/src/train.py --scheme scheme3 --model-family random_forest --split-mode strat_y --file-suffix scheme3_random_forest_strat_y_demo --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml --save-files
python restructured_pipeline/src/evaluate.py --scheme scheme3 --model-family random_forest --run-name scheme3_random_forest_strat_y_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
python restructured_pipeline/src/evaluate.py --scheme scheme3 --model-family random_forest --run-name scheme3_random_forest_strat_y_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
python restructured_pipeline/src/predict.py --scheme scheme3 --model-family random_forest --run-name scheme3_random_forest_strat_y_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
python restructured_pipeline/src/predict.py --scheme scheme3 --model-family random_forest --run-name scheme3_random_forest_strat_y_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
```

RF + `strat_y_K`

```bash
python restructured_pipeline/src/train.py --scheme scheme3 --model-family random_forest --split-mode strat_y_K --file-suffix scheme3_random_forest_strat_y_K_demo --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml --save-files
python restructured_pipeline/src/evaluate.py --scheme scheme3 --model-family random_forest --run-name scheme3_random_forest_strat_y_K_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
python restructured_pipeline/src/evaluate.py --scheme scheme3 --model-family random_forest --run-name scheme3_random_forest_strat_y_K_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
python restructured_pipeline/src/predict.py --scheme scheme3 --model-family random_forest --run-name scheme3_random_forest_strat_y_K_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
python restructured_pipeline/src/predict.py --scheme scheme3 --model-family random_forest --run-name scheme3_random_forest_strat_y_K_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
```

XGB + `strat_y`

```bash
python restructured_pipeline/src/train.py --scheme scheme3 --model-family xgboost --split-mode strat_y --file-suffix scheme3_xgboost_strat_y_demo --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml --save-files
python restructured_pipeline/src/evaluate.py --scheme scheme3 --model-family xgboost --run-name scheme3_xgboost_strat_y_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
python restructured_pipeline/src/evaluate.py --scheme scheme3 --model-family xgboost --run-name scheme3_xgboost_strat_y_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
python restructured_pipeline/src/predict.py --scheme scheme3 --model-family xgboost --run-name scheme3_xgboost_strat_y_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
python restructured_pipeline/src/predict.py --scheme scheme3 --model-family xgboost --run-name scheme3_xgboost_strat_y_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
```

XGB + `strat_y_K`

```bash
python restructured_pipeline/src/train.py --scheme scheme3 --model-family xgboost --split-mode strat_y_K --file-suffix scheme3_xgboost_strat_y_K_demo --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml --save-files
python restructured_pipeline/src/evaluate.py --scheme scheme3 --model-family xgboost --run-name scheme3_xgboost_strat_y_K_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
python restructured_pipeline/src/evaluate.py --scheme scheme3 --model-family xgboost --run-name scheme3_xgboost_strat_y_K_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
python restructured_pipeline/src/predict.py --scheme scheme3 --model-family xgboost --run-name scheme3_xgboost_strat_y_K_demo --inference-mode inference_test_set --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
python restructured_pipeline/src/predict.py --scheme scheme3 --model-family xgboost --run-name scheme3_xgboost_strat_y_K_demo --inference-mode inference_entire_data --project-config restructured_pipeline/configs/project.yaml --model-config restructured_pipeline/configs/models_scheme3.yaml
```

## End-To-End Workflow

The intended saved-run workflow is:

1. Train once with `train.py --save-files`
2. Evaluate the held-out test split with
   `evaluate.py --inference-mode inference_test_set`
3. Export held-out test predictions with
   `predict.py --inference-mode inference_test_set`
4. Export full-dataset predictions with
   `predict.py --inference-mode inference_entire_data`

The bash wrapper runs that end-to-end flow in one command and emits both
prediction export files automatically:

- `all_predictions_<RUN_NAME>_predict_inference_test_data.csv`
- `all_predictions_<RUN_NAME>_predict_inference_entire_data.csv`

Wrapper note:

- for Scheme 1, `run_full_pipeline.sh` resolves the variant automatically from
  `--model-family + --split-mode`
- the wrapper stages a per-run wrong-predictions comparison report; the
  consolidated all-schemes full-dataset comparison is a separate direct
  `compare_wrong_predictions.py --discover-entire-data-files` workflow

Related naming rule:

- held-out evaluation/report outputs use `inference_test_data` in generated
  filenames
- held-out prediction export outputs use `inference_test_data` in generated
  filenames
- full-dataset evaluation/prediction outputs use `inference_entire_data` in
  generated filenames
- this naming convention is shared across Scheme 1, Scheme 2, and Scheme 3

Example:

```bash
bash restructured_pipeline/run_full_pipeline.sh \
  --scheme scheme2 \
  --model-family xgboost \
  --split-mode strat_y_K \
  --run-name scheme2_xgboost_strat_y_K_wrapper_demo \
  --eval-inference-mode inference_test_set
```

## Experiment Matrix

`src/experiment_matrix.py` runs the full batch experiment matrix across:

- schemes: `scheme1`, `scheme2`, `scheme3`
- model families: `random_forest`, `xgboost`
- split modes: `strat_y`, `strat_y_K`
- inference modes: held-out `test_data` and `entire_data`

For each trained base run, the matrix runner:

1. trains with `train.py --save-files`
2. evaluates on the held-out test split
3. evaluates on the entire dataset
4. exports held-out prediction files
5. exports entire-dataset prediction files
6. writes hierarchical stage-summary and per-class report workbooks for
   Scheme 2 and Scheme 3 training runs
7. writes one performance workbook and one tolerance workbook per
   `scheme/model_family`

Main workbook output:

- `restructured_pipeline/outputs/metrics/<scheme>/<model_family>/model_performance_for_all_exp_<MATRIX_RUN_NAME>.xlsx`
- sheet names:
  - `model_performance_for_all_exp`
  - `strat_y_test_data`
  - `strat_y__K_test_data`
  - `strat_y_entire_data`
  - `strat_y__K_entire_data`

Each final performance workbook contains exactly these five sheets.

Tolerance workbook output:

- `restructured_pipeline/outputs/metrics/<scheme>/<model_family>/model_performance_for_all_exp_tolerance_<MATRIX_RUN_NAME>.xlsx`
- sheet names:
  - `tolerance_eval_all_exp`
  - the same experiment sheets listed above

Each final tolerance workbook contains exactly these five sheets. Temporary
per-tolerance staging sheets are consolidated and removed.

Tolerance rows use the configured values from `project.yaml` and reuse the
shared tolerance metric logic already used by the pipeline:

- `0.0`
- `0.1`
- `0.2`
- `0.5`

Important split note:

- `strat_y_K` still prefers `target + K` stratification
- if a combined `target + K` class would make stratification fail, those row(s)
  are forced into the training split first and the remaining rows still use true
  `target + K` stratification

Run the Python matrix runner directly:

```bash
python restructured_pipeline/src/experiment_matrix.py \
  --matrix-run-name april_matrix
```

Run the same workflow through the bash wrapper:

```bash
restructured_pipeline/run_full_pipeline.sh \
  --full-experiment-matrix \
  --matrix-run-name april_matrix
```

## Reduced Artifact Set

The active pipeline now avoids redundant file writes during training,
evaluation, prediction export, and matrix runs.

Removed redundant outputs:

- duplicate wrong-prediction CSV exports alongside the active Excel files
- per-run classification-report CSV files
- per-run metrics JSON sidecars
- evaluation and prediction JSON log files
- temporary tolerance sheets left behind in the final tolerance workbook

Kept outputs:

- saved model artifacts and split indices needed for saved-run reuse
- prediction CSV files for held-out and entire-data inference
- wrong-prediction Excel files
- confusion matrices
- Scheme 2 and Scheme 3 stage summary / per-class Excel reports
- one performance workbook per `scheme/model_family`
- one tolerance workbook per `scheme/model_family`

## Compare Wrong Predictions

`compare_wrong_predictions.py` is the Python version of the old
`compare_wrong_predictions.ipynb` workflow. It reads
`wrong_predictions_*.xlsx` files, collects failure rows that meet a
time-gap threshold, adds comparison columns, and writes one Excel workbook with
one sheet per `inference_on`.

It supports these input modes:

- staged model roots via `--model-root NAME=PATH`
- direct saved wrong-predictions workbooks via
  `--wrong-predictions-file [NAME=]PATH`
- direct full-dataset auto-discovery via `--discover-entire-data-files`

Example: analyze one saved wrong-predictions workbook directly

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --wrong-predictions-file restructured_pipeline/outputs/wrong_cases/scheme2/random_forest/wrong_predictions_scheme2_random_forest_strat_y.xlsx
```

Example: build the consolidated full-dataset workbook across all saved scheme
runs on disk

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --discover-entire-data-files \
  --wrong-cases-root restructured_pipeline/outputs/wrong_cases \
  --threshold 0.5 \
  --time-gap-highlight-threshold 100
```

Important runtime note:

- wrong-case and failure-case runtime comparisons are computed from the active
  BnB algorithm columns
- row filtering uses `time_gap_ratio >= --threshold`
- yellow fill is applied only to the `time_gap` cell when
  `time_gap > --time-gap-highlight-threshold`
- total-time columns may still appear in saved CSV exports, but they are not
  used to define `fastest_algo`, `fastest_runtime`, `predicted_time`, or
  `time_gap_ratio`
- full comparison details are documented in
  `restructured_pipeline/docs/comparison_wrong_predictions.md`

### Wrong Predictions Analysis (Unified)

`compare_wrong_predictions.py` is the single entry point for wrong-predictions
analysis. It combines:

- comparison logic:
  discovery, loading, threshold filtering, metadata derivation, and
  failure-case column construction
- reporting logic:
  Excel writing, sheet generation by `inference_on`, autosizing, severity
  fills, and yellow `time_gap` highlighting

Current behavior:

- `--mode compare`:
  runs the full compare + report flow
- `--mode report`:
  reapplies the Excel formatting/writing layer to an existing workbook
- `top3_predicted_detailed` is written in a canonical rounded format:
  probabilities and time ratios use 4 decimal places

Example:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --discover-entire-data-files \
  --wrong-cases-root restructured_pipeline/outputs/wrong_cases \
  --threshold 0.5 \
  --time-gap-highlight-threshold 100 \
  --output-path restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data.xlsx
```

Example: formatting-only mode

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --mode report \
  --input-workbook restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data.xlsx \
  --output-path restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data.xlsx
```

Generated outputs:

- formatted compare workbook by `inference_on`
- severity-based row fills from `failure_severity`
- yellow highlight on `time_gap` cells above
  `--time-gap-highlight-threshold`
- canonical `top3_predicted_detailed` formatting with 4-decimal rounding

### Consolidated Tolerance Wrong Predictions Report

`compare_wrong_predictions.py` can also write a second workbook that applies
tolerance-based wrong-prediction filtering across all schemes and models.

Tolerance definition:

- compute `tolerance_threshold = fastest_runtime * (1 + tolerance / 100)`
- keep rows where `predicted_time > tolerance_threshold`
- supported tolerances:
  `0`, `10`, `20`, `50`

Command:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --discover-entire-data-files \
  --wrong-cases-root restructured_pipeline/outputs/wrong_cases \
  --generate-tolerance-report
```

Optional explicit output path:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --discover-entire-data-files \
  --wrong-cases-root restructured_pipeline/outputs/wrong_cases \
  --generate-tolerance-report \
  --tolerance-output-path restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all.xlsx
```

Output:

- `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all.xlsx`

Sheet naming:

- `scheme_1_tol_0`
- `scheme_1_tol_10`
- `scheme_1_tol_20`
- `scheme_1_tol_50`
- `scheme_2_tol_0`
- `scheme_2_tol_10`
- `scheme_2_tol_20`
- `scheme_2_tol_50`
- `scheme_3_tol_0`
- `scheme_3_tol_10`
- `scheme_3_tol_20`
- `scheme_3_tol_50`

Sheet contents:

- one sheet per `(scheme, tolerance)` pair
- rows are filtered independently per tolerance from the dataframe supplied to
  the tolerance writer
- the normal compare workbook is still written unchanged
- `top3_predicted_detailed` remains rounded to 4 decimals
- formatting reuses the standard reporting layer:
  severity row fills plus the `time_gap` cell highlight rule

### Correct Pipeline Order

1. Build the strict compare dataset with `time_gap_ratio >= --threshold`
2. Apply tolerance filtering on top of that strict dataset
3. Optionally apply important-failure filtering on the tolerance workbook

Dedicated pipeline note:

- the standalone `--generate-tolerance-report` path still operates on the
  already-built compare dataframe
- the dedicated `--run-test-data-pipeline` and `--run-entire-data-pipeline`
  paths rebuild the tolerance base directly from saved wrong-predictions files
  with `threshold = 0.0`, then apply the tolerance masks independently
- this keeps the compare workbook unchanged while ensuring the tolerance sheets
  are not forced to reuse the already-strict subset

### Important Failure Filtering (Tolerance Reports)

`compare_wrong_predictions.py` can also perform a second-stage filter on an
existing tolerance workbook and write an important-failures workbook with the
same 12 scheme/tolerance sheets.

Filtering rules:

- exclude rows where:
  `fastest_runtime < 5.0` and `predicted_runtime < 5.0`
- exclude rows where:
  `predicted_runtime <= 1.2 * fastest_runtime`
- runtime-column fallback:
  use `fastest_time` and `predicted_time` if the runtime-named columns are not
  present

Command:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --input-tolerance-report restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all.xlsx \
  --generate-important-failure-report
```

Optional explicit output path:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --input-tolerance-report restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all.xlsx \
  --generate-important-failure-report \
  --important-failure-output-path restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_important_failures.xlsx
```

Output:

- `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_important_failures.xlsx`

Workbook contents:

- same 12 scheme/tolerance sheet names as the tolerance input workbook
- original column order preserved per sheet
- optional summary sheet:
  `filter_summary`
- `top3_predicted_detailed` remains rounded to 4 decimals
- the important-failure stage must consume the tolerance workbook after the
  correct per-tolerance filtering step

### Test Data Wrong Predictions Pipeline

`compare_wrong_predictions.py` can also run the full reporting pipeline for
held-out test data only.

Command:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --run-test-data-pipeline
```

Outputs:

- `restructured_pipeline/outputs/reports/test_data/compare_wrong_predictions_test_data.xlsx`
- `restructured_pipeline/outputs/reports/test_data/compare_wrong_predictions_tolerance_report_test_data.xlsx`
- `restructured_pipeline/outputs/reports/test_data/compare_wrong_predictions_tolerance_important_failures_test_data.xlsx`

Notes:

- this mirrors the entire-dataset pipeline in the same order:
  compare -> tolerance -> important failures
- it uses only saved `predict_inference_test_data` wrong-predictions files
- it writes only inside `restructured_pipeline/outputs/reports/test_data/`
- it is intended for held-out evaluation analysis and does not mix in
  entire-dataset rows
- the compare workbook uses the configured strict threshold
- the tolerance workbook is rebuilt from the full saved test-data
  wrong-predictions pool with `threshold = 0.0`, then filtered independently
  at `0`, `10`, `20`, and `50`

### Entire Data Wrong Predictions Pipeline

`compare_wrong_predictions.py` can also run the same three-stage reporting
pipeline for entire-dataset wrong-predictions files only.

Command:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --run-entire-data-pipeline
```

Outputs:

- `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx`
- `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_tolerance_report_entire_data.xlsx`
- `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx`

Notes:

- this mirrors the same order:
  compare -> tolerance -> important failures
- it uses only saved `predict_inference_entire_data` wrong-predictions files
- it writes only inside `restructured_pipeline/outputs/reports/entire_data/`
- it does not mix in held-out test rows
- the compare workbook uses the configured strict threshold
- the tolerance workbook is rebuilt from the full saved entire-data
  wrong-predictions pool with `threshold = 0.0`, then filtered independently
  at `0`, `10`, `20`, and `50`

## Failure Analysis Enrichment

`failure_analysis_summary.py` now supports a second reporting-only mode that
builds a cleaner entire-dataset failure workbook without changing any
training, prediction, or core compare logic.

Command:

```bash
python restructured_pipeline/src/failure_analysis_summary.py \
  --mode enrich
```

Main workbook:

- default input workbook:
  `restructured_pipeline/outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data.xlsx`
- default output workbook:
  `restructured_pipeline/outputs/reports/failure_analysis_summary_all_scheme_entire_data.xlsx`
- each run rewrites this six-sheet workbook from scratch
- exact sheet set:
  - `scheme_1_RF`
  - `scheme_1_XGB`
  - `scheme_2_RF`
  - `scheme_2_XGB`
  - `scheme_3_RF`
  - `scheme_3_XGB`
- this workbook is entire-dataset only
- no tolerance sheets are generated
- no row or cell highlighting is applied

Top-K analysis columns:

- for each `K = 1..6`, the workbook writes:
  - `top<K>_min_time_ratio`
  - `top<K>_candidate`
- `top<K>_candidate` stores the best-runtime candidate found within the first
  `K` probability-sorted predictions
- tuple format:
  `(predicted_time, predicted_algo, prob)`
- `top<K>_min_time_ratio` uses the same candidate and is computed as:
  `predicted_time / fastest_time`
- `all_predictions_sorted_by_prob` stores the full probability ranking as one
  compact tuple list:
  `(time_ratio, predicted_algo, predicted_time, prob)`
- `fastest_algo_tuple` stores:
  `(fastest_time, fastest_algo, rank)`
- `rank` is the 1-based position where the fastest algorithm first appears in
  the probability-sorted prediction order

Removed from this workbook:

- tolerance sheets and tolerance-sheet metadata
- highlighting rules based on `time_gap`, `time_gap_ratio`, or recovery
  thresholds
- `inference_on`, `with_holdout`, `strat_on`, `scheme`,
  `source_model`, `failure_case_filter_reason`,
  `fastest_algo`, `fastest_time`,
  `predicted_algo`, `predicted_time`,
  `predicted_probability`, `actual_time_ratio`,
  `failure_severity`, `time_gap`, `time_gap_ratio`
- all `prob_...` export columns

### Mixed Failure-Case Combinations

`mixed_failure_case_combinations.py` builds a reporting-only workbook that
mixes postprocessed failure rows across allowed scheme pairs.

Command:

```bash
python restructured_pipeline/src/mixed_failure_case_combinations.py
```

Default output:

- `restructured_pipeline/outputs/reports/mixed_failure_case_combinations.xlsx`

Input mapping:

- master split/source workbook:
  `restructured_pipeline/outputs/reports/existing_code_run_20260511/failure_analysis_summary_unfiltered_with_generated_filtered_reference.xlsx`
- filtered source workbook:
  `restructured_pipeline/outputs/reports/existing_code_run_20260511/failure_analysis_summary_filtered_all_scheme_entire_data_from_one_sheet.xlsx`
- the master workbook is used only to map `source_split_label`
  such as `strat_y` or `strat_y_K` back onto the filtered rows

Allowed pairings and rules:

- `scheme 1` with `scheme 2`
- `scheme 1` with `scheme 3`
- `scheme 2` and `scheme 3` must not be paired directly
- base pairwise sheets use full names such as:
  - `scheme1_top1__scheme2_top1`
  - `scheme1_top1__scheme2_top2`
  - `scheme1_top2__scheme2_top1`
  - `scheme1_top2__scheme2_top2`
  - `scheme1_top1__scheme3_top1`
  - `scheme1_top1__scheme3_top2`
  - `scheme1_top2__scheme3_top1`
  - `scheme1_top2__scheme3_top2`
- top-3 base variants are also generated:
  - `scheme1_top1__scheme2_top3`
  - `scheme1_top2__scheme2_top3`
  - `scheme1_top3__scheme2_top1`
  - `scheme1_top3__scheme2_top2`
  - same pattern for `scheme 3`
- threshold variants use compact names to stay within Excel's 31-character
  sheet-name limit:
  - `s1_t1__s2_t1__thr1p5`
  - `s1_t1__s2_t1__thr2p0`
- split-aware variants use compact suffixes:
  - `__straty`
  - `__stratyK`
  - `__entire`
- model-aware variants use:
  - `__RF`
  - `__XGB`
- the current source workbooks are entire-data only, so no `test_data` sheets
  are generated
- for each source row, the selected candidate is taken from:
  - `top1_candidate` when the rule uses `top1`
  - `top2_candidate` when the rule uses `top2`
  - `top3_candidate` when the rule uses `top3`
- rows are kept only when the selected minimum ratio is strictly greater than
  the active threshold, currently `1.5` or `2.0` depending on the sheet family

Recovered columns:

- `fastest_algo` and `fastest_runtime` are parsed from `fastest_algo_tuple`
- `predicted_algo`, `predicted_time`, and `predicted_probability` are parsed
  from the selected `top1_candidate` or `top2_candidate`
- `top3_predicted_detailed` is rebuilt from the first three entries in
  `all_predictions_sorted_by_prob`

### XGB-Only Mixed Failure-Case Workbook

`mixed_failure_case_combinations_xgb_only.py` builds a narrower workbook that
uses only the filtered `XGB` failure-analysis sheets and writes only the base
pairwise combinations plus the top-3 variants.

Command:

```bash
python restructured_pipeline/src/mixed_failure_case_combinations_xgb_only.py
```

Default output:

- `restructured_pipeline/outputs/reports/mixed_failure_case_combinations_xgb_only.xlsx`

Rules:

- source workbook:
  `failure_analysis_summary_filtered_all_scheme_entire_data_from_one_sheet.xlsx`
- only `scheme_1_XGB`, `scheme_2_XGB`, and `scheme_3_XGB` are read
- allowed pairings only:
  - `scheme 1 + scheme 2`
  - `scheme 1 + scheme 3`
- forbidden pairing:
  - `scheme 2 + scheme 3`
- threshold filtering is applied before rows are written
- rows are also restricted to:
  `strat_y_K`
- active filtering rule:
  for each sheet side, use the top-k named in that sheet and require:
  `selected topK_min_time_ratio > 1.20`
- example:
  `scheme1_top3__scheme2_top1` means
  `scheme 1` rows must satisfy `top3_min_time_ratio > 1.20`
  and `scheme 2` rows must satisfy `top1_min_time_ratio > 1.20`
- this rule is applied consistently across all generated sheets
- no split-aware variants are generated
- no `RF` rows are included
- a `summary` sheet is written first with:
  `sheet_name`, `scheme_pair`, `topk_pairing`, `row_count`,
  `pairing_family`, `rows_before_filter`, `rows_after_filter`,
  `rows_removed`, `min_ratio_threshold`, `required_split_label`,
  and `filter_rule`
- a second reporting-only sheet is also written:
  `shared_failure_summary`
- a shared failure means both source schemes in the same mixed sheet contain
  the same `(dataset, k_value)` pair
- rows that belong to those overlap cases are highlighted yellow across the
  full row in the affected mixed sheet
- `shared_failure_summary` reports:
  `sheet_name`, `scheme_pair`, `total_rows`, `shared_failure_pairs`,
  `highlighted_rows`, `unique_datasets_with_shared_failures`,
  `unique_dataset_k_overlap_cases`, and the most repeated overlap pairs

Generated sheets:

- base top1/top2 pairings:
  - `scheme1_top1__scheme2_top1`
  - `scheme1_top1__scheme2_top2`
  - `scheme1_top2__scheme2_top1`
  - `scheme1_top2__scheme2_top2`
  - `scheme1_top1__scheme3_top1`
  - `scheme1_top1__scheme3_top2`
  - `scheme1_top2__scheme3_top1`
  - `scheme1_top2__scheme3_top2`
- top-3 variants:
  - `scheme1_top3__scheme2_top1`
  - `scheme1_top3__scheme2_top2`
  - `scheme1_top2__scheme2_top3`
  - `scheme1_top1__scheme2_top3`
  - `scheme1_top1__scheme3_top3`
  - `scheme1_top2__scheme3_top3`
  - `scheme1_top3__scheme3_top1`
  - `scheme1_top3__scheme3_top2`

Exact output columns on every sheet:

- `dataset`
- `k_value`
- `scheme`
- `fastest_algo_rank`
- `fastest_algo_tuple`
- `top1_min_time_ratio`
- `top1_candidate`
- `top2_min_time_ratio`
- `top2_candidate`
- `top3_min_time_ratio`
- `top3_candidate`
- `top4_min_time_ratio`
- `top4_candidate`
- `top5_min_time_ratio`
- `top5_candidate`
- `top6_min_time_ratio`
- `top6_candidate`
- `all_predictions_sorted_by_prob`
- `source_scheme`
- `source_rule`

### Failure Counts By Split, Scheme, And Model

`failure_counts_by_split_scheme_model.py` builds a reporting-only workbook that
separates failures by split mode, scheme, model, and partition.

Command:

```bash
python restructured_pipeline/src/failure_counts_by_split_scheme_model.py
```

Default output:

- `restructured_pipeline/outputs/reports/failure_counts_by_split_scheme_model.xlsx`

Behavior:

- `important_failures_split_counts` is built from the filtered one-sheet
  important-failure workbook and matched back to the normalized wrong-case
  exports to recover:
  - `split_mode`: `strat_y` vs `strat_y_K`
  - `partition`: `train` vs `test`
- `all_wrong_split_counts` is built from all saved wrong-prediction exports,
  not just important failures
- training counts are derived as:
  `entire_data wrong predictions - test_data wrong predictions`
- unknown rows are retained through the `unknown_partition` column if a split
  or partition cannot be resolved

Workbook sheets:

- `important_failures_split_counts`
- `all_wrong_split_counts`
- `important_by_scheme_model`
- `important_by_split_scheme`
- `all_wrong_by_scheme_model`
- `all_wrong_by_split_scheme`

### Failure Train/Test Split Count Summary

`failure_train_test_split_count_summary.py` builds a second reporting-only
workbook that combines:

- important-failure counts from the filtered one-sheet workbook
- all wrong-prediction counts from saved wrong-case exports
- train/test breakdowns for the corrected XGB mixed-combination workbook

Command:

```bash
python restructured_pipeline/src/failure_train_test_split_count_summary.py
```

Default output:

- `restructured_pipeline/outputs/reports/failure_train_test_split_count_summary.xlsx`

Workbook sheets:

- `important_failure_counts`
- `all_wrong_predictions_counts`
- `mixed_combo_train_test_counts`
- `analysis_notes`
- `pivot_summary`
- `important_by_scheme_model`
- `important_by_split_scheme`
- `all_wrong_by_scheme_model`
- `all_wrong_by_split_scheme`

Notes:

- if the root-level filtered one-sheet workbook is missing, the script falls
  back to the saved copy under `existing_code_run_20260511/`
- train/test origin is derived by matching entire-data rows against saved
  `predict_inference_test_data` wrong-prediction exports
- mixed-combination counts use
  `mixed_failure_case_combinations_xgb_only_corrected.xlsx`
  and treat the workbook `summary` sheet as the summary source when a
  `failure case summary` sheet is not present

### XGB strat_y_K Train/Test Failure Analysis

`xgb_stratyk_train_test_failure_analysis.py` builds a narrower report for
`XGB + strat_y_K` important failures and produces a highlighted copy of the
corrected mixed-combination workbook.

Command:

```bash
python restructured_pipeline/src/xgb_stratyk_train_test_failure_analysis.py
```

Outputs:

- `restructured_pipeline/outputs/reports/xgb_strat_yK_train_test_failure_counts.xlsx`
- `restructured_pipeline/outputs/reports/mixed_failure_case_combinations_xgb_only_corrected_train_highlighted.xlsx`

Behavior:

- filters important failures to:
  `model == XGB` and `split_mode == strat_y_K`
- derives train/test origin using the same saved wrong-case matching logic as
  the broader split-count report
- copies the corrected mixed-combination workbook and highlights only
  training-side rows in light yellow
- adds `train_test_failure_summary` to the copied workbook
