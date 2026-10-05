# Scheme Guide

This guide explains how the active runnable pipeline under
`restructured_pipeline/` handles Scheme 1, Scheme 2, and Scheme 3 after the
dataset update to `ml-data.csv`.

## Core Dataset Rules

### Input file

The primary dataset path is:

- `input_data/ml-data.csv`

### Runtime columns

The dataset includes both:

- BnB runtime columns
- total-time columns

The pipeline uses only the BnB runtime columns for target creation, split
logic, model training, evaluation, prediction scoring, wrong-case generation,
and failure-case reporting.

The active BnB algorithm labels are:

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

The total-time columns may still remain in the dataframe and saved CSV files,
but they are not used as supervised targets.

## Main Files And Their Roles

### Shared pipeline files

- `restructured_pipeline/configs/project.yaml`
  Shared dataset path, BnB target columns, feature lists, split rules,
  tolerances, and output locations.
- `restructured_pipeline/configs/models_scheme1.yaml`
  Scheme 1 model families, variants, and hyperparameters.
- `restructured_pipeline/configs/models_scheme2.yaml`
  Scheme 2 stage-wise model settings.
- `restructured_pipeline/configs/models_scheme3.yaml`
  Scheme 3 stage-wise model settings.
- `restructured_pipeline/src/features.py`
  Data loading, preprocessing, BnB target derivation, and stage-weight logic.
- `restructured_pipeline/src/data_splits.py`
  Train/test split helpers.
- `restructured_pipeline/src/metrics.py`
  Classification and runtime metrics.
- `restructured_pipeline/src/write_utils.py`
  Shared report helpers and wrong-case utilities.

### Scheme-specific model files

- `restructured_pipeline/src/models/flat.py`
  Scheme 1 model wrappers.
- `restructured_pipeline/src/models/hierarchical.py`
  Scheme 2 staged classifier and evaluation logic.
- `restructured_pipeline/src/models/hierarchical_scheme3.py`
  Scheme 3 staged classifier and evaluation logic.

### Pipeline entry points

- `restructured_pipeline/src/train.py`
- `restructured_pipeline/src/evaluate.py`
- `restructured_pipeline/src/predict.py`
- `restructured_pipeline/src/compare_wrong_predictions.py`
- `restructured_pipeline/run_full_pipeline.sh`

## Shared Input Preparation

The common preprocessing flow lives in `src/features.py` and is used by
`train.py -> load_and_preprocess_data`.

### Loading and cleanup

`load_excel_data` currently:

- reads `ml-data.csv` as a comma-separated file
- renames legacy structural columns:
  - `shell_sizes -> kcore_list`
  - `core_number -> core_max`
  - `k -> k_value`
- replaces timeout marker `x` with `1800` inside the configured BnB runtime
  columns
- removes rows where all configured BnB runtime columns are timeouts
- converts the configured BnB runtime columns to numeric float columns

### Feature engineering

`process_kcore_list_to_statistics` derives:

- `shell_num`
- `core_min`
- `shell_avgSize`
- `shell_maxSize`
- `shell_minSize`
- `shell_stdSize`

The main feature list remains:

- `k_value`
- `n`
- `m`
- `d_avg`
- `d_max`
- `core_max`
- `core_min`
- `shell_num`
- `shell_avgSize`
- `shell_maxSize`
- `shell_minSize`
- `shell_stdSize`

### Base weights

`assign_weights` still computes the shared `weight` column with
`W_TIME_RATIO`, but the calculation now uses only the BnB runtime columns.

## Scheme 1

### Target definition

Scheme 1 is a flat 10-class classifier over the active BnB labels.

The target column is:

- `target`

It is created during split preparation by expanding each row across all
fastest-within-tolerance BnB labels.

### Split logic

Scheme 1 uses:

- `prepare_train_test_strat_Y`
- `prepare_train_test_strat_Y_K`

Those helpers:

1. read the 10 BnB runtime columns
2. find the minimum BnB runtime in the row
3. collect all algorithms within the tie tolerance
4. create one expanded row per tied algorithm
5. set:
   - `target = alg`
   - `weight = original_weight / number_of_tied_algs`

### Training and inference

Scheme 1 training lives in:

- `train.py -> train_scheme1`
- `models/flat.py`

Inference is built by `build_flat_results` in `train.py`, which:

- recomputes `fastest_algorithms` from the BnB columns
- predicts on the feature matrix
- writes `predicted`, `correct`, `prob_<algorithm>`, and `top3_predicted`

## Scheme 2

### Target definition

Scheme 2 adds hierarchical targets in `assign_scheme2_targets_and_weights`:

- `y_algo`
- `y_group`
- `y_sparse_subgroup`

They are derived from the BnB fastest algorithm:

- `y_algo`
  The single fastest BnB label from the 10 BnB runtime columns
- `y_group`
  `Dense` for `kPEX-S-bnb`, `kPEX-R-bnb`, `kPEX-P-bnb`, `kPEX-B-bnb`;
  otherwise `Sparse`
- `y_sparse_subgroup`
  `AltRB` for `UMKP+-*-bnb`, `No-AltRB` for `UMKP-*-bnb`, and `None` for dense
  rows

### Stage structure

Scheme 2 factorizes the final decision into:

1. Stage 1: `Sparse` vs `Dense`
2. Stage 2: if `Sparse`, `AltRB` vs `No-AltRB`
3. Stage 3: if `AltRB`, one of:
   - `UMKP+-S-bnb`
   - `UMKP+-R-bnb`
   - `UMKP+-P-bnb`
4. Stage 4: if `No-AltRB`, one of:
   - `UMKP-S-bnb`
   - `UMKP-R-bnb`
   - `UMKP-P-bnb`
5. Stage 5: if `Dense`, one of:
   - `kPEX-S-bnb`
   - `kPEX-R-bnb`
   - `kPEX-P-bnb`
   - `kPEX-B-bnb`

### Stage weights

Scheme 2 still computes:

- `weight_group`
- `weight_sparse_subgroup`
- `weight_altRB`
- `weight_noAltRB`
- `weight_dense`

All of those weight calculations now depend on the BnB runtime columns only.

### Training and inference

Scheme 2 training lives in:

- `train.py -> train_scheme2`
- `models/hierarchical.py`

The final Scheme 2 prediction is always one of the 10 BnB labels listed above.
Final `prob_<algorithm>` columns and time metrics are also computed against the
BnB labels.

## Scheme 3

### Target definition

Scheme 3 adds:

- `y_algo`
- `y_group`
- `y_sparse_subgroup`
- `y_router4`

All four are derived from the fastest BnB label.

### Stage structure

Scheme 3 is the 3-stage hierarchy:

1. Stage 1: `Sparse` vs `Dense`
2. Stage 2: if `Sparse`, `AltRB` vs `No-AltRB`
3. Stage 3: 4-class router

Router mapping:

- `Class1`: `UMKP-S-bnb`, `UMKP+-S-bnb`, `kPEX-S-bnb`
- `Class2`: `UMKP-R-bnb`, `UMKP+-R-bnb`, `kPEX-R-bnb`
- `Class3`: `UMKP-P-bnb`, `UMKP+-P-bnb`, `kPEX-P-bnb`
- `Class4`: `kPEX-B-bnb`

Final prediction rule:

- if the router predicts `Class4`, the final label is `kPEX-B-bnb`
- otherwise the router class plus the Stage 1 / Stage 2 path resolve the final
  `*-bnb` label

### Probability construction

Scheme 3 still writes:

- final `prob_<algorithm>` columns
- stage-local `prob_stage*` columns

The final BnB probabilities are assembled from:

- Stage 1 group probabilities
- Stage 2 sparse-subgroup probabilities
- Stage 3 router probabilities

Implemented joint logic:

- `P(UMKP-S-bnb) = P(Sparse) * P(No-AltRB | Sparse) * P(Class1)`
- `P(UMKP-R-bnb) = P(Sparse) * P(No-AltRB | Sparse) * P(Class2)`
- `P(UMKP-P-bnb) = P(Sparse) * P(No-AltRB | Sparse) * P(Class3)`
- `P(UMKP+-S-bnb) = P(Sparse) * P(AltRB | Sparse) * P(Class1)`
- `P(UMKP+-R-bnb) = P(Sparse) * P(AltRB | Sparse) * P(Class2)`
- `P(UMKP+-P-bnb) = P(Sparse) * P(AltRB | Sparse) * P(Class3)`
- `P(kPEX-S-bnb) = P(Dense) * P(Class1)`
- `P(kPEX-R-bnb) = P(Dense) * P(Class2)`
- `P(kPEX-P-bnb) = P(Dense) * P(Class3)`
- `P(kPEX-B-bnb) = P(Class4)`

## Outputs

When `train.py --save-files` is used, the pipeline writes:

- `model_<run_name>.pkl`
- `all_predictions_<run_name>.csv`
- `wrong_predictions_<run_name>.csv`
- `wrong_predictions_<run_name>.xlsx`
- `metrics_<run_name>.json`
- `model_performance_for_all_exp.xlsx`
- split index CSVs
- run logs and evaluation/prediction logs

Those outputs now keep the BnB labels consistently in:

- `predicted`
- `fastest_algorithms`
- `y_algo`
- `prob_<algorithm>`
- wrong-case and failure-case analysis columns

## Reporting And Failure Cases

`compare_wrong_predictions.py` and the Excel reporting layer still work off the
saved wrong-prediction files. Runtime-gap fields such as:

- `fastest_time`
- `predicted_time`
- `time_gap`
- `time_gap_ratio`

are computed from the active BnB algorithm columns. Total-time columns can
still remain in the exported CSVs, but they are not used to define the
supervised target or failure severity.

## End-To-End Wrapper

`run_full_pipeline.sh` now performs:

1. training with `--save-files`
2. held-out evaluation
3. held-out prediction export
4. full-dataset prediction export
5. wrong-case comparison report generation

For one run name, it emits both prediction export files:

- `all_predictions_<RUN_NAME>_predict_inference_test_set.csv`
- `all_predictions_<RUN_NAME>_predict_inference_entire_data.csv`
