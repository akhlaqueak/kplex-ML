# Scheme 3

This document describes the active Scheme 3 pipeline under
`restructured_pipeline/` after the dataset update to `ml-data.csv`.

## Files

- `restructured_pipeline/configs/project.yaml`
- `restructured_pipeline/configs/models_scheme3.yaml`
- `restructured_pipeline/src/features.py`
- `restructured_pipeline/src/data_splits.py`
- `restructured_pipeline/src/models/hierarchical_scheme3.py`
- `restructured_pipeline/src/train.py`
- `restructured_pipeline/src/evaluate.py`
- `restructured_pipeline/src/predict.py`

## Dataset And Target Rule

Scheme 3 now reads:

- `input_data/ml-data.csv`

The dataset includes both BnB time columns and total-time columns. Scheme 3
uses only the BnB columns for:

- `y_algo`
- `y_group`
- `y_sparse_subgroup`
- `y_router4`
- sample weights
- evaluation metrics
- wrong-case and failure-case runtime comparisons

The active final labels are:

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

## Hierarchy

Scheme 3 uses three stages:

1. Stage 1: `Sparse` vs `Dense`
2. Stage 2: if Stage 1 is `Sparse`, `AltRB` vs `No-AltRB`
3. Stage 3: one shared router across `Class1`, `Class2`, `Class3`, `Class4`

The stage target columns are:

- `y_group`
- `y_sparse_subgroup`
- `y_router4`
- `y_algo`

## Stage 1

Stage 1 predicts:

- `Sparse`
- `Dense`

This is derived from the fastest BnB algorithm:

- `Dense` for `kPEX-S-bnb`, `kPEX-R-bnb`, `kPEX-P-bnb`, `kPEX-B-bnb`
- `Sparse` for all `UMKP*-bnb` and `UMKP+*-bnb` labels

## Stage 2

Stage 2 is only meaningful on rows where Stage 1 is `Sparse`.

It predicts:

- `AltRB`
- `No-AltRB`

This is derived from the fastest sparse BnB algorithm:

- `AltRB` for `UMKP+-S-bnb`, `UMKP+-R-bnb`, `UMKP+-P-bnb`
- `No-AltRB` for `UMKP-S-bnb`, `UMKP-R-bnb`, `UMKP-P-bnb`

## Stage 3

Stage 3 is a 4-class router across the full dataset.

Router class mapping:

- `Class1`: `UMKP-S-bnb`, `UMKP+-S-bnb`, `kPEX-S-bnb`
- `Class2`: `UMKP-R-bnb`, `UMKP+-R-bnb`, `kPEX-R-bnb`
- `Class3`: `UMKP-P-bnb`, `UMKP+-P-bnb`, `kPEX-P-bnb`
- `Class4`: `kPEX-B-bnb`

The router target column is:

- `y_router4`

The mapping logic is suffix-based for Classes 1-3:

- `Class1` for `*-S-bnb`
- `Class2` for `*-R-bnb`
- `Class3` for `*-P-bnb`
- `Class4` only for `kPEX-B-bnb`

## Final Prediction Rule

The final algorithm is resolved from the stage outputs in
`HierarchicalClassifier3`.

If Stage 3 predicts `Class4`, the final prediction is:

- `kPEX-B-bnb`

Otherwise:

- if Stage 1 predicts `Dense`, the final label is `kPEX-S-bnb`,
  `kPEX-R-bnb`, or `kPEX-P-bnb`
- if Stage 1 predicts `Sparse` and Stage 2 predicts `AltRB`, the final label
  is `UMKP+-S-bnb`, `UMKP+-R-bnb`, or `UMKP+-P-bnb`
- if Stage 1 predicts `Sparse` and Stage 2 predicts `No-AltRB`, the final
  label is `UMKP-S-bnb`, `UMKP-R-bnb`, or `UMKP-P-bnb`

## Probability Construction

Scheme 3 writes final `prob_<algorithm>` columns and stage-local probability
columns.

Final BnB algorithm probabilities are assembled from:

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

Stage-local probability columns are:

- `prob_stage1_Sparse`
- `prob_stage1_Dense`
- `prob_stage2_AltRB`
- `prob_stage2_No-AltRB`
- `prob_stage3_Class1`
- `prob_stage3_Class2`
- `prob_stage3_Class3`
- `prob_stage3_Class4`

## Scheme 3 vs Scheme 2

Scheme 2 uses five stages:

- Stage 1: `Sparse` vs `Dense`
- Stage 2: `AltRB` vs `No-AltRB`
- Stage 3: `AltRB` leaf classifier
- Stage 4: `No-AltRB` leaf classifier
- Stage 5: `Dense` leaf classifier

Scheme 3 uses three stages:

- Stage 1: `Sparse` vs `Dense`
- Stage 2: `AltRB` vs `No-AltRB`
- Stage 3: one shared 4-class router

The special Scheme 3 direct-output case is now:

- `Class4 -> kPEX-B-bnb`

## Running Scheme 3

With `--save-files`, training writes:

- `model_<run_name>.pkl`
- `all_predictions_<run_name>.csv`

Later `evaluate.py` and `predict.py` prefer the saved model artifact when it is
available, and fall back to the saved predictions export when needed.

Train Scheme 3 RF:

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme3 \
  --model-family random_forest \
  --split-mode strat_y \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme3.yaml \
  --save-files
```

Train Scheme 3 XGB:

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme3 \
  --model-family xgboost \
  --split-mode strat_y \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme3.yaml \
  --save-files
```

Evaluate a saved Scheme 3 run:

```bash
python restructured_pipeline/src/evaluate.py \
  --scheme scheme3 \
  --model-family random_forest \
  --run-name <saved_run_name> \
  --inference-mode inference_test_set \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme3.yaml
```

Export Scheme 3 predictions from a saved run:

```bash
python restructured_pipeline/src/predict.py \
  --scheme scheme3 \
  --model-family random_forest \
  --run-name <saved_run_name> \
  --inference-mode inference_test_set \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme3.yaml
```
