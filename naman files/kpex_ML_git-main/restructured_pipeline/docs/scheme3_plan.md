# Scheme 3

This document describes the active Scheme 3 pipeline under
`restructured_pipeline/` after the 24-algorithm dataset update.

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

Scheme 3 reads the configured raw dataset:

- `input_data/ml-data.csv`

The dataset includes both BnB runtime columns and matching total-time columns.
Scheme 3 uses only the 24 `-bnb` runtime columns for target construction,
sample weights, metrics, wrong-case exports, and failure-analysis runtime
comparisons. The matching non-`-bnb` total-time columns remain ignored by the
ML target logic unless a future config explicitly changes that behavior.

The active final labels are:

- `kPEX-S-bnb`
- `kPEX-R-bnb`
- `kPEX-P-bnb`
- `kPEX-B-bnb`
- `kPEX-oriented-S-bnb`
- `kPEX-oriented-R-bnb`
- `kPEX-oriented-P-bnb`
- `kPEX-oriented-B-bnb`
- `UMKP-S-bnb`
- `UMKP-R-bnb`
- `UMKP-P-bnb`
- `UMKP-B-bnb`
- `UMKP-oriented-S-bnb`
- `UMKP-oriented-R-bnb`
- `UMKP-oriented-P-bnb`
- `UMKP-oriented-B-bnb`
- `UMKP+-S-bnb`
- `UMKP+-R-bnb`
- `UMKP+-P-bnb`
- `UMKP+-B-bnb`
- `UMKP+-oriented-S-bnb`
- `UMKP+-oriented-R-bnb`
- `UMKP+-oriented-P-bnb`
- `UMKP+-oriented-B-bnb`

## Hierarchy

Scheme 3 now uses four stages:

1. Stage 1: `Sparse` vs `Dense`
2. Stage 2: if Stage 1 is `Sparse`, `AltRB` vs `No-AltRB`
3. Stage 3: `Oriented` vs `NonOriented`
4. Stage 4: one shared router across `ClassB`, `ClassS`, `ClassR`, `ClassP`

The stage target columns are:

- `y_group`
- `y_sparse_subgroup`
- `y_orientation`
- `y_router4`
- `y_algo`

## Stage 1

Stage 1 predicts:

- `Sparse`
- `Dense`

This is derived from the fastest BnB algorithm:

- `Dense` for every `kPEX...-bnb` label
- `Sparse` for every `UMKP...-bnb` and `UMKP+...-bnb` label

## Stage 2

Stage 2 is only meaningful on rows where Stage 1 is `Sparse`.

It predicts:

- `AltRB`
- `No-AltRB`

This is derived from the fastest sparse BnB algorithm:

- `AltRB` for every `UMKP+...-bnb` label
- `No-AltRB` for every `UMKP...-bnb` label that is not `UMKP+`
- dense rows keep the existing not-applicable/null treatment

## Stage 3

Stage 3 predicts:

- `Oriented`
- `NonOriented`

This is derived from the final label text:

- `Oriented` for labels containing `-oriented-`
- `NonOriented` for labels without `-oriented-`

## Stage 4

Stage 4 is a 4-class router across the full dataset.

Router class mapping:

- `ClassB`: every `*-B-bnb` label
- `ClassS`: every `*-S-bnb` label
- `ClassR`: every `*-R-bnb` label
- `ClassP`: every `*-P-bnb` label

The router target column is:

- `y_router4`

Backward-compatible `Class1` / `Class2` / `Class3` / `Class4` aliases may still
be understood by old artifacts, but the active 24-label config uses
`ClassB` / `ClassS` / `ClassR` / `ClassP`.

## Tree View

```text
Sparse
  No-AltRB
    UMKP-oriented-B/S/R/P
    UMKP-B/S/R/P
  AltRB
    UMKP+-oriented-B/S/R/P
    UMKP+-B/S/R/P

Dense
  kPEX-oriented-B/S/R/P
  kPEX-B/S/R/P
```

## Final Prediction Rule

The final algorithm is resolved from the stage outputs in
`HierarchicalClassifier3`.

If Stage 1 predicts `Dense`:

- family is `kPEX`
- `Oriented` becomes `kPEX-oriented`
- `NonOriented` remains `kPEX`
- Stage 4 supplies the `B`, `S`, `R`, or `P` suffix

If Stage 1 predicts `Sparse` and Stage 2 predicts `AltRB`:

- family is `UMKP+`
- `Oriented` becomes `UMKP+-oriented`
- `NonOriented` remains `UMKP+`
- Stage 4 supplies the `B`, `S`, `R`, or `P` suffix

If Stage 1 predicts `Sparse` and Stage 2 predicts `No-AltRB`:

- family is `UMKP`
- `Oriented` becomes `UMKP-oriented`
- `NonOriented` remains `UMKP`
- Stage 4 supplies the `B`, `S`, `R`, or `P` suffix

The resolved final label must exactly match one of the 24 active `-bnb`
runtime labels.

## Probability Construction

Scheme 3 writes final `prob_<algorithm>` columns for all 24 final labels and
stage-local probability columns.

Final BnB algorithm probabilities are assembled from:

- Stage 1 group probabilities
- Stage 2 sparse-subgroup probabilities for sparse families
- Stage 3 orientation probabilities
- Stage 4 router probabilities

Examples:

- `P(kPEX-oriented-B-bnb) = P(Dense) * P(Oriented) * P(ClassB)`
- `P(kPEX-S-bnb) = P(Dense) * P(NonOriented) * P(ClassS)`
- `P(UMKP+-oriented-P-bnb) = P(Sparse) * P(AltRB | Sparse) * P(Oriented) * P(ClassP)`
- `P(UMKP-R-bnb) = P(Sparse) * P(No-AltRB | Sparse) * P(NonOriented) * P(ClassR)`

Stage-local probability columns are:

- `prob_stage1_Sparse`
- `prob_stage1_Dense`
- `prob_stage2_AltRB`
- `prob_stage2_No-AltRB`
- `prob_stage3_Oriented`
- `prob_stage3_NonOriented`
- `prob_stage4_ClassB`
- `prob_stage4_ClassS`
- `prob_stage4_ClassR`
- `prob_stage4_ClassP`

## Scheme 3 vs Scheme 2

Scheme 2 remains the older branch-local hierarchy:

- Stage 1: `Sparse` vs `Dense`
- Stage 2: `AltRB` vs `No-AltRB`
- Stage 3: `AltRB` leaf classifier
- Stage 4: `No-AltRB` leaf classifier
- Stage 5: `Dense` leaf classifier

Scheme 3 is the active 24-label hierarchical path:

- Stage 1: `Sparse` vs `Dense`
- Stage 2: `AltRB` vs `No-AltRB`
- Stage 3: `Oriented` vs `NonOriented`
- Stage 4: one shared `B` / `S` / `R` / `P` router

Do not migrate or remove Scheme 2 behavior as part of Scheme 3 cleanup unless a
separate validation task explicitly covers it.

## Running Scheme 3

With `--save-files`, training writes:

- `model_<run_name>.pkl`
- `all_predictions_<run_name>.csv`
- `wrong_predictions_<run_name>.xlsx`

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
  --split-mode strat_y_K \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme3.yaml \
  --save-files
```

Evaluate a saved Scheme 3 run:

```bash
python restructured_pipeline/src/evaluate.py \
  --scheme scheme3 \
  --model-family xgboost \
  --run-name <saved_run_name> \
  --inference-mode inference_test_set \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme3.yaml
```

Export Scheme 3 predictions from a saved run:

```bash
python restructured_pipeline/src/predict.py \
  --scheme scheme3 \
  --model-family xgboost \
  --run-name <saved_run_name> \
  --inference-mode inference_test_set \
  --project-config restructured_pipeline/configs/project.yaml \
  --model-config restructured_pipeline/configs/models_scheme3.yaml
```
