# Wrong-Predictions Comparison

This document describes the active wrong-predictions comparison workflow
implemented by:

- `restructured_pipeline/src/compare_wrong_predictions.py`

## What The Comparison Does

The comparison workflow reads saved `wrong_predictions_*.xlsx` workbooks,
filters rows that meet the configured runtime-gap threshold, derives a compact
set of comparison columns, and writes a formatted Excel workbook for review.

It is the active Python replacement for the older notebook-based
wrong-predictions comparison flow.

## Input Files

The comparison expects wrong-case Excel files written by:

- `restructured_pipeline/src/train.py`
- `restructured_pipeline/src/predict.py`

Current wrong-case workbook naming:

- training wrong cases:
  `wrong_predictions_<RUN_NAME>.xlsx`
- held-out prediction wrong cases:
  `wrong_predictions_<RUN_NAME>_predict_inference_test_data.xlsx`
- full-dataset prediction wrong cases:
  `wrong_predictions_<RUN_NAME>_predict_inference_entire_data.xlsx`

Current storage location:

- `restructured_pipeline/outputs/wrong_cases/<scheme>/<model_family>/`

## Supported Comparison Modes

`compare_wrong_predictions.py` supports three input modes.

### 1. Staged Model-Root Comparison

Use `--model-root NAME=PATH` when the input folder already contains staged
strategy subfolders such as:

- `expand_and_split/`
- `split_and_expand/`

This is the mode used by `run_full_pipeline.sh` for the per-run comparison
report.

Example:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --model-root scheme2_random_forest=restructured_pipeline/outputs/reports/compare_inputs/scheme2_random_forest_demo \
  --threshold 0.5
```

### 2. Direct Saved Workbook Comparison

Use `--wrong-predictions-file [NAME=]PATH` when you want to compare one or more
explicit saved wrong-case workbooks directly.

Example:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --wrong-predictions-file restructured_pipeline/outputs/wrong_cases/scheme2/random_forest/wrong_predictions_scheme2_random_forest_strat_y_demo_predict_inference_entire_data.xlsx
```

### 3. Consolidated Full-Dataset Comparison From Disk

Use `--discover-entire-data-files` to auto-discover:

- `wrong_predictions_*predict_inference_entire_data*.xlsx`

under the wrong-cases root directory.

This is the active consolidated workflow for comparing full-dataset wrong
predictions across all schemes and model families from disk.

Example:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --discover-entire-data-files \
  --wrong-cases-root restructured_pipeline/outputs/wrong_cases \
  --threshold 0.5 \
  --time-gap-highlight-threshold 100
```

Default consolidated output path:

- `restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_schema_entire_data.xlsx`

## How Rows Are Chosen

The comparison first loads each workbook into a dataframe and ensures
`time_gap_ratio` is available. If the source workbook is missing
`time_gap_ratio`, it is rebuilt from:

- `fastest_time`
- `predicted_time`
- `time_gap`

Current row filter:

- keep rows where `time_gap_ratio >= --threshold`

Default threshold:

- `0.5`

This filter controls which rows appear in the comparison workbook.

## How Highlighting Works

Formatting is applied by the same script after the workbook is written.

Two different highlight rules are active:

1. Row severity fill:
   derived from `failure_severity`, which is computed from `time_gap_ratio`
2. Special yellow cell highlight:
   applied only to the `time_gap` cell when
   `time_gap > --time-gap-highlight-threshold`

Default yellow-cell threshold:

- `100.0`

Important distinction:

- the row filter uses `time_gap_ratio`
- the special yellow highlight uses the absolute `time_gap` value
- only the `time_gap` cell is turned yellow
- the `time_gap_ratio` cell is not turned yellow by the special rule

## How Schemes And Models Are Compared

The comparison dataframe keeps all probability columns and adds a common set of
review columns. For direct saved-workbook and auto-discovery modes, the code
derives:

- `scheme` from the filename token:
  `scheme1`, `scheme2`, `scheme3`
- `model` from the filename token:
  `random_forest -> RF`,
  `xgboost -> XGB`

The `source_model` column is preserved from the current aggregation logic:

- in staged model-root mode it reflects the supplied model-root label
- in direct-file mode it reflects the supplied file label or discovered file
  stem

## Output Workbook Contents

The comparison workbook contains one sheet per `inference_on` value.

Common sheet values:

- `on 20% test set`
- `on entire dataset`

For the consolidated full-dataset discovery workflow, the workbook typically
contains only:

- `on entire dataset`

Current leading output columns are written in this order:

1. `inference_on`
2. `with_holdout`
3. `strat_on`
4. `k_value`
5. `dataset`
6. `scheme`
7. `model`
8. `source_model`
9. `failure_severity`
10. `fastest_algo`
11. `fastest_runtime`
12. `predicted_algo`
13. `predicted_probability`
14. `fastest_time`
15. `predicted_time`
16. `time_gap`
17. `time_gap_ratio`
18. `time_ratio`
19. `top3_predicted_detailed`

After those columns, the workflow appends all remaining `prob_*` columns.

## Consolidated Full-Dataset Workflow

Use this path when the goal is one workbook across all schemes for the entire
dataset.

1. Generate saved full-dataset wrong-case workbooks with `predict.py` using
   `--inference-mode inference_entire_data`
2. Ensure the workbooks are under
   `restructured_pipeline/outputs/wrong_cases/<scheme>/<model_family>/`
3. Run:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --discover-entire-data-files \
  --wrong-cases-root restructured_pipeline/outputs/wrong_cases \
  --threshold 0.5 \
  --time-gap-highlight-threshold 100
```

4. Review the consolidated workbook:

- `restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_schema_entire_data.xlsx`

## Unified Script Notes

`compare_wrong_predictions.py` is the single maintained entry point for both
comparison and Excel reporting.

- `--mode compare` runs discovery/loading, filtering, column derivation, and
  workbook writing
- `--mode report` reapplies workbook formatting to an existing compare output
- `top3_predicted_detailed` is exported in a canonical rounded format using 4
  decimal places for probabilities and time ratios

## Important Failure Filtering

Use this path when the goal is to keep only the important failure cases from an
already-generated consolidated compare workbook.

The active filter rules are:

- exclude rows where both runtimes are below `--small-runtime-threshold`
- exclude rows where `predicted_runtime <= --slowdown-ratio-threshold * fastest_runtime`
- keep all remaining rows as important failures

Runtime-column compatibility:

- the filter prefers `fastest_runtime` / `predicted_runtime` when present
- it falls back to `fastest_time` / `predicted_time` for the current compare
  workbook format

Default thresholds:

- `--small-runtime-threshold 5.0`
- `--slowdown-ratio-threshold 1.2`

Example:

```bash
python restructured_pipeline/src/failure_case_filter.py \
  --input-workbook restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data.xlsx \
  --output-path restructured_pipeline/outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data.xlsx \
  --small-runtime-threshold 5.0 \
  --slowdown-ratio-threshold 1.2 \
  --time-gap-highlight-threshold 100
```

The filtered workbook keeps the original compare-report columns, adds
`failure_case_filter_reason`, writes only important failures to the data sheets,
and includes a summary sheet with exclusion counts.

## Relationship To The Wrapper Script

`restructured_pipeline/run_full_pipeline.sh` still performs a different,
per-run comparison workflow:

- it stages the training wrong-case workbook under `expand_and_split`
- it stages the held-out prediction wrong-case workbook under
  `split_and_expand`
- it then calls `compare_wrong_predictions.py --model-root ...`

That wrapper path is useful for one run at a time.

The consolidated full-dataset workflow described above is the cross-scheme,
file-based comparison path for saved entire-dataset wrong-case workbooks.
