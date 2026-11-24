# High-Risk Cleanup Checklist

Date: 2026-06-23

Purpose:

- record the items that are still intentionally risky to remove or merge
- make future cleanup safer by tying each risky item to its dependencies,
  validation gates, and exact smoke tests
- complement, not replace,
  `docs/compatibility_paths_and_removal_checklist.md`

Use this document before deleting, merging, or rewriting any of the items
below.

## Current Do-Not-Touch List

| Item | Current Status | Recommended Action |
|------|------|------|
| compatibility alias families | still used as compatibility paths and fallbacks | `DO NOT TOUCH YET` |
| `run_full_pipeline.sh` staging compare-input fallback | still kept as wrapper fallback mode | `DO NOT TOUCH YET` |
| duplicated hierarchical helpers in `models/hierarchical.py` and `models/hierarchical_scheme3.py` | still duplicated on purpose pending larger split | `DO NOT TOUCH YET` |
| Scheme 1 hyperparameter dual source in `models/flat.py` and `configs/models_scheme1.yaml` | still a two-source runtime risk area | `DO NOT TOUCH YET` |
| `failure_case_filter.py` | still required by the canonical filtered enrich workflow | `DO NOT TOUCH YET` |

## 1. Compatibility Alias Families

Scope:

- old compare workbook fallbacks
- `all_schema` typo paths
- dated report fallback paths

Why this is risky:

- these aliases still protect older commands, downstream workbook readers, and
  compatibility fallbacks that are intentionally kept in place
- removing one alias can silently break compare, filtered enrich, or summary
  workflows that do not yet use only canonical paths
- some warnings were added specifically to make alias usage visible before
  removal; deleting paths too early would skip that transition window

What depends on it:

- `src/compare_wrong_predictions.py`
- `src/failure_case_filter.py`
- `src/failure_analysis_summary.py`
- workflow docs that still describe fallback behavior
- operators with older saved output paths

Validate before removal:

1. Confirm all runnable docs use canonical paths only.
2. Confirm no helper or discovery branch still searches the alias family.
3. Confirm canonical-path workflows run without compatibility warnings.
4. Confirm deprecated-path workflows still warn before the alias family is
   removed.
5. Remove only one alias family at a time.

Proposed future replacement:

- canonical filenames documented in
  `docs/compatibility_paths_and_removal_checklist.md`
- warning-backed deprecation window followed by one-family-at-a-time removal

Exact smoke tests required:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --run-test-data-pipeline
```

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --run-entire-data-pipeline
```

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --input-workbook restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx \
  --generate-tolerance-report \
  --tolerance-output-path /private/tmp/high_risk_tol.xlsx
```

```bash
python restructured_pipeline/src/failure_case_filter.py \
  --input-workbook restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx \
  --output-path /private/tmp/high_risk_filtered.xlsx
```

```bash
python restructured_pipeline/src/failure_analysis_summary.py \
  --mode enrich \
  --input-workbook /private/tmp/high_risk_filtered.xlsx \
  --enriched-failure-workbook-output /private/tmp/high_risk_failure_summary.xlsx
```

Keep this tied to the full compatibility-removal sequence in
`docs/compatibility_paths_and_removal_checklist.md`.

## 2. `run_full_pipeline.sh` Staging Compare-Input Fallback

Why this is risky:

- direct compare mode is now preferred, but the staged `expand_and_split/` and
  `split_and_expand/` layout is still the documented fallback path
- wrapper users may still rely on the old compare-input naming semantics for
  debugging or compatibility
- removing staging too early would strand the fallback path before direct mode
  is the only documented and validated wrapper path

What depends on it:

- `restructured_pipeline/run_full_pipeline.sh --compare-input-mode staging`
- `src/compare_wrong_predictions.py --model-root ...`
- `docs/comparison_wrong_predictions.md`
- any operator workflow that still depends on staged filenames

Validate before removal:

1. Confirm direct wrapper mode is the only recommended path in docs.
2. Confirm direct wrapper mode preserves all expected metadata:
   `source_model`, `with_holdout`, `strat_on`, `inference_on`.
3. Confirm no fallback scripts or operator notes still require staged files.
4. Confirm staging remains unnecessary across one validated wrapper run.

Proposed future replacement:

- wrapper direct-file compare only:
  `compare_wrong_predictions.py --wrong-predictions-file ...`
- keep `--source-model-label` as the metadata-preserving bridge

Exact smoke tests required:

```bash
bash restructured_pipeline/run_full_pipeline.sh \
  --python-bin /opt/miniconda3/envs/kpex/bin/python \
  --scheme scheme1 \
  --model-family random_forest \
  --split-mode strat_y \
  --run-name scheme1_rf_high_risk_direct \
  --compare-input-mode direct
```

```bash
bash restructured_pipeline/run_full_pipeline.sh \
  --python-bin /opt/miniconda3/envs/kpex/bin/python \
  --scheme scheme1 \
  --model-family random_forest \
  --split-mode strat_y \
  --run-name scheme1_rf_high_risk_staging \
  --compare-input-mode staging
```

Validation checks:

- compare workbook exists
- tolerance workbook exists
- important-failure workbook exists
- no blank `scheme` / `model`
- direct mode creates no staged compare-input directory
- staging mode still works unchanged until removal day

## 3. Duplicated Hierarchical Helpers

Scope:

- `src/models/hierarchical.py`
- `src/models/hierarchical_scheme3.py`

Affected duplicated helpers:

- `get_fastest_algorithms`
- `_prepare_X_values`
- `_transform_optional_labels`

Why this is risky:

- the duplication is real, but these modules sit inside active Scheme 2 and
  Scheme 3 training paths with different surrounding stage logic
- a premature helper merge could subtly change data preparation, class-label
  handling, or training-time behavior for hierarchical models
- the repo already plans to revisit this during the larger module split, which
  is the safer time to centralize them

What depends on it:

- Scheme 2 RF training
- Scheme 2 XGB training
- Scheme 3 RF training
- Scheme 3 XGB training
- any saved-run reload path that depends on their current preprocessing

Validate before removal or merge:

1. Start only after the planned hierarchical module split work begins.
2. Compare class distributions and output row counts before and after helper
   extraction.
3. Confirm training artifacts, confusion matrices, and wrong-case workbooks are
   unchanged for representative Scheme 2 and Scheme 3 runs.

Proposed future replacement:

- one shared hierarchical helper module or base preprocessing layer introduced
  during the planned large-module split

Exact smoke tests required:

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme2 \
  --model-family random_forest \
  --split-mode strat_y \
  --save-files \
  --file-suffix scheme2_rf_high_risk
```

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme2 \
  --model-family xgboost \
  --split-mode strat_y \
  --save-files \
  --file-suffix scheme2_xgb_high_risk
```

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme3 \
  --model-family random_forest \
  --split-mode strat_y \
  --save-files \
  --file-suffix scheme3_rf_high_risk
```

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme3 \
  --model-family xgboost \
  --split-mode strat_y \
  --save-files \
  --file-suffix scheme3_xgb_high_risk
```

Recommended comparison checks:

- no import errors
- stage summary workbook presence
- wrong-case workbook presence
- stable row counts
- no unexpected label drift

## 4. Scheme 1 Hyperparameter Dual Source

Scope:

- `src/models/flat.py`
- `configs/models_scheme1.yaml`

Why this is risky:

- Scheme 1 still has hyperparameter values in both code presets and YAML
- consolidating too early could change variant resolution or silently alter the
  trained model configuration
- `train.py` currently routes Scheme 1 through both variant naming and YAML
  configuration behavior, so this is a source-of-truth problem with real model
  regression risk

What depends on it:

- Scheme 1 RF variants:
  `rf_strat_y`, `rf_strat_y_K`
- Scheme 1 XGB variants:
  `xgb_strat_y`, `xgb_strat_y_K`
- wrapper variant auto-resolution for Scheme 1
- any operator workflow that relies on the current variant names

Validate before removal or consolidation:

1. Identify one canonical source of truth first.
2. Confirm all four Scheme 1 variants resolve to the expected hyperparameters.
3. Confirm training metrics and wrong-case counts remain within expected
   tolerance after consolidation.
4. Confirm docs still map variant names to the correct configuration.

Proposed future replacement:

- keep YAML as the single configurable source of truth
- reduce `models/flat.py` to thin constructors and validation helpers only

Exact smoke tests required:

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme1 \
  --model-family random_forest \
  --split-mode strat_y \
  --variant rf_strat_y \
  --save-files \
  --file-suffix scheme1_rf_strat_y_high_risk
```

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme1 \
  --model-family random_forest \
  --split-mode strat_y_K \
  --variant rf_strat_y_K \
  --save-files \
  --file-suffix scheme1_rf_strat_y_K_high_risk
```

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme1 \
  --model-family xgboost \
  --split-mode strat_y \
  --variant xgb_strat_y \
  --save-files \
  --file-suffix scheme1_xgb_strat_y_high_risk
```

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme1 \
  --model-family xgboost \
  --split-mode strat_y_K \
  --variant xgb_strat_y_K \
  --save-files \
  --file-suffix scheme1_xgb_strat_y_K_high_risk
```

Recommended comparison checks:

- confusion matrix written
- wrong-case workbook written
- metrics workbook updated successfully
- no variant-resolution mismatch

## 5. `failure_case_filter.py`

Why this is risky:

- this script may look redundant because important-failure filtering also
  exists inside `compare_wrong_predictions.py`
- however, `failure_analysis_summary.py --mode enrich` still expects the
  single-sheet filtered workbook shape produced by `failure_case_filter.py`
- removing it now would break the canonical filtered enrich workflow unless
  enrich mode is taught to consume the integrated important-failure workbook
  directly

What depends on it:

- `src/failure_analysis_summary.py --mode enrich`
- the canonical filtered failure-summary workflow
- docs that still point to the single-sheet filtered workbook

Validate before removal:

1. Teach enrich mode to consume the integrated important-failure workbook or
   provide a canonical conversion step.
2. Prove the replacement preserves row counts, columns, and expected sheet
   shape.
3. Revalidate the full filtered failure-summary workflow end to end.

Proposed future replacement:

- either direct enrich-mode support for
  `compare_wrong_predictions_tolerance_important_failures*.xlsx`
- or one explicit canonical converter from tolerance-important-failure workbook
  shape to enrich-input workbook shape

Exact smoke tests required:

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --run-entire-data-pipeline
```

```bash
python restructured_pipeline/src/failure_case_filter.py \
  --input-workbook restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx \
  --output-path /private/tmp/high_risk_filtered.xlsx
```

```bash
python restructured_pipeline/src/failure_analysis_summary.py \
  --mode enrich \
  --input-workbook /private/tmp/high_risk_filtered.xlsx \
  --enriched-failure-workbook-output /private/tmp/high_risk_failure_summary.xlsx
```

Validation checks:

- filtered workbook created
- enrich workbook created
- expected sheets present
- row counts preserved
- no blank `scheme` / `model`

## Recommended Future Order

1. Finish all low-risk and medium-risk cleanup first.
2. Remove compatibility alias families one family at a time.
3. Retire the staging wrapper fallback only after direct wrapper mode is the
   sole validated path.
4. Merge hierarchical helpers only during the planned large-module split.
5. Consolidate Scheme 1 hyperparameters only with a dedicated regression pass.
6. Retire `failure_case_filter.py` only after enrich mode no longer depends on
   its single-sheet workbook shape.

Until those validations are complete, every item in this file remains
`DO NOT TOUCH YET`.
