# Future High-Risk Cleanup Plan

Date: 2026-06-24

Purpose:

- define a future-only cleanup plan for the remaining high-risk items
- turn the current `DO NOT TOUCH YET` list into a staged, validation-first
  execution plan
- avoid accidental removal of compatibility code before the required evidence
  exists

This document is planning-only. It does not approve any code removal by
itself.

## Overall Recommended Order

1. Compatibility alias families
2. `run_full_pipeline.sh` staging fallback
3. `failure_case_filter.py`
4. Scheme 1 hyperparameter dual source
5. Duplicated hierarchical helpers

Why this order:

- alias cleanup and wrapper staging cleanup affect operator-visible paths and
  must be resolved before larger internal refactors
- `failure_case_filter.py` can only be retired after a replacement workflow is
  proven
- Scheme 1 hyperparameter consolidation needs a focused regression pass
- hierarchical helper deduplication is safest when done as part of the planned
  larger module split

## 1. Compatibility Alias Families

Risk level:

- HIGH

Dependencies:

- `src/compare_wrong_predictions.py`
- `src/failure_case_filter.py`
- `src/failure_analysis_summary.py`
- `docs/comparison_wrong_predictions.md`
- `docs/compatibility_paths_and_removal_checklist.md`
- saved operator outputs that may still use old filenames

Estimated effort:

- Medium for planning and doc cleanup
- Medium to High for staged removal and revalidation

### Exact Alias Families

| Alias Family | Exact Alias Paths | Canonical Replacement |
|------|------|------|
| compare workbook typo alias | `restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_schema_entire_data.xlsx` | `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx` |
| older compare workbook fallbacks | `restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data.xlsx` | `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx` |
| older compare workbook fallback variant | `restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data_pre_consistency_fix.xlsx` | `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx` |
| standalone tolerance old filename | `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report.xlsx` | `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all.xlsx` |
| legacy failure-analysis input candidate | `restructured_pipeline/outputs/reports/failure_case_analysis_cleaned.xlsx` | canonical summary-mode / enrich-mode input flow documented in current compare and failure-analysis docs |
| legacy failure-analysis input candidate | `restructured_pipeline/outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data_fixed.xlsx` | `restructured_pipeline/outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data.xlsx` or the later canonical conversion target if enrich mode changes |
| top-level test-data compare fallback | `restructured_pipeline/outputs/reports/compare_wrong_predictions_test_data.xlsx` | `restructured_pipeline/outputs/reports/test_data/compare_wrong_predictions_test_data.xlsx` |
| important-failure fallback alias | `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_important_failures.xlsx` when consumed as fallback for enrich-related paths | `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx` |

### Validation Required Before Removal

1. Update all runnable docs to use only canonical paths.
2. Confirm no helper or discovery code still searches the alias family.
3. Confirm canonical-path workflows emit no compatibility warnings.
4. Confirm deprecated-path workflows still warn before the alias is removed.
5. Remove one alias family at a time.

### Exact Smoke Tests

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
  --tolerance-output-path /private/tmp/future_high_risk_tol.xlsx
```

```bash
python restructured_pipeline/src/failure_case_filter.py \
  --input-workbook restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx \
  --output-path /private/tmp/future_high_risk_filtered.xlsx
```

```bash
python restructured_pipeline/src/failure_analysis_summary.py \
  --mode enrich \
  --input-workbook /private/tmp/future_high_risk_filtered.xlsx \
  --enriched-failure-workbook-output /private/tmp/future_high_risk_failure_summary.xlsx
```

### Removal Order

1. `all_schema` compare typo alias
2. older compare workbook fallbacks
3. standalone tolerance old filename
4. legacy failure-analysis input candidates
5. top-level test-data compare fallback
6. important-failure fallback alias if enrich-mode replacement work has already finished

Recommended rule:

- never remove more than one alias family in a single validation pass

## 2. `run_full_pipeline.sh` Staging Fallback

Risk level:

- HIGH

Dependencies:

- `run_full_pipeline.sh --compare-input-mode staging`
- `src/compare_wrong_predictions.py --model-root ...`
- `docs/comparison_wrong_predictions.md`
- operator workflows that still rely on staged compare-input folders

Estimated effort:

- Medium

### When It Can Be Removed

Remove the staging fallback only when all of the following are true:

1. direct wrapper mode is the sole recommended path in docs
2. direct wrapper mode is validated for all metadata expectations
3. no external workflow still depends on staged filenames
4. either:
   - staging mode has been repaired and proven unnecessary, or
   - staging mode remains metadata-inferior and has been formally deprecated

Current evidence from
`docs/run_full_pipeline_direct_vs_staging_validation.md`:

- direct mode is stable and preserves `scheme`, `model`, `strat_on`,
  `inference_on`
- staging mode still runs, but in the validated wrapper path it leaves
  `scheme` and `model` blank and produces empty Scheme 1 tolerance and
  important-failure sheet counts

### Required Direct-Mode Validation

Direct mode must pass:

- compare workbook exists
- tolerance workbook exists
- important-failure workbook exists
- no blank `scheme`
- no blank `model`
- `strat_on` populated
- `inference_on` populated
- no staged compatibility folders created

### Exact Smoke Tests

```bash
bash restructured_pipeline/run_full_pipeline.sh \
  --python-bin /opt/miniconda3/envs/kpex/bin/python \
  --scheme scheme1 \
  --model-family random_forest \
  --split-mode strat_y \
  --variant rf_strat_y \
  --run-name scheme1_rf_future_direct \
  --compare-input-mode direct \
  --skip-full-predict
```

```bash
bash restructured_pipeline/run_full_pipeline.sh \
  --python-bin /opt/miniconda3/envs/kpex/bin/python \
  --scheme scheme1 \
  --model-family random_forest \
  --split-mode strat_y \
  --variant rf_strat_y \
  --run-name scheme1_rf_future_staging \
  --compare-input-mode staging \
  --skip-full-predict
```

### Fallback-Removal Steps

1. Keep direct mode as preferred and rerun direct validation.
2. Decide whether staging will be:
   - repaired first, or
   - deprecated as legacy fallback before removal
3. Remove wrapper docs that recommend staging.
4. Remove the staged compare-input directory creation branch.
5. Remove staged filename copying.
6. Remove the `--compare-input-mode staging` CLI path.
7. Re-run direct wrapper smoke tests.

Recommended note:

- do not start staging removal until the metadata gap recorded in
  `docs/run_full_pipeline_direct_vs_staging_validation.md` is explicitly
  resolved one way or the other

## 3. Duplicated Hierarchical Helpers

Risk level:

- HIGH

Dependencies:

- `src/models/hierarchical.py`
- `src/models/hierarchical_scheme3.py`
- Scheme 2 RF/XGB training
- Scheme 3 RF/XGB training
- any saved-run reuse path depending on current preprocessing

Estimated effort:

- High

### Proposed Shared Helper / Module Design

Recommended target:

- new helper module:
  `src/models/hierarchical_shared.py`

Suggested contents:

- `get_fastest_algorithms(row, algorithms)`
- `prepare_hierarchical_X_values(results_df, features, kcore_list=False, for_nn=False)`
- `transform_optional_labels(labels, le)`

Design rules:

- helpers should stay data-preparation-only
- no stage-routing behavior should be moved into the shared module yet
- Scheme 2 and Scheme 3 should keep their classifier-specific assembly logic
- move only exact duplicate helpers first

### Required Scheme 2 / Scheme 3 Regression Tests

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme2 \
  --model-family random_forest \
  --split-mode strat_y \
  --save-files \
  --file-suffix scheme2_rf_future_high_risk
```

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme2 \
  --model-family xgboost \
  --split-mode strat_y \
  --save-files \
  --file-suffix scheme2_xgb_future_high_risk
```

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme3 \
  --model-family random_forest \
  --split-mode strat_y \
  --save-files \
  --file-suffix scheme3_rf_future_high_risk
```

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme3 \
  --model-family xgboost \
  --split-mode strat_y \
  --save-files \
  --file-suffix scheme3_xgb_future_high_risk
```

Regression checks:

- no import errors
- stage summary workbook present for hierarchical runs
- wrong-case workbook present
- confusion matrix present
- row counts unchanged
- no unexpected label drift

Recommended execution note:

- do this only inside the planned larger module split cycle

## 4. Scheme 1 Hyperparameter Dual Source

Risk level:

- HIGH

Dependencies:

- `src/models/flat.py`
- `configs/models_scheme1.yaml`
- Scheme 1 RF variants
- Scheme 1 XGB variants
- wrapper variant auto-resolution

Estimated effort:

- Medium to High

### Should YAML Become Canonical?

Recommended answer:

- Yes, YAML should become the canonical long-term source of truth

Reason:

- model variant configuration is easier to audit, compare, and document in one
  declarative file
- code should become thin constructor logic plus validation only

### How To Compare Before / After Hyperparameters

For each Scheme 1 variant:

- extract current code preset from `models/flat.py`
- extract current YAML values from `configs/models_scheme1.yaml`
- create a one-row comparison table per variant:
  - parameter name
  - code value
  - YAML value
  - final runtime override value used by `train.py`

Variants to compare:

- `rf_strat_y`
- `rf_strat_y_K`
- `xgb_strat_y`
- `xgb_strat_y_K`

Comparison goal:

- prove the final runtime configuration is identical before any removal

### Required Scheme 1 RF / XGB Validation

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme1 \
  --model-family random_forest \
  --split-mode strat_y \
  --variant rf_strat_y \
  --save-files \
  --file-suffix scheme1_rf_strat_y_future_high_risk
```

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme1 \
  --model-family random_forest \
  --split-mode strat_y_K \
  --variant rf_strat_y_K \
  --save-files \
  --file-suffix scheme1_rf_strat_y_K_future_high_risk
```

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme1 \
  --model-family xgboost \
  --split-mode strat_y \
  --variant xgb_strat_y \
  --save-files \
  --file-suffix scheme1_xgb_strat_y_future_high_risk
```

```bash
python restructured_pipeline/src/train.py \
  --scheme scheme1 \
  --model-family xgboost \
  --split-mode strat_y_K \
  --variant xgb_strat_y_K \
  --save-files \
  --file-suffix scheme1_xgb_strat_y_K_future_high_risk
```

Validation checks:

- confusion matrix written
- wrong-case workbook written
- metrics workbook updated
- no variant-resolution mismatch
- no parameter drift between pre-change and post-change runs

## 5. `failure_case_filter.py`

Risk level:

- HIGH

Dependencies:

- `src/failure_case_filter.py`
- `src/failure_analysis_summary.py --mode enrich`
- canonical filtered enrich workflow
- `failure_case_analysis_important_failures_all_scheme_entire_data.xlsx`

Estimated effort:

- Medium

### Replacement Needed Before Retiring It

At least one of these must exist first:

1. `failure_analysis_summary.py --mode enrich` directly consumes the integrated
   important-failure workbook family produced by
   `compare_wrong_predictions.py`

or

2. a small canonical converter transforms the integrated important-failure
   workbook shape into the current single-sheet enrich-input shape

Recommended direction:

- prefer teaching enrich mode to consume integrated important-failure
  workbooks directly, if workbook semantics can be preserved cleanly

### Should Enrich Mode Consume Integrated Important-Failure Workbooks Directly?

Recommended answer:

- Yes, if and only if the enrich input contract can be defined clearly and the
  row counts and per-row top-K fields stay unchanged

Why:

- it would remove one extra report transformation step
- it would make the compare -> important-failure -> enrich chain more direct

When not to do it:

- if enrich mode still relies on a shape that cannot be reconstructed without
  ambiguity from the integrated workbook

### Exact Row-Count Validation Needed

```bash
python restructured_pipeline/src/compare_wrong_predictions.py \
  --run-entire-data-pipeline
```

```bash
python restructured_pipeline/src/failure_case_filter.py \
  --input-workbook restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx \
  --output-path /private/tmp/future_failure_filter_filtered.xlsx
```

```bash
python restructured_pipeline/src/failure_analysis_summary.py \
  --mode enrich \
  --input-workbook /private/tmp/future_failure_filter_filtered.xlsx \
  --enriched-failure-workbook-output /private/tmp/future_failure_filter_enrich.xlsx
```

If a direct enrich replacement is built, then compare:

- old filtered workbook total rows vs replacement input total rows
- old enrich workbook sheet counts vs replacement enrich workbook sheet counts
- per-sheet row counts:
  - `scheme_1_RF`
  - `scheme_1_XGB`
  - `scheme_2_RF`
  - `scheme_2_XGB`
  - `scheme_3_RF`
  - `scheme_3_XGB`
- no blank `scheme`
- no blank `model`

Retirement rule:

- do not retire `failure_case_filter.py` until the replacement path matches
  row counts and output shape across the full enrich validation pass

## Recommended Future Execution Pattern

For every high-risk item above:

1. create a dedicated validation task first
2. run `bash restructured_pipeline/run_cleanup_smoke_tests.sh`
3. run the item-specific smoke tests from this plan
4. change one high-risk surface at a time
5. compare row counts, workbook presence, metadata completeness, and warnings
6. update docs immediately after each successful removal or consolidation

## Final Recommendation

Use this document together with:

- `docs/high_risk_cleanup_checklist.md`
- `docs/compatibility_paths_and_removal_checklist.md`
- `docs/run_full_pipeline_direct_vs_staging_validation.md`

This plan should guide the next high-risk cleanup cycle, but none of the items
here should be implemented without an explicitly approved follow-up task.
