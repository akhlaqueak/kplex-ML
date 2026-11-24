# Compatibility Paths And Removal Checklist

Date: 2026-06-23

This document records the current canonical output filenames for the active
pipeline and the older compatibility aliases that are still accepted for now.

The goal is to make later cleanup safer:

- canonical paths should be used in new commands, scripts, and docs
- deprecated aliases should continue working during the transition
- warnings should appear only when a deprecated alias is actually used

## Canonical Output Filenames

| Artifact Type | Canonical Output Filename | Notes |
|------|------|------|
| training full-dataset predictions | `restructured_pipeline/outputs/predictions/<scheme>/<model_family>/all_predictions_<RUN>.csv` | Canonical full-dataset prediction export written by `train.py --save-files`. |
| held-out prediction export | `restructured_pipeline/outputs/predictions/<scheme>/<model_family>/all_predictions_<RUN>_predict_inference_test_data.csv` | Test-set export written by `predict.py --inference-mode inference_test_set`. |
| training full-dataset wrong cases | `restructured_pipeline/outputs/wrong_cases/<scheme>/<model_family>/wrong_predictions_<RUN>.xlsx` | Canonical full-dataset wrong-case workbook written by `train.py --save-files`. |
| held-out prediction wrong cases | `restructured_pipeline/outputs/wrong_cases/<scheme>/<model_family>/wrong_predictions_<RUN>_predict_inference_test_data.xlsx` | Test-set wrong-case workbook written by `predict.py --inference-mode inference_test_set`. |
| direct or discovery compare workbook for entire data | `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx` | Preferred canonical compare workbook for entire-data-only reporting. |
| test-data compare workbook | `restructured_pipeline/outputs/reports/test_data/compare_wrong_predictions_test_data.xlsx` | Canonical held-out compare workbook. |
| entire-data tolerance workbook | `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_tolerance_report_entire_data.xlsx` | Canonical tolerance workbook for the dedicated entire-data pipeline. |
| test-data tolerance workbook | `restructured_pipeline/outputs/reports/test_data/compare_wrong_predictions_tolerance_report_test_data.xlsx` | Canonical tolerance workbook for the dedicated test-data pipeline. |
| shared compare tolerance workbook | `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all.xlsx` | Canonical output for the standalone `--generate-tolerance-report` workflow. |
| entire-data important-failure workbook | `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx` | Canonical important-failure workbook for the dedicated entire-data pipeline. |
| test-data important-failure workbook | `restructured_pipeline/outputs/reports/test_data/compare_wrong_predictions_tolerance_important_failures_test_data.xlsx` | Canonical important-failure workbook for the dedicated test-data pipeline. |
| shared important-failure workbook | `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_important_failures.xlsx` | Canonical output for standalone important-failure filtering on the shared tolerance workbook. |
| filtered important-failure workbook for failure-analysis enrichment | `restructured_pipeline/outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data.xlsx` | Canonical enrich-mode input workbook consumed by `failure_analysis_summary.py --mode enrich`. |
| filtered failure-analysis summary | `restructured_pipeline/outputs/reports/failure_analysis_summary_all_scheme_entire_data.xlsx` | Canonical filtered six-sheet enrichment workbook. |
| unfiltered failure-analysis summary | `restructured_pipeline/outputs/reports/failure_analysis_summary_unfiltered_all_scheme_entire_data.xlsx` | Canonical unfiltered six-sheet summary workbook. |
| mixed failure-case workbook | `restructured_pipeline/outputs/reports/mixed_failure_case_combinations.xlsx` | Requires explicit `--master-workbook` and `--filtered-workbook` inputs. |

## Compatibility Aliases Kept Temporarily

| Deprecated Alias Or Fallback | Canonical Replacement | Status |
|------|------|------|
| `restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_schema_entire_data.xlsx` | `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx` | Kept for compatibility. Warn when used as the compare output alias. |
| `restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data.xlsx` | `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx` | Kept as an older compare input fallback in downstream scripts. |
| `restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data_pre_consistency_fix.xlsx` | `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx` | Kept only as an old fallback input path. |
| `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report.xlsx` | `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all.xlsx` | Kept as a standalone tolerance-input alias. |
| `restructured_pipeline/outputs/reports/failure_case_analysis_cleaned.xlsx` | `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all_entire_data_in_one_sheet.xlsx` | Kept as a legacy summary-mode input candidate. |
| `restructured_pipeline/outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data_fixed.xlsx` | `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all_entire_data_in_one_sheet.xlsx` | Kept as a legacy summary-mode input candidate. |
| `restructured_pipeline/outputs/reports/compare_wrong_predictions_test_data.xlsx` | `restructured_pipeline/outputs/reports/test_data/compare_wrong_predictions_test_data.xlsx` | Kept as a fallback input for summary-mode test-data comparison. |
| `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_important_failures.xlsx` | `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx` | Kept as a fallback input for filtered enrichment. |
| staged `expand_and_split/` and `split_and_expand/` compare inputs created by `run_full_pipeline.sh` | direct `--wrong-predictions-file` or `--discover-entire-data-files` compare inputs | Kept until wrapper-driven compare is rewritten away from the staging shim. |

## Warning Coverage

Warnings are expected only when a deprecated alias is selected.

Current warning behavior:

- `compare_wrong_predictions.py`
  warns when the deprecated `all_schema` compare output alias is used
- `compare_wrong_predictions.py`
  warns when the old `compare_wrong_predictions_tolerance_report.xlsx` input
  alias is used for important-failure generation
- `failure_case_filter.py`
  warns when the deprecated `all_schema` compare workbook is selected as the
  default input
- `failure_analysis_summary.py`
  warns when a deprecated fallback candidate or an explicitly supplied
  deprecated compatibility path is used

Canonical paths should not emit compatibility warnings.

## Entire-Data Prediction Export Note

`predict.py --inference-mode inference_entire_data` no longer rewrites a second
full-dataset CSV/XLSX pair when the training-time full-dataset outputs already
exist and match exactly.

Current behavior:

- canonical training outputs remain the source of truth
- duplicate predict-named entire-data paths are created only as compatibility
  aliases when reuse is possible
- `--force-export` preserves the old behavior when a separate duplicate export
  is explicitly needed

## Compatibility-Removal Checklist

Before removing any alias or fallback:

1. Update all runnable docs to use the canonical path only.
2. Confirm `README.md`, `agents.md`, and workflow docs no longer recommend the alias.
3. Re-run:
   - entire-data compare pipeline
   - test-data compare pipeline
   - standalone tolerance report
   - standalone important-failure report
   - `failure_analysis_summary.py --mode enrich`
4. Confirm no active wrapper or helper still depends on the alias.
5. Confirm no generated workbook discovery code still searches the alias.
6. Confirm canonical-path runs produce no warnings.
7. Confirm deprecated-path runs still warn before the alias is removed.
8. Remove only one alias family at a time:
   - `all_schema` compare typo
   - old compare workbook fallbacks
   - tolerance-report old filename
   - legacy failure-analysis input candidates
   - top-level test-data compare fallback
9. Re-run smoke tests after each removal.
10. Do not remove the `run_full_pipeline.sh` staging shim until the wrapper has
    been migrated to direct compare inputs and revalidated.
