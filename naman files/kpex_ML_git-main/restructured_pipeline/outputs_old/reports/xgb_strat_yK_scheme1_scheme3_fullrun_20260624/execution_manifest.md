# Execution Manifest

- Run label: `xgb_strat_yK_scheme1_scheme3_fullrun_20260624`
- Generated at: `2026-06-24T18:56:01Z`
- Status: `partial_failure`
- Requested scope: `Scheme 1 + Scheme 3`, `XGB only`, `strat_y_K only`

## Scope Validation

- RF training was not run.
- Scheme 2 training was not run.
- `strat_y` training was not run.
- Successful scoped reports contain only Scheme 1 and Scheme 3 XGB nonzero rows.
- Some fixed-shape report workbooks still emit zero-row Scheme 2 and RF sheets.
- Tolerance counts are monotonic: validated.
- Important-failure counts do not exceed their parent tolerance rows: validated.

## Runs Executed

| Scheme | Model | Split | Run Name |
|---|---|---|---|
| `scheme1` | `xgboost` | `strat_y_K` | `scheme1_xgboost_strat_y_K_xgb_strat_yK_scheme1_scheme3_fullrun_20260624` |
| `scheme3` | `xgboost` | `strat_y_K` | `scheme3_xgboost_strat_y_K_xgb_strat_yK_scheme1_scheme3_fullrun_20260624` |

## Command Outcome

| Step | Result | Exit |
|---|---|---|
| Preflight imports/config check | PASS | `0` |
| Train Scheme 1 XGB | PASS | `0` |
| Train Scheme 3 XGB | PASS | `0` |
| Evaluate Scheme 1 test | PASS | `0` |
| Evaluate Scheme 1 entire | PASS | `0` |
| Evaluate Scheme 3 test | PASS | `0` |
| Evaluate Scheme 3 entire | PASS | `0` |
| Predict Scheme 1 test | PASS | `0` |
| Predict Scheme 1 entire | PASS | `0` |
| Predict Scheme 3 test | PASS | `0` |
| Predict Scheme 3 entire | PASS | `0` |
| Compare wrong predictions test | PASS | `0` |
| Compare wrong predictions entire | PASS | `0` |
| Failure-case filter | PASS | `0` |
| Failure-analysis enrich | PASS | `0` |
| Failure-analysis summary mode | PASS | `0` |
| Unfiltered failure-analysis summary | FAIL | `1` |

## Key Outputs

| Output | Notes |
|---|---|
| `restructured_pipeline/outputs/predictions/scheme1/xgboost/all_predictions_scheme1_xgboost_strat_y_K_xgb_strat_yK_scheme1_scheme3_fullrun_20260624.csv` | Training-time full-dataset predictions |
| `restructured_pipeline/outputs/predictions/scheme3/xgboost/all_predictions_scheme3_xgboost_strat_y_K_xgb_strat_yK_scheme1_scheme3_fullrun_20260624.csv` | Training-time full-dataset predictions |
| `restructured_pipeline/outputs/reports/xgb_strat_yK_scheme1_scheme3_fullrun_20260624/compare_wrong_predictions_test_data.xlsx` | Strict compare on held-out test wrong cases |
| `restructured_pipeline/outputs/reports/xgb_strat_yK_scheme1_scheme3_fullrun_20260624/compare_wrong_predictions_entire_data.xlsx` | Strict compare on full-dataset wrong cases |
| `restructured_pipeline/outputs/reports/xgb_strat_yK_scheme1_scheme3_fullrun_20260624/compare_wrong_predictions_tolerance_report_test_data.xlsx` | Tolerance workbook for test compare |
| `restructured_pipeline/outputs/reports/xgb_strat_yK_scheme1_scheme3_fullrun_20260624/compare_wrong_predictions_tolerance_report_entire_data.xlsx` | Tolerance workbook for entire compare |
| `restructured_pipeline/outputs/reports/xgb_strat_yK_scheme1_scheme3_fullrun_20260624/compare_wrong_predictions_tolerance_important_failures_test_data.xlsx` | Important-failure workbook for test compare |
| `restructured_pipeline/outputs/reports/xgb_strat_yK_scheme1_scheme3_fullrun_20260624/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx` | Important-failure workbook for entire compare |
| `restructured_pipeline/outputs/reports/xgb_strat_yK_scheme1_scheme3_fullrun_20260624/failure_case_analysis_important_failures_entire_data.xlsx` | Filtered failure workbook required by enrich mode |
| `restructured_pipeline/outputs/reports/xgb_strat_yK_scheme1_scheme3_fullrun_20260624/failure_analysis_summary_entire_data.xlsx` | Enriched failure workbook |
| `restructured_pipeline/outputs/reports/xgb_strat_yK_scheme1_scheme3_fullrun_20260624/failure_analysis_summary_entire_vs_test.xlsx` | Entire-vs-test summary workbook |
| `restructured_pipeline/outputs/reports/xgb_strat_yK_scheme1_scheme3_fullrun_20260624/failure_analysis_summary_unfiltered_entire_data.xlsx` | Exists but invalid because the generating command failed validation |

## Row Counts

- Compare test: `87` rows on sheet `on 20% test set`
- Compare entire: `89` rows on sheet `on entire dataset`
- Tolerance test:
  - Scheme 1: `48 / 48 / 48 / 48`
  - Scheme 3: `39 / 39 / 39 / 39`
- Important failures test:
  - Scheme 1: `28 / 28 / 28 / 28`
  - Scheme 3: `17 / 17 / 17 / 17`
- Tolerance entire:
  - Scheme 1: `48 / 48 / 48 / 48`
  - Scheme 3: `41 / 41 / 41 / 41`
- Important failures entire:
  - Scheme 1: `28 / 28 / 28 / 28`
  - Scheme 3: `17 / 17 / 17 / 17`
- Filtered failure workbook:
  - `on entire dataset`: `45`
  - `filter_summary`: `2`
- Enriched failure workbook:
  - `scheme_1_XGB`: `28`
  - `scheme_3_XGB`: `17`
  - all RF / Scheme 2 sheets: `0`

## Entire-Data Prediction Export Behavior

- `predict.py --inference-mode inference_entire_data` physically regenerated exports for both Scheme 1 and Scheme 3.
- No symlink or alias reuse was observed for this run.

## Provenance Highlights

| Output Type | Script | CLI Path | Writer Function |
|---|---|---|---|
| Training predictions / training wrong cases | `restructured_pipeline/src/train.py` | `main -> run_training -> train_scheme1/train_scheme3` | `save_dataframe_outputs` |
| Model artifacts | `restructured_pipeline/src/train.py` | `main -> run_training -> train_scheme1/train_scheme3` | `save_model_artifact` |
| Split files | `restructured_pipeline/src/train.py` | `main -> run_training -> train_scheme1/train_scheme3` | `save_split_indices` |
| Evaluation confusion outputs | `restructured_pipeline/src/evaluate.py` | `main -> run_evaluation` | `save_final_reports` |
| Held-out and full-data predict exports | `restructured_pipeline/src/predict.py` | `main -> run_prediction` | `resolve_prediction_output_paths`, `maybe_reuse_canonical_entire_data_outputs`, `train.save_dataframe_outputs` |
| Strict compare | `restructured_pipeline/src/compare_wrong_predictions.py` | `main -> run_compare_mode` | `build_compare_wrong_predictions_from_files -> write_by_inference_on` |
| Tolerance reports | `restructured_pipeline/src/compare_wrong_predictions.py` | `main -> run_compare_mode` | `build_tolerance_sheet_frames -> write_tolerance_sheet_frames` |
| Integrated important failures | `restructured_pipeline/src/compare_wrong_predictions.py` | `main -> run_compare_mode` | `build_important_failure_sheet_frames -> write_important_failures_report_from_tolerance_frames` |
| Filtered important failures | `restructured_pipeline/src/failure_case_filter.py` | `main -> run_failure_case_filter` | `write_summary_sheet` |
| Enriched summary | `restructured_pipeline/src/failure_analysis_summary.py` | `main -> run_enrichment_mode` | `run_enrichment_mode` |
| Entire-vs-test summary | `restructured_pipeline/src/failure_analysis_summary.py` | `main -> run_summary_mode` | `run_summary_mode` |
| Unfiltered summary | `restructured_pipeline/src/failure_analysis_unfiltered_summary.py` | `main` | `load_unfiltered_records -> write_workbook -> validate_output` |

## Warnings

- XGBoost emitted `use_label_encoder` unused-parameter warnings during training.
- Entire-data compare computed a missing `time_gap_ratio` for the Scheme 3 training wrong-case workbook.
- Fixed-shape report workbooks still emit zero-row RF and Scheme 2 sheets.

## Failure And Stop Condition

- Failed command:
  - `restructured_pipeline/src/failure_analysis_unfiltered_summary.py`
- Exact error:
  - `ValueError: Unfiltered sheet scheme_3_XGB has fewer rows than filtered reference: 0 < 17`
- Failed output:
  - `restructured_pipeline/outputs/reports/xgb_strat_yK_scheme1_scheme3_fullrun_20260624/failure_analysis_summary_unfiltered_entire_data.xlsx`
- Status of failed output:
  - File exists, but it is not a valid successful artifact and should not be used.

## Skipped Downstream Steps

- `restructured_pipeline/src/mixed_failure_case_combinations.py`
- any XGB-only corrected mixed-failure workflow

Reason:
Execution stopped immediately after the unfiltered summary command failed, per requested stop-on-failure behavior.

## Recommended Next Action

Inspect and fix the Scheme 3 ingestion or validation path inside `restructured_pipeline/src/failure_analysis_unfiltered_summary.py`, then rerun only the failed unfiltered summary step and the downstream mixed-failure workflows.
