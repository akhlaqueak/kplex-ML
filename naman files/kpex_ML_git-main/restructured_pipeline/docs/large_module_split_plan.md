# Large Module Split Plan

Date: 2026-06-23

Scope:

- `restructured_pipeline/src/compare_wrong_predictions.py`
- `restructured_pipeline/src/failure_analysis_summary.py`
- `restructured_pipeline/src/train.py`
- `restructured_pipeline/src/models/hierarchical.py`
- `restructured_pipeline/src/models/hierarchical_scheme3.py`

This phase is planning only.

No runtime code is split here.

## Goals

- reduce file size and responsibility overlap
- isolate I/O from dataframe logic
- make smoke-test coverage more targeted
- lower the risk of future cleanup in duplicate-heavy areas

## Recommended Next Target

Start with:

- `src/compare_wrong_predictions.py`

Why this first:

- it is the most operationally busy reporting entry point
- recent phases already stabilized its direct-file flow and in-memory pipeline
- its responsibilities are easier to separate cleanly into parsing, compare,
  tolerance, important-failure, workbook writing, and CLI layers
- it has the highest payoff for future cleanup in `run_full_pipeline.sh`

## Risk Ranking

| Module | Size | Risk | Why |
|------|------|------|------|
| `src/compare_wrong_predictions.py` | `1820` lines | `MEDIUM-HIGH` | Heavy CLI surface and compatibility paths, but responsibilities are now more separable than before. |
| `src/failure_analysis_summary.py` | `711` lines | `HIGH` | Mixes active enrich flow, summary mode, validation, and compatibility fallbacks. The older legacy helper cluster has already been removed. |
| `src/train.py` | `1113` lines | `HIGH` | Central entry point shared by evaluation/prediction via saved-model helpers; mistakes ripple into the whole pipeline. |
| `src/models/hierarchical.py` | `1079` lines | `HIGH` | Interleaves classifier internals, preprocessing, fit/evaluate logic, and reporting helpers. |
| `src/models/hierarchical_scheme3.py` | `915` lines | `HIGH` | Same coupling pattern as Scheme 2, plus router-specific logic that must stay behavior-identical. |

## Proposed Module Structure

### `src/compare_wrong_predictions.py`

Current responsibilities:

- CLI parsing and mode dispatch
- staged-root and direct-file input loading
- metadata derivation from filenames
- strict compare dataframe construction
- prediction-detail recomputation and validation
- tolerance dataframe generation
- important-failure dataframe generation
- workbook writing and formatting
- dedicated test-data and entire-data pipeline orchestration
- compatibility path handling and warnings

Proposed modules:

- `src/compare_wrong_predictions.py`
  thin entry point and compatibility re-export layer
- `src/compare_pipeline.py`
  top-level orchestration for compare, test-data pipeline, and entire-data pipeline
- `src/compare_inputs.py`
  path parsing, discovery, direct-file normalization, metadata derivation
- `src/compare_core.py`
  row collection, prediction-detail rebuilding, severity derivation, dataframe assembly
- `src/compare_reports.py`
  workbook writing, sheet formatting, summary-sheet writing, validation helpers
- `src/compare_tolerance.py`
  tolerance sheet generation and validation
- `src/compare_important_failures.py`
  important-failure filtering and report writing

Functions to move:

- `compare_inputs.py`
  `parse_named_paths`, `parse_named_files`, `discover_wrong_predictions_files`,
  `infer_strategy_from_filename`, `_has_compare_metadata`,
  `_has_split_metadata`, `derive_scheme_and_model_columns`,
  `normalize_direct_file_metadata`, `resolve_output_path`,
  `resolve_tolerance_output_path`,
  `resolve_tolerance_important_failures_input_path`,
  `resolve_tolerance_important_failures_output_path`
- `compare_core.py`
  `collect_rows_meeting_threshold`, `collect_rows_from_wrong_predictions_file`,
  `derive_failure_severity`, `get_algorithm_columns`,
  `_build_ranked_algorithm_predictions`,
  `_recompute_prediction_details_for_row`,
  `build_prediction_consistency_diagnostics`,
  `validate_prediction_consistency`, `add_prediction_detail_columns`,
  `build_model_comparison_dataframe`,
  `build_file_comparison_dataframe`,
  `build_compare_wrong_predictions_dataframe`,
  `build_compare_wrong_predictions_from_files`,
  `finalize_comparison_dataframe`,
  `build_combined_dataframe_from_args`,
  `build_pipeline_dataframe_from_saved_files`
- `compare_reports.py`
  `autosize_columns`, `highlight_rows_by_failure_severity`,
  `highlight_abnormal_time_gap_cells`, `write_by_inference_on`,
  `_write_formatted_named_sheets`, `_ensure_columns`,
  `_resolve_runtime_column`, `_write_summary_sheet`,
  `write_formatted_workbook_from_existing`
- `compare_tolerance.py`
  `validate_consolidated_tolerance_report`,
  `generate_tolerance_reports`, `build_tolerance_sheet_frames`,
  `write_tolerance_sheet_frames`, `generate_consolidated_tolerance_report`
- `compare_important_failures.py`
  `filter_tolerance_sheet_for_important_failures`,
  `build_important_failure_sheet_frames`,
  `write_important_failures_report_from_tolerance_frames`,
  `generate_tolerance_important_failures_report`
- `compare_pipeline.py`
  `validate_test_data_rows`, `validate_entire_data_rows`,
  `resolve_test_data_output_paths`, `resolve_entire_data_output_paths`,
  `generate_compare_test_data`, `generate_tolerance_report_test_data`,
  `generate_important_failures_test_data`, `run_test_data_pipeline`,
  `generate_compare_entire_data`, `generate_tolerance_report_entire_data`,
  `generate_important_failures_entire_data`, `run_entire_data_pipeline`,
  `run_compare_mode`

Order of movement:

1. Move pure workbook-formatting helpers.
2. Move input/path parsing and metadata helpers.
3. Move pure compare-core dataframe builders.
4. Move tolerance helpers.
5. Move important-failure helpers.
6. Move dedicated pipeline orchestration.
7. Leave CLI parser and `main()` in the entry point until the end.

Risks:

- filename-derived metadata can silently drift
- output workbook names and sheet names must remain unchanged
- warning behavior for compatibility aliases must not regress
- direct-file and `--model-root` modes must both continue working

Smoke tests required:

- direct-file compare using training + held-out wrong-case inputs
- `--generate-tolerance-report --generate-important-failure-report`
- `--run-test-data-pipeline`
- `--run-entire-data-pipeline`
- wrapper run through `run_full_pipeline.sh` in direct mode
- deprecated-path warning check for `all_schema` compare alias

### `src/failure_analysis_summary.py`

Current responsibilities:

- summary-mode workbook selection and aggregation
- enrich-mode six-sheet filtered workbook generation
- runtime lookup construction from project config
- top-k candidate parsing and formatting
- workbook writing and validation
- old dataset-summary and heatmap helper cluster
- compatibility-path fallback selection
- CLI parsing and mode dispatch

Proposed modules:

- `src/failure_analysis_summary.py`
  thin entry point and compatibility re-export layer
- `src/failure_summary_paths.py`
  candidate-path resolution and compatibility warnings
- `src/failure_summary_summary_mode.py`
  summary-mode loading, counts, scheme summaries, workbook build
- `src/failure_summary_enrich.py`
  enrich-mode dataframe construction and six-sheet export
- `src/failure_summary_topk.py`
  top-k tuple parsing, predicted-probability extraction, decision formatting
- `src/failure_summary_validation.py`
  post-write validation helpers
- `src/failure_summary_legacy.py`
  currently unused dataset-summary, heatmap, and green-highlight helpers

Functions to move:

- `failure_summary_paths.py`
  `resolve_existing_path`, `resolve_explicit_or_existing_path`,
  `load_entire_important`, `load_entire_base`, `load_test_compare`
- `failure_summary_summary_mode.py`
  `percent`, `normalize_severity_label`, `resolve_failure_severity_mask`,
  `compute_important_subset_from_compare`, `build_scheme_summary`,
  `write_section`, `autosize`, `build_workbook`, `validate_counts`,
  `run_summary_mode`
- `failure_summary_enrich.py`
  `_build_runtime_lookup`, `_resolve_runtime_row`,
  `_build_sorted_prediction_entries`, `_select_best_topk_entry`,
  `_build_failure_analysis_row`, `_build_failure_analysis_sheet`,
  `_load_entire_failure_dataframe`, `_write_failure_analysis_workbook`,
  `run_enrichment_mode`
- `failure_summary_topk.py`
  `_safe_float`, `_round_or_none`, `_format_float`,
  `_format_candidate_tuple`, `_format_fastest_tuple`,
  `_format_all_predictions`, `_parse_top3_item`,
  `parse_top3_predicted_detailed`, `derive_predicted_probability`,
  `select_best_recovery_entry`, `format_decision_tuple`,
  `parse_decision_tuple`, `extract_time_ratio_from_decision_value`
- `failure_summary_validation.py`
  `_validate_failure_analysis_workbook`
- `failure_summary_legacy.py`
  No longer a current split target. The older legacy helper cluster referenced
  in earlier audits has already been removed from `src/failure_analysis_summary.py`.

Order of movement:

1. Move path-resolution helpers.
2. Move top-k parsing/formatting helpers.
3. Move enrich-mode dataframe build and write helpers.
4. Move summary-mode helpers.
5. Move validation helpers.
6. Move legacy unused cluster into a clearly named legacy module last.

Risks:

- enrich-mode workbook column order is highly sensitive
- summary mode and enrich mode share similar but different assumptions
- current compatibility candidates must remain intact until explicitly removed
- legacy helper cluster may still be used outside the documented pipeline

Smoke tests required:

- `failure_analysis_summary.py --mode enrich`
- summary mode using canonical filtered and compare inputs
- row-count comparison against a known-good filtered workbook
- check that all six enrich sheets still exist and contain no blank `scheme`/`model`

### `src/train.py`

Current responsibilities:

- config loading and normalization
- output-directory creation
- run-name creation
- preprocessing and feature preparation
- flat results construction
- wrong-prediction extraction
- artifact writing
- confusion matrix generation
- model artifact save/load
- saved-run result reconstruction used by other entry points
- scheme-specific training for Scheme 1, 2, and 3
- CLI parsing and dispatch

Proposed modules:

- `src/train.py`
  thin entry point and top-level dispatch
- `src/train_config.py`
  config loading, normalization, run-name helpers, output-dir helpers
- `src/train_outputs.py`
  csv/xlsx output writers, metrics writers, split-index writers
- `src/train_confusion.py`
  confusion-label and confusion-matrix helpers
- `src/train_saved_run.py`
  model artifact save/load and result reconstruction helpers
- `src/train_scheme1.py`
  Scheme 1 training flow
- `src/train_scheme2.py`
  Scheme 2 training flow
- `src/train_scheme3.py`
  Scheme 3 training flow
- optional later:
  `src/train_results.py`
  if flat result-building and wrong-case extraction grows further

Functions to move:

- `train_config.py`
  `load_yaml_config`, `normalize_scheme`, `normalize_model_family`,
  `resolve_model_config_path`, `ensure_output_dirs`, `create_run_name`,
  `load_and_preprocess_data`
- `train_outputs.py`
  `save_dataframe_outputs`, `save_metrics_outputs`,
  `save_hierarchical_stage_outputs`, `save_flat_reports`,
  `save_split_indices`
- `train_confusion.py`
  `build_confusion_labels`, `save_confusion_matrix`,
  `save_scheme2_confusion_matrices`, `save_scheme3_confusion_matrices`
- `train_saved_run.py`
  `save_model_artifact`, `resolve_model_artifact_path`,
  `load_model_artifact`, `build_results_from_model_artifact`
- `train_results.py` or leave in entry point temporarily
  `build_flat_results`, `build_wrong_predictions`
- `train_scheme1.py`
  `train_scheme1`
- `train_scheme2.py`
  `train_scheme2`
- `train_scheme3.py`
  `train_scheme3`

Order of movement:

1. Move config and output helpers.
2. Move confusion helpers.
3. Move saved-run helpers used by evaluate/predict.
4. Move scheme-specific training functions.
5. Decide whether to keep `build_flat_results` and `build_wrong_predictions`
   in `train.py` or promote them to `train_results.py`.
6. Shrink `run_training()` and keep `main()` last.

Risks:

- `evaluate.py` and `predict.py` rely on saved-run helper stability
- Scheme 1 variant handling is tightly coupled to config semantics
- model artifact contents must not change shape
- output naming must remain stable for downstream compare/failure scripts

Smoke tests required:

- Scheme 1 RF training with `--save-files`
- Scheme 2 XGB training with `--save-files`
- Scheme 3 RF training with `--save-files`
- held-out evaluation using the saved artifact
- entire-data prediction using the saved artifact
- wrapper run for one representative model

### `src/models/hierarchical.py`

Current responsibilities:

- scheme-specific classifier class
- split helper selection and column normalization
- feature preparation and label encoding
- model builders for RF and XGB
- fit flows for RF and XGB
- stage-level evaluation and probability assembly
- full evaluation/report export

Proposed modules:

- `src/models/hierarchical.py`
  thin compatibility module exporting the public Scheme 2 API
- `src/models/hierarchical_core.py`
  classifier class and shared runtime helpers
- `src/models/hierarchical_data.py`
  split-function resolution, column normalization, train/test preparation
- `src/models/hierarchical_builders.py`
  RF and XGB builder functions
- `src/models/hierarchical_fit.py`
  RF and XGB fit functions
- `src/models/hierarchical_eval.py`
  stage evaluation, probability assembly, and full evaluation flow

Functions/classes to move:

- `hierarchical_core.py`
  `get_fastest_algorithms`, `HierarchicalClassifier5`,
  `_prepare_X_values`, `_transform_optional_labels`
- `hierarchical_data.py`
  `_get_hier_split_function`, `_ensure_scheme2_columns`,
  `prepare_train_test_data_hier`, `create_label_encoders`
- `hierarchical_builders.py`
  `build_hierarchical_random_forest_classifier`,
  `build_hierarchical_xgb_classifier`,
  `_resolve_stage_local_num_class`
- `hierarchical_fit.py`
  `fit_hierarchical_random_forest_classifier`,
  `fit_hierarchical_xgb_classifier`
- `hierarchical_eval.py`
  `_decode_leaf_classes`, `_add_stage_probabilities`,
  `evaluate_hierarchical_stages`, `run_full_evaluation_hier5`,
  `save_stage_reports`

Order of movement:

1. Move pure data-prep and builder helpers.
2. Move evaluation helpers.
3. Move fit functions.
4. Move the classifier class last if imports stay manageable.

Risks:

- staged classifier internals are tightly coupled to fit and eval code
- label encoder behavior must remain identical
- stage probability columns are consumed by downstream reporting

Smoke tests required:

- Scheme 2 RF training
- Scheme 2 XGB training
- saved stage-report outputs
- confusion-matrix generation from Scheme 2 training flow

### `src/models/hierarchical_scheme3.py`

Current responsibilities:

- scheme-specific classifier class
- Scheme 3 split and column normalization
- feature preparation and label encoding
- router-specific RF and XGB builders
- fit flows for RF and XGB
- stage-level evaluation and probability assembly
- full evaluation/export for Scheme 3

Proposed modules:

- `src/models/hierarchical_scheme3.py`
  thin compatibility module exporting the public Scheme 3 API
- `src/models/hierarchical_scheme3_core.py`
  classifier class and shared runtime helpers
- `src/models/hierarchical_scheme3_data.py`
  split-function resolution, column normalization, train/test preparation
- `src/models/hierarchical_scheme3_builders.py`
  RF and XGB builder functions
- `src/models/hierarchical_scheme3_fit.py`
  RF and XGB fit functions
- `src/models/hierarchical_scheme3_eval.py`
  stage evaluation, router decoding, probability assembly, full evaluation

Functions/classes to move:

- `hierarchical_scheme3_core.py`
  `get_fastest_algorithms`, `HierarchicalClassifier3`,
  `_prepare_X_values`, `_transform_optional_labels`
- `hierarchical_scheme3_data.py`
  `_get_hier3_split_function`, `_ensure_scheme3_columns`,
  `create_label_encoders_scheme3`, `prepare_train_test_data_hier3`
- `hierarchical_scheme3_builders.py`
  `build_hierarchical_scheme3_random_forest_classifier`,
  `build_hierarchical_scheme3_xgb_classifier`,
  `_resolve_router_num_class`
- `hierarchical_scheme3_fit.py`
  `fit_hierarchical_scheme3_random_forest_classifier`,
  `fit_hierarchical_scheme3_xgb_classifier`
- `hierarchical_scheme3_eval.py`
  `_decode_router_predictions`, `_add_stage_probabilities`,
  `evaluate_hierarchical_scheme3_stages`,
  `run_full_evaluation_scheme3`

Order of movement:

1. Move split/data helpers.
2. Move builder helpers.
3. Move evaluation helpers.
4. Move fit functions.
5. Move the classifier class last.

Risks:

- Scheme 3 router decoding is easy to break silently
- stage output column names must remain stable
- shared helper extraction with Scheme 2 may look tempting, but should not be
  mixed into the same split unless both modules are already independently stable

Smoke tests required:

- Scheme 3 RF training
- Scheme 3 XGB training
- saved stage-report outputs
- Scheme 3 confusion-matrix generation

## Recommended Split Order Across The Repo

1. `src/compare_wrong_predictions.py`
2. `src/failure_analysis_summary.py`
3. `src/train.py`
4. `src/models/hierarchical.py`
5. `src/models/hierarchical_scheme3.py`

Why this order:

- compare and failure-analysis modules are already the main reporting bottlenecks
- `train.py` should wait until compare/report refactors settle
- hierarchical modules should come after the non-hierarchical training entry
  point is simpler and the saved-run interfaces are stable

## Global Guardrails For The Later Split

- keep the original top-level module paths importing and re-exporting the old
  public names during the transition
- move pure helpers before moving orchestration
- preserve output filenames, sheet names, and CLI flags unless a separate phase
  explicitly changes them
- after each movement step:
  run the smallest smoke test that exercises the moved code
- do not combine:
  module split,
  behavioral cleanup,
  and compatibility-path removal
  in the same implementation phase
