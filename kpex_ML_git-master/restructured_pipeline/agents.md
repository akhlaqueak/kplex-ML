# Refactor Continuation Guide

## Context

- Goal: keep moving the ML workflow into a config-driven pipeline.
- Legacy notebooks and old code remain intact.
- New work stays inside `restructured_pipeline/`.

## Hard Constraints

- Reuse existing code and logic first.
- Keep function names and variable names consistent where possible.
- Make minimal changes only.
- Do not delete, move, or rename legacy notebook files.
- Keep the runnable path inside the new architecture.

## Active Dataset Assumptions

- Primary input file: `input_data/ml-data.csv`
- The dataset contains both BnB times and total times.
- All supervised target construction, training, evaluation, prediction, and
  failure-case runtime metrics must use the BnB columns only.
- Total-time columns may remain in exported dataframes for compatibility, but
  they must not drive `y_algo`, `fastest_algorithms`, `correct`,
  `predicted_time`, or `time_gap_ratio`.

Active BnB labels:

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

## Current Status

- Shared pipeline modules are implemented.
  Files: `src/features.py`, `src/data_splits.py`, `src/metrics.py`,
  `src/write_utils.py`
- Scheme 1, Scheme 2, and Scheme 3 model runners are implemented inside the
  active pipeline.
  Files: `src/models/flat.py`, `src/models/hierarchical.py`,
  `src/models/hierarchical_scheme3.py`
- Training, evaluation, and prediction entry points are implemented.
  Files: `src/train.py`, `src/evaluate.py`, `src/predict.py`
- Full experiment-matrix orchestration is implemented.
  Files: `src/experiment_matrix.py`, `src/tolerance_reports.py`
- Failure-case analysis is implemented as Python code.
  File: `src/compare_wrong_predictions.py`
- Primary docs are kept inside `restructured_pipeline/`.
  Files: `README.md`, `docs/comparison_wrong_predictions.md`, `agents.md`

## Documentation Pointers

- `README.md`
  Main operator doc for training, evaluation, prediction, output locations,
  naming rules, and the runnable command matrix.
- `docs/comparison_wrong_predictions.md`
  Wrong-predictions comparison inputs, thresholds, highlighting, and the
  consolidated full-dataset workflow.
- `agents.md`
  Current implementation status, constraints, and the active pipeline path.

## End-To-End Flow

- Train with `src/train.py`
- Re-evaluate saved outputs with `src/evaluate.py`
- Re-export held-out test predictions and wrong cases with `src/predict.py`
- Re-export full-dataset predictions and wrong cases with `src/predict.py`
- Run failure-case analysis and Excel reporting with
  `src/compare_wrong_predictions.py`

The intended pipeline path remains end-to-end from training through
failure-case analysis.

Saved-run naming rule:

- training keeps `all_predictions_<RUN_NAME>.csv` as the base full-dataset
  export written by `train.py --save-files`
- held-out evaluation and prediction outputs use `inference_test_data` in
  generated filenames
- full-dataset evaluation and prediction outputs use `inference_entire_data`
  in generated filenames

## Runnable Combinations

- Scheme 1 supports:
  `random_forest/rf` and `xgboost/xgb`
- Scheme 2 supports:
  `random_forest/rf` and `xgboost/xgb`
- Scheme 3 supports:
  `random_forest/rf` and `xgboost/xgb`
- All three schemes support both split modes:
  `strat_y` and `strat_y_K`
- Scheme 1 variant mapping is fixed in code and in the wrapper:
  - `random_forest + strat_y -> rf_strat_y`
  - `random_forest + strat_y_K -> rf_strat_y_K`
  - `xgboost + strat_y -> xgb_strat_y`
  - `xgboost + strat_y_K -> xgb_strat_y_K`

## Split Options

- `strat_y`: split from the BnB fastest label
- `strat_y_K`: split from the BnB fastest label and `K`

Compatibility rule:

- `strat_y_K` still prefers combined `target + K` stratification
- if a combined class would make the split fail, those row(s) are forced into
  the training set first and the remaining rows still use true `target + K`
  stratification

These split modes are already wired through the current pipeline:

- Scheme 1:
  `prepare_train_test_strat_Y`, `prepare_train_test_strat_Y_K`
- Scheme 2:
  `prepare_train_test_strat_Y_hier`, `prepare_train_test_strat_Y_K_hier`
- Scheme 3:
  `prepare_train_test_strat_Y_hier3`, `prepare_train_test_strat_Y_K_hier3`

## Failure-Case Analysis Flow

- Read `wrong_predictions_*.xlsx`
- Support either staged strategy folders or direct saved wrong-predictions
  workbook paths as input
- Support direct full-dataset consolidation from disk with
  `--discover-entire-data-files`
- Reuse shared helpers for filename parsing and aggregation
- Rebuild `time_gap_ratio` when the source file is missing derived BnB time-gap
  columns
- Collect rows where `time_gap_ratio >= threshold`
- Add the analysis columns used for failure-case review
- Export one Excel workbook with one sheet per `inference_on`
- Apply Excel-only highlighting in the reporting layer
- Optionally export a second tolerance workbook with one sheet per
  `(scheme, tolerance)` pair

Active comparison workflows:

- Wrapper-driven per-run comparison:
  `run_full_pipeline.sh` stages training wrong cases plus held-out prediction
  wrong cases, then calls `src/compare_wrong_predictions.py --model-root ...`
- Direct saved-workbook comparison:
  `src/compare_wrong_predictions.py --wrong-predictions-file ...`
- Consolidated full-dataset comparison across saved files:
  `src/compare_wrong_predictions.py --discover-entire-data-files`
  writes
  `restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_schema_entire_data.xlsx`

Highlighting rule in the active reporting layer:

- row severity fill is driven by `failure_severity`
- yellow fill is applied only to the `time_gap` cell when
  `time_gap > --time-gap-highlight-threshold`
  and the default highlight threshold is `100.0`

Tolerance-report rule in the active reporting layer:

- optional flag:
  `src/compare_wrong_predictions.py --generate-tolerance-report`
- workbook path:
  `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all.xlsx`
- supported tolerances:
  `0`, `10`, `20`, `50`
- sheet names:
  `scheme_1_tol_0`,
  `scheme_1_tol_10`,
  `scheme_1_tol_20`,
  `scheme_1_tol_50`,
  `scheme_2_tol_0`,
  `scheme_2_tol_10`,
  `scheme_2_tol_20`,
  `scheme_2_tol_50`,
  `scheme_3_tol_0`,
  `scheme_3_tol_10`,
  `scheme_3_tol_20`,
  `scheme_3_tol_50`
- keep rows where:
  `predicted_time > fastest_runtime * (1 + tolerance / 100)`
- the standalone `--generate-tolerance-report` path operates on the dataframe
  supplied by the compare stage
- the dedicated `--run-test-data-pipeline` and
  `--run-entire-data-pipeline` paths rebuild the tolerance base from saved
  wrong-predictions files with `threshold=0.0`, then apply the tolerance masks
  independently per sheet
- validation must confirm:
  exact 12 sheets,
  exact exported column order,
  no duplicated exported rows,
  and non-increasing row counts as tolerance increases
- dedicated test/entire pipelines must not silently reuse the already-strict
  compare subset for all tolerance sheets
- formatting rule:
  reuse the standard reporting layer, including severity row fills, and
  highlight `time_gap` cells above threshold
- this is an additional workbook only; the normal compare workbook is not
  replaced or renamed

Second-stage important-failure rule:

- optional flag:
  `src/compare_wrong_predictions.py --generate-important-failure-report`
- default input candidates:
  `compare_wrong_predictions_tolerance_report_all.xlsx`,
  `compare_wrong_predictions_tolerance_report.xlsx`
- workbook path:
  `restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_important_failures.xlsx`
- relationship:
  `compare workbook -> tolerance workbook -> important-failure workbook`
- pipeline order is fixed:
  strict compare dataset first, tolerance second, important-failure filtering
  last
- keep rows only when they are not excluded by either rule:
  `fastest_runtime < 5.0 and predicted_runtime < 5.0`
  or
  `predicted_runtime <= 1.2 * fastest_runtime`
- runtime fallback:
  use `fastest_time` / `predicted_time` when runtime-named columns are absent
- preserve the same 12 scheme/tolerance sheets and append a `filter_summary`
  sheet for per-sheet counts

Test-data wrong-predictions pipeline:

- optional flag:
  `src/compare_wrong_predictions.py --run-test-data-pipeline`
- source files:
  saved `wrong_predictions_*predict_inference_test_data*.xlsx`
- output folder:
  `restructured_pipeline/outputs/reports/test_data/`
- output files:
  `compare_wrong_predictions_test_data.xlsx`,
  `compare_wrong_predictions_tolerance_report_test_data.xlsx`,
  `compare_wrong_predictions_tolerance_important_failures_test_data.xlsx`
- behavior:
  mirrors the entire-dataset pipeline in the same order and must remain fully
  separated from the entire-dataset outputs
- compare workbook uses the configured strict threshold
- tolerance workbook must be rebuilt from the full saved
  `predict_inference_test_data` wrong-predictions pool with `threshold=0.0`
  before per-tolerance filtering
- validation:
  all rows must belong to `inference_on = on 20% test set`

Entire-data wrong-predictions pipeline:

- optional flag:
  `src/compare_wrong_predictions.py --run-entire-data-pipeline`
- source files:
  saved `wrong_predictions_*predict_inference_entire_data*.xlsx`
- output folder:
  `restructured_pipeline/outputs/reports/entire_data/`
- output files:
  `compare_wrong_predictions_entire_data.xlsx`,
  `compare_wrong_predictions_tolerance_report_entire_data.xlsx`,
  `compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx`
- behavior:
  mirrors the compare -> tolerance -> important-failure order and must remain
  separated from test-data outputs
- compare workbook uses the configured strict threshold
- tolerance workbook must be rebuilt from the full saved
  `predict_inference_entire_data` wrong-predictions pool with `threshold=0.0`
  before per-tolerance filtering
- validation:
  all rows must belong to `inference_on = on entire dataset`

Failure-analysis enrichment:

- reporting-only entry point:
  `src/failure_analysis_summary.py --mode enrich`
- does not modify training, prediction, or core compare logic
- default input workbook:
  `outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data.xlsx`
- default output workbook:
  `outputs/reports/failure_analysis_summary_all_scheme_entire_data.xlsx`
- rerunning the script regenerates this workbook from scratch
- final sheet set is fixed:
  - `scheme_1_RF`
  - `scheme_1_XGB`
  - `scheme_2_RF`
  - `scheme_2_XGB`
  - `scheme_3_RF`
  - `scheme_3_XGB`
- no tolerance sheets are generated by this reporting layer
- no highlighting is applied in this workbook
- per-row top-K recovery view:
  - `top1_min_time_ratio` through `top6_min_time_ratio`
  - `top1_candidate` through `top6_candidate`
- `top<K>_candidate` uses tuple format:
  `(predicted_time, predicted_algo, prob)`
- `top<K>_min_time_ratio` is computed from that same candidate as:
  `predicted_time / fastest_time`
- `all_predictions_sorted_by_prob` stores the full ranked tuple list:
  `(time_ratio, predicted_algo, predicted_time, prob)`
- `fastest_algo_tuple` stores:
  `(fastest_time, fastest_algo, rank)`
- `rank` is the 1-based position where the fastest algorithm first appears in
  the probability-sorted prediction order
- removed from this workbook:
  `inference_on`, `with_holdout`, `strat_on`, `scheme`,
  `source_model`, `failure_case_filter_reason`,
  `fastest_algo`, `fastest_time`,
  `predicted_algo`, `predicted_time`,
  `predicted_probability`, `actual_time_ratio`,
  `failure_severity`, `time_gap`, `time_gap_ratio`,
  all `prob_...` columns, and tolerance-sheet metadata

Mixed failure-case combinations:

- reporting-only entry point:
  `src/mixed_failure_case_combinations.py`
- default output workbook:
  `outputs/reports/mixed_failure_case_combinations.xlsx`
- uses the unfiltered six-sheet workbook only for split/source mapping:
  `failure_analysis_summary_unfiltered_with_generated_filtered_reference.xlsx`
- uses the filtered six-sheet workbook as the failure pool:
  `failure_analysis_summary_filtered_all_scheme_entire_data_from_one_sheet.xlsx`
- allowed scheme pairings only:
  - `scheme 1 + scheme 2`
  - `scheme 1 + scheme 3`
- forbidden pairing:
  - `scheme 2 + scheme 3`
- exported base sheets include:
  - `scheme1_top1__scheme2_top1`
  - `scheme1_top1__scheme2_top2`
  - `scheme1_top2__scheme2_top1`
  - `scheme1_top2__scheme2_top2`
  - `scheme1_top1__scheme3_top1`
  - `scheme1_top1__scheme3_top2`
  - `scheme1_top2__scheme3_top1`
  - `scheme1_top2__scheme3_top2`
- additional base variants also include `top3` on either side:
  - `scheme1_top1__scheme2_top3`
  - `scheme1_top2__scheme2_top3`
  - `scheme1_top3__scheme2_top1`
  - `scheme1_top3__scheme2_top2`
  - same pattern for `scheme 3`
- threshold-aware variants use compact Excel-safe names:
  - `s1_t1__s2_t1__thr1p5`
  - `s1_t1__s2_t1__thr2p0`
- split-aware variants use:
  - `__straty`
  - `__stratyK`
  - `__entire`
- model-aware variants use:
  - `__RF`
  - `__XGB`
- current source inputs are entire-data only, so no `test_data` sheets are
  generated
- `source_split_label` is mapped from the master workbook
- rows are kept only when the selected `top1`, `top2`, or `top3` minimum
  ratio is strictly greater than the sheet threshold
- selected-candidate fields are reconstructed from:
  `top1_candidate`, `top2_candidate`, `top3_candidate`, `fastest_algo_tuple`,
  and `all_predictions_sorted_by_prob`

XGB-only mixed failure-case workbook:

- reporting-only entry point:
  `src/mixed_failure_case_combinations_xgb_only.py`
- default output workbook:
  `outputs/reports/mixed_failure_case_combinations_xgb_only.xlsx`
- source workbook:
  `failure_analysis_summary_filtered_all_scheme_entire_data_from_one_sheet.xlsx`
- only reads:
  - `scheme_1_XGB`
  - `scheme_2_XGB`
  - `scheme_3_XGB`
- allowed pairings only:
  - `scheme 1 + scheme 2`
  - `scheme 1 + scheme 3`
- forbidden pairing:
  - `scheme 2 + scheme 3`
- each mixed sheet applies a minimum-ratio threshold before export
- rows are further restricted to:
  `strat_y_K`
- active filtering rule:
  for each sheet side, use the top-k named in that sheet and require:
  selected `topK_min_time_ratio > 1.20`
- example:
  `scheme1_top3__scheme3_top2` means
  `scheme 1` rows must satisfy `top3_min_time_ratio > 1.20`
  and `scheme 3` rows must satisfy `top2_min_time_ratio > 1.20`
- this rule is applied uniformly across all XGB-only mixed sheets
- active threshold rule:
  selected minimum ratio `> 1.20`
- generated sheet families:
  - base top1/top2 pairings
  - top-3 variants
- no split-aware variants
- no `RF` rows
- workbook starts with a `summary` sheet containing:
  `sheet_name`, `scheme_pair`, `topk_pairing`, `row_count`,
  `pairing_family`, `rows_before_filter`, `rows_after_filter`,
  `rows_removed`, `min_ratio_threshold`, `required_split_label`,
  and `filter_rule`
- workbook also adds `shared_failure_summary`
- shared-failure rule:
  within any mixed sheet, if both allowed source schemes appear for the same
  `(dataset, k_value)` pair, that overlap is a shared failure
- highlighting rule:
  highlight the full row yellow only for those shared-failure overlap rows
- `shared_failure_summary` records:
  `sheet_name`, `scheme_pair`, `total_rows`, `shared_failure_pairs`,
  `highlighted_rows`, `unique_datasets_with_shared_failures`,
  `unique_dataset_k_overlap_cases`, and top repeated overlap pairs
- exact exported columns:
  `dataset`, `k_value`, `scheme`,
  `fastest_algo_rank`, `fastest_algo_tuple`,
  `top1_min_time_ratio`, `top1_candidate`,
  `top2_min_time_ratio`, `top2_candidate`,
  `top3_min_time_ratio`, `top3_candidate`,
  `top4_min_time_ratio`, `top4_candidate`,
  `top5_min_time_ratio`, `top5_candidate`,
  `top6_min_time_ratio`, `top6_candidate`,
  `all_predictions_sorted_by_prob`,
  `source_scheme`, `source_rule`

Split-count reporting workbook:

- reporting-only entry point:
  `src/failure_counts_by_split_scheme_model.py`
- output workbook:
  `outputs/reports/failure_counts_by_split_scheme_model.xlsx`
- purpose:
  count failures by `split_mode`, `scheme`, `model`, and partition
  (`train`, `test`, `unknown`)
- `important_failures_split_counts` uses the filtered one-sheet
  important-failure workbook and matches rows back to normalized wrong-case
  exports to recover split mode and partition
- `all_wrong_split_counts` uses all saved wrong-prediction exports, not only
  important failures
- train-count logic for all wrong predictions:
  `entire_data wrong predictions - test_data wrong predictions`
- workbook sheets:
  `important_failures_split_counts`,
  `all_wrong_split_counts`,
  `important_by_scheme_model`,
  `important_by_split_scheme`,
  `all_wrong_by_scheme_model`,
  `all_wrong_by_split_scheme`

Train/test split count summary workbook:

- reporting-only entry point:
  `src/failure_train_test_split_count_summary.py`
- output workbook:
  `outputs/reports/failure_train_test_split_count_summary.xlsx`
- combines:
  - important-failure counts from the filtered one-sheet workbook
  - all wrong-prediction counts from saved wrong-case exports
  - train/test counts for the corrected XGB mixed-combination workbook
- train/test origin is derived by matching entire-data rows against saved
  `predict_inference_test_data` wrong-prediction exports
- if the root-level filtered one-sheet workbook is absent, the script falls
  back to the saved copy under `existing_code_run_20260511/`
- mixed-combination analysis uses
  `mixed_failure_case_combinations_xgb_only_corrected.xlsx`
  and falls back to the workbook `summary` sheet when no
  `failure case summary` sheet exists
- workbook sheets:
  `important_failure_counts`,
  `all_wrong_predictions_counts`,
  `mixed_combo_train_test_counts`,
  `analysis_notes`,
  `pivot_summary`,
  `important_by_scheme_model`,
  `important_by_split_scheme`,
  `all_wrong_by_scheme_model`,
  `all_wrong_by_split_scheme`

XGB strat_y_K train/test failure analysis:

- reporting-only entry point:
  `src/xgb_stratyk_train_test_failure_analysis.py`
- outputs:
  `outputs/reports/xgb_strat_yK_train_test_failure_counts.xlsx`
  and
  `outputs/reports/mixed_failure_case_combinations_xgb_only_corrected_train_highlighted.xlsx`
- purpose:
  isolate `XGB + strat_y_K` important failures by scheme and partition, then
  apply the same train/test origin detection to each corrected mixed XGB
  combination sheet
- copied mixed workbook behavior:
  highlight training-side rows only, using a light-yellow full-row fill
  and add `train_test_failure_summary`

## Source Fields For Failure-Case Analysis

- `source_model`
- `failure_severity`
- `fastest_algo`
- `fastest_runtime`
- `predicted_algo`
- `predicted_probability`
- `top3_predicted_detailed`

All runtime-derived fields above are based on the active BnB columns.

## Runner Script

- End-to-end bash entry point: `restructured_pipeline/run_full_pipeline.sh`
- Supports:
  `scheme1`, `scheme2`, `scheme3`
- Supports split modes:
  `strat_y`, `strat_y_K`
- Supports full experiment-matrix mode:
  `--full-experiment-matrix --matrix-run-name <name>`
- Runs:
  training, evaluation, held-out prediction export, full-dataset prediction
  export, wrong-prediction staging, and failure-case Excel report generation
- One wrapper run writes both prediction export files automatically:
  - `all_predictions_<RUN_NAME>_predict_inference_test_data.csv`
  - `all_predictions_<RUN_NAME>_predict_inference_entire_data.csv`
- In matrix mode, the wrapper calls `src/experiment_matrix.py` and writes:
  - `model_performance_for_all_exp_<MATRIX_RUN_NAME>.xlsx`
  - `model_performance_for_all_exp_tolerance_<MATRIX_RUN_NAME>.xlsx`
  - one workbook pair per `scheme/model_family`
  - Scheme 2 / Scheme 3 training runs also keep their stage-summary and
    per-class report workbooks
  - final sheet set:
    `model_performance_for_all_exp`,
    `strat_y_test_data`,
    `strat_y__K_test_data`,
    `strat_y_entire_data`,
    `strat_y__K_entire_data`

## Scheme 3 Implementation

- Keep all new work inside `restructured_pipeline/`.
- Use config file: `configs/models_scheme3.yaml`.
- Use model module: `src/models/hierarchical_scheme3.py`.
- Implemented hierarchy:
  - Stage 1: `Sparse` vs `Dense`
  - Stage 2: `Sparse -> AltRB` vs `No-AltRB`
  - Stage 3: 4-class router
    - `Class1`: `UMKP-S-bnb` / `UMKP+-S-bnb` / `kPEX-S-bnb`
    - `Class2`: `UMKP-R-bnb` / `UMKP+-R-bnb` / `kPEX-R-bnb`
    - `Class3`: `UMKP-P-bnb` / `UMKP+-P-bnb` / `kPEX-P-bnb`
    - `Class4`: `kPEX-B-bnb`
- Final target remains the fastest BnB algorithm label.
- Special router behavior:
  - if Stage 3 predicts `Class4`, final output is `kPEX-B-bnb` directly
- Implemented model families:
  - Scheme 3 RF for all stages
  - Scheme 3 XGB for all stages

## Execution Rules

- Extract, do not redesign.
- Preserve behavior unless a small compatibility extension is required.
- Keep failure-case computation and Excel formatting as separate layers.
- Do not reintroduce notebook-only dependencies into the new pipeline.

## Remaining Work

- No open integration blockers remain for the active pipeline path.
- The active matrix path now covers, for every scheme:
  - `strat_y` and `strat_y_K`
  - `random_forest` and `xgboost`
  - `inference_test_set` and `inference_entire_data`
- Redundant artifact writes have been reduced in the active pipeline:
  - wrong-prediction exports keep `.xlsx` only
  - classification-report CSV files are not written
  - per-run metrics JSON and evaluation/prediction log JSON files are not written
  - Scheme 2 / Scheme 3 still keep the stage summary and per-class Excel reports
  - tolerance workbooks keep only the summary sheet plus the four experiment sheets
- Remaining work is operational rather than structural:
  - run the full production matrix with the non-validation model configs
  - review the generated workbooks and failure-case outputs
