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

Phase 1 cleanup status:

- completed on `2026-06-23`
- removed unused active-pipeline helper:
  `src/write_utils.py:write_metrics_to_excel`
- removed the deprecated notebook-era compatibility helper after repo-wide
  caller validation
- removed dead, unreferenced CLI flags from
  `src/failure_analysis_summary.py`
- validation smoke tests passed for:
  Scheme 1 RF training, held-out evaluation, entire-data prediction export,
  and `src/failure_analysis_summary.py --mode enrich`

Phase 2 shared-helper extraction status:

- completed on `2026-06-23`
- added shared helper modules:
  `src/saved_run_utils.py`
  `src/reporting_utils.py`
  `src/topk_utils.py`
- `src/evaluate.py` and `src/predict.py` now share the saved-run inference
  helpers through `src/saved_run_utils.py`
- repeated reporting helpers are shared through
  `src/reporting_utils.py`
- repeated top-3 probability ranking is shared through
  `src/topk_utils.py`
- compatibility aliases were retained in the original entry-point modules
- validation smoke tests passed for:
  Scheme 1 RF training, held-out evaluation, entire-data prediction export,
  compare wrong predictions discovery mode, and
  `src/failure_analysis_summary.py --mode enrich`
- pre-existing discovery-mode blank `scheme` / `model` rows still come from
  legacy wrong-case inputs; no new blank-label rows were introduced by Phase 2

Phase 3 duplicate full-dataset export status:

- completed on `2026-06-23`
- `src/predict.py --inference-mode inference_entire_data` now reuses the
  canonical training-time full-dataset outputs when they already match:
  `all_predictions_<RUN_NAME>.csv`
  `wrong_predictions_<RUN_NAME>.xlsx`
- the duplicate-named entire-data export paths are kept as aliases so the
  current compare workflow can still read:
  `all_predictions_<RUN_NAME>_predict_inference_entire_data.csv`
  `wrong_predictions_<RUN_NAME>_predict_inference_entire_data.xlsx`
- added `src/predict.py --force-export` to opt back into writing a second
  physical entire-data export
- held-out test-set prediction export behavior is unchanged
- validation smoke tests passed for:
  fresh Scheme 1 RF training, entire-data prediction reuse via aliases,
  held-out test-set prediction export, and a downstream compare smoke test
  reading the alias-named entire-data wrong-predictions workbook

## Documentation Pointers

- `README.md`
  Main operator doc for training, evaluation, prediction, output locations,
  naming rules, and the runnable command matrix.
- `docs/comparison_wrong_predictions.md`
  Wrong-predictions comparison inputs, thresholds, highlighting, and the
  consolidated full-dataset workflow.
- `docs/high_risk_cleanup_checklist.md`
  Future-cleanup guardrails for compatibility aliases, staging fallback,
  hierarchical helper duplication, Scheme 1 hyperparameter dual sources, and
  `failure_case_filter.py`.
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

## Cleanup Guardrails

High-risk items that must NOT be removed without a dedicated validation task:

1. Compatibility alias families
   Why it exists:
   older compare, tolerance, and failure-analysis paths still accept
   compatibility filenames such as the `all_schema` typo alias and older
   fallback workbook names.
   What depends on it:
   `src/compare_wrong_predictions.py`,
   `src/failure_case_filter.py`,
   `src/failure_analysis_summary.py`, and older saved output paths.
   Required validation before removal:
   run the alias-removal smoke tests from
   `docs/high_risk_cleanup_checklist.md` and remove only one alias family at a
   time.

2. `run_full_pipeline.sh --compare-input-mode staging`
   Why it exists:
   staged compare-input copying is still the documented fallback when direct
   wrapper compare is not the path being exercised.
   What depends on it:
   `run_full_pipeline.sh --compare-input-mode staging`,
   `src/compare_wrong_predictions.py --model-root ...`, and fallback wrapper
   workflows.
   Required validation before removal:
   prove direct wrapper mode is the sole documented path and rerun the direct
   and staging wrapper smoke tests from
   `docs/high_risk_cleanup_checklist.md`.

3. Duplicated hierarchical helpers in Scheme 2 / Scheme 3 modules
   Why it exists:
   the duplicated helpers sit inside active hierarchical training paths and are
   intentionally deferred until the planned larger module split.
   What depends on it:
   Scheme 2 RF/XGB training and Scheme 3 RF/XGB training.
   Required validation before removal:
   perform the dedicated hierarchical smoke-test set from
   `docs/high_risk_cleanup_checklist.md` and compare artifacts, row counts, and
   label behavior before and after any helper merge.

4. Scheme 1 hyperparameter dual source
   Why it exists:
   Scheme 1 still keeps hyperparameter presets in both `models/flat.py` and
   `configs/models_scheme1.yaml`, and a premature consolidation could change
   actual model configuration.
   What depends on it:
   Scheme 1 RF/XGB variant resolution, wrapper variant mapping, and current
   training behavior.
   Required validation before removal:
   choose one canonical source first, then rerun the dedicated Scheme 1
   regression commands from `docs/high_risk_cleanup_checklist.md`.

5. `failure_case_filter.py`
   Why it exists:
   it still produces the single-sheet filtered workbook shape that
   `failure_analysis_summary.py --mode enrich` consumes.
   What depends on it:
   the canonical filtered enrich workflow and its current workbook contract.
   Required validation before removal:
   teach enrich mode to consume the replacement workbook shape first, then
   rerun the filtered enrich smoke tests from
   `docs/high_risk_cleanup_checklist.md`.

Use `docs/high_risk_cleanup_checklist.md` as the required validation checklist
before any future removal attempt touching the items above.

Saved-run naming rule:

- training keeps `all_predictions_<RUN_NAME>.csv` as the base full-dataset
  export written by `train.py --save-files`
- held-out evaluation and prediction outputs use `inference_test_data` in
  generated filenames
- full-dataset evaluation and prediction outputs use `inference_entire_data`
  in generated filenames
- after Phase 3, full-dataset `predict.py` export names are aliases to the
  canonical training-time files by default unless `--force-export` is used
- `run_full_pipeline.sh` still enables the full-dataset `predict.py` step by
  default for backward compatibility, but `--skip-full-predict` can skip that
  wrapper step when only the direct compare path is needed

Phase 4 compare/tolerance/important-failure smoothing status:

- completed on `2026-06-23`
- `src/compare_wrong_predictions.py` now carries one canonical base compare
  dataframe forward into the tolerance and important-failure stages during the
  active pipeline flows
- `--run-test-data-pipeline` and `--run-entire-data-pipeline` no longer
  rebuild the strict compare dataframe twice
- compare mode with tolerance + important-failure generation no longer rereads
  the tolerance workbook between stages
- existing CLI entry points and output workbook names are preserved
- standalone workbook-driven compatibility paths still read from disk:
  `--mode report` and
  `--generate-important-failure-report --input-tolerance-report ...`
- validation smoke tests passed for:
  direct-file compare, `--run-test-data-pipeline`,
  `--run-entire-data-pipeline`, monotonic tolerance counts,
  important-failure counts never exceeding parent tolerance counts, and
  zero blank `scheme` / `model` values in the validated smoke outputs

Phase 5 unfiltered/mixed-workflow canonicalization status:

- completed on `2026-06-23`
- canonical unfiltered workflow:
  `src/failure_analysis_unfiltered_summary.py`
- removed the deprecated compatibility-only unfiltered summary script after
  repo-wide caller validation
- `src/mixed_failure_case_combinations.py` now requires explicit
  `--master-workbook` and `--filtered-workbook` inputs
- dated `existing_code_run_20260511/...` fallback defaults were removed from
  the mixed-failure script
- validation smoke tests passed for:
  canonical unfiltered workbook generation and mixed failure-case
  combinations with explicit input paths

Phase 6 compatibility-path preparation status:

- completed on `2026-06-23`
- canonical filenames and deprecated aliases documented in:
  `docs/compatibility_paths_and_removal_checklist.md`
- warnings now fire when deprecated compatibility paths are used in:
  `src/compare_wrong_predictions.py`,
  `src/failure_case_filter.py`,
  `src/failure_analysis_summary.py`
- canonical-path workflows should remain warning-free
- the following compatibility branches remain intentionally untouched:
  - `all_schema` typo compatibility
  - older compare/failure-analysis fallback candidates
  - `run_full_pipeline.sh` compare-input staging shim

Phase 7 wrapper compare-staging refactor status:

- completed on `2026-06-23`
- `run_full_pipeline.sh` now defaults to direct-file compare inputs
- staged compare-input copying remains available only as a fallback through:
  `--compare-input-mode staging`
- wrapper runs can now also request explicit per-run paths for:
  - tolerance reports
  - important-failure reports
- `src/compare_wrong_predictions.py` now supports
  `--source-model-label`
  so direct-file wrapper comparisons keep the expected exported
  `source_model` label while deriving `with_holdout` and `strat_on` from
  synthetic direct-file metadata names
- validation smoke test passed for one Scheme 1 RF direct wrapper run with:
  compare workbook generation, tolerance workbook generation,
  important-failure workbook generation, correct split metadata,
  zero blank `scheme` / `model` rows, and no staged compare-input files

Phase 8 large-module split planning status:

- completed on `2026-06-23`
- added planning document:
  `docs/large_module_split_plan.md`
- no runtime module split was performed in this phase
- recommended next split target:
  `src/compare_wrong_predictions.py`
- planned split order:
  `src/compare_wrong_predictions.py`,
  `src/failure_analysis_summary.py`,
  `src/train.py`,
  `src/models/hierarchical.py`,
  `src/models/hierarchical_scheme3.py`

Phase 9 final validation status:

- completed on `2026-06-23`
- passed validation for:
  - Scheme 1 RF training
  - Scheme 2 XGB training
  - Scheme 3 RF training
  - held-out and entire-data evaluation
  - held-out and entire-data prediction export
  - direct-file compare reporting
  - tolerance reporting
  - integrated important-failure reporting
  - canonical filtered important-failure reporting
  - filtered failure-analysis summary
  - canonical unfiltered summary
- entire-data prediction export still behaves as intended after Phase 3:
  canonical training outputs are reused and predict-named entire-data paths are
  compatibility symlink aliases
- later closeout validation confirmed:
  `src/mixed_failure_case_combinations.py` now runs without the earlier
  merge-key dtype mismatch

Low-risk unused-function cleanup completed on `2026-06-23`.

- removed:
  `src/features.py:export_sorted`
  `src/models/flat.py:get_feature_importances`

Cleanup smoke-test runner:

- use:
  `bash restructured_pipeline/run_cleanup_smoke_tests.sh`
- run it after cleanup edits, before deleting medium-risk items, and before any
  high-risk compatibility-removal attempt
- it validates:
  - Python syntax
  - bash syntax for `run_full_pipeline.sh`
  - imports for active modules
  - one Scheme 1 RF training smoke run
  - held-out evaluation
  - held-out and entire-data prediction export
  - direct compare reporting
  - failure-case filter
  - `failure_analysis_summary.py --mode enrich`
  - canonical unfiltered summary generation
  - mixed failure-case combinations
- for the mixed-failure step, it reuses an existing validated multi-scheme
  filtered/unfiltered workbook pair when available, because a single Scheme 1
  smoke run is not enough to exercise the cross-scheme pairing rules
- smoke-report outputs go to a temporary `/private/tmp/...` directory
- failures should be treated as cleanup blockers until explained or fixed

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
  should now prefer
  `restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_entire_data.xlsx`
- deprecated compatibility alias still accepted:
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
- requires explicit input paths:
  `--master-workbook <UNFILTERED_SIX_SHEET_WORKBOOK>`
  `--filtered-workbook <FILTERED_SIX_SHEET_WORKBOOK>`
- recommended unfiltered source:
  `failure_analysis_summary_unfiltered_all_scheme_entire_data.xlsx`
- recommended filtered source:
  `failure_analysis_summary_all_scheme_entire_data.xlsx`
- allowed scheme pairings only:
  - `scheme 1 + scheme 2`
  - `scheme 1 + scheme 3`
- forbidden pairing:
  - `scheme 2 + scheme 3`
- exported sheets:
  - `s1_top2__s2_top1`
  - `s1_top1__s2_top2`
  - `s1_top2__s3_top1`
  - `s1_top1__s3_top2`
- `source_split_label` is mapped from the master workbook
- rows are kept only when the selected `top1` or `top2` minimum ratio is
  strictly greater than `2.0`
- selected-candidate fields are reconstructed from:
  `top1_candidate`, `top2_candidate`, `fastest_algo_tuple`,
  and `all_predictions_sorted_by_prob`

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
- `--skip-full-predict` skips only the wrapper's full-dataset `predict.py`
  export step; training outputs, held-out prediction export, and compare
  reporting still run
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
  - Stage 3: `Oriented` vs `NonOriented`
  - Stage 4: 4-class router
    - `ClassB`: every `*-B-bnb` label
    - `ClassS`: every `*-S-bnb` label
    - `ClassR`: every `*-R-bnb` label
    - `ClassP`: every `*-P-bnb` label
- Final target remains the fastest BnB algorithm label.
- Final output is resolved by combining:
  - Stage 1 family: `Dense -> kPEX`, `Sparse -> UMKP/UMKP+`
  - Stage 2 sparse branch: `AltRB -> UMKP+`, `No-AltRB -> UMKP`
  - Stage 3 orientation: add `-oriented` only for `Oriented`
  - Stage 4 router suffix: `B`, `S`, `R`, or `P`
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
