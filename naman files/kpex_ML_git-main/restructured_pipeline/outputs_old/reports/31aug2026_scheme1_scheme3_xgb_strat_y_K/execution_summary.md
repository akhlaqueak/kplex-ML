# 31aug2026 Scheme 1 / Scheme 3 XGB strat_y_K Execution Summary

Run date: `31aug2026`
Report folder: `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K`

## Scope

- Included: Scheme 1 XGB strat_y_K
- Included: Scheme 3 XGB strat_y_K
- Excluded: RF, Scheme 2, strat_y

## Model And Prediction Metrics

| Scope | Rows | Accuracy | Tol0 | Tol10 | Tol20 | Tol50 | Wrong | Avg Ratio | Median Ratio | Max Ratio | Avg Gap | Median Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scheme1_test | 1149 | 0.3699 | 0.3699 | 0.5039 | 0.5988 | 0.7145 | 724 | 809.633287 | 1.237165 | 580645.16129 | 59.387911 | 0.009 |
| scheme1_entire | 4279 | 0.1837 | 0.1837 | 0.3557 | 0.4730 | 0.6163 | 3493 | 307.183161 | 1.283889 | 580645.16129 | 81.677964 | 0.0967 |
| scheme3_test | 854 | 0.4426 | 0.4426 | 0.6944 | 0.7541 | 0.8162 | 476 | 684.757916 | 1.01648 | 227848.101266 | 46.648562 | 0.0014 |
| scheme3_entire | 4279 | 0.8042 | 0.8042 | 0.9084 | 0.9290 | 0.9469 | 838 | 996.735172 | 1.0 | 2571428.571429 | 12.614058 | 0.0 |

## Important Failure Counts

| Scope | Scheme 1 tol0 | Scheme 2 tol0 | Scheme 3 tol0 | Notes |
|---|---:|---:|---:|---|
| test_data integrated | 162 | 0 | 84 | From compare tolerance important-failure workbook |
| entire_data integrated | 888 | 0 | 103 | From compare tolerance important-failure workbook |
| entire_data filtered input | 888 | 0 | 103 | Single-sheet input for enrich mode; total 991 |

## Generated Workbooks

- `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/compare_wrong_predictions_test_data_31aug2026.xlsx`: sheets=['on 20% test set'], rows={'on 20% test set': 485}
- `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/compare_wrong_predictions_tolerance_report_test_data_31aug2026.xlsx`: sheets=['scheme_1_tol_0', 'scheme_1_tol_10', 'scheme_1_tol_20', 'scheme_1_tol_50', 'scheme_2_tol_0', 'scheme_2_tol_10', 'scheme_2_tol_20', 'scheme_2_tol_50', 'scheme_3_tol_0', 'scheme_3_tol_10', 'scheme_3_tol_20', 'scheme_3_tol_50'], rows={'scheme_1_tol_0': 328, 'scheme_1_tol_10': 328, 'scheme_1_tol_20': 328, 'scheme_1_tol_50': 328, 'scheme_2_tol_0': 0, 'scheme_2_tol_10': 0, 'scheme_2_tol_20': 0, 'scheme_2_tol_50': 0, 'scheme_3_tol_0': 157, 'scheme_3_tol_10': 157, 'scheme_3_tol_20': 157, 'scheme_3_tol_50': 157}
- `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/compare_wrong_predictions_tolerance_important_failures_test_data_31aug2026.xlsx`: sheets=['scheme_1_tol_0', 'scheme_1_tol_10', 'scheme_1_tol_20', 'scheme_1_tol_50', 'scheme_2_tol_0', 'scheme_2_tol_10', 'scheme_2_tol_20', 'scheme_2_tol_50', 'scheme_3_tol_0', 'scheme_3_tol_10', 'scheme_3_tol_20', 'scheme_3_tol_50', 'filter_summary'], rows={'scheme_1_tol_0': 162, 'scheme_1_tol_10': 162, 'scheme_1_tol_20': 162, 'scheme_1_tol_50': 162, 'scheme_2_tol_0': 0, 'scheme_2_tol_10': 0, 'scheme_2_tol_20': 0, 'scheme_2_tol_50': 0, 'scheme_3_tol_0': 84, 'scheme_3_tol_10': 84, 'scheme_3_tol_20': 84, 'scheme_3_tol_50': 84, 'filter_summary': 13}
- `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/compare_wrong_predictions_entire_data_31aug2026.xlsx`: sheets=['on entire dataset'], rows={'on entire dataset': 1869}
- `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/compare_wrong_predictions_tolerance_report_entire_data_31aug2026.xlsx`: sheets=['scheme_1_tol_0', 'scheme_1_tol_10', 'scheme_1_tol_20', 'scheme_1_tol_50', 'scheme_2_tol_0', 'scheme_2_tol_10', 'scheme_2_tol_20', 'scheme_2_tol_50', 'scheme_3_tol_0', 'scheme_3_tol_10', 'scheme_3_tol_20', 'scheme_3_tol_50'], rows={'scheme_1_tol_0': 1642, 'scheme_1_tol_10': 1642, 'scheme_1_tol_20': 1642, 'scheme_1_tol_50': 1642, 'scheme_2_tol_0': 0, 'scheme_2_tol_10': 0, 'scheme_2_tol_20': 0, 'scheme_2_tol_50': 0, 'scheme_3_tol_0': 227, 'scheme_3_tol_10': 227, 'scheme_3_tol_20': 227, 'scheme_3_tol_50': 227}
- `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/compare_wrong_predictions_tolerance_important_failures_entire_data_31aug2026.xlsx`: sheets=['scheme_1_tol_0', 'scheme_1_tol_10', 'scheme_1_tol_20', 'scheme_1_tol_50', 'scheme_2_tol_0', 'scheme_2_tol_10', 'scheme_2_tol_20', 'scheme_2_tol_50', 'scheme_3_tol_0', 'scheme_3_tol_10', 'scheme_3_tol_20', 'scheme_3_tol_50', 'filter_summary'], rows={'scheme_1_tol_0': 888, 'scheme_1_tol_10': 888, 'scheme_1_tol_20': 888, 'scheme_1_tol_50': 888, 'scheme_2_tol_0': 0, 'scheme_2_tol_10': 0, 'scheme_2_tol_20': 0, 'scheme_2_tol_50': 0, 'scheme_3_tol_0': 103, 'scheme_3_tol_10': 103, 'scheme_3_tol_20': 103, 'scheme_3_tol_50': 103, 'filter_summary': 13}
- `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/failure_case_analysis_important_failures_scheme1_scheme3_xgb_strat_y_K_31aug2026.xlsx`: sheets=['on entire dataset', 'filter_summary'], rows={'on entire dataset': 991, 'filter_summary': 2}
- `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/failure_analysis_summary_filtered_scheme1_scheme3_xgb_strat_y_K_31aug2026.xlsx`: sheets=['scheme_1_RF', 'scheme_1_XGB', 'scheme_2_RF', 'scheme_2_XGB', 'scheme_3_RF', 'scheme_3_XGB'], rows={'scheme_1_RF': 0, 'scheme_1_XGB': 888, 'scheme_2_RF': 0, 'scheme_2_XGB': 0, 'scheme_3_RF': 0, 'scheme_3_XGB': 103}
- `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/failure_analysis_summary_unfiltered_scheme1_scheme3_xgb_strat_y_K_31aug2026.xlsx`: sheets=['summary', 'scheme_1_RF', 'scheme_1_XGB', 'scheme_2_RF', 'scheme_2_XGB', 'scheme_3_RF', 'scheme_3_XGB'], rows={'summary': 2, 'scheme_1_RF': 0, 'scheme_1_XGB': 3493, 'scheme_2_RF': 0, 'scheme_2_XGB': 0, 'scheme_3_RF': 0, 'scheme_3_XGB': 838}

## Top Confusions

### scheme1_test
- `UMKP-oriented-B-bnb` -> `kPEX-oriented-B-bnb`: 140
- `UMKP-oriented-P-bnb` -> `kPEX-oriented-B-bnb`: 98
- `UMKP-oriented-S-bnb` -> `kPEX-oriented-B-bnb`: 49
- `UMKP+-oriented-S-bnb` -> `kPEX-oriented-B-bnb`: 47
- `UMKP+-oriented-B-bnb` -> `kPEX-oriented-B-bnb`: 42

### scheme1_entire
- `UMKP-oriented-B-bnb` -> `kPEX-oriented-B-bnb`: 680
- `UMKP-oriented-P-bnb` -> `kPEX-oriented-B-bnb`: 506
- `UMKP+-oriented-B-bnb` -> `kPEX-oriented-B-bnb`: 239
- `UMKP-oriented-S-bnb` -> `kPEX-oriented-B-bnb`: 210
- `UMKP+-oriented-S-bnb` -> `kPEX-oriented-B-bnb`: 210

### scheme3_test
- `kPEX-oriented-B-bnb` -> `UMKP-oriented-P-bnb`: 18
- `UMKP-oriented-P-bnb` -> `kPEX-oriented-B-bnb`: 15
- `UMKP-oriented-B-bnb` -> `UMKP-oriented-P-bnb`: 15
- `UMKP-oriented-P-bnb` -> `UMKP-oriented-B-bnb`: 13
- `kPEX-oriented-B-bnb` -> `kPEX-oriented-P-bnb`: 12

### scheme3_entire
- `UMKP+-oriented-B-bnb` -> `UMKP-oriented-B-bnb`: 32
- `UMKP-oriented-B-bnb` -> `UMKP-oriented-P-bnb`: 24
- `kPEX-oriented-B-bnb` -> `UMKP-oriented-P-bnb`: 24
- `UMKP-oriented-P-bnb` -> `kPEX-oriented-P-bnb`: 23
- `kPEX-oriented-B-bnb` -> `UMKP-oriented-B-bnb`: 22

## Validation

- PASS: test_data: compare has no RF rows - restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/compare_wrong_predictions_test_data_31aug2026.xlsx
- PASS: test_data: compare has no Scheme 2 rows - restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/compare_wrong_predictions_test_data_31aug2026.xlsx
- PASS: test_data: compare only strat_y_K - ['y and K']
- PASS: test_data: compare no blank scheme/model/strat/inference - checked key columns
- PASS: test_data: scheme_1_tol_0 no RF unless expected empty - 328
- PASS: test_data: scheme_1_tol_0 only y and K - ['y and K']
- PASS: test_data: scheme_1_tol_10 no RF unless expected empty - 328
- PASS: test_data: scheme_1_tol_10 only y and K - ['y and K']
- PASS: test_data: scheme_1_tol_20 no RF unless expected empty - 328
- PASS: test_data: scheme_1_tol_20 only y and K - ['y and K']
- PASS: test_data: scheme_1_tol_50 no RF unless expected empty - 328
- PASS: test_data: scheme_1_tol_50 only y and K - ['y and K']
- PASS: test_data: scheme_1 tolerance monotonic - [328, 328, 328, 328]
- PASS: test_data: scheme_1 important <= tolerance - important=[162, 162, 162, 162], tolerance=[328, 328, 328, 328]
- PASS: test_data: scheme_2 tolerance monotonic - [0, 0, 0, 0]
- PASS: test_data: scheme_2 important <= tolerance - important=[0, 0, 0, 0], tolerance=[0, 0, 0, 0]
- PASS: test_data: scheme_3_tol_0 no RF unless expected empty - 157
- PASS: test_data: scheme_3_tol_0 only y and K - ['y and K']
- PASS: test_data: scheme_3_tol_10 no RF unless expected empty - 157
- PASS: test_data: scheme_3_tol_10 only y and K - ['y and K']
- PASS: test_data: scheme_3_tol_20 no RF unless expected empty - 157
- PASS: test_data: scheme_3_tol_20 only y and K - ['y and K']
- PASS: test_data: scheme_3_tol_50 no RF unless expected empty - 157
- PASS: test_data: scheme_3_tol_50 only y and K - ['y and K']
- PASS: test_data: scheme_3 tolerance monotonic - [157, 157, 157, 157]
- PASS: test_data: scheme_3 important <= tolerance - important=[84, 84, 84, 84], tolerance=[157, 157, 157, 157]
- PASS: entire_data: compare has no RF rows - restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/compare_wrong_predictions_entire_data_31aug2026.xlsx
- PASS: entire_data: compare has no Scheme 2 rows - restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/compare_wrong_predictions_entire_data_31aug2026.xlsx
- PASS: entire_data: compare only strat_y_K - ['y and K']
- PASS: entire_data: compare no blank scheme/model/strat/inference - checked key columns
- PASS: entire_data: scheme_1_tol_0 no RF unless expected empty - 1642
- PASS: entire_data: scheme_1_tol_0 only y and K - ['y and K']
- PASS: entire_data: scheme_1_tol_10 no RF unless expected empty - 1642
- PASS: entire_data: scheme_1_tol_10 only y and K - ['y and K']
- PASS: entire_data: scheme_1_tol_20 no RF unless expected empty - 1642
- PASS: entire_data: scheme_1_tol_20 only y and K - ['y and K']
- PASS: entire_data: scheme_1_tol_50 no RF unless expected empty - 1642
- PASS: entire_data: scheme_1_tol_50 only y and K - ['y and K']
- PASS: entire_data: scheme_1 tolerance monotonic - [1642, 1642, 1642, 1642]
- PASS: entire_data: scheme_1 important <= tolerance - important=[888, 888, 888, 888], tolerance=[1642, 1642, 1642, 1642]
- PASS: entire_data: scheme_2 tolerance monotonic - [0, 0, 0, 0]
- PASS: entire_data: scheme_2 important <= tolerance - important=[0, 0, 0, 0], tolerance=[0, 0, 0, 0]
- PASS: entire_data: scheme_3_tol_0 no RF unless expected empty - 227
- PASS: entire_data: scheme_3_tol_0 only y and K - ['y and K']
- PASS: entire_data: scheme_3_tol_10 no RF unless expected empty - 227
- PASS: entire_data: scheme_3_tol_10 only y and K - ['y and K']
- PASS: entire_data: scheme_3_tol_20 no RF unless expected empty - 227
- PASS: entire_data: scheme_3_tol_20 only y and K - ['y and K']
- PASS: entire_data: scheme_3_tol_50 no RF unless expected empty - 227
- PASS: entire_data: scheme_3_tol_50 only y and K - ['y and K']
- PASS: entire_data: scheme_3 tolerance monotonic - [227, 227, 227, 227]
- PASS: entire_data: scheme_3 important <= tolerance - important=[103, 103, 103, 103], tolerance=[227, 227, 227, 227]
- PASS: filtered_failure_input: no RF rows - restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/failure_case_analysis_important_failures_scheme1_scheme3_xgb_strat_y_K_31aug2026.xlsx
- PASS: filtered_failure_input: no Scheme 2 rows - restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/failure_case_analysis_important_failures_scheme1_scheme3_xgb_strat_y_K_31aug2026.xlsx
- PASS: enriched_failure_analysis: no RF rows - restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/failure_analysis_summary_filtered_scheme1_scheme3_xgb_strat_y_K_31aug2026.xlsx
- PASS: unfiltered_failure_analysis: no RF rows - restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/failure_analysis_summary_unfiltered_scheme1_scheme3_xgb_strat_y_K_31aug2026.xlsx
- PASS: unfiltered_failure_analysis: no Scheme 2 rows - restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/failure_analysis_summary_unfiltered_scheme1_scheme3_xgb_strat_y_K_31aug2026.xlsx
- PASS: unfiltered_failure_analysis: only strat_y_K - ['strat_y_K']
- PASS: scheme1_test: predicted labels valid - {"invalid": [], "blank": 0}
- PASS: scheme1_entire: predicted labels valid - {"invalid": [], "blank": 0}
- PASS: scheme3_test: predicted labels valid - {"invalid": [], "blank": 0}
- PASS: scheme3_entire: predicted labels valid - {"invalid": [], "blank": 0}
- WARN: scheme1_entire: duplicate full-dataset export was not skipped - predict.py reported wrote_entire_data_export; pred_byte_equal=True; wrong_size_equal=True
- WARN: scheme3_entire: duplicate full-dataset export was not skipped - predict.py reported wrote_entire_data_export; pred_byte_equal=True; wrong_size_equal=False

## Warnings And Skips

- Root README.md is absent; used restructured_pipeline/README.md as the active workflow README.
- Initial test-data compare run with raw xgb filenames produced blank scheme/model metadata; corrected by rerunning with direct-file NAME labels containing xgboost. Final report files are from the corrected run.
- predict.py entire-data runs reported wrote_entire_data_export for both schemes; duplicate physical full-dataset prediction/wrong-case files were generated instead of canonical alias reuse.
- Mixed failure-case combinations skipped: current PAIRING_RULES include Scheme 1 + Scheme 2 pairings, so the script cannot cleanly generate a Scheme 1 + Scheme 3-only workbook without wrong-scope/empty Scheme 2 validation issues.
- Mixed failure-case combinations was skipped because the current script cannot cleanly scope to Scheme 1 + Scheme 3 only without Scheme 1 + Scheme 2 pairing rules.

## Provenance

| Output File | Report Type | Generated By Script | Main Function / CLI Path | Input Files | Rows / Sheets | Notes |
|---|---|---|---|---|---|---|
| `model_*.pkl, all_predictions_<run>.csv, wrong_predictions_<run>.xlsx, split CSVs, metrics/confusion matrices` | training artifacts | `restructured_pipeline/src/train.py` | main -> run_training -> train_scheme1/train_scheme3 -> save_model_artifact/save_dataframe_outputs/save_metrics_outputs/save_split_indices | project.yaml, models_scheme*.yaml, input_data/ml-data.csv | Scheme1 train/test 9350/2338 expanded rows; Scheme3 train/test 3425/854 rows | Scheme3 also writes stage summary/per-class reports. |
| `all_predictions_*_predict_inference_*.csv and wrong_predictions_*_predict_inference_*.xlsx` | prediction export | `restructured_pipeline/src/predict.py` | main -> run_prediction -> load_results_for_prediction/select_prediction_subset/save_dataframe_outputs | saved model artifacts and split indices | test: 1149/854 rows; entire: 4279/4279 rows | Entire-data export regenerated physical files in this run. |
| `compare_wrong_predictions_*_31aug2026.xlsx` | strict compare workbook | `restructured_pipeline/src/compare_wrong_predictions.py` | main -> run_compare_mode -> build_combined_dataframe_from_args -> build_compare_wrong_predictions_from_files -> write_by_inference_on | direct wrong_predictions files | test compare 485 rows; entire compare 1869 rows | Direct NAME labels used xgboost token for metadata parsing. |
| `compare_wrong_predictions_tolerance_report_*_31aug2026.xlsx` | tolerance workbook | `restructured_pipeline/src/compare_wrong_predictions.py` | run_compare_mode -> build_tolerance_sheet_frames -> write_tolerance_sheet_frames | in-memory strict compare dataframe | 12 tolerance sheets per scope | Scheme2 sheets are empty compatibility sheets. |
| `compare_wrong_predictions_tolerance_important_failures_*_31aug2026.xlsx` | integrated important-failure workbook | `restructured_pipeline/src/compare_wrong_predictions.py` | run_compare_mode -> write_important_failures_report_from_tolerance_frames -> build_important_failure_sheet_frames | in-memory tolerance sheet frames | 12 tolerance important-failure sheets plus summary | Counts do not exceed tolerance parent rows. |
| `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/failure_case_analysis_important_failures_scheme1_scheme3_xgb_strat_y_K_31aug2026.xlsx` | single-sheet filtered failure-analysis input | `restructured_pipeline/src/failure_case_filter.py` | main -> run_failure_case_filter -> annotate_failure_cases -> write_by_inference_on | restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/compare_wrong_predictions_entire_data_31aug2026.xlsx | 991 rows, sheet on entire dataset | Required by enrich mode shape. |
| `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/failure_analysis_summary_filtered_scheme1_scheme3_xgb_strat_y_K_31aug2026.xlsx` | filtered six-sheet failure-analysis summary | `restructured_pipeline/src/failure_analysis_summary.py` | main -> run_enrichment_mode -> _build_failure_analysis_sheet -> _write_failure_analysis_workbook | restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/failure_case_analysis_important_failures_scheme1_scheme3_xgb_strat_y_K_31aug2026.xlsx | 6 sheets; Scheme1 XGB 888 rows, Scheme3 XGB 103 rows | RF/Scheme2 sheets are empty. |
| `restructured_pipeline/outputs/reports/31aug2026_scheme1_scheme3_xgb_strat_y_K/failure_analysis_summary_unfiltered_scheme1_scheme3_xgb_strat_y_K_31aug2026.xlsx` | unfiltered six-sheet failure-analysis summary | `restructured_pipeline/src/failure_analysis_unfiltered_summary.py` | main -> load_unfiltered_records -> build_summary_sheet/build_detail_sheets -> write_workbook | scoped scheme[13]/xgboost entire-data wrong-case glob | 6 detail sheets plus summary; Scheme1 XGB 3493 rows, Scheme3 XGB 838 rows | Canonical unfiltered workflow. |
