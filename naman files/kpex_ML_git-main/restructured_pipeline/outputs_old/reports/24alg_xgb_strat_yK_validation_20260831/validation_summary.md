# 24-Algorithm XGB strat_y_K Validation Summary

Validation run: `24alg_xgb_strat_yK_validation_20260831`

## Conclusion

Status: **PASS WITH WARNINGS**. Scheme 1 and Scheme 3 XGB train, predict, and generate scoped compare/failure reports with valid 24-label predictions. Remaining warnings are duplicate physical entire-data prediction CSVs and the mixed-combination script's inability to scope to Scheme 1 + Scheme 3 only.

## Accuracy And Runtime Metrics

| Scope | Rows | Accuracy | Tol 0 | Tol 10 | Tol 20 | Tol 50 | Avg Ratio | Median Ratio | Max Ratio | Avg Gap | Median Gap | Wrong | Important Tol0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scheme1_test | 1149 | 36.99% | 36.99% | 50.39% | 59.88% | 71.45% | 809.6333 | 1.2372 | 580645.1613 | 59.3879 | 0.0090 | 724 | 162 |
| scheme1_entire | 4279 | 18.37% | 18.37% | 35.57% | 47.30% | 61.63% | 307.1832 | 1.2839 | 580645.1613 | 81.6780 | 0.0967 | 3493 | 888 |
| scheme3_test | 854 | 44.26% | 44.26% | 69.44% | 75.41% | 81.62% | 684.7579 | 1.0165 | 227848.1013 | 46.6486 | 0.0014 | 476 | 84 |
| scheme3_entire | 4279 | 80.42% | 80.42% | 90.84% | 92.90% | 94.70% | 996.7352 | 1.0000 | 2571428.5714 | 12.6141 | 0.0000 | 838 | 103 |

## Training Artifacts

- `scheme1`: train rows `9350`, test rows `2338`, classes seen `24`, model `restructured_pipeline/outputs/models/scheme1/xgboost/model_24alg_xgb_strat_yK_validation_20260831_scheme1.pkl`.
- `scheme3`: train rows `3425`, test rows `854`, classes seen `24`, model `restructured_pipeline/outputs/models/scheme3/xgboost/model_24alg_xgb_strat_yK_validation_20260831_scheme3.pkl`.

## Inconsistencies And Warnings

- `scheme1`: duplicate entire-data prediction CSV generated - {'entire_prediction_export_behavior': 'wrote_entire_data_export', 'entire_prediction_is_symlink': False, 'entire_prediction_hash_matches_training_canonical': True}
- `scheme3`: duplicate entire-data prediction CSV generated - {'entire_prediction_export_behavior': 'wrote_entire_data_export', 'entire_prediction_is_symlink': False, 'entire_prediction_hash_matches_training_canonical': True}
- `mixed_failure_case_combinations`: scoped Scheme 1 + Scheme 3-only run unsupported - Script exits because Scheme 1 + Scheme 2 pairing sheets become one-sided when Scheme 2 is intentionally excluded.

## Top Failure Patterns

### scheme1_test

Top confused pairs: `[{'fastest': 'UMKP-oriented-B-bnb', 'predicted': 'kPEX-oriented-B-bnb', 'count': 140}, {'fastest': 'UMKP-oriented-P-bnb', 'predicted': 'kPEX-oriented-B-bnb', 'count': 98}, {'fastest': 'UMKP-oriented-S-bnb', 'predicted': 'kPEX-oriented-B-bnb', 'count': 49}, {'fastest': 'UMKP+-oriented-S-bnb', 'predicted': 'kPEX-oriented-B-bnb', 'count': 47}, {'fastest': 'UMKP+-oriented-B-bnb', 'predicted': 'kPEX-oriented-B-bnb', 'count': 42}]`

Hardest datasets: `[{'dataset': 'Hook_1498', 'failure_count': 6}, {'dataset': 'pkustk14', 'failure_count': 6}, {'dataset': 'barrier2-11', 'failure_count': 5}, {'dataset': 'crankseg_1', 'failure_count': 5}, {'dataset': 'crystk03', 'failure_count': 5}]`

Hardest k values: `[{'k_value': 15, 'failure_count': 79}, {'k_value': 20, 'failure_count': 76}, {'k_value': 6, 'failure_count': 73}, {'k_value': 10, 'failure_count': 72}, {'k_value': 9, 'failure_count': 70}]`

### scheme1_entire

Top confused pairs: `[{'fastest': 'UMKP-oriented-B-bnb', 'predicted': 'kPEX-oriented-B-bnb', 'count': 680}, {'fastest': 'UMKP-oriented-P-bnb', 'predicted': 'kPEX-oriented-B-bnb', 'count': 506}, {'fastest': 'UMKP+-oriented-B-bnb', 'predicted': 'kPEX-oriented-B-bnb', 'count': 239}, {'fastest': 'UMKP+-oriented-S-bnb', 'predicted': 'kPEX-oriented-B-bnb', 'count': 210}, {'fastest': 'UMKP-oriented-S-bnb', 'predicted': 'kPEX-oriented-B-bnb', 'count': 210}]`

Hardest datasets: `[{'dataset': 'af_1_k101', 'failure_count': 11}, {'dataset': 'af_4_k101', 'failure_count': 11}, {'dataset': 'af_5_k101', 'failure_count': 11}, {'dataset': 'af_shell10', 'failure_count': 11}, {'dataset': 'af_shell1', 'failure_count': 11}]`

Hardest k values: `[{'k_value': 20, 'failure_count': 392}, {'k_value': 15, 'failure_count': 370}, {'k_value': 10, 'failure_count': 346}, {'k_value': 6, 'failure_count': 332}, {'k_value': 5, 'failure_count': 328}]`

### scheme3_test

Top confused pairs: `[{'fastest': 'kPEX-oriented-B-bnb', 'predicted': 'UMKP-oriented-P-bnb', 'count': 18}, {'fastest': 'UMKP-oriented-B-bnb', 'predicted': 'UMKP-oriented-P-bnb', 'count': 15}, {'fastest': 'UMKP-oriented-P-bnb', 'predicted': 'kPEX-oriented-B-bnb', 'count': 15}, {'fastest': 'UMKP-oriented-P-bnb', 'predicted': 'UMKP-oriented-B-bnb', 'count': 13}, {'fastest': 'UMKP-oriented-B-bnb', 'predicted': 'UMKP+-oriented-B-bnb', 'count': 12}]`

Hardest datasets: `[{'dataset': 'RM07R', 'failure_count': 5}, {'dataset': 'nd3k', 'failure_count': 4}, {'dataset': 'opt1', 'failure_count': 4}, {'dataset': 'ship_001', 'failure_count': 4}, {'dataset': 'soc-digg', 'failure_count': 4}]`

Hardest k values: `[{'k_value': 15, 'failure_count': 60}, {'k_value': 10, 'failure_count': 57}, {'k_value': 20, 'failure_count': 51}, {'k_value': 7, 'failure_count': 47}, {'k_value': 5, 'failure_count': 45}]`

### scheme3_entire

Top confused pairs: `[{'fastest': 'UMKP+-oriented-B-bnb', 'predicted': 'UMKP-oriented-B-bnb', 'count': 32}, {'fastest': 'UMKP-oriented-B-bnb', 'predicted': 'UMKP-oriented-P-bnb', 'count': 24}, {'fastest': 'kPEX-oriented-B-bnb', 'predicted': 'UMKP-oriented-P-bnb', 'count': 24}, {'fastest': 'UMKP-oriented-P-bnb', 'predicted': 'kPEX-oriented-P-bnb', 'count': 23}, {'fastest': 'kPEX-oriented-B-bnb', 'predicted': 'UMKP-oriented-B-bnb', 'count': 22}]`

Hardest datasets: `[{'dataset': 'barrier2-1', 'failure_count': 8}, {'dataset': 'barrier2-9', 'failure_count': 7}, {'dataset': 'pattern1', 'failure_count': 7}, {'dataset': 'reactome', 'failure_count': 7}, {'dataset': 'barrier2-2', 'failure_count': 6}]`

Hardest k values: `[{'k_value': 15, 'failure_count': 112}, {'k_value': 10, 'failure_count': 107}, {'k_value': 7, 'failure_count': 87}, {'k_value': 20, 'failure_count': 81}, {'k_value': 6, 'failure_count': 80}]`

## Generated Reports

- `reports_dir`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831`
- `compare_test`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/compare_wrong_predictions_test_data.xlsx`
- `compare_entire`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/compare_wrong_predictions_entire_data.xlsx`
- `tolerance_test`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/compare_wrong_predictions_tolerance_report_test_data.xlsx`
- `tolerance_entire`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/compare_wrong_predictions_tolerance_report_entire_data.xlsx`
- `important_test`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/compare_wrong_predictions_tolerance_important_failures_test_data.xlsx`
- `important_entire`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx`
- `filtered_test`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/failure_case_analysis_important_failures_test_data.xlsx`
- `filtered_entire`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/failure_case_analysis_important_failures_entire_data.xlsx`
- `failure_summary_test`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/failure_analysis_summary_test_data.xlsx`
- `failure_summary_entire`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/failure_analysis_summary_entire_data.xlsx`
- `unfiltered_test`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/failure_analysis_summary_unfiltered_test_data.xlsx`
- `unfiltered_entire`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/failure_analysis_summary_unfiltered_entire_data.xlsx`
- `mixed_scoped_attempt`: `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/mixed_failure_case_combinations_scoped_attempt.xlsx`

## Provenance

| Output | Generated By | Main Function / CLI Path | Input Files | Rows / Sheets | Notes |
|---|---|---|---|---|---|
| `restructured_pipeline/outputs/predictions/scheme1/xgboost/all_predictions_24alg_xgb_strat_yK_validation_20260831_scheme1.csv` | `train.py` | `main() -> train_scheme1() -> build_flat_results()` | `input_data/ml-data.csv`, `configs/project.yaml`, `configs/models_scheme1.yaml` | 4279 rows | Canonical training-time full-dataset prediction export. |
| `restructured_pipeline/outputs/predictions/scheme3/xgboost/all_predictions_24alg_xgb_strat_yK_validation_20260831_scheme3.csv` | `train.py` | `main() -> train_scheme3() -> evaluate_hierarchical_scheme3()` | `input_data/ml-data.csv`, `configs/project.yaml`, `configs/models_scheme3.yaml` | 4279 rows | Canonical Scheme 3 full-dataset prediction export. |
| `restructured_pipeline/outputs/predictions/scheme1/xgboost/all_predictions_24alg_xgb_strat_yK_validation_20260831_scheme1_predict_inference_test_data.csv` | `predict.py` | `main()` | `restructured_pipeline/outputs/models/scheme1/xgboost/model_24alg_xgb_strat_yK_validation_20260831_scheme1.pkl`, `restructured_pipeline/data/splits/scheme1/xgboost/test_indices_24alg_xgb_strat_yK_validation_20260831_scheme1.csv` | 1149 rows | Held-out test-set export. |
| `restructured_pipeline/outputs/predictions/scheme3/xgboost/all_predictions_24alg_xgb_strat_yK_validation_20260831_scheme3_predict_inference_test_data.csv` | `predict.py` | `main()` | `restructured_pipeline/outputs/models/scheme3/xgboost/model_24alg_xgb_strat_yK_validation_20260831_scheme3.pkl`, `restructured_pipeline/data/splits/scheme3/xgboost/test_indices_24alg_xgb_strat_yK_validation_20260831_scheme3.csv` | 854 rows | Held-out test-set export. |
| `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/compare_wrong_predictions_test_data.xlsx` | `compare_wrong_predictions.py` | `main() -> run_compare_mode() -> build_compare_wrong_predictions_from_files()` | `scheme1/scheme3 test wrong-case workbooks` | 485 rows / 1 sheet | Strict threshold >= 0.5. |
| `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/compare_wrong_predictions_tolerance_report_test_data.xlsx` | `compare_wrong_predictions.py` | `run_compare_mode() -> build_tolerance_sheet_frames() -> write_tolerance_sheet_frames()` | `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/compare_wrong_predictions_test_data.xlsx` | 12 sheets | Tolerance sheets 0/10/20/50 for scheme 1/2/3; Scheme 2 sheets empty by scope. |
| `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/compare_wrong_predictions_tolerance_important_failures_test_data.xlsx` | `compare_wrong_predictions.py` | `run_compare_mode() -> build_tolerance_important_failure_frames()` | `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/compare_wrong_predictions_tolerance_report_test_data.xlsx` | 12 tolerance sheets + filter_summary | Important-failure rows never exceed tolerance parent rows. |
| `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/failure_case_analysis_important_failures_entire_data.xlsx` | `failure_case_filter.py` | `main() -> filter_important_failures()` | `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/compare_wrong_predictions_entire_data.xlsx` | 991 rows / 1 sheet | Single-sheet input for enrich mode. |
| `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/failure_analysis_summary_entire_data.xlsx` | `failure_analysis_summary.py` | `main() -> run_enrichment_mode()` | `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/failure_case_analysis_important_failures_entire_data.xlsx` | 6 sheets | RF/Scheme 2 sheets empty by scope. |
| `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/failure_analysis_summary_unfiltered_entire_data.xlsx` | `failure_analysis_unfiltered_summary.py` | `main() -> load_unfiltered_records()` | `scheme1/scheme3 entire-data wrong-case workbooks`, `restructured_pipeline/outputs/reports/24alg_xgb_strat_yK_validation_20260831/failure_analysis_summary_entire_data.xlsx` | summary + 6 sheets | RF/Scheme 2 sheets empty by scope. |
