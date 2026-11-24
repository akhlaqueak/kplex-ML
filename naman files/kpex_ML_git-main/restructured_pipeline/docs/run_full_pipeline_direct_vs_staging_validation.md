# `run_full_pipeline.sh` Direct vs Staging Validation

Date: 2026-06-23

## Goal

Validate that:

- `run_full_pipeline.sh --compare-input-mode direct` is stable for the current
  preferred wrapper path
- `run_full_pipeline.sh --compare-input-mode staging` still executes as the
  legacy fallback
- staging mode remains untouched and is not removed in this validation

This was a validation-only pass. No compare semantics, compatibility paths, or
staging behavior were modified.

## Validation Setup

Representative run:

- scheme: `scheme1`
- model family: `random_forest`
- split mode: `strat_y`

To keep the validation lightweight, both wrapper runs used:

- `--skip-full-predict`

That does not change the compare-input semantics being validated here, because
both direct and staging compare paths consume:

- the training wrong-case workbook
- the held-out prediction wrong-case workbook

The compare reports themselves were written to `/private/tmp` so the validation
did not overwrite canonical report locations.

## Commands Run

### Direct Mode

```bash
MPLCONFIGDIR=/private/tmp/mpl_wrapper_compare_direct \
bash restructured_pipeline/run_full_pipeline.sh \
  --python-bin /opt/miniconda3/envs/kpex/bin/python \
  --scheme scheme1 \
  --model-family random_forest \
  --split-mode strat_y \
  --variant rf_strat_y \
  --run-name scheme1_random_forest_strat_y_direct_validation_20260623 \
  --compare-input-mode direct \
  --compare-input-root /private/tmp/run_full_pipeline_mode_validation/direct_compare_inputs \
  --report-output-path /private/tmp/run_full_pipeline_mode_validation/direct/compare_wrong_predictions_direct.xlsx \
  --tolerance-output-path /private/tmp/run_full_pipeline_mode_validation/direct/compare_wrong_predictions_tolerance_direct.xlsx \
  --important-failure-output-path /private/tmp/run_full_pipeline_mode_validation/direct/compare_wrong_predictions_important_direct.xlsx \
  --skip-full-predict
```

### Staging Mode

```bash
MPLCONFIGDIR=/private/tmp/mpl_wrapper_compare_staging \
bash restructured_pipeline/run_full_pipeline.sh \
  --python-bin /opt/miniconda3/envs/kpex/bin/python \
  --scheme scheme1 \
  --model-family random_forest \
  --split-mode strat_y \
  --variant rf_strat_y \
  --run-name scheme1_random_forest_strat_y_staging_validation_20260623 \
  --compare-input-mode staging \
  --compare-input-root /private/tmp/run_full_pipeline_mode_validation/staging_compare_inputs \
  --report-output-path /private/tmp/run_full_pipeline_mode_validation/staging/compare_wrong_predictions_staging.xlsx \
  --tolerance-output-path /private/tmp/run_full_pipeline_mode_validation/staging/compare_wrong_predictions_tolerance_staging.xlsx \
  --important-failure-output-path /private/tmp/run_full_pipeline_mode_validation/staging/compare_wrong_predictions_important_staging.xlsx \
  --skip-full-predict
```

## Output Files Created

### Direct Mode

- `/private/tmp/run_full_pipeline_mode_validation/direct/compare_wrong_predictions_direct.xlsx`
- `/private/tmp/run_full_pipeline_mode_validation/direct/compare_wrong_predictions_tolerance_direct.xlsx`
- `/private/tmp/run_full_pipeline_mode_validation/direct/compare_wrong_predictions_important_direct.xlsx`

### Staging Mode

- `/private/tmp/run_full_pipeline_mode_validation/staging/compare_wrong_predictions_staging.xlsx`
- `/private/tmp/run_full_pipeline_mode_validation/staging/compare_wrong_predictions_tolerance_staging.xlsx`
- `/private/tmp/run_full_pipeline_mode_validation/staging/compare_wrong_predictions_important_staging.xlsx`

### Staging Compatibility Folders

Created only in staging mode:

- `/private/tmp/run_full_pipeline_mode_validation/staging_compare_inputs/expand_and_split/`
- `/private/tmp/run_full_pipeline_mode_validation/staging_compare_inputs/split_and_expand/`

Direct mode did not create:

- `/private/tmp/run_full_pipeline_mode_validation/direct_compare_inputs`

## Metadata Checks

Compare workbook sheet set for both modes:

- `on 20% test set`
- `on entire dataset`

### Direct Mode

| Check | Result | Evidence |
|------|------|------|
| compare workbook exists | pass | workbook written to `/private/tmp/run_full_pipeline_mode_validation/direct/compare_wrong_predictions_direct.xlsx` |
| tolerance workbook exists | pass | workbook written to `/private/tmp/run_full_pipeline_mode_validation/direct/compare_wrong_predictions_tolerance_direct.xlsx` |
| important-failure workbook exists | pass | workbook written to `/private/tmp/run_full_pipeline_mode_validation/direct/compare_wrong_predictions_important_direct.xlsx` |
| blank `scheme` values | pass | `0` blank rows in both sheets |
| blank `model` values | pass | `0` blank rows in both sheets |
| `strat_on` populated | pass | all rows populated with `y` |
| `inference_on` populated | pass | all rows populated with `on 20% test set` or `on entire dataset` |
| staged compatibility folders created | pass | no direct-mode compare-input root was created |
| Scheme 1 tolerance rows | pass | `87, 87, 87, 87` across `tol_0, tol_10, tol_20, tol_50` |
| Scheme 1 important-failure rows | pass | `56, 56, 56, 56` across `tol_0, tol_10, tol_20, tol_50` |

Observed direct compare metadata:

- sheet `on 20% test set`
  - rows: `41`
  - `scheme`: `scheme 1`
  - `model`: `RF`
  - `strat_on`: `y`
  - `inference_on`: `on 20% test set`
- sheet `on entire dataset`
  - rows: `46`
  - `scheme`: `scheme 1`
  - `model`: `RF`
  - `strat_on`: `y`
  - `inference_on`: `on entire dataset`

### Staging Mode

| Check | Result | Evidence |
|------|------|------|
| compare workbook exists | pass | workbook written to `/private/tmp/run_full_pipeline_mode_validation/staging/compare_wrong_predictions_staging.xlsx` |
| tolerance workbook exists | pass | workbook written to `/private/tmp/run_full_pipeline_mode_validation/staging/compare_wrong_predictions_tolerance_staging.xlsx` |
| important-failure workbook exists | pass | workbook written to `/private/tmp/run_full_pipeline_mode_validation/staging/compare_wrong_predictions_important_staging.xlsx` |
| blank `scheme` values | fail | all `41` / `46` data rows were blank in the two compare sheets |
| blank `model` values | fail | all `41` / `46` data rows were blank in the two compare sheets |
| `strat_on` populated | pass | all rows populated with `y` |
| `inference_on` populated | pass | all rows populated with `on 20% test set` or `on entire dataset` |
| staging compatibility folders created | pass | `expand_and_split/` and `split_and_expand/` were created as expected |
| staging mode still runs as fallback | pass | wrapper completed successfully and produced all three report workbooks |
| Scheme 1 tolerance rows | fail for metadata equivalence | `0, 0, 0, 0` across the Scheme 1 tolerance sheets |
| Scheme 1 important-failure rows | fail for metadata equivalence | `0, 0, 0, 0` across the Scheme 1 important-failure sheets |

Observed staging compare metadata:

- sheet `on 20% test set`
  - rows: `41`
  - `scheme`: blank
  - `model`: blank
  - `strat_on`: `y`
  - `inference_on`: `on 20% test set`
- sheet `on entire dataset`
  - rows: `46`
  - `scheme`: blank
  - `model`: blank
  - `strat_on`: `y`
  - `inference_on`: `on entire dataset`

## Validation Conclusion

### Is Direct Mode Safe As The Preferred Mode?

Yes.

For this representative wrapper validation, direct mode was stable and produced
the expected metadata:

- compare workbook present
- tolerance workbook present
- important-failure workbook present
- no blank `scheme`
- no blank `model`
- `strat_on` populated
- `inference_on` populated
- no staged compatibility folders created

This supports keeping direct mode as the preferred wrapper path.

### Should Staging Remain Fallback?

Yes, but only as fallback.

Reason:

- staging mode still executes successfully and still produces the expected
  fallback staged compare-input folders
- however, in this validation it did not preserve `scheme` / `model` metadata
  in the compare workbook
- that metadata loss propagated into the tolerance and important-failure
  outputs, leaving Scheme 1 sheet counts at `0`

So the current evidence supports:

- keep staging mode available
- do not remove or refactor it in this task
- do not treat staging mode as metadata-equivalent to direct mode without a
  future dedicated validation or repair task

## Recommended Status

- direct mode: `safe as preferred wrapper mode`
- staging mode: `keep as untouched legacy fallback`

This matches the current high-risk guidance in
`docs/high_risk_cleanup_checklist.md`:

- direct mode should remain the preferred path
- staging should remain available until a future dedicated validation task
  decides whether it can be repaired or retired
