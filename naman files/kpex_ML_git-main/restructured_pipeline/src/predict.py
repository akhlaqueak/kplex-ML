"""Prediction entry point for the restructured pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


DEFAULT_PROJECT_CONFIG = Path("restructured_pipeline/configs/project.yaml")
from restructured_pipeline.src.saved_run_utils import (
    _build_signature_series,
    _normalize_bool_series,
    _parse_literal,
    load_results_for_inference,
    load_saved_results_df,
    load_saved_wrong_cases_df,
    normalize_inference_mode,
    output_mode_tag,
    resolve_base_run_name,
    resolve_predictions_path,
    select_inference_subset,
)


def create_prediction_run_name(base_run_name, inference_mode, file_suffix=None):
    if file_suffix:
        return file_suffix
    return f"{base_run_name}_predict_{output_mode_tag(inference_mode)}"


def resolve_prediction_output_paths(output_dirs, filename_prefixes, run_name):
    predictions_path = output_dirs["predictions"] / f"{filename_prefixes['all_predictions_prefix']}_{run_name}.csv"
    wrong_cases_xlsx_path = output_dirs["wrong_cases"] / f"{filename_prefixes['wrong_predictions_prefix']}_{run_name}.xlsx"
    return predictions_path, wrong_cases_xlsx_path


def _dataframes_match(left_df, right_df):
    from pandas.testing import assert_frame_equal

    def normalize_duplicate_columns(df):
        normalized = df.copy()
        seen = {}
        renamed = []
        for col in normalized.columns:
            count = seen.get(col, 0)
            renamed.append(col if count == 0 else f"{col}.{count}")
            seen[col] = count + 1
        normalized.columns = renamed
        return normalized

    left_df = normalize_duplicate_columns(left_df)
    right_df = normalize_duplicate_columns(right_df)

    if list(left_df.columns) != list(right_df.columns):
        return False

    try:
        assert_frame_equal(
            left_df.reset_index(drop=True),
            right_df.reset_index(drop=True),
            check_dtype=False,
            check_like=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        return True
    except AssertionError:
        return False


def _ensure_alias_path(alias_path: Path, target_path: Path) -> str:
    if alias_path == target_path:
        return "canonical_path"

    if alias_path.is_symlink():
        try:
            if alias_path.resolve() == target_path.resolve():
                return "existing_alias"
        except FileNotFoundError:
            return "broken_alias_preserved"
        return "different_alias_preserved"

    if alias_path.exists():
        return "existing_file_preserved"

    alias_path.parent.mkdir(parents=True, exist_ok=True)
    relative_target = Path(os.path.relpath(target_path, start=alias_path.parent))
    alias_path.symlink_to(relative_target)
    return "created_alias"


def maybe_reuse_canonical_entire_data_outputs(
    results_df,
    wrong_df,
    output_dirs,
    filename_prefixes,
    base_run_name,
    prediction_run_name,
):
    canonical_predictions_path, canonical_wrong_cases_path = resolve_prediction_output_paths(
        output_dirs,
        filename_prefixes,
        base_run_name,
    )
    export_predictions_path, export_wrong_cases_path = resolve_prediction_output_paths(
        output_dirs,
        filename_prefixes,
        prediction_run_name,
    )

    if not canonical_predictions_path.exists() or not canonical_wrong_cases_path.exists():
        return None

    canonical_results_df = load_saved_results_df(canonical_predictions_path)
    canonical_wrong_df = load_saved_wrong_cases_df(canonical_wrong_cases_path)
    if not _dataframes_match(results_df, canonical_results_df):
        return None
    if not _dataframes_match(wrong_df, canonical_wrong_df):
        return None

    predictions_alias_status = _ensure_alias_path(export_predictions_path, canonical_predictions_path)
    wrong_cases_alias_status = _ensure_alias_path(export_wrong_cases_path, canonical_wrong_cases_path)
    return {
        "predictions_path": str(export_predictions_path),
        "wrong_cases_xlsx_path": str(export_wrong_cases_path),
        "canonical_predictions_path": str(canonical_predictions_path),
        "canonical_wrong_cases_xlsx_path": str(canonical_wrong_cases_path),
        "export_behavior": "reused_canonical_entire_data_outputs",
        "predictions_alias_status": predictions_alias_status,
        "wrong_cases_alias_status": wrong_cases_alias_status,
    }


def load_results_for_prediction(
    project_config,
    scheme,
    model_family,
    base_run_name,
    tol,
    predictions_path=None,
    model_path=None,
):
    return load_results_for_inference(
        project_config=project_config,
        scheme=scheme,
        model_family=model_family,
        base_run_name=base_run_name,
        tol=tol,
        predictions_path=predictions_path,
        model_path=model_path,
    )


def select_prediction_subset(results_df, project_config, scheme, model_family, inference_mode, base_run_name):
    return select_inference_subset(
        results_df,
        project_config=project_config,
        scheme=scheme,
        model_family=model_family,
        inference_mode=inference_mode,
        base_run_name=base_run_name,
        consumer_label="predict",
    )


def run_prediction(
    scheme,
    model_family,
    variant=None,
    split_mode=None,
    tol=None,
    inference_mode="inference_test_set",
    save_files=True,
    force_export=False,
    file_suffix=None,
    run_name=None,
    predictions_path=None,
    model_path=None,
    project_config_path=DEFAULT_PROJECT_CONFIG,
    model_config_path=None,
):
    from restructured_pipeline.src.train import (
        build_wrong_predictions,
        load_yaml_config,
        normalize_model_family,
        normalize_scheme,
        save_dataframe_outputs,
    )

    scheme = normalize_scheme(scheme)
    model_family = normalize_model_family(model_family)
    inference_mode = normalize_inference_mode(inference_mode)

    project_config = load_yaml_config(project_config_path)
    tol = project_config["tolerances"]["tie_tolerance"] if tol is None else tol
    base_run_name = resolve_base_run_name(
        scheme=scheme,
        model_family=model_family,
        variant=variant,
        split_mode=split_mode,
        tol=tol,
        run_name=run_name,
        project_config_path=project_config_path,
        model_config_path=model_config_path,
    )

    results_df, output_dirs, load_metadata = load_results_for_prediction(
        project_config=project_config,
        scheme=scheme,
        model_family=model_family,
        base_run_name=base_run_name,
        tol=tol,
        predictions_path=predictions_path,
        model_path=model_path,
    )
    results_subset = select_prediction_subset(
        results_df,
        project_config=project_config,
        scheme=scheme,
        model_family=model_family,
        inference_mode=inference_mode,
        base_run_name=base_run_name,
    )

    algorithms = project_config["targets"]["scheme1"]["algorithms"]
    epsilon = project_config["evaluation"]["epsilon"]
    wrong_subset = build_wrong_predictions(results_subset, algorithms, epsilon=epsilon)

    prediction_run_name = create_prediction_run_name(
        base_run_name=base_run_name,
        inference_mode=inference_mode,
        file_suffix=file_suffix,
    )

    artifacts = {}
    if save_files:
        if inference_mode == "inference_entire_data" and not force_export:
            reused_artifacts = maybe_reuse_canonical_entire_data_outputs(
                results_subset,
                wrong_subset,
                output_dirs,
                project_config["output_locations"]["filenames"],
                base_run_name,
                prediction_run_name,
            )
            if reused_artifacts is not None:
                artifacts.update(reused_artifacts)
            else:
                artifacts.update(
                    save_dataframe_outputs(
                        results_subset,
                        wrong_subset,
                        output_dirs,
                        project_config["output_locations"]["filenames"],
                        prediction_run_name,
                    )
                )
                artifacts["export_behavior"] = "wrote_entire_data_export"
        else:
            artifacts.update(
                save_dataframe_outputs(
                    results_subset,
                    wrong_subset,
                    output_dirs,
                    project_config["output_locations"]["filenames"],
                    prediction_run_name,
                )
            )
            artifacts["export_behavior"] = "wrote_prediction_export"

    return {
        "scheme": scheme,
        "model_family": model_family,
        "variant": variant,
        "split_mode": split_mode,
        "base_run_name": base_run_name,
        "prediction_run_name": prediction_run_name,
        "inference_mode": inference_mode,
        "results_source": load_metadata["results_source"],
        "loaded_predictions_path": load_metadata["loaded_predictions_path"],
        "loaded_model_path": load_metadata["loaded_model_path"],
        "results_df": results_subset,
        "wrong_df": wrong_subset,
        "artifacts": artifacts,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate predictions from saved pipeline outputs.")
    parser.add_argument("--scheme", required=True, choices=["scheme1", "scheme2", "scheme3"])
    parser.add_argument("--model-family", required=True, choices=["rf", "random_forest", "xgb", "xgboost"])
    parser.add_argument("--variant", default=None, help="Scheme 1 variant name, e.g. rf_strat_y or xgb_strat_y.")
    parser.add_argument("--split-mode", default=None, help="Optional split mode override, e.g. strat_y or strat_y_K.")
    parser.add_argument(
        "--inference-mode",
        default="inference_test_set",
        choices=["test", "test_set", "inference_test_set", "entire", "entire_data", "full", "inference_entire_data"],
        help="Write predictions for the held-out test set or the entire dataset.",
    )
    parser.add_argument("--tol", type=float, default=None, help="Tolerance used when resolving the saved run name.")
    parser.add_argument("--run-name", default=None, help="Existing saved training run name to load.")
    parser.add_argument("--predictions-path", default=None, help="Optional explicit all_predictions CSV path.")
    parser.add_argument("--model-path", default=None, help="Optional explicit saved model artifact path.")
    parser.add_argument("--file-suffix", default=None, help="Optional explicit output suffix.")
    parser.add_argument("--project-config", default=str(DEFAULT_PROJECT_CONFIG))
    parser.add_argument("--model-config", default=None, help="Optional explicit model config path.")
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="For entire-data prediction only, write a physical duplicate export instead of reusing the canonical training-time files.",
    )
    parser.add_argument(
        "--no-save-files",
        action="store_true",
        help="Skip writing CSV/XLSX prediction artifacts; still prints the summary.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    result = run_prediction(
        scheme=args.scheme,
        model_family=args.model_family,
        variant=args.variant,
        split_mode=args.split_mode,
        tol=args.tol,
        inference_mode=args.inference_mode,
        save_files=not args.no_save_files,
        force_export=args.force_export,
        file_suffix=args.file_suffix,
        run_name=args.run_name,
        predictions_path=args.predictions_path,
        model_path=args.model_path,
        project_config_path=args.project_config,
        model_config_path=args.model_config,
    )

    print(f"Prediction export complete: {result['prediction_run_name']}")
    print(f"Base run: {result['base_run_name']}")
    print(f"Rows exported: {len(result['results_df'])}")
    print(f"Wrong predictions: {len(result['wrong_df'])}")
    if not args.no_save_files:
        export_behavior = result["artifacts"].get("export_behavior")
        if export_behavior == "reused_canonical_entire_data_outputs":
            print("Export behavior: reused canonical entire-data outputs; duplicate-named export paths were aliased instead of rewritten.")
        elif export_behavior:
            print(f"Export behavior: {export_behavior}")
        print("Artifacts:")
        for key, value in sorted(result["artifacts"].items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
