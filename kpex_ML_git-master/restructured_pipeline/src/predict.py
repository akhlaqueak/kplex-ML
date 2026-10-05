"""Prediction entry point for the restructured pipeline."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


DEFAULT_PROJECT_CONFIG = Path("restructured_pipeline/configs/project.yaml")


def _normalize_scheme4_family(scheme4_family):
    if scheme4_family is None:
        return None
    mapping = {
        "oriented": "oriented",
        "forward": "forward",
        "non_oriented": "forward",
        "non-oriented": "forward",
    }
    key = str(scheme4_family).strip().lower()
    if key not in mapping:
        raise ValueError(
            "Unsupported Scheme 4 family: "
            f"{scheme4_family!r}. Expected 'oriented' or 'forward'."
        )
    return mapping[key]


def _normalize_scheme5_family(scheme5_family):
    if scheme5_family is None:
        return None
    key = str(scheme5_family).strip().lower()
    mapping = {"b": "b", "binary": "b", "other": "other", "non_b": "other", "non-b": "other"}
    if key not in mapping:
        raise ValueError(f"Unsupported Scheme 5 family: {scheme5_family!r}. Expected 'b' or 'other'.")
    return mapping[key]


def _ensure_output_dirs(project_config, scheme, model_family, scheme4_family=None, scheme5_family=None, experiment_subdir=None):
    """Call train.ensure_output_dirs with the Scheme-4 family when needed."""
    from restructured_pipeline.src.train import ensure_output_dirs

    if scheme == "scheme4":
        if scheme4_family is None:
            raise ValueError("--scheme4-family is required for Scheme 4.")
        return ensure_output_dirs(project_config, scheme, model_family, subfamily=scheme4_family, experiment_subdir=experiment_subdir)
    if scheme == "scheme5":
        if scheme5_family is None:
            raise ValueError("--scheme5-family is required for Scheme 5.")
        return ensure_output_dirs(project_config, scheme, model_family, subfamily=scheme5_family, experiment_subdir=experiment_subdir)
    return ensure_output_dirs(project_config, scheme, model_family, experiment_subdir=experiment_subdir)


def _resolve_active_algorithms(results_df, model_artifact=None):
    """Resolve the classifier vocabulary without assuming Scheme 1's 24 classes."""
    if model_artifact is not None:
        saved = model_artifact.get("algorithms")
        if saved:
            return list(saved)

    # Saved prediction CSVs contain one prob_<algorithm> column per active class.
    prob_algorithms = [
        str(column)[5:]
        for column in results_df.columns
        if str(column).startswith("prob_")
    ]
    if prob_algorithms:
        return prob_algorithms

    raise ValueError(
        "Could not resolve the active algorithm vocabulary from the saved model "
        "artifact or prob_* prediction columns."
    )


def normalize_inference_mode(inference_mode):
    mapping = {
        "test": "inference_test_set",
        "test_set": "inference_test_set",
        "inference_test_set": "inference_test_set",
        "entire": "inference_entire_data",
        "entire_data": "inference_entire_data",
        "full": "inference_entire_data",
        "inference_entire_data": "inference_entire_data",
    }
    if inference_mode not in mapping:
        raise ValueError(f"Unsupported inference_mode: {inference_mode}")
    return mapping[inference_mode]


def output_mode_tag(inference_mode):
    normalized = normalize_inference_mode(inference_mode)
    mapping = {
        "inference_test_set": "inference_test_data",
        "inference_entire_data": "inference_entire_data",
    }
    return mapping[normalized]


def _parse_literal(value):
    if value is None:
        return None
    if isinstance(value, float):
        try:
            if value != value:
                return None
        except Exception:
            pass
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return value
    return value


def _normalize_bool_series(series):
    if str(series.dtype) == "bool":
        return series

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    return series.apply(
        lambda value: mapping.get(str(value).strip().lower(), value) if value is not None else value
    )


def load_saved_results_df(predictions_path):
    import pandas as pd

    results_df = pd.read_csv(predictions_path)

    for col in ["fastest_algorithms", "top3_predicted", "tol_fastest"]:
        if col in results_df.columns:
            results_df[col] = results_df[col].apply(_parse_literal)

    if "correct" in results_df.columns:
        results_df["correct"] = _normalize_bool_series(results_df["correct"])

    return results_df


def create_prediction_run_name(base_run_name, inference_mode, file_suffix=None):
    if file_suffix:
        return file_suffix
    return f"{base_run_name}_predict_{output_mode_tag(inference_mode)}"


def resolve_base_run_name(
    scheme,
    model_family,
    variant,
    split_mode,
    tol,
    run_name,
    project_config_path,
    model_config_path,
    scheme4_family=None,
    scheme5_family=None,
    experiment_subdir=None
):
    if run_name:
        return run_name

    from restructured_pipeline.src.train import (
        create_run_name,
        load_yaml_config,
        normalize_model_family,
        normalize_scheme,
        resolve_model_config_path,
    )

    scheme = normalize_scheme(scheme)
    model_family = normalize_model_family(model_family)
    project_config = load_yaml_config(project_config_path)
    model_config = load_yaml_config(resolve_model_config_path(scheme, model_config_path))

    if scheme == "scheme4":
        scheme4_family = _normalize_scheme4_family(scheme4_family)
        if scheme4_family is None:
            raise ValueError("--scheme4-family is required for Scheme 4.")
        if model_family != "xgboost":
            raise ValueError("Scheme 4 currently supports XGBoost only.")
        family_cfg = model_config["scheme4"][scheme4_family][model_family]["variants"]
        if variant is None:
            variant = next(iter(family_cfg))
        if variant not in family_cfg:
            raise ValueError(
                f"Unknown Scheme 4 variant {variant!r} for family {scheme4_family!r}."
            )
        split_mode = family_cfg[variant]["split_mode"]
        return f"scheme4_{scheme4_family}_{model_family}_{variant}"

    if scheme == "scheme5":
        scheme5_family = _normalize_scheme5_family(scheme5_family)
        if scheme5_family is None:
            raise ValueError("--scheme5-family is required for Scheme 5.")
        if model_family not in {"random_forest", "xgboost"}:
            raise ValueError("Scheme 5 supports random_forest and xgboost.")
        family_cfg = model_config["scheme5"][scheme5_family][model_family]["variants"]
        if variant is None:
            variant = next(iter(family_cfg))
        if variant not in family_cfg:
            raise ValueError(f"Unknown Scheme 5 variant {variant!r} for family {scheme5_family!r}.")
        return f"scheme5_{scheme5_family}_{model_family}_{variant}"

    if scheme == "scheme1":
        family_cfg = model_config["scheme1"][model_family]["variants"]
        if variant is None:
            variant = next(iter(family_cfg))
        if variant not in family_cfg:
            raise ValueError(f"Unknown variant '{variant}' for {model_family}")
        split_mode = family_cfg[variant]["split_mode"]
        return create_run_name(
            scheme,
            model_family,
            split_mode,
            variant=variant,
            tol=tol,
        )

    split_mode = split_mode or project_config["split_rules"]["defaults"]["stratify_on_target_only"]
    return create_run_name(
        scheme,
        model_family,
        split_mode,
        variant=None,
        tol=tol,
    )


def resolve_predictions_path(
    project_config, scheme, model_family, base_run_name, predictions_path=None, scheme4_family=None, scheme5_family=None, experiment_subdir=None
):
    output_dirs = _ensure_output_dirs(
        project_config, scheme, model_family, scheme4_family=scheme4_family, scheme5_family=scheme5_family, experiment_subdir=experiment_subdir
    )
    if predictions_path is not None:
        return Path(predictions_path), output_dirs

    filename_prefixes = project_config["output_locations"]["filenames"]
    candidate = output_dirs["predictions"] / f"{filename_prefixes['all_predictions_prefix']}_{base_run_name}.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Saved predictions not found at {candidate}. Run train.py with --save-files first "
            "or pass --predictions-path/--run-name."
        )
    return candidate, output_dirs


def load_results_for_prediction(
    project_config,
    scheme,
    model_family,
    base_run_name,
    tol,
    predictions_path=None,
    model_path=None,
    scheme4_family=None,
    scheme5_family=None,
    experiment_subdir=None,
):
    from restructured_pipeline.src.train import (
        build_results_from_model_artifact,
        load_model_artifact,
    )

    if predictions_path is not None:
        resolved_predictions_path, output_dirs = resolve_predictions_path(
            project_config=project_config,
            scheme=scheme,
            model_family=model_family,
            base_run_name=base_run_name,
            predictions_path=predictions_path,
            scheme4_family=scheme4_family,
            scheme5_family=scheme5_family,
        )
        loaded_df = load_saved_results_df(resolved_predictions_path)
        return loaded_df, output_dirs, {
            "results_source": "saved_predictions",
            "loaded_predictions_path": str(resolved_predictions_path),
            "loaded_model_path": None,
            "active_algorithms": _resolve_active_algorithms(loaded_df),
        }

    # load_model_artifact() retains the legacy signature and does not accept
    # a Scheme-4 family argument. Resolve the family-aware output directory
    # here, then pass an explicit model path to the legacy loader.
    output_dirs = _ensure_output_dirs(
        project_config,
        scheme,
        model_family,
        scheme4_family=scheme4_family,
        scheme5_family=scheme5_family,
        experiment_subdir=experiment_subdir,
    )

    resolved_input_model_path = model_path
    if resolved_input_model_path is None and (scheme in {"scheme4", "scheme5"} or experiment_subdir):
        resolved_input_model_path = (
            output_dirs["models"] / f"model_{base_run_name}.pkl"
        )

    model_artifact, resolved_model_path, _legacy_output_dirs = load_model_artifact(
        project_config,
        scheme,
        model_family,
        base_run_name,
        model_path=resolved_input_model_path,
    )
    if model_artifact is not None:
        _, _, results_df, _ = build_results_from_model_artifact(
            project_config,
            scheme,
            model_artifact,
            tol=tol,
        )
        return results_df, output_dirs, {
            "results_source": "saved_model",
            "loaded_predictions_path": None,
            "loaded_model_path": str(resolved_model_path),
            "active_algorithms": list(model_artifact.get("algorithms", [])),
        }

    resolved_predictions_path, output_dirs = resolve_predictions_path(
        project_config=project_config,
        scheme=scheme,
        model_family=model_family,
        base_run_name=base_run_name,
        predictions_path=predictions_path,
        scheme4_family=scheme4_family,
        scheme5_family=scheme5_family,
        experiment_subdir=experiment_subdir,
    )
    loaded_df = load_saved_results_df(resolved_predictions_path)
    return loaded_df, output_dirs, {
        "results_source": "saved_predictions",
        "loaded_predictions_path": str(resolved_predictions_path),
        "loaded_model_path": None,
    }


def load_test_split_reference(
    project_config, scheme, model_family, base_run_name, scheme4_family=None, scheme5_family=None, experiment_subdir=None
):
    from restructured_pipeline.src.train import load_and_preprocess_data

    import pandas as pd

    output_dirs = _ensure_output_dirs(
        project_config, scheme, model_family, scheme4_family=scheme4_family, scheme5_family=scheme5_family, experiment_subdir=experiment_subdir
    )
    split_path = output_dirs["splits"] / f"test_indices_{base_run_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            "Saved split indices were not found for this run. "
            "Run train.py with --save-files first so predict can reproduce the test-set routing."
        )

    split_df = pd.read_csv(split_path)
    split_index_col = "source_row_index" if "source_row_index" in split_df.columns else "index"
    cat_df = load_and_preprocess_data(project_config, scheme)
    return cat_df.loc[split_df[split_index_col]].copy()


def _build_signature_series(df, signature_cols):
    return df[signature_cols].astype(str).agg("||".join, axis=1)


def select_prediction_subset(
    results_df, project_config, scheme, model_family, inference_mode, base_run_name, scheme4_family=None, scheme5_family=None, experiment_subdir=None
):
    if inference_mode == "inference_entire_data":
        return results_df.copy()

    reference_df = load_test_split_reference(
        project_config, scheme, model_family, base_run_name, scheme4_family=scheme4_family, scheme5_family=scheme5_family, experiment_subdir=experiment_subdir
    )
    if "source_row_index" in results_df.columns and "source_row_index" in reference_df.columns:
        return results_df.loc[results_df["source_row_index"].isin(reference_df["source_row_index"])].copy()

    signature_cols = ["dataset"] + [
        col for col in project_config["features"]["feature_cols"] if col in results_df.columns
    ]
    reference_signature = set(_build_signature_series(reference_df, signature_cols))
    result_signature = _build_signature_series(results_df, signature_cols)
    return results_df.loc[result_signature.isin(reference_signature)].copy()


def run_prediction(
    scheme,
    model_family,
    variant=None,
    split_mode=None,
    tol=None,
    inference_mode="inference_test_set",
    save_files=True,
    file_suffix=None,
    run_name=None,
    predictions_path=None,
    model_path=None,
    project_config_path=DEFAULT_PROJECT_CONFIG,
    model_config_path=None,
    scheme4_family=None,
    scheme5_family=None,
    experiment_subdir=None,
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
    scheme4_family = _normalize_scheme4_family(scheme4_family)
    if scheme == "scheme4" and scheme4_family is None:
        raise ValueError("--scheme4-family is required for Scheme 4.")
    if scheme != "scheme4" and scheme4_family is not None:
        raise ValueError("--scheme4-family is only valid with --scheme scheme4.")
    if scheme == "scheme5" and scheme5_family is None:
        raise ValueError("--scheme5-family is required for Scheme 5.")
    if scheme != "scheme5" and scheme5_family is not None:
        raise ValueError("--scheme5-family is only valid with --scheme scheme5.")
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
        scheme4_family=scheme4_family,
        scheme5_family=scheme5_family,
        experiment_subdir=experiment_subdir,
    )

    results_df, output_dirs, load_metadata = load_results_for_prediction(
        project_config=project_config,
        scheme=scheme,
        model_family=model_family,
        base_run_name=base_run_name,
        tol=tol,
        predictions_path=predictions_path,
        model_path=model_path,
        scheme4_family=scheme4_family,
        scheme5_family=scheme5_family,
        experiment_subdir=experiment_subdir,
    )
    results_subset = select_prediction_subset(
        results_df,
        project_config=project_config,
        scheme=scheme,
        model_family=model_family,
        inference_mode=inference_mode,
        base_run_name=base_run_name,
        scheme4_family=scheme4_family,
        scheme5_family=scheme5_family,
        experiment_subdir=experiment_subdir,
    )

    active_algorithms = load_metadata.get("active_algorithms") or _resolve_active_algorithms(
        results_subset
    )
    epsilon = project_config["evaluation"]["epsilon"]
    wrong_subset = build_wrong_predictions(results_subset, active_algorithms, epsilon=epsilon)

    prediction_run_name = create_prediction_run_name(
        base_run_name=base_run_name,
        inference_mode=inference_mode,
        file_suffix=file_suffix,
    )

    artifacts = {}
    if save_files:
        artifacts.update(
            save_dataframe_outputs(
                results_subset,
                wrong_subset,
                output_dirs,
                project_config["output_locations"]["filenames"],
                prediction_run_name,
            )
        )

    return {
        "scheme": scheme,
        "model_family": model_family,
        "scheme4_family": scheme4_family,
        "scheme5_family": scheme5_family,
        "algorithms": active_algorithms,
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
    parser.add_argument("--scheme", required=True, choices=["scheme1", "scheme2", "scheme3", "scheme4", "scheme5"])
    parser.add_argument("--model-family", required=True, choices=["rf", "random_forest", "xgb", "xgboost"])
    parser.add_argument(
        "--scheme4-family",
        default=None,
        choices=["oriented", "forward"],
        help="Scheme 4 subproblem: oriented or forward. Required for Scheme 4.",
    )
    parser.add_argument("--scheme5-family", default=None, choices=["b", "other"], help="Scheme 5 subproblem: 6 B algorithms or 18 Other algorithms.")
    parser.add_argument("--experiment-subdir", default=None, help="Optional output subdirectory below scheme/model, e.g. a Scheme 1 weighting scheme.")
    parser.add_argument("--variant", default=None, help="Flat-model variant name, e.g. xgb_strat_y or xgb_strat_y_K.")
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
        scheme4_family=args.scheme4_family,
        scheme5_family=args.scheme5_family,
        experiment_subdir=args.experiment_subdir,
        variant=args.variant,
        split_mode=args.split_mode,
        tol=args.tol,
        inference_mode=args.inference_mode,
        save_files=not args.no_save_files,
        file_suffix=args.file_suffix,
        run_name=args.run_name,
        predictions_path=args.predictions_path,
        model_path=args.model_path,
        project_config_path=args.project_config,
        model_config_path=args.model_config,
    )

    print(f"Prediction export complete: {result['prediction_run_name']}")
    print(f"Base run: {result['base_run_name']}")
    print(f"Rows written: {len(result['results_df'])}")
    print(f"Wrong predictions: {len(result['wrong_df'])}")
    if not args.no_save_files:
        print("Artifacts:")
        for key, value in sorted(result["artifacts"].items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
