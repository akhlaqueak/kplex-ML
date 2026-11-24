"""Shared helpers for saved-run inference entry points."""

from __future__ import annotations

from pathlib import Path

from restructured_pipeline.src.reporting_utils import parse_saved_run_literal


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
    return parse_saved_run_literal(value)


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


def _normalize_saved_results_df(results_df):
    literal_cols = {"fastest_algorithms", "top3_predicted", "tol_fastest"}
    bool_cols = {"correct", "correct_tol"}

    for col in results_df.columns:
        base_col = str(col).split(".")[0]
        if base_col in literal_cols:
            results_df[col] = results_df[col].apply(_parse_literal)
        if base_col in bool_cols:
            results_df[col] = _normalize_bool_series(results_df[col])

    return results_df


def load_saved_results_df(predictions_path):
    import pandas as pd

    results_df = pd.read_csv(predictions_path)
    return _normalize_saved_results_df(results_df)


def load_saved_wrong_cases_df(wrong_cases_path):
    import pandas as pd

    wrong_df = pd.read_excel(wrong_cases_path)
    return _normalize_saved_results_df(wrong_df)


def resolve_base_run_name(
    scheme,
    model_family,
    variant,
    split_mode,
    tol,
    run_name,
    project_config_path,
    model_config_path,
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


def resolve_predictions_path(project_config, scheme, model_family, base_run_name, predictions_path=None):
    from restructured_pipeline.src.train import ensure_output_dirs

    output_dirs = ensure_output_dirs(project_config, scheme, model_family)
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


def load_results_for_inference(
    project_config,
    scheme,
    model_family,
    base_run_name,
    tol,
    predictions_path=None,
    model_path=None,
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
        )
        return load_saved_results_df(resolved_predictions_path), output_dirs, {
            "results_source": "saved_predictions",
            "loaded_predictions_path": str(resolved_predictions_path),
            "loaded_model_path": None,
        }

    model_artifact, resolved_model_path, output_dirs = load_model_artifact(
        project_config,
        scheme,
        model_family,
        base_run_name,
        model_path=model_path,
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
        }

    resolved_predictions_path, output_dirs = resolve_predictions_path(
        project_config=project_config,
        scheme=scheme,
        model_family=model_family,
        base_run_name=base_run_name,
        predictions_path=predictions_path,
    )
    return load_saved_results_df(resolved_predictions_path), output_dirs, {
        "results_source": "saved_predictions",
        "loaded_predictions_path": str(resolved_predictions_path),
        "loaded_model_path": None,
    }


def load_test_split_reference(project_config, scheme, model_family, base_run_name, consumer_label="predict/evaluate"):
    from restructured_pipeline.src.train import ensure_output_dirs, load_and_preprocess_data

    import pandas as pd

    output_dirs = ensure_output_dirs(project_config, scheme, model_family)
    split_path = output_dirs["splits"] / f"test_indices_{base_run_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            "Saved split indices were not found for this run. "
            f"Run train.py with --save-files first so {consumer_label} can reproduce the test-set routing."
        )

    split_df = pd.read_csv(split_path)
    split_index_col = "source_row_index" if "source_row_index" in split_df.columns else "index"
    cat_df = load_and_preprocess_data(project_config, scheme)
    return cat_df.loc[split_df[split_index_col]].copy()


def _build_signature_series(df, signature_cols):
    return df[signature_cols].astype(str).agg("||".join, axis=1)


def select_inference_subset(
    results_df,
    project_config,
    scheme,
    model_family,
    inference_mode,
    base_run_name,
    consumer_label="predict/evaluate",
):
    if inference_mode == "inference_entire_data":
        return results_df.copy()

    reference_df = load_test_split_reference(
        project_config,
        scheme,
        model_family,
        base_run_name,
        consumer_label=consumer_label,
    )
    if "source_row_index" in results_df.columns and "source_row_index" in reference_df.columns:
        return results_df.loc[results_df["source_row_index"].isin(reference_df["source_row_index"])].copy()

    signature_cols = ["dataset"] + [
        col for col in project_config["features"]["feature_cols"] if col in results_df.columns
    ]
    reference_signature = set(_build_signature_series(reference_df, signature_cols))
    result_signature = _build_signature_series(results_df, signature_cols)
    return results_df.loc[result_signature.isin(reference_signature)].copy()
