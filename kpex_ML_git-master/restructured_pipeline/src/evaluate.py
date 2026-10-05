"""Evaluation entry point for the restructured pipeline.

"""

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

    if "correct_tol" in results_df.columns:
        results_df["correct_tol"] = _normalize_bool_series(results_df["correct_tol"])

    return results_df


def create_evaluation_run_name(base_run_name, inference_mode, file_suffix=None):
    if file_suffix:
        return file_suffix
    return f"{base_run_name}_eval_{output_mode_tag(inference_mode)}"


def _normalize_top_predictions(value):
    value = _parse_literal(value)
    if value is None:
        return []
    if isinstance(value, list):
        labels = []
        for item in value:
            if isinstance(item, tuple) and item:
                labels.append(item[0])
            else:
                labels.append(item)
        return labels
    return [value]


def has_probability_columns(results_df, algorithms):
    return all(f"prob_{alg}" in results_df.columns for alg in algorithms)


def compute_top_k_metrics(results_df, algorithms, tol=0.0):
    from restructured_pipeline.src.metrics import get_tol_fastest_algorithms

    ranked_predictions = []
    if "top3_predicted" in results_df.columns:
        ranked_predictions = results_df["top3_predicted"].apply(_normalize_top_predictions)
    elif has_probability_columns(results_df, algorithms):
        ranked_predictions = results_df.apply(
            lambda row: sorted(
                algorithms,
                key=lambda alg: row[f"prob_{alg}"],
                reverse=True,
            )[:3],
            axis=1,
        )
    else:
        ranked_predictions = results_df["predicted"].apply(lambda value: [value])

    top_k_metrics = {}
    for k in (1, 2, 3):
        top_k_metrics[f"top{k}_accuracy"] = ranked_predictions.apply(
            lambda preds, top_k=k: any(pred in preds[:top_k] for pred in _normalize_top_predictions(preds))
        ).mean()

    if "fastest_algorithms" in results_df.columns:
        for k in (1, 2, 3):
            top_k_metrics[f"top{k}_accuracy"] = ranked_predictions.combine(
                results_df["fastest_algorithms"],
                lambda preds, fastest, top_k=k: any(pred in (fastest or []) for pred in preds[:top_k]),
            ).mean()

    if tol >= 0:
        tol_fastest = results_df.apply(
            lambda row: get_tol_fastest_algorithms(row, algorithms, tol=tol),
            axis=1,
        )
        for k in (1, 2, 3):
            top_k_metrics[f"top{k}_accuracy_with_tolerance"] = ranked_predictions.combine(
                tol_fastest,
                lambda preds, fastest, top_k=k: any(pred in (fastest or []) for pred in preds[:top_k]),
            ).mean()

    return top_k_metrics


def evaluate_results(results_df, algorithms, tol=0.0, epsilon=1e-3):
    from restructured_pipeline.src.metrics import (
        evaluate_algorithm_performance,
        evaluate_time_metrics,
        evaluate_time_metrics_with_tol,
        evaluate_with_tolerance,
    )

    classification_metrics = evaluate_algorithm_performance(results_df, algorithms)
    time_metrics = evaluate_time_metrics(results_df, algorithms, epsilon=epsilon)

    if has_probability_columns(results_df, algorithms):
        classification_metrics_with_tolerance = evaluate_with_tolerance(results_df, algorithms, tol=tol)
    else:
        classification_metrics_with_tolerance = None

    time_metrics_with_tolerance = evaluate_time_metrics_with_tol(
        results_df,
        algorithms,
        tol=tol,
        epsilon=epsilon,
    )
    top_k_metrics = compute_top_k_metrics(results_df, algorithms, tol=tol)

    return (
        classification_metrics,
        time_metrics,
        classification_metrics_with_tolerance,
        time_metrics_with_tolerance,
        top_k_metrics,
    )


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

    if scheme == "scheme4":
        if scheme4_family not in {"oriented", "forward"}:
            raise ValueError("Scheme 4 requires scheme4_family='oriented' or 'forward'.")
        family_cfg = model_config["scheme4"][scheme4_family][model_family]["variants"]
        if variant is None:
            variant = next(iter(family_cfg))
        if variant not in family_cfg:
            raise ValueError(f"Unknown Scheme 4 variant {variant!r} for {scheme4_family}/{model_family}.")
        split_mode = family_cfg[variant]["split_mode"]
        return create_run_name(
            scheme, model_family, split_mode, variant=variant, tol=tol,
            subfamily=scheme4_family,
        )

    if scheme == "scheme5":
        if scheme5_family not in {"b", "other"}:
            raise ValueError("Scheme 5 requires scheme5_family='b' or 'other'.")
        family_cfg = model_config["scheme5"][scheme5_family][model_family]["variants"]
        if variant is None:
            variant = next(iter(family_cfg))
        if variant not in family_cfg:
            raise ValueError(f"Unknown Scheme 5 variant {variant!r} for {scheme5_family}/{model_family}.")
        split_mode = family_cfg[variant]["split_mode"]
        return f"scheme5_{scheme5_family}_{model_family}_{variant}"

    split_mode = split_mode or project_config["split_rules"]["defaults"]["stratify_on_target_only"]
    return create_run_name(
        scheme,
        model_family,
        split_mode,
        variant=None,
        tol=tol,
    )


def resolve_predictions_path(project_config, scheme, model_family, base_run_name, predictions_path=None, scheme4_family=None, scheme5_family=None, experiment_subdir=None):
    from restructured_pipeline.src.train import ensure_output_dirs

    output_dirs = ensure_output_dirs(project_config, scheme, model_family, subfamily=(scheme4_family if scheme == "scheme4" else scheme5_family if scheme == "scheme5" else None), experiment_subdir=experiment_subdir)
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


def load_results_for_evaluation(
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
            experiment_subdir=experiment_subdir,
        )
        return load_saved_results_df(resolved_predictions_path), output_dirs, {
            "results_source": "saved_predictions",
            "loaded_predictions_path": str(resolved_predictions_path),
            "loaded_model_path": None,
        }

    from restructured_pipeline.src.train import ensure_output_dirs
    output_dirs = ensure_output_dirs(project_config, scheme, model_family, subfamily=(scheme4_family if scheme == "scheme4" else scheme5_family if scheme == "scheme5" else None), experiment_subdir=experiment_subdir)
    resolved_input_model_path = model_path
    if resolved_input_model_path is None and (scheme in {"scheme4", "scheme5"} or experiment_subdir):
        resolved_input_model_path = output_dirs["models"] / f"model_{base_run_name}.pkl"
    model_artifact, resolved_model_path, _ = load_model_artifact(
        project_config, scheme, model_family, base_run_name,
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
    return load_saved_results_df(resolved_predictions_path), output_dirs, {
        "results_source": "saved_predictions",
        "loaded_predictions_path": str(resolved_predictions_path),
        "loaded_model_path": None,
    }


def load_test_split_reference(project_config, scheme, model_family, base_run_name, scheme4_family=None, scheme5_family=None, experiment_subdir=None):
    from restructured_pipeline.src.train import ensure_output_dirs, load_and_preprocess_data

    import pandas as pd

    output_dirs = ensure_output_dirs(project_config, scheme, model_family, subfamily=(scheme4_family if scheme == "scheme4" else scheme5_family if scheme == "scheme5" else None), experiment_subdir=experiment_subdir)
    split_path = output_dirs["splits"] / f"test_indices_{base_run_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            "Saved split indices were not found for this run. "
            "Run train.py with --save-files first so predict/evaluate can reproduce the test-set routing."
        )

    split_df = pd.read_csv(split_path)
    split_index_col = "source_row_index" if "source_row_index" in split_df.columns else "index"
    cat_df = load_and_preprocess_data(project_config, scheme)
    return cat_df.loc[split_df[split_index_col]].copy()


def _build_signature_series(df, signature_cols):
    return df[signature_cols].astype(str).agg("||".join, axis=1)


def select_results_subset(results_df, project_config, scheme, model_family, inference_mode, base_run_name, scheme4_family=None, scheme5_family=None, experiment_subdir=None):
    if inference_mode == "inference_entire_data":
        return results_df.copy()

    reference_df = load_test_split_reference(project_config, scheme, model_family, base_run_name, scheme4_family=scheme4_family, scheme5_family=scheme5_family, experiment_subdir=experiment_subdir)
    if "source_row_index" in results_df.columns and "source_row_index" in reference_df.columns:
        return results_df.loc[results_df["source_row_index"].isin(reference_df["source_row_index"])].copy()

    signature_cols = ["dataset"] + [
        col for col in project_config["features"]["feature_cols"] if col in results_df.columns
    ]
    reference_signature = set(_build_signature_series(reference_df, signature_cols))
    result_signature = _build_signature_series(results_df, signature_cols)
    return results_df.loc[result_signature.isin(reference_signature)].copy()


def save_final_reports(results_df, algorithms, output_dirs, run_name):
    from restructured_pipeline.src.features import get_fastest_algorithm
    from restructured_pipeline.src.train import build_confusion_labels, save_confusion_matrix

    y_true = results_df.apply(lambda row: get_fastest_algorithm(row, algorithms), axis=1)
    y_pred = results_df["predicted"]
    labels = build_confusion_labels(y_true, y_pred, algorithms)

    confusion_path = output_dirs["metrics"] / f"confusion_matrix_{run_name}.png"
    save_confusion_matrix(y_true, y_pred, labels, confusion_path, f"Confusion Matrix: {run_name}")

    return {"confusion_matrix_path": str(confusion_path)}


def run_evaluation(
    scheme,
    model_family,
    variant=None,
    split_mode=None,
    inference_mode="inference_test_set",
    tol=None,
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
        ensure_output_dirs,
        load_yaml_config,
        normalize_model_family,
        normalize_scheme,
        save_metrics_outputs,
    )

    scheme = normalize_scheme(scheme)
    model_family = normalize_model_family(model_family)
    inference_mode = normalize_inference_mode(inference_mode)
    if scheme == "scheme4" and scheme4_family not in {"oriented", "forward"}:
        raise ValueError("Scheme 4 requires --scheme4-family oriented or forward.")
    if scheme == "scheme5" and scheme5_family not in {"b", "other"}:
        raise ValueError("Scheme 5 requires --scheme5-family b or other.")

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

    results_df, output_dirs, load_metadata = load_results_for_evaluation(
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
    # Recover the actual training split mode when evaluating a saved run.
    # --run-name identifies the artifact but does not populate split_mode.
    if split_mode is None and load_metadata.get("loaded_model_path"):
        from restructured_pipeline.src.train import load_model_artifact

        reload_model_path = model_path
        if reload_model_path is None and (scheme in {"scheme4", "scheme5"} or experiment_subdir):
            reload_model_path = output_dirs["models"] / f"model_{base_run_name}.pkl"
        model_artifact, _, _ = load_model_artifact(
            project_config, scheme, model_family, base_run_name, model_path=reload_model_path
        )

        if model_artifact is not None:
            split_mode = model_artifact.get("split_mode")

        if split_mode not in {"strat_y", "strat_y_K"}:
            raise ValueError(
                f"Could not resolve split_mode for saved run '{base_run_name}'. "
                f"Resolved value: {split_mode!r}"
            )

    results_subset = select_results_subset(
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

    # Resolve the active classifier vocabulary. Scheme 4 uses the 12 prob_* columns/artifact vocabulary.
    if scheme in {"scheme4", "scheme5"}:
        algorithms = [col[len("prob_"):] for col in results_subset.columns if col.startswith("prob_")]
        expected = 12 if scheme == "scheme4" else (6 if scheme5_family == "b" else 18)
        if len(algorithms) != expected:
            raise ValueError(f"Expected {expected} {scheme} probability columns, found {len(algorithms)}.")
    else:
        algorithms = project_config["targets"]["scheme1"]["algorithms"]
    epsilon = project_config["evaluation"]["epsilon"]
    (
        classification_metrics,
        time_metrics,
        classification_metrics_with_tolerance,
        time_metrics_with_tolerance,
        top_k_metrics,
    ) = evaluate_results(results_subset, algorithms, tol=tol, epsilon=epsilon)

    evaluation_run_name = create_evaluation_run_name(
        base_run_name=base_run_name,
        inference_mode=inference_mode,
        file_suffix=file_suffix,
    )

    artifacts = {}
    if save_files:
        artifacts.update(
            save_metrics_outputs(
                classification_metrics,
                time_metrics,
                output_dirs,
                project_config["output_locations"]["filenames"]["metrics_workbook"],
                evaluation_run_name,
                sheet_name=(
                    f"{split_mode}_test_data"
                    if inference_mode == "inference_test_set"
                    else f"{split_mode}_entire_data"
                ),
            )
        )
        artifacts.update(save_final_reports(results_subset, algorithms, output_dirs, evaluation_run_name))

    return {
        "scheme": scheme,
        "model_family": model_family,
        "scheme4_family": scheme4_family,
        "scheme5_family": scheme5_family,
        "variant": variant,
        "split_mode": split_mode,
        "base_run_name": base_run_name,
        "evaluation_run_name": evaluation_run_name,
        "inference_mode": inference_mode,
        "classification_metrics": classification_metrics,
        "time_metrics": time_metrics,
        "classification_metrics_with_tolerance": classification_metrics_with_tolerance,
        "time_metrics_with_tolerance": time_metrics_with_tolerance,
        "top_k_metrics": top_k_metrics,
        "results_df": results_subset,
        "results_source": load_metadata["results_source"],
        "loaded_predictions_path": load_metadata["loaded_predictions_path"],
        "loaded_model_path": load_metadata["loaded_model_path"],
        "artifacts": artifacts,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate saved pipeline outputs on the test split or the entire dataset.")
    parser.add_argument("--scheme", required=True, choices=["scheme1", "scheme2", "scheme3", "scheme4", "scheme5"])
    parser.add_argument("--model-family", required=True, choices=["rf", "random_forest", "xgb", "xgboost"])
    parser.add_argument("--scheme4-family", choices=["oriented", "forward"], default=None)
    parser.add_argument("--scheme5-family", choices=["b", "other"], default=None)
    parser.add_argument("--experiment-subdir", default=None, help="Optional output subdirectory below scheme/model, e.g. a Scheme 1 weighting scheme.")
    parser.add_argument("--variant", default=None, help="Scheme 1 variant name, e.g. rf_strat_y or xgb_strat_y.")
    parser.add_argument("--split-mode", default=None, help="Optional split mode override, e.g. strat_y or strat_y_K.")
    parser.add_argument(
        "--inference-mode",
        default="inference_test_set",
        choices=["test", "test_set", "inference_test_set", "entire", "entire_data", "full", "inference_entire_data"],
        help="Evaluate on the held-out test set or the entire dataset.",
    )
    parser.add_argument("--tol", type=float, default=None, help="Tolerance for evaluation metrics.")
    parser.add_argument("--run-name", default=None, help="Existing saved training run name to evaluate.")
    parser.add_argument("--predictions-path", default=None, help="Optional explicit all_predictions CSV path.")
    parser.add_argument("--model-path", default=None, help="Optional explicit saved model artifact path.")
    parser.add_argument("--file-suffix", default=None, help="Optional explicit output suffix.")
    parser.add_argument("--project-config", default=str(DEFAULT_PROJECT_CONFIG))
    parser.add_argument("--model-config", default=None, help="Optional explicit model config path.")
    parser.add_argument(
        "--no-save-files",
        action="store_true",
        help="Skip writing reports and metrics files; still prints the evaluation summary.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    result = run_evaluation(
        scheme=args.scheme,
        model_family=args.model_family,
        variant=args.variant,
        split_mode=args.split_mode,
        inference_mode=args.inference_mode,
        tol=args.tol,
        save_files=not args.no_save_files,
        file_suffix=args.file_suffix,
        run_name=args.run_name,
        predictions_path=args.predictions_path,
        model_path=args.model_path,
        project_config_path=args.project_config,
        model_config_path=args.model_config,
        scheme4_family=args.scheme4_family,
        scheme5_family=args.scheme5_family,
        experiment_subdir=args.experiment_subdir,
    )

    print(f"Evaluation complete: {result['evaluation_run_name']}")
    print(f"Base run: {result['base_run_name']}")
    print(f"Rows evaluated: {len(result['results_df'])}")
    print(f"Accuracy: {result['classification_metrics']['accuracy']:.4f}")
    print(f"Time ratio: {result['time_metrics']['time_ratio']:.4f}")
    if result["classification_metrics_with_tolerance"] is not None:
        print(
            "Tolerance accuracy: "
            f"{result['classification_metrics_with_tolerance']['accuracy']:.4f}"
        )
    print(f"Top-3 accuracy: {result['top_k_metrics']['top3_accuracy']:.4f}")
    if not args.no_save_files:
        print("Artifacts:")
        for key, value in sorted(result["artifacts"].items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
