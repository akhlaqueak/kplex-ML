"""Evaluation entry point for the restructured pipeline."""

from __future__ import annotations

import argparse
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
    normalize_inference_mode,
    output_mode_tag,
    resolve_base_run_name,
    resolve_predictions_path,
    select_inference_subset,
)


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


def load_results_for_evaluation(
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


def select_results_subset(results_df, project_config, scheme, model_family, inference_mode, base_run_name):
    return select_inference_subset(
        results_df,
        project_config=project_config,
        scheme=scheme,
        model_family=model_family,
        inference_mode=inference_mode,
        base_run_name=base_run_name,
        consumer_label="predict/evaluate",
    )


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

    results_df, output_dirs, load_metadata = load_results_for_evaluation(
        project_config=project_config,
        scheme=scheme,
        model_family=model_family,
        base_run_name=base_run_name,
        tol=tol,
        predictions_path=predictions_path,
        model_path=model_path,
    )
    results_subset = select_results_subset(
        results_df,
        project_config=project_config,
        scheme=scheme,
        model_family=model_family,
        inference_mode=inference_mode,
        base_run_name=base_run_name,
    )

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
            )
        )
        artifacts.update(save_final_reports(results_subset, algorithms, output_dirs, evaluation_run_name))

    return {
        "scheme": scheme,
        "model_family": model_family,
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
    parser.add_argument("--scheme", required=True, choices=["scheme1", "scheme2", "scheme3"])
    parser.add_argument("--model-family", required=True, choices=["rf", "random_forest", "xgb", "xgboost"])
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
