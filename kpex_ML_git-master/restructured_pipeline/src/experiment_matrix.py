"""Run the full scheme/model/split/inference experiment matrix.
Keeping mine- No changes
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from restructured_pipeline.src.evaluate import normalize_inference_mode, run_evaluation
from restructured_pipeline.src.predict import run_prediction
from restructured_pipeline.src.tolerance_reports import write_tolerance_workbook
from restructured_pipeline.src.train import (
    DEFAULT_PROJECT_CONFIG,
    ensure_output_dirs,
    load_yaml_config,
    normalize_model_family,
    normalize_scheme,
    run_training,
)
from restructured_pipeline.src.write_utils import write_metrics_to_excel_file


DEFAULT_SCHEMES = ["scheme1", "scheme2", "scheme3"]
DEFAULT_MODEL_FAMILIES = ["random_forest", "xgboost"]
DEFAULT_SPLIT_MODES = ["strat_y", "strat_y_K"]
SUMMARY_SHEET_NAME = "model_performance_for_all_exp"
TOLERANCE_SUMMARY_SHEET_NAME = "tolerance_eval_all_exp"
SPLIT_MODE_ORDER = {"strat_y": 0, "strat_y_K": 1}
INFERENCE_MODE_ORDER = {"inference_test_set": 0, "inference_entire_data": 1}


def resolve_scheme1_variant(model_family, split_mode):
    mapping = {
        ("random_forest", "strat_y"): "rf_strat_y",
        ("random_forest", "strat_y_K"): "rf_strat_y_K",
        ("xgboost", "strat_y"): "xgb_strat_y",
        ("xgboost", "strat_y_K"): "xgb_strat_y_K",
    }
    key = (model_family, split_mode)
    if key not in mapping:
        raise ValueError(f"Unsupported scheme1 combination: {model_family} + {split_mode}")
    return mapping[key]


def build_experiment_sheet_name(split_mode, inference_mode):
    split_tag = "strat_y__K" if split_mode == "strat_y_K" else "strat_y"
    inference_tag = {
        "inference_test_set": "test_data",
        "inference_entire_data": "entire_data",
    }[normalize_inference_mode(inference_mode)]
    return f"{split_tag}_{inference_tag}"


def build_base_run_name(scheme, model_family, split_mode, matrix_run_name):
    return f"{scheme}_{model_family}_{split_mode}_{matrix_run_name}"


def order_experiment_records(experiment_records):
    return sorted(
        experiment_records,
        key=lambda record: (
            SPLIT_MODE_ORDER.get(record["split_mode"], 99),
            INFERENCE_MODE_ORDER.get(record["inference_mode"], 99),
        ),
    )


def resolve_matrix_workbook_path(project_config, scheme, model_family, matrix_run_name):
    output_dirs = ensure_output_dirs(project_config, scheme, model_family)
    base_name = Path(project_config["output_locations"]["filenames"]["metrics_workbook"]).stem
    workbook_path = output_dirs["metrics"] / f"{base_name}_{matrix_run_name}.xlsx"
    tolerance_path = output_dirs["metrics"] / f"{base_name}_tolerance_{matrix_run_name}.xlsx"
    return output_dirs, workbook_path, tolerance_path


def build_summary_row(experiment_record):
    evaluation_result = experiment_record["evaluation_result"]
    prediction_result = experiment_record["prediction_result"]
    classification_metrics = evaluation_result["classification_metrics"]
    time_metrics = evaluation_result["time_metrics"]
    tolerance_metrics = evaluation_result["classification_metrics_with_tolerance"] or {}
    tolerance_time_metrics = evaluation_result["time_metrics_with_tolerance"] or {}
    top_k_metrics = evaluation_result["top_k_metrics"]

    return {
        "experiment_sheet": experiment_record["sheet_name"],
        "scheme": experiment_record["scheme"],
        "model_family": experiment_record["model_family"],
        "split_mode": experiment_record["split_mode"],
        "inference_mode": experiment_record["inference_mode"],
        "base_run_name": experiment_record["base_run_name"],
        "evaluation_run_name": evaluation_result["evaluation_run_name"],
        "prediction_run_name": prediction_result["prediction_run_name"],
        "row_count": len(evaluation_result["results_df"]),
        "accuracy": classification_metrics["accuracy"],
        "tolerance_accuracy": tolerance_metrics.get("accuracy"),
        "time_ratio": time_metrics["time_ratio"],
        "tolerance_time_ratio": tolerance_time_metrics.get("time_ratio"),
        "avg_overhead_per_case": time_metrics["avg_overhead_per_case"],
        "tolerance_avg_overhead_per_case": tolerance_time_metrics.get("avg_overhead_per_case"),
        "top1_accuracy": top_k_metrics["top1_accuracy"],
        "top2_accuracy": top_k_metrics["top2_accuracy"],
        "top3_accuracy": top_k_metrics["top3_accuracy"],
        "top1_accuracy_with_tolerance": top_k_metrics.get("top1_accuracy_with_tolerance"),
        "top2_accuracy_with_tolerance": top_k_metrics.get("top2_accuracy_with_tolerance"),
        "top3_accuracy_with_tolerance": top_k_metrics.get("top3_accuracy_with_tolerance"),
        "model_path": experiment_record["train_result"]["artifacts"].get("model_artifact_path"),
        "stage_summary_report_path": experiment_record["train_result"]["artifacts"].get("stage_summary_report_path"),
        "stage_per_class_report_path": experiment_record["train_result"]["artifacts"].get("stage_per_class_report_path"),
        "confusion_matrix_path": evaluation_result["artifacts"].get("confusion_matrix_path"),
        "prediction_csv_path": prediction_result["artifacts"].get("predictions_path"),
        "prediction_wrong_cases_xlsx_path": prediction_result["artifacts"].get("wrong_cases_xlsx_path"),
    }


def write_performance_workbook(experiment_records, workbook_path):
    workbook_path = Path(workbook_path)
    if workbook_path.exists():
        workbook_path.unlink()

    ordered_records = order_experiment_records(experiment_records)
    summary_df = pd.DataFrame([build_summary_row(record) for record in ordered_records])
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name=SUMMARY_SHEET_NAME, index=False)

    for record in ordered_records:
        evaluation_result = record["evaluation_result"]
        write_metrics_to_excel_file(
            evaluation_result["classification_metrics"],
            evaluation_result["time_metrics"],
            str(workbook_path),
            sheet_name=record["sheet_name"],
        )

    return str(workbook_path)


def strip_experiment_record_for_manifest(experiment_record):
    evaluation_result = experiment_record["evaluation_result"]
    prediction_result = experiment_record["prediction_result"]

    return {
        "sheet_name": experiment_record["sheet_name"],
        "scheme": experiment_record["scheme"],
        "model_family": experiment_record["model_family"],
        "split_mode": experiment_record["split_mode"],
        "inference_mode": experiment_record["inference_mode"],
        "base_run_name": experiment_record["base_run_name"],
        "evaluation_run_name": evaluation_result["evaluation_run_name"],
        "prediction_run_name": prediction_result["prediction_run_name"],
        "evaluation_artifacts": evaluation_result["artifacts"],
        "prediction_artifacts": prediction_result["artifacts"],
    }


def run_experiment_matrix(
    schemes=None,
    model_families=None,
    split_modes=None,
    tol=None,
    matrix_run_name=None,
    project_config_path=DEFAULT_PROJECT_CONFIG,
    model_config_scheme1=None,
    model_config_scheme2=None,
    model_config_scheme3=None,
):
    project_config = load_yaml_config(project_config_path)
    schemes = [normalize_scheme(value) for value in (schemes or DEFAULT_SCHEMES)]
    model_families = [normalize_model_family(value) for value in (model_families or DEFAULT_MODEL_FAMILIES)]
    split_modes = split_modes or DEFAULT_SPLIT_MODES
    tol = project_config["tolerances"]["tie_tolerance"] if tol is None else tol
    matrix_run_name = matrix_run_name or datetime.now().strftime("matrix_%Y%m%d_%H%M%S")
    tolerance_values = project_config["evaluation"]["tolerance_values"]
    epsilon = project_config["evaluation"]["epsilon"]
    algorithms = project_config["targets"]["scheme1"]["algorithms"]

    model_configs = {
        "scheme1": model_config_scheme1,
        "scheme2": model_config_scheme2,
        "scheme3": model_config_scheme3,
    }

    records_by_group = {}
    for scheme in schemes:
        for model_family in model_families:
            group_key = (scheme, model_family)
            records_by_group[group_key] = []

            for split_mode in split_modes:
                variant = resolve_scheme1_variant(model_family, split_mode) if scheme == "scheme1" else None
                base_run_name = build_base_run_name(scheme, model_family, split_mode, matrix_run_name)
                model_config_path = model_configs[scheme]

                train_result = run_training(
                    scheme=scheme,
                    model_family=model_family,
                    variant=variant,
                    split_mode=split_mode,
                    tol=tol,
                    save_files=True,
                    file_suffix=base_run_name,
                    project_config_path=project_config_path,
                    model_config_path=model_config_path,
                )

                for inference_mode in ("inference_test_set", "inference_entire_data"):
                    evaluation_result = run_evaluation(
                        scheme=scheme,
                        model_family=model_family,
                        variant=variant,
                        split_mode=split_mode,
                        inference_mode=inference_mode,
                        tol=tol,
                        save_files=True,
                        run_name=base_run_name,
                        project_config_path=project_config_path,
                        model_config_path=model_config_path,
                    )
                    prediction_result = run_prediction(
                        scheme=scheme,
                        model_family=model_family,
                        variant=variant,
                        split_mode=split_mode,
                        inference_mode=inference_mode,
                        tol=tol,
                        save_files=True,
                        run_name=base_run_name,
                        project_config_path=project_config_path,
                        model_config_path=model_config_path,
                    )

                    records_by_group[group_key].append(
                        {
                            "sheet_name": build_experiment_sheet_name(split_mode, inference_mode),
                            "scheme": scheme,
                            "model_family": model_family,
                            "split_mode": split_mode,
                            "inference_mode": inference_mode,
                            "base_run_name": base_run_name,
                            "train_result": train_result,
                            "evaluation_result": evaluation_result,
                            "prediction_result": prediction_result,
                            "results_df": prediction_result["results_df"],
                        }
                    )

    manifest = {
        "matrix_run_name": matrix_run_name,
        "schemes": schemes,
        "model_families": model_families,
        "split_modes": split_modes,
        "tol": tol,
        "tolerance_values": tolerance_values,
        "groups": {},
    }

    for (scheme, model_family), experiment_records in records_by_group.items():
        ordered_records = order_experiment_records(experiment_records)
        output_dirs, workbook_path, tolerance_path = resolve_matrix_workbook_path(
            project_config,
            scheme,
            model_family,
            matrix_run_name,
        )
        performance_workbook_path = write_performance_workbook(ordered_records, workbook_path)
        tolerance_workbook_path = write_tolerance_workbook(
            ordered_records,
            tolerance_values=tolerance_values,
            output_path=tolerance_path,
            algorithms=algorithms,
            epsilon=epsilon,
            summary_sheet_name=TOLERANCE_SUMMARY_SHEET_NAME,
        )

        manifest["groups"][f"{scheme}:{model_family}"] = {
            "performance_workbook_path": performance_workbook_path,
            "tolerance_workbook_path": tolerance_workbook_path,
            "experiments": [strip_experiment_record_for_manifest(record) for record in ordered_records],
        }

    matrix_log_dir = Path(project_config["paths"]["pipeline"]["logs_dir"]) / "experiment_matrix"
    matrix_log_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = matrix_log_dir / f"experiment_matrix_{matrix_run_name}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run the full scheme/model/split/inference experiment matrix.")
    parser.add_argument("--schemes", nargs="*", default=None, choices=DEFAULT_SCHEMES)
    parser.add_argument("--model-families", nargs="*", default=None, choices=["rf", "random_forest", "xgb", "xgboost"])
    parser.add_argument("--split-modes", nargs="*", default=None, choices=DEFAULT_SPLIT_MODES)
    parser.add_argument("--tol", type=float, default=None, help="Base tie tolerance for training run naming.")
    parser.add_argument("--matrix-run-name", default=None, help="Shared suffix used for all runs in the matrix.")
    parser.add_argument("--project-config", default=str(DEFAULT_PROJECT_CONFIG))
    parser.add_argument("--model-config-scheme1", default=None)
    parser.add_argument("--model-config-scheme2", default=None)
    parser.add_argument("--model-config-scheme3", default=None)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    result = run_experiment_matrix(
        schemes=args.schemes,
        model_families=args.model_families,
        split_modes=args.split_modes,
        tol=args.tol,
        matrix_run_name=args.matrix_run_name,
        project_config_path=args.project_config,
        model_config_scheme1=args.model_config_scheme1,
        model_config_scheme2=args.model_config_scheme2,
        model_config_scheme3=args.model_config_scheme3,
    )

    print(f"Experiment matrix complete: {result['matrix_run_name']}")
    print(f"Manifest: {result['manifest_path']}")
    for group_name, payload in sorted(result["groups"].items()):
        print(group_name)
        print(f"  performance_workbook_path: {payload['performance_workbook_path']}")
        print(f"  tolerance_workbook_path: {payload['tolerance_workbook_path']}")


if __name__ == "__main__":
    main()
