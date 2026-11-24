"""Single training entry point for the restructured pipeline."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from restructured_pipeline.src.features import (
    W_TIME_RATIO,
    algorithms as legacy_algorithms,
    assign_scheme2_targets_and_weights,
    assign_scheme3_targets_and_weights,
    assign_weights,
    feature_cols as legacy_feature_cols,
    get_fastest_algorithm,
    load_excel_data,
    post_process_data,
    process_kcore_list_to_statistics,
)
from restructured_pipeline.src.metrics import (
    evaluate_algorithm_performance,
    evaluate_time_metrics,
    evaluate_time_metrics_with_tol,
    evaluate_with_tolerance,
)
from restructured_pipeline.src.models.flat import fit_random_forest_classifier, fit_xgb_classifier
from restructured_pipeline.src.topk_utils import get_top3
from restructured_pipeline.src.models.hierarchical import (
    evaluate_hierarchical_stages,
    fit_hierarchical_random_forest_classifier,
    fit_hierarchical_xgb_classifier,
    run_full_evaluation_hier5,
    save_stage_reports,
)
from restructured_pipeline.src.models.hierarchical_scheme3 import (
    evaluate_hierarchical_scheme3_stages,
    fit_hierarchical_scheme3_random_forest_classifier,
    fit_hierarchical_scheme3_xgb_classifier,
    run_full_evaluation_scheme3,
)
from restructured_pipeline.src.write_utils import write_metrics_to_excel_file


DEFAULT_PROJECT_CONFIG = Path("restructured_pipeline/configs/project.yaml")
DEFAULT_SCHEME1_CONFIG = Path("restructured_pipeline/configs/models_scheme1.yaml")
DEFAULT_SCHEME2_CONFIG = Path("restructured_pipeline/configs/models_scheme2.yaml")
DEFAULT_SCHEME3_CONFIG = Path("restructured_pipeline/configs/models_scheme3.yaml")


def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_scheme(scheme):
    mapping = {"scheme1": "scheme1", "scheme2": "scheme2", "scheme3": "scheme3"}
    if scheme not in mapping:
        raise ValueError(f"Unsupported scheme: {scheme}")
    return mapping[scheme]


def normalize_model_family(model_family):
    mapping = {
        "rf": "random_forest",
        "random_forest": "random_forest",
        "xgb": "xgboost",
        "xgboost": "xgboost",
    }
    if model_family not in mapping:
        raise ValueError(f"Unsupported model_family: {model_family}")
    return mapping[model_family]


def resolve_model_config_path(scheme, model_config_path=None):
    if model_config_path is not None:
        return Path(model_config_path)
    if scheme == "scheme1":
        return DEFAULT_SCHEME1_CONFIG
    if scheme == "scheme2":
        return DEFAULT_SCHEME2_CONFIG
    return DEFAULT_SCHEME3_CONFIG


def ensure_output_dirs(project_config, scheme, model_family):
    pipeline_paths = project_config["paths"]["pipeline"]
    dirs = {
        "models": Path(pipeline_paths["models_dir"]) / scheme / model_family,
        "predictions": Path(pipeline_paths["predictions_dir"]) / scheme / model_family,
        "wrong_cases": Path(pipeline_paths["wrong_cases_dir"]) / scheme / model_family,
        "metrics": Path(pipeline_paths["metrics_dir"]) / scheme / model_family,
        "logs": Path(pipeline_paths["logs_dir"]) / scheme / model_family,
        "splits": Path(pipeline_paths["splits_data_dir"]) / scheme / model_family,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def create_run_name(scheme, model_family, split_mode, variant=None, tol=0.0, file_suffix=None):
    if file_suffix:
        return file_suffix

    base = f"{scheme}_{model_family}"
    if variant:
        base = f"{base}_{variant}"
    else:
        base = f"{base}_{split_mode}"

    if tol > 0:
        base = f"{base}_tol_{int(round(tol * 100))}"
    return base


def load_and_preprocess_data(project_config, scheme):
    data_cfg = project_config["data"]
    eval_cfg = project_config["evaluation"]
    file_path = data_cfg["file_path"]

    cat_df = load_excel_data(
        file_path,
        timeout_replacement=data_cfg["timeout_replacement"],
        y_cols=data_cfg["raw_target_columns"],
        sep=data_cfg["sep"],
        quotechar=data_cfg["quotechar"],
        engine=data_cfg["engine"],
        rename_columns=data_cfg["rename_columns"],
    )
    assign_weights(cat_df, W_TIME_RATIO, eval_cfg["epsilon"])
    process_kcore_list_to_statistics(cat_df)
    cat_df = post_process_data(cat_df)
    cat_df["source_row_index"] = cat_df.index

    for col in legacy_feature_cols:
        if col not in cat_df.columns:
            continue
        numeric_col = pd.to_numeric(cat_df[col], errors="coerce")
        if not numeric_col.isna().any():
            continue
        fill_value = numeric_col.median()
        if pd.isna(fill_value):
            fill_value = 0.0
        cat_df[col] = numeric_col.fillna(fill_value)

    if scheme == "scheme2":
        cat_df = assign_scheme2_targets_and_weights(cat_df)
    elif scheme == "scheme3":
        cat_df = assign_scheme3_targets_and_weights(cat_df)

    return cat_df


def build_flat_results(cat_df, clf, features, algorithms, le=None, epsilon=1e-3, tol=0.0):
    results_df = cat_df.copy()
    results_df["fastest_algorithms"] = results_df.apply(
        lambda row: [alg for alg in algorithms if row[alg] == min(row[a] for a in algorithms)],
        axis=1,
    )

    X_full = results_df[features]
    y_pred = clf.predict(X_full)
    if le is not None:
        results_df["predicted"] = le.inverse_transform(y_pred)
        class_labels = list(le.classes_)
    else:
        results_df["predicted"] = y_pred
        class_labels = list(clf.classes_)

    results_df["correct"] = results_df.apply(
        lambda row: row["predicted"] in row["fastest_algorithms"],
        axis=1,
    )

    proba_cols = []
    if hasattr(clf, "predict_proba"):
        probas = clf.predict_proba(X_full)
        for i, cls in enumerate(class_labels):
            col = f"prob_{cls}"
            results_df[col] = probas[:, i]
            proba_cols.append(col)

    if tol > 0:
        cls_metrics = evaluate_with_tolerance(results_df, algorithms, tol=tol)
        time_metrics = evaluate_time_metrics_with_tol(results_df, algorithms, tol=tol, epsilon=epsilon)
    else:
        cls_metrics = evaluate_algorithm_performance(results_df, algorithms)
        time_metrics = evaluate_time_metrics(results_df, algorithms, epsilon=epsilon)

    if proba_cols:
        results_df["top3_predicted"] = results_df.apply(lambda row: get_top3(row, algorithms), axis=1)
    else:
        results_df["top3_predicted"] = None

    wrong_df = build_wrong_predictions(results_df, algorithms, epsilon=epsilon)
    return cls_metrics, time_metrics, results_df, wrong_df


def build_wrong_predictions(results_df, algorithms, epsilon=1e-3):
    wrong_df = results_df[~results_df["correct"]].copy()
    if wrong_df.empty:
        return wrong_df

    def calculate_time_metrics(row):
        algo_times = {alg: row[alg] for alg in algorithms}
        fastest_time = min(algo_times.values())
        fastest_algos = [alg for alg, t in algo_times.items() if t == fastest_time]
        predicted_time = algo_times[row["predicted"]]
        time_gap = predicted_time - fastest_time
        return pd.Series(
            {
                "fastest_time": fastest_time,
                "predicted_time": predicted_time,
                "time_gap": time_gap,
                "time_gap_ratio": time_gap / fastest_time if fastest_time > 0 else predicted_time / epsilon,
                "time_ratio": predicted_time / fastest_time if fastest_time > 0 else predicted_time / epsilon,
                "fastest_algorithms": fastest_algos,
            }
        )

    time_metrics = wrong_df.apply(calculate_time_metrics, axis=1)
    wrong_df = pd.concat([wrong_df, time_metrics], axis=1)
    return wrong_df.sort_values("weight", ascending=False)


def save_dataframe_outputs(
    results_df,
    wrong_df,
    output_dirs,
    filename_prefixes,
    run_name,
):
    predictions_path = output_dirs["predictions"] / f"{filename_prefixes['all_predictions_prefix']}_{run_name}.csv"
    wrong_cases_xlsx_path = output_dirs["wrong_cases"] / f"{filename_prefixes['wrong_predictions_prefix']}_{run_name}.xlsx"

    results_df.to_csv(predictions_path, index=False)
    wrong_df.to_excel(wrong_cases_xlsx_path, index=False)

    artifacts = {
        "predictions_path": str(predictions_path),
        "wrong_cases_xlsx_path": str(wrong_cases_xlsx_path),
    }

    return artifacts


def save_metrics_outputs(cls_metrics, time_metrics, output_dirs, metrics_workbook, run_name):
    metrics_workbook_path = output_dirs["metrics"] / metrics_workbook
    write_metrics_to_excel_file(
        cls_metrics,
        time_metrics,
        str(metrics_workbook_path),
        sheet_name=run_name[:31],
    )

    return {"metrics_workbook_path": str(metrics_workbook_path)}


def save_hierarchical_stage_outputs(stage_summary_df, per_class_reports, output_dirs, filename_prefixes, run_name):
    return {
        "stage_summary_report_path": save_stage_reports(
            stage_summary_df,
            f"{filename_prefixes['stage_summary_report_prefix']}_{run_name}",
            output_dir=str(output_dirs["metrics"]),
            to_excel=True,
        ),
        "stage_per_class_report_path": save_stage_reports(
            per_class_reports,
            f"{filename_prefixes['stage_per_class_report_prefix']}_{run_name}",
            output_dir=str(output_dirs["metrics"]),
            to_excel=True,
        ),
    }


def build_confusion_labels(y_true, y_pred, base_labels):
    labels = list(base_labels)
    extras = sorted((set(pd.Series(y_true).dropna()) | set(pd.Series(y_pred).dropna())) - set(labels))
    return labels + extras


def save_confusion_matrix(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_flat_reports(results_df, algorithms, output_dirs, run_name):
    y_true = results_df.apply(lambda row: get_fastest_algorithm(row, algorithms), axis=1)
    y_pred = results_df["predicted"]
    labels = build_confusion_labels(y_true, y_pred, algorithms)

    confusion_path = output_dirs["metrics"] / f"confusion_matrix_{run_name}.png"
    save_confusion_matrix(y_true, y_pred, labels, confusion_path, f"Scheme 1 Confusion Matrix: {run_name}")

    return {"confusion_matrix_path": str(confusion_path)}


def save_scheme2_confusion_matrices(results_df, clf, features, algorithms, output_dirs, run_name):
    X_full = results_df[features].values
    outputs = {}

    if clf.uses_label_encoding:
        stage1_pred = clf.le_group.inverse_transform(clf.group_clf.predict(X_full))
        save_confusion_matrix(
            results_df["y_group"],
            stage1_pred,
            build_confusion_labels(results_df["y_group"], stage1_pred, ["Sparse", "Dense"]),
            output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage1.png",
            f"Stage 1: {run_name}",
        )
        outputs["stage1_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage1.png")

        sparse_mask = results_df["y_group"] == "Sparse"
        if sparse_mask.any():
            stage2_pred = clf.le_sparse.inverse_transform(clf.sparse_subgroup_clf.predict(X_full[sparse_mask]))
            save_confusion_matrix(
                results_df.loc[sparse_mask, "y_sparse_subgroup"],
                stage2_pred,
                build_confusion_labels(results_df.loc[sparse_mask, "y_sparse_subgroup"], stage2_pred, ["AltRB", "No-AltRB"]),
                output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage2.png",
                f"Stage 2: {run_name}",
            )
            outputs["stage2_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage2.png")

        alt_mask = results_df["y_sparse_subgroup"] == "AltRB"
        if alt_mask.any():
            y_pred_local = clf.altRB_clf.predict(X_full[alt_mask])
            stage3_pred = clf.le_algo.inverse_transform([clf.alt_map[i] for i in y_pred_local])
            labels = build_confusion_labels(results_df.loc[alt_mask, "y_algo"], stage3_pred, sorted(results_df.loc[alt_mask, "y_algo"].unique()))
            save_confusion_matrix(
                results_df.loc[alt_mask, "y_algo"],
                stage3_pred,
                labels,
                output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage3.png",
                f"Stage 3: {run_name}",
            )
            outputs["stage3_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage3.png")

        noalt_mask = results_df["y_sparse_subgroup"] == "No-AltRB"
        if noalt_mask.any():
            y_pred_local = clf.noAltRB_clf.predict(X_full[noalt_mask])
            stage4_pred = clf.le_algo.inverse_transform([clf.noalt_map[i] for i in y_pred_local])
            labels = build_confusion_labels(results_df.loc[noalt_mask, "y_algo"], stage4_pred, sorted(results_df.loc[noalt_mask, "y_algo"].unique()))
            save_confusion_matrix(
                results_df.loc[noalt_mask, "y_algo"],
                stage4_pred,
                labels,
                output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage4.png",
                f"Stage 4: {run_name}",
            )
            outputs["stage4_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage4.png")

        dense_mask = results_df["y_group"] == "Dense"
        if dense_mask.any():
            y_pred_local = clf.dense_clf.predict(X_full[dense_mask])
            stage5_pred = clf.le_algo.inverse_transform([clf.dense_map[i] for i in y_pred_local])
            labels = build_confusion_labels(results_df.loc[dense_mask, "y_algo"], stage5_pred, sorted(results_df.loc[dense_mask, "y_algo"].unique()))
            save_confusion_matrix(
                results_df.loc[dense_mask, "y_algo"],
                stage5_pred,
                labels,
                output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage5.png",
                f"Stage 5: {run_name}",
            )
            outputs["stage5_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage5.png")
    else:
        stage1_pred = clf.group_clf.predict(X_full)
        save_confusion_matrix(
            results_df["y_group"],
            stage1_pred,
            build_confusion_labels(results_df["y_group"], stage1_pred, ["Sparse", "Dense"]),
            output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage1.png",
            f"Stage 1: {run_name}",
        )
        outputs["stage1_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage1.png")

        sparse_mask = results_df["y_group"] == "Sparse"
        if sparse_mask.any():
            stage2_pred = clf.sparse_subgroup_clf.predict(X_full[sparse_mask])
            save_confusion_matrix(
                results_df.loc[sparse_mask, "y_sparse_subgroup"],
                stage2_pred,
                build_confusion_labels(results_df.loc[sparse_mask, "y_sparse_subgroup"], stage2_pred, ["AltRB", "No-AltRB"]),
                output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage2.png",
                f"Stage 2: {run_name}",
            )
            outputs["stage2_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage2.png")

        alt_mask = results_df["y_sparse_subgroup"] == "AltRB"
        if alt_mask.any():
            stage3_pred = clf.altRB_clf.predict(X_full[alt_mask])
            labels = build_confusion_labels(results_df.loc[alt_mask, "y_algo"], stage3_pred, sorted(results_df.loc[alt_mask, "y_algo"].unique()))
            save_confusion_matrix(
                results_df.loc[alt_mask, "y_algo"],
                stage3_pred,
                labels,
                output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage3.png",
                f"Stage 3: {run_name}",
            )
            outputs["stage3_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage3.png")

        noalt_mask = results_df["y_sparse_subgroup"] == "No-AltRB"
        if noalt_mask.any():
            stage4_pred = clf.noAltRB_clf.predict(X_full[noalt_mask])
            labels = build_confusion_labels(results_df.loc[noalt_mask, "y_algo"], stage4_pred, sorted(results_df.loc[noalt_mask, "y_algo"].unique()))
            save_confusion_matrix(
                results_df.loc[noalt_mask, "y_algo"],
                stage4_pred,
                labels,
                output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage4.png",
                f"Stage 4: {run_name}",
            )
            outputs["stage4_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage4.png")

        dense_mask = results_df["y_group"] == "Dense"
        if dense_mask.any():
            stage5_pred = clf.dense_clf.predict(X_full[dense_mask])
            labels = build_confusion_labels(results_df.loc[dense_mask, "y_algo"], stage5_pred, sorted(results_df.loc[dense_mask, "y_algo"].unique()))
            save_confusion_matrix(
                results_df.loc[dense_mask, "y_algo"],
                stage5_pred,
                labels,
                output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage5.png",
                f"Stage 5: {run_name}",
            )
            outputs["stage5_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage5.png")

    final_pred = results_df["predicted"]
    final_labels = build_confusion_labels(results_df["y_algo"], final_pred, algorithms)
    final_path = output_dirs["metrics"] / f"confusion_matrix_{run_name}_final.png"
    save_confusion_matrix(results_df["y_algo"], final_pred, final_labels, final_path, f"Final Scheme 2 Confusion Matrix: {run_name}")
    outputs["final_confusion_matrix_path"] = str(final_path)
    return outputs


def save_scheme3_confusion_matrices(results_df, clf, features, algorithms, output_dirs, run_name):
    X_full = results_df[features].values
    outputs = {}
    has_orientation_stage = getattr(clf, "_has_orientation_stage", lambda: False)()

    if clf.uses_label_encoding:
        stage1_pred = clf.le_group.inverse_transform(clf.group_clf.predict(X_full))
        save_confusion_matrix(
            results_df["y_group"],
            stage1_pred,
            build_confusion_labels(results_df["y_group"], stage1_pred, ["Sparse", "Dense"]),
            output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage1.png",
            f"Stage 1: {run_name}",
        )
        outputs["stage1_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage1.png")

        sparse_mask = results_df["y_group"] == "Sparse"
        if sparse_mask.any():
            stage2_pred = clf.le_sparse.inverse_transform(clf.sparse_subgroup_clf.predict(X_full[sparse_mask]))
            save_confusion_matrix(
                results_df.loc[sparse_mask, "y_sparse_subgroup"],
                stage2_pred,
                build_confusion_labels(
                    results_df.loc[sparse_mask, "y_sparse_subgroup"],
                    stage2_pred,
                    ["AltRB", "No-AltRB"],
                ),
                output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage2.png",
                f"Stage 2: {run_name}",
            )
            outputs["stage2_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage2.png")

        if has_orientation_stage:
            stage3_pred = clf.le_orientation.inverse_transform(clf.orientation_clf.predict(X_full))
            save_confusion_matrix(
                results_df["y_orientation"],
                stage3_pred,
                build_confusion_labels(
                    results_df["y_orientation"],
                    stage3_pred,
                    ["Oriented", "NonOriented"],
                ),
                output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage3.png",
                f"Stage 3: {run_name}",
            )
            outputs["stage3_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage3.png")

        y_pred_local = clf.router4_clf.predict(X_full)
        stage3_pred = clf.le_router.inverse_transform([clf.router_map[i] for i in y_pred_local])
        router_stage_number = "stage4" if has_orientation_stage else "stage3"
        save_confusion_matrix(
            results_df["y_router4"],
            stage3_pred,
            build_confusion_labels(results_df["y_router4"], stage3_pred, ["ClassB", "ClassS", "ClassR", "ClassP"]),
            output_dirs["metrics"] / f"confusion_matrix_{run_name}_{router_stage_number}.png",
            f"Stage {'4' if has_orientation_stage else '3'}: {run_name}",
        )
        outputs["router_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_{router_stage_number}.png")
    else:
        stage1_pred = clf.group_clf.predict(X_full)
        save_confusion_matrix(
            results_df["y_group"],
            stage1_pred,
            build_confusion_labels(results_df["y_group"], stage1_pred, ["Sparse", "Dense"]),
            output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage1.png",
            f"Stage 1: {run_name}",
        )
        outputs["stage1_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage1.png")

        sparse_mask = results_df["y_group"] == "Sparse"
        if sparse_mask.any():
            stage2_pred = clf.sparse_subgroup_clf.predict(X_full[sparse_mask])
            save_confusion_matrix(
                results_df.loc[sparse_mask, "y_sparse_subgroup"],
                stage2_pred,
                build_confusion_labels(
                    results_df.loc[sparse_mask, "y_sparse_subgroup"],
                    stage2_pred,
                    ["AltRB", "No-AltRB"],
                ),
                output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage2.png",
                f"Stage 2: {run_name}",
            )
            outputs["stage2_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage2.png")

        if has_orientation_stage:
            stage3_pred = clf.orientation_clf.predict(X_full)
            save_confusion_matrix(
                results_df["y_orientation"],
                stage3_pred,
                build_confusion_labels(
                    results_df["y_orientation"],
                    stage3_pred,
                    ["Oriented", "NonOriented"],
                ),
                output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage3.png",
                f"Stage 3: {run_name}",
            )
            outputs["stage3_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_stage3.png")

        stage3_pred = clf.router4_clf.predict(X_full)
        router_stage_number = "stage4" if has_orientation_stage else "stage3"
        save_confusion_matrix(
            results_df["y_router4"],
            stage3_pred,
            build_confusion_labels(results_df["y_router4"], stage3_pred, ["ClassB", "ClassS", "ClassR", "ClassP"]),
            output_dirs["metrics"] / f"confusion_matrix_{run_name}_{router_stage_number}.png",
            f"Stage {'4' if has_orientation_stage else '3'}: {run_name}",
        )
        outputs["router_confusion_matrix_path"] = str(output_dirs["metrics"] / f"confusion_matrix_{run_name}_{router_stage_number}.png")

    final_pred = results_df["predicted"]
    final_labels = build_confusion_labels(results_df["y_algo"], final_pred, algorithms)
    final_path = output_dirs["metrics"] / f"confusion_matrix_{run_name}_final.png"
    save_confusion_matrix(results_df["y_algo"], final_pred, final_labels, final_path, f"Final Scheme 3 Confusion Matrix: {run_name}")
    outputs["final_confusion_matrix_path"] = str(final_path)
    return outputs


def save_split_indices(split_data, output_dirs, run_name):
    train_idx_path = output_dirs["splits"] / f"train_indices_{run_name}.csv"
    test_idx_path = output_dirs["splits"] / f"test_indices_{run_name}.csv"

    pd.DataFrame({"source_row_index": split_data["X_train"].index}).to_csv(train_idx_path, index=False)
    pd.DataFrame({"source_row_index": split_data["X_test"].index}).to_csv(test_idx_path, index=False)

    return {
        "train_indices_path": str(train_idx_path),
        "test_indices_path": str(test_idx_path),
    }


def save_model_artifact(output_dirs, run_name, clf, le, scheme, model_family, split_mode=None, variant=None, feature_cols=None, algorithms=None):
    model_path = output_dirs["models"] / f"model_{run_name}.pkl"
    payload = {
        "scheme": scheme,
        "model_family": model_family,
        "split_mode": split_mode,
        "variant": variant,
        "feature_cols": list(feature_cols or []),
        "algorithms": list(algorithms or []),
        "classifier": clf,
        "label_encoder": le,
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {"model_artifact_path": str(model_path)}


def resolve_model_artifact_path(project_config, scheme, model_family, base_run_name, model_path=None):
    output_dirs = ensure_output_dirs(project_config, scheme, model_family)
    if model_path is not None:
        return Path(model_path), output_dirs
    candidate = output_dirs["models"] / f"model_{base_run_name}.pkl"
    return candidate, output_dirs


def load_model_artifact(project_config, scheme, model_family, base_run_name, model_path=None):
    artifact_path, output_dirs = resolve_model_artifact_path(
        project_config,
        scheme,
        model_family,
        base_run_name,
        model_path=model_path,
    )
    if not artifact_path.exists():
        if model_path is not None:
            raise FileNotFoundError(f"Saved model artifact not found at {artifact_path}.")
        return None, artifact_path, output_dirs
    with open(artifact_path, "rb") as f:
        return pickle.load(f), artifact_path, output_dirs


def build_results_from_model_artifact(project_config, scheme, model_artifact, tol=0.0):
    cat_df = load_and_preprocess_data(project_config, scheme)
    clf = model_artifact["classifier"]
    le = model_artifact.get("label_encoder")
    features = model_artifact.get("feature_cols") or project_config["features"]["feature_cols"]
    algorithms = model_artifact.get("algorithms") or project_config["targets"]["scheme1"]["algorithms"]
    epsilon = project_config["evaluation"]["epsilon"]

    if scheme == "scheme1":
        return build_flat_results(
            cat_df,
            clf,
            features,
            algorithms,
            le=le,
            epsilon=epsilon,
            tol=tol,
        )
    if scheme == "scheme2":
        return run_full_evaluation_hier5(
            cat_df,
            clf,
            features=features,
            algorithms=algorithms,
            le=le,
            epsilon=epsilon,
            tol=tol,
            save_files=False,
        )
    return run_full_evaluation_scheme3(
        cat_df,
        clf,
        features=features,
        algorithms=algorithms,
        le=le,
        epsilon=epsilon,
        tol=tol,
        save_files=False,
    )


def train_scheme1(cat_df, project_config, model_config, model_family, variant, tol=0.0, save_files=False, file_suffix=None):
    scheme_cfg = model_config["scheme1"]
    family_cfg = scheme_cfg[model_family]
    variants = family_cfg["variants"]

    if variant is None:
        variant = next(iter(variants))
    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}' for {model_family}")

    variant_cfg = variants[variant]
    split_mode = variant_cfg["split_mode"]
    run_name = create_run_name("scheme1", model_family, split_mode, variant=variant, tol=tol, file_suffix=file_suffix)
    output_dirs = ensure_output_dirs(project_config, "scheme1", model_family)

    split_kwargs = {
        "weight_col": scheme_cfg["shared"]["sample_weight_col"],
        "test_size": project_config["split_rules"]["defaults"]["test_size"],
        "random_state": project_config["split_rules"]["defaults"]["random_state"],
        "tie_tolerance": project_config["split_rules"]["defaults"]["tie_tolerance"],
    }
    model_kwargs = dict(variant_cfg["hyperparameters"])

    if model_family == "random_forest":
        clf, split_data = fit_random_forest_classifier(
            cat_df,
            model_name=variant,
            split_mode=split_mode,
            model_kwargs=model_kwargs,
            split_kwargs=split_kwargs,
        )
        le = None
    else:
        sample_weight_transform_mode = variant_cfg.get(
            "sample_weight_transform",
            family_cfg["defaults"].get("sample_weight_transform", "raw"),
        )
        clf, le, split_data = fit_xgb_classifier(
            cat_df,
            model_name=variant,
            split_mode=split_mode,
            model_kwargs=model_kwargs,
            split_kwargs=split_kwargs,
            sample_weight_transform_mode=sample_weight_transform_mode,
        )

    cls_metrics, time_metrics, results_df, wrong_df = build_flat_results(
        cat_df,
        clf,
        project_config["features"]["feature_cols"],
        project_config["targets"]["scheme1"]["algorithms"],
        le=le,
        epsilon=project_config["evaluation"]["epsilon"],
        tol=tol,
    )

    artifacts = {}
    if save_files:
        artifacts.update(
            save_model_artifact(
                output_dirs,
                run_name,
                clf,
                le,
                "scheme1",
                model_family,
                split_mode=split_mode,
                variant=variant,
                feature_cols=project_config["features"]["feature_cols"],
                algorithms=project_config["targets"]["scheme1"]["algorithms"],
            )
        )
        artifacts.update(
            save_dataframe_outputs(
                results_df,
                wrong_df,
                output_dirs,
                project_config["output_locations"]["filenames"],
                run_name,
            )
        )
        artifacts.update(
            save_metrics_outputs(
                cls_metrics,
                time_metrics,
                output_dirs,
                project_config["output_locations"]["filenames"]["metrics_workbook"],
                run_name,
            )
        )
        artifacts.update(save_flat_reports(results_df, project_config["targets"]["scheme1"]["algorithms"], output_dirs, run_name))
        artifacts.update(save_split_indices(split_data, output_dirs, run_name))

    return {
        "scheme": "scheme1",
        "model_family": model_family,
        "variant": variant,
        "run_name": run_name,
        "classifier": clf,
        "label_encoder": le,
        "split_data": split_data,
        "classification_metrics": cls_metrics,
        "time_metrics": time_metrics,
        "results_df": results_df,
        "wrong_df": wrong_df,
        "artifacts": artifacts,
    }


def train_scheme2(cat_df, project_config, model_config, model_family, split_mode=None, tol=0.0, save_files=False, file_suffix=None):
    scheme_cfg = model_config["scheme2"]
    family_cfg = scheme_cfg[model_family]
    stage_cfg = family_cfg["stages"]
    stage_params = family_cfg["stage_wise_parameters"]

    split_mode = split_mode or project_config["split_rules"]["defaults"]["stratify_on_target_only"]
    run_name = create_run_name("scheme2", model_family, split_mode, tol=tol, file_suffix=file_suffix)
    output_dirs = ensure_output_dirs(project_config, "scheme2", model_family)

    model_kwargs = {
        "group_clf_kwargs": dict(stage_cfg["group_clf"]["hyperparameters"]),
        "sparse_subgroup_clf_kwargs": dict(stage_cfg["sparse_subgroup_clf"]["hyperparameters"]),
        "altRB_clf_kwargs": dict(stage_cfg["altRB_clf"]["hyperparameters"]),
        "noAltRB_clf_kwargs": dict(stage_cfg["noAltRB_clf"]["hyperparameters"]),
        "dense_clf_kwargs": dict(stage_cfg["dense_clf"]["hyperparameters"]),
        "tau_group": stage_params["tau_group"],
        "tau_sparse": stage_params["tau_sparse"],
        "tau_algo": stage_params["tau_algo"],
        "fallback_label": stage_params["fallback_label"],
    }
    split_kwargs = {
        "weight_col": project_config["split_rules"]["defaults"]["weight_col"],
        "test_size": project_config["split_rules"]["defaults"]["test_size"],
        "random_state": project_config["split_rules"]["defaults"]["random_state"],
        "tie_tolerance": project_config["split_rules"]["defaults"]["tie_tolerance"],
    }

    if model_family == "random_forest":
        clf, split_data = fit_hierarchical_random_forest_classifier(
            cat_df,
            split_mode=split_mode,
            model_kwargs=model_kwargs,
            split_kwargs=split_kwargs,
        )
        le = None
    else:
        clf, split_data = fit_hierarchical_xgb_classifier(
            cat_df,
            split_mode=split_mode,
            model_kwargs=model_kwargs,
            split_kwargs=split_kwargs,
        )
        le = split_data["le_algo"]

    cls_metrics, time_metrics, results_df, wrong_df = run_full_evaluation_hier5(
        cat_df,
        clf,
        features=project_config["features"]["feature_cols"],
        algorithms=project_config["targets"]["scheme1"]["algorithms"],
        le=le,
        epsilon=project_config["evaluation"]["epsilon"],
        tol=tol,
        save_files=False,
    )

    X_full = results_df[project_config["features"]["feature_cols"]].values
    stage_summary_df, per_class_reports = evaluate_hierarchical_stages(
        results_df,
        clf,
        X_full,
        project_config["targets"]["scheme1"]["algorithms"],
    )

    artifacts = {}
    if save_files:
        artifacts.update(
            save_model_artifact(
                output_dirs,
                run_name,
                clf,
                le,
                "scheme2",
                model_family,
                split_mode=split_mode,
                variant=None,
                feature_cols=project_config["features"]["feature_cols"],
                algorithms=project_config["targets"]["scheme1"]["algorithms"],
            )
        )
        artifacts.update(
            save_dataframe_outputs(
                results_df,
                wrong_df,
                output_dirs,
                project_config["output_locations"]["filenames"],
                run_name,
            )
        )
        artifacts.update(
            save_metrics_outputs(
                cls_metrics,
                time_metrics,
                output_dirs,
                project_config["output_locations"]["filenames"]["metrics_workbook"],
                run_name,
            )
        )
        artifacts.update(
            save_hierarchical_stage_outputs(
                stage_summary_df,
                per_class_reports,
                output_dirs,
                project_config["output_locations"]["filenames"],
                run_name,
            )
        )
        artifacts.update(
            save_scheme2_confusion_matrices(
                results_df,
                clf,
                project_config["features"]["feature_cols"],
                project_config["targets"]["scheme1"]["algorithms"],
                output_dirs,
                run_name,
            )
        )
        artifacts.update(save_split_indices(split_data, output_dirs, run_name))

    return {
        "scheme": "scheme2",
        "model_family": model_family,
        "variant": None,
        "run_name": run_name,
        "classifier": clf,
        "label_encoder": le,
        "split_data": split_data,
        "classification_metrics": cls_metrics,
        "time_metrics": time_metrics,
        "stage_summary_df": stage_summary_df,
        "per_class_reports": per_class_reports,
        "results_df": results_df,
        "wrong_df": wrong_df,
        "artifacts": artifacts,
    }


def train_scheme3(cat_df, project_config, model_config, model_family, split_mode=None, tol=0.0, save_files=False, file_suffix=None):
    scheme_cfg = model_config["scheme3"]
    family_cfg = scheme_cfg[model_family]
    stage_cfg = family_cfg["stages"]
    stage_params = family_cfg["stage_wise_parameters"]

    split_mode = split_mode or project_config["split_rules"]["defaults"]["stratify_on_target_only"]
    run_name = create_run_name("scheme3", model_family, split_mode, tol=tol, file_suffix=file_suffix)
    output_dirs = ensure_output_dirs(project_config, "scheme3", model_family)

    model_kwargs = {
        "group_clf_kwargs": dict(stage_cfg["group_clf"]["hyperparameters"]),
        "sparse_subgroup_clf_kwargs": dict(stage_cfg["sparse_subgroup_clf"]["hyperparameters"]),
        "orientation_clf_kwargs": dict(stage_cfg["orientation_clf"]["hyperparameters"]),
        "router4_clf_kwargs": dict(stage_cfg["router4_clf"]["hyperparameters"]),
        "tau_group": stage_params["tau_group"],
        "tau_sparse": stage_params["tau_sparse"],
        "tau_orientation": stage_params.get("tau_orientation", 0.0),
        "tau_router": stage_params["tau_router"],
        "fallback_label": stage_params["fallback_label"],
    }
    split_kwargs = {
        "weight_col": project_config["split_rules"]["defaults"]["weight_col"],
        "test_size": project_config["split_rules"]["defaults"]["test_size"],
        "random_state": project_config["split_rules"]["defaults"]["random_state"],
        "tie_tolerance": project_config["split_rules"]["defaults"]["tie_tolerance"],
    }

    if model_family == "random_forest":
        clf, split_data = fit_hierarchical_scheme3_random_forest_classifier(
            cat_df,
            split_mode=split_mode,
            model_kwargs=model_kwargs,
            split_kwargs=split_kwargs,
        )
        le = None
    else:
        clf, split_data = fit_hierarchical_scheme3_xgb_classifier(
            cat_df,
            split_mode=split_mode,
            model_kwargs=model_kwargs,
            split_kwargs=split_kwargs,
        )
        le = split_data["le_algo"]

    cls_metrics, time_metrics, results_df, wrong_df = run_full_evaluation_scheme3(
        cat_df,
        clf,
        features=project_config["features"]["feature_cols"],
        algorithms=project_config["targets"]["scheme1"]["algorithms"],
        le=le,
        epsilon=project_config["evaluation"]["epsilon"],
        tol=tol,
        save_files=False,
    )

    X_full = results_df[project_config["features"]["feature_cols"]].values
    stage_summary_df, per_class_reports = evaluate_hierarchical_scheme3_stages(
        results_df,
        clf,
        X_full,
        project_config["targets"]["scheme1"]["algorithms"],
    )

    artifacts = {}
    if save_files:
        artifacts.update(
            save_model_artifact(
                output_dirs,
                run_name,
                clf,
                le,
                "scheme3",
                model_family,
                split_mode=split_mode,
                variant=None,
                feature_cols=project_config["features"]["feature_cols"],
                algorithms=project_config["targets"]["scheme1"]["algorithms"],
            )
        )
        artifacts.update(
            save_dataframe_outputs(
                results_df,
                wrong_df,
                output_dirs,
                project_config["output_locations"]["filenames"],
                run_name,
            )
        )
        artifacts.update(
            save_metrics_outputs(
                cls_metrics,
                time_metrics,
                output_dirs,
                project_config["output_locations"]["filenames"]["metrics_workbook"],
                run_name,
            )
        )
        artifacts.update(
            save_hierarchical_stage_outputs(
                stage_summary_df,
                per_class_reports,
                output_dirs,
                project_config["output_locations"]["filenames"],
                run_name,
            )
        )
        artifacts.update(
            save_scheme3_confusion_matrices(
                results_df,
                clf,
                project_config["features"]["feature_cols"],
                project_config["targets"]["scheme1"]["algorithms"],
                output_dirs,
                run_name,
            )
        )
        artifacts.update(save_split_indices(split_data, output_dirs, run_name))

    return {
        "scheme": "scheme3",
        "model_family": model_family,
        "variant": None,
        "run_name": run_name,
        "classifier": clf,
        "label_encoder": le,
        "split_data": split_data,
        "classification_metrics": cls_metrics,
        "time_metrics": time_metrics,
        "stage_summary_df": stage_summary_df,
        "per_class_reports": per_class_reports,
        "results_df": results_df,
        "wrong_df": wrong_df,
        "artifacts": artifacts,
    }


def run_training(
    scheme,
    model_family,
    variant=None,
    split_mode=None,
    tol=None,
    save_files=None,
    file_suffix=None,
    project_config_path=DEFAULT_PROJECT_CONFIG,
    model_config_path=None,
):
    scheme = normalize_scheme(scheme)
    model_family = normalize_model_family(model_family)

    project_config = load_yaml_config(project_config_path)
    model_config = load_yaml_config(resolve_model_config_path(scheme, model_config_path))

    tol = project_config["tolerances"]["tie_tolerance"] if tol is None else tol
    save_files = project_config["evaluation"]["save_files_default"] if save_files is None else save_files

    cat_df = load_and_preprocess_data(project_config, scheme)

    if scheme == "scheme1":
        run_result = train_scheme1(
            cat_df,
            project_config,
            model_config,
            model_family,
            variant=variant,
            tol=tol,
            save_files=save_files,
            file_suffix=file_suffix,
        )
    elif scheme == "scheme2":
        run_result = train_scheme2(
            cat_df,
            project_config,
            model_config,
            model_family,
            split_mode=split_mode,
            tol=tol,
            save_files=save_files,
            file_suffix=file_suffix,
        )
    else:
        run_result = train_scheme3(
            cat_df,
            project_config,
            model_config,
            model_family,
            split_mode=split_mode,
            tol=tol,
            save_files=save_files,
            file_suffix=file_suffix,
        )

    return run_result


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train Scheme 1, Scheme 2, or Scheme 3 models in the restructured pipeline.")
    parser.add_argument("--scheme", required=True, choices=["scheme1", "scheme2", "scheme3"])
    parser.add_argument("--model-family", required=True, choices=["rf", "random_forest", "xgb", "xgboost"])
    parser.add_argument("--variant", default=None, help="Scheme 1 variant name, e.g. rf_strat_y or xgb_strat_y.")
    parser.add_argument("--split-mode", default=None, help="Optional split mode override, e.g. strat_y or strat_y_K.")
    parser.add_argument("--tol", type=float, default=None, help="Tolerance for evaluation metrics.")
    parser.add_argument("--save-files", action="store_true", help="Write predictions, metrics, reports, and wrong cases.")
    parser.add_argument("--file-suffix", default=None, help="Optional explicit file suffix/run name.")
    parser.add_argument("--project-config", default=str(DEFAULT_PROJECT_CONFIG))
    parser.add_argument("--model-config", default=None, help="Optional explicit model config path.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    result = run_training(
        scheme=args.scheme,
        model_family=args.model_family,
        variant=args.variant,
        split_mode=args.split_mode,
        tol=args.tol,
        save_files=args.save_files,
        file_suffix=args.file_suffix,
        project_config_path=args.project_config,
        model_config_path=args.model_config,
    )

    print(f"Run complete: {result['run_name']}")
    print(f"Scheme: {result['scheme']}")
    print(f"Model family: {result['model_family']}")
    print(f"Accuracy: {result['classification_metrics']['accuracy']:.4f}")
    print(f"Time ratio: {result['time_metrics']['time_ratio']:.4f}")
    if args.save_files:
        print("Artifacts:")
        for key, value in sorted(result["artifacts"].items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
