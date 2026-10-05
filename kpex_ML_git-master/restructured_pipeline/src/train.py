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
    W_TIME_DIFF_RELATIVE,
    W_LOG_TIME_RATIO,
    W_TIME_MIN,
    W_LOG_TIME_MIN,
    W_TIME_MIN_X_W_LOG_TIME_RATIO,
    W_LOG_TIME_MIN_X_W_LOG_TIME_RATIO,
    W_LOG_TIME_MIN_PLUS_W_LOG_TIME_RATIO,
    ORIENTED_ALGOS,
    NON_ORIENTED_ALGOS,
    # B_ALGOS,
    # OTHER_ALGOS,
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
DEFAULT_SCHEME4_CONFIG = Path("restructured_pipeline/configs/models_scheme4.yaml")
DEFAULT_SCHEME5_CONFIG = Path("restructured_pipeline/configs/models_scheme5.yaml")


# Scheme 1 sample-weight choices. The five kth-min schemes are implemented
# by features.assign_weights()/compute_weight().
WEIGHT_SCHEME_ALIASES = {
    "time_ratio": W_TIME_RATIO,
    "time_diff_relative": W_TIME_DIFF_RELATIVE,
    "log_time_ratio": W_LOG_TIME_RATIO,
    "time_min": W_TIME_MIN,
    "log_time_min": W_LOG_TIME_MIN,
    "time_min_x_log_time_ratio": W_TIME_MIN_X_W_LOG_TIME_RATIO,
    "log_time_min_x_log_time_ratio": W_LOG_TIME_MIN_X_W_LOG_TIME_RATIO,
    "log_time_min_plus_log_time_ratio": W_LOG_TIME_MIN_PLUS_W_LOG_TIME_RATIO,
    "2min_x_log_2min_ratio": "2min_x_log_2min_ratio",
    "3min_x_log_3min_ratio": "3min_x_log_3min_ratio",
    "4min_x_log_4min_ratio": "4min_x_log_4min_ratio",
    "5min_x_log_5min_ratio": "5min_x_log_5min_ratio",
    "6min_x_log_6min_ratio": "6min_x_log_6min_ratio",
}
SUPPORTED_WEIGHT_SCHEMES = tuple(WEIGHT_SCHEME_ALIASES)


def normalize_weight_scheme(weight_scheme):
    if weight_scheme in WEIGHT_SCHEME_ALIASES.values():
        return weight_scheme
    key = str(weight_scheme).strip()
    if key not in WEIGHT_SCHEME_ALIASES:
        raise ValueError(
            f"Unsupported weight scheme {weight_scheme!r}. "
            f"Expected one of: {SUPPORTED_WEIGHT_SCHEMES}"
        )
    return WEIGHT_SCHEME_ALIASES[key]


def weight_scheme_name(weight_scheme):
    normalized = normalize_weight_scheme(weight_scheme)
    for name, value in WEIGHT_SCHEME_ALIASES.items():
        if value == normalized:
            return name
    raise ValueError(f"Could not name weight scheme: {weight_scheme!r}")



def load_yaml_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_scheme(scheme):
    mapping = {"scheme1": "scheme1", "scheme2": "scheme2", "scheme3": "scheme3", "scheme4": "scheme4", "scheme5": "scheme5"}
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
    if scheme == "scheme3":
        return DEFAULT_SCHEME3_CONFIG
    if scheme == "scheme4":
        return DEFAULT_SCHEME4_CONFIG
    return DEFAULT_SCHEME5_CONFIG


def ensure_output_dirs(project_config, scheme, model_family, subfamily=None, experiment_subdir=None):
    pipeline_paths = project_config["paths"]["pipeline"]
    parts = [scheme]
    if subfamily:
        parts.append(subfamily)
    parts.append(model_family)
    if experiment_subdir:
        parts.append(str(experiment_subdir))
    dirs = {
        "models": Path(pipeline_paths["models_dir"]).joinpath(*parts),
        "predictions": Path(pipeline_paths["predictions_dir"]).joinpath(*parts),
        "wrong_cases": Path(pipeline_paths["wrong_cases_dir"]).joinpath(*parts),
        "metrics": Path(pipeline_paths["metrics_dir"]).joinpath(*parts),
        "logs": Path(pipeline_paths["logs_dir"]).joinpath(*parts),
        "splits": Path(pipeline_paths["splits_data_dir"]).joinpath(*parts),
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


def load_and_preprocess_data(project_config, scheme, weight_scheme=W_TIME_RATIO):
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
    normalized_weight_scheme = normalize_weight_scheme(weight_scheme)
    assign_weights(cat_df, normalized_weight_scheme, eval_cfg["epsilon"])
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

def build_scheme3_orientation_branch_masks(cat_df):
    """
    Branch masks for Scheme 3 orientation training.

    This keeps the orientation classifier branch-local instead of training it
    on the whole dataset.
    """
    sparse_mask = cat_df["y_group"].eq("Sparse")
    dense_mask = cat_df["y_group"].eq("Dense")

    return {
        "dense": dense_mask,
        "sparse_altRB": sparse_mask & cat_df["y_sparse_subgroup"].eq("AltRB"),
        "sparse_noAltRB": sparse_mask & cat_df["y_sparse_subgroup"].eq("No-AltRB"),
    }

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
        def get_top3(row):
            probs = {alg: row[f"prob_{alg}"] for alg in algorithms}
            return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]

        results_df["top3_predicted"] = results_df.apply(get_top3, axis=1)
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


def save_metrics_outputs(
    cls_metrics,
    time_metrics,
    output_dirs,
    metrics_workbook,
    run_name,
    sheet_name=None,
):
    metrics_workbook_path = output_dirs["metrics"] / metrics_workbook

    if sheet_name is None:
        sheet_name = run_name[:31]

    write_metrics_to_excel_file(
        cls_metrics,
        time_metrics,
        str(metrics_workbook_path),
        sheet_name=sheet_name,
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


def save_flat_reports(results_df, algorithms, output_dirs, run_name, scheme_label="Scheme 1"):
    y_true = results_df.apply(lambda row: get_fastest_algorithm(row, algorithms), axis=1)
    y_pred = results_df["predicted"]
    labels = build_confusion_labels(y_true, y_pred, algorithms)

    confusion_path = output_dirs["metrics"] / f"confusion_matrix_{run_name}.png"
    save_confusion_matrix(y_true, y_pred, labels, confusion_path, f"{scheme_label} Confusion Matrix: {run_name}")

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
            orientation_branches = {
                "dense": (
                    results_df["y_group"] == "Dense",
                    clf.orientation_dense_clf,
                    "Dense",
                ),
                "altRB": (
                    (results_df["y_group"] == "Sparse")
                    & (results_df["y_sparse_subgroup"] == "AltRB"),
                    clf.orientation_altRB_clf,
                    "Sparse AltRB",
                ),
                "noAltRB": (
                    (results_df["y_group"] == "Sparse")
                    & (results_df["y_sparse_subgroup"] == "No-AltRB"),
                    clf.orientation_noAltRB_clf,
                    "Sparse No-AltRB",
                ),
            }

            for branch_key, (branch_mask, branch_clf, branch_title) in orientation_branches.items():
                if not branch_mask.any():
                    continue

                stage3_pred = clf.le_orientation.inverse_transform(
                    branch_clf.predict(X_full[branch_mask])
                )

                confusion_path = (
                    output_dirs["metrics"]
                    / f"confusion_matrix_{run_name}_stage3_{branch_key}.png"
                )

                save_confusion_matrix(
                    results_df.loc[branch_mask, "y_orientation"],
                    stage3_pred,
                    build_confusion_labels(
                        results_df.loc[branch_mask, "y_orientation"],
                        stage3_pred,
                        ["Oriented", "NonOriented"],
                    ),
                    confusion_path,
                    f"Stage 3 ({branch_title}): {run_name}",
                )

                outputs[f"stage3_{branch_key}_confusion_matrix_path"] = str(
                    confusion_path
                )
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
            orientation_branches = {
                "dense": (
                    results_df["y_group"] == "Dense",
                    clf.orientation_dense_clf,
                    "Dense",
                ),
                "altRB": (
                    (results_df["y_group"] == "Sparse")
                    & (results_df["y_sparse_subgroup"] == "AltRB"),
                    clf.orientation_altRB_clf,
                    "Sparse AltRB",
                ),
                "noAltRB": (
                    (results_df["y_group"] == "Sparse")
                    & (results_df["y_sparse_subgroup"] == "No-AltRB"),
                    clf.orientation_noAltRB_clf,
                    "Sparse No-AltRB",
                ),
            }

            for branch_key, (branch_mask, branch_clf, branch_title) in orientation_branches.items():
                if not branch_mask.any():
                    continue

                stage3_pred = branch_clf.predict(X_full[branch_mask])

                confusion_path = (
                    output_dirs["metrics"]
                    / f"confusion_matrix_{run_name}_stage3_{branch_key}.png"
                )

                save_confusion_matrix(
                    results_df.loc[branch_mask, "y_orientation"],
                    stage3_pred,
                    build_confusion_labels(
                        results_df.loc[branch_mask, "y_orientation"],
                        stage3_pred,
                        ["Oriented", "NonOriented"],
                    ),
                    confusion_path,
                    f"Stage 3 ({branch_title}): {run_name}",
                )

                outputs[f"stage3_{branch_key}_confusion_matrix_path"] = str(
                    confusion_path
                )

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


def save_model_artifact(output_dirs, run_name, clf, le, scheme, model_family, split_mode=None, variant=None, feature_cols=None, algorithms=None, scheme4_family=None, scheme5_family=None):
    model_path = output_dirs["models"] / f"model_{run_name}.pkl"
    payload = {
        "scheme": scheme,
        "model_family": model_family,
        "scheme4_family": scheme4_family,
        "scheme5_family": scheme5_family,
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
    # Prediction/evaluation does not depend on the training sample-weight formula.
    cat_df = load_and_preprocess_data(project_config, scheme)
    clf = model_artifact["classifier"]
    le = model_artifact.get("label_encoder")
    features = model_artifact.get("feature_cols") or project_config["features"]["feature_cols"]
    algorithms = model_artifact.get("algorithms") or project_config["targets"]["scheme1"]["algorithms"]
    epsilon = project_config["evaluation"]["epsilon"]

    if scheme in {"scheme1", "scheme4", "scheme5"}:
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


def train_scheme1(cat_df, project_config, model_config, model_family, variant, tol=0.0, save_files=False, file_suffix=None, weight_scheme=W_TIME_RATIO):
    scheme_cfg = model_config["scheme1"]
    family_cfg = scheme_cfg[model_family]
    variants = family_cfg["variants"]

    if variant is None:
        variant = next(iter(variants))
    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}' for {model_family}")

    variant_cfg = variants[variant]
    split_mode = variant_cfg["split_mode"]
    run_name = create_run_name(
        "scheme1", model_family, split_mode, variant=variant, tol=tol, file_suffix=file_suffix,
    )
    weight_name = weight_scheme_name(weight_scheme)
    # Every Scheme 1 weighting experiment is isolated by a readable folder name.
    if file_suffix is None:
        run_name = f"{run_name}_weight_{weight_name}"
    output_dirs = ensure_output_dirs(
        project_config, "scheme1", model_family,
        experiment_subdir=weight_name,
    )

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
        clf, le, split_data = fit_xgb_classifier(
            cat_df,
            model_name=variant,
            split_mode=split_mode,
            model_kwargs=model_kwargs,
            split_kwargs=split_kwargs,
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
                sheet_name=f"{split_mode}_entire_data",
            )
        )
        artifacts.update(save_flat_reports(results_df, project_config["targets"]["scheme1"]["algorithms"], output_dirs, run_name))
        artifacts.update(save_split_indices(split_data, output_dirs, run_name))

    return {
        "scheme": "scheme1",
        "model_family": model_family,
        "variant": variant,
        "weight_scheme": weight_scheme_name(weight_scheme),
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



def train_scheme4(
    cat_df,
    project_config,
    model_config,
    model_family,
    scheme4_family,
    variant,
    tol=0.0,
    save_files=False,
    file_suffix=None,
):
    """Train one independent Scheme 4 12-class XGB model."""
    if model_family != "xgboost":
        raise ValueError("Scheme 4 currently supports XGBoost only.")
    if scheme4_family not in {"oriented", "forward"}:
        raise ValueError("Scheme 4 requires --scheme4-family oriented or forward.")

    active_algorithms = (
        list(ORIENTED_ALGOS)
        if scheme4_family == "oriented"
        else list(NON_ORIENTED_ALGOS)
    )
    if len(active_algorithms) != 12:
        raise ValueError(
            f"Scheme 4 {scheme4_family} vocabulary must contain exactly 12 algorithms; "
            f"found {len(active_algorithms)}."
        )

    scheme_cfg = model_config["scheme4"]
    family_cfg = scheme_cfg[scheme4_family][model_family]
    variants = family_cfg["variants"]
    if variant is None:
        variant = next(iter(variants))
    if variant not in variants:
        raise ValueError(
            f"Unknown Scheme 4 variant {variant!r} for {scheme4_family}/{model_family}."
        )

    variant_cfg = variants[variant]
    split_mode = variant_cfg["split_mode"]
    run_name = file_suffix or f"scheme4_{scheme4_family}_{model_family}_{variant}"
    if tol > 0 and file_suffix is None:
        run_name += f"_tol_{int(round(tol * 100))}"
    output_dirs = ensure_output_dirs(
        project_config, "scheme4", model_family, subfamily=scheme4_family
    )

    split_kwargs = {
        "weight_col": scheme_cfg.get("shared", {}).get(
            "sample_weight_col", project_config["split_rules"]["defaults"]["weight_col"]
        ),
        "test_size": project_config["split_rules"]["defaults"]["test_size"],
        "random_state": project_config["split_rules"]["defaults"]["random_state"],
        "tie_tolerance": project_config["split_rules"]["defaults"]["tie_tolerance"],
        "algorithm_cols": active_algorithms,
    }
    model_kwargs = dict(variant_cfg["hyperparameters"])

    clf, le, split_data = fit_xgb_classifier(
        cat_df,
        model_name=variant,
        split_mode=split_mode,
        model_kwargs=model_kwargs,
        split_kwargs=split_kwargs,
    )

    cls_metrics, time_metrics, results_df, wrong_df = build_flat_results(
        cat_df,
        clf,
        project_config["features"]["feature_cols"],
        active_algorithms,
        le=le,
        epsilon=project_config["evaluation"]["epsilon"],
        tol=tol,
    )

    artifacts = {}
    if save_files:
        artifacts.update(save_model_artifact(
            output_dirs, run_name, clf, le, "scheme4", model_family,
            split_mode=split_mode, variant=variant,
            feature_cols=project_config["features"]["feature_cols"],
            algorithms=active_algorithms, scheme4_family=scheme4_family,
        ))
        artifacts.update(save_dataframe_outputs(
            results_df, wrong_df, output_dirs,
            project_config["output_locations"]["filenames"], run_name,
        ))
        artifacts.update(save_metrics_outputs(
            cls_metrics, time_metrics, output_dirs,
            project_config["output_locations"]["filenames"]["metrics_workbook"],
            run_name, sheet_name=f"{scheme4_family}_{split_mode}_entire_data",
        ))
        artifacts.update(save_flat_reports(
            results_df, active_algorithms, output_dirs, run_name,
            scheme_label=f"Scheme 4 {scheme4_family.title()}",
        ))
        artifacts.update(save_split_indices(split_data, output_dirs, run_name))

    return {
        "scheme": "scheme4",
        "scheme4_family": scheme4_family,
        "model_family": model_family,
        "variant": variant,
        "split_mode": split_mode,
        "run_name": run_name,
        "classifier": clf,
        "label_encoder": le,
        "algorithms": active_algorithms,
        "split_data": split_data,
        "classification_metrics": cls_metrics,
        "time_metrics": time_metrics,
        "results_df": results_df,
        "wrong_df": wrong_df,
        "artifacts": artifacts,
    }


def train_scheme5(
    cat_df, project_config, model_config, model_family, scheme5_family, variant,
    tol=0.0, save_files=False, file_suffix=None,
):
    """Train one independent Scheme 5 flat model: B(6) or Other(18), using RF or XGB."""
    if model_family not in {"random_forest", "xgboost"}:
        raise ValueError("Scheme 5 supports random_forest and xgboost.")
    if scheme5_family not in {"b", "other"}:
        raise ValueError("Scheme 5 requires --scheme5-family b or other.")

    active_algorithms = list(B_ALGOS if scheme5_family == "b" else OTHER_ALGOS)
    expected = 6 if scheme5_family == "b" else 18
    if len(active_algorithms) != expected:
        raise ValueError(
            f"Scheme 5 {scheme5_family} must contain exactly {expected} algorithms; "
            f"found {len(active_algorithms)}."
        )

    scheme_cfg = model_config["scheme5"]
    family_cfg = scheme_cfg[scheme5_family][model_family]
    variants = family_cfg["variants"]
    if variant is None:
        variant = next(iter(variants))
    if variant not in variants:
        raise ValueError(
            f"Unknown Scheme 5 variant {variant!r} for {scheme5_family}/{model_family}."
        )

    variant_cfg = variants[variant]
    split_mode = variant_cfg["split_mode"]
    run_name = file_suffix or f"scheme5_{scheme5_family}_{model_family}_{variant}"
    if tol > 0 and file_suffix is None:
        run_name += f"_tol_{int(round(tol * 100))}"

    output_dirs = ensure_output_dirs(
        project_config, "scheme5", model_family, subfamily=scheme5_family
    )
    split_kwargs = {
        "weight_col": scheme_cfg.get("shared", {}).get(
            "sample_weight_col", project_config["split_rules"]["defaults"]["weight_col"]
        ),
        "test_size": project_config["split_rules"]["defaults"]["test_size"],
        "random_state": project_config["split_rules"]["defaults"]["random_state"],
        "tie_tolerance": project_config["split_rules"]["defaults"]["tie_tolerance"],
        "algorithm_cols": active_algorithms,
    }
    model_kwargs = dict(variant_cfg["hyperparameters"])

    if model_family == "random_forest":
        clf, split_data = fit_random_forest_classifier(
            cat_df, model_name=variant, split_mode=split_mode,
            model_kwargs=model_kwargs, split_kwargs=split_kwargs,
        )
        le = None
    else:
        clf, le, split_data = fit_xgb_classifier(
            cat_df, model_name=variant, split_mode=split_mode,
            model_kwargs=model_kwargs, split_kwargs=split_kwargs,
        )

    cls_metrics, time_metrics, results_df, wrong_df = build_flat_results(
        cat_df, clf, project_config["features"]["feature_cols"], active_algorithms,
        le=le, epsilon=project_config["evaluation"]["epsilon"], tol=tol,
    )

    artifacts = {}
    if save_files:
        artifacts.update(save_model_artifact(
            output_dirs, run_name, clf, le, "scheme5", model_family,
            split_mode=split_mode, variant=variant,
            feature_cols=project_config["features"]["feature_cols"],
            algorithms=active_algorithms, scheme5_family=scheme5_family,
        ))
        artifacts.update(save_dataframe_outputs(
            results_df, wrong_df, output_dirs,
            project_config["output_locations"]["filenames"], run_name,
        ))
        artifacts.update(save_metrics_outputs(
            cls_metrics, time_metrics, output_dirs,
            project_config["output_locations"]["filenames"]["metrics_workbook"],
            run_name, sheet_name=f"{scheme5_family}_{model_family}_{split_mode}_entire_data",
        ))
        artifacts.update(save_flat_reports(
            results_df, active_algorithms, output_dirs, run_name,
            scheme_label=f"Scheme 5 {scheme5_family.upper() if scheme5_family == 'b' else 'Other'} {model_family}",
        ))
        artifacts.update(save_split_indices(split_data, output_dirs, run_name))

    return {
        "scheme": "scheme5", "scheme5_family": scheme5_family,
        "model_family": model_family, "variant": variant,
        "split_mode": split_mode, "run_name": run_name,
        "classifier": clf, "label_encoder": le,
        "algorithms": active_algorithms, "split_data": split_data,
        "classification_metrics": cls_metrics, "time_metrics": time_metrics,
        "results_df": results_df, "wrong_df": wrong_df, "artifacts": artifacts,
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
                sheet_name=f"{split_mode}_entire_data",
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
         # Scheme 3 change:
        # orientation classifier should be trained branch-locally, not on the full dataset
        # "orientation_branch_scope": "branch_local",
        # "orientation_branch_masks": orientation_branch_masks,
    }
    split_kwargs = {
        "weight_col": project_config["split_rules"]["defaults"]["weight_col"],
        "test_size": project_config["split_rules"]["defaults"]["test_size"],
        "random_state": project_config["split_rules"]["defaults"]["random_state"],
        "tie_tolerance": project_config["split_rules"]["defaults"]["tie_tolerance"],
    }

    orientation_branch_masks = build_scheme3_orientation_branch_masks(cat_df)

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
                sheet_name=f"{split_mode}_entire_data",
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
    scheme4_family=None,
    scheme5_family=None,
    weight_scheme=W_TIME_RATIO,
):
    scheme = normalize_scheme(scheme)
    model_family = normalize_model_family(model_family)

    project_config = load_yaml_config(project_config_path)
    model_config = load_yaml_config(resolve_model_config_path(scheme, model_config_path))

    tol = project_config["tolerances"]["tie_tolerance"] if tol is None else tol
    save_files = project_config["evaluation"]["save_files_default"] if save_files is None else save_files

    normalized_weight_scheme = normalize_weight_scheme(weight_scheme)

    # The selectable weighting experiment is for Scheme 1. Other schemes keep
    # their established time_ratio preprocessing behavior.
    preprocessing_weight_scheme = normalized_weight_scheme if scheme == "scheme1" else W_TIME_RATIO
    cat_df = load_and_preprocess_data(
        project_config, scheme, weight_scheme=preprocessing_weight_scheme
    )

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
            weight_scheme=normalized_weight_scheme,
        )
    elif scheme == "scheme4":
        run_result = train_scheme4(
            cat_df, project_config, model_config, model_family,
            scheme4_family=scheme4_family, variant=variant, tol=tol,
            save_files=save_files, file_suffix=file_suffix,
        )
    elif scheme == "scheme5":
        run_result = train_scheme5(
            cat_df, project_config, model_config, model_family,
            scheme5_family=scheme5_family, variant=variant, tol=tol,
            save_files=save_files, file_suffix=file_suffix,
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
    parser = argparse.ArgumentParser(description="Train Scheme 1, Scheme 2, Scheme 3, Scheme 4, or Scheme 5 models in the restructured pipeline.")
    parser.add_argument("--scheme", required=True, choices=["scheme1", "scheme2", "scheme3", "scheme4", "scheme5"])
    parser.add_argument("--model-family", required=True, choices=["rf", "random_forest", "xgb", "xgboost"])
    parser.add_argument("--scheme4-family", choices=["oriented", "forward"], default=None, help="Required for Scheme 4; selects the independent 12-algorithm family.")
    parser.add_argument("--scheme5-family", choices=["b", "other"], default=None, help="Required for Scheme 5; selects the independent 6-B or 18-Other family.")
    # parser.add_argument("--scheme5-family", choices=["b", "other"], default=None, help="Required for Scheme 5; selects the independent 6-B or 18-Other family.")
    parser.add_argument("--variant", default=None, help="Scheme 1 variant name, e.g. rf_strat_y or xgb_strat_y.")
    parser.add_argument("--split-mode", default=None, help="Optional split mode override, e.g. strat_y or strat_y_K.")
    parser.add_argument("--tol", type=float, default=None, help="Tolerance for evaluation metrics.")
    parser.add_argument(
        "--weight-scheme",
        default="time_ratio",
        choices=SUPPORTED_WEIGHT_SCHEMES,
        help=(
            "Scheme 1 sample-weight formula. Supports the legacy weighting formulas "
            "and kth-min experimental formulas."
        ),
    )
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
        scheme4_family=args.scheme4_family,
        scheme5_family=args.scheme5_family,
        weight_scheme=args.weight_scheme,
    )

    print(f"Run complete: {result['run_name']}")
    print(f"Scheme: {result['scheme']}")
    print(f"Model family: {result['model_family']}")
    if result.get("weight_scheme") is not None:
        print(f"Weight scheme: {result['weight_scheme']}")
    if result.get("scheme4_family"):
        print(f"Scheme 4 family: {result['scheme4_family']}")
    if result.get("scheme5_family"):
        print(f"Scheme 5 family: {result['scheme5_family']}")
    print(f"Accuracy: {result['classification_metrics']['accuracy']:.4f}")
    print(f"Time ratio: {result['time_metrics']['time_ratio']:.4f}")
    if args.save_files:
        print("Artifacts:")
        for key, value in sorted(result["artifacts"].items()):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
