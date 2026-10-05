"""Unified wrong-predictions comparison and Excel reporting entry point.
Scheme 4/5 support: run one independent family at a time so family probability spaces are never mixed.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from restructured_pipeline.src.write_utils import (
    aggregate_master_dataframes,
    ensure_time_gap_ratio,
    extract_filename_info,
)

from restructured_pipeline.src.features import algorithms as ACTIVE_ALGORITHMS


DEFAULT_OUTPUT_PATH = Path("restructured_pipeline/outputs/reports/all_errors_agg_all_models_and_experiments.xlsx")
DEFAULT_CONSOLIDATED_ENTIRE_DATA_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_schema_entire_data.xlsx"
)
DEFAULT_TOLERANCE_REPORT_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all.xlsx"
)
DEFAULT_TOLERANCE_IMPORTANT_FAILURES_INPUT_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all.xlsx"),
    Path("restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report.xlsx"),
]
DEFAULT_TOLERANCE_IMPORTANT_FAILURES_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_important_failures.xlsx"
)
DEFAULT_WRONG_CASES_ROOT = Path("restructured_pipeline/outputs/wrong_cases")
ENTIRE_DATA_WRONG_PREDICTIONS_GLOB = "wrong_predictions_*predict_inference_entire_data*.xlsx"
TEST_DATA_WRONG_PREDICTIONS_GLOB = "wrong_predictions_*predict_inference_test_data*.xlsx"
DEFAULT_TEST_DATA_REPORTS_DIR = Path("restructured_pipeline/outputs/reports/test_data")
DEFAULT_TEST_DATA_COMPARE_OUTPUT_PATH = DEFAULT_TEST_DATA_REPORTS_DIR / "compare_wrong_predictions_test_data.xlsx"
DEFAULT_TEST_DATA_TOLERANCE_OUTPUT_PATH = (
    DEFAULT_TEST_DATA_REPORTS_DIR / "compare_wrong_predictions_tolerance_report_test_data.xlsx"
)
DEFAULT_TEST_DATA_IMPORTANT_FAILURES_OUTPUT_PATH = (
    DEFAULT_TEST_DATA_REPORTS_DIR / "compare_wrong_predictions_tolerance_important_failures_test_data.xlsx"
)
DEFAULT_ENTIRE_DATA_REPORTS_DIR = Path("restructured_pipeline/outputs/reports/entire_data")
DEFAULT_ENTIRE_DATA_COMPARE_OUTPUT_PATH = (
    DEFAULT_ENTIRE_DATA_REPORTS_DIR / "compare_wrong_predictions_entire_data.xlsx"
)
DEFAULT_ENTIRE_DATA_TOLERANCE_OUTPUT_PATH = (
    DEFAULT_ENTIRE_DATA_REPORTS_DIR / "compare_wrong_predictions_tolerance_report_entire_data.xlsx"
)
DEFAULT_ENTIRE_DATA_IMPORTANT_FAILURES_OUTPUT_PATH = (
    DEFAULT_ENTIRE_DATA_REPORTS_DIR / "compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"
)
STRATEGY_FOLDERS = ("expand_and_split", "split_and_expand")
RATIO_EPSILON = 1e-12
DEFAULT_TOLERANCE_VALUES = (0, 10, 20, 50)
DEFAULT_IMPORTANT_FAILURE_SUMMARY_SHEET = "filter_summary"
DEFAULT_SMALL_RUNTIME_THRESHOLD = 5.0
DEFAULT_SLOWDOWN_RATIO_THRESHOLD = 1.2
TEST_SET_INFERENCE_LABEL = "on 20% test set"
ENTIRE_DATA_INFERENCE_LABEL = "on entire dataset"
SEVERITY_FILLS = {
    "\u22645%": PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid"),
    "5\u201320%": PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid"),
    "20\u2013100%": PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid"),
    ">100%": PatternFill(start_color="F4CCCC", end_color="F4CCCC", fill_type="solid"),
}
YELLOW_FILL = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
FINAL_COLUMN_ORDER = [
    "inference_on",
    "with_holdout",
    "strat_on",
    "k_value",
    "dataset",
    "scheme",
    "model",
    "source_model",
    "failure_severity",
    "fastest_algo",
    "fastest_runtime",
    "predicted_algo",
    "predicted_probability",
    "fastest_time",
    "predicted_time",
    "time_gap",
    "time_gap_ratio",
    "time_ratio",
    "top3_predicted_detailed",
]
TOLERANCE_REPORT_COLUMNS = [
    "inference_on",
    "strat_on",
    "k_value",
    "dataset",
    "scheme",
    "model",
    "failure_severity",
    "fastest_algo",
    "fastest_runtime",
    "predicted_algo",
    "predicted_time",
    "time_gap",
    "time_ratio",
    "top3_predicted_detailed",
    *[f"prob_{algo}" for algo in ACTIVE_ALGORITHMS],

    "time_gap_ratio",
]
TOLERANCE_REPORT_SHEET_NAMES = [
    "scheme_1_tol_0",
    "scheme_1_tol_10",
    "scheme_1_tol_20",
    "scheme_1_tol_50",
    "scheme_2_tol_0",
    "scheme_2_tol_10",
    "scheme_2_tol_20",
    "scheme_2_tol_50",
    "scheme_3_tol_0",
    "scheme_3_tol_10",
    "scheme_3_tol_20",
    "scheme_3_tol_50",
    "scheme_4_tol_0",
    "scheme_4_tol_10",
    "scheme_4_tol_20",
    "scheme_4_tol_50",
    "scheme_5_tol_0",
    "scheme_5_tol_10",
    "scheme_5_tol_20",
    "scheme_5_tol_50",
]


def _parse_literal(value):
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            matches = re.findall(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)", stripped)
            if matches:
                parsed = []
                for algo, probability, time_ratio in matches:
                    prob_value = _safe_float(probability)
                    ratio_value = _safe_float(time_ratio)
                    if prob_value is None:
                        continue
                    parsed.append((algo.strip(), prob_value, ratio_value))
                if parsed:
                    return parsed
            return value
    return value


def _ensure_list(value):
    value = _parse_literal(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    return [value]


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _compute_ratio(numerator, denominator):
    if numerator is None or denominator is None:
        return pd.NA
    if denominator > 0:
        return numerator / denominator
    return numerator / RATIO_EPSILON


def _extract_top3_top1(value):
    top3 = _ensure_list(value)
    if not top3:
        return None
    first = top3[0]
    if not isinstance(first, (list, tuple)) or not first:
        return None
    return first[0]


def _format_top3_for_export(value) -> str:
    top3 = _ensure_list(value)
    formatted = []
    for item in top3:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        algo = str(item[0])
        probability = _safe_float(item[1])
        time_ratio = _safe_float(item[2]) if len(item) >= 3 else None
        if probability is None:
            continue
        time_ratio_text = "nan" if time_ratio is None else f"{time_ratio:.4f}"
        formatted.append(f"({algo}, {probability:.4f}, {time_ratio_text})")
    return "[" + ", ".join(formatted) + "]"


def autosize_columns(worksheet):
    for column_cells in worksheet.columns:
        values = [cell.value for cell in column_cells if cell.value is not None]
        max_width = max((len(str(value)) for value in values), default=0)
        worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_width + 2, 12), 40)


def highlight_rows_by_failure_severity(worksheet, severity_col: str = "failure_severity"):
    headers = [cell.value for cell in worksheet[1]]
    if severity_col not in headers:
        return

    severity_idx = headers.index(severity_col) + 1
    last_col = len(headers)

    for row_idx in range(2, worksheet.max_row + 1):
        severity = worksheet.cell(row=row_idx, column=severity_idx).value
        fill = SEVERITY_FILLS.get(severity)
        if fill is None:
            continue

        for col_idx in range(1, last_col + 1):
            worksheet.cell(row=row_idx, column=col_idx).fill = fill


def highlight_abnormal_time_gap_cells(
    worksheet,
    threshold: float,
    time_gap_col: str = "time_gap",
):
    headers = [cell.value for cell in worksheet[1]]
    if time_gap_col not in headers:
        return

    time_gap_idx = headers.index(time_gap_col) + 1

    for row_idx in range(2, worksheet.max_row + 1):
        time_gap_value = worksheet.cell(row=row_idx, column=time_gap_idx).value
        try:
            if time_gap_value is None or float(time_gap_value) <= threshold:
                continue
        except Exception:
            continue

        worksheet.cell(row=row_idx, column=time_gap_idx).fill = YELLOW_FILL


def write_by_inference_on(
    df: pd.DataFrame,
    output_path: str,
    severity_col: str = "failure_severity",
    threshold: float = 0.5,
    time_gap_highlight_threshold: float = 100.0,
):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for inf_on in df["inference_on"].drop_duplicates():
            sheet_name = str(inf_on)[:31]
            subset = df[df["inference_on"] == inf_on].copy()
            if "top3_predicted_detailed" in subset.columns:
                subset["top3_predicted_detailed"] = subset["top3_predicted_detailed"].apply(_format_top3_for_export)
            subset.to_excel(writer, sheet_name=sheet_name, index=False)

    workbook = load_workbook(output)
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        highlight_rows_by_failure_severity(worksheet, severity_col=severity_col)
        highlight_abnormal_time_gap_cells(
            worksheet,
            threshold=time_gap_highlight_threshold,
        )
        autosize_columns(worksheet)

    workbook.save(output)
    workbook.close()

    print(f"Wrote {len(df)} rows to '{output}' with sheets: {list(df['inference_on'].drop_duplicates())}")


def _write_formatted_named_sheets(
    sheet_frames: dict[str, pd.DataFrame],
    output_path: str | Path,
    severity_col: str = "failure_severity",
    time_gap_highlight_threshold: float = 100.0,
    apply_row_severity_fills: bool = True,
):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, subset in sheet_frames.items():
            sheet_df = subset.copy()
            if "top3_predicted_detailed" in sheet_df.columns:
                sheet_df["top3_predicted_detailed"] = sheet_df["top3_predicted_detailed"].apply(_format_top3_for_export)
            sheet_df.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)

    workbook = load_workbook(output)
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        if apply_row_severity_fills:
            highlight_rows_by_failure_severity(worksheet, severity_col=severity_col)
        highlight_abnormal_time_gap_cells(
            worksheet,
            threshold=time_gap_highlight_threshold,
        )
        autosize_columns(worksheet)

    workbook.save(output)
    workbook.close()


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    return normalized[columns]


def _resolve_runtime_column(columns, preferred_name, fallback_name):
    if preferred_name in columns:
        return preferred_name
    if fallback_name in columns:
        return fallback_name
    raise KeyError(f"Neither '{preferred_name}' nor '{fallback_name}' exists in the input dataframe.")


def _write_summary_sheet(output_path: str | Path, summary_df: pd.DataFrame, summary_sheet_name: str) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        summary_df.to_excel(writer, sheet_name=summary_sheet_name, index=False)

    workbook = load_workbook(output_path)
    worksheet = workbook[summary_sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    autosize_columns(worksheet)
    workbook.save(output_path)
    workbook.close()


def _print_stage_summary(stage_name: str, df: pd.DataFrame) -> None:
    total_rows = int(len(df))
    print(f"[{stage_name}] total_rows={total_rows}")
    if "scheme" in df.columns:
        scheme_counts = df["scheme"].fillna("").astype(str).value_counts().sort_index()
        if not scheme_counts.empty:
            counts_text = ", ".join(f"{scheme}={count}" for scheme, count in scheme_counts.items())
            print(f"[{stage_name}] rows_by_scheme: {counts_text}")


def validate_consolidated_tolerance_report(sheet_frames: dict[str, pd.DataFrame]) -> None:
    sheet_names = list(sheet_frames.keys())
    if len(sheet_names) != len(TOLERANCE_REPORT_SHEET_NAMES):
        raise ValueError(
            "Tolerance report validation failed: expected "
            f"{len(TOLERANCE_REPORT_SHEET_NAMES)} sheets, found {len(sheet_names)}."
        )
    if sheet_names != TOLERANCE_REPORT_SHEET_NAMES:
        raise ValueError(
            "Tolerance report validation failed: unexpected sheet ordering/names. "
            f"Expected {TOLERANCE_REPORT_SHEET_NAMES}, found {sheet_names}."
        )

    previous_by_scheme: dict[str, int] = {}
    for sheet_name, df in sheet_frames.items():
        if list(df.columns) != TOLERANCE_REPORT_COLUMNS:
            raise ValueError(
                f"Tolerance report validation failed for {sheet_name}: "
                "column order does not match the required export order."
            )
        missing_columns = [column for column in TOLERANCE_REPORT_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Tolerance report validation failed for {sheet_name}: missing columns {missing_columns}."
            )
        duplicate_check_df = df.copy()
        if "top3_predicted_detailed" in duplicate_check_df.columns:
            duplicate_check_df["top3_predicted_detailed"] = duplicate_check_df["top3_predicted_detailed"].apply(
                _format_top3_for_export
            )
        duplicate_count = int(duplicate_check_df.duplicated(subset=TOLERANCE_REPORT_COLUMNS).sum())
        if duplicate_count:
            raise ValueError(
                f"Tolerance report validation failed for {sheet_name}: "
                f"{duplicate_count} duplicated exported rows detected."
            )
        if "top3_predicted_detailed" in df.columns:
            _ = df["top3_predicted_detailed"].apply(_format_top3_for_export)

        scheme_prefix = "_".join(sheet_name.split("_")[:2])
        current_count = int(len(df))
        previous_count = previous_by_scheme.get(scheme_prefix)
        if previous_count is not None and current_count > previous_count:
            raise ValueError(
                f"Tolerance report validation failed for {sheet_name}: "
                "row counts increased as tolerance became less strict."
            )
        previous_by_scheme[scheme_prefix] = current_count


def validate_test_data_rows(df: pd.DataFrame, stage_name: str) -> None:
    if "inference_on" not in df.columns:
        raise ValueError(f"{stage_name} validation failed: missing 'inference_on' column.")
    invalid = df[df["inference_on"] != TEST_SET_INFERENCE_LABEL]
    if not invalid.empty:
        raise ValueError(
            f"{stage_name} validation failed: found {len(invalid)} rows outside '{TEST_SET_INFERENCE_LABEL}'."
        )


def validate_entire_data_rows(df: pd.DataFrame, stage_name: str) -> None:
    if "inference_on" not in df.columns:
        raise ValueError(f"{stage_name} validation failed: missing 'inference_on' column.")
    invalid = df[df["inference_on"] != ENTIRE_DATA_INFERENCE_LABEL]
    if not invalid.empty:
        raise ValueError(
            f"{stage_name} validation failed: found {len(invalid)} rows outside '{ENTIRE_DATA_INFERENCE_LABEL}'."
        )


def resolve_test_data_output_paths(scheme4_family: str | None = None, scheme5_family: str | None = None, report_subdir: str | None = None, report_tag: str | None = None) -> dict[str, Path]:
    if scheme4_family:
        report_dir = DEFAULT_TEST_DATA_REPORTS_DIR / "scheme4" / scheme4_family
    elif scheme5_family:
        report_dir = DEFAULT_TEST_DATA_REPORTS_DIR / "scheme5" / scheme5_family
        report_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{report_tag}_" if report_tag else ""
        return {
            "compare": report_dir / f"{prefix}compare_wrong_predictions_test_data.xlsx",
            "tolerance": report_dir / f"{prefix}compare_wrong_predictions_tolerance_report_test_data.xlsx",
            "important_failures": report_dir / f"{prefix}compare_wrong_predictions_tolerance_important_failures_test_data.xlsx",
        }
    if report_subdir:
        report_dir = DEFAULT_TEST_DATA_REPORTS_DIR / report_subdir
        report_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{report_tag}_" if report_tag else ""
        return {
            "compare": report_dir / f"{prefix}compare_wrong_predictions_test_data.xlsx",
            "tolerance": report_dir / f"{prefix}compare_wrong_predictions_tolerance_report_test_data.xlsx",
            "important_failures": report_dir / f"{prefix}compare_wrong_predictions_tolerance_important_failures_test_data.xlsx",
        }
    DEFAULT_TEST_DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "compare": DEFAULT_TEST_DATA_COMPARE_OUTPUT_PATH,
        "tolerance": DEFAULT_TEST_DATA_TOLERANCE_OUTPUT_PATH,
        "important_failures": DEFAULT_TEST_DATA_IMPORTANT_FAILURES_OUTPUT_PATH,
    }


def resolve_entire_data_output_paths(scheme4_family: str | None = None, scheme5_family: str | None = None, report_subdir: str | None = None, report_tag: str | None = None) -> dict[str, Path]:
    if scheme4_family:
        report_dir = DEFAULT_ENTIRE_DATA_REPORTS_DIR / "scheme4" / scheme4_family
    elif scheme5_family:
        report_dir = DEFAULT_ENTIRE_DATA_REPORTS_DIR / "scheme5" / scheme5_family
        report_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{report_tag}_" if report_tag else ""
        return {
            "compare": report_dir / f"{prefix}compare_wrong_predictions_entire_data.xlsx",
            "tolerance": report_dir / f"{prefix}compare_wrong_predictions_tolerance_report_entire_data.xlsx",
            "important_failures": report_dir / f"{prefix}compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx",
        }
    if report_subdir:
        report_dir = DEFAULT_ENTIRE_DATA_REPORTS_DIR / report_subdir
        report_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{report_tag}_" if report_tag else ""
        return {
            "compare": report_dir / f"{prefix}compare_wrong_predictions_entire_data.xlsx",
            "tolerance": report_dir / f"{prefix}compare_wrong_predictions_tolerance_report_entire_data.xlsx",
            "important_failures": report_dir / f"{prefix}compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx",
        }
    DEFAULT_ENTIRE_DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "compare": DEFAULT_ENTIRE_DATA_COMPARE_OUTPUT_PATH,
        "tolerance": DEFAULT_ENTIRE_DATA_TOLERANCE_OUTPUT_PATH,
        "important_failures": DEFAULT_ENTIRE_DATA_IMPORTANT_FAILURES_OUTPUT_PATH,
    }


def parse_named_paths(values):
    named_paths = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got: {value}")
        name, raw_path = value.split("=", 1)
        named_paths[name] = Path(raw_path).expanduser()
    return named_paths


def parse_named_files(values):
    named_files = {}
    for index, value in enumerate(values, start=1):
        if "=" in value:
            name, raw_path = value.split("=", 1)
        else:
            raw_path = value
            name = Path(raw_path).stem or f"wrong_predictions_file_{index:02d}"
        named_files[name] = Path(raw_path).expanduser()
    return named_files


def discover_wrong_predictions_files(
    root: str | Path,
    pattern: str = ENTIRE_DATA_WRONG_PREDICTIONS_GLOB,
    analysis_tag: str | None = None,
    prediction_tag: str | None = None,
    scheme4_family: str | None = None,
    scheme5_family: str | None = None,
    model_family: str | None = None,
    experiment_token: str | None = None,
) -> dict[str, Path]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"Wrong-predictions root not found: {root_path}")

    candidate_paths = sorted(root_path.rglob(pattern))

    print()
    print("=== WRONG-PREDICTION DISCOVERY ===")
    print(f"Root:         {root_path}")
    print(f"Pattern:      {pattern}")
    print(f"Analysis tag: {analysis_tag if analysis_tag else '<none>'}")
    print(f"Prediction tag: {prediction_tag if prediction_tag else '<none>'}")
    print(f"Candidates before tag filtering: {len(candidate_paths)}")

    if analysis_tag and prediction_tag:
        exact_generation_token = f"_{analysis_tag}_{prediction_tag}_predict_inference_"
        candidate_paths = [
            path for path in candidate_paths
            if exact_generation_token.lower() in path.name.lower()
        ]
        print(f"Exact generation token: {exact_generation_token}")
    elif analysis_tag:
        candidate_paths = [path for path in candidate_paths if analysis_tag in path.name]
    elif prediction_tag:
        candidate_paths = [
            path for path in candidate_paths
            if f"_{prediction_tag}_predict_inference_".lower() in path.name.lower()
        ]

    print(f"Candidates after tag filtering: {len(candidate_paths)}")

    if experiment_token:
        candidate_paths = [
            path for path in candidate_paths
            if experiment_token.lower() in path.name.lower()
        ]
        print(f"Candidates after experiment-token filtering ({experiment_token}): {len(candidate_paths)}")

    if scheme4_family:
        family_token = f"scheme4_{scheme4_family}_".lower()
        candidate_paths = [path for path in candidate_paths if family_token in path.name.lower()]
        print(f"Candidates after Scheme 4 family filtering ({scheme4_family}): {len(candidate_paths)}")
    if scheme5_family:
        family_token = f"scheme5_{scheme5_family}_".lower()
        candidate_paths = [path for path in candidate_paths if family_token in path.name.lower()]
        print(f"Candidates after Scheme 5 family filtering ({scheme5_family}): {len(candidate_paths)}")


    if model_family:
        model_token = "_random_forest_" if model_family == "random_forest" else "_xgboost_"
        candidate_paths = [path for path in candidate_paths if model_token in path.name.lower()]
        print(f"Candidates after model-family filtering ({model_family}): {len(candidate_paths)}")
    discovered = {}
    for path in candidate_paths:
        key = str(path.relative_to(root_path))
        discovered[key] = path
        print(f"  {path}")

    print("==================================")
    print()

    if not discovered:
        tag_message = f" containing analysis tag {analysis_tag!r}" if analysis_tag else ""
        raise FileNotFoundError(
            f"No wrong-predictions files matching {pattern!r}{tag_message} were found under {root_path}."
        )

    return discovered


def validate_discovered_experiment_files(
    wrong_predictions_files: dict[str, Path],
) -> None:
    experiment_groups = {}

    for path in wrong_predictions_files.values():
        name = path.name.lower()

        if "scheme1" in name:
            scheme = "scheme1"
        elif "scheme2" in name:
            scheme = "scheme2"
        elif "scheme3" in name:
            scheme = "scheme3"
        elif "scheme4" in name:
            scheme = "scheme4"
        elif "scheme5" in name:
            scheme = "scheme5"
        else:
            scheme = "unknown"

        if "random_forest" in name:
            model = "random_forest"
        elif "xgboost" in name:
            model = "xgboost"
        else:
            model = "unknown"

        if "strat_y_k" in name:
            split_mode = "strat_y_K"
        elif "strat_y" in name:
            split_mode = "strat_y"
        else:
            split_mode = "unknown"

        if "inference_test_data" in name:
            scope = "test_data"
        elif "inference_entire_data" in name:
            scope = "entire_data"
        else:
            scope = "unknown"

        key = (scheme, model, split_mode, scope)
        experiment_groups.setdefault(key, []).append(path)

    duplicates = {
        key: paths for key, paths in experiment_groups.items() if len(paths) > 1
    }

    if duplicates:
        print()
        print("DUPLICATE EXPERIMENT FILES DETECTED")
        for key, paths in duplicates.items():
            print(f"\nExperiment: {key}")
            for path in paths:
                print(f"  {path}")
        raise ValueError(
            "Multiple wrong-prediction files were discovered for the same "
            "scheme/model/split/scope. Restrict discovery using --analysis-tag "
            "or remove/archive obsolete outputs."
        )


def collect_rows_meeting_threshold(
    model_folder: str,
    filename_pattern: str = "wrong_predictions_",
    threshold: float = 0.5,
    sheet_name: str = "Sheet1",
) -> pd.DataFrame:
    collected = []
    template_df = None

    for fname in sorted(os.listdir(model_folder)):
        if not (fname.startswith(filename_pattern) and fname.endswith(".xlsx")):
            continue

        path = os.path.join(model_folder, fname)
        print(f"Reading {path}...")

        df = pd.read_excel(path, sheet_name=sheet_name)
        template_df = df.copy()
        df = ensure_time_gap_ratio(df)

        mask = df["time_gap_ratio"] >= threshold
        if mask.any():
            temp = df.loc[mask].copy()
            temp["filename"] = fname
            collected.append(temp)
            print(f"  -> Collected {mask.sum()} rows >= {threshold}")
        else:
            print(f"  -> No rows meet {threshold}")

    if collected:
        collected_df = pd.concat(collected, ignore_index=True)
        return collected_df.sort_values(by="dataset")

    if template_df is not None:
        empty = template_df.iloc[0:0].copy()
        empty["filename"] = pd.Series(dtype="object")
        return empty

    return pd.DataFrame(columns=["filename"])


def collect_rows_from_wrong_predictions_file(
    wrong_predictions_path: str | Path,
    threshold: float = 0.5,
    sheet_name: str = "Sheet1",
) -> pd.DataFrame:
    path = Path(wrong_predictions_path)
    if not path.exists():
        raise FileNotFoundError(f"Wrong-predictions file not found: {path}")

    print(f"Reading {path}...")
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = ensure_time_gap_ratio(df)

    mask = df["time_gap_ratio"] >= threshold
    if not mask.any():
        print(f"  -> No rows meet {threshold}")
        empty = df.iloc[0:0].copy()
        empty["filename"] = pd.Series(dtype="object")
        return empty

    filtered = df.loc[mask].copy()
    filtered["filename"] = path.name
    print(f"  -> Collected {mask.sum()} rows >= {threshold}")
    return filtered.sort_values(by="dataset")


def derive_failure_severity(df: pd.DataFrame) -> pd.Series:
    bins = [-float("inf"), 0.05, 0.20, 1.0, float("inf")]
    labels = ["\u22645%", "5\u201320%", "20\u2013100%", ">100%"]
    severity = pd.cut(df["time_gap_ratio"], bins=bins, labels=labels, include_lowest=True)
    return severity.astype("object").fillna("")


def get_algorithm_columns(df: pd.DataFrame) -> list[str]:
    algorithms = []
    for col in df.columns:
        if not col.endswith("-bnb"):
            continue
        if f"prob_{col}" not in df.columns:
            continue
        algorithms.append(col)
    return algorithms


def _build_ranked_algorithm_predictions(
    row: pd.Series,
    algorithms: list[str],
    fastest_time_value,
) -> list[tuple[str, float, float | None]]:
    ranked = []
    for alg in algorithms:
        probability = _safe_float(row.get(f"prob_{alg}"))
        runtime = _safe_float(row.get(alg))
        if probability is None:
            continue

        time_ratio = _compute_ratio(runtime, fastest_time_value) if runtime is not None else pd.NA
        ranked.append((alg, probability, None if time_ratio is pd.NA else time_ratio))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:3]


def _recompute_prediction_details_for_row(row: pd.Series, algorithms: list[str]) -> dict[str, object]:
    fastest_time = _safe_float(row.get("fastest_time", row.get("fastest_runtime")))
    ranked = _build_ranked_algorithm_predictions(row, algorithms, fastest_time)

    predicted_algo = row.get("predicted")
    predicted_probability = pd.NA
    predicted_time = pd.NA

    if ranked:
        predicted_algo = ranked[0][0]
        predicted_probability = ranked[0][1]
        predicted_time_value = _safe_float(row.get(predicted_algo))
        if predicted_time_value is not None:
            predicted_time = predicted_time_value
    elif predicted_algo in algorithms:
        predicted_probability_value = _safe_float(row.get(f"prob_{predicted_algo}"))
        predicted_time_value = _safe_float(row.get(predicted_algo))
        if predicted_probability_value is not None:
            predicted_probability = predicted_probability_value
        if predicted_time_value is not None:
            predicted_time = predicted_time_value

    if fastest_time is None or predicted_time is pd.NA:
        time_gap = pd.NA
        time_gap_ratio = pd.NA
        time_ratio = pd.NA
    else:
        time_gap = predicted_time - fastest_time
        time_gap_ratio = _compute_ratio(time_gap, fastest_time)
        time_ratio = _compute_ratio(predicted_time, fastest_time)

    return {
        "predicted_algo": predicted_algo,
        "predicted_probability": predicted_probability,
        "predicted_time": predicted_time,
        "time_gap": time_gap,
        "time_gap_ratio": time_gap_ratio,
        "time_ratio": time_ratio,
        "top3_predicted_detailed": ranked,
    }


def build_prediction_consistency_diagnostics(
    df: pd.DataFrame,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    diagnostics = []

    for row_idx, row in df.iterrows():
        top3 = _ensure_list(row.get("top3_predicted_detailed"))
        top1_algo = _extract_top3_top1(row.get("top3_predicted_detailed"))
        predicted_algo = row.get("predicted_algo")
        predicted_probability = _safe_float(row.get("predicted_probability"))
        fastest_time = _safe_float(row.get("fastest_time", row.get("fastest_runtime")))
        predicted_time = _safe_float(row.get("predicted_time"))
        stored_time_gap_ratio = _safe_float(row.get("time_gap_ratio"))

        top3_probabilities = []
        for item in top3:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            probability = _safe_float(item[1])
            if probability is None:
                continue
            top3_probabilities.append(probability)
        top3_unsorted = any(
            top3_probabilities[idx] < top3_probabilities[idx + 1] - tolerance
            for idx in range(len(top3_probabilities) - 1)
        )

        expected_time_gap_ratio = pd.NA
        if fastest_time is not None and predicted_time is not None:
            expected_time_gap_ratio = _compute_ratio(predicted_time - fastest_time, fastest_time)

        time_gap_ratio_mismatch = False
        if stored_time_gap_ratio is not None:
            if expected_time_gap_ratio is pd.NA:
                time_gap_ratio_mismatch = True
            else:
                time_gap_ratio_mismatch = abs(stored_time_gap_ratio - expected_time_gap_ratio) > tolerance

        expected_predicted_probability = None
        if predicted_algo and f"prob_{predicted_algo}" in row.index:
            expected_predicted_probability = _safe_float(row.get(f"prob_{predicted_algo}"))

        predicted_probability_mismatch = False
        if predicted_probability is not None and expected_predicted_probability is not None:
            predicted_probability_mismatch = abs(predicted_probability - expected_predicted_probability) > tolerance

        top1_mismatch = bool(predicted_algo) and bool(top1_algo) and predicted_algo != top1_algo
        if not (top1_mismatch or top3_unsorted or time_gap_ratio_mismatch or predicted_probability_mismatch):
            continue

        diagnostics.append(
            {
                "row_index": row_idx,
                "dataset": row.get("dataset"),
                "scheme": row.get("scheme", ""),
                "model": row.get("model", ""),
                "source_model": row.get("source_model", ""),
                "inference_on": row.get("inference_on", ""),
                "predicted_algo": predicted_algo,
                "top1_algo": top1_algo,
                "predicted_probability": predicted_probability,
                "expected_predicted_probability": expected_predicted_probability,
                "fastest_time": fastest_time,
                "predicted_time": predicted_time,
                "time_gap_ratio": stored_time_gap_ratio,
                "expected_time_gap_ratio": expected_time_gap_ratio,
                "top3_predicted_detailed": row.get("top3_predicted_detailed"),
                "anomaly_top1_mismatch": top1_mismatch,
                "anomaly_top3_unsorted": top3_unsorted,
                "anomaly_time_gap_ratio_mismatch": time_gap_ratio_mismatch,
                "anomaly_predicted_probability_mismatch": predicted_probability_mismatch,
            }
        )

    return pd.DataFrame(diagnostics)


def validate_prediction_consistency(df: pd.DataFrame, tolerance: float = 1e-6) -> dict[str, int]:
    diagnostics = build_prediction_consistency_diagnostics(df, tolerance=tolerance)
    if diagnostics.empty:
        return {
            "top1_mismatch_count": 0,
            "top3_unsorted_count": 0,
            "time_gap_ratio_mismatch_count": 0,
            "predicted_probability_mismatch_count": 0,
        }

    return {
        "top1_mismatch_count": int(diagnostics["anomaly_top1_mismatch"].sum()),
        "top3_unsorted_count": int(diagnostics["anomaly_top3_unsorted"].sum()),
        "time_gap_ratio_mismatch_count": int(diagnostics["anomaly_time_gap_ratio_mismatch"].sum()),
        "predicted_probability_mismatch_count": int(diagnostics["anomaly_predicted_probability_mismatch"].sum()),
    }


def add_prediction_detail_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    algorithms = get_algorithm_columns(df)

    if "fastest_algorithms" in df.columns:
        df["fastest_algorithms"] = df["fastest_algorithms"].apply(_ensure_list)
    else:
        df["fastest_algorithms"] = [[] for _ in range(len(df))]

    if algorithms and not df["fastest_algorithms"].map(bool).all():
        df["fastest_algorithms"] = df.apply(
            lambda row: [alg for alg in algorithms if row.get(alg) == row[algorithms].min()],
            axis=1,
        )

    if "fastest_time" not in df.columns and algorithms:
        df["fastest_time"] = df[algorithms].min(axis=1)

    df["fastest_algo"] = df["fastest_algorithms"].apply(lambda value: value[0] if value else "")
    df["fastest_runtime"] = df.get("fastest_time", pd.Series([pd.NA] * len(df), index=df.index))

    if algorithms:
        recomputed_columns = df.apply(
            lambda row: pd.Series(
                _recompute_prediction_details_for_row(row, algorithms),
                index=[
                    "predicted_algo",
                    "predicted_probability",
                    "predicted_time",
                    "time_gap",
                    "time_gap_ratio",
                    "time_ratio",
                    "top3_predicted_detailed",
                ],
            ),
            axis=1,
        )
        for column in recomputed_columns.columns:
            df[column] = recomputed_columns[column]
    else:
        df["predicted_algo"] = df.get("predicted", pd.Series([pd.NA] * len(df), index=df.index))
        df["predicted_probability"] = pd.NA
        df["top3_predicted_detailed"] = [[] for _ in range(len(df))]

    validation = validate_prediction_consistency(df)
    if validation["top1_mismatch_count"] or validation["top3_unsorted_count"]:
        raise ValueError(
            "Prediction-detail validation failed after recomputing reporting fields: "
            f"{validation}"
        )

    return df


def build_model_comparison_dataframe(
    model_root: Path,
    threshold: float,
    filename_pattern: str,
    sheet_name: str,
) -> pd.DataFrame:
    master_dict = {}

    for strategy_name in STRATEGY_FOLDERS:
        strategy_folder = model_root / strategy_name
        if not strategy_folder.is_dir():
            continue

        strategy_df = collect_rows_meeting_threshold(
            model_folder=str(strategy_folder),
            filename_pattern=filename_pattern,
            threshold=threshold,
            sheet_name=sheet_name,
        )
        if strategy_df.empty:
            continue

        strategy_df = add_prediction_detail_columns(strategy_df)
        master_dict[strategy_name] = extract_filename_info(strategy_df)

    if not master_dict:
        return pd.DataFrame()

    combined_df = aggregate_master_dataframes(master_dict, output_dir=str(model_root), save_files=False)
    return combined_df.rename(columns={"source_model": "strategy"})


def infer_strategy_from_filename(filename: str) -> str:
    if "_with_holdout_" in filename:
        return "split_and_expand"
    if "_without_holdout_" in filename:
        return "expand_and_split"
    if "predict_inference_test_data" in filename or "eval_inference_test_data" in filename:
        return "split_and_expand"
    return "expand_and_split"


def derive_scheme_and_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    metadata_source = df.get("filename", pd.Series([""] * len(df), index=df.index)).fillna("")
    if "source_model" in df.columns:
        metadata_source = metadata_source.where(
            metadata_source.astype(str).str.len() > 0,
            df["source_model"].fillna("").astype(str),
        )

    extracted = metadata_source.astype(str).str.extract(r"(scheme[12345])_(?:(?:oriented|forward|b|other)_)?(random_forest|xgboost)")
    df["scheme"] = extracted[0].map(
        {
            "scheme1": "scheme 1",
            "scheme2": "scheme 2",
            "scheme3": "scheme 3",
            "scheme4": "scheme 4",
            "scheme5": "scheme 5",
        }
    ).fillna("")
    df["model"] = extracted[1].map(
        {
            "random_forest": "RF",
            "xgboost": "XGB",
        }
    ).fillna("")
    return df


def finalize_comparison_dataframe(combined_df: pd.DataFrame) -> pd.DataFrame:
    combined_df = combined_df.copy()
    combined_df["with_holdout"] = combined_df["strategy"].eq("split_and_expand")
    combined_df["failure_severity"] = derive_failure_severity(combined_df)
    combined_df = derive_scheme_and_model_columns(combined_df)

    probability_cols = sorted([col for col in combined_df.columns if col.startswith("prob_")])

    retained_cols = [col for col in FINAL_COLUMN_ORDER if col in combined_df.columns]
    retained_cols.extend(col for col in probability_cols if col not in retained_cols)

    combined_df = combined_df[retained_cols]
    combined_df = combined_df.sort_values(
        by=["inference_on", "with_holdout", "strat_on", "k_value", "dataset", "source_model"],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)
    return combined_df


def normalize_direct_file_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "filename" not in df.columns:
        df["filename"] = ""

    if "strategy" not in df.columns:
        df["strategy"] = df["filename"].apply(infer_strategy_from_filename)

    missing_inference_mask = ~df["filename"].str.contains(
        r"_(?:inference_)?(?:entire_data|test_data)(?:_|\.|$)",
        regex=True,
    )
    if missing_inference_mask.any():
        df.loc[missing_inference_mask, "filename"] = df.loc[missing_inference_mask, "filename"].apply(
            lambda value: value.replace(
                "wrong_predictions_",
                "wrong_predictions_inference_entire_data_without_holdout_",
                1,
            )
            if "wrong_predictions_" in value
            else value
        )

    return extract_filename_info(df)


def build_file_comparison_dataframe(
    wrong_predictions_path: Path,
    threshold: float,
    sheet_name: str,
) -> pd.DataFrame:
    filtered_df = collect_rows_from_wrong_predictions_file(
        wrong_predictions_path=wrong_predictions_path,
        threshold=threshold,
        sheet_name=sheet_name,
    )
    if filtered_df.empty:
        return filtered_df

    filtered_df = add_prediction_detail_columns(filtered_df)
    return normalize_direct_file_metadata(filtered_df)


def build_compare_wrong_predictions_dataframe(
    model_roots: dict[str, Path],
    threshold: float,
    filename_pattern: str,
    sheet_name: str,
) -> pd.DataFrame:
    master_dict = {}

    for model_name, model_root in model_roots.items():
        combined_df = build_model_comparison_dataframe(
            model_root=model_root,
            threshold=threshold,
            filename_pattern=filename_pattern,
            sheet_name=sheet_name,
        )
        if combined_df.empty:
            print(f"Skipping {model_name}: no matching wrong prediction files found.")
            continue

        master_dict[model_name] = combined_df

    if not master_dict:
        raise FileNotFoundError("No wrong prediction Excel files were found in the supplied model roots.")

    combined_df = aggregate_master_dataframes(
        master_dict,
        output_dir="restructured_pipeline/outputs/reports",
        save_files=False,
    )
    return finalize_comparison_dataframe(combined_df)


def build_compare_wrong_predictions_from_files(
    wrong_predictions_files: dict[str, Path],
    threshold: float,
    sheet_name: str,
) -> pd.DataFrame:
    master_dict = {}

    for model_name, wrong_predictions_path in wrong_predictions_files.items():
        combined_df = build_file_comparison_dataframe(
            wrong_predictions_path=wrong_predictions_path,
            threshold=threshold,
            sheet_name=sheet_name,
        )
        if combined_df.empty:
            print(f"Skipping {model_name}: no rows met the threshold.")
            continue
        master_dict[model_name] = combined_df

    if not master_dict:
        raise FileNotFoundError("No rows meeting the threshold were found in the supplied wrong-predictions files.")

    combined_df = aggregate_master_dataframes(
        master_dict,
        output_dir="restructured_pipeline/outputs/reports",
        save_files=False,
    )
    if "strategy" not in combined_df.columns:
        combined_df["strategy"] = combined_df["filename"].apply(infer_strategy_from_filename)
    return finalize_comparison_dataframe(combined_df)


def resolve_output_path(
    output_path: str | None,
    model_roots: dict[str, Path],
    wrong_predictions_files: dict[str, Path],
) -> Path:
    if output_path:
        return Path(output_path)

    if wrong_predictions_files and not model_roots:
        if len(wrong_predictions_files) == 1:
            only_path = next(iter(wrong_predictions_files.values()))
            stem = only_path.stem
            if stem.startswith("wrong_predictions_"):
                stem = stem[len("wrong_predictions_"):]
            return Path("restructured_pipeline/outputs/reports") / f"compare_wrong_predictions_{stem}.xlsx"
        return Path("restructured_pipeline/outputs/reports/compare_wrong_predictions_from_files.xlsx")

    return DEFAULT_OUTPUT_PATH


def resolve_tolerance_output_path(output_path: str | None) -> Path:
    if output_path:
        return Path(output_path)
    return DEFAULT_TOLERANCE_REPORT_OUTPUT_PATH


def resolve_tolerance_important_failures_input_path(input_path: str | None = None) -> Path:
    if input_path:
        return Path(input_path)
    for candidate in DEFAULT_TOLERANCE_IMPORTANT_FAILURES_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_TOLERANCE_IMPORTANT_FAILURES_INPUT_CANDIDATES[0]


def resolve_tolerance_important_failures_output_path(output_path: str | None = None) -> Path:
    if output_path:
        return Path(output_path)
    return DEFAULT_TOLERANCE_IMPORTANT_FAILURES_OUTPUT_PATH


def build_combined_dataframe_from_args(args, threshold: float) -> pd.DataFrame:
    model_roots = parse_named_paths(args.model_root or [])
    wrong_predictions_files = parse_named_files(args.wrong_predictions_file or [])
    if args.discover_entire_data_files:
        wrong_predictions_files.update(
            discover_wrong_predictions_files(
                args.wrong_cases_root,
                analysis_tag=getattr(args, "analysis_tag", None),
                prediction_tag=getattr(args, "prediction_tag", None),
                scheme4_family=getattr(args, "scheme4_family", None),
                scheme5_family=getattr(args, "scheme5_family", None),
                model_family=getattr(args, "model_family", None),
                experiment_token=getattr(args, "experiment_token", None),
            )
        )

    if wrong_predictions_files:
        return build_compare_wrong_predictions_from_files(
            wrong_predictions_files=wrong_predictions_files,
            threshold=threshold,
            sheet_name=args.sheet_name,
        )

    return build_compare_wrong_predictions_dataframe(
        model_roots=model_roots,
        threshold=threshold,
        filename_pattern=args.filename_pattern,
        sheet_name=args.sheet_name,
    )


def build_pipeline_dataframe_from_saved_files(
    args,
    pattern: str,
    threshold: float,
) -> pd.DataFrame:
    wrong_predictions_files = discover_wrong_predictions_files(
        args.wrong_cases_root,
        pattern=pattern,
        analysis_tag=getattr(args, "analysis_tag", None),
        prediction_tag=getattr(args, "prediction_tag", None),
        scheme4_family=getattr(args, "scheme4_family", None),
        scheme5_family=getattr(args, "scheme5_family", None),
        model_family=getattr(args, "model_family", None),
        experiment_token=getattr(args, "experiment_token", None),
    )
    validate_discovered_experiment_files(wrong_predictions_files)
    print(f"[pipeline discovery] selected_files={len(wrong_predictions_files)}")
    return build_compare_wrong_predictions_from_files(
        wrong_predictions_files=wrong_predictions_files,
        threshold=threshold,
        sheet_name=args.sheet_name,
    )


def generate_tolerance_reports(
    df: pd.DataFrame,
    output_path: str | Path,
    tolerance_values: tuple[int, ...] = DEFAULT_TOLERANCE_VALUES,
    time_gap_highlight_threshold: float = 100.0,
) -> Path:
    return generate_consolidated_tolerance_report(
        df=df,
        output_path=output_path,
        tolerance_values=tolerance_values,
        time_gap_highlight_threshold=time_gap_highlight_threshold,
    )


def generate_consolidated_tolerance_report(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    tolerance_values: tuple[int, ...] = DEFAULT_TOLERANCE_VALUES,
    time_gap_highlight_threshold: float = 100.0,
) -> Path:
    tolerance_source = df.copy()
    if tolerance_source.empty:
        raise ValueError("Cannot generate consolidated tolerance report from an empty dataframe.")

    tolerance_source["fastest_runtime"] = pd.to_numeric(tolerance_source["fastest_runtime"], errors="coerce")
    tolerance_source["predicted_time"] = pd.to_numeric(tolerance_source["predicted_time"], errors="coerce")
    _print_stage_summary("tolerance_base_dataset", tolerance_source)

    sheet_frames: dict[str, pd.DataFrame] = {}
    for scheme_number in (1, 2, 3, 4, 5):
        scheme_label = f"scheme {scheme_number}"
        scheme_df = tolerance_source[tolerance_source["scheme"] == scheme_label].copy()
        for tolerance in tolerance_values:
            tolerance_df = scheme_df.copy()
            tolerance_factor = 1.0 + (float(tolerance) / 100.0)
            tolerance_mask = tolerance_df["predicted_time"] > (tolerance_df["fastest_runtime"] * tolerance_factor)
            tolerance_df = tolerance_df.loc[tolerance_mask].copy()
            tolerance_df = _ensure_columns(tolerance_df, TOLERANCE_REPORT_COLUMNS)
            tolerance_df = tolerance_df.sort_values(
                by=["inference_on", "strat_on", "k_value", "dataset", "model"],
                ascending=[True, True, True, True, True],
            ).reset_index(drop=True)
            sheet_name = f"scheme_{scheme_number}_tol_{tolerance}"
            sheet_frames[sheet_name] = tolerance_df
            print(
                f"[tolerance_report] sheet={sheet_name} tolerance={tolerance}% rows={len(tolerance_df)}"
            )

    validate_consolidated_tolerance_report(sheet_frames)
    output = resolve_tolerance_output_path(str(output_path) if output_path is not None else None)
    _write_formatted_named_sheets(
        sheet_frames,
        output_path=output,
        time_gap_highlight_threshold=time_gap_highlight_threshold,
    )
    print(f"Wrote tolerance report to '{output}' with sheets: {list(sheet_frames.keys())}")
    return output


def filter_tolerance_sheet_for_important_failures(
    df: pd.DataFrame,
    small_runtime_threshold: float = DEFAULT_SMALL_RUNTIME_THRESHOLD,
    slowdown_ratio_threshold: float = DEFAULT_SLOWDOWN_RATIO_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, object]]:
    filtered_df = df.copy()
    fastest_col = _resolve_runtime_column(filtered_df.columns, "fastest_runtime", "fastest_time")
    predicted_col = _resolve_runtime_column(filtered_df.columns, "predicted_runtime", "predicted_time")

    fastest = pd.to_numeric(filtered_df[fastest_col], errors="coerce")
    predicted = pd.to_numeric(filtered_df[predicted_col], errors="coerce")

    excluded_small_runtime = fastest.lt(small_runtime_threshold) & predicted.lt(small_runtime_threshold)
    excluded_near_optimal = predicted.le(slowdown_ratio_threshold * fastest)
    keep_mask = ~(excluded_small_runtime | excluded_near_optimal)

    summary = {
        "runtime_fastest_column": fastest_col,
        "runtime_predicted_column": predicted_col,
        "total_rows": int(len(filtered_df)),
        "small_runtime_threshold": float(small_runtime_threshold),
        "slowdown_ratio_threshold": float(slowdown_ratio_threshold),
        "excluded_small_runtime_count": int(excluded_small_runtime.sum()),
        "excluded_near_optimal_count": int(excluded_near_optimal.sum()),
        "excluded_by_both_rules_count": int((excluded_small_runtime & excluded_near_optimal).sum()),
        "remaining_important_failures": int(keep_mask.sum()),
    }
    return filtered_df.loc[keep_mask].copy(), summary


def generate_tolerance_important_failures_report(
    input_tolerance_report: str | Path,
    output_path: str | Path | None = None,
    small_runtime_threshold: float = DEFAULT_SMALL_RUNTIME_THRESHOLD,
    slowdown_ratio_threshold: float = DEFAULT_SLOWDOWN_RATIO_THRESHOLD,
    time_gap_highlight_threshold: float = 100.0,
    summary_sheet_name: str = DEFAULT_IMPORTANT_FAILURE_SUMMARY_SHEET,
) -> dict[str, object]:
    input_path = Path(input_tolerance_report)
    if not input_path.exists():
        raise FileNotFoundError(f"Tolerance report not found: {input_path}")

    workbook_sheets = pd.read_excel(input_path, sheet_name=None)
    sheet_frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []

    for sheet_name, sheet_df in workbook_sheets.items():
        if not str(sheet_name).startswith("scheme_"):
            continue
        important_df, summary = filter_tolerance_sheet_for_important_failures(
            sheet_df,
            small_runtime_threshold=small_runtime_threshold,
            slowdown_ratio_threshold=slowdown_ratio_threshold,
        )
        sheet_frames[sheet_name] = important_df
        summary_rows.append({"sheet_name": sheet_name, **summary})
        print(
            "[important_failure_report] "
            f"sheet={sheet_name} total_rows={summary['total_rows']} "
            f"remaining={summary['remaining_important_failures']} "
            f"small_runtime_excluded={summary['excluded_small_runtime_count']} "
            f"near_optimal_excluded={summary['excluded_near_optimal_count']}"
        )

    if not sheet_frames:
        raise ValueError("The input tolerance workbook did not contain any scheme tolerance sheets.")

    resolved_output = resolve_tolerance_important_failures_output_path(
        str(output_path) if output_path is not None else None
    )
    _write_formatted_named_sheets(
        sheet_frames,
        output_path=resolved_output,
        time_gap_highlight_threshold=time_gap_highlight_threshold,
    )

    summary_df = pd.DataFrame(summary_rows)
    total_row = {
        "sheet_name": "ALL",
        "runtime_fastest_column": "",
        "runtime_predicted_column": "",
        "total_rows": int(summary_df["total_rows"].sum()),
        "small_runtime_threshold": float(small_runtime_threshold),
        "slowdown_ratio_threshold": float(slowdown_ratio_threshold),
        "excluded_small_runtime_count": int(summary_df["excluded_small_runtime_count"].sum()),
        "excluded_near_optimal_count": int(summary_df["excluded_near_optimal_count"].sum()),
        "excluded_by_both_rules_count": int(summary_df["excluded_by_both_rules_count"].sum()),
        "remaining_important_failures": int(summary_df["remaining_important_failures"].sum()),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
    _write_summary_sheet(resolved_output, summary_df, summary_sheet_name)
    print(
        "[important_failure_report] "
        f"total_rows={total_row['total_rows']} "
        f"remaining={total_row['remaining_important_failures']} "
        f"small_runtime_excluded={total_row['excluded_small_runtime_count']} "
        f"near_optimal_excluded={total_row['excluded_near_optimal_count']}"
    )

    return {
        "input_tolerance_report": str(input_path),
        "output_path": str(resolved_output),
        "sheet_count": len(sheet_frames),
        "summary_sheet_name": summary_sheet_name,
        "total_rows": int(total_row["total_rows"]),
        "excluded_small_runtime_count": int(total_row["excluded_small_runtime_count"]),
        "excluded_near_optimal_count": int(total_row["excluded_near_optimal_count"]),
        "excluded_by_both_rules_count": int(total_row["excluded_by_both_rules_count"]),
        "remaining_important_failures": int(total_row["remaining_important_failures"]),
    }


def generate_compare_test_data(args) -> tuple[pd.DataFrame, Path]:
    combined_df = build_pipeline_dataframe_from_saved_files(
        args,
        pattern=TEST_DATA_WRONG_PREDICTIONS_GLOB,
        threshold=args.threshold,
    )
    validate_test_data_rows(combined_df, "test_data_compare")
    output_paths = resolve_test_data_output_paths(getattr(args, "scheme4_family", None), getattr(args, "scheme5_family", None), getattr(args, "report_subdir", None), getattr(args, "report_tag", None))
    write_by_inference_on(
        combined_df,
        output_path=str(output_paths["compare"]),
        threshold=args.threshold,
        time_gap_highlight_threshold=args.time_gap_highlight_threshold,
    )
    print(f"[test_data_compare] output={output_paths['compare']}")
    _print_stage_summary("test_data_compare", combined_df)
    return combined_df, output_paths["compare"]


def generate_tolerance_report_test_data(args) -> Path:
    tolerance_base_df = build_pipeline_dataframe_from_saved_files(
        args,
        pattern=TEST_DATA_WRONG_PREDICTIONS_GLOB,
        threshold=0.0,
    )
    validate_test_data_rows(tolerance_base_df, "test_data_tolerance_base")
    output_paths = resolve_test_data_output_paths(getattr(args, "scheme4_family", None), getattr(args, "scheme5_family", None), getattr(args, "report_subdir", None), getattr(args, "report_tag", None))
    tolerance_output = generate_consolidated_tolerance_report(
        df=tolerance_base_df,
        output_path=output_paths["tolerance"],
        tolerance_values=DEFAULT_TOLERANCE_VALUES,
        time_gap_highlight_threshold=args.time_gap_highlight_threshold,
    )
    print(f"[test_data_tolerance] output={tolerance_output}")
    return tolerance_output


def generate_important_failures_test_data(args, tolerance_output_path: str | Path) -> dict[str, object]:
    output_paths = resolve_test_data_output_paths(getattr(args, "scheme4_family", None), getattr(args, "scheme5_family", None), getattr(args, "report_subdir", None), getattr(args, "report_tag", None))
    result = generate_tolerance_important_failures_report(
        input_tolerance_report=tolerance_output_path,
        output_path=output_paths["important_failures"],
        small_runtime_threshold=args.small_runtime_threshold,
        slowdown_ratio_threshold=args.slowdown_ratio_threshold,
        time_gap_highlight_threshold=args.time_gap_highlight_threshold,
        summary_sheet_name=args.summary_sheet_name,
    )
    print(f"[test_data_important_failures] output={result['output_path']}")
    return result


def run_test_data_pipeline(args) -> dict[str, str]:
    output_paths = resolve_test_data_output_paths(getattr(args, "scheme4_family", None), getattr(args, "scheme5_family", None), getattr(args, "report_subdir", None), getattr(args, "report_tag", None))
    _, compare_output = generate_compare_test_data(args)
    tolerance_output = generate_tolerance_report_test_data(args)
    important_result = generate_important_failures_test_data(args, tolerance_output)
    return {
        "compare_output": str(compare_output),
        "tolerance_output": str(tolerance_output),
        "important_failures_output": str(output_paths["important_failures"]),
        "important_failures_count": str(important_result["remaining_important_failures"]),
    }


def generate_compare_entire_data(args) -> tuple[pd.DataFrame, Path]:
    combined_df = build_pipeline_dataframe_from_saved_files(
        args,
        pattern=ENTIRE_DATA_WRONG_PREDICTIONS_GLOB,
        threshold=args.threshold,
    )
    validate_entire_data_rows(combined_df, "entire_data_compare")
    output_paths = resolve_entire_data_output_paths(getattr(args, "scheme4_family", None), getattr(args, "scheme5_family", None), getattr(args, "report_subdir", None), getattr(args, "report_tag", None))
    write_by_inference_on(
        combined_df,
        output_path=str(output_paths["compare"]),
        threshold=args.threshold,
        time_gap_highlight_threshold=args.time_gap_highlight_threshold,
    )
    print(f"[entire_data_compare] output={output_paths['compare']}")
    _print_stage_summary("entire_data_compare", combined_df)
    return combined_df, output_paths["compare"]


def generate_tolerance_report_entire_data(args) -> Path:
    tolerance_base_df = build_pipeline_dataframe_from_saved_files(
        args,
        pattern=ENTIRE_DATA_WRONG_PREDICTIONS_GLOB,
        threshold=0.0,
    )
    validate_entire_data_rows(tolerance_base_df, "entire_data_tolerance_base")
    output_paths = resolve_entire_data_output_paths(getattr(args, "scheme4_family", None), getattr(args, "scheme5_family", None), getattr(args, "report_subdir", None), getattr(args, "report_tag", None))
    tolerance_output = generate_consolidated_tolerance_report(
        df=tolerance_base_df,
        output_path=output_paths["tolerance"],
        tolerance_values=DEFAULT_TOLERANCE_VALUES,
        time_gap_highlight_threshold=args.time_gap_highlight_threshold,
    )
    print(f"[entire_data_tolerance] output={tolerance_output}")
    return tolerance_output


def generate_important_failures_entire_data(args, tolerance_output_path: str | Path) -> dict[str, object]:
    output_paths = resolve_entire_data_output_paths(getattr(args, "scheme4_family", None), getattr(args, "scheme5_family", None), getattr(args, "report_subdir", None), getattr(args, "report_tag", None))
    result = generate_tolerance_important_failures_report(
        input_tolerance_report=tolerance_output_path,
        output_path=output_paths["important_failures"],
        small_runtime_threshold=args.small_runtime_threshold,
        slowdown_ratio_threshold=args.slowdown_ratio_threshold,
        time_gap_highlight_threshold=args.time_gap_highlight_threshold,
        summary_sheet_name=args.summary_sheet_name,
    )
    print(f"[entire_data_important_failures] output={result['output_path']}")
    return result


def run_entire_data_pipeline(args) -> dict[str, str]:
    output_paths = resolve_entire_data_output_paths(getattr(args, "scheme4_family", None), getattr(args, "scheme5_family", None), getattr(args, "report_subdir", None), getattr(args, "report_tag", None))
    _, compare_output = generate_compare_entire_data(args)
    tolerance_output = generate_tolerance_report_entire_data(args)
    important_result = generate_important_failures_entire_data(args, tolerance_output)
    return {
        "compare_output": str(compare_output),
        "tolerance_output": str(tolerance_output),
        "important_failures_output": str(output_paths["important_failures"]),
        "important_failures_count": str(important_result["remaining_important_failures"]),
    }


def write_formatted_workbook_from_existing(
    input_workbook: str | Path,
    output_path: str | Path | None = None,
    time_gap_highlight_threshold: float = 100.0,
) -> Path:
    input_path = Path(input_workbook)
    if not input_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {input_path}")

    sheets = pd.read_excel(input_path, sheet_name=None)
    frames = []
    for _, df in sheets.items():
        if df.empty or "inference_on" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        raise ValueError("Input workbook did not contain any compare-report sheets.")

    combined_df = pd.concat(frames, ignore_index=True)
    destination = Path(output_path) if output_path else input_path
    write_by_inference_on(
        combined_df,
        output_path=str(destination),
        time_gap_highlight_threshold=time_gap_highlight_threshold,
    )
    return destination


def run_compare_mode(args) -> Path:
    model_roots = parse_named_paths(args.model_root or [])
    wrong_predictions_files = parse_named_files(args.wrong_predictions_file or [])
    if args.discover_entire_data_files:
        wrong_predictions_files.update(
            discover_wrong_predictions_files(
                args.wrong_cases_root,
                analysis_tag=getattr(args, "analysis_tag", None),
                prediction_tag=getattr(args, "prediction_tag", None),
                scheme4_family=getattr(args, "scheme4_family", None),
                scheme5_family=getattr(args, "scheme5_family", None),
                model_family=getattr(args, "model_family", None),
                experiment_token=getattr(args, "experiment_token", None),
            )
        )
        if not args.output_path:
            args.output_path = str(DEFAULT_CONSOLIDATED_ENTIRE_DATA_OUTPUT_PATH)

    combined_df = build_combined_dataframe_from_args(args, threshold=args.threshold)
    print(f"[strict_compare] threshold={args.threshold}")
    _print_stage_summary("strict_compare", combined_df)

    resolved_output_path = resolve_output_path(
        output_path=args.output_path,
        model_roots=model_roots,
        wrong_predictions_files=wrong_predictions_files,
    )
    write_by_inference_on(
        combined_df,
        output_path=str(resolved_output_path),
        threshold=args.threshold,
        time_gap_highlight_threshold=args.time_gap_highlight_threshold,
    )

    if args.generate_tolerance_report:
        tolerance_output_path = generate_tolerance_reports(
            combined_df,
            output_path=args.tolerance_output_path,
            time_gap_highlight_threshold=args.time_gap_highlight_threshold,
        )
        if args.generate_important_failure_report:
            generate_tolerance_important_failures_report(
                input_tolerance_report=tolerance_output_path,
                output_path=args.important_failure_output_path,
                small_runtime_threshold=args.small_runtime_threshold,
                slowdown_ratio_threshold=args.slowdown_ratio_threshold,
                time_gap_highlight_threshold=args.time_gap_highlight_threshold,
                summary_sheet_name=args.summary_sheet_name,
            )
    elif args.generate_important_failure_report:
        generate_tolerance_important_failures_report(
            input_tolerance_report=resolve_tolerance_important_failures_input_path(args.input_tolerance_report),
            output_path=args.important_failure_output_path,
            small_runtime_threshold=args.small_runtime_threshold,
            slowdown_ratio_threshold=args.slowdown_ratio_threshold,
            time_gap_highlight_threshold=args.time_gap_highlight_threshold,
            summary_sheet_name=args.summary_sheet_name,
        )
    return resolved_output_path


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Unified wrong-predictions comparison and report writer.")
    parser.add_argument(
        "--mode",
        default="compare",
        choices=["compare", "report"],
        help="Run full compare+report generation, or rewrite formatting on an existing workbook.",
    )
    parser.add_argument(
        "--model-root",
        action="append",
        help="Model root in NAME=PATH form. PATH should contain expand_and_split/ and/or split_and_expand/.",
    )
    parser.add_argument(
        "--wrong-predictions-file",
        action="append",
        help="Wrong-predictions workbook path, optionally in NAME=PATH form.",
    )
    parser.add_argument(
        "--discover-entire-data-files",
        action="store_true",
        help="Auto-discover wrong_predictions_*predict_inference_entire_data*.xlsx under --wrong-cases-root.",
    )
    parser.add_argument(
        "--run-test-data-pipeline",
        action="store_true",
        help="Run the compare -> tolerance -> important-failures pipeline using only held-out test-data wrong-predictions files.",
    )
    parser.add_argument(
        "--run-entire-data-pipeline",
        action="store_true",
        help="Run the compare -> tolerance -> important-failures pipeline using only entire-data wrong-predictions files.",
    )
    parser.add_argument(
        "--wrong-cases-root",
        default=str(DEFAULT_WRONG_CASES_ROOT),
        help="Root folder used with --discover-entire-data-files.",
    )
    parser.add_argument(
        "--scheme4-family",
        choices=["oriented", "forward"],
        default=None,
        help="For Scheme 4 reporting, restrict discovered files to one independent family. Do not mix oriented and forward in one compare workbook.",
    )
    parser.add_argument(
        "--scheme5-family",
        choices=["b", "other"],
        default=None,
        help="For Scheme 5 reporting, restrict discovery to the independent B or Other family.",
    )
    parser.add_argument(
        "--model-family",
        choices=["random_forest", "xgboost"],
        default=None,
        help="Optionally restrict automatic wrong-prediction discovery to one model family.",
    )
    parser.add_argument(
        "--analysis-tag",
        default=None,
        help=(
            "Optional downstream analysis tag used to restrict automatic "
            "wrong-prediction discovery, e.g. 09sep2026. Only filenames "
            "containing this tag are loaded."
        ),
    )
    parser.add_argument(
        "--prediction-tag",
        default=None,
        help=(
            "Optional prediction-generation tag. When used together with "
            "--analysis-tag, discovery requires the exact filename token "
            "_<analysis-tag>_<prediction-tag>_predict_inference_, preventing "
            "different prediction generations of the same trained run from being mixed."
        ),
    )
    parser.add_argument(
        "--experiment-token",
        default=None,
        help="Optional filename token used to isolate one weighted experiment during wrong-case discovery.",
    )
    parser.add_argument(
        "--report-subdir",
        default=None,
        help="Optional subdirectory below reports/test_data and reports/entire_data for isolated experiment outputs.",
    )
    parser.add_argument(
        "--report-tag",
        default=None,
        help="Optional filename prefix inside --report-subdir, e.g. strat_y or strat_y_K.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Excel output path for the aggregated comparison workbook.",
    )
    parser.add_argument(
        "--input-workbook",
        default=None,
        help="Existing workbook to reformat when --mode report is used.",
    )
    parser.add_argument(
        "--generate-tolerance-report",
        action="store_true",
        help="Write an additional tolerance-based wrong-predictions workbook after the normal compare report.",
    )
    parser.add_argument(
        "--tolerance-output-path",
        default=None,
        help="Excel output path for the tolerance-based wrong-predictions workbook.",
    )
    parser.add_argument(
        "--generate-important-failure-report",
        action="store_true",
        help="Write an additional second-stage important-failures workbook from a tolerance report.",
    )
    parser.add_argument(
        "--input-tolerance-report",
        default=None,
        help="Existing tolerance workbook to read when only the important-failure report should be generated.",
    )
    parser.add_argument(
        "--important-failure-output-path",
        default=None,
        help="Excel output path for the tolerance important-failures workbook.",
    )
    parser.add_argument(
        "--small-runtime-threshold",
        type=float,
        default=DEFAULT_SMALL_RUNTIME_THRESHOLD,
        help="Exclude rows where both fastest and predicted runtimes are below this value.",
    )
    parser.add_argument(
        "--slowdown-ratio-threshold",
        type=float,
        default=DEFAULT_SLOWDOWN_RATIO_THRESHOLD,
        help="Exclude rows where predicted runtime is within this multiple of the fastest runtime.",
    )
    parser.add_argument(
        "--summary-sheet-name",
        default=DEFAULT_IMPORTANT_FAILURE_SUMMARY_SHEET,
        help="Summary sheet name for the tolerance important-failures workbook.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Keep rows with time_gap_ratio >= threshold.",
    )
    parser.add_argument(
        "--time-gap-highlight-threshold",
        type=float,
        default=100.0,
        help="Highlight only time_gap cells with time_gap > this threshold.",
    )
    parser.add_argument(
        "--filename-pattern",
        default="wrong_predictions_",
        help="Filename prefix used to discover wrong prediction Excel files.",
    )
    parser.add_argument(
        "--sheet-name",
        default="Sheet1",
        help="Worksheet name to read from each wrong prediction workbook.",
    )
    return parser


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.mode == "report":
        if not args.input_workbook:
            parser.error("--input-workbook is required when --mode report is used.")
        output_path = write_formatted_workbook_from_existing(
            input_workbook=args.input_workbook,
            output_path=args.output_path,
            time_gap_highlight_threshold=args.time_gap_highlight_threshold,
        )
        print(f"Formatted workbook written to: {output_path}")
        return

    if args.run_test_data_pipeline:
        result = run_test_data_pipeline(args)
        print("Test-data wrong-predictions pipeline complete.")
        print(f"Compare workbook: {result['compare_output']}")
        print(f"Tolerance workbook: {result['tolerance_output']}")
        print(f"Important-failures workbook: {result['important_failures_output']}")
        print(f"Important failures: {result['important_failures_count']}")
        return

    if args.run_entire_data_pipeline:
        result = run_entire_data_pipeline(args)
        print("Entire-data wrong-predictions pipeline complete.")
        print(f"Compare workbook: {result['compare_output']}")
        print(f"Tolerance workbook: {result['tolerance_output']}")
        print(f"Important-failures workbook: {result['important_failures_output']}")
        print(f"Important failures: {result['important_failures_count']}")
        return

    has_compare_inputs = bool(args.model_root or args.wrong_predictions_file or args.discover_entire_data_files)
    if args.generate_important_failure_report and not has_compare_inputs and not args.generate_tolerance_report:
        result = generate_tolerance_important_failures_report(
            input_tolerance_report=resolve_tolerance_important_failures_input_path(args.input_tolerance_report),
            output_path=args.important_failure_output_path,
            small_runtime_threshold=args.small_runtime_threshold,
            slowdown_ratio_threshold=args.slowdown_ratio_threshold,
            time_gap_highlight_threshold=args.time_gap_highlight_threshold,
            summary_sheet_name=args.summary_sheet_name,
        )
        print(f"Important-failures workbook written to: {result['output_path']}")
        return

    if not args.model_root and not args.wrong_predictions_file and not args.discover_entire_data_files:
        parser.error(
            "Supply at least one --model-root, --wrong-predictions-file, or --discover-entire-data-files, "
            "or use --generate-important-failure-report with --input-tolerance-report."
        )

    run_compare_mode(args)


__all__ = [
    "DEFAULT_ENTIRE_DATA_COMPARE_OUTPUT_PATH",
    "DEFAULT_ENTIRE_DATA_IMPORTANT_FAILURES_OUTPUT_PATH",
    "DEFAULT_ENTIRE_DATA_REPORTS_DIR",
    "DEFAULT_ENTIRE_DATA_TOLERANCE_OUTPUT_PATH",
    "DEFAULT_IMPORTANT_FAILURE_SUMMARY_SHEET",
    "DEFAULT_SMALL_RUNTIME_THRESHOLD",
    "DEFAULT_SLOWDOWN_RATIO_THRESHOLD",
    "DEFAULT_TEST_DATA_COMPARE_OUTPUT_PATH",
    "DEFAULT_TEST_DATA_IMPORTANT_FAILURES_OUTPUT_PATH",
    "DEFAULT_TEST_DATA_REPORTS_DIR",
    "DEFAULT_TEST_DATA_TOLERANCE_OUTPUT_PATH",
    "DEFAULT_TOLERANCE_REPORT_OUTPUT_PATH",
    "DEFAULT_TOLERANCE_IMPORTANT_FAILURES_INPUT_CANDIDATES",
    "DEFAULT_TOLERANCE_IMPORTANT_FAILURES_OUTPUT_PATH",
    "DEFAULT_TOLERANCE_VALUES",
    "DEFAULT_CONSOLIDATED_ENTIRE_DATA_OUTPUT_PATH",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_WRONG_CASES_ROOT",
    "ENTIRE_DATA_INFERENCE_LABEL",
    "ENTIRE_DATA_WRONG_PREDICTIONS_GLOB",
    "TOLERANCE_REPORT_SHEET_NAMES",
    "TOLERANCE_REPORT_COLUMNS",
    "TEST_DATA_WRONG_PREDICTIONS_GLOB",
    "TEST_SET_INFERENCE_LABEL",
    "autosize_columns",
    "build_argument_parser",
    "build_combined_dataframe_from_args",
    "build_pipeline_dataframe_from_saved_files",
    "build_compare_wrong_predictions_dataframe",
    "build_compare_wrong_predictions_from_files",
    "build_file_comparison_dataframe",
    "build_model_comparison_dataframe",
    "build_prediction_consistency_diagnostics",
    "collect_rows_from_wrong_predictions_file",
    "collect_rows_meeting_threshold",
    "derive_failure_severity",
    "discover_wrong_predictions_files",
    "filter_tolerance_sheet_for_important_failures",
    "finalize_comparison_dataframe",
    "generate_compare_entire_data",
    "generate_compare_test_data",
    "generate_consolidated_tolerance_report",
    "generate_important_failures_entire_data",
    "generate_important_failures_test_data",
    "generate_tolerance_important_failures_report",
    "generate_tolerance_reports",
    "generate_tolerance_report_entire_data",
    "generate_tolerance_report_test_data",
    "highlight_abnormal_time_gap_cells",
    "highlight_rows_by_failure_severity",
    "main",
    "parse_named_files",
    "parse_named_paths",
    "resolve_entire_data_output_paths",
    "resolve_test_data_output_paths",
    "run_compare_mode",
    "run_entire_data_pipeline",
    "run_test_data_pipeline",
    "validate_prediction_consistency",
    "validate_discovered_experiment_files",
    "validate_consolidated_tolerance_report",
    "validate_entire_data_rows",
    "validate_test_data_rows",
    "write_by_inference_on",
    "write_formatted_workbook_from_existing",
]


if __name__ == "__main__":
    main()
