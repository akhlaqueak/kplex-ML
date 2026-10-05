"""Build failure summary workbooks and enrich important-failure artifacts.
Need to resolve the error!!!"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
 
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from restructured_pipeline.src.compare_wrong_predictions import (
    autosize_columns,
    highlight_abnormal_time_gap_cells,
    highlight_rows_by_failure_severity,
)
from restructured_pipeline.src.failure_case_filter import annotate_failure_cases
from restructured_pipeline.src.features import algorithms as ALL_ALGOS, load_excel_data


DEFAULT_ENTIRE_IMPORTANT_PATH = Path(
    "restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all_entire_data_in_one_sheet.xlsx"
)
DEFAULT_ENTIRE_IMPORTANT_CANDIDATES = [
    DEFAULT_ENTIRE_IMPORTANT_PATH,
    Path("restructured_pipeline/outputs/reports/failure_case_analysis_cleaned.xlsx"),
    Path("restructured_pipeline/outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data_fixed.xlsx"),
]
DEFAULT_ENTIRE_BASE_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data.xlsx"),
    Path("restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data_pre_consistency_fix.xlsx"),
    Path("restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_schema_entire_data.xlsx"),
]
DEFAULT_TEST_COMPARE_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/test_data/compare_wrong_predictions_test_data.xlsx"),
    Path("restructured_pipeline/outputs/reports/compare_wrong_predictions_test_data.xlsx"),
]
DEFAULT_OUTPUT_PATH = Path("restructured_pipeline/outputs/reports/failure_analysis_summary_entire_vs_test.xlsx")
DEFAULT_FAILURE_ANALYSIS_WORKBOOK_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/failure_analysis_summary_all_scheme_entire_data.xlsx"
)
DEFAULT_PROJECT_CONFIG_PATH = Path("restructured_pipeline/configs/project.yaml")
DEFAULT_FAILURE_ANALYSIS_INPUT_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data.xlsx"),
    Path("restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"),
    Path("restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_important_failures.xlsx"),
]

DEFAULT_ENTIRE_IMPORTANT_FAILURES_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"),
    Path("restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_important_failures.xlsx"),
]
DEFAULT_TEST_IMPORTANT_FAILURES_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/test_data/compare_wrong_predictions_tolerance_important_failures_test_data.xlsx"),
]
DEFAULT_DATASET_FAILURE_SUMMARY_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/failure_case_analysis_dataset_failure_summary.xlsx"
)
DEFAULT_REPEAT_FAILURE_TABLE_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/failure_case_analysis_repeat_failure_table.xlsx"
)
DEFAULT_HEATMAP_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/failure_case_analysis_dataset_scheme_heatmap.png"
)
DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_LAYERED_HIGH_CONFIDENCE_THRESHOLD = 0.9
DEFAULT_MIN_TIME_RATIO_HIGHLIGHT_THRESHOLD = 2.0
DEFAULT_TIME_GAP_HIGHLIGHT_THRESHOLD = 100.0
DEFAULT_CATASTROPHIC_TIME_RATIO_THRESHOLD = 5.0
DEFAULT_CATASTROPHIC_PREDICTED_TIME_THRESHOLD = 1000.0
DEFAULT_FILTER_SUMMARY_SHEET = "filter_summary"
DEFAULT_ENRICHMENT_MODE = "enrich"
BEST_TOP2_CANDIDATE_COL = "best_top2_candidate"
BEST_TOP3_CANDIDATE_COL = "best_top3_candidate"
BEST_TOP2_TIME_RATIO_COL = "best_top2_time_ratio"
BEST_TOP3_TIME_RATIO_COL = "best_top3_time_ratio"

SCHEMES = ("scheme 1", "scheme 2", "scheme 3")
MODELS = ("RF", "XGB")
SEVERITY_LABELS = (">100%", "20–100%")
SPLIT_MODES = ("strat_y", "strat_y_K")
DATA_SCOPES = ("test_data", "entire_data")

FAILURE_ANALYSIS_SHEET_ORDER = [
    ("scheme 1", "RF",  "strat_y",   "scheme_1_RF_strat_y"),
    ("scheme 1", "RF",  "strat_y_K", "scheme_1_RF_strat_y_K"),
    ("scheme 1", "XGB", "strat_y",   "scheme_1_XGB_strat_y"),
    ("scheme 1", "XGB", "strat_y_K", "scheme_1_XGB_strat_y_K"),

    ("scheme 3", "RF",  "strat_y",   "scheme_3_RF_strat_y"),
    ("scheme 3", "RF",  "strat_y_K", "scheme_3_RF_strat_y_K"),
    ("scheme 3", "XGB", "strat_y",   "scheme_3_XGB_strat_y"),
    ("scheme 3", "XGB", "strat_y_K", "scheme_3_XGB_strat_y_K"),
]

TOP_K_VALUES = tuple(range(1, 7))
FORBIDDEN_ANALYSIS_COLUMNS = {
    "inference_on",
    "with_holdout",
    "strat_on",
    "source_model",
    "failure_case_filter_reason",
    "failure_severity",
    "fastest_algo",
    "fastest_time",
    "predicted_algo",
    "predicted_time",
    "predicted_probability",
    "time_gap",
    "time_gap_ratio",
    "tolerance",
    "split_scope",
    "source_sheet",
    "actual_time_ratio",
    "high_confidence_wrong_prediction_flag",
    BEST_TOP2_CANDIDATE_COL,
    BEST_TOP3_CANDIDATE_COL,
    BEST_TOP2_TIME_RATIO_COL,
    BEST_TOP3_TIME_RATIO_COL,
}
SUMMARY_PROBABILITY_COLUMNS = [f"prob_{algo}" for algo in ALL_ALGOS]
 
TOP3_ENTRY_PATTERN = re.compile(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)")
GREEN_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")


def resolve_existing_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve {label} from candidates: {candidates}")


def percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(100.0 * numerator / denominator):.1f}%"


def normalize_severity_label(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return text.replace("â€“", "–")


def resolve_failure_severity_mask(df: pd.DataFrame, severity_label: str) -> pd.Series:
    return df["failure_severity"].map(normalize_severity_label).eq(severity_label)


def load_entire_important(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="on entire dataset")


def load_entire_base(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="on entire dataset")


def load_test_compare(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="on 20% test set")


def compute_important_subset_from_compare(df: pd.DataFrame) -> pd.DataFrame:
    important_df, _ = annotate_failure_cases(
        df,
        small_runtime_threshold=5.0,
        slowdown_ratio_threshold=1.2,
    )
    return important_df


def build_scheme_summary(base_df: pd.DataFrame, important_df: pd.DataFrame, scheme: str) -> dict[str, pd.DataFrame]:
    base_scheme = base_df[base_df["scheme"] == scheme].copy()
    important_scheme = important_df[important_df["scheme"] == scheme].copy()

    total_rows = int(len(base_scheme))
    important_rows = int(len(important_scheme))
    filtered_out = total_rows - important_rows

    overall = pd.DataFrame(
        [
            ("Total rows", total_rows),
            ("Important failures", important_rows),
            ("Important %", percent(important_rows, total_rows)),
            ("Filtered out", filtered_out),
            ("Filtered out %", percent(filtered_out, total_rows)),
        ],
        columns=["Metric", "Value"],
    )

    severity_rows = []
    for severity in SEVERITY_LABELS:
        count = int(resolve_failure_severity_mask(important_scheme, severity).sum())
        severity_rows.append((severity, count, percent(count, important_rows)))
    severity = pd.DataFrame(severity_rows, columns=["Severity", "Count", "%"])

    model_rows = []
    model_severity_rows = []
    for model in MODELS:
        base_model = base_scheme[base_scheme["model"] == model]
        important_model = important_scheme[important_scheme["model"] == model]
        model_total = int(len(base_model))
        model_important = int(len(important_model))
        model_rows.append((model, model_total, model_important, percent(model_important, model_total)))
        for severity_label in SEVERITY_LABELS:
            count = int(resolve_failure_severity_mask(important_model, severity_label).sum())
            model_severity_rows.append((model, severity_label, count, percent(count, model_important)))

    model_breakdown = pd.DataFrame(model_rows, columns=["Model", "Total", "Important", "%"])
    model_severity = pd.DataFrame(model_severity_rows, columns=["Model", "Severity", "Count", "%"])

    return {
        "overall": overall,
        "severity": severity,
        "model_breakdown": model_breakdown,
        "model_severity": model_severity,
        "total_rows": total_rows,
        "important_rows": important_rows,
    }


def write_section(worksheet, start_row: int, title: str, dataframe: pd.DataFrame) -> int:
    worksheet.cell(row=start_row, column=1, value=title).font = Font(bold=True)
    header_row = start_row + 1
    for column_index, column_name in enumerate(dataframe.columns, start=1):
        cell = worksheet.cell(row=header_row, column=column_index, value=column_name)
        cell.font = Font(bold=True)
    for row_offset, values in enumerate(dataframe.itertuples(index=False), start=1):
        for column_index, value in enumerate(values, start=1):
            worksheet.cell(row=header_row + row_offset, column=column_index, value=value)
    return header_row + len(dataframe) + 2


def autosize(worksheet) -> None:
    for column_cells in worksheet.columns:
        max_width = max((len(str(cell.value)) for cell in column_cells if cell.value is not None), default=0)
        worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_width + 2, 12), 32)


def build_workbook(
    entire_base_df: pd.DataFrame,
    entire_important_df: pd.DataFrame,
    test_base_df: pd.DataFrame,
    test_important_df: pd.DataFrame,
    output_path: Path,
) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)

    overview_rows = []
    datasets = [
        ("entire_data", entire_base_df, entire_important_df),
        ("test_data", test_base_df, test_important_df),
    ]

    for scheme in SCHEMES:
        scheme_number = scheme.replace("scheme ", "scheme_")
        for dataset_label, base_df, important_df in datasets:
            sheet_name = f"{scheme_number}_{dataset_label}"
            worksheet = workbook.create_sheet(title=sheet_name[:31])
            summary = build_scheme_summary(base_df, important_df, scheme)

            row = 1
            row = write_section(worksheet, row, "Section 1: Overall", summary["overall"])
            row = write_section(worksheet, row, "Section 2: Severity breakdown", summary["severity"])
            row = write_section(worksheet, row, "Section 3: Model breakdown", summary["model_breakdown"])
            write_section(worksheet, row, "Section 4: Model severity breakdown", summary["model_severity"])
            autosize(worksheet)

            overview_rows.append(
                (
                    scheme,
                    dataset_label,
                    summary["total_rows"],
                    summary["important_rows"],
                    percent(summary["important_rows"], summary["total_rows"]),
                )
            )

    overview = workbook.create_sheet(title="summary_overview")
    overview_df = pd.DataFrame(overview_rows, columns=["Scheme", "Dataset", "Total", "Important", "%"])
    write_section(overview, 1, "Summary Overview", overview_df)
    autosize(overview)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    workbook.close()


def validate_counts(
    entire_base_df: pd.DataFrame,
    entire_important_df: pd.DataFrame,
    test_base_df: pd.DataFrame,
    test_important_df: pd.DataFrame,
) -> None:
    for scheme in SCHEMES:
        entire_total = int((entire_base_df["scheme"] == scheme).sum())
        entire_important = int((entire_important_df["scheme"] == scheme).sum())
        test_total = int((test_base_df["scheme"] == scheme).sum())
        test_important = int((test_important_df["scheme"] == scheme).sum())

        if entire_important > entire_total:
            raise ValueError(f"Entire-data counts invalid for {scheme}: important exceeds total.")
        if test_important > test_total:
            raise ValueError(f"Test-data counts invalid for {scheme}: important exceeds total.")

        for model in MODELS:
            model_entire_total = int(((entire_base_df["scheme"] == scheme) & (entire_base_df["model"] == model)).sum())
            model_entire_important = int(
                ((entire_important_df["scheme"] == scheme) & (entire_important_df["model"] == model)).sum()
            )
            model_test_total = int(((test_base_df["scheme"] == scheme) & (test_base_df["model"] == model)).sum())
            model_test_important = int(
                ((test_important_df["scheme"] == scheme) & (test_important_df["model"] == model)).sum()
            )
            if model_entire_important > model_entire_total:
                raise ValueError(f"Entire-data model counts invalid for {scheme}/{model}.")
            if model_test_important > model_test_total:
                raise ValueError(f"Test-data model counts invalid for {scheme}/{model}.")


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _format_float(value: float | None) -> str:
    rounded = _round_or_none(value)
    return "None" if rounded is None else f"{rounded:.4f}"


def _format_candidate_tuple(entry: dict[str, object] | None):
    if entry is None:
        return pd.NA
    return (
        f"({_format_float(_safe_float(entry.get('predicted_time')))}, "
        f"{entry['algo']!r}, "
        f"{_format_float(_safe_float(entry.get('probability')))})"
    )


def _format_fastest_tuple(fastest_time: float | None, fastest_algo: str | None, rank: int | None):
    if fastest_algo is None:
        return pd.NA
    return f"({_format_float(fastest_time)}, {fastest_algo!r}, {rank if rank is not None else 'None'})"


def _format_all_predictions(entries: list[dict[str, object]]):
    formatted_entries = [
        (
            f"({_format_float(_safe_float(entry.get('time_ratio')))}, "
            f"{entry['algo']!r}, "
            f"{_format_float(_safe_float(entry.get('predicted_time')))}, "
            f"{_format_float(_safe_float(entry.get('probability')))})"
        )
        for entry in entries
    ]
    return "[" + ", ".join(formatted_entries) + "]"


def _build_runtime_lookup(project_config_path: Path) -> pd.DataFrame:
    with open(project_config_path, "r", encoding="utf-8") as handle:
        project_config = yaml.safe_load(handle)

    data_cfg = project_config["data"]
    runtime_df = load_excel_data(
        data_cfg["file_path"],
        timeout_replacement=data_cfg["timeout_replacement"],
        y_cols=data_cfg["raw_target_columns"],
        sep=data_cfg["sep"],
        quotechar=data_cfg["quotechar"],
        engine=data_cfg["engine"],
        rename_columns=data_cfg["rename_columns"],
    )
    lookup = runtime_df[["dataset", "k_value", *ALL_ALGOS]].copy()
    lookup["dataset"] = lookup["dataset"].astype(str)
    lookup["k_value"] = pd.to_numeric(lookup["k_value"], errors="raise").astype(int)
    if lookup.duplicated(subset=["dataset", "k_value"]).any():
        duplicates = lookup.loc[lookup.duplicated(subset=["dataset", "k_value"], keep=False), ["dataset", "k_value"]]
        raise ValueError(f"Duplicate runtime lookup keys detected: {duplicates.head(10).to_dict(orient='records')}")
    return lookup.set_index(["dataset", "k_value"])


def _resolve_runtime_row(runtime_lookup: pd.DataFrame, dataset, k_value) -> pd.Series:
    dataset_key = str(dataset)
    k_key = int(pd.to_numeric(pd.Series([k_value]), errors="raise").iloc[0])
    try:
        return runtime_lookup.loc[(dataset_key, k_key)]
    except KeyError as exc:
        raise KeyError(f"Missing runtime lookup row for dataset={dataset_key!r}, k_value={k_key}.") from exc


def _build_sorted_prediction_entries(row: pd.Series, runtime_row: pd.Series) -> list[dict[str, object]]:
    fastest_time = _safe_float(row.get("fastest_time", row.get("fastest_runtime")))
    entries: list[dict[str, object]] = []
    for algo_order, algo in enumerate(ALL_ALGOS):
        probability = _safe_float(row.get(f"prob_{algo}"))
        predicted_time = _safe_float(runtime_row.get(algo))
        time_ratio = None
        if predicted_time is not None and fastest_time not in (None, 0):
            time_ratio = predicted_time / fastest_time
        entries.append(
            {
                "algo": algo,
                "probability": probability,
                "predicted_time": predicted_time,
                "time_ratio": time_ratio,
                "algo_order": algo_order,
            }
        )

    entries.sort(
        key=lambda entry: (
            float("-inf") if entry["probability"] is None else -float(entry["probability"]),
            entry["algo_order"],
        )
    )
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
    return entries


def _select_best_topk_entry(entries: list[dict[str, object]], top_k: int) -> dict[str, object] | None:
    candidates = [entry for entry in entries[:top_k] if _safe_float(entry.get("time_ratio")) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda entry: (float(entry["time_ratio"]), int(entry["rank"])))


def _build_failure_analysis_row(row: pd.Series, runtime_lookup: pd.DataFrame) -> dict[str, object]:
    runtime_row = _resolve_runtime_row(runtime_lookup, row.get("dataset"), row.get("k_value"))
    sorted_entries = _build_sorted_prediction_entries(row, runtime_row)

    fastest_algo = row.get("fastest_algo")
    fastest_time = _safe_float(row.get("fastest_time", row.get("fastest_runtime")))
    predicted_algo = row.get("predicted_algo")
    predicted_time = _safe_float(row.get("predicted_time"))
    predicted_probability = _safe_float(row.get("predicted_probability"))
    actual_time_ratio = None
    if predicted_time is not None and fastest_time not in (None, 0):
        actual_time_ratio = predicted_time / fastest_time

    fastest_rank = next((int(entry["rank"]) for entry in sorted_entries if entry["algo"] == fastest_algo), None)

    record = {
        "dataset": row.get("dataset"),
        "k_value": int(pd.to_numeric(pd.Series([row.get('k_value')]), errors="raise").iloc[0]),
        "scheme": row.get("scheme"),
        "model": row.get("model"),
        "split_mode": row.get("split_mode"),
        "data_scope": row.get("data_scope"),
        "fastest_algo_rank": fastest_rank,
        "fastest_algo_tuple": _format_fastest_tuple(fastest_time, fastest_algo, fastest_rank),
    }

    for top_k in TOP_K_VALUES:
        best_entry = _select_best_topk_entry(sorted_entries, top_k)
        record[f"top{top_k}_min_time_ratio"] = _round_or_none(
            _safe_float(best_entry.get("time_ratio")) if best_entry is not None else None
        )
        record[f"top{top_k}_candidate"] = _format_candidate_tuple(best_entry)

    record["all_predictions_sorted_by_prob"] = _format_all_predictions(sorted_entries)
    return record


def _build_failure_analysis_sheet(df: pd.DataFrame, runtime_lookup: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        columns = [
            "dataset",
            "k_value",
            "scheme",
            "model",
            "split_mode",
            "data_scope",
            "fastest_algo_rank",
            "fastest_algo_tuple",
        ]
        for top_k in TOP_K_VALUES:
            columns.extend([f"top{top_k}_min_time_ratio", f"top{top_k}_candidate"])
        columns.append("all_predictions_sorted_by_prob")
        return pd.DataFrame(columns=columns)

    records = [_build_failure_analysis_row(row, runtime_lookup) for _, row in df.iterrows()]
    sheet_df = pd.DataFrame(records)
    return sheet_df.sort_values(
        by=[
            "data_scope",
            "top1_min_time_ratio",
            "dataset",
            "k_value",
            "fastest_algo_rank",
        ],
        ascending=[
            True,
            False,
            True,
            True,
            True,
        ],
        na_position="last",
    ).reset_index(drop=True)

def build_failure_analysis_scope_summary(
    sheet_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    rows = []

    metadata_lookup = {
        sheet_name: (
            scheme,
            model,
            split_mode,
        )
        for (
            scheme,
            model,
            split_mode,
            sheet_name,
        ) in FAILURE_ANALYSIS_SHEET_ORDER
    }

    for sheet_name, df in sheet_frames.items():
        scheme, model, split_mode = metadata_lookup[
            sheet_name
        ]

        test_count = int(
            df["data_scope"]
            .eq("test_data")
            .sum()
        )

        entire_count = int(
            df["data_scope"]
            .eq("entire_data")
            .sum()
        )

        rows.append(
            {
                "scheme": scheme,
                "model": model,
                "split_mode": split_mode,
                "sheet": sheet_name,
                "total_rows": int(len(df)),
                "test_data_failures": test_count,
                "entire_data_failures": entire_count,
            }
        )

    return pd.DataFrame(rows)

def _load_failure_dataframe(
    workbook_path: Path,
    data_scope: str,
    tolerance: int = 0,
) -> pd.DataFrame:
    """
    Load all failure rows for one evaluation scope at one tolerance.

    Supports sheet names such as:
      scheme_1_tol_0
      scheme_1_tol_0_y
      scheme_1_tol_0_y_K
      scheme_3_tol_0

    Only the requested tolerance is loaded.
    """
    sheets = pd.read_excel(
        workbook_path,
        sheet_name=None,
    )

    print(
        f"\n[LOAD FAILURE WORKBOOK] "
        f"scope={data_scope}, path={workbook_path}"
    )

    # ---------------------------------------------------------
    # Legacy single-sheet compatibility
    # ---------------------------------------------------------

    legacy_names = (
        "on entire dataset",
        "on 20% test set",
    )

    for legacy_name in legacy_names:
        if legacy_name in sheets:
            frame = sheets[legacy_name].copy()

            frame["data_scope"] = data_scope
            frame["source_sheet"] = legacy_name
            frame["tolerance"] = tolerance

            print(
                f"  Using legacy sheet "
                f"{legacy_name!r}: {len(frame)} rows"
            )

            return frame

    # ---------------------------------------------------------
    # Current tolerance workbook
    # ---------------------------------------------------------

    selected_frames = []

    # Matches:
    #
    # scheme_1_tol_0
    # scheme_1_tol_0_y
    # scheme_1_tol_0_y_K
    # scheme_3_tol_0
    #
    tolerance_pattern = re.compile(
        rf"(?:^|_)tol_{int(tolerance)}(?:_|$)",
        flags=re.IGNORECASE,
    )

    for sheet_name, sheet_df in sheets.items():

        print(
            f"  sheet={sheet_name!r}, "
            f"rows={len(sheet_df)}"
        )

        if "scheme" in sheet_df.columns:
            print(
                "    schemes="
                + str(
                    sheet_df["scheme"]
                    .value_counts(dropna=False)
                    .to_dict()
                )
            )

        if "model" in sheet_df.columns:
            print(
                "    models="
                + str(
                    sheet_df["model"]
                    .value_counts(dropna=False)
                    .to_dict()
                )
            )

        if "strat_on" in sheet_df.columns:
            print(
                "    strat_on="
                + str(
                    sheet_df["strat_on"]
                    .value_counts(dropna=False)
                    .to_dict()
                )
            )

        if sheet_name == DEFAULT_FILTER_SUMMARY_SHEET:
            continue

        if not tolerance_pattern.search(
            str(sheet_name)
        ):
            continue

        # Ignore empty sheets instead of concatenating them.
        # This also removes the pandas FutureWarning caused by
        # concatenating empty/all-NA frames.
        if sheet_df.empty:
            continue

        frame = sheet_df.copy()

        frame["source_sheet"] = sheet_name
        frame["data_scope"] = data_scope
        frame["tolerance"] = tolerance

        selected_frames.append(frame)

    if not selected_frames:
        raise ValueError(
            f"No tol_{tolerance} failure sheets found "
            f"in {workbook_path}. "
            f"Available sheets: {list(sheets)}"
        )

    combined = pd.concat(
        selected_frames,
        ignore_index=True,
        sort=False,
    )

    print(
        f"\n[{data_scope}] Selected "
        f"tol_{tolerance} rows: {len(combined)}"
    )

    if "scheme" in combined.columns:
        print(
            f"[{data_scope}] Raw scheme counts:"
        )
        print(
            combined["scheme"]
            .value_counts(dropna=False)
            .to_string()
        )

    if "strat_on" in combined.columns:
        print(
            f"[{data_scope}] Raw stratification counts:"
        )
        print(
            combined["strat_on"]
            .value_counts(dropna=False)
            .to_string()
        )

    return combined


def normalize_failure_analysis_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:
    normalized = df.copy()

    # ========================================================
    # 1. Normalize scheme
    # ========================================================

    if "scheme" not in normalized.columns:
        raise ValueError(
            "Failure-analysis dataframe does not contain "
            "required column 'scheme'."
        )

    scheme_map = {
        "scheme1": "scheme 1",
        "scheme_1": "scheme 1",
        "scheme 1": "scheme 1",
        "1": "scheme 1",

        "scheme2": "scheme 2",
        "scheme_2": "scheme 2",
        "scheme 2": "scheme 2",
        "2": "scheme 2",

        "scheme3": "scheme 3",
        "scheme_3": "scheme 3",
        "scheme 3": "scheme 3",
        "3": "scheme 3",
    }

    original_scheme = normalized["scheme"].copy()

    normalized["scheme"] = (
        normalized["scheme"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(scheme_map)
    )

    # Fail instead of silently losing an unknown scheme.
    unknown_scheme_mask = (
        normalized["scheme"].isna()
        & original_scheme.notna()
    )

    if unknown_scheme_mask.any():
        unknown_values = sorted(
            original_scheme.loc[unknown_scheme_mask]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unrecognized scheme metadata values: "
            f"{unknown_values}"
        )

    # ========================================================
    # 2. Normalize model
    # ========================================================

    if "model" not in normalized.columns:
        raise ValueError(
            "Failure-analysis dataframe does not contain "
            "required column 'model'."
        )

    model_map = {
        "rf": "RF",
        "random_forest": "RF",
        "random forest": "RF",

        "xgb": "XGB",
        "xgboost": "XGB",
    }

    original_model = normalized["model"].copy()

    normalized["model"] = (
        normalized["model"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(model_map)
    )

    unknown_model_mask = (
        normalized["model"].isna()
        & original_model.notna()
    )

    if unknown_model_mask.any():
        unknown_values = sorted(
            original_model.loc[unknown_model_mask]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unrecognized model metadata values: "
            f"{unknown_values}"
        )

    # ========================================================
    # 3. Define split normalizer BEFORE using it
    # ========================================================

    def normalize_split(value):
        if value is None or pd.isna(value):
            return pd.NA

        original = str(value).strip()

        if not original:
            return pd.NA

        text = (
            original
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        while "__" in text:
            text = text.replace("__", "_")

        compact = text.replace("_", "")

        # ----------------------------------------------------
        # strat_y_K variants
        # ----------------------------------------------------

        strat_yk_values = {
            "strat_y_k",
            "strat_yk",
            "straty_k",
            "stratyk",
            "y_k",
            "y+k",
            "y_and_k",
            "yandk",
        }

        if text in strat_yk_values:
            return "strat_y_K"

        # Defensive handling of historical spellings.
        if (
            "strat" in compact
            and "y" in compact
            and "k" in compact
        ):
            return "strat_y_K"

        # ----------------------------------------------------
        # strat_y variants
        # ----------------------------------------------------

        strat_y_values = {
            "strat_y",
            "straty",
            "y",
        }

        if text in strat_y_values:
            return "strat_y"

        if (
            "strat" in compact
            and "y" in compact
        ):
            return "strat_y"

        # Preserve unknown value so diagnostics expose it.
        return original

    # ========================================================
    # 4. Resolve the source split column
    # ========================================================

    split_candidates = [
        candidate
        for candidate in (
            "split_mode",
            "strat_on",
            "stratified_on",
        )
        if candidate in normalized.columns
    ]

    if not split_candidates:
        raise ValueError(
            "Could not determine stratification mode. "
            "Expected at least one of: "
            "'split_mode', 'strat_on', 'stratified_on'."
        )

    # Prefer the column containing the largest number
    # of actual non-null values.
    split_source = max(
        split_candidates,
        key=lambda column: int(
            normalized[column].notna().sum()
        ),
    )

    print(
        f"\n[failure-analysis] "
        f"Using '{split_source}' as split metadata source."
    )

    print(
        "[failure-analysis] Raw split values:"
    )

    print(
        normalized[split_source]
        .value_counts(dropna=False)
        .to_string()
    )

    # ========================================================
    # 5. NOW call normalize_split()
    # ========================================================

    normalized["split_mode"] = (
        normalized[split_source]
        .apply(normalize_split)
    )

    print(
        "\n[failure-analysis] Normalized split values:"
    )

    print(
        normalized["split_mode"]
        .value_counts(dropna=False)
        .to_string()
    )

    # ========================================================
    # 6. Validate normalized split values
    # ========================================================

    valid_split_modes = {
        "strat_y",
        "strat_y_K",
    }

    invalid_split_mask = (
        normalized["split_mode"].notna()
        & ~normalized["split_mode"].isin(
            valid_split_modes
        )
    )

    if invalid_split_mask.any():
        invalid_values = sorted(
            normalized.loc[
                invalid_split_mask,
                "split_mode",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unrecognized split metadata after normalization: "
            f"{invalid_values}. "
            f"Source column was '{split_source}'."
        )

    return normalized


def _write_failure_analysis_workbook(
    sheet_frames: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df = (
        build_failure_analysis_scope_summary(
            sheet_frames
        )
    )

    # Write ONCE.
    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="failure_scope_summary",
            index=False,
        )

        for (
            _,
            _,
            _,
            sheet_name,
        ) in FAILURE_ANALYSIS_SHEET_ORDER:

            sheet_frames[
                sheet_name
            ].to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

    # Then reopen for formatting.
    workbook = load_workbook(output_path)

    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]

        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = (
            worksheet.dimensions
        )

        autosize_columns(worksheet)

    workbook.save(output_path)
    workbook.close()


def _validate_failure_analysis_workbook(
    output_path: Path,
    sheet_frames: dict[str, pd.DataFrame],
) -> dict[str, object]:

    workbook = load_workbook(output_path)

    experiment_sheet_names = [
        sheet_name
        for _, _, _, sheet_name
        in FAILURE_ANALYSIS_SHEET_ORDER
    ]

    expected_sheet_names = [
        "failure_scope_summary",
        *experiment_sheet_names,
    ]

    actual_sheet_names = workbook.sheetnames

    if actual_sheet_names != expected_sheet_names:
        raise ValueError(
            "Failure analysis workbook sheet names mismatch: "
            f"expected {expected_sheet_names}, "
            f"found {actual_sheet_names}."
        )

    highlighted_cells = 0

    for sheet_name in experiment_sheet_names:
        worksheet = workbook[sheet_name]

        for row in worksheet.iter_rows():
            for cell in row:
                if cell.fill.fill_type not in (None, "none"):
                    highlighted_cells += 1

    workbook.close()

    if highlighted_cells != 0:
        raise ValueError(
            "Failure analysis workbook unexpectedly contains highlighted cells."
        )

    read_back_sheets = pd.read_excel(
        output_path,
        sheet_name=None,
    )

    removed_column_violations: dict[str, list[str]] = {}
    tuple_validation_rows = 0

    for sheet_name in experiment_sheet_names:
        exported_df = read_back_sheets[sheet_name]

        forbidden_found = [
            column
            for column in exported_df.columns
            if (
                column in FORBIDDEN_ANALYSIS_COLUMNS
                or str(column).startswith("prob_")
            )
        ]

        if forbidden_found:
            removed_column_violations[sheet_name] = forbidden_found

        for _, row in exported_df.iterrows():
            if pd.isna(
                row.get("all_predictions_sorted_by_prob")
            ):
                continue

            all_predictions = ast.literal_eval(
                str(row["all_predictions_sorted_by_prob"])
            )

            fastest_tuple = ast.literal_eval(
                str(row["fastest_algo_tuple"])
            )

            fastest_rank = (
                int(fastest_tuple[2])
                if fastest_tuple[2] != "None"
                else None
            )

            actual_fastest_rank = next(
                (
                    idx
                    for idx, entry
                    in enumerate(all_predictions, start=1)
                    if entry[1] == fastest_tuple[1]
                ),
                None,
            )

            if fastest_rank != actual_fastest_rank:
                raise ValueError(
                    f"fastest_algo rank mismatch in sheet "
                    f"{sheet_name} for dataset={row['dataset']}, "
                    f"k={row['k_value']}."
                )

            for top_k in TOP_K_VALUES:
                prefix = all_predictions[:top_k]

                _, best_entry = min(
                    enumerate(prefix, start=1),
                    key=lambda item: (
                        float(item[1][0]),
                        item[0],
                    ),
                )

                exported_ratio = _safe_float(
                    row[f"top{top_k}_min_time_ratio"]
                )

                if (
                    exported_ratio is None
                    or abs(
                        exported_ratio
                        - float(best_entry[0])
                    )
                    > 1e-9
                ):
                    raise ValueError(
                        f"top{top_k}_min_time_ratio mismatch "
                        f"in sheet {sheet_name} "
                        f"for dataset={row['dataset']}, "
                        f"k={row['k_value']}."
                    )

                exported_candidate = ast.literal_eval(
                    str(row[f"top{top_k}_candidate"])
                )

                expected_candidate = (
                    round(float(best_entry[2]), 4),
                    best_entry[1],
                    round(float(best_entry[3]), 4),
                )

                if exported_candidate != expected_candidate:
                    raise ValueError(
                        f"top{top_k}_candidate mismatch "
                        f"in sheet {sheet_name} "
                        f"for dataset={row['dataset']}, "
                        f"k={row['k_value']}."
                    )

            tuple_validation_rows += 1

    if removed_column_violations:
        raise ValueError(
            "Forbidden columns found in failure analysis workbook: "
            f"{removed_column_violations}"
        )

    return {
        "sheet_count": len(expected_sheet_names),
        "experiment_sheet_count": len(experiment_sheet_names),
        "sheet_names": expected_sheet_names,
        "highlighted_cells": highlighted_cells,
        "validated_rows": tuple_validation_rows,
        "row_counts": {
            sheet_name: int(len(sheet_frames[sheet_name]))
            for sheet_name in experiment_sheet_names
        },
    }


def _parse_top3_item(item) -> dict[str, object] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        return None
    algo = str(item[0]).strip()
    second = _safe_float(item[1])
    third = _safe_float(item[2])
    if second is None and third is None:
        return None

    probability = None
    time_ratio = None
    if second is not None and third is not None:
        if second <= 1.0 and third >= 1.0:
            probability = second
            time_ratio = third
        elif third <= 1.0 and second >= 1.0:
            time_ratio = second
            probability = third
        else:
            probability = second
            time_ratio = third
    elif second is not None:
        probability = second if second <= 1.0 else None
        time_ratio = second if second > 1.0 else None
    elif third is not None:
        probability = third if third <= 1.0 else None
        time_ratio = third if third > 1.0 else None

    return {
        "algo": algo,
        "probability": probability,
        "time_ratio": time_ratio,
    }


def parse_top3_predicted_detailed(value) -> list[dict[str, object]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        parsed = None
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = None
        if parsed is None:
            matches = TOP3_ENTRY_PATTERN.findall(stripped)
            if not matches:
                return []
            parsed = [(algo.strip(), second.strip(), third.strip()) for algo, second, third in matches]
    else:
        parsed = value

    if isinstance(parsed, tuple):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    entries = []
    for item in parsed:
        parsed_item = _parse_top3_item(item)
        if parsed_item is not None:
            entries.append(parsed_item)
    return entries


def derive_predicted_probability(row: pd.Series, parsed_top3: list[dict[str, object]]) -> float | None:
    predicted_algo = row.get("predicted_algo")
    if isinstance(predicted_algo, str):
        probability_col = f"prob_{predicted_algo}"
        if probability_col in row.index:
            probability = _safe_float(row.get(probability_col))
            if probability is not None:
                return probability
    for entry in parsed_top3:
        if entry["algo"] == predicted_algo:
            return _safe_float(entry["probability"])
    if parsed_top3:
        return _safe_float(parsed_top3[0]["probability"])
    return None


def select_best_recovery_entry(parsed_top3: list[dict[str, object]], top_k: int) -> dict[str, object] | None:
    candidates = [entry for entry in parsed_top3[:top_k] if _safe_float(entry.get("time_ratio")) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda entry: float(entry["time_ratio"]))


def format_decision_tuple(entry: dict[str, object] | None):
    if entry is None:
        return pd.NA
    algo = str(entry["algo"])
    time_ratio = _round_or_none(_safe_float(entry.get("time_ratio")))
    probability = _round_or_none(_safe_float(entry.get("probability")))
    if time_ratio is None or probability is None:
        return pd.NA
    return f"({algo}, {time_ratio:.4f}, {probability:.4f})"


def parse_decision_tuple(value) -> dict[str, object] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = None

        if isinstance(parsed, tuple) and len(parsed) >= 3:
            algo = str(parsed[0]).strip()
            time_ratio = _safe_float(parsed[1])
            probability = _safe_float(parsed[2])
            if time_ratio is None or probability is None:
                return None
            return {"algo": algo, "time_ratio": time_ratio, "probability": probability}

        matches = TOP3_ENTRY_PATTERN.findall(stripped)
        if matches:
            algo, second, third = matches[0]
            time_ratio = _safe_float(second)
            probability = _safe_float(third)
            if time_ratio is None or probability is None:
                return None
            return {"algo": algo.strip(), "time_ratio": time_ratio, "probability": probability}
        return None

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        algo = str(value[0]).strip()
        time_ratio = _safe_float(value[1])
        probability = _safe_float(value[2])
        if time_ratio is None or probability is None:
            return None
        return {"algo": algo, "time_ratio": time_ratio, "probability": probability}
    return None


def extract_time_ratio_from_decision_value(value) -> float | None:
    parsed = parse_decision_tuple(value)
    if parsed is None:
        return None
    return _safe_float(parsed.get("time_ratio"))


def enrich_failure_sheet(df: pd.DataFrame, high_confidence_threshold: float) -> pd.DataFrame:
    enriched = df.copy()

    top2_candidate_values = []
    top3_candidate_values = []
    top2_ratio_values = []
    top3_ratio_values = []
    high_confidence_flags = []

    for _, row in enriched.iterrows():
        parsed_top3 = parse_top3_predicted_detailed(row.get("top3_predicted_detailed"))
        top2_entry = select_best_recovery_entry(parsed_top3, top_k=2)
        top3_entry = select_best_recovery_entry(parsed_top3, top_k=3)
        top2_candidate_values.append(format_decision_tuple(top2_entry))
        top3_candidate_values.append(format_decision_tuple(top3_entry))
        top2_ratio_values.append(_round_or_none(_safe_float(top2_entry.get("time_ratio"))) if top2_entry else pd.NA)
        top3_ratio_values.append(_round_or_none(_safe_float(top3_entry.get("time_ratio"))) if top3_entry else pd.NA)

        predicted_probability = derive_predicted_probability(row, parsed_top3)
        high_confidence_flags.append(bool(predicted_probability is not None and predicted_probability >= high_confidence_threshold))

    for column_name in (
        "min_time_ratio_top2",
        "min_time_ratio_top3",
        BEST_TOP2_CANDIDATE_COL,
        BEST_TOP3_CANDIDATE_COL,
        BEST_TOP2_TIME_RATIO_COL,
        BEST_TOP3_TIME_RATIO_COL,
        "high_confidence_wrong_prediction_flag",
    ):
        if column_name in enriched.columns:
            enriched = enriched.drop(columns=[column_name])

    enriched[BEST_TOP2_CANDIDATE_COL] = top2_candidate_values
    enriched[BEST_TOP3_CANDIDATE_COL] = top3_candidate_values
    enriched[BEST_TOP2_TIME_RATIO_COL] = top2_ratio_values
    enriched[BEST_TOP3_TIME_RATIO_COL] = top3_ratio_values
    enriched["high_confidence_wrong_prediction_flag"] = high_confidence_flags
    return enriched


def load_failure_workbook_sheets(
    workbook_path: Path,
    summary_sheet_name: str = DEFAULT_FILTER_SUMMARY_SHEET,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    sheets = pd.read_excel(workbook_path, sheet_name=None)
    data_sheets: dict[str, pd.DataFrame] = {}
    summary_df = None
    for sheet_name, df in sheets.items():
        if sheet_name == summary_sheet_name:
            summary_df = df
        else:
            data_sheets[sheet_name] = df
    return data_sheets, summary_df


def collect_failure_rows_from_sheets(
    sheet_frames: dict[str, pd.DataFrame],
    split_scope: str,
) -> pd.DataFrame:
    frames = []
    for sheet_name, df in sheet_frames.items():
        subset = df.copy()
        subset["split_scope"] = split_scope
        subset["source_sheet"] = sheet_name
        tolerance_match = re.search(r"_tol_(\d+)$", sheet_name)
        subset["tolerance"] = int(tolerance_match.group(1)) if tolerance_match else pd.NA
        frames.append(subset)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def highlight_min_time_ratio_cells(
    worksheet,
    column_names: tuple[str, str] = (BEST_TOP2_CANDIDATE_COL, BEST_TOP3_CANDIDATE_COL),
    threshold: float = DEFAULT_MIN_TIME_RATIO_HIGHLIGHT_THRESHOLD,
) -> None:
    headers = [cell.value for cell in worksheet[1]]
    for column_name in column_names:
        if column_name not in headers:
            continue
        column_idx = headers.index(column_name) + 1
        for row_idx in range(2, worksheet.max_row + 1):
            value = worksheet.cell(row=row_idx, column=column_idx).value
            numeric_value = extract_time_ratio_from_decision_value(value)
            if numeric_value is not None and numeric_value < threshold:
                worksheet.cell(row=row_idx, column=column_idx).fill = GREEN_FILL


def write_enriched_failure_workbook(
    sheet_frames: dict[str, pd.DataFrame],
    summary_df: pd.DataFrame | None,
    output_path: Path,
    time_gap_highlight_threshold: float,
    min_time_ratio_highlight_threshold: float,
    summary_sheet_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheet_frames.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        if summary_df is not None:
            summary_df.to_excel(writer, sheet_name=summary_sheet_name[:31], index=False)

    workbook = load_workbook(output_path)
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        if sheet_name != summary_sheet_name:
            highlight_rows_by_failure_severity(worksheet, severity_col="failure_severity")
            highlight_abnormal_time_gap_cells(worksheet, threshold=time_gap_highlight_threshold)
            highlight_min_time_ratio_cells(
                worksheet,
                threshold=min_time_ratio_highlight_threshold,
            )
        autosize_columns(worksheet)

    workbook.save(output_path)
    workbook.close()


def collect_failure_rows(
    workbook_path: Path,
    split_scope: str,
    summary_sheet_name: str = DEFAULT_FILTER_SUMMARY_SHEET,
) -> pd.DataFrame:
    data_sheets, _ = load_failure_workbook_sheets(workbook_path, summary_sheet_name=summary_sheet_name)
    return collect_failure_rows_from_sheets(data_sheets, split_scope=split_scope)


def add_dataset_summary_counts(df: pd.DataFrame) -> pd.DataFrame:
    summary_df = df.copy()
    key_all = ["dataset", "k_value", "fastest_algo", "predicted_algo"]
    key_scheme = key_all + ["scheme"]
    key_scheme_model = key_scheme + ["model"]

    summary_df["failure_count_in_scheme"] = summary_df.groupby(key_scheme)["dataset"].transform("size")
    summary_df["failure_count_in_scheme_model"] = summary_df.groupby(key_scheme_model)["dataset"].transform("size")
    summary_df["failure_count_all_schemes"] = summary_df.groupby(key_all)["dataset"].transform("size")
    summary_df["failure_count_in_split_scope"] = summary_df.groupby(key_all + ["split_scope"])["dataset"].transform("size")
    return summary_df


def ensure_predicted_probability_column(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["predicted_probability"] = normalized.apply(
        lambda row: derive_predicted_probability(row, parse_top3_predicted_detailed(row.get("top3_predicted_detailed"))),
        axis=1,
    )
    return normalized


def summarize_repeat_failures(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["dataset", "k_value", "fastest_algo", "predicted_algo"], dropna=False)
        .agg(
            total_failure_count=("dataset", "size"),
            scheme_breakdown=("scheme", lambda values: "; ".join(f"{k}:{v}" for k, v in pd.Series(values).value_counts().sort_index().items())),
            model_breakdown=("model", lambda values: "; ".join(f"{k}:{v}" for k, v in pd.Series(values).value_counts().sort_index().items())),
            split_breakdown=("split_scope", lambda values: "; ".join(f"{k}:{v}" for k, v in pd.Series(values).value_counts().sort_index().items())),
            mean_time_gap=("time_gap", "mean"),
            mean_time_ratio=("time_ratio", "mean"),
            avg_probability=("predicted_probability", "mean"),
        )
        .reset_index()
        .sort_values(
            by=["total_failure_count", "dataset", "predicted_algo"],
            ascending=[False, True, True],
        )
    )
    return grouped


def build_executive_summary(
    raw_df: pd.DataFrame,
    deduped_df: pd.DataFrame,
    repeat_failure_df: pd.DataFrame,
    dataset_scheme_matrix_df: pd.DataFrame,
) -> pd.DataFrame:
    worst_scheme = deduped_df["scheme"].value_counts().idxmax() if not deduped_df.empty else ""
    worst_dataset = deduped_df["dataset"].value_counts().idxmax() if not deduped_df.empty else ""
    worst_repeated_confusion = ""
    max_repeat_count = 0
    highest_mean_time_ratio = 0.0
    if not repeat_failure_df.empty:
        top_confusion = repeat_failure_df.iloc[0]
        worst_repeated_confusion = (
            f"{top_confusion['dataset']} | K={top_confusion['k_value']} | "
            f"{top_confusion['fastest_algo']} -> {top_confusion['predicted_algo']}"
        )
        max_repeat_count = int(top_confusion["total_failure_count"])
        highest_mean_time_ratio = float(repeat_failure_df["mean_time_ratio"].max())
    worst_model = deduped_df["model"].value_counts().idxmax() if not deduped_df.empty else ""
    summary_rows = [
        ("Total row-level failures", int(len(raw_df))),
        ("Unique repeat failure patterns", int(len(repeat_failure_df))),
        (
            "Unique dataset-scheme pairs",
            int(
                len(
                    dataset_scheme_matrix_df.melt(
                        id_vars=["Dataset", "Total"],
                        value_vars=[column for column in ["Scheme 1", "Scheme 2", "Scheme 3"] if column in dataset_scheme_matrix_df.columns],
                    ).query("value > 0")
                )
            ),
        ),
        ("Worst scheme", worst_scheme),
        ("Worst model", worst_model),
        ("Worst dataset", worst_dataset),
        ("Worst repeated confusion", worst_repeated_confusion),
        ("Max repeat count", max_repeat_count),
        ("Highest mean time ratio", round(highest_mean_time_ratio, 4)),
        (
            "High-confidence wrong predictions",
            int(pd.to_numeric(deduped_df["predicted_probability"], errors="coerce").ge(DEFAULT_LAYERED_HIGH_CONFIDENCE_THRESHOLD).sum()),
        ),
    ]
    return pd.DataFrame(summary_rows, columns=["Metric", "Value"])


def build_dataset_ranking(df: pd.DataFrame) -> pd.DataFrame:
    ranking = (
        df.groupby("dataset", dropna=False)
        .agg(
            failures=("dataset", "size"),
            schemes_affected=("scheme", lambda values: ", ".join(sorted(pd.Series(values).dropna().astype(str).unique()))),
            models_affected=("model", lambda values: ", ".join(sorted(pd.Series(values).dropna().astype(str).unique()))),
            avg_time_ratio=("time_ratio", "mean"),
        )
        .reset_index()
        .sort_values(by=["failures", "avg_time_ratio", "dataset"], ascending=[False, False, True])
    )
    ranking["avg_time_ratio"] = ranking["avg_time_ratio"].round(4)
    ranking["max_time_ratio"] = (
        df.groupby("dataset", dropna=False)["time_ratio"].max().reindex(ranking["dataset"]).to_numpy()
    )
    ranking["max_time_ratio"] = ranking["max_time_ratio"].round(4)
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    return ranking.rename(
        columns={
            "dataset": "Dataset",
            "failures": "Total Failures",
            "schemes_affected": "Schemes Affected",
            "models_affected": "Models Affected",
            "avg_time_ratio": "Avg Time Ratio",
            "max_time_ratio": "Max Time Ratio",
        }
    )[["Rank", "Dataset", "Total Failures", "Schemes Affected", "Models Affected", "Avg Time Ratio", "Max Time Ratio"]].head(20)


def build_dataset_scheme_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.pivot_table(
            index="dataset",
            columns="scheme",
            values="predicted_algo",
            aggfunc="size",
            fill_value=0,
        )
        .reindex(columns=list(SCHEMES), fill_value=0)
        .reset_index()
    )
    pivot["Total"] = pivot[list(SCHEMES)].sum(axis=1)
    return pivot.sort_values(by=["Total", *list(SCHEMES)], ascending=[False, False, False, False]).rename(
        columns={"dataset": "Dataset", "scheme 1": "Scheme 1", "scheme 2": "Scheme 2", "scheme 3": "Scheme 3"}
    )


def build_top_20_confusions(repeat_failure_df: pd.DataFrame) -> pd.DataFrame:
    top_confusions = repeat_failure_df.copy().sort_values(
        by=["total_failure_count", "mean_time_ratio", "dataset"],
        ascending=[False, False, True],
    ).head(20).reset_index(drop=True)
    top_confusions.insert(0, "Rank", range(1, len(top_confusions) + 1))
    top_confusions["mean_time_ratio"] = top_confusions["mean_time_ratio"].round(4)
    top_confusions["avg_probability"] = top_confusions["avg_probability"].round(4)
    return top_confusions.rename(
        columns={
            "dataset": "Dataset",
            "k_value": "K",
            "fastest_algo": "Fastest Algo",
            "predicted_algo": "Predicted Algo",
            "total_failure_count": "Count",
            "mean_time_ratio": "Avg Time Ratio",
            "avg_probability": "Avg Probability",
        }
    )[["Rank", "Dataset", "K", "Fastest Algo", "Predicted Algo", "Count", "Avg Time Ratio", "Avg Probability"]]


def build_scheme_model_summary(
    df: pd.DataFrame,
    catastrophic_time_ratio_threshold: float,
    catastrophic_predicted_time_threshold: float,
) -> pd.DataFrame:
    catastrophic_mask = (
        pd.to_numeric(df["time_ratio"], errors="coerce").ge(catastrophic_time_ratio_threshold)
        | pd.to_numeric(df["predicted_time"], errors="coerce").ge(catastrophic_predicted_time_threshold)
    )
    summary = (
        df.assign(
            catastrophic_flag=catastrophic_mask,
            high_confidence_flag=pd.to_numeric(df["predicted_probability"], errors="coerce").ge(
                DEFAULT_LAYERED_HIGH_CONFIDENCE_THRESHOLD
            ),
        )
        .groupby(["scheme", "model"], dropna=False)
        .agg(
            failure_count=("dataset", "size"),
            unique_datasets=("dataset", "nunique"),
            avg_time_ratio=("time_ratio", "mean"),
            high_confidence_count=("high_confidence_flag", "sum"),
            catastrophic_count=("catastrophic_flag", "sum"),
        )
        .reset_index()
        .sort_values(by=["failure_count", "avg_time_ratio", "scheme", "model"], ascending=[False, False, True, True])
    )
    summary["avg_time_ratio"] = summary["avg_time_ratio"].round(4)
    return summary.rename(
        columns={
            "scheme": "Scheme",
            "model": "Model",
            "failure_count": "Failure Count",
            "unique_datasets": "Unique Datasets",
            "avg_time_ratio": "Avg Time Ratio",
            "high_confidence_count": "High Confidence Count",
            "catastrophic_count": "Catastrophic Count",
        }
    )


def build_tolerance_trend_summary(df: pd.DataFrame) -> pd.DataFrame:
    trend = (
        df.groupby(["scheme", "model", "tolerance"], dropna=False)
        .size()
        .reset_index(name="failure_count")
        .pivot_table(
            index=["scheme", "model"],
            columns="tolerance",
            values="failure_count",
            fill_value=0,
        )
        .reset_index()
    )
    trend = trend.rename(
        columns={
            0: "Tol 0",
            10: "Tol 10",
            20: "Tol 20",
            50: "Tol 50",
        }
    )
    for column in ["Tol 0", "Tol 10", "Tol 20", "Tol 50"]:
        if column not in trend.columns:
            trend[column] = 0
    return trend[["scheme", "model", "Tol 0", "Tol 10", "Tol 20", "Tol 50"]].sort_values(
        by=["scheme", "model"]
    )


def deduplicate_tolerance_rows(df: pd.DataFrame) -> pd.DataFrame:
    dedupe_key = [
        "dataset",
        "k_value",
        "scheme",
        "model",
        "split_scope",
        "fastest_algo",
        "fastest_runtime",
        "predicted_algo",
        "predicted_time",
        "time_gap",
        "time_ratio",
    ]
    working = df.copy()
    working["tolerances_present"] = (
        working.groupby(dedupe_key)["tolerance"]
        .transform(lambda values: ", ".join(str(int(value)) for value in sorted(pd.Series(values).dropna().astype(int).unique())))
    )
    deduped = working.sort_values(by=["dataset", "scheme", "model", "split_scope", "tolerance"]).drop_duplicates(
        subset=dedupe_key,
        keep="first",
    )
    return deduped.reset_index(drop=True)


def build_high_confidence_wrong_predictions(df: pd.DataFrame) -> pd.DataFrame:
    high_conf_df = df[pd.to_numeric(df["predicted_probability"], errors="coerce").ge(DEFAULT_LAYERED_HIGH_CONFIDENCE_THRESHOLD)].copy()
    high_conf_df = high_conf_df.drop_duplicates(
        subset=[
            "dataset",
            "k_value",
            "scheme",
            "model",
            "fastest_algo",
            "predicted_algo",
            "predicted_probability",
            "time_ratio",
        ],
        keep="first",
    )
    selected_columns = [
        "dataset",
        "k_value",
        "scheme",
        "model",
        "fastest_algo",
        "predicted_algo",
        "predicted_probability",
        "time_ratio",
    ]
    return high_conf_df[selected_columns].rename(
        columns={
            "dataset": "Dataset",
            "k_value": "K",
            "scheme": "Scheme",
            "model": "Model",
            "fastest_algo": "Fastest Algo",
            "predicted_algo": "Predicted Algo",
            "predicted_probability": "Probability",
            "time_ratio": "Time Ratio",
        }
    ).sort_values(
        by=["Probability", "Time Ratio", "Dataset"],
        ascending=[False, False, True],
    )


def build_near_miss_recovery_potential(
    df: pd.DataFrame,
    min_time_ratio_highlight_threshold: float,
) -> pd.DataFrame:
    near_miss_df = df.copy()
    near_miss_df["top2_recovery_ratio"] = near_miss_df[BEST_TOP2_CANDIDATE_COL].apply(extract_time_ratio_from_decision_value)
    near_miss_df["top3_recovery_ratio"] = near_miss_df[BEST_TOP3_CANDIDATE_COL].apply(extract_time_ratio_from_decision_value)
    near_miss_df["top2_recovery_possible"] = near_miss_df["top2_recovery_ratio"].lt(min_time_ratio_highlight_threshold)
    near_miss_df["top3_recovery_possible"] = near_miss_df["top3_recovery_ratio"].lt(min_time_ratio_highlight_threshold)
    near_miss_df = near_miss_df.drop_duplicates(
        subset=[
            "dataset",
            "k_value",
            "scheme",
            "model",
            "fastest_algo",
            "predicted_algo",
            "time_ratio",
            BEST_TOP2_CANDIDATE_COL,
            BEST_TOP3_CANDIDATE_COL,
        ],
        keep="first",
    )
    selected_columns = [
        "dataset",
        "k_value",
        "scheme",
        "model",
        "fastest_algo",
        "predicted_algo",
        "time_ratio",
        BEST_TOP2_CANDIDATE_COL,
        "top2_recovery_possible",
        BEST_TOP3_CANDIDATE_COL,
        "top3_recovery_possible",
    ]
    near_miss_df = near_miss_df[selected_columns].rename(
        columns={
            "dataset": "Dataset",
            "k_value": "K",
            "scheme": "Scheme",
            "model": "Model",
            "fastest_algo": "Fastest Algo",
            "predicted_algo": "Predicted Algo",
            "time_ratio": "Actual Time Ratio",
            BEST_TOP2_CANDIDATE_COL: "Best Top2 Candidate",
            BEST_TOP3_CANDIDATE_COL: "Best Top3 Candidate",
            "top2_recovery_possible": "Recoverable by Top2?",
            "top3_recovery_possible": "Recoverable by Top3?",
        }
    )
    return near_miss_df.sort_values(
        by=["Recoverable by Top3?", "Recoverable by Top2?", "Actual Time Ratio", "Dataset"],
        ascending=[False, False, False, True],
    )


def build_catastrophic_failures(
    df: pd.DataFrame,
    catastrophic_time_ratio_threshold: float,
    catastrophic_predicted_time_threshold: float,
) -> pd.DataFrame:
    catastrophic_mask = (
        pd.to_numeric(df["time_ratio"], errors="coerce").ge(catastrophic_time_ratio_threshold)
        | pd.to_numeric(df["predicted_time"], errors="coerce").ge(catastrophic_predicted_time_threshold)
    )
    catastrophic_df = df[catastrophic_mask].copy()
    catastrophic_df = catastrophic_df.drop_duplicates(
        subset=[
            "dataset",
            "k_value",
            "scheme",
            "model",
            "fastest_algo",
            "fastest_runtime",
            "predicted_algo",
            "predicted_time",
            "time_ratio",
        ],
        keep="first",
    )
    selected_columns = [
        "dataset",
        "k_value",
        "scheme",
        "model",
        "fastest_algo",
        "fastest_runtime",
        "predicted_algo",
        "predicted_time",
        "time_ratio",
    ]
    catastrophic_df = catastrophic_df[selected_columns].rename(
        columns={
            "dataset": "Dataset",
            "k_value": "K",
            "scheme": "Scheme",
            "model": "Model",
            "fastest_algo": "Fastest Algo",
            "predicted_algo": "Predicted Algo",
            "fastest_runtime": "Fastest Time",
            "predicted_time": "Predicted Time",
            "time_ratio": "Time Ratio",
        }
    )
    return catastrophic_df.sort_values(
        by=["Time Ratio", "Predicted Time", "Dataset"],
        ascending=[False, False, True],
    )


def build_raw_detail_sheet(df: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [
        "dataset",
        "k_value",
        "scheme",
        "model",
        "split_scope",
        "source_sheet",
        "tolerance",
        "fastest_algo",
        "fastest_runtime",
        "predicted_algo",
        "predicted_time",
        "time_gap",
        "time_ratio",
        "predicted_probability",
        "high_confidence_wrong_prediction_flag",
        "failure_count_in_scheme",
        "failure_count_in_scheme_model",
        "failure_count_all_schemes",
        "failure_count_in_split_scope",
        "top3_predicted_detailed",
        BEST_TOP2_CANDIDATE_COL,
        BEST_TOP3_CANDIDATE_COL,
        BEST_TOP2_TIME_RATIO_COL,
        BEST_TOP3_TIME_RATIO_COL,
    ] + [column for column in SUMMARY_PROBABILITY_COLUMNS if column in df.columns]
    return df[selected_columns].sort_values(
        by=["failure_count_all_schemes", "dataset", "scheme", "model", "split_scope", "tolerance"],
        ascending=[False, True, True, True, True, True],
    )


def write_dataset_failure_summary(
    raw_df: pd.DataFrame,
    deduped_df: pd.DataFrame,
    repeat_failure_df: pd.DataFrame,
    output_path: Path,
    min_time_ratio_highlight_threshold: float,
    catastrophic_time_ratio_threshold: float,
    catastrophic_predicted_time_threshold: float,
) -> None:
    dataset_scheme_summary_df = build_dataset_scheme_heatmap_data(deduped_df)
    executive_summary_df = build_executive_summary(raw_df, deduped_df, repeat_failure_df, dataset_scheme_summary_df)
    dataset_ranking_df = build_dataset_ranking(deduped_df)
    top_20_confusions_df = build_top_20_confusions(repeat_failure_df)
    scheme_model_summary_df = build_scheme_model_summary(
        deduped_df,
        catastrophic_time_ratio_threshold=catastrophic_time_ratio_threshold,
        catastrophic_predicted_time_threshold=catastrophic_predicted_time_threshold,
    )
    high_confidence_df = build_high_confidence_wrong_predictions(deduped_df)
    near_miss_df = build_near_miss_recovery_potential(
        deduped_df,
        min_time_ratio_highlight_threshold=min_time_ratio_highlight_threshold,
    )
    catastrophic_df = build_catastrophic_failures(
        deduped_df,
        catastrophic_time_ratio_threshold=catastrophic_time_ratio_threshold,
        catastrophic_predicted_time_threshold=catastrophic_predicted_time_threshold,
    )
    tolerance_trend_df = build_tolerance_trend_summary(raw_df)
    raw_detail_df = build_raw_detail_sheet(raw_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        executive_summary_df.to_excel(writer, sheet_name="executive_summary", index=False)
        dataset_ranking_df.to_excel(writer, sheet_name="top_20_dataset_failures", index=False)
        top_20_confusions_df.to_excel(writer, sheet_name="top_20_confusions", index=False)
        scheme_model_summary_df.to_excel(writer, sheet_name="scheme_model_summary", index=False)
        dataset_scheme_summary_df.to_excel(writer, sheet_name="dataset_scheme_matrix", index=False)
        high_confidence_df.to_excel(writer, sheet_name="high_confidence_failures", index=False)
        catastrophic_df.to_excel(writer, sheet_name="catastrophic_failures", index=False)
        near_miss_df.to_excel(writer, sheet_name="topk_recovery", index=False)
        tolerance_trend_df.to_excel(writer, sheet_name="tolerance_trend_summary", index=False)
        raw_detail_df.to_excel(writer, sheet_name="raw_failure_rows", index=False)

    workbook = load_workbook(output_path)
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        if sheet_name == "topk_recovery":
            highlight_min_time_ratio_cells(
                worksheet,
                column_names=("Best Top2 Candidate", "Best Top3 Candidate"),
                threshold=min_time_ratio_highlight_threshold,
            )
        if sheet_name in {"catastrophic_failures", "raw_failure_rows"}:
            highlight_min_time_ratio_cells(
                worksheet,
                threshold=min_time_ratio_highlight_threshold,
            )
        autosize_columns(worksheet)
    workbook.save(output_path)
    workbook.close()


def write_repeat_failure_table(repeat_failure_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        repeat_failure_df.to_excel(writer, sheet_name="repeat_failures", index=False)

    workbook = load_workbook(output_path)
    worksheet = workbook["repeat_failures"]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    autosize_columns(worksheet)
    workbook.save(output_path)
    workbook.close()


def validate_dataset_summary_workbook(
    output_path: Path,
    raw_failure_df: pd.DataFrame,
    deduped_failure_df: pd.DataFrame,
) -> dict[str, int]:
    workbook_sheets = pd.read_excel(output_path, sheet_name=None)
    expected_sheet_order = [
        "executive_summary",
        "top_20_dataset_failures",
        "top_20_confusions",
        "scheme_model_summary",
        "dataset_scheme_matrix",
        "high_confidence_failures",
        "catastrophic_failures",
        "topk_recovery",
        "tolerance_trend_summary",
        "raw_failure_rows",
    ]
    actual_sheet_order = list(workbook_sheets)
    if actual_sheet_order != expected_sheet_order:
        raise ValueError(
            f"Dataset summary workbook sheet order mismatch: expected {expected_sheet_order}, found {actual_sheet_order}."
        )

    raw_detail_rows = len(workbook_sheets["raw_failure_rows"])
    if raw_detail_rows != len(raw_failure_df):
        raise ValueError("raw_failure_rows sheet row count does not match the combined failure dataframe.")

    systematic_confusions = workbook_sheets["top_20_confusions"]
    if systematic_confusions.empty:
        raise ValueError("top_20_confusions sheet is empty.")

    heatmap_data = workbook_sheets["dataset_scheme_matrix"]
    numeric_columns = [column for column in ["Scheme 1", "Scheme 2", "Scheme 3"] if column in heatmap_data.columns]
    if int(heatmap_data[numeric_columns].to_numpy().sum()) != len(deduped_failure_df):
        raise ValueError("dataset_scheme_matrix sheet counts do not match the combined failure row count.")

    topk_recovery_rows = len(workbook_sheets["topk_recovery"])
    if topk_recovery_rows <= 0 or topk_recovery_rows > len(deduped_failure_df):
        raise ValueError("topk_recovery sheet row count is outside the expected deduplicated range.")

    return {
        "raw_detail_rows": raw_detail_rows,
        "systematic_confusion_rows": len(systematic_confusions),
        "heatmap_dataset_rows": len(heatmap_data),
        "topk_recovery_rows": topk_recovery_rows,
    }


def create_dataset_scheme_heatmap(combined_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    pivot = (
        combined_df.pivot_table(
            index="dataset",
            columns="scheme",
            values="predicted_algo",
            aggfunc="size",
            fill_value=0,
        )
        .reindex(columns=list(SCHEMES), fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values(by="total", ascending=False).drop(columns=["total"])

    figure_height = min(max(6.0, 0.28 * max(len(pivot), 1)), 36.0)
    fig, ax = plt.subplots(figsize=(9, figure_height))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_title("Dataset x Scheme Important Failure Counts")
    ax.set_xlabel("Scheme")
    ax.set_ylabel("Dataset")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(image, ax=ax, label="Failure count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pivot


def validate_enriched_failure_sheets(
    original_sheets: dict[str, pd.DataFrame],
    enriched_sheets: dict[str, pd.DataFrame],
    high_confidence_threshold: float,
    min_time_ratio_highlight_threshold: float,
) -> dict[str, object]:
    results = {
        "green_highlight_candidate_count_top2": 0,
        "green_highlight_candidate_count_top3": 0,
        "high_confidence_true_count": 0,
    }
    for sheet_name, original_df in original_sheets.items():
        enriched_df = enriched_sheets[sheet_name]
        for column_name in (
            BEST_TOP2_CANDIDATE_COL,
            BEST_TOP3_CANDIDATE_COL,
            BEST_TOP2_TIME_RATIO_COL,
            BEST_TOP3_TIME_RATIO_COL,
            "high_confidence_wrong_prediction_flag",
        ):
            if column_name not in enriched_df.columns:
                raise ValueError(f"Missing required derived column '{column_name}' in sheet {sheet_name}.")

        if len(original_df) != len(enriched_df):
            raise ValueError(f"Row count changed unexpectedly for sheet {sheet_name}.")

        green_top2 = enriched_df[BEST_TOP2_CANDIDATE_COL].apply(extract_time_ratio_from_decision_value).map(
            lambda value: value is not None and value < min_time_ratio_highlight_threshold
        )
        green_top3 = enriched_df[BEST_TOP3_CANDIDATE_COL].apply(extract_time_ratio_from_decision_value).map(
            lambda value: value is not None and value < min_time_ratio_highlight_threshold
        )
        results["green_highlight_candidate_count_top2"] += int(green_top2.fillna(False).sum())
        results["green_highlight_candidate_count_top3"] += int(green_top3.fillna(False).sum())
        results["high_confidence_true_count"] += int(
            enriched_df["high_confidence_wrong_prediction_flag"].fillna(False).astype(bool).sum()
        )

        if not enriched_df["high_confidence_wrong_prediction_flag"].dropna().isin([True, False]).all():
            raise ValueError(f"High-confidence flag contains non-boolean values in sheet {sheet_name}.")

        for _, row in enriched_df.iterrows():
            parsed_top3 = parse_top3_predicted_detailed(row.get("top3_predicted_detailed"))
            expected_top2 = select_best_recovery_entry(parsed_top3, top_k=2)
            expected_top3 = select_best_recovery_entry(parsed_top3, top_k=3)
            actual_top2 = parse_decision_tuple(row.get(BEST_TOP2_CANDIDATE_COL))
            actual_top3 = parse_decision_tuple(row.get(BEST_TOP3_CANDIDATE_COL))
            if expected_top2 is None and actual_top2 is not None:
                raise ValueError(f"Unexpected {BEST_TOP2_CANDIDATE_COL} value in sheet {sheet_name}.")
            if expected_top2 is not None and (
                actual_top2 is None
                or actual_top2["algo"] != expected_top2["algo"]
                or abs(actual_top2["time_ratio"] - round(float(expected_top2["time_ratio"]), 4)) > 1e-9
                or abs(actual_top2["probability"] - round(float(expected_top2["probability"]), 4)) > 1e-9
            ):
                raise ValueError(f"{BEST_TOP2_CANDIDATE_COL} mismatch in sheet {sheet_name}.")
            if expected_top3 is None and actual_top3 is not None:
                raise ValueError(f"Unexpected {BEST_TOP3_CANDIDATE_COL} value in sheet {sheet_name}.")
            if expected_top3 is not None and (
                actual_top3 is None
                or actual_top3["algo"] != expected_top3["algo"]
                or abs(actual_top3["time_ratio"] - round(float(expected_top3["time_ratio"]), 4)) > 1e-9
                or abs(actual_top3["probability"] - round(float(expected_top3["probability"]), 4)) > 1e-9
            ):
                raise ValueError(f"{BEST_TOP3_CANDIDATE_COL} mismatch in sheet {sheet_name}.")

            actual_top2_ratio = _safe_float(row.get(BEST_TOP2_TIME_RATIO_COL))
            actual_top3_ratio = _safe_float(row.get(BEST_TOP3_TIME_RATIO_COL))
            expected_top2_ratio = round(float(expected_top2["time_ratio"]), 4) if expected_top2 is not None else None
            expected_top3_ratio = round(float(expected_top3["time_ratio"]), 4) if expected_top3 is not None else None
            if expected_top2_ratio is None and actual_top2_ratio is not None:
                raise ValueError(f"Unexpected {BEST_TOP2_TIME_RATIO_COL} value in sheet {sheet_name}.")
            if expected_top2_ratio is not None and (actual_top2_ratio is None or abs(actual_top2_ratio - expected_top2_ratio) > 1e-9):
                raise ValueError(f"{BEST_TOP2_TIME_RATIO_COL} mismatch in sheet {sheet_name}.")
            if expected_top3_ratio is None and actual_top3_ratio is not None:
                raise ValueError(f"Unexpected {BEST_TOP3_TIME_RATIO_COL} value in sheet {sheet_name}.")
            if expected_top3_ratio is not None and (actual_top3_ratio is None or abs(actual_top3_ratio - expected_top3_ratio) > 1e-9):
                raise ValueError(f"{BEST_TOP3_TIME_RATIO_COL} mismatch in sheet {sheet_name}.")

            predicted_probability = derive_predicted_probability(row, parsed_top3)
            expected_flag = bool(predicted_probability is not None and predicted_probability >= high_confidence_threshold)
            if bool(row.get("high_confidence_wrong_prediction_flag")) != expected_flag:
                raise ValueError(f"high_confidence_wrong_prediction_flag mismatch in sheet {sheet_name}.")

    return results


def validate_green_highlighting(
    output_path: Path,
    min_time_ratio_highlight_threshold: float,
) -> dict[str, int]:
    workbook = load_workbook(output_path)
    results = {
        "green_cells_top2": 0,
        "green_cells_top3": 0,
        "non_green_cells_top2_below_threshold": 0,
        "non_green_cells_top3_below_threshold": 0,
    }
    green_colors = {"00C6EFCE", "FFC6EFCE"}

    for sheet_name in workbook.sheetnames:
        if sheet_name == DEFAULT_FILTER_SUMMARY_SHEET:
            continue
        worksheet = workbook[sheet_name]
        headers = [cell.value for cell in worksheet[1]]
        for column_name, result_key, missing_key in (
            (BEST_TOP2_CANDIDATE_COL, "green_cells_top2", "non_green_cells_top2_below_threshold"),
            (BEST_TOP3_CANDIDATE_COL, "green_cells_top3", "non_green_cells_top3_below_threshold"),
        ):
            if column_name not in headers:
                continue
            column_idx = headers.index(column_name) + 1
            for row_idx in range(2, worksheet.max_row + 1):
                cell = worksheet.cell(row=row_idx, column=column_idx)
                value = extract_time_ratio_from_decision_value(cell.value)
                if value is None or value >= min_time_ratio_highlight_threshold:
                    continue
                fill_color = cell.fill.start_color.rgb
                if fill_color in green_colors:
                    results[result_key] += 1
                else:
                    results[missing_key] += 1

    workbook.close()
    if results["non_green_cells_top2_below_threshold"] or results["non_green_cells_top3_below_threshold"]:
        raise ValueError("Green highlighting validation failed for min_time_ratio columns.")
    return results


def run_summary_mode(args) -> None:
    entire_important_arg = Path(args.entire_important_workbook)
    if entire_important_arg.exists():
        entire_important_path = entire_important_arg
    else:
        entire_important_path = resolve_existing_path(
            DEFAULT_ENTIRE_IMPORTANT_CANDIDATES,
            "entire-data important-failures workbook for summary mode",
        )
    entire_base_path = Path(args.entire_base_workbook) if args.entire_base_workbook else resolve_existing_path(
        DEFAULT_ENTIRE_BASE_CANDIDATES,
        "entire-data base compare workbook",
    )
    test_compare_path = Path(args.test_compare_workbook) if args.test_compare_workbook else resolve_existing_path(
        DEFAULT_TEST_COMPARE_CANDIDATES,
        "test-data compare workbook",
    )
    output_path = Path(args.output_path)

    entire_important_df = load_entire_important(entire_important_path)
    entire_base_df = load_entire_base(entire_base_path)
    test_base_df = load_test_compare(test_compare_path)
    test_important_df = compute_important_subset_from_compare(test_base_df)

    validate_counts(entire_base_df, entire_important_df, test_base_df, test_important_df)
    build_workbook(entire_base_df, entire_important_df, test_base_df, test_important_df, output_path)

    print(f"Wrote failure summary workbook to: {output_path}")
    print(f"Entire data source: important={entire_important_path}, base={entire_base_path}")
    print(f"Test data source: compare={test_compare_path}")
    for scheme in SCHEMES:
        print(
            f"{scheme}: "
            f"entire total={int((entire_base_df['scheme'] == scheme).sum())}, "
            f"entire important={int((entire_important_df['scheme'] == scheme).sum())}, "
            f"test total={int((test_base_df['scheme'] == scheme).sum())}, "
            f"test important={int((test_important_df['scheme'] == scheme).sum())}"
        )


def run_enrichment_mode(args) -> None:
    entire_path = (
        Path(args.failure_workbook)
        if args.failure_workbook
        else resolve_existing_path(
            DEFAULT_ENTIRE_IMPORTANT_FAILURES_CANDIDATES,
            "entire-data important-failures workbook",
        )
    )

    test_path = (
        Path(args.test_failure_workbook)
        if args.test_failure_workbook
        else resolve_existing_path(
            DEFAULT_TEST_IMPORTANT_FAILURES_CANDIDATES,
            "test-data important-failures workbook",
        )
    )

    output_path = Path(
        args.enriched_failure_workbook_output
        or DEFAULT_FAILURE_ANALYSIS_WORKBOOK_OUTPUT_PATH
    )

    runtime_lookup = _build_runtime_lookup(
        Path(args.project_config)
    )

    # --------------------------------------------
    # Load tol_0 canonical failures independently.
    # --------------------------------------------

    entire_df = _load_failure_dataframe(
        entire_path,
        data_scope="entire_data",
        tolerance=0,
    )

    test_df = _load_failure_dataframe(
        test_path,
        data_scope="test_data",
        tolerance=0,
    )

    entire_df = normalize_failure_analysis_metadata(
        entire_df
    )

    test_df = normalize_failure_analysis_metadata(
        test_df
    )

    source_df = pd.concat(
        [entire_df, test_df],
        ignore_index=True,
        sort=False,
    )

    # --------------------------------------------
    # Diagnostics
    # --------------------------------------------

    print("\n=== Failure-analysis input diagnostics ===")
    print(f"Entire-data failures: {len(entire_df)}")
    print(f"Test-data failures:   {len(test_df)}")
    print(f"Combined rows:        {len(source_df)}")

    print("\nScheme counts:")
    print(
        source_df["scheme"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nModel counts:")
    print(
        source_df["model"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nSplit-mode counts:")
    print(
        source_df["split_mode"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nData-scope counts:")
    print(
        source_df["data_scope"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("==========================================\n")

    print("\nExperiment combination counts:")

    combination_counts = (
        source_df
        .groupby(
            [
                "scheme",
                "model",
                "split_mode",
                "data_scope",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .sort_values(
            [
                "scheme",
                "model",
                "split_mode",
                "data_scope",
            ]
        )
    )

    print(
        combination_counts.to_string(index=False)
    )

    print()

    # --------------------------------------------
    # Build the 8 experiment sheets.
    # --------------------------------------------

    sheet_frames: dict[str, pd.DataFrame] = {}

    for (
        scheme_name,
        model_name,
        split_mode,
        sheet_name,
    ) in FAILURE_ANALYSIS_SHEET_ORDER:

        subset = source_df[
            (source_df["scheme"] == scheme_name)
            & (source_df["model"] == model_name)
            & (source_df["split_mode"] == split_mode)
        ].copy()

        #''' for diagonistic checks '''
        if subset.empty:
            available = (
                source_df[
                    [
                        "scheme",
                        "model",
                        "split_mode",
                        "data_scope",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    [
                        "scheme",
                        "model",
                        "split_mode",
                        "data_scope",
                    ]
                )
            )

            raise ValueError(
                f"No failure rows found for expected experiment "
                f"{sheet_name!r} "
                f"(scheme={scheme_name!r}, "
                f"model={model_name!r}, "
                f"split_mode={split_mode!r}).\n"
                "Available metadata combinations:\n"
                f"{available.to_string(index=False)}"
            )

        test_count = int(
            subset["data_scope"].eq("test_data").sum()
        )

        entire_count = int(
            subset["data_scope"].eq("entire_data").sum()
        )

        print(
            f"{sheet_name}: "
            f"total={len(subset)}, "
            f"test={test_count}, "
            f"entire={entire_count}"
        )

        sheet_frames[sheet_name] = (
            _build_failure_analysis_sheet(
                subset,
                runtime_lookup,
            )
        )

    _write_failure_analysis_workbook(
        sheet_frames,
        output_path,
    )

    validation = _validate_failure_analysis_workbook(
        output_path,
        sheet_frames,
    )

    print(f"\nFailure analysis workbook: {output_path}")
    print(f"Entire source: {entire_path}")
    print(f"Test source:   {test_path}")

    print(
        "Validation: "
        f"sheet_count={validation['sheet_count']}, "
        f"sheet_names={validation['sheet_names']}, "
        f"validated_rows={validation['validated_rows']}, "
        f"row_counts={validation['row_counts']}"
    )

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build failure summary workbooks and enrich failure-analysis outputs.")
    parser.add_argument(
        "--mode",
        default=DEFAULT_ENRICHMENT_MODE,
        choices=["summary", "enrich"],
        help="Run the original entire-vs-test summary workbook flow or build the six-sheet entire-data failure-analysis workbook.",
    )
    parser.add_argument("--entire-important-workbook", default=str(DEFAULT_ENTIRE_IMPORTANT_PATH))
    parser.add_argument("--entire-base-workbook", default=None)
    parser.add_argument("--test-compare-workbook", default=None)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--failure-workbook",
        default=None,
        help="Entire-data important-failures workbook to enrich in --mode enrich.",
    )
    parser.add_argument(
        "--test-failure-workbook",
        default=None,
        help="Test-data important-failures workbook used for dataset-centric summaries in --mode enrich.",
    )
    parser.add_argument(
        "--enriched-failure-workbook-output",
        default=str(DEFAULT_FAILURE_ANALYSIS_WORKBOOK_OUTPUT_PATH),
        help="Output path for the six-sheet entire-data failure-analysis workbook.",
    )
    parser.add_argument(
        "--project-config",
        default=str(DEFAULT_PROJECT_CONFIG_PATH),
        help="Project config used to resolve the raw dataset for runtime lookups in the failure-analysis workbook.",
    )
    parser.add_argument(
        "--dataset-summary-output-path",
        default=str(DEFAULT_DATASET_FAILURE_SUMMARY_OUTPUT_PATH),
        help="Output workbook path for the dataset-centric failure summary.",
    )
    parser.add_argument(
        "--repeat-failure-output-path",
        default=str(DEFAULT_REPEAT_FAILURE_TABLE_OUTPUT_PATH),
        help="Output workbook path for the grouped repeat-failure table.",
    )
    parser.add_argument(
        "--heatmap-output-path",
        default=str(DEFAULT_HEATMAP_OUTPUT_PATH),
        help="Output image path for the dataset × scheme heatmap.",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
        help="Threshold applied to the predicted probability when setting high_confidence_wrong_prediction_flag.",
    )
    parser.add_argument(
        "--min-time-ratio-highlight-threshold",
        type=float,
        default=DEFAULT_MIN_TIME_RATIO_HIGHLIGHT_THRESHOLD,
        help="Highlight min_time_ratio_top2/min_time_ratio_top3 cells in green when below this threshold.",
    )
    parser.add_argument(
        "--time-gap-highlight-threshold",
        type=float,
        default=DEFAULT_TIME_GAP_HIGHLIGHT_THRESHOLD,
        help="Preserve the existing rule that highlights time_gap cells above this threshold in yellow.",
    )
    parser.add_argument(
        "--catastrophic-time-ratio-threshold",
        type=float,
        default=DEFAULT_CATASTROPHIC_TIME_RATIO_THRESHOLD,
        help="Include rows in the catastrophic sheet when time_ratio is at or above this threshold.",
    )
    parser.add_argument(
        "--catastrophic-predicted-time-threshold",
        type=float,
        default=DEFAULT_CATASTROPHIC_PREDICTED_TIME_THRESHOLD,
        help="Include rows in the catastrophic sheet when predicted_time is at or above this threshold.",
    )
    parser.add_argument(
        "--summary-sheet-name",
        default=DEFAULT_FILTER_SUMMARY_SHEET,
        help="Summary sheet name for important-failure workbooks.",
    )
    return parser


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "summary":
        run_summary_mode(args)
        return

    run_enrichment_mode(args)


if __name__ == "__main__":
    main()
