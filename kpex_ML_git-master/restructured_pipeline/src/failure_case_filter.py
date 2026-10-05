"""Filter consolidated compare-reports down to important failure cases only.
NOTE: standalone CLI is deprecated/commented because compare_wrong_predictions.py now owns the active compare -> tolerance -> important-failure pipeline; annotate_failure_cases remains the shared helper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from restructured_pipeline.src.compare_wrong_predictions import autosize_columns, write_by_inference_on


DEFAULT_INPUT_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data.xlsx"),
    Path("restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_schema_entire_data.xlsx"),
]
DEFAULT_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data.xlsx"
)
DEFAULT_SUMMARY_SHEET = "filter_summary"
DEFAULT_SMALL_RUNTIME_THRESHOLD = 5.0
DEFAULT_SLOWDOWN_RATIO_THRESHOLD = 1.2
DEFAULT_TIME_GAP_HIGHLIGHT_THRESHOLD = 100.0
REASON_COLUMN = "failure_case_filter_reason"
SMALL_RUNTIME_REASON = "excluded_small_runtime"
SLOWDOWN_REASON = "excluded_within_ratio_threshold"
IMPORTANT_REASON = "important_failure"


def resolve_default_input_path() -> Path:
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_INPUT_CANDIDATES[0]


def resolve_runtime_column(columns, preferred_name, fallback_name):
    if preferred_name in columns:
        return preferred_name
    if fallback_name in columns:
        return fallback_name
    raise KeyError(f"Neither '{preferred_name}' nor '{fallback_name}' exists in the input workbook.")


def insert_reason_column(df: pd.DataFrame, reason_series: pd.Series) -> pd.DataFrame:
    df = df.copy()
    if REASON_COLUMN in df.columns:
        df[REASON_COLUMN] = reason_series
        return df

    insert_at = len(df.columns)
    if "failure_severity" in df.columns:
        insert_at = df.columns.get_loc("failure_severity") + 1
    df.insert(insert_at, REASON_COLUMN, reason_series)
    return df


def annotate_failure_cases(
    df: pd.DataFrame,
    small_runtime_threshold: float,
    slowdown_ratio_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fastest_col = resolve_runtime_column(df.columns, "fastest_runtime", "fastest_time")
    predicted_col = resolve_runtime_column(df.columns, "predicted_runtime", "predicted_time")

    annotated_df = df.copy()
    fastest = pd.to_numeric(annotated_df[fastest_col], errors="coerce")
    predicted = pd.to_numeric(annotated_df[predicted_col], errors="coerce")

    excluded_small_runtime = fastest.lt(small_runtime_threshold) & predicted.lt(small_runtime_threshold)
    excluded_within_ratio = predicted.le(slowdown_ratio_threshold * fastest)

    reason_values = []
    for is_small, is_within in zip(excluded_small_runtime, excluded_within_ratio):
        reasons = []
        if bool(is_small):
            reasons.append(SMALL_RUNTIME_REASON)
        if bool(is_within):
            reasons.append(SLOWDOWN_REASON)
        if not reasons:
            reasons.append(IMPORTANT_REASON)
        reason_values.append(";".join(reasons))

    annotated_df = insert_reason_column(annotated_df, pd.Series(reason_values, index=annotated_df.index))
    important_df = annotated_df.loc[~(excluded_small_runtime | excluded_within_ratio)].copy()

    summary_rows = [
        {
            "sheet_name": annotated_df["inference_on"].iloc[0] if not annotated_df.empty and "inference_on" in annotated_df.columns else "",
            "total_rows": int(len(annotated_df)),
            "small_runtime_threshold": float(small_runtime_threshold),
            "slowdown_ratio_threshold": float(slowdown_ratio_threshold),
            "excluded_small_runtime_count": int(excluded_small_runtime.sum()),
            "excluded_within_ratio_count": int(excluded_within_ratio.sum()),
            "excluded_by_both_rules_count": int((excluded_small_runtime & excluded_within_ratio).sum()),
            "excluded_any_count": int((excluded_small_runtime | excluded_within_ratio).sum()),
            "important_failure_count": int(len(important_df)),
        }
    ]

    return important_df, pd.DataFrame(summary_rows)


def load_compare_workbook(input_workbook: str | Path) -> dict[str, pd.DataFrame]:
    workbook_path = Path(input_workbook)
    if not workbook_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {workbook_path}")
    return pd.read_excel(workbook_path, sheet_name=None)


def write_summary_sheet(output_path: str | Path, summary_df: pd.DataFrame, summary_sheet_name: str) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        summary_df.to_excel(writer, sheet_name=summary_sheet_name, index=False)

    workbook = load_workbook(output_path)
    worksheet = workbook[summary_sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    autosize_columns(worksheet)
    workbook.save(output_path)
    workbook.close()


def run_failure_case_filter(
    input_workbook: str | Path,
    output_path: str | Path,
    small_runtime_threshold: float = DEFAULT_SMALL_RUNTIME_THRESHOLD,
    slowdown_ratio_threshold: float = DEFAULT_SLOWDOWN_RATIO_THRESHOLD,
    time_gap_highlight_threshold: float = DEFAULT_TIME_GAP_HIGHLIGHT_THRESHOLD,
    summary_sheet_name: str = DEFAULT_SUMMARY_SHEET,
):
    workbook_sheets = load_compare_workbook(input_workbook)

    important_frames = []
    summary_frames = []
    for _, df in workbook_sheets.items():
        if df.empty:
            continue
        important_df, summary_df = annotate_failure_cases(
            df,
            small_runtime_threshold=small_runtime_threshold,
            slowdown_ratio_threshold=slowdown_ratio_threshold,
        )
        if not important_df.empty:
            important_frames.append(important_df)
        summary_frames.append(summary_df)

    if not summary_frames:
        raise ValueError("The input workbook did not contain any data rows.")

    summary_df = pd.concat(summary_frames, ignore_index=True)
    total_row = {
        "sheet_name": "ALL",
        "total_rows": int(summary_df["total_rows"].sum()),
        "small_runtime_threshold": float(small_runtime_threshold),
        "slowdown_ratio_threshold": float(slowdown_ratio_threshold),
        "excluded_small_runtime_count": int(summary_df["excluded_small_runtime_count"].sum()),
        "excluded_within_ratio_count": int(summary_df["excluded_within_ratio_count"].sum()),
        "excluded_by_both_rules_count": int(summary_df["excluded_by_both_rules_count"].sum()),
        "excluded_any_count": int(summary_df["excluded_any_count"].sum()),
        "important_failure_count": int(summary_df["important_failure_count"].sum()),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

    important_df = pd.concat(important_frames, ignore_index=True) if important_frames else pd.DataFrame(columns=[REASON_COLUMN])
    write_by_inference_on(
        important_df,
        output_path=str(output_path),
        time_gap_highlight_threshold=time_gap_highlight_threshold,
    )
    write_summary_sheet(output_path, summary_df, summary_sheet_name)

    return {
        "input_workbook": str(Path(input_workbook)),
        "output_path": str(Path(output_path)),
        "small_runtime_threshold": small_runtime_threshold,
        "slowdown_ratio_threshold": slowdown_ratio_threshold,
        "time_gap_highlight_threshold": time_gap_highlight_threshold,
        "important_failure_count": int(total_row["important_failure_count"]),
        "excluded_small_runtime_count": int(total_row["excluded_small_runtime_count"]),
        "excluded_within_ratio_count": int(total_row["excluded_within_ratio_count"]),
        "excluded_by_both_rules_count": int(total_row["excluded_by_both_rules_count"]),
        "excluded_any_count": int(total_row["excluded_any_count"]),
    }


# -----------------------------------------------------------------------------
# DEPRECATED STANDALONE CLI (commented out): active pipeline is integrated in
# compare_wrong_predictions.py; keep this source text temporarily for reference.
# -----------------------------------------------------------------------------
# def build_arg_parser():
#     parser = argparse.ArgumentParser(
#         description="Filter consolidated compare-reports to important failure cases only."
#     )
#     parser.add_argument(
#         "--input-workbook",
#         default=str(resolve_default_input_path()),
#         help="Existing consolidated compare-report workbook to read.",
#     )
#     parser.add_argument(
#         "--output-path",
#         default=str(DEFAULT_OUTPUT_PATH),
#         help="Excel workbook path for the filtered important-failures report.",
#     )
#     parser.add_argument(
#         "--small-runtime-threshold",
#         type=float,
#         default=DEFAULT_SMALL_RUNTIME_THRESHOLD,
#         help="Exclude rows where both fastest and predicted runtimes are below this value.",
#     )
#     parser.add_argument(
#         "--slowdown-ratio-threshold",
#         type=float,
#         default=DEFAULT_SLOWDOWN_RATIO_THRESHOLD,
#         help="Exclude rows where predicted runtime is within this multiple of the fastest runtime.",
#     )
#     parser.add_argument(
#         "--time-gap-highlight-threshold",
#         type=float,
#         default=DEFAULT_TIME_GAP_HIGHLIGHT_THRESHOLD,
#         help="Highlight only time_gap cells above this absolute threshold in yellow.",
#     )
#     parser.add_argument(
#         "--summary-sheet-name",
#         default=DEFAULT_SUMMARY_SHEET,
#         help="Worksheet name used for the summary of exclusions and remaining rows.",
#     )
#     return parser
#
#
# def main(argv=None):
#     parser = build_arg_parser()
#     args = parser.parse_args(argv)
#
#     result = run_failure_case_filter(
#         input_workbook=args.input_workbook,
#         output_path=args.output_path,
#         small_runtime_threshold=args.small_runtime_threshold,
#         slowdown_ratio_threshold=args.slowdown_ratio_threshold,
#         time_gap_highlight_threshold=args.time_gap_highlight_threshold,
#         summary_sheet_name=args.summary_sheet_name,
#     )
#
#     print(f"Failure-case filter complete: {result['output_path']}")
#     print(f"Input workbook: {result['input_workbook']}")
#     print(f"Small runtime threshold: {result['small_runtime_threshold']}")
#     print(f"Slowdown ratio threshold: {result['slowdown_ratio_threshold']}")
#     print(f"Important failures: {result['important_failure_count']}")
#     print(f"Excluded by small runtime rule: {result['excluded_small_runtime_count']}")
#     print(f"Excluded by slowdown ratio rule: {result['excluded_within_ratio_count']}")
#     print(f"Excluded by both rules: {result['excluded_by_both_rules_count']}")
#     print(f"Excluded by any rule: {result['excluded_any_count']}")
#
#
# if __name__ == "__main__":
#     main()
