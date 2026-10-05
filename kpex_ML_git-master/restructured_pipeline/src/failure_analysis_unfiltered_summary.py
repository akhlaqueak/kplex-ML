"""Build an unfiltered six-sheet failure-analysis workbook from saved wrong predictions."""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from restructured_pipeline.src.compare_wrong_predictions import autosize_columns
from restructured_pipeline.src.failure_analysis_summary import (
    ALL_ALGOS,
    DEFAULT_FAILURE_ANALYSIS_WORKBOOK_OUTPUT_PATH,
    DEFAULT_PROJECT_CONFIG_PATH,
    FAILURE_ANALYSIS_SHEET_ORDER,
    TOP_K_VALUES,
    _build_runtime_lookup,
    _build_sorted_prediction_entries,
    _format_all_predictions,
    _format_candidate_tuple,
    _format_fastest_tuple,
    _round_or_none,
    _safe_float,
    _select_best_topk_entry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_GLOB = "restructured_pipeline/outputs/wrong_cases/*/*/wrong_predictions_*_predict_inference_entire_data.xlsx"
DEFAULT_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/failure_analysis_summary_unfiltered_all_scheme_entire_data.xlsx"
)
SUMMARY_SHEET_NAME = "summary"
STRAT_ORDER = {"strat_y": 0, "strat_y_K": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-glob",
        default=DEFAULT_RAW_GLOB,
        help="Glob for existing entire-dataset wrong-predictions workbooks.",
    )
    parser.add_argument(
        "--project-config-path",
        type=Path,
        default=DEFAULT_PROJECT_CONFIG_PATH,
        help="Project config used to rebuild the runtime lookup for top-K analysis.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination workbook path.",
    )
    parser.add_argument(
        "--filtered-reference-path",
        type=Path,
        default=DEFAULT_FAILURE_ANALYSIS_WORKBOOK_OUTPUT_PATH,
        help="Filtered six-sheet workbook used only for validation count comparisons.",
    )
    return parser.parse_args()


def _sheet_sort_key(scheme: str, model: str) -> int:
    mapping = {(scheme_name, model_name): idx for idx, (scheme_name, model_name, _) in enumerate(FAILURE_ANALYSIS_SHEET_ORDER)}
    return mapping[(scheme, model)]


def _parse_list_literal(value) -> list[object]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
    return []


def _infer_scheme_model_strat(path: Path) -> tuple[str, str, str]:
    text = path.as_posix()
    if "scheme1" in text:
        scheme = "scheme 1"
    elif "scheme2" in text:
        scheme = "scheme 2"
    elif "scheme3" in text:
        scheme = "scheme 3"
    else:
        raise ValueError(f"Could not infer scheme from {path}")

    if "xgboost" in text:
        model = "XGB"
    elif "random_forest" in text:
        model = "RF"
    else:
        raise ValueError(f"Could not infer model from {path}")

    if "strat_y_K" in path.name:
        stratified_on = "strat_y_K"
    elif "strat_y" in path.name:
        stratified_on = "strat_y"
    else:
        raise ValueError(f"Could not infer split mode from {path}")

    return scheme, model, stratified_on


def _resolve_fastest_algo(row: pd.Series) -> str | None:
    for column in ("fastest_algorithms", "fastest_algorithms.1"):
        parsed = _parse_list_literal(row.get(column))
        if parsed:
            return str(parsed[0])

    fastest_time = _safe_float(row.get("fastest_time"))
    if fastest_time is None:
        return None

    matching_algos = []
    for algo in ALL_ALGOS:
        algo_time = _safe_float(row.get(algo))
        if algo_time is None:
            continue
        if abs(algo_time - fastest_time) <= 1e-12:
            matching_algos.append(algo)
    return matching_algos[0] if matching_algos else None


def _normalize_wrong_prediction_row(
    row: pd.Series,
    scheme: str,
    model: str,
    stratified_on: str,
    runtime_lookup: pd.DataFrame,
) -> dict[str, object]:
    runtime_row = runtime_lookup.loc[(str(row.get("dataset")), int(pd.to_numeric(pd.Series([row.get("k_value")]), errors="raise").iloc[0]))]
    fastest_algo = _resolve_fastest_algo(row)
    predicted_algo = row.get("predicted_algo", row.get("predicted"))
    predicted_time = _safe_float(row.get("predicted_time"))
    fastest_time = _safe_float(row.get("fastest_time"))
    predicted_probability = None
    if isinstance(predicted_algo, str):
        predicted_probability = _safe_float(row.get(f"prob_{predicted_algo}"))

    enriched_row = row.copy()
    enriched_row["fastest_algo"] = fastest_algo
    enriched_row["predicted_algo"] = predicted_algo
    enriched_row["predicted_probability"] = predicted_probability
    enriched_row["fastest_time"] = fastest_time
    enriched_row["predicted_time"] = predicted_time

    sorted_entries = _build_sorted_prediction_entries(enriched_row, runtime_row)
    fastest_rank = next((int(entry["rank"]) for entry in sorted_entries if entry["algo"] == fastest_algo), None)

    record: dict[str, object] = {
        "dataset": row.get("dataset"),
        "k_value": int(pd.to_numeric(pd.Series([row.get("k_value")]), errors="raise").iloc[0]),
        "model": model,
        "stratified_on": stratified_on,
        "fastest_algo_rank": fastest_rank,
        "fastest_algo_tuple": _format_fastest_tuple(fastest_time, fastest_algo, fastest_rank),
        "time_ratio": _round_or_none(_safe_float(row.get("time_ratio"))),
        "predicted_time": _round_or_none(predicted_time),
        "fastest_time": _round_or_none(fastest_time),
    }

    for top_k in TOP_K_VALUES:
        best_entry = _select_best_topk_entry(sorted_entries, top_k)
        record[f"top{top_k}_min_time_ratio"] = _round_or_none(
            _safe_float(best_entry.get("time_ratio")) if best_entry is not None else None
        )
        record[f"top{top_k}_candidate"] = _format_candidate_tuple(best_entry)

    record["all_predictions_sorted_by_prob"] = _format_all_predictions(sorted_entries)
    record["_scheme"] = scheme
    return record


def load_unfiltered_records(raw_glob: str, runtime_lookup: pd.DataFrame) -> pd.DataFrame:
    raw_paths = sorted(REPO_ROOT.glob(raw_glob))
    if not raw_paths:
        raise FileNotFoundError(f"No wrong-predictions workbooks found for glob: {raw_glob}")

    records: list[dict[str, object]] = []
    for path in raw_paths:
        scheme, model, stratified_on = _infer_scheme_model_strat(path)
        df = pd.read_excel(path)
        for _, row in df.iterrows():
            records.append(_normalize_wrong_prediction_row(row, scheme, model, stratified_on, runtime_lookup))

    combined = pd.DataFrame(records)
    return combined


def build_summary_sheet(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["_scheme", "model", "stratified_on"], dropna=False)
        .agg(
            wrong_prediction_count=("dataset", "size"),
            unique_datasets=("dataset", "nunique"),
            avg_time_ratio=("time_ratio", "mean"),
            max_time_ratio=("time_ratio", "max"),
            avg_predicted_time=("predicted_time", "mean"),
            avg_fastest_time=("fastest_time", "mean"),
        )
        .reset_index()
        .rename(columns={"_scheme": "scheme"})
    )
    for column in ("avg_time_ratio", "max_time_ratio", "avg_predicted_time", "avg_fastest_time"):
        summary[column] = summary[column].round(4)
    summary["_sheet_key"] = summary.apply(lambda row: _sheet_sort_key(row["scheme"], row["model"]), axis=1)
    summary["_strat_key"] = summary["stratified_on"].map(STRAT_ORDER).fillna(99)
    summary = summary.sort_values(["_sheet_key", "_strat_key"]).drop(columns=["_sheet_key", "_strat_key"]).reset_index(drop=True)
    return summary


def build_detail_sheets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sheet_frames: dict[str, pd.DataFrame] = {}
    for scheme, model, sheet_name in FAILURE_ANALYSIS_SHEET_ORDER:
        sheet_df = df[(df["_scheme"] == scheme) & (df["model"] == model)].copy()
        sheet_df = sheet_df.drop(columns=["_scheme", "time_ratio", "predicted_time", "fastest_time"])
        sheet_df = sheet_df.sort_values(
            by=["top1_min_time_ratio", "stratified_on", "dataset", "k_value", "fastest_algo_rank"],
            ascending=[False, True, True, True, True],
            na_position="last",
        ).reset_index(drop=True)
        sheet_frames[sheet_name] = sheet_df
    return sheet_frames


def write_workbook(summary_df: pd.DataFrame, sheet_frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name=SUMMARY_SHEET_NAME, index=False)
        for _, _, sheet_name in FAILURE_ANALYSIS_SHEET_ORDER:
            sheet_frames[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

    workbook = load_workbook(output_path)
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        autosize_columns(worksheet)
    workbook.save(output_path)
    workbook.close()


def validate_output(
    output_path: Path,
    summary_df: pd.DataFrame,
    raw_detail_df: pd.DataFrame,
    filtered_reference_path: Path,
) -> dict[str, object]:
    output_path = REPO_ROOT / output_path
    filtered_reference_path = REPO_ROOT / filtered_reference_path
    workbook = pd.ExcelFile(output_path)
    expected_sheets = [SUMMARY_SHEET_NAME, *[sheet_name for _, _, sheet_name in FAILURE_ANALYSIS_SHEET_ORDER]]
    if workbook.sheet_names != expected_sheets:
        raise ValueError(f"Unexpected sheets: expected {expected_sheets}, found {workbook.sheet_names}")

    summary_read_back = pd.read_excel(output_path, sheet_name=SUMMARY_SHEET_NAME)
    if len(summary_read_back) != len(summary_df):
        raise ValueError("Summary sheet row count mismatch after export.")

    detail_frames = {sheet_name: pd.read_excel(output_path, sheet_name=sheet_name) for sheet_name in expected_sheets[1:]}
    if sum(len(frame) for frame in detail_frames.values()) != len(raw_detail_df):
        raise ValueError("Exported detail row count does not match the unfiltered source row count.")
    for sheet_name, frame in detail_frames.items():
        if frame["stratified_on"].isna().any():
            raise ValueError(f"Missing stratified_on values in {sheet_name}")
        scheme, model = next((scheme_name, model_name) for scheme_name, model_name, candidate in FAILURE_ANALYSIS_SHEET_ORDER if candidate == sheet_name)
        expected_count = int(
            summary_read_back[
                (summary_read_back["scheme"] == scheme)
                & (summary_read_back["model"] == model)
            ]["wrong_prediction_count"].sum()
        )
        if len(frame) != expected_count:
            raise ValueError(f"Summary count mismatch for {sheet_name}: detail={len(frame)}, summary={expected_count}")

    low_runtime_source = int(((raw_detail_df["fastest_time"] < 5) & (raw_detail_df["predicted_time"] < 5)).sum())
    within_ratio_source = int((raw_detail_df["predicted_time"] <= (1.2 * raw_detail_df["fastest_time"])).sum())

    filtered_counts = {}
    if filtered_reference_path.exists():
        filtered_xls = pd.ExcelFile(filtered_reference_path)
        for _, _, sheet_name in FAILURE_ANALYSIS_SHEET_ORDER:
            filtered_counts[sheet_name] = len(pd.read_excel(filtered_reference_path, sheet_name=sheet_name))
            if len(detail_frames[sheet_name]) < filtered_counts[sheet_name]:
                raise ValueError(
                    f"Unfiltered sheet {sheet_name} has fewer rows than filtered reference: "
                    f"{len(detail_frames[sheet_name])} < {filtered_counts[sheet_name]}"
                )

    return {
        "output_path": str(output_path),
        "sheet_names": expected_sheets,
        "detail_row_counts": {sheet_name: int(len(detail_frames[sheet_name])) for sheet_name in detail_frames},
        "filtered_reference_row_counts": filtered_counts,
        "summary_rows": int(len(summary_df)),
        "rows_with_fastest_and_predicted_below_5_seconds": low_runtime_source,
        "rows_with_predicted_time_at_most_1_2x_fastest": within_ratio_source,
    }


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)
    runtime_lookup = _build_runtime_lookup(REPO_ROOT / args.project_config_path)
    detail_df = load_unfiltered_records(args.raw_glob, runtime_lookup)
    summary_df = build_summary_sheet(detail_df)
    sheet_frames = build_detail_sheets(detail_df)
    write_workbook(summary_df, sheet_frames, args.output_path)
    validation = validate_output(args.output_path, summary_df, detail_df, args.filtered_reference_path)

    print(f"Wrote {validation['output_path']}")
    print(f"Sheets: {validation['sheet_names']}")
    print("Detail row counts:")
    for sheet_name, count in validation["detail_row_counts"].items():
        print(f"- {sheet_name}: {count}")
    print("Filtered reference row counts:")
    for sheet_name, count in validation["filtered_reference_row_counts"].items():
        print(f"- {sheet_name}: {count}")
    print(
        "Filter-preserved rows: "
        f"fastest<5 and predicted<5 -> {validation['rows_with_fastest_and_predicted_below_5_seconds']}, "
        f"predicted<=1.2x fastest -> {validation['rows_with_predicted_time_at_most_1_2x_fastest']}"
    )


if __name__ == "__main__":
    main()
