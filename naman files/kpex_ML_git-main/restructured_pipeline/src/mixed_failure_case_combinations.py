"""Build mixed failure-case combination sheets from existing postprocessed workbooks."""

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

from restructured_pipeline.src.compare_wrong_predictions import autosize_columns
from restructured_pipeline.src.failure_analysis_summary import FAILURE_ANALYSIS_SHEET_ORDER
from restructured_pipeline.src.reporting_utils import (
    _parse_literal as shared_parse_literal,
    _safe_float,
    parse_all_predictions_sorted as shared_parse_all_predictions_sorted,
    parse_candidate_prediction_tuple as shared_parse_candidate_prediction_tuple,
    parse_fastest_algo_tuple as shared_parse_fastest_algo_tuple,
    round_or_none as shared_round_or_none,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = Path("restructured_pipeline/outputs/reports/mixed_failure_case_combinations.xlsx")
SUMMARY_SHEET = "summary"
PAIRING_RULES = [
    ("scheme 1", 2, "scheme 2", 1, "s1_top2__s2_top1"),
    ("scheme 1", 1, "scheme 2", 2, "s1_top1__s2_top2"),
    ("scheme 1", 2, "scheme 3", 1, "s1_top2__s3_top1"),
    ("scheme 1", 1, "scheme 3", 2, "s1_top1__s3_top2"),
]
MIX_OUTPUT_COLUMNS = [
    "dataset",
    "k_value",
    "scheme",
    "model",
    "source_split_label",
    "source_scheme",
    "source_rule",
    "selection_topk",
    "selected_min_time_ratio",
    "selected_candidate",
    "fastest_algo",
    "predicted_algo",
    "fastest_runtime",
    "predicted_time",
    "time_gap",
    "time_ratio",
    "predicted_probability",
    "top3_predicted_detailed",
    "top1_min_time_ratio",
    "top1_candidate",
    "top2_min_time_ratio",
    "top2_candidate",
    "fastest_algo_tuple",
]
MASTER_MERGE_COLUMNS = [
    "dataset",
    "k_value",
    "fastest_algo_tuple",
    "top1_candidate",
    "top2_candidate",
    "top1_min_time_ratio",
    "top2_min_time_ratio",
]
NORMALIZED_MERGE_KEY_PREFIX = "__merge_key__"
STRING_KEY_COLUMNS = {
    "dataset",
    "scheme",
    "model",
    "fastest_algo",
    "predicted_algo",
}
NUMERIC_KEY_COLUMNS = {
    "k_value",
    "top1_min_time_ratio",
    "top2_min_time_ratio",
}
LITERAL_KEY_COLUMNS = {
    "fastest_algo_tuple",
    "top1_candidate",
    "top2_candidate",
    "top3_candidate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master-workbook",
        type=Path,
        required=True,
        help="Explicit unfiltered six-sheet workbook used to map split/source labels.",
    )
    parser.add_argument(
        "--filtered-workbook",
        type=Path,
        required=True,
        help="Explicit filtered six-sheet failure-analysis workbook used as the source pool.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination workbook path.",
    )
    parser.add_argument(
        "--min-ratio-threshold",
        type=float,
        default=2.0,
        help="Keep rows only when the selected top-k minimum ratio is strictly above this threshold.",
    )
    return parser.parse_args()


def resolve_input_path(explicit: Path, label: str) -> Path:
    resolved = REPO_ROOT / explicit if not explicit.is_absolute() else explicit
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"{label} not found: {resolved}")


def _round_or_none(value):
    return shared_round_or_none(value)


def _parse_literal(value):
    return shared_parse_literal(value)


def _parse_candidate_tuple(value):
    return shared_parse_candidate_prediction_tuple(value)


def _parse_fastest_tuple(value):
    return shared_parse_fastest_algo_tuple(value)


def _parse_all_predictions(value):
    return shared_parse_all_predictions_sorted(value)


def _format_top3_detailed(entries: list[dict[str, object]]) -> str:
    formatted = []
    for entry in entries[:3]:
        ratio = _round_or_none(entry.get("time_ratio"))
        predicted_time = _round_or_none(entry.get("predicted_time"))
        probability = _round_or_none(entry.get("probability"))
        ratio_text = "None" if ratio is None else f"{ratio:.4f}"
        time_text = "None" if predicted_time is None else f"{predicted_time:.4f}"
        prob_text = "None" if probability is None else f"{probability:.4f}"
        formatted.append(f"({ratio_text}, {entry['algo']!r}, {time_text}, {prob_text})")
    return "[" + ", ".join(formatted) + "]"


def _normalized_merge_column_name(column: str) -> str:
    return f"{NORMALIZED_MERGE_KEY_PREFIX}{column}"


def _is_missing_key_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (tuple, list, dict, set)):
        return False
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _format_numeric_key(value, *, integer_only: bool = False) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return ""
    if integer_only:
        return str(int(round(numeric)))
    return f"{numeric:.12g}"


def _canonicalize_literal_key_value(value, *, nested: bool = False) -> str:
    if _is_missing_key_value(value):
        return "None" if nested else ""
    if isinstance(value, tuple):
        return "(" + ", ".join(_canonicalize_literal_key_value(item, nested=True) for item in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(_canonicalize_literal_key_value(item, nested=True) for item in value) + "]"
    if isinstance(value, str):
        stripped = value.strip()
        return repr(stripped) if nested else stripped
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)

    numeric = _safe_float(value)
    if numeric is not None:
        return f"{numeric:.12g}"

    return str(value).strip()


def _normalize_literal_merge_key(value) -> str:
    if _is_missing_key_value(value):
        return ""

    parsed = _parse_literal(value)
    if isinstance(parsed, (tuple, list)):
        return _canonicalize_literal_key_value(parsed)

    if isinstance(value, str):
        return value.strip()
    return _canonicalize_literal_key_value(value)


def _normalize_merge_key_series(series: pd.Series, column: str) -> pd.Series:
    if column in STRING_KEY_COLUMNS:
        return series.apply(lambda value: "" if _is_missing_key_value(value) else str(value).strip())
    if column in NUMERIC_KEY_COLUMNS:
        integer_only = column == "k_value"
        return series.apply(lambda value: _format_numeric_key(value, integer_only=integer_only))
    if column in LITERAL_KEY_COLUMNS:
        return series.apply(_normalize_literal_merge_key)
    return series.apply(lambda value: "" if _is_missing_key_value(value) else str(value).strip())


def _add_normalized_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    merge_candidate_columns = STRING_KEY_COLUMNS | NUMERIC_KEY_COLUMNS | LITERAL_KEY_COLUMNS | set(MASTER_MERGE_COLUMNS)
    for column in merge_candidate_columns:
        if column in normalized.columns:
            normalized[_normalized_merge_column_name(column)] = _normalize_merge_key_series(normalized[column], column)
    return normalized


def _select_effective_merge_keys(master_df: pd.DataFrame, filtered_df: pd.DataFrame) -> list[str]:
    effective_keys = [
        _normalized_merge_column_name("dataset"),
        _normalized_merge_column_name("k_value"),
    ]

    for column in MASTER_MERGE_COLUMNS[2:]:
        normalized_col = _normalized_merge_column_name(column)
        if normalized_col not in master_df.columns or normalized_col not in filtered_df.columns:
            continue
        master_has_values = master_df[normalized_col].ne("").any()
        filtered_has_values = filtered_df[normalized_col].ne("").any()
        if master_has_values and filtered_has_values:
            effective_keys.append(normalized_col)

    return effective_keys


def _drop_normalized_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [column for column in df.columns if str(column).startswith(NORMALIZED_MERGE_KEY_PREFIX)]
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)


def _candidate_merge_key_sets(master_df: pd.DataFrame, filtered_df: pd.DataFrame) -> list[list[str]]:
    dataset_key = _normalized_merge_column_name("dataset")
    k_value_key = _normalized_merge_column_name("k_value")
    fastest_tuple_key = _normalized_merge_column_name("fastest_algo_tuple")

    key_sets: list[list[str]] = []

    def add_key_set(keys: list[str]) -> None:
        seen = set()
        deduped = []
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        if deduped and deduped not in key_sets:
            key_sets.append(deduped)

    add_key_set(_select_effective_merge_keys(master_df, filtered_df))

    if (
        fastest_tuple_key in master_df.columns
        and fastest_tuple_key in filtered_df.columns
        and master_df[fastest_tuple_key].ne("").any()
        and filtered_df[fastest_tuple_key].ne("").any()
    ):
        add_key_set([dataset_key, k_value_key, fastest_tuple_key])

    add_key_set([dataset_key, k_value_key])
    return key_sets


def load_workbook_frames(workbook_path: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for scheme, model, sheet_name in FAILURE_ANALYSIS_SHEET_ORDER:
        try:
            sheet_df = pd.read_excel(workbook_path, sheet_name=sheet_name)
        except ValueError:
            continue
        sheet_df = sheet_df.copy()
        sheet_df["scheme"] = scheme
        sheet_df["model"] = model
        sheet_df["source_sheet"] = sheet_name
        frames[sheet_name] = sheet_df
    if not frames:
        raise ValueError(f"No scheme/model sheets found in {workbook_path}")
    return frames


def attach_split_labels(master_frames: dict[str, pd.DataFrame], filtered_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    mapped_frames: dict[str, pd.DataFrame] = {}
    for sheet_name, filtered_df in filtered_frames.items():
        master_df = _add_normalized_merge_keys(master_frames[sheet_name])
        filtered_df = _add_normalized_merge_keys(filtered_df)
        merged = None
        for merge_keys in _candidate_merge_key_sets(master_df, filtered_df):
            merge_df = (
                master_df[merge_keys + ["stratified_on"]]
                .drop_duplicates()
                .groupby(merge_keys, dropna=False)["stratified_on"]
                .agg(lambda values: "|".join(sorted({str(value) for value in values if pd.notna(value)})))
                .reset_index()
            )
            candidate_merged = filtered_df.merge(merge_df, on=merge_keys, how="left", validate="many_to_one")
            if not candidate_merged["stratified_on"].isna().any():
                merged = candidate_merged
                break

        if merged is None:
            raise ValueError(f"Missing stratified_on mapping for rows in {sheet_name}")
        merged = merged.rename(columns={"stratified_on": "source_split_label"})
        mapped_frames[sheet_name] = _drop_normalized_merge_keys(merged)
    return mapped_frames


def _select_rows_for_rule(df: pd.DataFrame, source_rule: str, selection_topk: int, threshold: float) -> pd.DataFrame:
    ratio_col = f"top{selection_topk}_min_time_ratio"
    candidate_col = f"top{selection_topk}_candidate"
    selection_mask = df[ratio_col].apply(lambda value: (_safe_float(value) or -1) > threshold).astype(bool)
    selected = df.loc[selection_mask].copy()
    selected["source_rule"] = source_rule
    selected["selection_topk"] = selection_topk
    selected["selected_min_time_ratio"] = selected[ratio_col].apply(_safe_float)
    selected["selected_candidate"] = selected[candidate_col]
    selected["source_scheme"] = selected["scheme"]

    fastest_info = selected["fastest_algo_tuple"].apply(_parse_fastest_tuple)
    candidate_info = selected["selected_candidate"].apply(_parse_candidate_tuple)
    top3_info = selected["all_predictions_sorted_by_prob"].apply(_parse_all_predictions)

    selected["fastest_algo"] = fastest_info.apply(lambda item: item["algo"] if item else pd.NA)
    selected["fastest_runtime"] = fastest_info.apply(lambda item: _round_or_none(item["fastest_runtime"]) if item else pd.NA)
    selected["predicted_algo"] = candidate_info.apply(lambda item: item["algo"] if item else pd.NA)
    selected["predicted_time"] = candidate_info.apply(lambda item: _round_or_none(item["predicted_time"]) if item else pd.NA)
    selected["predicted_probability"] = candidate_info.apply(lambda item: _round_or_none(item["probability"]) if item else pd.NA)
    selected["time_gap"] = selected.apply(
        lambda row: _round_or_none((_safe_float(row["predicted_time"]) or 0) - (_safe_float(row["fastest_runtime"]) or 0))
        if _safe_float(row["predicted_time"]) is not None and _safe_float(row["fastest_runtime"]) is not None
        else pd.NA,
        axis=1,
    )
    selected["time_ratio"] = selected["selected_min_time_ratio"].apply(_round_or_none)
    selected["top3_predicted_detailed"] = top3_info.apply(_format_top3_detailed)

    for column in ("top1_min_time_ratio", "top2_min_time_ratio", "selected_min_time_ratio"):
        if column in selected.columns:
            selected[column] = selected[column].apply(_round_or_none)
    return selected


def build_mix_sheet(
    combined_df: pd.DataFrame,
    left_scheme: str,
    left_topk: int,
    right_scheme: str,
    right_topk: int,
    source_rule: str,
    threshold: float,
) -> pd.DataFrame:
    left_rows = _select_rows_for_rule(
        combined_df[combined_df["scheme"] == left_scheme].copy(),
        source_rule=source_rule,
        selection_topk=left_topk,
        threshold=threshold,
    )
    right_rows = _select_rows_for_rule(
        combined_df[combined_df["scheme"] == right_scheme].copy(),
        source_rule=source_rule,
        selection_topk=right_topk,
        threshold=threshold,
    )
    mix_df = pd.concat([left_rows, right_rows], ignore_index=True, sort=False)
    mix_df = mix_df[MIX_OUTPUT_COLUMNS].sort_values(["dataset", "k_value", "scheme", "model"]).reset_index(drop=True)
    return mix_df


def build_summary(summary_rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(summary_rows)


def write_workbook(output_path: Path, sheet_frames: dict[str, pd.DataFrame], summary_df: pd.DataFrame) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name=SUMMARY_SHEET, index=False)
        for sheet_name, frame in sheet_frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

    workbook = load_workbook(output_path)
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        autosize_columns(worksheet)
    workbook.save(output_path)
    workbook.close()


def validate_output(sheet_frames: dict[str, pd.DataFrame], threshold: float) -> dict[str, object]:
    expected_sheet_names = [rule[-1] for rule in PAIRING_RULES]
    actual_sheet_names = list(sheet_frames.keys())
    if actual_sheet_names != expected_sheet_names:
        raise ValueError(f"Unexpected sheet ordering: expected {expected_sheet_names}, found {actual_sheet_names}")

    validation_rows = []
    for sheet_name, frame in sheet_frames.items():
        if frame.empty:
            raise ValueError(f"{sheet_name} is empty.")
        schemes = sorted(frame["source_scheme"].dropna().unique().tolist())
        if schemes not in (["scheme 1", "scheme 2"], ["scheme 1", "scheme 3"]):
            raise ValueError(f"{sheet_name} contains disallowed schemes: {schemes}")
        if frame["selected_min_time_ratio"].apply(lambda value: (_safe_float(value) or -1) <= threshold).any():
            raise ValueError(f"{sheet_name} contains rows at or below the threshold.")
        sorted_frame = frame.sort_values(["dataset", "k_value", "scheme", "model"]).reset_index(drop=True)
        if not frame.reset_index(drop=True).equals(sorted_frame):
            raise ValueError(f"{sheet_name} is not sorted by dataset and k_value.")
        if frame["source_split_label"].isna().any():
            raise ValueError(f"{sheet_name} is missing source split labels.")
        validation_rows.append(
            {
                "sheet_name": sheet_name,
                "rows": int(len(frame)),
                "scheme_1_rows": int((frame["source_scheme"] == "scheme 1").sum()),
                "scheme_2_rows": int((frame["source_scheme"] == "scheme 2").sum()),
                "scheme_3_rows": int((frame["source_scheme"] == "scheme 3").sum()),
            }
        )

    return {
        "sheet_names": actual_sheet_names,
        "sheet_counts": validation_rows,
        "threshold": threshold,
    }


def main() -> None:
    args = parse_args()
    master_path = resolve_input_path(args.master_workbook, "master workbook")
    filtered_path = resolve_input_path(args.filtered_workbook, "filtered workbook")
    output_path = REPO_ROOT / args.output_path if not args.output_path.is_absolute() else args.output_path

    master_frames = load_workbook_frames(master_path)
    filtered_frames = load_workbook_frames(filtered_path)
    mapped_frames = attach_split_labels(master_frames, filtered_frames)
    combined_df = pd.concat(mapped_frames.values(), ignore_index=True, sort=False)

    sheet_frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    for left_scheme, left_topk, right_scheme, right_topk, sheet_name in PAIRING_RULES:
        source_rule = f"{left_scheme.replace(' ', '')}_top{left_topk}+{right_scheme.replace(' ', '')}_top{right_topk}"
        frame = build_mix_sheet(
            combined_df,
            left_scheme=left_scheme,
            left_topk=left_topk,
            right_scheme=right_scheme,
            right_topk=right_topk,
            source_rule=source_rule,
            threshold=args.min_ratio_threshold,
        )
        sheet_frames[sheet_name] = frame
        summary_rows.append(
            {
                "sheet_name": sheet_name,
                "source_rule": source_rule,
                "allowed_schemes": f"{left_scheme}, {right_scheme}",
                "threshold_rule": f"selected_min_time_ratio > {args.min_ratio_threshold}",
                "total_rows": int(len(frame)),
                "scheme_1_rows": int((frame["source_scheme"] == "scheme 1").sum()),
                "scheme_2_rows": int((frame["source_scheme"] == "scheme 2").sum()),
                "scheme_3_rows": int((frame["source_scheme"] == "scheme 3").sum()),
                "duplicate_rows_retained": int(frame.duplicated(subset=MIX_OUTPUT_COLUMNS, keep=False).sum()),
            }
        )

    summary_df = build_summary(summary_rows)
    write_workbook(output_path, sheet_frames, summary_df)
    validation = validate_output(sheet_frames, args.min_ratio_threshold)

    print(f"Mixed workbook: {output_path}")
    print(f"Master workbook: {master_path}")
    print(f"Filtered workbook: {filtered_path}")
    print(f"Generated sheets: {validation['sheet_names']}")
    for row in validation["sheet_counts"]:
        print(
            f"{row['sheet_name']}: rows={row['rows']}, "
            f"scheme1={row['scheme_1_rows']}, scheme2={row['scheme_2_rows']}, scheme3={row['scheme_3_rows']}"
        )


if __name__ == "__main__":
    main()
