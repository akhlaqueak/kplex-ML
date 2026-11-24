"""Shared helpers for report parsing, metadata normalization, and workbook writing."""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


TOP3_TUPLE_PATTERN = re.compile(r"\(([^,]+),\s*([^,]+),\s*([^)]+)\)")


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def round_or_none(value):
    if value is None:
        return None
    return round(float(value), 4)


def resolve_runtime_column(
    columns,
    preferred_name,
    fallback_name,
    *,
    missing_context: str = "input dataframe",
):
    if preferred_name in columns:
        return preferred_name
    if fallback_name in columns:
        return fallback_name
    raise KeyError(
        f"Neither '{preferred_name}' nor '{fallback_name}' exists in the {missing_context}."
    )


def parse_nullable_literal(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (tuple, list)):
        return value
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None


def parse_saved_run_literal(value):
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


def parse_list_literal(value) -> list[object]:
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


def parse_report_literal(value):
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            matches = TOP3_TUPLE_PATTERN.findall(stripped)
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


def parse_candidate_prediction_tuple(value):
    parsed = parse_nullable_literal(value)
    if not isinstance(parsed, tuple) or len(parsed) < 3:
        return None
    return {
        "predicted_time": _safe_float(parsed[0]),
        "algo": str(parsed[1]),
        "probability": _safe_float(parsed[2]),
    }


def parse_fastest_algo_tuple(value):
    parsed = parse_nullable_literal(value)
    if not isinstance(parsed, tuple) or len(parsed) < 3:
        return None
    return {
        "fastest_runtime": _safe_float(parsed[0]),
        "algo": str(parsed[1]),
        "rank": int(parsed[2]) if parsed[2] not in (None, "None") else None,
    }


def parse_all_predictions_sorted(value):
    parsed = parse_nullable_literal(value)
    if not isinstance(parsed, list):
        return []
    entries = []
    for item in parsed:
        if not isinstance(item, tuple) or len(item) < 4:
            continue
        entries.append(
            {
                "time_ratio": _safe_float(item[0]),
                "algo": str(item[1]),
                "predicted_time": _safe_float(item[2]),
                "probability": _safe_float(item[3]),
            }
        )
    return entries


def _parse_literal(value):
    return parse_nullable_literal(value)


def _parse_report_literal(value):
    return parse_report_literal(value)


def warn_deprecated_path_usage(
    path: str | Path,
    canonical_path: str | Path,
    label: str,
    reason: str | None = None,
) -> None:
    resolved_path = Path(path)
    resolved_canonical = Path(canonical_path)
    if resolved_path == resolved_canonical:
        return

    message = (
        f"Using deprecated {label} path '{resolved_path}'. "
        f"Canonical path is '{resolved_canonical}'."
    )
    if reason:
        message += f" {reason}"
    warnings.warn(message, UserWarning, stacklevel=2)


def write_summary_sheet(output_path: str | Path, summary_df: pd.DataFrame, summary_sheet_name: str, autosize_columns) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        summary_df.to_excel(writer, sheet_name=summary_sheet_name, index=False)

    workbook = load_workbook(output_path)
    worksheet = workbook[summary_sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    autosize_columns(worksheet)
    workbook.save(output_path)
    workbook.close()


__all__ = [
    "_parse_literal",
    "_parse_report_literal",
    "_safe_float",
    "parse_all_predictions_sorted",
    "parse_candidate_prediction_tuple",
    "parse_fastest_algo_tuple",
    "parse_list_literal",
    "parse_nullable_literal",
    "parse_report_literal",
    "parse_saved_run_literal",
    "resolve_runtime_column",
    "round_or_none",
    "warn_deprecated_path_usage",
    "write_summary_sheet",
]
