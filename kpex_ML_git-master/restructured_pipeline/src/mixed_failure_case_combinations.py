"""Build mixed failure-case combination sheets from existing postprocessed workbooks."""

from __future__ import annotations

import argparse
import ast
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER_CANDIDATES = [
    Path(
        "restructured_pipeline/outputs/reports/existing_code_run_20260511/"
        "failure_analysis_summary_unfiltered_with_generated_filtered_reference.xlsx"
    ),
    Path("restructured_pipeline/outputs/reports/failure_analysis_summary_unfiltered_with_generated_filtered_reference.xlsx"),
]
DEFAULT_FILTERED_CANDIDATES = [
    Path(
        "restructured_pipeline/outputs/reports/existing_code_run_20260511/"
        "failure_analysis_summary_filtered_all_scheme_entire_data_from_one_sheet.xlsx"
    ),
    Path("restructured_pipeline/outputs/reports/failure_analysis_summary_filtered_all_scheme_entire_data_from_one_sheet.xlsx"),
]
DEFAULT_OUTPUT_PATH = Path("restructured_pipeline/outputs/reports/mixed_failure_case_combinations.xlsx")
SUMMARY_SHEET = "summary"
DEFAULT_SCOPE = "entire_data"
MASTER_MERGE_COLUMNS = [
    "dataset",
    "k_value",
    "fastest_algo_tuple",
    "top1_candidate",
    "top2_candidate",
    "top3_candidate",
    "top1_min_time_ratio",
    "top2_min_time_ratio",
    "top3_min_time_ratio",
]
MIX_OUTPUT_COLUMNS = [
    "dataset",
    "k_value",
    "scheme",
    "model",
    "source_split_label",
    "source_scope",
    "source_scheme",
    "source_rule",
    "selection_topk",
    "selected_min_time_ratio",
    "selected_candidate",
    "threshold_rule",
    "variant_kind",
    "variant_label",
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
    "top3_min_time_ratio",
    "top3_candidate",
    "fastest_algo_tuple",
]
SCHEME_PAIRS = [("scheme 1", "scheme 2"), ("scheme 1", "scheme 3")]
TOPK_PAIRINGS = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (2, 3),
    (3, 1),
    (3, 2),
]
BASE_PAIRINGS = {(1, 1), (1, 2), (2, 1), (2, 2)}
THRESHOLDS = (1.5, 2.0)
SPLIT_VARIANTS = ("strat_y", "strat_y_K")
MODEL_VARIANTS = ("RF", "XGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-workbook", type=Path, default=None)
    parser.add_argument("--filtered-workbook", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=2.0,
        help="Default threshold used for the base pairwise sheets.",
    )
    return parser.parse_args()


def resolve_existing_path(explicit: Path | None, candidates: list[Path], label: str) -> Path:
    if explicit is not None:
        resolved = REPO_ROOT / explicit if not explicit.is_absolute() else explicit
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"{label} not found: {resolved}")
    for candidate in candidates:
        resolved = REPO_ROOT / candidate
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Could not resolve {label} from candidates: {candidates}")


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _round_or_none(value):
    if value is None:
        return None
    return round(float(value), 4)


def _parse_literal(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (tuple, list)):
        return value
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None


def _parse_candidate_tuple(value):
    parsed = _parse_literal(value)
    if not isinstance(parsed, tuple) or len(parsed) < 3:
        return None
    return {
        "predicted_time": _safe_float(parsed[0]),
        "algo": str(parsed[1]),
        "probability": _safe_float(parsed[2]),
    }


def _parse_fastest_tuple(value):
    parsed = _parse_literal(value)
    if not isinstance(parsed, tuple) or len(parsed) < 3:
        return None
    rank_value = parsed[2]
    rank = None if rank_value in (None, "None") else int(rank_value)
    return {
        "fastest_runtime": _safe_float(parsed[0]),
        "algo": str(parsed[1]),
        "rank": rank,
    }


def _parse_all_predictions(value):
    parsed = _parse_literal(value)
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
        sheet_df["source_scope"] = DEFAULT_SCOPE
        frames[sheet_name] = sheet_df
    if not frames:
        raise ValueError(f"No scheme/model sheets found in {workbook_path}")
    return frames


def attach_split_labels(master_frames: dict[str, pd.DataFrame], filtered_frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    mapped_frames: dict[str, pd.DataFrame] = {}
    for sheet_name, filtered_df in filtered_frames.items():
        master_df = master_frames[sheet_name]
        merge_df = (
            master_df[MASTER_MERGE_COLUMNS + ["stratified_on"]]
            .drop_duplicates()
            .groupby(MASTER_MERGE_COLUMNS, dropna=False)["stratified_on"]
            .agg(lambda values: "|".join(sorted({str(value) for value in values if pd.notna(value)})))
            .reset_index()
        )
        merged = filtered_df.merge(merge_df, on=MASTER_MERGE_COLUMNS, how="left", validate="many_to_one")
        if merged["stratified_on"].isna().any():
            raise ValueError(f"Missing stratified_on mapping for rows in {sheet_name}")
        merged = merged.rename(columns={"stratified_on": "source_split_label"})
        mapped_frames[sheet_name] = merged
    return mapped_frames


def _scheme_short(scheme_name: str) -> str:
    return "s1" if scheme_name == "scheme 1" else "s2" if scheme_name == "scheme 2" else "s3"


def _scheme_long(scheme_name: str) -> str:
    return scheme_name.replace(" ", "")


def _threshold_suffix(threshold: float) -> str:
    return str(threshold).replace(".", "p")


def _sheet_name(
    left_scheme: str,
    left_topk: int,
    right_scheme: str,
    right_topk: int,
    variant_kind: str,
    variant_label: str | None,
) -> str:
    if variant_kind == "base":
        return f"{_scheme_long(left_scheme)}_top{left_topk}__{_scheme_long(right_scheme)}_top{right_topk}"
    base = f"{_scheme_short(left_scheme)}_t{left_topk}__{_scheme_short(right_scheme)}_t{right_topk}"
    if variant_kind == "threshold":
        return f"{base}__{variant_label}"
    if variant_kind == "split":
        return f"{base}__{variant_label}"
    if variant_kind == "model":
        return f"{base}__{variant_label}"
    raise ValueError(f"Unknown variant kind: {variant_kind}")


def _rule_label(left_scheme: str, left_topk: int, right_scheme: str, right_topk: int) -> str:
    return f"{_scheme_long(left_scheme)}_top{left_topk} + {_scheme_long(right_scheme)}_top{right_topk}"


def _select_rows(
    df: pd.DataFrame,
    scheme_name: str,
    selection_topk: int,
    threshold: float,
    split_label: str | None,
    scope_label: str | None,
    model_label: str | None,
    source_rule: str,
    variant_kind: str,
    variant_label: str | None,
) -> pd.DataFrame:
    ratio_col = f"top{selection_topk}_min_time_ratio"
    candidate_col = f"top{selection_topk}_candidate"

    selected = df[df["scheme"] == scheme_name].copy()
    if split_label is not None:
        selected = selected[selected["source_split_label"].astype(str) == split_label].copy()
    if scope_label is not None:
        selected = selected[selected["source_scope"].astype(str) == scope_label].copy()
    if model_label is not None:
        selected = selected[selected["model"].astype(str) == model_label].copy()

    selected = selected[selected[ratio_col].apply(lambda value: (_safe_float(value) or -1) > threshold)].copy()
    selected["source_scheme"] = selected["scheme"]
    selected["source_rule"] = source_rule
    selected["selection_topk"] = selection_topk
    selected["selected_min_time_ratio"] = selected[ratio_col].apply(_round_or_none)
    selected["selected_candidate"] = selected[candidate_col]
    selected["threshold_rule"] = f"selected_min_time_ratio > {threshold}"
    selected["variant_kind"] = variant_kind
    selected["variant_label"] = variant_label if variant_label is not None else ""

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

    for column in ("top1_min_time_ratio", "top2_min_time_ratio", "top3_min_time_ratio"):
        if column in selected.columns:
            selected[column] = selected[column].apply(_round_or_none)
    return selected


def build_mix_sheet(
    combined_df: pd.DataFrame,
    left_scheme: str,
    left_topk: int,
    right_scheme: str,
    right_topk: int,
    threshold: float,
    variant_kind: str,
    variant_label: str | None = None,
    split_label: str | None = None,
    scope_label: str | None = None,
    model_label: str | None = None,
) -> pd.DataFrame:
    source_rule = _rule_label(left_scheme, left_topk, right_scheme, right_topk)
    left_rows = _select_rows(
        combined_df,
        scheme_name=left_scheme,
        selection_topk=left_topk,
        threshold=threshold,
        split_label=split_label,
        scope_label=scope_label,
        model_label=model_label,
        source_rule=source_rule,
        variant_kind=variant_kind,
        variant_label=variant_label,
    )
    right_rows = _select_rows(
        combined_df,
        scheme_name=right_scheme,
        selection_topk=right_topk,
        threshold=threshold,
        split_label=split_label,
        scope_label=scope_label,
        model_label=model_label,
        source_rule=source_rule,
        variant_kind=variant_kind,
        variant_label=variant_label,
    )
    mix_df = pd.concat([left_rows, right_rows], ignore_index=True, sort=False)
    mix_df = mix_df[MIX_OUTPUT_COLUMNS].sort_values(["dataset", "k_value", "scheme", "model"]).reset_index(drop=True)
    return mix_df


def write_workbook(output_path: Path, summary_df: pd.DataFrame, sheet_frames: dict[str, pd.DataFrame]) -> None:
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


def validate_output(summary_df: pd.DataFrame, sheet_frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    base_expected = {
        # "scheme1_top1__scheme2_top1",
        # "scheme1_top1__scheme2_top2",
        # "scheme1_top2__scheme2_top1",
        # "scheme1_top2__scheme2_top2",
        "scheme1_top1__scheme3_top1",
        "scheme1_top1__scheme3_top2",
        "scheme1_top2__scheme3_top1",
        "scheme1_top2__scheme3_top2",
    }
    missing_base = sorted(base_expected - set(sheet_frames))
    if missing_base:
        raise ValueError(f"Missing required base sheets: {missing_base}")

    if any({"scheme 2", "scheme 3"} == set(frame["source_scheme"].dropna().unique()) for frame in sheet_frames.values()):
        raise ValueError("Found an illegal scheme2+scheme3 mixed sheet.")

    unavailable_test_rows = int((summary_df.get("scope_label", pd.Series(dtype=object)) == "test_data").sum())
    if unavailable_test_rows != 0:
        raise ValueError("Test-data sheets should not have been generated from entire-data-only sources.")

    for sheet_name, frame in sheet_frames.items():
        if frame.empty:
            continue
        sorted_frame = frame.sort_values(["dataset", "k_value", "scheme", "model"]).reset_index(drop=True)
        if not frame.reset_index(drop=True).equals(sorted_frame):
            raise ValueError(f"{sheet_name} is not sorted by dataset and k_value.")
        if frame["source_scheme"].isna().any() or frame["source_rule"].isna().any():
            raise ValueError(f"{sheet_name} is missing provenance columns.")
        threshold = float(str(frame["threshold_rule"].iloc[0]).split(">")[1].strip())
        if frame["selected_min_time_ratio"].apply(lambda value: (_safe_float(value) or -1) <= threshold).any():
            raise ValueError(f"{sheet_name} contains rows at or below its threshold.")
    return {
        "generated_sheet_count": int(len(sheet_frames)),
        "sheet_names": list(sheet_frames.keys()),
        "summary_rows": int(len(summary_df)),
    }


def main() -> None:
    args = parse_args()
    master_path = resolve_existing_path(args.master_workbook, DEFAULT_MASTER_CANDIDATES, "master workbook")
    filtered_path = resolve_existing_path(args.filtered_workbook, DEFAULT_FILTERED_CANDIDATES, "filtered workbook")
    output_path = REPO_ROOT / args.output_path if not args.output_path.is_absolute() else args.output_path

    master_frames = load_workbook_frames(master_path)
    filtered_frames = load_workbook_frames(filtered_path)
    mapped_frames = attach_split_labels(master_frames, filtered_frames)
    combined_df = pd.concat(mapped_frames.values(), ignore_index=True, sort=False)

    sheet_frames: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, object]] = []
    available_scopes = sorted(set(combined_df["source_scope"].dropna().astype(str)))

    for left_scheme, right_scheme in SCHEME_PAIRS:
        pair_label = f"{left_scheme} + {right_scheme}"
        for left_topk, right_topk in TOPK_PAIRINGS:
            base_name = _sheet_name(left_scheme, left_topk, right_scheme, right_topk, "base", None)
            base_frame = build_mix_sheet(
                combined_df,
                left_scheme=left_scheme,
                left_topk=left_topk,
                right_scheme=right_scheme,
                right_topk=right_topk,
                threshold=args.default_threshold,
                variant_kind="base",
            )
            sheet_frames[base_name] = base_frame
            summary_rows.append(
                {
                    "sheet_name": base_name,
                    "scheme_pair": pair_label,
                    "topk_pairing": f"top{left_topk}+top{right_topk}",
                    "threshold": args.default_threshold,
                    "split_label": "",
                    "scope_label": DEFAULT_SCOPE,
                    "model_family": "",
                    "variant_kind": "base",
                    "row_count": int(len(base_frame)),
                    "scheme_1_rows": int((base_frame["source_scheme"] == "scheme 1").sum()),
                    "scheme_2_rows": int((base_frame["source_scheme"] == "scheme 2").sum()),
                    "scheme_3_rows": int((base_frame["source_scheme"] == "scheme 3").sum()),
                    "duplicates_retained": int(base_frame.duplicated(subset=MIX_OUTPUT_COLUMNS, keep=False).sum()),
                }
            )

            for threshold in THRESHOLDS:
                threshold_label = _threshold_suffix(threshold)
                threshold_name = _sheet_name(left_scheme, left_topk, right_scheme, right_topk, "threshold", f"thr{threshold_label}")
                threshold_frame = build_mix_sheet(
                    combined_df,
                    left_scheme=left_scheme,
                    left_topk=left_topk,
                    right_scheme=right_scheme,
                    right_topk=right_topk,
                    threshold=threshold,
                    variant_kind="threshold",
                    variant_label=f"thr{threshold_label}",
                )
                sheet_frames[threshold_name] = threshold_frame
                summary_rows.append(
                    {
                        "sheet_name": threshold_name,
                        "scheme_pair": pair_label,
                        "topk_pairing": f"top{left_topk}+top{right_topk}",
                        "threshold": threshold,
                        "split_label": "",
                        "scope_label": DEFAULT_SCOPE,
                        "model_family": "",
                        "variant_kind": "threshold",
                        "row_count": int(len(threshold_frame)),
                        "scheme_1_rows": int((threshold_frame["source_scheme"] == "scheme 1").sum()),
                        "scheme_2_rows": int((threshold_frame["source_scheme"] == "scheme 2").sum()),
                        "scheme_3_rows": int((threshold_frame["source_scheme"] == "scheme 3").sum()),
                        "duplicates_retained": int(threshold_frame.duplicated(subset=MIX_OUTPUT_COLUMNS, keep=False).sum()),
                    }
                )

            for split_label, split_suffix in (("strat_y", "straty"), ("strat_y_K", "stratyK")):
                split_name = _sheet_name(left_scheme, left_topk, right_scheme, right_topk, "split", split_suffix)
                split_frame = build_mix_sheet(
                    combined_df,
                    left_scheme=left_scheme,
                    left_topk=left_topk,
                    right_scheme=right_scheme,
                    right_topk=right_topk,
                    threshold=args.default_threshold,
                    variant_kind="split",
                    variant_label=split_suffix,
                    split_label=split_label,
                )
                sheet_frames[split_name] = split_frame
                summary_rows.append(
                    {
                        "sheet_name": split_name,
                        "scheme_pair": pair_label,
                        "topk_pairing": f"top{left_topk}+top{right_topk}",
                        "threshold": args.default_threshold,
                        "split_label": split_label,
                        "scope_label": "",
                        "model_family": "",
                        "variant_kind": "split",
                        "row_count": int(len(split_frame)),
                        "scheme_1_rows": int((split_frame["source_scheme"] == "scheme 1").sum()),
                        "scheme_2_rows": int((split_frame["source_scheme"] == "scheme 2").sum()),
                        "scheme_3_rows": int((split_frame["source_scheme"] == "scheme 3").sum()),
                        "duplicates_retained": int(split_frame.duplicated(subset=MIX_OUTPUT_COLUMNS, keep=False).sum()),
                    }
                )

            for scope in available_scopes:
                scope_suffix = "entire" if scope == "entire_data" else "test" if scope == "test_data" else scope.replace("_", "")
                scope_name = _sheet_name(left_scheme, left_topk, right_scheme, right_topk, "split", scope_suffix)
                scope_frame = build_mix_sheet(
                    combined_df,
                    left_scheme=left_scheme,
                    left_topk=left_topk,
                    right_scheme=right_scheme,
                    right_topk=right_topk,
                    threshold=args.default_threshold,
                    variant_kind="split",
                    variant_label=scope_suffix,
                    scope_label=scope,
                )
                sheet_frames[scope_name] = scope_frame
                summary_rows.append(
                    {
                        "sheet_name": scope_name,
                        "scheme_pair": pair_label,
                        "topk_pairing": f"top{left_topk}+top{right_topk}",
                        "threshold": args.default_threshold,
                        "split_label": "",
                        "scope_label": scope,
                        "model_family": "",
                        "variant_kind": "split",
                        "row_count": int(len(scope_frame)),
                        "scheme_1_rows": int((scope_frame["source_scheme"] == "scheme 1").sum()),
                        "scheme_2_rows": int((scope_frame["source_scheme"] == "scheme 2").sum()),
                        "scheme_3_rows": int((scope_frame["source_scheme"] == "scheme 3").sum()),
                        "duplicates_retained": int(scope_frame.duplicated(subset=MIX_OUTPUT_COLUMNS, keep=False).sum()),
                    }
                )

            for model_family in MODEL_VARIANTS:
                model_name = _sheet_name(left_scheme, left_topk, right_scheme, right_topk, "model", model_family)
                model_frame = build_mix_sheet(
                    combined_df,
                    left_scheme=left_scheme,
                    left_topk=left_topk,
                    right_scheme=right_scheme,
                    right_topk=right_topk,
                    threshold=args.default_threshold,
                    variant_kind="model",
                    variant_label=model_family,
                    model_label=model_family,
                )
                sheet_frames[model_name] = model_frame
                summary_rows.append(
                    {
                        "sheet_name": model_name,
                        "scheme_pair": pair_label,
                        "topk_pairing": f"top{left_topk}+top{right_topk}",
                        "threshold": args.default_threshold,
                        "split_label": "",
                        "scope_label": DEFAULT_SCOPE,
                        "model_family": model_family,
                        "variant_kind": "model",
                        "row_count": int(len(model_frame)),
                        "scheme_1_rows": int((model_frame["source_scheme"] == "scheme 1").sum()),
                        "scheme_2_rows": int((model_frame["source_scheme"] == "scheme 2").sum()),
                        "scheme_3_rows": int((model_frame["source_scheme"] == "scheme 3").sum()),
                        "duplicates_retained": int(model_frame.duplicated(subset=MIX_OUTPUT_COLUMNS, keep=False).sum()),
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["variant_kind", "scheme_pair", "topk_pairing", "threshold", "split_label", "scope_label", "model_family", "sheet_name"]
    ).reset_index(drop=True)
    write_workbook(output_path, summary_df, sheet_frames)
    validation = validate_output(summary_df, sheet_frames)

    print(f"Mixed workbook: {output_path}")
    print(f"Master workbook: {master_path}")
    print(f"Filtered workbook: {filtered_path}")
    print(f"Generated sheet count: {validation['generated_sheet_count']}")
    print(f"Generated sheets: {validation['sheet_names'][:12]}{' ...' if len(validation['sheet_names']) > 12 else ''}")
    print("Summary by variant kind:")
    print(summary_df.groupby("variant_kind", dropna=False)["sheet_name"].count().rename("sheet_count").to_string())


if __name__ == "__main__":
    main()
