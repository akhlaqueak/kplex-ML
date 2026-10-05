"""Build mixed Top-K failure workbooks for Scheme 4 or Scheme 5.

The filtering rule intentionally matches the older mixed Scheme-1/Scheme-3
analysis: for a pairing such as B2+O3, filter the B-family important failures
by top2_min_time_ratio > 1.20 and the Other-family important failures by
top3_min_time_ratio > 1.20, independently, then concatenate the surviving
rows. Therefore the number of rows normally changes with the Top-K pairing.

If the same dataset/K is a surviving failure in both families, both rows are
kept (one row per family). Shared surviving dataset/K rows are highlighted
yellow. No synthetic 'Both' row is created.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

THRESHOLD = 1.20
MAX_TOPK = 6
YELLOW_FILL = PatternFill(fill_type="solid", start_color="FFF59D", end_color="FFF59D")

# Columns deliberately omitted from the exported combination sheets:
#   global_fastest_algo
#   global_fastest_time
#   Predicted_fastest_algo
#   fastest_time
#   RankOfFastest_algorithm_among_24
OUTPUT_COLUMNS = [
    "dataset",
    "k_value",
    "failure_belongs_to",
    "fastest_algo_tuple",
    "top1_min_time_ratio",
    "top1_candidate",
    "top2_min_time_ratio",
    "top2_candidate",
    "top3_min_time_ratio",
    "top3_candidate",
    "top4_min_time_ratio",
    "top4_candidate",
    "top5_min_time_ratio",
    "top5_candidate",
    "top6_min_time_ratio",
    "top6_candidate",
    "all_predictions_sorted_by_prob",
    "source_scheme",
    "source_rule",
]


def _normalize_failure_frame(path: Path, split: str, family_label: str) -> pd.DataFrame:
    """Load the family-specific unique important-failure sheet without losing Top-K columns."""
    sheet = f"{split}_unique"
    df = pd.read_excel(path, sheet_name=sheet).copy()

    required = {
        "dataset",
        "k_value",
        "fastest_algo_tuple",
        *[f"top{k}_min_time_ratio" for k in range(1, MAX_TOPK + 1)],
        *[f"top{k}_candidate" for k in range(1, MAX_TOPK + 1)],
        "all_predictions_sorted_by_prob",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path}/{sheet} missing required columns: {missing}")

    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["k_value"] = pd.to_numeric(df["k_value"], errors="raise").astype(int)

    # The *_unique sheet should already contain one row per dataset/K. Validate
    # rather than silently changing the population.
    duplicates = df.duplicated(["dataset", "k_value"], keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, ["dataset", "k_value"]].head(20).to_dict("records")
        raise ValueError(
            f"{path}/{sheet} contains duplicate dataset/K rows; expected a unique view. "
            f"Examples: {sample}"
        )

    df["failure_belongs_to"] = family_label

    # Preserve the old highlighting meaning internally, without exporting the
    # global/family-fastest columns the user asked to remove.
    if {"fastest_algo", "global_fastest_algo"}.issubset(df.columns):
        df["_family_global_mismatch"] = (
            df["fastest_algo"].astype(str) != df["global_fastest_algo"].astype(str)
        )
    elif "family_fastest_is_global" in df.columns:
        df["_family_global_mismatch"] = ~df["family_fastest_is_global"].fillna(False).astype(bool)
    else:
        df["_family_global_mismatch"] = False

    return df.reset_index(drop=True)


def _filter_for_topk(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Old rule: keep only failures whose selected family Top-K ratio is > 1.20."""
    ratio_col = f"top{k}_min_time_ratio"
    ratios = pd.to_numeric(df[ratio_col], errors="coerce")
    return df.loc[ratios.gt(THRESHOLD)].copy()


def _shared_keys(left: pd.DataFrame, right: pd.DataFrame) -> set[tuple[str, int]]:
    left_keys = set(zip(left["dataset"].astype(str), left["k_value"].astype(int)))
    right_keys = set(zip(right["dataset"].astype(str), right["k_value"].astype(int)))
    return left_keys & right_keys


def build(
    left_workbook: Path,
    right_workbook: Path,
    split: str,
    scheme: str,
):
    """Build all 3x3 Top-K pairings using the legacy independent-side filter."""
    if scheme == "scheme4":
        left_label, right_label = "Oriented", "Forward"
        letters = ("O", "F")
        source_scheme = "scheme 4"
    else:
        left_label, right_label = "B", "Other"
        letters = ("B", "O")
        source_scheme = "scheme 5"

    left = _normalize_failure_frame(left_workbook, split, left_label)
    right = _normalize_failure_frame(right_workbook, split, right_label)

    left_before = len(left)
    right_before = len(right)
    baseline_rows = left_before + right_before

    frames: dict[str, pd.DataFrame] = {}
    overlap_by_sheet: dict[str, set[tuple[str, int]]] = {}
    summary_rows = []

    for left_k in range(1, 4):
        for right_k in range(1, 4):
            left_filtered = _filter_for_topk(left, left_k)
            right_filtered = _filter_for_topk(right, right_k)

            rule = f"{letters[0]}{left_k}+{letters[1]}{right_k}"
            sheet_name = f"{letters[0]}{left_k}_{letters[1]}{right_k}"

            left_filtered["source_scheme"] = source_scheme
            right_filtered["source_scheme"] = source_scheme
            left_filtered["source_rule"] = rule
            right_filtered["source_rule"] = rule

            shared = _shared_keys(left_filtered, right_filtered)
            overlap_by_sheet[sheet_name] = shared

            combined = pd.concat(
                [left_filtered, right_filtered],
                ignore_index=True,
                sort=False,
            )
            combined = combined.sort_values(
                ["dataset", "k_value", "failure_belongs_to"],
                kind="stable",
            ).reset_index(drop=True)

            # Keep internal highlight flag until workbook formatting, but export
            # only the requested user-facing columns.
            highlight_flags = combined.get(
                "_family_global_mismatch",
                pd.Series(False, index=combined.index),
            ).fillna(False).astype(bool).tolist()

            export = combined[OUTPUT_COLUMNS].copy()
            export.attrs["family_global_mismatch_flags"] = highlight_flags
            frames[sheet_name] = export

            remaining = len(export)
            recovered = baseline_rows - remaining

            summary_rows.append(
                {
                    "combination": rule,
                    # "baseline_important_failure_rows": int(baseline_rows),
                    "important_failure_rows": int(remaining),
                    "b_failure_rows" if scheme == "scheme5" else "oriented_failure_rows": int(len(left_filtered)),
                    "other_failure_rows" if scheme == "scheme5" else "forward_failure_rows": int(len(right_filtered)),
                    "shared_dataset_k_cases": int(len(shared)),
                    # "recovered_le_1_2x": int(recovered),
                    # "remaining_failures_gt_1_2x": int(remaining),
                    # "recovery_percent": round(100.0 * recovered / baseline_rows, 2) if baseline_rows else 0.0,
                    "filter_rule": (
                        f"{left_label} top{left_k}_min_time_ratio > {THRESHOLD:.2f}; "
                        f"{right_label} top{right_k}_min_time_ratio > {THRESHOLD:.2f}; "
                        "filter each family independently, then concatenate"
                    ),
                }
            )

    return pd.DataFrame(summary_rows), frames, overlap_by_sheet


def _highlight_workbook(
    output_path: Path,
    frames: dict[str, pd.DataFrame],
    overlap_by_sheet: dict[str, set[tuple[str, int]]],
) -> None:
    """Apply only the established yellow highlights.

    A row is yellow when either:
      1. its family optimum differs from the global optimum (legacy family
         failure-analysis highlight), or
      2. the same surviving dataset/K occurs on both sides of this Top-K pair.

    The columns used to determine (1) are intentionally not exported.
    """
    wb = load_workbook(output_path)
    try:
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            if ws.title == "summary":
                continue

            headers = [c.value for c in ws[1]]
            dataset_col = headers.index("dataset") + 1
            k_col = headers.index("k_value") + 1
            flags = frames[ws.title].attrs.get("family_global_mismatch_flags", [])
            shared = overlap_by_sheet.get(ws.title, set())

            for excel_row in range(2, ws.max_row + 1):
                data_idx = excel_row - 2
                dataset = str(ws.cell(excel_row, dataset_col).value)
                try:
                    kval = int(ws.cell(excel_row, k_col).value)
                except (TypeError, ValueError):
                    kval = None

                mismatch = bool(flags[data_idx]) if data_idx < len(flags) else False
                is_shared = kval is not None and (dataset, kval) in shared
                if mismatch or is_shared:
                    for col in range(1, ws.max_column + 1):
                        ws.cell(excel_row, col).fill = YELLOW_FILL

        wb.save(output_path)
    finally:
        wb.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=["scheme4", "scheme5"], default="scheme4")
    parser.add_argument("--split-mode", choices=["strat_y", "strat_y_K"], required=True)
    parser.add_argument("--output-path", type=Path, required=True)

    # Keep the existing CLI names so current bash files do not break.
    parser.add_argument("--oriented-workbook", type=Path)
    parser.add_argument("--forward-workbook", type=Path)
    parser.add_argument("--oriented-predictions", type=Path)
    parser.add_argument("--forward-predictions", type=Path)
    parser.add_argument("--b-workbook", type=Path)
    parser.add_argument("--other-workbook", type=Path)
    parser.add_argument("--b-predictions", type=Path)
    parser.add_argument("--other-predictions", type=Path)
    args = parser.parse_args()

    if args.scheme == "scheme4":
        left, right = args.oriented_workbook, args.forward_workbook
    else:
        left, right = args.b_workbook, args.other_workbook

    if left is None or right is None:
        parser.error("The two family failure-analysis workbooks are required for the selected scheme.")

    # Prediction arguments remain accepted for backward-compatible bash files,
    # but the old mixed-analysis rule operates on the enriched family Top-K
    # columns, so the raw prediction CSVs are not needed here.
    summary, frames, overlap_by_sheet = build(
        left_workbook=left,
        right_workbook=right,
        split=args.split_mode,
        scheme=args.scheme,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)

    # _highlight_workbook(args.output_path, frames, overlap_by_sheet)

    print(summary.to_string(index=False))
    print(f"Wrote: {args.output_path}")


if __name__ == "__main__":
    main()
