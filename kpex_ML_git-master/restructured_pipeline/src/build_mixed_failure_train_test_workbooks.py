"""Build split-specific mixed XGB failure workbooks with train/test origin.

Inputs:
  1) failure_train_test_split_count_summary_09sep2026.xlsx
  2) failure_analysis_summary_scheme1_scheme3_09sep2026.xlsx
  3) mixed_failure_case_combinations_xgb_strat_y_09sep2026.xlsx
  4) mixed_failure_case_combinations_xgb_strat_y_K_09sep2026.xlsx

Outputs:
  - mixed_failure_train_test_shared_strat_y_09sep2026.xlsx
  - mixed_failure_train_test_shared_strat_y_K_09sep2026.xlsx

For each Top-k combination:
  shared failure pair = same (dataset, k_value) occurs for both Scheme 1 and Scheme 3.
  shared_rows = 2 * shared_failure_pairs.
  remaining_rows = total_rows - shared_rows.

Each row is labelled test/train by matching (dataset, k_value) against the
corresponding XGB experiment's test_data important-failure rows in the enriched
failure-analysis workbook. The aggregate count workbook is used as a validation
source for the underlying Scheme 1/3 XGB train/test important-failure totals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

TOPK_SHEETS_PREFIX = "scheme1_"
DROP_OUTPUT_COLUMNS = {"model", "split_mode", "data_scope"}
SUMMARY_SHEETS = {"summary", "shared_failure_summary"}
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
HEADER_FONT = Font(color="FFFFFF", bold=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--count-workbook", type=Path, required=True)
    p.add_argument("--failure-analysis-workbook", type=Path, required=True)
    p.add_argument("--mixed-strat-y-workbook", type=Path, required=True)
    p.add_argument("--mixed-strat-y-k-workbook", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("restructured_pipeline/outputs/reports"))
    p.add_argument("--tag", default="09sep2026")
    return p.parse_args()


def normalize_identity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).str.strip()
    out["k_value"] = pd.to_numeric(out["k_value"], errors="raise").astype(int)
    return out


def build_test_membership(failure_analysis_path: Path) -> dict[tuple[str, str], set[tuple[str, int]]]:
    mapping = {}
    specs = [
        ("strat_y", "scheme 1", "scheme_1_XGB_strat_y"),
        ("strat_y", "scheme 3", "scheme_3_XGB_strat_y"),
        ("strat_y_K", "scheme 1", "scheme_1_XGB_strat_y_K"),
        ("strat_y_K", "scheme 3", "scheme_3_XGB_strat_y_K"),
    ]
    for split_mode, scheme, sheet in specs:
        df = normalize_identity(pd.read_excel(failure_analysis_path, sheet_name=sheet))
        test = df[df["data_scope"].eq("test_data")]
        mapping[(split_mode, scheme)] = set(zip(test["dataset"], test["k_value"]))
    return mapping


def load_count_validation(count_path: Path, split_mode: str) -> pd.DataFrame:
    df = pd.read_excel(count_path, sheet_name="important_failure_counts")
    return df[(df["Model"].eq("XGB")) & (df["Split"].eq(split_mode)) & (df["Scheme"].isin(["scheme 1", "scheme 3"]))].copy()


def classify_rows(df: pd.DataFrame, split_mode: str, membership) -> pd.DataFrame:
    out = normalize_identity(df)
    out["origin"] = [
        "test" if (dataset, int(k)) in membership[(split_mode, scheme)] else "train"
        for dataset, k, scheme in zip(out["dataset"], out["k_value"], out["scheme"])
    ]
    return out


def analyze_topk_sheet(df: pd.DataFrame, split_mode: str, membership):
    df = classify_rows(df, split_mode, membership)
    groups = {key: g.copy() for key, g in df.groupby(["dataset", "k_value"], sort=False)}
    shared = {
        key: g for key, g in groups.items()
        if set(g["scheme"].astype(str)) == {"scheme 1", "scheme 3"}
    }
    shared_keys = set(shared)
    shared_df = pd.concat(shared.values(), ignore_index=True) if shared else df.iloc[0:0].copy()
    remaining_df = df[~df.set_index(["dataset", "k_value"]).index.isin(shared_keys)].copy()

    pair_counts = {
        "both_test_pairs": 0,
        "both_train_pairs": 0,
        "scheme1_test_scheme3_train_pairs": 0,
        "scheme1_train_scheme3_test_pairs": 0,
    }
    for g in shared.values():
        origins = dict(zip(g["scheme"], g["origin"]))
        s1, s3 = origins["scheme 1"], origins["scheme 3"]
        if s1 == "test" and s3 == "test":
            pair_counts["both_test_pairs"] += 1
        elif s1 == "train" and s3 == "train":
            pair_counts["both_train_pairs"] += 1
        elif s1 == "test":
            pair_counts["scheme1_test_scheme3_train_pairs"] += 1
        else:
            pair_counts["scheme1_train_scheme3_test_pairs"] += 1

    shared_pairs = len(shared)
    shared_rows = len(shared_df)
    if shared_rows != 2 * shared_pairs:
        raise ValueError(f"Expected exactly two rows per shared pair, got pairs={shared_pairs}, rows={shared_rows}.")

    summary = {
        "split_mode": split_mode,
        "total_rows": len(df),
        "shared_failure_pairs": shared_pairs,
        "shared_rows": shared_rows,
        "shared_train_rows": int(shared_df["origin"].eq("train").sum()),
        "shared_test_rows": int(shared_df["origin"].eq("test").sum()),
        "remaining_rows": len(remaining_df),
        "remaining_train_rows": int(remaining_df["origin"].eq("train").sum()),
        "remaining_test_rows": int(remaining_df["origin"].eq("test").sum()),
        **pair_counts,
    }
    if summary["remaining_rows"] != summary["total_rows"] - 2 * summary["shared_failure_pairs"]:
        raise ValueError("remaining_rows validation failed.")
    if summary["shared_train_rows"] + summary["shared_test_rows"] != summary["shared_rows"]:
        raise ValueError("shared train/test row validation failed.")
    if summary["remaining_train_rows"] + summary["remaining_test_rows"] != summary["remaining_rows"]:
        raise ValueError("remaining train/test row validation failed.")

    # User-requested detail: shared/highlighted rows only, sorted dataset -> K -> scheme.
    shared_df = shared_df.drop(columns=[c for c in DROP_OUTPUT_COLUMNS if c in shared_df.columns])
    first = [c for c in ["dataset", "k_value", "scheme", "origin"] if c in shared_df.columns]
    rest = [c for c in shared_df.columns if c not in first]
    shared_df = shared_df[first + rest].sort_values(["dataset", "k_value", "scheme"], kind="stable").reset_index(drop=True)
    return summary, shared_df


def style_workbook(path: Path) -> None:
    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
        for col in ws.columns:
            letter = col[0].column_letter
            max_len = max((len(str(c.value)) for c in col if c.value is not None), default=0)
            ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 40)
    wb.save(path)
    wb.close()


def build_one(mixed_path: Path, split_mode: str, output_path: Path, membership, count_path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(mixed_path)
    topk_sheets = [s for s in xl.sheet_names if s.startswith(TOPK_SHEETS_PREFIX)]
    summaries = []
    details = {}
    for sheet in topk_sheets:
        df = pd.read_excel(mixed_path, sheet_name=sheet)
        summary, shared_df = analyze_topk_sheet(df, split_mode, membership)
        summary = {"topk_combination": sheet, **summary}
        summaries.append(summary)
        details[sheet] = shared_df

    summary_df = pd.DataFrame(summaries).sort_values("topk_combination", kind="stable").reset_index(drop=True)
    source_counts = load_count_validation(count_path, split_mode)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        source_counts.to_excel(writer, sheet_name="source_failure_counts", index=False)
        for sheet, df in details.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    style_workbook(output_path)
    return summary_df


def main() -> None:
    args = parse_args()
    membership = build_test_membership(args.failure_analysis_workbook)

    outputs = [
        ("strat_y", args.mixed_strat_y_workbook, args.output_dir / f"mixed_failure_train_test_shared_strat_y_{args.tag}.xlsx"),
        ("strat_y_K", args.mixed_strat_y_k_workbook, args.output_dir / f"mixed_failure_train_test_shared_strat_y_K_{args.tag}.xlsx"),
    ]
    for split_mode, source, output in outputs:
        summary = build_one(source, split_mode, output, membership, args.count_workbook)
        print(f"\n{split_mode}: {output}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
