from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


DEFAULT_ROOT = Path(
    "restructured_pipeline/outputs/reports/scheme1/xgboost"
)

# Presentation names used only in the consolidated workbook.
WEIGHT_DISPLAY_NAMES = {
    "time_ratio": "Time Ratio",
    "time_diff_relative": "Relative Time Difference",
    "log_time_ratio": "Log(Time Ratio)",
    "time_min": "Minimum Runtime",
    "log_time_min": "Log(1 + Minimum Runtime)",
    "time_min_x_log_time_ratio": "Min Runtime × Log(Time Ratio)",
    "log_time_min_x_log_time_ratio": "Log(1 + Min Runtime) × Log(Time Ratio)",
    "log_time_min_plus_log_time_ratio": "Log(1 + Min Runtime) + Log(Time Ratio)",
    "2min_x_log_2min_ratio": "2nd Min × Log(2nd-Min Ratio)",
    "3min_x_log_3min_ratio": "3rd Min × Log(3rd-Min Ratio)",
    "4min_x_log_4min_ratio": "4th Min × Log(4th-Min Ratio)",
    "5min_x_log_5min_ratio": "5th Min × Log(5th-Min Ratio)",
    "6min_x_log_6min_ratio": "6th Min × Log(6th-Min Ratio)",
}

SPLIT_DISPLAY_NAMES = {
    "strat_y": "Target",
    "strat_y_K": "Target + K",
}

MASTER_COLUMNS = [
    "Weighting Scheme",
    "Stratification",
    "Test N",
    "Test Wrong",
    "Test Important",
    "Test Important / Wrong (%)",
    "Test Important / All (%)",
    "Entire N",
    "Entire Wrong",
    "Entire Important",
    "Entire Important / Wrong (%)",
    "Entire Important / All (%)",
    # "Test–Entire Overlap",
]

DETAIL_PREFERRED_COLUMNS = [
    "Weighting Scheme",
    "Weighting Scheme ID",
    "Stratification",
    "Model",
    "Evaluation Scope",
    "Total Predictions",
    "Wrong Predictions",
    "Important Failures",
    "Important / Wrong (%)",
    "Important / All (%)",
    # "Test–Entire Overlap",
    "Source Workbook",
]


def discover_failure_workbooks(root: Path, date_tag: str) -> list[Path]:
    """Find one dated failure-analysis workbook per weighting-scheme folder."""
    filename = f"failure_analysis_{date_tag}.xlsx"
    matches = sorted(root.rglob(filename))

    # Do not accidentally ingest a previously generated master workbook.
    matches = [
        p for p in matches
        if p.parent != root and not p.name.startswith("failure_analysis_master_summary_")
    ]

    if not matches:
        raise FileNotFoundError(
            f"No '{filename}' files found under:\n{root}"
        )

    return matches


def _first_existing(columns, candidates):
    return next((c for c in candidates if c in columns), None)


def _normalize_split(value):
    text = str(value).strip()
    lowered = text.lower()

    if lowered in {"strat_y", "target", "target only"}:
        return "strat_y"
    if lowered in {
        "strat_y_k",
        "strat_y+k",
        "target + k",
        "target+k",
        "target and k",
    }:
        return "strat_y_K"

    return text


def _normalize_scope(value):
    text = str(value).strip().lower()

    if text in {
        "test_data",
        "held-out test",
        "held out test",
        "test",
        "20% test",
    }:
        return "test_data"

    if text in {
        "entire_data",
        "entire dataset",
        "entire data",
        "all data",
    }:
        return "entire_data"

    return str(value).strip()


def read_summary(path: Path, root: Path):
    """Read the summary sheet and attach experiment identity."""
    excel = pd.ExcelFile(path)

    if not excel.sheet_names:
        raise ValueError(f"No worksheets found in {path}")

    # Prefer the explicit summary sheet; otherwise preserve compatibility
    # with older workbooks whose first worksheet was the summary.
    summary_sheet = (
        "summary"
        if "summary" in excel.sheet_names
        else excel.sheet_names[0]
    )

    summary = pd.read_excel(path, sheet_name=summary_sheet)

    relative_parent = path.parent.relative_to(root)
    weighting_scheme_id = relative_parent.parts[0]
    weighting_scheme = WEIGHT_DISPLAY_NAMES.get(
        weighting_scheme_id,
        weighting_scheme_id.replace("_", " ").title(),
    )

    summary.insert(0, "Weighting Scheme", weighting_scheme)
    summary.insert(1, "Weighting Scheme ID", weighting_scheme_id)
    summary["Source Workbook"] = str(path.relative_to(root))

    split_col = _first_existing(
        summary.columns,
        ["Stratification", "split_mode", "Split"],
    )
    scope_col = _first_existing(
        summary.columns,
        ["Evaluation Scope", "scope", "data_scope"],
    )

    if split_col is None:
        raise ValueError(
            f"{path}: summary has no stratification column. "
            f"Columns={list(summary.columns)}"
        )
    if scope_col is None:
        raise ValueError(
            f"{path}: summary has no evaluation-scope column. "
            f"Columns={list(summary.columns)}"
        )

    summary["_split_id"] = summary[split_col].map(_normalize_split)
    summary["_scope_id"] = summary[scope_col].map(_normalize_scope)

    # Standardize presentation columns without deleting original fields.
    summary["Stratification"] = summary["_split_id"].map(
        SPLIT_DISPLAY_NAMES
    ).fillna(summary["_split_id"])
    summary["Evaluation Scope"] = summary["_scope_id"].map(
        {
            "test_data": "Held-out Test",
            "entire_data": "Entire Data",
        }
    ).fillna(summary["_scope_id"])

    return weighting_scheme_id, weighting_scheme, summary


def _find_metric_column(df, candidates, required=True):
    col = _first_existing(df.columns, candidates)
    if col is None and required:
        raise ValueError(
            "Could not find required summary metric. "
            f"Tried {candidates}; available columns={list(df.columns)}"
        )
    return col


def _to_number(value):
    if pd.isna(value):
        return pd.NA
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def _pick_single_row(df, weight_id, split_id, scope_id):
    subset = df[
        df["Weighting Scheme ID"].eq(weight_id)
        & df["_split_id"].eq(split_id)
        & df["_scope_id"].eq(scope_id)
    ]

    if subset.empty:
        return None

    if len(subset) > 1:
        raise ValueError(
            "Expected one summary row but found duplicates for "
            f"weight={weight_id}, split={split_id}, scope={scope_id}."
        )

    return subset.iloc[0]


# def _extract_overlap(test_row, entire_row):
#     candidates = [
#         "Test–Entire Overlap",
#         "Test-Entire Overlap",
#         "overlapping_dataset_k_cases",
#         "Overlapping Cases",
#         "Overlap",
#     ]

#     for row in (test_row, entire_row):
#         if row is None:
#             continue
#         for col in candidates:
#             if col in row.index and pd.notna(row[col]):
#                 return _to_number(row[col])

#     return pd.NA


def build_research_master(detailed: pd.DataFrame) -> pd.DataFrame:
    """
    Create one publication-style row per weighting scheme × stratification.

    Test and entire-data statistics are placed side by side rather than
    repeated as separate rows.
    """
    total_col = _find_metric_column(
        detailed,
        ["Total Predictions", "total_predictions", "Rows", "rows"],
    )
    wrong_col = _find_metric_column(
        detailed,
        ["Wrong Predictions", "wrong_predictions"],
    )
    important_col = _find_metric_column(
        detailed,
        ["Important Failures", "important_failures"],
    )
    imp_wrong_col = _find_metric_column(
        detailed,
        [
            "Important / Wrong (%)",
            "important_given_wrong_percent",
            "Important/Wrong (%)",
        ],
        required=False,
    )
    imp_all_col = _find_metric_column(
        detailed,
        [
            "Important / All (%)",
            "important_of_all_percent",
            "Important/All (%)",
        ],
        required=False,
    )

    weight_order = list(dict.fromkeys(detailed["Weighting Scheme ID"]))
    rows = []

    for weight_id in weight_order:
        weight_name = detailed.loc[
            detailed["Weighting Scheme ID"].eq(weight_id),
            "Weighting Scheme",
        ].iloc[0]

        for split_id in ("strat_y", "strat_y_K"):
            test = _pick_single_row(
                detailed, weight_id, split_id, "test_data"
            )
            entire = _pick_single_row(
                detailed, weight_id, split_id, "entire_data"
            )

            # Skip split only if it is completely absent.
            if test is None and entire is None:
                continue

            def metric(row, col):
                if row is None or col is None:
                    return pd.NA
                return _to_number(row[col])

            def derived_percent(num, den):
                if pd.isna(num) or pd.isna(den) or float(den) == 0:
                    return pd.NA
                return round(100.0 * float(num) / float(den), 2)

            test_n = metric(test, total_col)
            test_wrong = metric(test, wrong_col)
            test_imp = metric(test, important_col)
            entire_n = metric(entire, total_col)
            entire_wrong = metric(entire, wrong_col)
            entire_imp = metric(entire, important_col)

            test_imp_wrong = (
                metric(test, imp_wrong_col)
                if imp_wrong_col
                else derived_percent(test_imp, test_wrong)
            )
            test_imp_all = (
                metric(test, imp_all_col)
                if imp_all_col
                else derived_percent(test_imp, test_n)
            )
            entire_imp_wrong = (
                metric(entire, imp_wrong_col)
                if imp_wrong_col
                else derived_percent(entire_imp, entire_wrong)
            )
            entire_imp_all = (
                metric(entire, imp_all_col)
                if imp_all_col
                else derived_percent(entire_imp, entire_n)
            )

            rows.append({
                "Weighting Scheme": weight_name,
                "Stratification": SPLIT_DISPLAY_NAMES.get(
                    split_id, split_id
                ),
                "Test N": test_n,
                "Test Wrong": test_wrong,
                "Test Important": test_imp,
                "Test Important / Wrong (%)": test_imp_wrong,
                "Test Important / All (%)": test_imp_all,
                "Entire N": entire_n,
                "Entire Wrong": entire_wrong,
                "Entire Important": entire_imp,
                "Entire Important / Wrong (%)": entire_imp_wrong,
                "Entire Important / All (%)": entire_imp_all,
                # "Test–Entire Overlap": _extract_overlap(test, entire),
                "_weight_id": weight_id,
                "_split_id": split_id,
            })

    master = pd.DataFrame(rows)

    split_order = {"strat_y": 0, "strat_y_K": 1}
    weight_rank = {name: i for i, name in enumerate(weight_order)}

    master["_weight_order"] = master["_weight_id"].map(weight_rank)
    master["_split_order"] = master["_split_id"].map(split_order).fillna(99)

    master = (
        master.sort_values(
            ["_weight_order", "_split_order"],
            kind="stable",
        )
        .drop(
            columns=[
                "_weight_id",
                "_split_id",
                "_weight_order",
                "_split_order",
            ]
        )
        .reset_index(drop=True)
    )

    return master[MASTER_COLUMNS]


def build_source_index(records):
    rows = []
    for weight_id, weight_name, path, row_count in records:
        rows.append({
            "Weighting Scheme": weight_name,
            "Weighting Scheme ID": weight_id,
            "Source Workbook": str(path),
            "Summary Rows Imported": row_count,
        })
    return pd.DataFrame(rows)


def build_detailed_summary(df: pd.DataFrame):
    """Keep the audit-level information but put important columns first."""
    result = df.copy()

    # Internal normalization helpers are not useful to readers.
    result = result.drop(
        columns=["_split_id", "_scope_id"],
        errors="ignore",
    )

    preferred = [
        c for c in DETAIL_PREFERRED_COLUMNS
        if c in result.columns
    ]
    remaining = [
        c for c in result.columns
        if c not in preferred
    ]

    return result[preferred + remaining]


def _style_standard_sheet(ws):
    # header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="000000", bold=True)

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 32

    for cell in ws[1]:
        # cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    for column_cells in ws.columns:
        letter = get_column_letter(column_cells[0].column)
        max_length = max(
            (
                len(str(cell.value))
                for cell in column_cells
                if cell.value is not None
            ),
            default=0,
        )
        ws.column_dimensions[letter].width = min(
            max(max_length + 2, 11),
            42,
        )


def _style_master_sheet(ws):
    """
    Add a two-level research-style header:
        Held-out Test | Entire Data | Cross-Scope
    """
    # Insert group-header row above the column names.
    ws.insert_rows(1)

    # Merge only true groups. A/B remain standalone identifiers.
    ws.merge_cells("A1:A2")
    ws.merge_cells("B1:B2")
    ws.merge_cells("C1:G1")
    ws.merge_cells("H1:L1")
    ws.merge_cells("M1:M2")

    ws["A1"] = "Weighting Scheme"
    ws["B1"] = "Stratification"
    ws["C1"] = "Held-out Test"
    ws["H1"] = "Entire Data"
    # ws["M1"] = "Cross-Scope"

    # Remove duplicate A/B/M labels from the second header row because those
    # cells are part of merged ranges.
    # ws["A2"] = None
    # ws["B2"] = None
    # ws["M2"] = None

    dark_fill = PatternFill("solid", fgColor="1F4E78")
    group_fill = PatternFill("solid", fgColor="D9EAF7")
    dark_font = Font(color="000000", bold=True)
    group_font = Font(color="000000", bold=True)

    for cell in ws[1]:
        # cell.fill = dark_fill
        cell.font = dark_font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    for cell in ws[2]:
        # cell.fill = group_fill
        cell.font = group_font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

    # A/B/M are merged through row 2, so keep the dark group style.
    for coord in ("A1", "B1", "M1"):
        # ws[coord].fill = dark_fill
        ws[coord].font = dark_font

    ws.freeze_panes = "C3"
    ws.auto_filter.ref = f"A2:L{ws.max_row}"
    ws.row_dimensions[1].height = 24
    ws.row_dimensions[2].height = 42

    widths = {
        "A": 38,
        "B": 18,
        "C": 12,
        "D": 13,
        "E": 15,
        "F": 23,
        "G": 21,
        "H": 12,
        "I": 13,
        "J": 15,
        "K": 25,
        "L": 23,
        # "M": 20,
    }
    for col, width in widths.items():
        ws.column_dimensions[col].width = width

    # Percentage fields are numeric percentages expressed as 0..100 in the
    # source summaries, so display two decimal places rather than Excel %.
    for row in range(3, ws.max_row + 1):
        for col in (6, 7, 11, 12):
            ws.cell(row, col).number_format = "0.00"

        for col in (3, 4, 5, 8, 9, 10):
            ws.cell(row, col).number_format = "0"


def format_workbook(path: Path):
    wb = load_workbook(path)

    for ws in wb.worksheets:
        if ws.title == "Master Summary":
            _style_master_sheet(ws)
        else:
            _style_standard_sheet(ws)

    wb.save(path)
    wb.close()


def build_master_workbook(
    root: Path,
    date_tag: str,
    output_path: Path,
):
    files = discover_failure_workbooks(root, date_tag)

    summaries = []
    source_records = []

    print(f"Found {len(files)} failure-analysis workbook(s).")

    for path in files:
        weight_id, weight_name, summary = read_summary(path, root)

        print(f"  {weight_id}: {path}")

        summaries.append(summary)
        source_records.append(
            (
                weight_id,
                weight_name,
                path.relative_to(root),
                len(summary),
            )
        )

    detailed_raw = pd.concat(
        summaries,
        ignore_index=True,
        sort=False,
    )

    master = build_research_master(detailed_raw)
    detailed = build_detailed_summary(detailed_raw)
    source_index = build_source_index(source_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        master.to_excel(
            writer,
            sheet_name="Master Summary",
            index=False,
        )
        detailed.to_excel(
            writer,
            sheet_name="Detailed Summary",
            index=False,
        )
        source_index.to_excel(
            writer,
            sheet_name="Source Index",
            index=False,
        )

    format_workbook(output_path)

    print()
    print("Master workbook written:")
    print(output_path)
    print()
    print("Weighting schemes:", source_index["Weighting Scheme ID"].nunique())
    print("Research-summary rows:", len(master))
    print("Detailed audit rows:", len(detailed))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Collect Scheme 1 XGB failure-analysis summaries and create "
            "a compact research-grade comparison workbook."
        )
    )

    parser.add_argument(
        "--date",
        required=True,
        help="Analysis date tag, e.g. 23sep2026",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Root containing Scheme 1 XGB weighting-scheme "
            "report folders."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output workbook.",
    )

    args = parser.parse_args()

    output = (
        args.output
        if args.output is not None
        else args.root
        / f"failure_analysis_master_summary_{args.date}.xlsx"
    )

    build_master_workbook(
        root=args.root,
        date_tag=args.date,
        output_path=output,
    )


if __name__ == "__main__":
    main()
