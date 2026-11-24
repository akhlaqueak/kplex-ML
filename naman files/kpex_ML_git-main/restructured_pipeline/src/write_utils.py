"""Shared reporting and Excel utilities for the restructured pipeline."""

from __future__ import annotations

import os

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.utils import get_column_letter

from restructured_pipeline.src.compare_metadata_utils import extract_compare_filename_info


def consolidate_experiments(filepath: str, prefix: str, master_name: str, delete_source_sheets: bool = False):
    wb = load_workbook(filepath)
    sheet_names = [s for s in wb.sheetnames if s.startswith(prefix)]
    if not sheet_names:
        print(f"No sheets starting with '{prefix}' found.")
        return

    all_data = []
    max_rows = 0
    for name in sheet_names:
        ws = wb[name]
        col = []
        for row in ws.iter_rows(min_col=1, max_col=2, values_only=True):
            col.append([row[0] or "", row[1] or ""])
        all_data.append(col)
        max_rows = max(max_rows, len(col))

    if master_name in wb.sheetnames:
        wb.remove(wb[master_name])
    master = wb.create_sheet(master_name)

    header = ["Tolerances"] + [i + 1 for i in range(len(sheet_names))]
    master.append(header)

    for i in range(max_rows):
        row = []
        label = all_data[0][i][0] if i < len(all_data[0]) else ""
        row.append(label)
        for data in all_data:
            val = data[i][1] if i < len(data) else ""
            row.append(val)
        master.append(row)

    master.column_dimensions["A"].width = 55
    for col_idx in range(2, len(sheet_names) + 2):
        master.column_dimensions[get_column_letter(col_idx)].width = 20

    if delete_source_sheets:
        for name in sheet_names:
            if name in wb.sheetnames:
                wb.remove(wb[name])

    wb.save(filepath)
    print(f"Consolidated {len(sheet_names)} sheets into '{master_name}' in {filepath}")


def aggregate_master_dataframes(master_dict, output_dir=".", filename_column="filename", save_files=False):
    os.makedirs(output_dir, exist_ok=True)

    combined_rows = []
    for model_name, df in master_dict.items():
        temp = df.copy()
        temp["source_model"] = model_name
        combined_rows.append(temp)

    if not combined_rows:
        return pd.DataFrame()

    combined_df = pd.concat(combined_rows, ignore_index=True)

    filenames = combined_df[filename_column].unique()
    for fname in filenames:
        subset = combined_df[combined_df[filename_column] == fname].copy()

        cols = subset.columns.tolist()
        cols.remove("dataset")
        cols.remove("source_model")
        new_order = ["dataset", "source_model"] + cols

        subset = subset[new_order]
        subset = subset.sort_values(by="dataset")

        if save_files:
            outfile = os.path.join(output_dir, f"highlighted_errors_{fname}_agg_all_models.xlsx")
            subset.to_excel(outfile, index=False)
            print(f"Written aggregated file: {outfile}")

    return combined_df


def extract_filename_info(df: pd.DataFrame, filename_col: str = "filename") -> pd.DataFrame:
    return extract_compare_filename_info(df, filename_col=filename_col)


def ensure_time_gap_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time_gap_ratio" in df.columns:
        return df

    algorithm_cols = []
    for col in df.columns:
        if not col.startswith("prob_"):
            continue
        algo = col.replace("prob_", "", 1)
        if algo in df.columns:
            algorithm_cols.append(algo)

    if "fastest_time" not in df.columns and algorithm_cols:
        df["fastest_time"] = df[algorithm_cols].min(axis=1)

    if "predicted_time" not in df.columns and "predicted" in df.columns and algorithm_cols:
        df["predicted_time"] = df.apply(lambda r: r.get(r["predicted"], pd.NA), axis=1)

    if "time_gap" not in df.columns and {"predicted_time", "fastest_time"}.issubset(df.columns):
        df["time_gap"] = df["predicted_time"] - df["fastest_time"]

    if {"time_gap", "fastest_time"}.issubset(df.columns):
        df["time_gap_ratio"] = df.apply(
            lambda r: r["time_gap"] / r["fastest_time"] if r["fastest_time"] else float("inf"),
            axis=1,
        )
        print("  -> Computed 'time_gap_ratio'")
        return df

    raise KeyError("missing time_gap_ratio and cannot compute it")


def write_metrics_to_excel_file(cls_metrics: dict, time_metrics: dict, filepath: str, sheet_name: str = "Metrics"):
    if os.path.exists(filepath):
        wb = load_workbook(filepath)
        if sheet_name in wb.sheetnames:
            std = wb[sheet_name]
            wb.remove(std)
    else:
        wb = Workbook()
        default = wb.sheetnames[0]
        if default == "Sheet":
            wb.remove(wb[default])

    ws = wb.create_sheet(title=sheet_name)

    rows = []
    rows.append(["Overall Accuracy counting ties as correct (in %)", f"{cls_metrics['accuracy'] * 100:.2f}"])
    rows.append([])
    rows.append(["Precision by Algorithm (in %):", ""])
    for alg, prec in cls_metrics["precision_per_alg"].items():
        rows.append([alg, f"{prec * 100:.2f}"])
    rows.append([])

    if "ap_per_alg" in cls_metrics:
        rows.append(["Avg Precision by Algorithm (in %):", ""])
        for alg, ap in cls_metrics["ap_per_alg"].items():
            if pd.isna(ap):
                rows.append([alg, "N/A"])
            else:
                rows.append([alg, f"{ap * 100:.2f}"])
        rows.append([])

    rows.append(["Recall by Algorithm (in %):", ""])
    for alg, rec in cls_metrics["recall_per_alg"].items():
        rows.append([alg, f"{rec * 100:.2f}"])
    rows.append([])
    rows.append(["Additional time metrics", ""])
    rows.append(["Total Predicted Time (in seconds)", f"{time_metrics['total_time_pred']:.2f}"])
    rows.append(["Total Optimal Time (in seconds)", f"{time_metrics['total_time_opt']:.2f}"])
    rows.append(["Time Ratio (pred/opt)", f"{time_metrics['time_ratio']:.3f}"])
    rows.append(
        [
            "Avg Overhead/case ((pred - opt)/opt + ε) (in %)",
            f"{time_metrics['avg_overhead_per_case'] * 100:.2f}",
        ]
    )

    for row in rows:
        ws.append(row)

    ws.column_dimensions["A"].width = 55
    ws.column_dimensions["B"].width = 20

    wb.save(filepath)
    wb.close()
    print(f"Metrics written to {filepath}")


__all__ = [
    "aggregate_master_dataframes",
    "consolidate_experiments",
    "ensure_time_gap_ratio",
    "extract_filename_info",
    "write_metrics_to_excel_file",
]
