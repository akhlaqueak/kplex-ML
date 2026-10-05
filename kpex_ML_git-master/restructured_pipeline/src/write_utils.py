"""Shared reporting and Excel utilities for the restructured pipeline."""

from __future__ import annotations

import os

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter


def highlight_and_collect_threshold(
    model_folder: str,
    filename_pattern: str = "wrong_predictions",
    threshold: float = 0.5,
    sheet_name: str = "Sheet1",
) -> pd.DataFrame:
    YELLOW = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    collected = []
    template_df = None

    for fname in os.listdir(model_folder):
        if not (fname.startswith(filename_pattern) and fname.endswith(".xlsx")):
            continue

        path = os.path.join(model_folder, fname)
        print(f"Processing {path}...")

        df = pd.read_excel(path, sheet_name=sheet_name)
        template_df = df.copy()

        df = ensure_time_gap_ratio(df)

        with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        mask = df["time_gap_ratio"] > threshold
        if mask.any():
            temp = df.loc[mask].copy()
            temp["filename"] = fname
            collected.append(temp)
            print(f"  -> Collected {mask.sum()} rows > {threshold}")
        else:
            print(f"  -> No rows exceed {threshold}")

        wb = load_workbook(path)
        ws = wb[sheet_name]

        headers = [c.value for c in ws[1]]
        col_idx = headers.index("time_gap_ratio") + 1
        last_col = len(headers)
        last_row = ws.max_row

        for row in range(2, last_row + 1):
            cell = ws.cell(row=row, column=col_idx).value
            try:
                if cell is not None and float(cell) > threshold:
                    for col in range(1, last_col + 1):
                        ws.cell(row=row, column=col).fill = YELLOW
            except Exception:
                continue

        wb.save(path)
        wb.close()
        print(f"  -> Highlighted and saved {fname}")

    if collected:
        collected_df = pd.concat(collected, ignore_index=True)
        collected_df = collected_df.sort_values(by="dataset")
        return collected_df

    if template_df is not None:
        empty = template_df.iloc[0:0].copy()
        empty["filename"] = pd.Series(dtype="object")
        return empty

    return pd.DataFrame(columns=["filename"])


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
    df = df.copy()
    fn = df[filename_col]

    df["inference_on"] = fn.str.contains(r"_(?:inference_)?entire_data(?:_|\.|$)", regex=True).map(
        {True: "on entire dataset", False: "on 20% test set"}
    )

    strat_match = fn.str.extract(r"strat_((?:y)(?:_K)?)(?:_exp_\d{2})?")
    df["strat_on"] = strat_match[0].map({"y": "y", "y_K": "y and K"}).fillna("")

    exp_match = fn.str.extract(r"_exp_(\d{2})")
    df["exp_num"] = exp_match[0].fillna("00")

    return df


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


def write_metrics_to_excel(cls_metrics: dict, time_metrics: dict, output_path: str):
    rows = []
    rows.append(["Overall Accuracy counting ties as correct (in %)", f"{cls_metrics['accuracy'] * 100:.2f}"])
    rows.append([])
    precisions = []
    rows.append(["Precision by Algorithm (in %):", ""])
    for alg, prec in cls_metrics["precision_per_alg"].items():
        rows.append([alg, f"{prec * 100:.2f}"])
        precisions.append(prec)
    mean_prec = sum(precisions) / len(precisions) if precisions else 0
    rows.append(["Mean Precision (in %):", f"{mean_prec * 100:.2f}"])
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
    recalls = []
    for alg, rec in cls_metrics["recall_per_alg"].items():
        rows.append([alg, f"{rec * 100:.2f}"])
        recalls.append(rec)

    mean_recall = sum(recalls) / len(recalls) if recalls else 0
    rows.append(["Mean Recall (in %):", f"{mean_recall * 100:.2f}"])
    rows.append([])

    rows.append(["Additional time metrics", ""])
    rows.append(["Total Predicted Time (in seconds)", f"{time_metrics['total_time_pred']:.2f}"])
    rows.append(["Total Optimal Time (in seconds)", f"{time_metrics['total_time_opt']:.2f}"])
    rows.append(["Time Ratio (pred/opt)", f"{time_metrics['time_ratio']:.3f}"])
    # rows.append(
    #     [
    #         "Avg Overhead/case ((pred - opt)/opt + ε) (in %)",
    #         f"{time_metrics['avg_overhead_per_case'] * 100:.2f}",
    #     ]
    # )

    df_out = pd.DataFrame(rows, columns=["Metric", "Value"])

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, header=False, sheet_name="Metrics")
        ws = writer.sheets["Metrics"]
        ws.column_dimensions["A"].width = 50
        ws.column_dimensions["B"].width = 20

    print(f"Metrics written to {output_path}")


def write_metrics_to_excel_file(cls_metrics: dict, time_metrics: dict, filepath: str, sheet_name: str = "Metrics"):

    invalid_chars = [":", "\\", "/", "?", "*", "[", "]"]

    for char in invalid_chars:
        sheet_name = sheet_name.replace(char, "-")

    sheet_name = sheet_name[:31]

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

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
    precisions = []
    for alg, prec in cls_metrics["precision_per_alg"].items():
        rows.append([alg, f"{prec * 100:.2f}"])
        precisions.append(prec)
    mean_prec = sum(precisions) / len(precisions) if precisions else 0
    rows.append(["Mean Precision (in %):", f"{mean_prec * 100:.2f}"])
    rows.append([])

    if "ap_per_alg" in cls_metrics:
        rows.append(["Avg Precision by Algorithm (in %):", ""])
        for alg, ap in cls_metrics["ap_per_alg"].items():
            if pd.isna(ap):
                rows.append([alg, "N/A"])
            else:
                rows.append([alg, f"{ap * 100:.2f}"])
        rows.append([])

    recalls = []
    rows.append(["Recall by Algorithm (in %):", ""])
    for alg, rec in cls_metrics["recall_per_alg"].items():
        rows.append([alg, f"{rec * 100:.2f}"])
        recalls.append(rec)

    mean_recall = sum(recalls) / len(recalls) if recalls else 0
    rows.append(["Mean Recall (in %):", f"{mean_recall * 100:.2f}"])
    
    rows.append([])
    rows.append(["Additional time metrics", ""])
    rows.append(["Total Predicted Time (in seconds)", f"{time_metrics['total_time_pred']:.2f}"])
    rows.append(["Total Optimal Time (in seconds)", f"{time_metrics['total_time_opt']:.2f}"])
    rows.append(["Time Ratio (pred/opt)", f"{time_metrics['time_ratio']:.3f}"])
    # rows.append(
    #     [
    #         "Avg Overhead/case ((pred - opt)/opt + ε) (in %)",
    #         f"{time_metrics['avg_overhead_per_case'] * 100:.2f}",
    #     ]
    # )

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
    "highlight_and_collect_threshold",
    "write_metrics_to_excel",
    "write_metrics_to_excel_file",
]
