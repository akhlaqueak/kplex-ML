import pandas as pd
from openpyxl import load_workbook
import matplotlib.pyplot as plt

import os
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill


###--- For highlighting the prediction errors----###

def highlight_and_collect_threshold(
    model_folder: str,
    filename_pattern: str = "wrong_predictions",
    threshold: float = 0.5,
    sheet_name: str = "Sheet1"
) -> pd.DataFrame:
    """
    For each Excel file in `model_folder` matching filename_pattern:
     1) Load with pandas
     2) Ensure 'time_gap_ratio' exists (compute if needed)
     3) Overwrite the sheet with the new column
     4) Collect rows where time_gap_ratio > threshold
     5) Open with openpyxl and highlight entire rows
     6) Save and return all collected rows
    """
    YELLOW = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    collected = []

    for fname in os.listdir(model_folder):
        if not (fname.startswith(filename_pattern) and fname.endswith(".xlsx")):
            continue
        path = os.path.join(model_folder, fname)
        print(f"Processing {path}...")

        # 1) Read with pandas
        df = pd.read_excel(path, sheet_name=sheet_name)

        # 2) Compute time_gap_ratio if missing
        if 'time_gap_ratio' not in df.columns:
            if {'time_gap','fastest_time'}.issubset(df.columns):
                df['time_gap_ratio'] = df.apply(
                    lambda r: r['time_gap'] / r['fastest_time'] if r['fastest_time'] else float('inf'),
                    axis=1
                )
                print("  → Computed 'time_gap_ratio'")
            else:
                raise KeyError(f"{fname} missing time_gap_ratio and cannot compute it")

        # 3) Rewrite sheet with the new column
        with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # 4) Collect rows exceeding threshold
        mask = df['time_gap_ratio'] > threshold
        if mask.any():
            temp = df.loc[mask].copy()
            temp['filename'] = fname
            collected.append(temp)
            print(f"  → Collected {mask.sum()} rows > {threshold}")
        else:
            print(f"  → No rows exceed {threshold}")

        # 5) Highlight via openpyxl
        wb = load_workbook(path)
        ws = wb[sheet_name]

        # Find the updated header
        headers = [c.value for c in ws[1]]
        col_idx = headers.index('time_gap_ratio') + 1
        last_col = len(headers)
        last_row = ws.max_row

        for row in range(2, last_row+1):
            cell = ws.cell(row=row, column=col_idx).value
            try:
                if cell is not None and float(cell) > threshold:
                    for col in range(1, last_col+1):
                        ws.cell(row=row, column=col).fill = YELLOW
            except:
                continue

        wb.save(path)
        wb.close()
        print(f"  → Highlighted and saved {fname}")

    # 6) Build result DataFrame
    if collected:
        collected_df = pd.concat(collected, ignore_index=True)
        collected_df = collected_df.sort_values(by="dataset")
        return collected_df
    else:
        # No matches: return empty DF with expected columns
        empty = df.copy()
        empty['filename'] = []
        return empty.iloc[0:0]


### ---- Consolidate all hard and soft metrics of experiments into one master sheet. ----###

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

def consolidate_experiments(
    filepath: str,
    prefix: str,
    master_name: str
):
    """
    Consolidate all sheets whose names start with `prefix` into one `master_name` sheet.
    """
    wb = load_workbook(filepath)
    sheet_names = [s for s in wb.sheetnames if s.startswith(prefix)]
    if not sheet_names:
        print(f"No sheets starting with '{prefix}' found.")
        return

    # Read all data from each sheet
    all_data = []
    max_rows = 0
    for name in sheet_names:
        ws = wb[name]
        # read columns A & B into a list of tuples (label, value or '')
        col = []
        for row in ws.iter_rows(min_col=1, max_col=2, values_only=True):
            # values_only returns e.g. (None,None) for blank rows
            col.append([row[0] or "", row[1] or ""])
        all_data.append(col)
        max_rows = max(max_rows, len(col))

    # Create or replace master sheet
    if master_name in wb.sheetnames:
        wb.remove(wb[master_name])
    master = wb.create_sheet(master_name)

    # 1) Header row
    header = ["Experiment number"] + [i+1 for i in range(len(sheet_names))]
    master.append(header)

    # 2) Append each row i (0‑indexed) from all_data
    for i in range(max_rows):
        row = []
        # label from first sheet
        label = all_data[0][i][0] if i < len(all_data[0]) else ""
        row.append(label)
        # values from each sheet j
        for sheet_idx, data in enumerate(all_data):
            val = data[i][1] if i < len(data) else ""
            row.append(val)
        master.append(row)

    # 3) Adjust column widths
    master.column_dimensions['A'].width = 55
    for col_idx in range(2, len(sheet_names)+2):
        master.column_dimensions[get_column_letter(col_idx)].width = 20

    wb.save(filepath)
    print(f"✅ Consolidated {len(sheet_names)} sheets into '{master_name}' in {filepath}")


### --- Make one aggregated excel file for prediction errors from all models ---- ###

import pandas as pd
import os

def aggregate_master_dataframes(master_dict, output_dir=".", filename_column="filename", save_files = False):
    """
    master_dict : dict like {'NN': master_for_NN, 'RF': master_for_RF}
    output_dir  : directory where Excel files will be saved
    filename_column : the column to group on (e.g. 'filename')
    
    Creates Excel files named agg_<filename>_all_models.xlsx, sorted by dataset,
    with 'dataset' first column and 'source_model' second column.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1) Build combined dataframe with a new 'source_model' column
    combined_rows = []
    for model_name, df in master_dict.items():
        temp = df.copy()
        temp["source_model"] = model_name
        combined_rows.append(temp)
    combined_df = pd.concat(combined_rows, ignore_index=True)
    
    # 2) Now group by filename and create individual files
    filenames = combined_df[filename_column].unique()
    for fname in filenames:
        subset = combined_df[combined_df[filename_column] == fname].copy()
        
        # 3) Ensure columns order: dataset, source_model, then others
        cols = subset.columns.tolist()
        cols.remove("dataset")
        cols.remove("source_model")
        new_order = ["dataset", "source_model"] + cols
        
        subset = subset[new_order]
        
        # 4) Sort by dataset
        subset = subset.sort_values(by="dataset")
        
        # 5) Write to Excel
        if save_files:
            outfile = os.path.join(output_dir, f"highlighted_errors_{fname}_agg_all_models.xlsx")
            subset.to_excel(outfile, index=False)
            print(f"Written aggregated file: {outfile}")

    return combined_df


import pandas as pd
import re

def extract_filename_info(df: pd.DataFrame, filename_col: str = 'filename') -> pd.DataFrame:
    """
    From a DataFrame with a filename column like:
      - wrong_predictions_with_holdout_strat_y_exp_01.xlsx
      - wrong_predictions_with_holdout_strat_y_K_exp_03.xlsx
    this function adds three new columns:
      - with_holdout:  True if '_with_holdout_' in the filename
      - strat_on:      'y' or 'y and K'
      - exp_num:       string of the two‑digit experiment number, e.g. '01','03'
    """
    df = df.copy()
    fn = df[filename_col]

    # # 1) with_holdout: True/False
    # df['with_holdout'] = fn.str.contains(r'_with_holdout_', regex=True)

    # 1) inference_on: 'on test set' vs. 'on entire dataset'
    df['inference_on'] = fn.str.contains(r'_inference_entire_data_', regex=True) \
        .map({True: 'on entire dataset', False: 'on 20% test set'})

    # 2) strat_on: extract 'y' or 'y_K', then map to 'y'/'y and K'
    strat_match = fn.str.extract(r'strat_((?:y)(?:_K)?)_exp')
    df['strat_on'] = strat_match[0].map({
        'y': 'y',
        'y_K': 'y and K'
    }).fillna('')

    # 3) exp_num: extract two‑digit experiment number
    df['exp_num'] = fn.str.extract(r'_exp_(\d{2})').fillna('00')

    # df = df.rename(columns={'source_model': 'strategy'})

    return df

# Example usage:
# master_df = extract_filename_info(master_for_RF_with_holdout, 'filename')
# print(master_df[['filename','with_holdout','strat_on','exp_num']].head())



def write_metrics_to_excel(
    cls_metrics: dict,
    time_metrics: dict,
    output_path: str
):
    """
    Writes classification and time‑performance metrics into an Excel file
    in a human‑readable, sectioned format using openpyxl.
    """
    # Build rows with proper formatting
    rows = []
    rows.append([
        "Overall Accuracy counting ties as correct (in %)",
        f"{cls_metrics['accuracy'] * 100:.2f}"
    ])
    rows.append([])

    rows.append(["Precision by Algorithm (in %):", ""])
    for alg, prec in cls_metrics['precision_per_alg'].items():
        rows.append([alg, f"{prec * 100:.2f}"])
    rows.append([])

    rows.append(["Recall by Algorithm (in %):", ""])
    for alg, rec in cls_metrics['recall_per_alg'].items():
        rows.append([alg, f"{rec * 100:.2f}"])
    rows.append([])

    rows.append(["Additional time metrics", ""])
    rows.append(["Total Predicted Time (in seconds)", f"{time_metrics['total_time_pred']:.2f}"])
    rows.append(["Total Optimal Time (in seconds)",  f"{time_metrics['total_time_opt']:.2f}"])
    rows.append(["Time Ratio (pred/opt)",            f"{time_metrics['time_ratio']:.3f}"])
    rows.append([
        "Avg Overhead/case ((pred - opt)/opt + ε) (in %)",
        f"{time_metrics['avg_overhead_per_case'] * 100:.2f}"
    ])

    df_out = pd.DataFrame(rows, columns=["Metric", "Value"])

    # Write using openpyxl
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, header=False, sheet_name="Metrics")
        ws = writer.sheets["Metrics"]
        ws.column_dimensions['A'].width = 50
        ws.column_dimensions['B'].width = 20

    print(f"Metrics written to {output_path}")
