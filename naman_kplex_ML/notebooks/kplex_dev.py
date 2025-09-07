import pandas as pd
import numpy as np
import os
import re 
from sklearn.base import is_classifier
from sklearn.preprocessing import StandardScaler
from metrics import evaluate_algorithm_performance, evaluate_time_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


#User drfined parameter: timeout = 1800, also alter and analyse weighting scheme in 

# ============== Data Loading ===============

y_columns = ["UMKP", "AltRB", "kPEX", "Pivot_Br", "S_Br"]

def load_excel_data(excel_path, timeout_replacement): # !!! API for data loading !!!
    xls = pd.ExcelFile(excel_path)
    all_sheets = xls.sheet_names  # list of sheet names
    dfs = []
    
    # Iterate over each sheet
    for sheet in all_sheets:
        df = pd.read_excel(excel_path, sheet_name=sheet, header=1)
        df[y_columns] = df[y_columns].replace("×", timeout_replacement)
        # we will compute instance weights later by ourselves
        df.drop(columns=["Unnamed: 0", "Unnamed: 8", "max/min", "(max-min)/min", "min"], inplace=True)
        df.rename(columns={"Unnamed: 7": "kcore_list"}, inplace=True)
        df.drop(df.index[-1], inplace=True) # remove last row
        match = re.search(r"\d+", sheet)
        k_value = int(match.group())
        df["k_value"] = k_value # add new column with sheet name
        
        # Remove rows where ALL y_columns equal timeout_replacement
        mask = (df[y_columns] == timeout_replacement).all(axis=1)
        df = df[~mask]  # Keep rows where NOT all values are timeout_replacement
        
        dfs.append(df)

    # Concatenate all into one big DataFrame
    cat_df = pd.concat(dfs, ignore_index=True)
    dfs.reverse() # go from k = 2 to 20
    
    return cat_df, dfs # cat_df is the merged df, dfs is the array of df (df = DataFrame)

# ============== Instance Weighting ===============

# Basic Schemes
W_TIME_RATIO = 1
W_TIME_DIFF_RELATIVE = 2
W_LOG_TIME_RATIO = 4
W_TIME_MIN = 8
W_LOG_TIME_MIN = 16

# Advanced Schemes
W_TIME_MIN_X_W_LOG_TIME_RATIO = W_TIME_MIN + W_LOG_TIME_RATIO
W_LOG_TIME_MIN_X_W_LOG_TIME_RATIO = W_LOG_TIME_MIN + W_LOG_TIME_RATIO
W_LOG_TIME_MIN_PLUS_W_LOG_TIME_RATIO = (W_LOG_TIME_MIN + W_LOG_TIME_RATIO) * 2

# basic score functions

def time_ratio(vals, epsilon):
    return vals.max() / vals.min() if vals.min() != 0 else vals.max() / epsilon

def time_diff_relative(vals, epsilon): # = time_ratio - 1
    return (vals.max() - vals.min()) / vals.min() if vals.min() != 0 else vals.max() / epsilon

def time_min(vals):
    return vals.min()

def compute_weight(row, scheme, epsilon): # assisting function
    vals = row[y_columns]
    if scheme == W_TIME_RATIO:
        return time_ratio(vals, epsilon)
    elif scheme == W_TIME_DIFF_RELATIVE:
        return time_diff_relative(vals, epsilon)
    elif scheme == W_LOG_TIME_RATIO:
        return np.log(time_ratio(vals, epsilon))
    elif scheme == W_TIME_MIN:
        return time_min(vals)
    elif scheme == W_LOG_TIME_MIN:
        return np.log(1 + time_min(vals))
    elif scheme == W_TIME_MIN_X_W_LOG_TIME_RATIO:
        return time_min(vals) * np.log(time_ratio(vals, epsilon))
    elif scheme == W_LOG_TIME_MIN_X_W_LOG_TIME_RATIO:
        return np.log(1 + time_min(vals)) * np.log(time_ratio(vals, epsilon))
    elif scheme == W_LOG_TIME_MIN_PLUS_W_LOG_TIME_RATIO:
        return np.log(1 + time_min(vals)) + np.log(time_ratio(vals, epsilon))
    raise ValueError(f"Unknown scheme: {scheme}")

def assign_weights(df, scheme, epsilon): # !!! API for instance weight assignment !!!
    df["weight"] = df.apply(compute_weight, scheme=scheme, epsilon=epsilon, axis=1)

def export_sorted(df, output_file = "sorted_data.csv"): # !!! API for viewing sorted weight !!!
    df_sorted = df.sort_values(by="weight", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(output_file, index=False)

# ============== Feature Engineering (for k-core list) ===============

def process_kcore_list_to_statistics(df, keep_col=False):
    # Create temporary lists to store the new features
    shell_num = []
    core_min = []
    shell_avgSize = []
    shell_maxSize = []
    shell_minSize = []
    shell_stdSize = []
    
    for lst in df['kcore_list']:
        # String to list
        lst = eval(lst)
        
        # Extract sizes from each tuple in the list
        sizes = [size for (k, size) in lst]
        
        shell_num.append(len(lst))
        core_min.append(lst[0][0])  # First element's k value
        shell_avgSize.append(np.mean(sizes))
        shell_maxSize.append(np.max(sizes))
        shell_minSize.append(np.min(sizes))
        shell_stdSize.append(np.std(sizes))
    
    # Add the new columns to the dataframe
    df['shell_num'] = shell_num
    df['core_min'] = core_min
    df['shell_avgSize'] = shell_avgSize
    df['shell_maxSize'] = shell_maxSize
    df['shell_minSize'] = shell_minSize
    df['shell_stdSize'] = shell_stdSize
    
    # Drop the original column
    if not keep_col:
        df.drop('kcore_list', axis=1, inplace=True)


# ============== Post-Processing Data ===============

def post_process_data(cat_df):
    """
    Post-process the input DataFrame:
    - Convert selected float columns to nullable integer (`Int64`).
    - Reorder columns to bring metadata and features up front, followed by metrics.

    Parameters
    ----------
    cat_df : pd.DataFrame
        The input DataFrame to process.

    Returns
    -------
    pd.DataFrame
        A cleaned and reordered DataFrame.
    """
    # Columns to convert to Int64 (preserving missing values)
    int_cols = ['n', 'm', 'd_max', 'core_max', 'shell_maxSize', 'shell_minSize']
    
    for col in int_cols:
        if col in cat_df.columns:
            cat_df[col] = pd.to_numeric(cat_df[col], errors='coerce').astype('Int64')

    # Desired column order
    new_order = [
        'dataset', 'k_value', 'n', 'm', 'd_avg', 'd_max', 'core_max', 
        'core_min', 'shell_num', 
        'shell_avgSize', 'shell_maxSize', 'shell_minSize', 'shell_stdSize',
        'weight',
        'UMKP', 'AltRB', 'kPEX', 'Pivot_Br', 'S_Br'
    ]
    
    # Reorder if all columns exist; else keep existing order with a warning
    missing_cols = [col for col in new_order if col not in cat_df.columns]
    if missing_cols:
        print(f"Warning: Columns missing in DataFrame and will be skipped in reordering: {missing_cols}")
    
    ordered_cols = [col for col in new_order if col in cat_df.columns]
    remaining_cols = [col for col in cat_df.columns if col not in ordered_cols]
    
    # Final column order: specified ones followed by any remaining columns
    return cat_df[ordered_cols + remaining_cols]


# ============== Re-evaluate Model on Entire Dataset ===============

def get_fastest_algorithms(row):
    algorithms = ['UMKP', 'AltRB', 'kPEX', 'Pivot_Br', 'S_Br']
    times = [row[alg] for alg in algorithms]
    min_time = min(times)
    return [alg for alg in algorithms if row[alg] == min_time]


import pandas as pd

def re_evaluate_full(model, df, feature_cols, algorithms):
    """
    Re-evaluate a fitted model on the entire DataFrame.

    Parameters
    ----------
    model : object
        A fitted classifier with a .predict(X) method.
    df : pd.DataFrame
        Original DataFrame containing both features and runtime columns.
    feature_cols : list of str
        Column names to use as X for prediction.
    algorithms : list of str
        Column names holding each algorithm’s runtime.

    Returns
    -------
    results_df : pd.DataFrame
        A copy of df with added columns:
          - 'fastest_algorithms'
          - 'predicted'
          - 'correct'
    """
    # 1. Work on a clean copy
    results_df = df.copy()

    # 2. Compute the true fastest algorithm(s) per row
    def get_fastest_algorithms(row):
        times = [row[alg] for alg in algorithms]
        min_time = min(times)
        return [alg for alg in algorithms if row[alg] == min_time]

    results_df['fastest_algorithms'] = results_df.apply(
        get_fastest_algorithms, axis=1
    )

    # 3. Predict on the full feature set
    X_full = results_df[feature_cols]
    results_df['predicted'] = model.predict(X_full)

     # 4. Mark correctness (ties count as correct)
    results_df['correct'] = results_df.apply(
        lambda row: row['predicted'] in row['fastest_algorithms'],
        axis=1
    )

    return results_df

# ============== CSV of all wrong predictions sorted by weight ===============

def generate_wrong_predictions(
    results_df: pd.DataFrame,
    algorithms: list,
    epsilon: float = 1e-3
):
    """
    Extract all incorrect predictions, compute time metrics, optionally add predict_proba,
    save to CSV, and print a summary.

    Parameters
    ----------
    results_df : DataFrame
        Must contain columns 'predicted', 'correct' (bool), one col per alg in `algorithms`,
        plus 'weight' if available.
    algorithms : list of str
        Names of the runtime columns (e.g. ['UMKP','AltRB',…]).
    epsilon : float
        Small constant to avoid div-by-zero.
    """
    df = results_df.copy()

    # 1) Filter incorrect predictions
    wrong_predictions = df[~df["correct"]].copy()

    # 2) Compute time metrics
    def calculate_time_metrics(row):
       # Get all algorithm times
        algo_times = {alg: row[alg] for alg in algorithms}
        
        # Find fastest time and algorithms
        fastest_time = min(algo_times.values())
        fastest_algos = [alg for alg, t in algo_times.items() if t == fastest_time]
        
        # Get predicted algorithm's time
        predicted_time = algo_times[row['predicted']]
        time_gap = predicted_time - fastest_time
        
        return pd.Series({
            'fastest_time': fastest_time,
            'predicted_time': predicted_time,
            'time_gap': time_gap,
            'time_gap_ratio': time_gap / fastest_time if fastest_time > 0 else predicted_time / epsilon,
            'time_ratio': predicted_time / fastest_time if fastest_time > 0 else predicted_time / epsilon
        })

    time_metrics = wrong_predictions.apply(calculate_time_metrics, axis=1)
    wrong_predictions = pd.concat([wrong_predictions, time_metrics], axis=1)

    wrong_predictions_sorted = wrong_predictions.sort_values("weight", ascending=False)

    # Top‐5 by weight

    # Print summary
    print(f"\nSaved {len(wrong_predictions_sorted)} wrong predictions with time metrics")
    print("\nTop 5 heaviest wrong predictions:")
    print(wrong_predictions_sorted[['dataset', 'k_value', 'weight', 'predicted', 'fastest_algorithms', 'time_ratio']].head(5))

    # Enhanced error analysis
    print("\nError Analysis:")
    print(f"Average time ratio: {wrong_predictions_sorted['time_ratio'].mean():.2f}x slower")
    print(f"Worst time ratio: {wrong_predictions_sorted['time_ratio'].max():.2f}x slower")
    print(f"Best time ratio: {wrong_predictions_sorted['time_ratio'].min():.2f}x slower")
    print(f"Median time gap: {wrong_predictions_sorted['time_gap'].median():.2f} seconds")
    print(f"Largest time gap: {wrong_predictions_sorted['time_gap'].max():.2f} seconds")
    print(f"Smallest time gap: {wrong_predictions_sorted['time_gap'].min():.2f} seconds")

    # Error severity breakdown
    print("\nError Severity Distribution:")

    # Severity breakdown
    bins  = [0, 1.05, 1.2, 2, np.inf]
    labels= ['≤5%','5–20%','20–100%','>100%']
    severity = pd.cut(wrong_predictions_sorted['time_ratio'], bins=bins, labels=labels)
    print(severity.value_counts().sort_index())


    # Per‐algorithm error summary
    print("\nAlgorithm‐wise Errors:")
    for alg in algorithms:
        errs = wrong_predictions[wrong_predictions["predicted"] == alg]
        avg_r= errs["time_ratio"].mean() if len(errs)>0 else np.nan
        print(f"  {alg}: {len(errs)} errors, avg  {avg_r:.2f}× slower")

    return wrong_predictions_sorted

# ============== Plot side‐by‐side comparisons of classification reports ===============

import matplotlib.pyplot as plt

def compare_classification_reports(report_a, report_b, label_a, label_b):
    """
    Plot side‐by‐side comparisons of precision, recall, and F1‐score 
    between two classification reports.

    Parameters
    ----------
    report_a, report_b : dict
        The output of sklearn.metrics.classification_report(..., output_dict=True)
    label_a, label_b : str
        Names to label the two reports/curves in the legend.
    """
    # Extract class names (skip accuracy and averages)
    classes = [c for c in report_a.keys() 
               if c not in ('accuracy','macro avg','weighted avg')]

    metrics = ['precision', 'recall', 'f1-score']
    x = range(len(classes))
    
    # Width for side‐by‐side bars
    width = 0.35

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8,4))
        
        vals_a = [report_a[c][metric] for c in classes]
        vals_b = [report_b[c][metric] for c in classes]
        
        ax.bar([i - width/2 for i in x], vals_a, width, label=label_a)
        ax.bar([i + width/2 for i in x], vals_b, width, label=label_b)
        
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Comparison of {metric.capitalize()} by Class')
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()

# ============== Perform end-to-end evaluation and a flag save_files to toggle writing CSV/Excel outputs ===============

def run_full_evaluation(
    cat_df: pd.DataFrame,
    clf,
    features: list,
    algorithms: list,
    le = None,
    drop_indices: list = None,
    epsilon: float = 1e-3,
    save_files: bool = False,
    output_dir: str = './',
    file_suffix: str = "without_holdout_strat_y",
    kcore_list: bool = False,
    for_nn: bool = False
):
    """
    Perform end-to-end evaluation:
      1. Identify fastest algorithms
      2. Predict with given classifier `clf`
      3. Compute correctness and metrics
      4. Optionally save full predictions and wrong-predictions Excel/CSV
    
    Prints key steps and metrics for notebook presentation.
    
        Parameters
    ----------
    cat_df : DataFrame
      Original data.
    clf : estimator
      Trained classifier with predict/predict_proba.
    le : LabelEncoder
      Fitted encoder for target strings.
    drop_indices : list[int], optional
      Row indices to drop before time‐metric evaluation.
    features : list[str]
      Feature column names.
    algorithms : list[str]
      Algorithm runtime column names.
    save_files : bool
      Whether to write out CSV/XLSX.
    output_dir : str
      Directory path to save outputs.
    file_suffix : str
        The suffix to append to output filenames, e.g.
        "without_holdout_strat_y", "without_holdout_strat_y_K", or
        "with_holdout_strat_y", etc.

    Returns
    -------
    out_dict_1 : dict  # accuracy/precision/recall
    out_dict_2 : dict  # time metrics
    results_df : pd.DataFrame  # full predictions with probabilities & time metrics
    wrong_df   : pd.DataFrame  # wrong predictions with time metrics
    """

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    # 1) Copy and compute true fastest
    results_df = cat_df.copy()
    results_df['fastest_algorithms'] = results_df.apply(
        get_fastest_algorithms, axis=1
    )

    # 2) Predict
    if kcore_list:
        X_full = results_df[features+['kcore_list']]        
        # X_full = results_df[features+['kcore_list', 'fastest_time']]        
        y_pred = clf.predict(X_full)
    elif for_nn:
        X_full = results_df[features+['fastest_time']]        
        y_pred = clf.predict(X_full)
    else:
        if isinstance(clf, SVC):
            X_full = results_df[features]
            X_full = scaler.fit_transform(X_full)
            y_pred = clf.predict(X_full)
        # elif hasattr(clf, 'model_') and hasattr(clf, 'temperature'):
        #     X_full = results_df[features]
        #     X_full = scaler.fit_transform(X_full)
        #     y_pred = clf.predict(X_full)
        else:
            X_full = results_df[features]
            y_pred = clf.predict(X_full)


    print("Predicting on full dataset...")


    if le is not None:
        # Map back to strings
        results_df['predicted'] = le.inverse_transform(y_pred)
    else:
        # y_pred already contains the algorithm names
        results_df['predicted'] = y_pred

    # 3) Mark correctness
    results_df['correct'] = results_df.apply(
        lambda row: row['predicted'] in row['fastest_algorithms'], axis=1
    )

    # 4) Evaluate classification metrics
    print("\nClassification performance:")
    out_dict_1 = evaluate_algorithm_performance(results_df, algorithms)

    # 5) Evaluate time metrics (drop indexes with 0 values, if present)
    df_time = results_df
    if drop_indices:
        df_time = df_time.drop(index=drop_indices, errors='ignore')
        print(f"\nDropped indices {drop_indices} before time‐metric evaluation.")
    print("\nTime performance metrics:")
    out_dict_2 = evaluate_time_metrics(df_time, algorithms, epsilon=1e-3)

    # 6) Attach prediction probabilities if available
    proba_cols = []
    if hasattr(clf, "predict_proba"):
        print("\nAdding prediction probabilities...")
        probas = clf.predict_proba(X_full)
        cls_labels = le.classes_ if le is not None else clf.classes_
        for i, cls in enumerate(cls_labels):
            col = f"prob_{cls}"
            # only add the column if we can actually write it
            results_df[col] = probas[:, i]
            proba_cols.append(col)

        # if le:
        #     for i, cls in enumerate(le.classes_):
        #         results_df[f'prob_{cls}'] = probas[:, i]
        # else:
        #     for i, cls in enumerate(clf.classes_):
        #         results_df[f'prob_{cls}'] = probas[:, i]

    # 7) Save full predictions
    if save_files:
        os.makedirs(output_dir, exist_ok=True)
        fname_all = f"all_predictions_{file_suffix}.csv"
        csv_path = os.path.join(output_dir, fname_all)
        output_cols = ['dataset', 'k_value', 'fastest_algorithms', 'predicted', 'correct'] + algorithms
        output_cols += proba_cols
        
        # if hasattr(clf, "predict_proba") and le:
        #     output_cols += [f'prob_{cls}' for cls in le.classes_]
        # else:
        #     output_cols += [f'prob_{cls}' for cls in clf.classes_]

        print(f"\nSaving full predictions to '{csv_path}'")
        results_df[output_cols].to_csv(csv_path, index=False)

    # 8) Generate wrong-predictions DataFrame
    print("\nGenerating wrong-predictions with time metrics...")
    wrong_df = generate_wrong_predictions(
        results_df=results_df, algorithms=algorithms, epsilon=1e-3
    )

    # 9) Save wrong predictions
    if save_files:
        fname_wrong = f"wrong_predictions_{file_suffix}.xlsx"
        excel_path = os.path.join(output_dir, fname_wrong)
        output_cols_time = [
            'dataset', 'k_value', 'weight',
            'fastest_time', 'predicted_time', 'time_gap', 'time_gap_ratio','time_ratio',
            'fastest_algorithms', 'predicted', 'correct'
        ] + algorithms
        # only extend by the proba columns we actually created
        output_cols_time += proba_cols

        # if hasattr(clf, "predict_proba") and le:
        #     output_cols_time += [f'prob_{cls}' for cls in le.classes_]
        # else:
        #     output_cols_time += [f'prob_{cls}' for cls in clf.classes_]
        print(f"\nSaving wrong predictions to '{excel_path}'")
        wrong_df[output_cols_time].to_excel(excel_path, index=False)

    return out_dict_1, out_dict_2, results_df, wrong_df



import os
from openpyxl import Workbook, load_workbook

def write_metrics_to_excel_file(
    cls_metrics: dict,
    time_metrics: dict,
    filepath: str,
    sheet_name: str = "Metrics"
):
    """
    Append a new sheet of metrics to an Excel file (or create it if missing).
    
    Parameters
    ----------
    cls_metrics : dict
        Classification metrics.
    time_metrics : dict
        Time performance metrics.
    filepath : str
        Path to the .xlsx file. If it doesn't exist, it will be created.
    sheet_name : str
        Name of the sheet to write (will be created or replaced).
    """
    # 1) Open or create workbook
    if os.path.exists(filepath):
        wb = load_workbook(filepath)
        # If sheet already exists, remove it so we can recreate
        if sheet_name in wb.sheetnames:
            std = wb[sheet_name]
            wb.remove(std)
    else:
        wb = Workbook()
        # remove default sheet if it still has the default name
        default = wb.sheetnames[0]
        if default == "Sheet":
            wb.remove(wb[default])

    # 2) Create new sheet
    ws = wb.create_sheet(title=sheet_name)

    # 3) Build rows with formatting
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

    # 4) Write rows into sheet
    for r in rows:
        ws.append(r)

    # 5) Adjust column widths
    ws.column_dimensions['A'].width = 55
    ws.column_dimensions['B'].width = 20

    # 6) Save back to disk
    wb.save(filepath)
    wb.close()
    print(f"Metrics written to {filepath}")


# ---- Ensemble Evaluation Function  --- #

## TO-DO Review:

def run_union_evaluation(
    cat_df: pd.DataFrame,
    clf1,
    clf2,
    features: list,
    algorithms: list,
    le=None,
    drop_indices: list = None,
    epsilon: float = 1e-3,
    save_files: bool = False,
    output_dir: str = './',
    file_suffix: str = "union_eval",
    kcore_list: bool = False,
    for_nn: bool = False
):
    """
    1) Compute true fastest (ties)
    2) Predict with clf1 & clf2
    3) Compute union‐based correctness:
         {pred1, pred2} ∩ true_ties ≠ ∅ => correct
    4) Evaluate & save everything exactly as before,
       plus the two ‘predicted_by_<model>’ columns and
       an 'is_correct_union' flag.
    """

    scaler = StandardScaler()

    # 1) true fastest
    results_df = cat_df.copy()
    results_df['fastest_algorithms'] = results_df.apply(
        get_fastest_algorithms, axis=1
    )

    # 2) predictions
    def get_preds(clf, suffix):
        if kcore_list:
            X_full = results_df[features + ['kcore_list']]
        elif for_nn:
            X_full = results_df[features + ['fastest_time']]
        else:
            X_full = results_df[features]
            if isinstance(clf, SVC):
                X_full = scaler.fit_transform(X_full)

        raw = clf.predict(X_full)
        if le is not None:
            return le.inverse_transform(raw)
        return raw

    name1 = clf1.__class__.__name__
    name2 = clf2.__class__.__name__
    col1 = f'predicted_by_{name1}'
    col2 = f'predicted_by_{name2}'

    print("Predicting with", name1, "and", name2, "on full dataset...")
    results_df[col1] = get_preds(clf1, name1)
    results_df[col2] = get_preds(clf2, name2)

    # 3) union‐based correctness
    def union_correct(row):
        preds = {row[col1], row[col2]}
        return not preds.isdisjoint(row['fastest_algorithms'])

    results_df['is_correct_union'] = results_df.apply(union_correct, axis=1)

    # 4) Now we can evaluate exactly as before:
    #    a) classification metrics on 'is_correct_union'
    print("\nUnion‐based classification performance:")
    out_dict_1 = {
        'accuracy': results_df['is_correct_union'].mean()
        # you could also break down per‐alg as needed...
    }

    #    b) time metrics (on *all* rows or drop_indices)
    df_time = results_df.drop(index=drop_indices, errors='ignore') if drop_indices else results_df
    print("\nUnion‐based time performance metrics:")
    out_dict_2 = evaluate_time_metrics(df_time, algorithms, epsilon)

    # 5) Optionally attach probabilities for each clf
    proba_cols = []
    for clf, name, col_prefix in [(clf1, name1, col1), (clf2, name2, col2)]:
        if hasattr(clf, "predict_proba"):
            print(f"\nAdding predict_proba from {name}...")
            if kcore_list or for_nn:
                X_full = results_df[features + (['kcore_list'] if kcore_list else ['fastest_time'])]
            else:
                X_full = results_df[features]
                if isinstance(clf, SVC):
                    X_full = scaler.fit_transform(X_full)
            probs = clf.predict_proba(X_full)
            cls_labels = le.classes_ if le is not None else clf.classes_
            for i, cls in enumerate(cls_labels):
                pcol = f"prob_{name}_{cls}"
                results_df[pcol] = probs[:, i]
                proba_cols.append(pcol)

    # 6) Save full predictions
    if save_files:
        os.makedirs(output_dir, exist_ok=True)
        all_cols = ['dataset','k_value','fastest_algorithms',
                    col1,col2,'is_correct_union'] + algorithms + proba_cols
        fname_all = f"union_all_{file_suffix}.csv"
        print(f"\nSaving full union predictions to {fname_all}")
        results_df[all_cols].to_csv(os.path.join(output_dir, fname_all), index=False)

    # 7) Generate & save “wrong” union‐cases if desired
    wrong_union = results_df[~results_df['is_correct_union']].copy()
    if save_files:
        fname_wrong = f"union_wrong_{file_suffix}.xlsx"
        print(f"Saving wrong‐union cases to {fname_wrong}")
        wrong_union.to_excel(os.path.join(output_dir, fname_wrong), index=False)

    return out_dict_1, out_dict_2, results_df, wrong_union





