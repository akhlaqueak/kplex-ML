import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def evaluate_algorithm_performance(results_df, algorithms):
    """
    Given a DataFrame of prediction results and a list of algorithm names,
    prints overall accuracy (with ties counted as correct), precision per algorithm,
    and recall per algorithm (where recall counts ties as positive cases).
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        Must contain columns:
         - 'correct' : bool, whether the predicted algorithm was in the true fastest set
         - 'predicted' : str, the algorithm you predicted
         - 'fastest_algorithms' : list-like, the set of true fastest algorithms (can be ties)
    algorithms : list of str
        The list of all algorithm names to evaluate (e.g. ['UMKP','AltRB',...]).
    """
    # 1) Overall accuracy
    out = {}
    accuracy = results_df['correct'].mean()
    out['accuracy'] = accuracy
    print(f"Accuracy (counting ties as correct): {accuracy:.2%}")
    
    # 2) Precision per algorithm
    out['precision_per_alg'] = dict()
    print("\nPrecision by Algorithm:")
    for alg in algorithms:
        mask = results_df['predicted'] == alg
        precision = results_df.loc[mask, 'correct'].mean() if mask.any() else 0.0
        count = mask.sum()
        out['precision_per_alg'][alg] = precision
        print(f"  {alg}: {precision:.2%} ({count} predictions)")
    
    
    # 3) Recall per algorithm
    out['recall_per_alg'] = dict()
    print("\nRecall by Algorithm:")
    for alg in algorithms:
        # Rows where this alg truly was among the fastest
        was_fastest = results_df['fastest_algorithms'].apply(lambda fastest: alg in fastest)
        total_fastest = was_fastest.sum()
        
        if total_fastest > 0:
            true_positives = (results_df['correct'] & was_fastest).sum()
            recall = true_positives / total_fastest
            out['recall_per_alg'][alg] = recall
        else:
            recall = 0.0
            true_positives = 0
            out['recall_per_alg'][alg] = 0
        
        print(f"  {alg}: {recall:.2%} ({true_positives}/{total_fastest} cases)")
    
    return out


def evaluate_time_metrics(results_df, algorithms, epsilon = 1e-3):
    """
    - total_time_pred: sum over chosen algo's runtime
    - total_time_opt: sum of the optimal (min) runtime
    - time_ratio: total_time_pred / total_time_opt
    - avg_overhead: mean((pred-runtime − optimal)/optimal)
    """
    # runtime of predicted algorithm
    rt_pred = results_df.apply(
        lambda r: r[r['predicted']], axis=1
    )
    rt_opt  = results_df[algorithms].min(axis=1)
    
    total_pred = rt_pred.sum()
    total_opt  = rt_opt.sum()
    ratio      = total_pred / total_opt
    
    overheads  = (rt_pred - rt_opt) / (rt_opt + epsilon)
    avg_overhd = overheads.mean()
    
    out = {
        'total_time_pred': total_pred,
        'total_time_opt': total_opt,
        'time_ratio': ratio,
        'avg_overhead_per_case': avg_overhd
    }

    print("Time Metrics:")
    print(f"  Total Predicted Time = {out['total_time_pred']:.2f}")
    print(f"  Total Optimal Time   = {out['total_time_opt']:.2f}")
    print(f"  Time Ratio (pred/opt)= {out['time_ratio']:.3f}")
    print(f"  Avg Overhead/case ((pred - opt) / opt)    = {out['avg_overhead_per_case']:.2%}\n")

    return out

#### ------ Evaluation metrics with tolerences -----####

def get_tol_fastest_algorithms(row, algorithms, tol=0.0):
    """
    Return all algorithms whose runtime is within (1 + tol)*min_runtime.
    """
    times    = np.array([row[alg] for alg in algorithms], dtype=float)
    min_time = times.min()
    thresh   = (1 + tol) * min_time
    return [alg for alg, t in zip(algorithms, times) if t <= thresh]

def mark_correct_with_tol(df, algorithms, tol=0.0):
    """
    Adds two columns to `df`:
      - 'tol_fastest' : the list of tol‐fastest algorithms
      - 'correct_tol' : True if predicted ∈ tol_fastest
    """
    df = df.copy()
    df['tol_fastest'] = df.apply(lambda r: get_tol_fastest_algorithms(r, algorithms, tol), axis=1)
    df['correct_tol'] = df.apply(lambda r: r['predicted'] in r['tol_fastest'], axis=1)
    return df

def evaluate_with_tolerance(df, algorithms, tol=0.0):
    """
    Like evaluate_algorithm_performance, but using the tol‐based ‘correct_tol’
    and ‘tol_fastest’ instead of the strict versions.
    """
    df = mark_correct_with_tol(df, algorithms, tol)
    out = {}
    out['accuracy'] = df['correct_tol'].mean()
    print(f"Accuracy (±{tol*100:.1f}% ties counted as correct): {out['accuracy']:.2%}")

    # Precision
    out['precision_per_alg'] = {}
    print("\nPrecision by Algorithm:")
    for alg in algorithms:
        mask      = df['predicted'] == alg
        prec      = df.loc[mask, 'correct_tol'].mean() if mask.any() else 0.0
        count     = mask.sum()
        out['precision_per_alg'][alg] = prec
        print(f"  {alg}: {prec:.2%} ({count} preds)")

    # Avg precision by algorithm
    out['ap_per_alg'] = {}
    print("\nAverage Precision (AP) by Algorithm:")
    for alg in algorithms:
        prob_col = f"prob_{alg}"
        if prob_col not in df.columns:
            raise ValueError(f"Missing probability column: {prob_col}")

        # Binary ground truth: is alg among tol-fastest?
        y_true = df['tol_fastest'].apply(lambda L: alg in L).astype(int).values
        
        # Predicted probability
        y_score = df[prob_col].values

        N = len(df)
        n_pos = y_true.sum()
        n_neg = N - n_pos

        # Only compute AP if positives exist
        if n_pos == 0:
            ap = np.nan
            print(f"  {alg}: N/A (no positive instances)")
        else:
            ap = average_precision_score(y_true, y_score)
            pos_rate = n_pos / N
            print(f"  {alg}: {ap:.4f}")
            # print(f"  {alg}: {ap:.4f}, (positives={n_pos}, negatives={n_neg}, pos_rate={pos_rate:.2%})")

        out['ap_per_alg'][alg] = ap

    # Recall
    out['recall_per_alg'] = {}
    print("\nRecall by Algorithm:")
    for alg in algorithms:
        is_true   = df['tol_fastest'].apply(lambda L: alg in L)
        total_true= is_true.sum()
        if total_true > 0:
            tp     = ((df['correct_tol']) & is_true).sum()
            rec    = tp / total_true
        else:
            tp, rec = 0, 0.0
        out['recall_per_alg'][alg] = rec
        print(f"  {alg}: {rec:.2%} ({tp}/{total_true})")

    return out


def evaluate_time_metrics_with_tol(
    results_df,
    algorithms,
    tol: float = 0.0,
    epsilon: float = 1e-3
):
    """
    - total_time_pred: sum over chosen algo's runtime (capped to optimal if within tol)
    - total_time_opt: sum of the optimal (min) runtime
    - time_ratio: total_time_pred / total_time_opt
    - avg_overhead: mean((effective_pred − optimal)/optimal)

    Parameters
    ----------
    results_df : DataFrame
      Must have 'predicted' and one column per alg in algorithms.
    algorithms : list[str]
      Names of the runtime columns.
    tol : float
      Relative tolerance, e.g. 0.05 allows pred_time ≤1.05*opt be counted as opt.
    epsilon : float
      For numeric stability in division.
    """
    # 1) True optimal times
    rt_opt = results_df[algorithms].min(axis=1)

    # 2) Raw predicted times
    rt_pred_raw = results_df.apply(lambda r: r[r['predicted']], axis=1)

    # 3) Cap predicted time to optimal if within tolerance
    thresh = (1 + tol) * rt_opt
    rt_pred = rt_pred_raw.where(rt_pred_raw > thresh, rt_opt)

    # 4) Summaries
    total_pred = rt_pred.sum()
    total_opt  = rt_opt.sum()
    ratio      = total_pred / (total_opt + epsilon)

    # 5) Overheads per case
    overheads  = (rt_pred - rt_opt) / (rt_opt + epsilon)
    avg_overhd = overheads.mean()

    out = {
        'total_time_pred': total_pred,
        'total_time_opt': total_opt,
        'time_ratio': ratio,
        'avg_overhead_per_case': avg_overhd
    }

    # 6) Print
    print("Time Metrics:")
    print(f"  Total Predicted Time = {out['total_time_pred']:.2f}")
    print(f"  Total Optimal Time   = {out['total_time_opt']:.2f}")
    print(f"  Time Ratio (pred/opt)= {out['time_ratio']:.3f}")
    print(f"  Avg Overhead/case ((pred - opt) / opt)    = {out['avg_overhead_per_case']:.2%}\n")

    return out