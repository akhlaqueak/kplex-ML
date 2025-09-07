import numpy as np
import pandas as pd


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

