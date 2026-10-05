"""Shared metric helpers reused from the legacy notebooks.
Checked"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def evaluate_algorithm_performance(results_df, algorithms):
    out = {}
    accuracy = results_df["correct"].mean()
    out["accuracy"] = accuracy
    print(f"Accuracy (counting ties as correct): {accuracy:.2%}")

    out["precision_per_alg"] = {}
    print("\nPrecision by Algorithm:")
    for alg in algorithms:
        mask = results_df["predicted"] == alg
        precision = results_df.loc[mask, "correct"].mean() if mask.any() else 0.0
        count = mask.sum()
        out["precision_per_alg"][alg] = precision
        print(f"  {alg}: {precision:.2%} ({count} predictions)")

    out["recall_per_alg"] = {}
    print("\nRecall by Algorithm:")
    for alg in algorithms:
        was_fastest = results_df["fastest_algorithms"].apply(lambda fastest: alg in fastest)
        total_fastest = was_fastest.sum()

        if total_fastest > 0:
            true_positives = (results_df["correct"] & was_fastest).sum()
            recall = true_positives / total_fastest
            out["recall_per_alg"][alg] = recall
        else:
            recall = 0.0
            true_positives = 0
            out["recall_per_alg"][alg] = 0.0

        print(f"  {alg}: {recall:.2%} ({true_positives}/{total_fastest} cases)")

    return out


def evaluate_time_metrics(results_df, algorithms, epsilon=1e-3):
    rt_pred = results_df.apply(lambda r: r[r["predicted"]], axis=1)
    rt_opt = results_df[algorithms].min(axis=1)

    total_pred = rt_pred.sum()
    total_opt = rt_opt.sum()
    ratio = total_pred / total_opt

    # overheads = (rt_pred - rt_opt) / (rt_opt + epsilon)
    # avg_overhd = overheads.mean()

    out = {
        "total_time_pred": total_pred,
        "total_time_opt": total_opt,
        "time_ratio": ratio,
        # "avg_overhead_per_case": avg_overhd,
    }

    print("Time Metrics:")
    print(f"  Total Predicted Time = {out['total_time_pred']:.2f}")
    print(f"  Total Optimal Time   = {out['total_time_opt']:.2f}")
    print(f"  Time Ratio (pred/opt)= {out['time_ratio']:.3f}")
    # print(f"  Avg Overhead/case ((pred - opt) / opt)    = {out['avg_overhead_per_case']:.2%}\n")

    return out


def get_tol_fastest_algorithms(row, algorithms, tol=0.0):
    times = np.array([row[alg] for alg in algorithms], dtype=float)
    min_time = times.min()
    thresh = (1 + tol) * min_time
    return [alg for alg, t in zip(algorithms, times) if t <= thresh]


def mark_correct_with_tol(df, algorithms, tol=0.0):
    df = df.copy()
    df["tol_fastest"] = df.apply(lambda r: get_tol_fastest_algorithms(r, algorithms, tol), axis=1)
    df["correct_tol"] = df.apply(lambda r: r["predicted"] in r["tol_fastest"], axis=1)
    return df


def evaluate_with_tolerance(df, algorithms, tol=0.0):
    df = mark_correct_with_tol(df, algorithms, tol)
    out = {}
    out["accuracy"] = df["correct_tol"].mean()
    print(f"Accuracy (+/-{tol*100:.1f}% ties counted as correct): {out['accuracy']:.2%}")

    out["precision_per_alg"] = {}
    print("\nPrecision by Algorithm:")
    for alg in algorithms:
        mask = df["predicted"] == alg
        prec = df.loc[mask, "correct_tol"].mean() if mask.any() else 0.0
        count = mask.sum()
        out["precision_per_alg"][alg] = prec
        print(f"  {alg}: {prec:.2%} ({count} preds)")

    # out["ap_per_alg"] = {}
    # print("\nAverage Precision (AP) by Algorithm:")
    # for alg in algorithms:
    #     prob_col = f"prob_{alg}"
    #     if prob_col not in df.columns:
    #         raise ValueError(f"Missing probability column: {prob_col}")

    #     y_true = df["tol_fastest"].apply(lambda labels: alg in labels).astype(int).values
    #     y_score = df[prob_col].values

    #     n_pos = y_true.sum()
    #     if n_pos == 0:
    #         ap = np.nan
    #         print(f"  {alg}: N/A (no positive instances)")
    #     else:
    #         ap = average_precision_score(y_true, y_score)
    #         print(f"  {alg}: {ap:.4f}")

    #     out["ap_per_alg"][alg] = ap

    out["recall_per_alg"] = {}
    print("\nRecall by Algorithm:")
    for alg in algorithms:
        is_true = df["tol_fastest"].apply(lambda labels: alg in labels)
        total_true = is_true.sum()
        if total_true > 0:
            tp = (df["correct_tol"] & is_true).sum()
            rec = tp / total_true
        else:
            tp = 0
            rec = 0.0
        out["recall_per_alg"][alg] = rec
        print(f"  {alg}: {rec:.2%} ({tp}/{total_true})")

    return out


def evaluate_time_metrics_with_tol(results_df, algorithms, tol=0.0, epsilon=1e-3):
    rt_opt = results_df[algorithms].min(axis=1)
    rt_pred_raw = results_df.apply(lambda r: r[r["predicted"]], axis=1)

    thresh = (1 + tol) * rt_opt
    rt_pred = rt_pred_raw.where(rt_pred_raw > thresh, rt_opt)

    total_pred = rt_pred.sum()
    total_opt = rt_opt.sum()
    ratio = total_pred / (total_opt + epsilon)

    # overheads = (rt_pred - rt_opt) / (rt_opt + epsilon)
    # avg_overhd = overheads.mean()

    out = {
        "total_time_pred": total_pred,
        "total_time_opt": total_opt,
        "time_ratio": ratio,
        # "avg_overhead_per_case": avg_overhd,
    }

    print("Time Metrics:")
    print(f"  Total Predicted Time = {out['total_time_pred']:.2f}")
    print(f"  Total Optimal Time   = {out['total_time_opt']:.2f}")
    print(f"  Time Ratio (pred/opt)= {out['time_ratio']:.3f}")
    # print(f"  Avg Overhead/case ((pred - opt) / opt)    = {out['avg_overhead_per_case']:.2%}\n")

    return out


__all__ = [
    "evaluate_algorithm_performance",
    "evaluate_time_metrics",
    "evaluate_time_metrics_with_tol",
    "evaluate_with_tolerance",
    "get_tol_fastest_algorithms",
    "mark_correct_with_tol",
]
