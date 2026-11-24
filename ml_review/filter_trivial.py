"""Does removing trivial instances (every algorithm finishes in < THRESH s, total time)
help the models on the instances that matter?

  A  train on all instances           -> evaluate on non-trivial test instances
  B  train on non-trivial instances   -> evaluate on non-trivial test instances
  B* model B evaluated on ALL test instances (in practice you can't know in advance
     that an instance is trivial, so the filtered model still meets easy ones)

Graph-grouped 5-fold CV x 3 repeats, same folds as the other scripts. The trivial
flag is based on total time and used for both targets.

Usage: python filter_trivial.py [--thresh 5] [--targets total bnb] [--repeats 3]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from model_zoo import catboost_clf, catboost_reg, lgbm_clf, run_classifier, run_pairwise, run_regressor, topk
from runtime_selector import OUT, load_all

MODELS = {
    "CLF  CatBoost": lambda a, b, c: run_classifier(catboost_clf, a, b, c),
    "CLF  LightGBM": lambda a, b, c: run_classifier(lgbm_clf, a, b, c),
    "CLF  Random forest": lambda a, b, c: run_classifier(
        lambda: RandomForestClassifier(500, min_samples_leaf=2, n_jobs=-1, random_state=0), a, b, c),
    "PAIR LightGBM pairwise voting": run_pairwise,
    "REG  Random forest": lambda a, b, c: run_regressor(
        lambda: RandomForestRegressor(400, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0), a, b, c),
    "REG  CatBoost (multi-output)": lambda a, b, c: run_regressor(catboost_reg, a, b, c),
}


def metrics(T, timed_out, scores, sbs_idx):
    r = np.arange(len(T))
    pick = scores.argmin(1)
    cost, best, sbs = T[r, pick], T.min(1), T[r, sbs_idx]
    out = topk(T, scores)
    out.update(time_ratio=cost.sum() / best.sum(),
               gap_closed=100 * (sbs.sum() - cost.sum()) / (sbs.sum() - best.sum()),
               pred_timeouts=int(timed_out[r, pick].sum()),
               excess_hours=(cost - best).sum() / 3600)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=5.0)
    ap.add_argument("--targets", nargs="*", default=["total", "bnb"])
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    _, _, T_total, _, _, _, _ = load_all("total")
    trivial = T_total.max(1) < a.thresh  # every algorithm under thresh seconds (total time)
    print(f"trivial instances (all algorithms < {a.thresh:g}s total time): {trivial.sum()} / {len(trivial)}")

    rows = []
    for target in a.targets:
        gdf, X, T, timed_out, algos, feats, _ = load_all(target)
        groups = gdf["dataset"].to_numpy()
        Xb = X[feats["base"]]
        for rep in range(a.repeats):
            t0 = time.time()
            sc = {(m, v): np.zeros(T.shape) for m in MODELS for v in ("A", "B")}
            sbs = {v: np.zeros(len(T), int) for v in ("A", "B")}
            for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=groups):
                tr_b = tr[~trivial[tr]]
                for v, idx in (("A", tr), ("B", tr_b)):
                    sbs[v][te] = T[idx].sum(0).argmin()
                    for m, fn in MODELS.items():
                        sc[(m, v)][te] = fn(Xb.iloc[idx], T[idx], Xb.iloc[te])
            nt = ~trivial
            for m in MODELS:
                for v, mask, label in (("A", nt, "A: train all, eval non-trivial"),
                                       ("B", nt, "B: train non-trivial, eval non-trivial"),
                                       ("B", np.ones(len(T), bool), "B*: train non-trivial, eval all")):
                    r = metrics(T[mask], timed_out[mask], sc[(m, v)][mask], sbs[v][mask])
                    rows.append(dict(target=target, model=m, setup=label, repeat=rep, **r))
            pd.DataFrame(rows).to_csv(OUT / f"filter_trivial_{a.thresh:g}s_raw.csv", index=False)
            print(f"[{target}] repeat {rep} done ({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    cols = ["top1", "top3", "top3_10pct", "time_ratio", "gap_closed", "pred_timeouts", "excess_hours"]
    summ = res.groupby(["target", "model", "setup"])[cols].mean()
    summ.to_csv(OUT / f"filter_trivial_{a.thresh:g}s_summary.csv")
    pd.set_option("display.width", 230)
    for t in a.targets:
        print(f"\n=== target={t}, threshold {a.thresh:g}s")
        print(summ.loc[t].round(2).to_string())


if __name__ == "__main__":
    main()
