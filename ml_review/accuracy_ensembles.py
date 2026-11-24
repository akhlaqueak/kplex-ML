"""Ensembles built from the saved out-of-fold scores of accuracy_study.py (no refitting).

Each method's 24-column scores are row-normalised to sum to 1 and averaged.
Reports: fixed ensembles chosen in advance, and a greedy forward selection
(note: the forward selection picks members on the same CV results it reports,
so its number is mildly optimistic; the fixed ensembles are not).

Usage: python accuracy_ensembles.py --target total
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from accuracy_study import CONFIGS, SCORES_DIR, TRIVIAL_THRESH, acc_metrics
from runtime_selector import OUT, load_all


def load_scores(target, repeats):
    S = {}
    for name in CONFIGS:
        mats = []
        for r in range(repeats):
            f = SCORES_DIR / f"{target}__{name}__rep{r}.npy"
            if not f.exists():
                break
            M = np.clip(np.load(f), 0, None)
            mats.append(M / np.clip(M.sum(1, keepdims=True), 1e-12, None))
        if len(mats) == repeats:
            S[name] = mats
    return S


def score(T, mats, mask, members):
    res = []
    for r in range(len(mats[members[0]])):
        avg = sum(mats[m][r] for m in members) / len(members)
        res.append(acc_metrics(T, avg, mask))
    return pd.DataFrame(res).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="total")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    _, _, Tt, _, _, _, _ = load_all("total")
    mask = Tt.max(1) >= TRIVIAL_THRESH
    _, _, T, _, _, _, _ = load_all(a.target)
    mats = load_scores(a.target, a.repeats)
    print(f"{len(mats)} methods with {a.repeats} repeats available")

    single = pd.DataFrame({m: score(T, mats, mask, [m]) for m in mats}).T.sort_values("top1", ascending=False)
    rows = {}
    fixed = {
        "Ens: flat CatBoost + LightGBM + XGBoost + RF": ["Flat CatBoost", "Flat LightGBM", "Flat XGBoost (student config)", "Flat Random forest"],
        "Ens: flat CatBoost + hier family>variant winner-only": ["Flat CatBoost", "Hier family > variant, winner-only, greedy"],
        "Ens: top-3 single methods": list(single.index[:3]),
        "Ens: top-5 single methods": list(single.index[:5]),
    }
    for name, members in fixed.items():
        if all(m in mats for m in members):
            rows[name] = score(T, mats, mask, members)

    # greedy forward selection on top-1 (with replacement allowed = implicit weighting)
    chosen, best = [], -1
    for _ in range(8):
        cand = {m: score(T, mats, mask, chosen + [m])["top1"] for m in mats}
        m, v = max(cand.items(), key=lambda kv: kv[1])
        if v <= best + 0.05:
            break
        chosen.append(m); best = v
    rows[f"Ens: greedy selection ({len(chosen)} members, optimistic)"] = score(T, mats, mask, chosen)

    out = pd.concat([pd.DataFrame(rows).T, single.head(5)])
    pd.set_option("display.width", 220)
    print(out[["top1", "top2", "top3", "top1_10pct", "top3_10pct", "time_ratio"]].round(2).to_string())
    print("\ngreedy members:", chosen)
    out.to_csv(OUT / f"accuracy_ensembles_{a.target}.csv")


if __name__ == "__main__":
    main()
