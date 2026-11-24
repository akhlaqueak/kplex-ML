"""Where does the runtime selector lose time, and what would help?

Diagnostics on top of runtime_selector.py (RF regression of log runtime, probe features):
  1. error decomposition (timeouts vs slow-but-finished picks)
  2. oracle-feature check: add the true optimum size (NOT deployable) to see if hardness info is the bottleneck
  3. learning curve: train on 25/50/75/100% of training graphs
  4. timeout guard: classifier for P(timeout) per algorithm, penalise risky picks
  5. parallel portfolio: run top-2 / top-3 predicted algorithms in parallel (wall-clock = fastest of them)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from runtime_selector import EPS, OUT, TIMEOUT, XLSX, evaluate, load_all

target = sys.argv[1] if len(sys.argv) > 1 else "total"
REPEATS = 3
gdf, X, T, timed_out, algos, feats, probe_cost = load_all(target)
groups = gdf["dataset"].to_numpy()
F = feats["probe"]

# true optimum size (diagnostic only)
x = pd.read_excel(XLSX).rename(columns={"k": "k_value"})
mkp = gdf[["dataset", "k_value"]].merge(x, on=["dataset", "k_value"], how="left")
opt = pd.DataFrame({a: pd.to_numeric(mkp[f"{a}-mkp"], errors="coerce") for a in algos}).max(1)
X = X.copy()
X["oracle_opt"] = opt.fillna(-1).to_numpy()
X["oracle_opt_minus_lb"] = np.where(X["lb"] >= 0, X["oracle_opt"] - X["lb"], -1)


def rf_reg(Xtr, Ttr, Xte, seed=0):
    m = RandomForestRegressor(400, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=seed)
    m.fit(Xtr, np.log10(Ttr + EPS))
    return m.predict(Xte)


recs, dec = [], []
for rep in range(REPEATS):
    P = {}
    sbs = np.zeros(len(T), int)
    rng = np.random.default_rng(rep)
    for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=groups):
        sbs[te] = T[tr].sum(0).argmin()
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        pred = rf_reg(Xtr[F], T[tr], Xte[F])
        P.setdefault("RF probe (baseline)", np.zeros(len(T), int))[te] = pred.argmin(1)

        # 2. oracle feature
        po = rf_reg(Xtr[F + ["oracle_opt", "oracle_opt_minus_lb"]], T[tr], Xte[F + ["oracle_opt", "oracle_opt_minus_lb"]])
        P.setdefault("+ true optimum size (not deployable)", np.zeros(len(T), int))[te] = po.argmin(1)

        # 3. learning curve on graphs
        tr_graphs = np.unique(groups[tr])
        for frac in (0.25, 0.5, 0.75):
            keep = np.isin(groups[tr], rng.choice(tr_graphs, int(frac * len(tr_graphs)), replace=False))
            pl = rf_reg(Xtr[F][keep], T[tr][keep], Xte[F])
            P.setdefault(f"train on {int(frac*100)}% of graphs", np.zeros(len(T), int))[te] = pl.argmin(1)

        # 4. timeout guard: P(timeout) per algorithm, penalised expected cost
        risk = np.zeros((len(te), T.shape[1]))
        clf = RandomForestClassifier(400, min_samples_leaf=3, n_jobs=-1, random_state=0)
        clf.fit(Xtr[F], timed_out[tr])  # multi-output classification
        for j, est_proba in enumerate(clf.predict_proba(Xte[F])):
            risk[:, j] = est_proba[:, 1] if est_proba.shape[1] == 2 else (1.0 if clf.classes_[j][0] else 0.0)
        pred_time = 10 ** pred
        for lam in (0.5, 1.0, 2.0):
            score = (1 - risk) * pred_time + risk * lam * TIMEOUT
            P.setdefault(f"timeout guard (penalty x{lam})", np.zeros(len(T), int))[te] = score.argmin(1)

        # 5. portfolios: store ranking for later
        P.setdefault("_rank", np.zeros((len(T), 3), int))[te] = np.argsort(pred, 1)[:, :3]

    rank = P.pop("_rank")
    for name, p in P.items():
        r = evaluate(T, timed_out, p, sbs, probe_cost)
        r.update(config=name, repeat=rep); recs.append(r)
    r_ = np.arange(len(T))
    for kk in (2, 3):  # wall-clock = fastest of the k chosen algorithms; CPU cost = k x that
        wall = T[r_[:, None], rank[:, :kk]].min(1)
        best_in = rank[r_, T[r_[:, None], rank[:, :kk]].argmin(1)]
        r = evaluate(T, timed_out, best_in, sbs, probe_cost)
        r["note"] = f"{kk} cores"
        r.update(config=f"run top-{kk} in parallel (wall-clock)", repeat=rep); recs.append(r)

    # 1. error decomposition for the baseline
    p = P["RF probe (baseline)"]
    cost = T[r_, p] + probe_cost
    best = T.min(1)
    excess = cost - best
    to_pick = timed_out[r_, p]
    dec.append({
        "excess_hours_total": excess.sum() / 3600,
        "from_timeout_picks_%": 100 * excess[to_pick].sum() / excess.sum(),
        "timeout_picks": int(to_pick.sum()),
        "of_which_some_algo_finishes_<60s_%": 100 * np.mean(best[to_pick] < 60),
        "from_probe_cost_%": 100 * probe_cost.sum() / excess.sum(),
        "from_other_picks_%": 100 * (excess[~to_pick].sum() - probe_cost[~to_pick].sum()) / excess.sum(),
    })
    print(f"repeat {rep} done", flush=True)

res = pd.DataFrame(recs)
summ = res.groupby("config")[["acc", "acc_10pct", "time_ratio", "gap_closed", "geo_slowdown", "pred_timeouts", "total_hours"]].mean()
summ[["acc", "acc_10pct", "gap_closed"]] *= 100
summ = summ.sort_values("time_ratio")
pd.set_option("display.width", 220)
print(f"\n=== target={target}  (graph-grouped 5-fold CV x {REPEATS})")
print(summ.round(2).to_string())
print("\nError decomposition of the baseline (mean over repeats):")
print(pd.DataFrame(dec).mean().round(1).to_string())
summ.to_csv(OUT / f"ceiling_{target}_summary.csv")
pd.DataFrame(dec).to_csv(OUT / f"ceiling_{target}_decomposition.csv", index=False)
