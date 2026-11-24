"""Single-core fallback schedules on top of the runtime selector.

Run the predicted-fastest algorithm with a cutoff; if it has not finished, kill it
and run the 2nd prediction (optionally a 3rd) to completion. Cost = wall-clock =
CPU time on one core. Compared with plain selection and with parallel top-k.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from runtime_selector import EPS, OUT, TIMEOUT, evaluate, load_all

target = sys.argv[1] if len(sys.argv) > 1 else "total"
gdf, X, T, timed_out, algos, feats, probe_cost = load_all(target)
groups = gdf["dataset"].to_numpy()
F = feats["probe"]
r_ = np.arange(len(T))
best = T.min(1)


def schedule_cost(rank, pred_log, cut):
    """cut: fixed seconds (number) or ('rel', c) = max(10, c x predicted time of #1)."""
    t1, t2, t3 = (T[r_, rank[:, i]] for i in range(3))
    if isinstance(cut, tuple):
        c = np.maximum(10.0, cut[1] * 10 ** pred_log[r_, rank[:, 0]])
        c2 = None
    elif isinstance(cut, list):  # three-stage: [c1, c2]
        c, c2 = cut
    else:
        c, c2 = float(cut), None
    first_ok = t1 <= c
    cost = np.where(first_ok, t1, c + (t2 if c2 is None else np.where(t2 <= c2, t2, c2 + t3)))
    final = np.where(first_ok, rank[:, 0], rank[:, 1] if c2 is None else np.where(t2 <= c2, rank[:, 1], rank[:, 2]))
    # never pay more than the timeout budget for the last stage's own run
    return np.minimum(cost, np.where(first_ok, TIMEOUT, cost)), final


rows = []
CUTS = [10, 30, 60, 120, 300, ("rel", 3), ("rel", 10), [30, 300], [60, 600]]
for rep in range(3):
    sbs = np.zeros(len(T), int)
    rank = np.zeros((len(T), 3), int)
    pred_log = np.zeros(T.shape)
    for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=groups):
        sbs[te] = T[tr].sum(0).argmin()
        m = RandomForestRegressor(400, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0)
        m.fit(X.iloc[tr][F], np.log10(T[tr] + EPS))
        pred_log[te] = m.predict(X.iloc[te][F])
        rank[te] = np.argsort(pred_log[te], 1)[:, :3]
    sbs_total = T[r_, sbs].sum()

    def rec(name, cost, final, cores=1):
        cost = cost + probe_cost
        rows.append(dict(config=name, repeat=rep, cores=cores,
                         time_ratio=cost.sum() / best.sum(),
                         gap_closed=100 * (sbs_total - cost.sum()) / (sbs_total - best.sum()),
                         geo_slowdown=float(np.exp(np.mean(np.log((cost + .01) / (best + .01))))),
                         unsolved_after_schedule=int(timed_out[r_, final].sum()),
                         total_hours=cost.sum() / 3600))

    rec("select #1 only", T[r_, rank[:, 0]], rank[:, 0])
    for cut in CUTS:
        cost, final = schedule_cost(rank, pred_log, cut)
        label = (f"#1 up to {cut[1]}x its predicted time (min 10s), then #2" if isinstance(cut, tuple)
                 else f"#1 up to {cut[0]}s, #2 up to {cut[1]}s, then #3" if isinstance(cut, list)
                 else f"#1 up to {cut}s, then #2")
        rec(label, cost, final)
    for kk in (2, 3):
        w = T[r_[:, None], rank[:, :kk]]
        rec(f"top-{kk} in parallel", w.min(1), rank[r_, w.argmin(1)], cores=kk)
    print(f"repeat {rep} done", flush=True)

res = pd.DataFrame(rows).groupby(["config", "cores"]).mean(numeric_only=True).drop(columns="repeat")
res = res.sort_values("time_ratio")
pd.set_option("display.width", 200)
print(f"\n=== target={target}")
print(res.round(2).to_string())
res.to_csv(OUT / f"schedule_{target}_summary.csv")
