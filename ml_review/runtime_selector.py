"""Runtime-prediction algorithm selector for the 24 k-plex algorithms.

Instead of classifying the winner, predict every algorithm's (log) runtime and
run the one with the smallest prediction. Evaluated with graph-grouped CV so no
graph appears in both train and test.

Data: kplex_ML_data.xlsx (per-phase times for all 24 algorithms) + the student's
graph features from ml-data.csv.

Feature sets
  base   : the student's 12 graph features
  eng    : base + 18 scale-free graph features (see algo_selection_eval.py)
  probe  : eng + features from running ONE cheap heuristic first
           (heuristic k-plex size = lower bound, heuristic runtime, gap to the
           core+k upper bound). The probe's runtime (capped) is charged to the
           selector's cost so the comparison stays fair.

Usage
  python runtime_selector.py --stage grid   # target/feature/model/rule sweep
  python runtime_selector.py --stage final  # best configs, 3 repeats, per-k table
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from algo_selection_eval import load as load_graph_features

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "kplex_ML_data.xlsx"
OUT = Path(__file__).parent / "results"
TIMEOUT = 1800.0
EPS = 1e-3
PROBE_ALGO = "UMKP-oriented-B"  # cheapest median heuristic time
PROBE_CAP = 5.0  # seconds; if the heuristic takes longer we stop it and go without its result


# ----------------------------------------------------------------------------- data
def load_all(target: str):
    gdf, X_eng, _, _, _, feats_eng = load_graph_features("bnb")  # graph features (drops 1 corrupt row)
    x = pd.read_excel(XLSX)
    x = gdf[["dataset", "k_value"]].merge(
        x.rename(columns={"k": "k_value"}), on=["dataset", "k_value"], how="left", validate="1:1"
    )
    algos = [c[: -len("-total")] for c in x.columns if str(c).endswith("-total")]
    num = lambda c: pd.to_numeric(x[c], errors="coerce")

    if target == "total":
        raw = pd.DataFrame({a: num(f"{a}-total") for a in algos})
        timed_out = raw.isna().to_numpy()
    else:  # bnb: "x" with a finished total means BnB was skipped (0 s); with an unfinished total it is a timeout
        tot = pd.DataFrame({a: num(f"{a}-total") for a in algos})
        raw = pd.DataFrame({a: num(f"{a}-bnb") for a in algos})
        raw = raw.where(raw.notna() | tot.isna(), 0.0)
        timed_out = raw.isna().to_numpy()
    T = raw.fillna(TIMEOUT).to_numpy()

    # ---- probe features: run PROBE_ALGO's heuristic, stop it at PROBE_CAP
    h_time = num(f"{PROBE_ALGO}-heur").fillna(TIMEOUT)
    finished = h_time <= PROBE_CAP
    lb = num(f"{PROBE_ALGO}-heur-kp").where(finished)  # identical across algorithms
    k = gdf["k_value"].astype(float)
    ub = gdf["core_max"].astype(float) + k  # max k-plex size <= core number + k
    P = pd.DataFrame(index=gdf.index)
    P["probe_finished"] = finished.astype(float)
    P["probe_log_time"] = np.log10(h_time.clip(upper=PROBE_CAP) + EPS)
    P["lb"] = lb
    P["lb_minus_k"] = lb - k
    P["ub_minus_lb"] = ub - lb
    P["lb_over_ub"] = lb / ub
    P["lb_over_core"] = lb / gdf["core_max"].astype(float).clip(lower=1)
    P["lb_eq_ub"] = (lb >= ub).astype(float).where(finished)
    P = P.fillna(-1)  # tree models: -1 = "probe did not finish"
    probe_cost = h_time.clip(upper=PROBE_CAP).to_numpy()

    X = pd.concat([X_eng, P], axis=1)
    feats = {
        "base": feats_eng[:12],
        "eng": feats_eng,
        "probe": feats_eng + list(P.columns),
    }
    return gdf, X, T, timed_out, algos, feats, probe_cost


# ----------------------------------------------------------------------------- metrics
def evaluate(T, timed_out, pred, sbs_row, extra_cost=None):
    r = np.arange(len(T))
    cost = T[r, pred] + (0 if extra_cost is None else extra_cost)
    best = T.min(1)
    sbs = T[r, sbs_row]
    return dict(
        acc=np.mean(T[r, pred] <= best),
        acc_10pct=np.mean(T[r, pred] <= best * 1.1),
        time_ratio=cost.sum() / best.sum(),
        gap_closed=(sbs.sum() - cost.sum()) / (sbs.sum() - best.sum()),
        geo_slowdown=float(np.exp(np.mean(np.log((cost + 0.01) / (best + 0.01))))),
        pred_timeouts=int(timed_out[r, pred].sum()),
        total_hours=cost.sum() / 3600,
    )


# ----------------------------------------------------------------------------- targets & models
def make_target(Ttr, timed_tr, mode):
    Y = Ttr.copy()
    if mode == "par10":  # timeouts count as 10x the limit -> model learns to fear them
        Y[timed_tr] = 10 * TIMEOUT
    Y = np.log10(Y + EPS)
    if mode == "relative":  # log slowdown vs the row's best: removes instance difficulty
        Y = Y - Y.min(1, keepdims=True)
    return Y


def per_tree(model, X):
    return np.stack([t.predict(X) for t in model.estimators_])  # trees x rows x algos


def fit_forest(kind, Xtr, Y, Xte, leaf, mf, n=400):
    cls = RandomForestRegressor if kind == "rf" else ExtraTreesRegressor
    m = cls(n_estimators=n, min_samples_leaf=leaf, max_features=mf, n_jobs=-1, random_state=0)
    m.fit(Xtr, Y)
    return per_tree(m, Xte.to_numpy())


def fit_lgbm(Xtr, Y, Xte, **kw):
    import lightgbm as lgb
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=15,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                  verbose=-1, n_jobs=8)
    params.update(kw)
    out = np.zeros((len(Xte), Y.shape[1]))
    for j in range(Y.shape[1]):
        out[:, j] = lgb.LGBMRegressor(random_state=j, **params).fit(Xtr, Y[:, j]).predict(Xte)
    return out[None]  # 1 x rows x algos (no per-tree spread)


def select(pt, rule):
    """pt: members x rows x algos of predicted log10 times."""
    if rule == "mean_log":
        return pt.mean(0).argmin(1)
    if rule == "expected_time":  # average of predicted *times*: punishes algorithms that sometimes blow up
        return (10.0 ** pt).mean(0).argmin(1)
    raise ValueError(rule)


# ----------------------------------------------------------------------------- CV driver
def cv_predict(X, T, timed_out, groups, feats, configs, rep):
    """configs: name -> dict(model=..., target=..., rules=[...]). Returns name/rule -> preds."""
    preds, sbs_row = {}, np.zeros(len(T), int)
    for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=groups):
        sbs_row[te] = T[tr].sum(0).argmin()
        cache = {}
        for name, c in configs.items():
            Xtr, Xte = X.iloc[tr][feats[c["features"]]], X.iloc[te][feats[c["features"]]]
            Y = make_target(T[tr], timed_out[tr], c["target"])
            parts = []
            for m in c["model"].split("+"):
                key = (m, c["features"], c["target"], c.get("leaf"), c.get("mf"))
                if key not in cache:
                    if m in ("rf", "et"):
                        cache[key] = fit_forest(m, Xtr, Y, Xte, c.get("leaf", 3), c.get("mf", 0.5))
                    else:
                        cache[key] = fit_lgbm(Xtr, Y, Xte)
                parts.append(cache[key])
            pt = np.concatenate(parts, 0) if len(parts) == 1 else np.concatenate(
                [p.mean(0, keepdims=True) if p.shape[0] > 1 else p for p in parts], 0)
            for rule in c["rules"]:
                preds.setdefault(f"{name} | {rule}", np.zeros(len(T), int))[te] = select(pt, rule)
    return preds, sbs_row


def summarize(records):
    res = pd.DataFrame(records)
    s = res.groupby("config").agg(["mean", "std"])
    cols = ["acc", "acc_10pct", "time_ratio", "gap_closed", "geo_slowdown", "pred_timeouts", "total_hours"]
    out = pd.DataFrame({c: s[(c, "mean")] for c in cols})
    out["time_ratio_sd"] = s[("time_ratio", "std")]
    out[["acc", "acc_10pct", "gap_closed"]] *= 100
    return out.sort_values("time_ratio")


def run_stage(target, configs, repeats, tag):
    gdf, X, T, to, algos, feats, probe_cost = load_all(target)
    groups = gdf["dataset"].to_numpy()
    recs = []
    for rep in range(repeats):
        preds, sbs_row = cv_predict(X, T, to, groups, feats, configs, rep)
        for name, p in preds.items():
            uses_probe = configs[name.split(" | ")[0]]["features"] == "probe"
            r = evaluate(T, to, p, sbs_row, probe_cost if uses_probe else None)
            r.update(config=name, repeat=rep)
            recs.append(r)
        r = evaluate(T, to, sbs_row, sbs_row); r.update(config="Single best solver", repeat=rep); recs.append(r)
        r = evaluate(T, to, T.argmin(1), sbs_row); r.update(config="Oracle (always fastest)", repeat=rep); recs.append(r)
        print(f"  [{target}] repeat {rep} done", flush=True)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(recs).to_csv(OUT / f"runtime_{tag}_{target}_raw.csv", index=False)
    summ = summarize(recs)
    summ.to_csv(OUT / f"runtime_{tag}_{target}_summary.csv")
    return summ, (gdf, X, T, to, algos, feats, probe_cost)


GRID = {
    # feature sets (RF, log target)
    "RF base log": dict(model="rf", features="base", target="log", rules=["mean_log", "expected_time"]),
    "RF eng log": dict(model="rf", features="eng", target="log", rules=["mean_log", "expected_time"]),
    "RF probe log": dict(model="rf", features="probe", target="log", rules=["mean_log", "expected_time"]),
    # timeout encoding / relative target
    "RF probe par10": dict(model="rf", features="probe", target="par10", rules=["mean_log", "expected_time"]),
    "RF probe relative": dict(model="rf", features="probe", target="relative", rules=["mean_log", "expected_time"]),
    # forest hyperparameters
    "RF probe log leaf1": dict(model="rf", features="probe", target="log", leaf=1, rules=["expected_time"]),
    "RF probe log leaf8": dict(model="rf", features="probe", target="log", leaf=8, rules=["expected_time"]),
    "RF probe log mf0.3": dict(model="rf", features="probe", target="log", mf=0.3, rules=["expected_time"]),
    "RF probe log mf1.0": dict(model="rf", features="probe", target="log", mf=1.0, rules=["expected_time"]),
    # other learners
    "ET probe log": dict(model="et", features="probe", target="log", mf=0.7, rules=["mean_log", "expected_time"]),
    "ET probe par10": dict(model="et", features="probe", target="par10", mf=0.7, rules=["mean_log", "expected_time"]),
    "LGBM probe log": dict(model="lgbm", features="probe", target="log", rules=["mean_log"]),
    "RF+ET+LGBM probe log": dict(model="rf+et+lgbm", features="probe", target="log", mf=0.7, rules=["mean_log", "expected_time"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["grid", "final"], default="grid")
    ap.add_argument("--targets", nargs="*", default=["total", "bnb"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--configs", nargs="*", default=None, help="subset of GRID names")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    cfgs = {k: v for k, v in GRID.items() if not a.configs or k in a.configs}
    for t in a.targets:
        t0 = time.time()
        summ, _ = run_stage(t, cfgs, a.repeats, a.stage)
        print(f"\n=== target={t}  ({time.time() - t0:.0f}s)")
        print(summ.round(2).to_string())


if __name__ == "__main__":
    main()
