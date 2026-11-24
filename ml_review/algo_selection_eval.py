"""Independent re-evaluation of the k-plex algorithm-selection ML.

Compares the student's Scheme-1 setup with alternatives under a leak-free
protocol (cross-validation grouped by graph), on either the BnB-time target
(what the student used) or the total-time target.

Usage:
    python algo_selection_eval.py --target bnb   [--protocol group|random] [--repeats 3]
    python algo_selection_eval.py --target total
"""

from __future__ import annotations

import argparse
import ast
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / "naman files/kpex_ML_git-main/input_data/ml-data.csv"
TIMEOUT = 1800.0
EPS = 1e-3  # seconds; floor for log-times (many BnB times are exactly 0)

STUDENT_FEATURES = [
    "k_value", "n", "m", "d_avg", "d_max", "core_max", "core_min",
    "shell_num", "shell_avgSize", "shell_maxSize", "shell_minSize", "shell_stdSize",
]


# ----------------------------------------------------------------------------- data
def parse_shells(s):
    s = re.sub(r"\)\s*\(", "),(", str(s).strip())
    return ast.literal_eval(s)


def load(target="bnb"):
    df = pd.read_csv(DATA)
    # one corrupted row (soc-youtube, k=20) has n == "x" and a shell list starting at level 28
    bad = pd.to_numeric(df["n"], errors="coerce").isna()
    if bad.any():
        print(f"dropping {bad.sum()} row(s) with non-numeric graph features: "
              f"{df.loc[bad, ['dataset', 'k']].values.tolist()}")
        df = df[~bad].reset_index(drop=True)
        df["n"] = df["n"].astype(float)
    bnb_cols = [c for c in df.columns if c.endswith("-bnb")]
    tot_cols = [c[:-4] for c in bnb_cols]
    cols = bnb_cols if target == "bnb" else tot_cols
    raw = df[cols].apply(pd.to_numeric, errors="coerce")  # the only non-numeric value is "x" (timeout)
    timed_out = raw.isna().to_numpy()
    T = raw.fillna(TIMEOUT).to_numpy()
    keep = ~timed_out.all(axis=1)
    df, T, timed_out = df[keep].reset_index(drop=True), T[keep], timed_out[keep]

    df = df.rename(columns={"shell_sizes": "kcore_list", "core_number": "core_max", "k": "k_value"})
    shells = df["kcore_list"].apply(parse_shells)
    sizes = shells.apply(lambda l: np.array([s for _, s in l], dtype=float))
    levels = shells.apply(lambda l: np.array([c for c, _ in l], dtype=float))

    # --- the student's 12 features (same definitions as features.py)
    df["shell_num"] = shells.apply(len)
    df["core_min"] = shells.apply(lambda l: l[0][0])
    df["shell_avgSize"] = sizes.apply(np.mean)
    df["shell_maxSize"] = sizes.apply(np.max)
    df["shell_minSize"] = sizes.apply(np.min)
    df["shell_stdSize"] = sizes.apply(np.std)

    # --- engineered features (scale-free ratios trees cannot build themselves)
    n, m, k = df["n"].astype(float), df["m"].astype(float), df["k_value"].astype(float)
    cm = df["core_max"].astype(float)
    f = pd.DataFrame(index=df.index)
    f["log_n"] = np.log10(n + 1)
    f["log_m"] = np.log10(m + 1)
    f["density"] = 2 * m / (n * (n - 1)).clip(lower=1)
    f["log_density"] = np.log10(f["density"] + 1e-12)
    f["m_over_n"] = m / n.clip(lower=1)
    f["dmax_over_n"] = df["d_max"] / n.clip(lower=1)
    f["davg_over_dmax"] = df["d_avg"] / df["d_max"].clip(lower=1)
    f["k_over_core"] = k / cm.clip(lower=1)
    f["core_minus_k"] = cm - k
    f["kplex_ub"] = cm + k  # max k-plex size <= core number + k
    f["davg_over_k"] = df["d_avg"] / k
    f["dmax_over_core"] = df["d_max"] / cm.clip(lower=1)
    top = sizes.apply(lambda a: a[-1])
    f["log_topcore"] = np.log10(top + 1)
    f["topcore_frac"] = top / n.clip(lower=1)
    f["topcore_vs_core"] = top / (cm + 1)  # ~1 => top core is (near) a clique
    f["shell_decay"] = sizes.apply(lambda a: np.log10(a[-1] / a[0]) / max(len(a) - 1, 1) if a[0] > 0 else 0)
    f["half_core_frac"] = [
        (s[lv >= (lv[0] + lv[-1]) / 2][0] / s[0]) if len(s) else 0 for s, lv in zip(sizes, levels)
    ]
    f["core2k_frac"] = [
        (s[lv >= 2 * kk][0] / s[0]) if (lv >= 2 * kk).any() else 0.0 for s, lv, kk in zip(sizes, levels, k)
    ]
    feats_eng = STUDENT_FEATURES + list(f.columns)
    X = pd.concat([df[STUDENT_FEATURES].astype(float), f.astype(float)], axis=1)
    return df, X, T, timed_out, cols, feats_eng


# ----------------------------------------------------------------------------- metrics
def evaluate(T, timed_out, pred, sbs_idx):
    rows = np.arange(len(T))
    t_pred = T[rows, pred]
    t_best = T.min(axis=1)
    t_sbs = T[rows, sbs_idx]
    return {
        "acc": np.mean(t_pred <= t_best),  # ties count as correct
        "acc_10pct": np.mean(t_pred <= t_best * 1.1),
        "acc_50pct": np.mean(t_pred <= t_best * 1.5),
        "time_ratio": t_pred.sum() / t_best.sum(),  # student's headline time metric
        "gap_closed": (t_sbs.sum() - t_pred.sum()) / (t_sbs.sum() - t_best.sum()),
        "geo_slowdown": float(np.exp(np.mean(np.log((t_pred + 0.01) / (t_best + 0.01))))),
        "pred_timeouts": int(timed_out[rows, pred].sum()),
    }


# ----------------------------------------------------------------------------- models
def student_weights(T, eps=1e-3):
    """W_TIME_RATIO from features.py: max/min, with min=0 replaced by eps."""
    mx, mn = T.max(axis=1), T.min(axis=1)
    return np.where(mn > 0, np.maximum(mx / np.where(mn > 0, mn, 1), 1.0), np.maximum(mx / eps, 1.0))


def expand_ties(X, T, w):
    """Student's tie expansion: one row per tied-fastest algorithm, weight split evenly."""
    rows, labels, weights = [], [], []
    for i in range(len(T)):
        tied = np.flatnonzero(T[i] <= T[i].min())
        for a in tied:
            rows.append(i); labels.append(a); weights.append(w[i] / len(tied))
    return X.iloc[rows], np.array(labels), np.array(weights)


def fit_predict_classifier(Xtr, Ttr, Xte, *, weighting, **rf_kw):
    if weighting == "student_raw":
        w = student_weights(Ttr)
    elif weighting == "log":
        w = np.log1p(student_weights(Ttr))
    else:
        w = np.ones(len(Ttr))
    Xe, y, we = expand_ties(Xtr, Ttr, w)
    clf = RandomForestClassifier(n_jobs=-1, random_state=0, **rf_kw)
    clf.fit(Xe, y, sample_weight=we)
    return clf.classes_[clf.predict_proba(Xte).argmax(axis=1)]


def fit_predict_xgb_student(Xtr, Ttr, Xte):
    from xgboost import XGBClassifier
    w = np.log1p(student_weights(Ttr))  # xgb_strat_y_K uses log1p
    Xe, y, we = expand_ties(Xtr, Ttr, w)
    classes = np.unique(y)
    remap = {c: i for i, c in enumerate(classes)}
    clf = XGBClassifier(objective="multi:softprob", n_estimators=400, max_depth=10, learning_rate=0.1,
                        subsample=0.8, random_state=42, n_jobs=-1, verbosity=0)
    clf.fit(Xe, np.vectorize(remap.get)(y), sample_weight=we)
    return classes[clf.predict_proba(Xte).argmax(axis=1)]


def logT(T):
    return np.log10(T + EPS)


def fit_predict_rf_reg(Xtr, Ttr, Xte, **kw):
    reg = RandomForestRegressor(n_jobs=-1, random_state=0, **kw)
    reg.fit(Xtr, logT(Ttr))
    return reg.predict(Xte)


def fit_predict_et_reg(Xtr, Ttr, Xte, **kw):
    reg = ExtraTreesRegressor(n_jobs=-1, random_state=0, **kw)
    reg.fit(Xtr, logT(Ttr))
    return reg.predict(Xte)


def fit_predict_lgbm_reg(Xtr, Ttr, Xte, **kw):
    import lightgbm as lgb
    Y = logT(Ttr)
    out = np.zeros((len(Xte), Y.shape[1]))
    params = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=4)
    params.update(kw)
    for j in range(Y.shape[1]):
        out[:, j] = lgb.LGBMRegressor(random_state=j, **params).fit(Xtr, Y[:, j]).predict(Xte)
    return out


# ----------------------------------------------------------------------------- driver
def make_models(feats_student, feats_eng):
    """name -> (feature list, fn(Xtr, Ttr, Xte) -> predicted algorithm index)."""
    argmin = lambda f: (lambda *a: f(*a).argmin(axis=1))
    rf_student = dict(n_estimators=200, max_depth=20, min_samples_split=4, class_weight="balanced")
    return {
        "Student RF (raw weights)": (feats_student, lambda a, b, c: fit_predict_classifier(a, b, c, weighting="student_raw", **rf_student)),
        "Student XGB (log1p weights)": (feats_student, fit_predict_xgb_student),
        "RF classifier, log weights": (feats_student, lambda a, b, c: fit_predict_classifier(a, b, c, weighting="log", n_estimators=500, min_samples_leaf=2)),
        "RF classifier, log w, +feats": (feats_eng, lambda a, b, c: fit_predict_classifier(a, b, c, weighting="log", n_estimators=500, min_samples_leaf=2)),
        "RF regression of log-time": (feats_student, argmin(lambda a, b, c: fit_predict_rf_reg(a, b, c, n_estimators=500, min_samples_leaf=3, max_features=0.5))),
        "RF regression, +feats": (feats_eng, argmin(lambda a, b, c: fit_predict_rf_reg(a, b, c, n_estimators=500, min_samples_leaf=3, max_features=0.5))),
        "ExtraTrees regression, +feats": (feats_eng, argmin(lambda a, b, c: fit_predict_et_reg(a, b, c, n_estimators=500, min_samples_leaf=3, max_features=0.7))),
        "LightGBM regression, +feats": (feats_eng, argmin(fit_predict_lgbm_reg)),
        "Ensemble RF+ET+LGBM reg, +feats": (feats_eng, argmin(lambda a, b, c: (
            fit_predict_rf_reg(a, b, c, n_estimators=500, min_samples_leaf=3, max_features=0.5)
            + fit_predict_et_reg(a, b, c, n_estimators=500, min_samples_leaf=3, max_features=0.7)
            + fit_predict_lgbm_reg(a, b, c)) / 3)),
    }


def run(target, protocol, repeats, only=None):
    df, X, T, timed_out, algos, feats_eng = load(target)
    groups = df["dataset"].to_numpy()
    models = make_models(STUDENT_FEATURES, feats_eng)
    if only:
        models = {k: v for k, v in models.items() if any(o.lower() in k.lower() for o in only)}
    print(f"target={target} protocol={protocol} rows={len(T)} graphs={len(set(groups))} algos={len(algos)}")

    records = []
    for rep in range(repeats):
        if protocol == "group":
            splitter = GroupKFold(n_splits=5, shuffle=True, random_state=rep).split(X, groups=groups)
        else:
            splitter = KFold(n_splits=5, shuffle=True, random_state=rep).split(X)
        preds = {name: np.zeros(len(T), dtype=int) for name in models}
        preds["Single best solver (SBS)"] = np.zeros(len(T), dtype=int)
        sbs_for_row = np.zeros(len(T), dtype=int)
        for tr, te in splitter:
            sbs = int(T[tr].sum(axis=0).argmin())
            sbs_for_row[te] = sbs
            preds["Single best solver (SBS)"][te] = sbs
            for name, (feats, fn) in models.items():
                preds[name][te] = fn(X.iloc[tr][feats], T[tr], X.iloc[te][feats])
        for name, p in preds.items():
            r = evaluate(T, timed_out, p, sbs_for_row)
            r.update(model=name, repeat=rep)
            records.append(r)
        print(f"  repeat {rep} done")

    res = pd.DataFrame(records)
    summary = res.groupby("model").agg(["mean", "std"])
    order = res.groupby("model")["time_ratio"].mean().sort_values().index
    summary = summary.loc[order]
    return res, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["bnb", "total"], default="bnb")
    ap.add_argument("--protocol", choices=["group", "random"], default="group")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    a = ap.parse_args()
    t0 = time.time()
    res, summary = run(a.target, a.protocol, a.repeats, a.only)
    Path(a.out).mkdir(exist_ok=True)
    tag = f"{a.target}_{a.protocol}"
    res.to_csv(Path(a.out) / f"cv_{tag}_raw.csv", index=False)
    summary.to_csv(Path(a.out) / f"cv_{tag}_summary.csv")
    cols = ["acc", "acc_10pct", "acc_50pct", "time_ratio", "gap_closed", "geo_slowdown", "pred_timeouts"]
    show = pd.DataFrame({c: summary[(c, "mean")] for c in cols})
    show[["acc", "acc_10pct", "acc_50pct", "gap_closed"]] *= 100
    pd.set_option("display.width", 200)
    print(show.round(2).to_string())
    print(f"time_ratio std across repeats: {summary[('time_ratio', 'std')].round(3).to_dict()}")
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
