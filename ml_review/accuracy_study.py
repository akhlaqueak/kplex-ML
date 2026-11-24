"""Accuracy study: which method predicts the fastest k-plex algorithm most often?

All methods output a 24-column score matrix (higher = more likely fastest), so
top-1 / top-3 accuracy can be computed the same way for flat, hierarchical,
one-vs-rest and stacked models. Out-of-fold scores are saved, so ensembles and
extra metrics can be computed later without refitting.

Protocol: graph-grouped 5-fold CV x repeats (same folds as the other scripts).
Training rows: trivial instances removed (every algorithm < 5 s total time).
Primary metric: top-1 accuracy (ties count) on non-trivial test instances.

Usage:
  python accuracy_study.py --target total --repeats 3 [--configs name ...]
  python accuracy_study.py --list
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from algo_selection_eval import student_weights
from runtime_selector import EPS, OUT, load_all

warnings.filterwarnings("ignore")
TRIVIAL_THRESH = 5.0
SCORES_DIR = OUT / "accuracy_scores"


# ----------------------------------------------------------------------------- algorithm factors
def fam(a):
    return "UMKP+" if a.startswith("UMKP+") else ("UMKP" if a.startswith("UMKP") else "kPEX")


def ori(a):
    return "oriented" if "-oriented-" in a else "plain"


def br(a):
    return a.split("-")[-1]  # S / R / P / B


FACTORS = {
    "family": fam,
    "orient": ori,
    "branch": br,
    "variant": lambda a: f"{ori(a)}-{br(a)}",       # orientation x branching (8)
    "fam_orient": lambda a: f"{fam(a)}-{ori(a)}",   # family x orientation (6)
}


# ----------------------------------------------------------------------------- base learners
def make_learner(kind, n_classes):
    if kind == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05, verbose=False,
                                  random_seed=0, thread_count=8,
                                  loss_function="MultiClass" if n_classes > 2 else "Logloss")
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                                  n_jobs=8, random_state=0)
    if kind == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=10, learning_rate=0.1, subsample=0.8,
                             random_state=42, n_jobs=-1, verbosity=0)
    if kind == "rf":
        return RandomForestClassifier(500, min_samples_leaf=2, n_jobs=-1, random_state=0)
    raise ValueError(kind)


def fit_proba(kind, X, labels, weights, Xte, classes):
    """Fit on labels; return probabilities aligned to `classes` (classes absent in training -> 0)."""
    present = np.array(sorted(set(labels)))
    out = np.zeros((len(Xte), len(classes)))
    col = {c: i for i, c in enumerate(classes)}
    if len(present) == 1:
        out[:, col[present[0]]] = 1.0
        return out
    y = np.searchsorted(present, labels)
    m = make_learner(kind, len(present))
    m.fit(X, y, sample_weight=weights)
    p = m.predict_proba(Xte)
    learned = np.asarray(getattr(m, "classes_", np.arange(len(present)))).astype(int)
    for j, c in enumerate(present[learned]):
        out[:, col[c]] = p[:, j]
    return out


def expand(Ttr, cands, value_of, tol=0.0, base_w=None):
    """Tie expansion: for each training row, the factor values of the algorithms (within `cands`)
    that are within (1+tol) of the best candidate; weight split evenly."""
    rows, labels, weights = [], [], []
    Tc = Ttr[:, cands]
    for i in range(len(Ttr)):
        tied = cands[Tc[i] <= Tc[i].min() * (1 + tol)]
        vals = sorted({value_of(a) for a in tied})
        w = 1.0 if base_w is None else base_w[i]
        for v in vals:
            rows.append(i); labels.append(v); weights.append(w / len(vals))
    return np.array(rows), np.array(labels), np.array(weights)


# ----------------------------------------------------------------------------- methods
def flat(Xtr, Ttr, Xte, algos, kind="cat", weighting="none", tol=0.0):
    base_w = np.log1p(student_weights(Ttr)) if weighting == "log" else None
    rows, lab, w = expand(Ttr, np.arange(len(algos)), lambda a: a, tol=tol, base_w=base_w)
    return fit_proba(kind, Xtr.iloc[rows], lab, w, Xte, np.arange(len(algos)))


def one_vs_rest(Xtr, Ttr, Xte, algos, kind="lgbm"):
    best = Ttr <= Ttr.min(1, keepdims=True)
    S = np.zeros((len(Xte), len(algos)))
    for j in range(len(algos)):
        S[:, j] = fit_proba(kind, Xtr, best[:, j].astype(int), None, Xte, np.array([0, 1]))[:, 1]
    return S


def hierarchical(Xtr, Ttr, Xte, algos, order, kind="cat", cond="all", combine="product"):
    """order: list of factor names.
    cond='all'    : every node is trained on every row, label = best algorithm *within* that node.
    cond='winner' : node trained only on rows whose overall winner lies inside the node
                    (the student's Scheme 2/3 style).
    combine='product': P(algo) = product of node probabilities along its path.
    combine='greedy' : top-1 = argmax at every stage; remaining algorithms ranked by product."""
    names = np.array(algos)
    overall_best = Ttr <= Ttr.min(1, keepdims=True)
    P = np.ones((len(Xte), len(algos)))
    G = np.ones((len(Xte), len(algos)))

    def node(cands, depth):
        if depth == len(order) or len(cands) == 1:
            return
        f = FACTORS[order[depth]]
        vals = np.array(sorted({f(names[a]) for a in cands}))
        if len(vals) == 1:
            node(cands, depth + 1)
            return
        rows_ok = np.ones(len(Ttr), bool) if cond == "all" else overall_best[:, cands].any(1)
        idx = np.flatnonzero(rows_ok)
        rows, lab, w = expand(Ttr[idx], cands, lambda a: f(names[a]))
        prob = fit_proba(kind, Xtr.iloc[idx[rows]], lab, w, Xte, vals)
        top = prob.argmax(1)
        for vi, v in enumerate(vals):
            sub = np.array([a for a in cands if f(names[a]) == v])
            P[:, sub] *= prob[:, [vi]]
            G[:, sub] *= (top == vi)[:, None]
            node(sub, depth + 1)

    node(np.arange(len(algos)), 0)
    if combine == "greedy":
        return P * G + 1e-9 * P
    return P


def factorized(Xtr, Ttr, Xte, algos, kind="cat", factors=("family", "orient", "branch")):
    names = np.array(algos)
    P = np.ones((len(Xte), len(algos)))
    for fn in factors:
        f = FACTORS[fn]
        vals = np.array(sorted({f(a) for a in names}))
        rows, lab, w = expand(Ttr, np.arange(len(algos)), lambda a: f(names[a]))
        prob = fit_proba(kind, Xtr.iloc[rows], lab, w, Xte, vals)
        for vi, v in enumerate(vals):
            P[:, [i for i, a in enumerate(names) if f(a) == v]] *= prob[:, [vi]]
    return P


def stacked(Xtr, Ttr, Xte, algos, groups_tr, kind="cat"):
    """Add out-of-fold predicted log-runtimes of all 24 algorithms (relative to the row's
    predicted best) as extra features."""
    def reg():
        return RandomForestRegressor(300, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0)
    Y = np.log10(Ttr + EPS)
    oof = np.zeros_like(Y)
    for itr, ite in GroupKFold(5, shuffle=True, random_state=99).split(Xtr, groups=groups_tr):
        oof[ite] = reg().fit(Xtr.iloc[itr], Y[itr]).predict(Xtr.iloc[ite])
    te_pred = reg().fit(Xtr, Y).predict(Xte)
    cols = [f"pred_{a}" for a in algos]

    def add(X, Z):
        return pd.concat([X.reset_index(drop=True),
                          pd.DataFrame(Z - Z.min(1, keepdims=True), columns=cols)], axis=1)
    return flat(add(Xtr, oof), Ttr, add(Xte, te_pred), algos, kind=kind)


# ----------------------------------------------------------------------------- registry
def H(order, **kw):
    return dict(method="hier", order=order, **kw)


CONFIGS = {
    # flat 24-class
    "Flat CatBoost": dict(method="flat", kind="cat", feats="eng"),
    "Flat CatBoost, student feats": dict(method="flat", kind="cat", feats="base"),
    "Flat CatBoost, +heuristic probe": dict(method="flat", kind="cat", feats="probe"),
    "Flat CatBoost, log weights": dict(method="flat", kind="cat", feats="eng", weighting="log"),
    "Flat CatBoost, near-ties 5%": dict(method="flat", kind="cat", feats="eng", tol=0.05),
    "Flat LightGBM": dict(method="flat", kind="lgbm", feats="eng"),
    "Flat XGBoost (student config)": dict(method="flat", kind="xgb", feats="base", weighting="log"),
    "Flat Random forest": dict(method="flat", kind="rf", feats="eng"),
    "One-vs-rest LightGBM": dict(method="ovr", kind="lgbm", feats="eng"),
    # hierarchical (CatBoost at every node)
    "Hier family > variant": H(["family", "variant"], feats="eng"),
    "Hier family > orient > branch": H(["family", "orient", "branch"], feats="eng"),
    "Hier family > orient > branch, greedy": H(["family", "orient", "branch"], feats="eng", combine="greedy"),
    "Hier family > orient > branch, winner-only": H(["family", "orient", "branch"], feats="eng", cond="winner"),
    "Hier family > variant, winner-only, greedy": H(["family", "variant"], feats="eng", cond="winner", combine="greedy"),
    "Hier orient > family > branch": H(["orient", "family", "branch"], feats="eng"),
    "Hier branch > family > orient": H(["branch", "family", "orient"], feats="eng"),
    "Hier family+orient > branch": H(["fam_orient", "branch"], feats="eng"),
    "Factorized family x orient x branch": dict(method="factor", feats="eng"),
    # stacking
    "Stacked: CatBoost + predicted runtimes": dict(method="stack", kind="cat", feats="eng"),
}


def run_config(cfg, Xtr, Ttr, Xte, algos, groups_tr):
    m = cfg["method"]
    if m == "flat":
        return flat(Xtr, Ttr, Xte, algos, kind=cfg["kind"], weighting=cfg.get("weighting", "none"),
                    tol=cfg.get("tol", 0.0))
    if m == "ovr":
        return one_vs_rest(Xtr, Ttr, Xte, algos, kind=cfg["kind"])
    if m == "hier":
        return hierarchical(Xtr, Ttr, Xte, algos, cfg["order"], kind=cfg.get("kind", "cat"),
                            cond=cfg.get("cond", "all"), combine=cfg.get("combine", "product"))
    if m == "factor":
        return factorized(Xtr, Ttr, Xte, algos, kind=cfg.get("kind", "cat"))
    if m == "stack":
        return stacked(Xtr, Ttr, Xte, algos, groups_tr, kind=cfg["kind"])
    raise ValueError(m)


# ----------------------------------------------------------------------------- metrics
def acc_metrics(T, S, mask):
    T, S = T[mask], S[mask]
    order = np.argsort(-S, axis=1, kind="stable")
    best = T.min(1, keepdims=True)
    r = np.arange(len(T))[:, None]

    def hit(k, tol):
        return 100 * (T[r, order[:, :k]] <= best * (1 + tol)).any(1).mean()

    pick = order[:, 0]
    return dict(top1=hit(1, 0), top2=hit(2, 0), top3=hit(3, 0), top1_10pct=hit(1, .1),
                top3_10pct=hit(3, .1), time_ratio=T[np.arange(len(T)), pick].sum() / best.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="total", choices=["total", "bnb"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(CONFIGS))
        return
    cfgs = {k: v for k, v in CONFIGS.items() if not a.configs or k in a.configs}
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    _, _, T_total, _, _, _, _ = load_all("total")
    trivial = T_total.max(1) < TRIVIAL_THRESH
    gdf, X, T, timed_out, algos, feats, _ = load_all(a.target)
    groups = gdf["dataset"].to_numpy()
    print(f"target={a.target}: {len(T)} instances, {trivial.sum()} trivial (not used for training)", flush=True)

    rows = []
    for rep in range(a.repeats):
        for name, cfg in cfgs.items():
            f = SCORES_DIR / f"{a.target}__{name}__rep{rep}.npy"
            if f.exists():
                S = np.load(f)
            else:
                t0 = time.time()
                S = np.zeros(T.shape)
                Xf = X[feats[cfg["feats"]]]
                for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=groups):
                    tr = tr[~trivial[tr]]
                    S[te] = run_config(cfg, Xf.iloc[tr], T[tr], Xf.iloc[te], algos, groups[tr])
                np.save(f, S)
                print(f"  rep {rep}  {time.time() - t0:6.0f}s  {name}", flush=True)
            for label, mask in (("non-trivial", ~trivial), ("all", np.ones(len(T), bool))):
                rows.append(dict(config=name, repeat=rep, eval_on=label, **acc_metrics(T, S, mask)))
        pd.DataFrame(rows).to_csv(OUT / f"accuracy_study_{a.target}_raw.csv", index=False)

    res = pd.DataFrame(rows)
    summ = res.groupby(["eval_on", "config"]).agg(["mean", "std"])
    cols = ["top1", "top2", "top3", "top1_10pct", "top3_10pct", "time_ratio"]
    show = pd.DataFrame({c: summ[(c, "mean")] for c in cols})
    show["top1_sd"] = summ[("top1", "std")]
    show.to_csv(OUT / f"accuracy_study_{a.target}_summary.csv")
    pd.set_option("display.width", 220)
    for ev in ("non-trivial", "all"):
        print(f"\n=== target={a.target}, evaluated on {ev} instances, sorted by top-1")
        print(show.loc[ev].sort_values("top1", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
