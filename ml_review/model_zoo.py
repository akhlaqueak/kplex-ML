"""Compare many model families for k-plex algorithm selection.

Three approaches, all evaluated with graph-grouped 5-fold CV (same folds as the
other scripts):
  * classifiers   : predict the winner (tie-expanded rows, log1p(time-ratio) weights
                    where the model accepts sample weights) -> rank by probability
  * regressors    : predict log10 runtime of all 24 algorithms -> rank by prediction
  * pairwise      : one model per algorithm pair predicts which is faster,
                    weighted by |log time difference|; rank by weighted votes

Tree models use the student's 12 features; scale-sensitive models (SVM, kNN,
logistic/ridge, MLP) use log-transformed + standardized engineered features.

Usage: python model_zoo.py [--targets total bnb] [--repeats 3] [--only name ...]
"""

from __future__ import annotations

import argparse
import inspect
import time
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, SVR

from algo_selection_eval import expand_ties, student_weights
from runtime_selector import EPS, OUT, evaluate, load_all

warnings.filterwarnings("ignore")


def slog(X):
    """signed log1p: tames heavy-tailed counts (n, m, degrees) for scale-sensitive models."""
    X = np.asarray(X, dtype=float)
    return np.sign(X) * np.log1p(np.abs(X))


def scaled(est):
    return make_pipeline(FunctionTransformer(slog), StandardScaler(), est)


# ----------------------------------------------------------------------------- classifiers
def _accepts_weight(est):
    final = est.steps[-1][1] if hasattr(est, "steps") else est
    return "sample_weight" in inspect.signature(final.fit).parameters


def run_classifier(make, Xtr, Ttr, Xte):
    Xe, y, w = expand_ties(Xtr, Ttr, np.log1p(student_weights(Ttr)))
    classes = np.unique(y)
    yi = np.searchsorted(classes, y)
    est = make()
    if _accepts_weight(est):
        key = f"{est.steps[-1][0]}__sample_weight" if hasattr(est, "steps") else "sample_weight"
        est.fit(Xe, yi, **{key: w})
    else:
        est.fit(Xe, yi)
    prob = np.zeros((len(Xte), Ttr.shape[1]))
    prob[:, classes[est.classes_]] = est.predict_proba(Xte)
    return -prob


def xgb_clf():
    from xgboost import XGBClassifier
    return XGBClassifier(objective="multi:softprob", n_estimators=400, max_depth=10, learning_rate=0.1,
                         subsample=0.8, random_state=42, n_jobs=-1, verbosity=0)


def lgbm_clf():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8,
                              random_state=0)


def catboost_clf():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=600, depth=6, learning_rate=0.08, loss_function="MultiClass",
                              verbose=False, random_seed=0, thread_count=8)


# ----------------------------------------------------------------------------- regressors (log runtime)
def run_regressor(make, Xtr, Ttr, Xte):
    Y = np.log10(Ttr + EPS)
    return make().fit(Xtr, Y).predict(Xte)


def catboost_reg():
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=800, depth=6, learning_rate=0.05, loss_function="MultiRMSE",
                             verbose=False, random_seed=0, thread_count=8)


# ----------------------------------------------------------------------------- pairwise voting
def run_pairwise(Xtr, Ttr, Xte):
    import lightgbm as lgb
    L = np.log10(Ttr + EPS)
    n_alg = Ttr.shape[1]
    votes = np.zeros((len(Xte), n_alg))
    for a, b in combinations(range(n_alg), 2):
        diff = L[:, a] - L[:, b]
        keep = np.abs(diff) > 1e-9  # ties carry no information
        y = (diff[keep] < 0).astype(int)  # 1 = a faster
        if y.min() == y.max():
            winner = a if y[0] == 1 else b
            votes[:, winner] += 1
            continue
        m = lgb.LGBMClassifier(n_estimators=80, learning_rate=0.1, num_leaves=7, min_child_samples=10,
                               verbose=-1, n_jobs=8, random_state=0)
        m.fit(Xtr[keep], y, sample_weight=np.abs(diff[keep]))
        p = m.predict_proba(Xte)[:, 1]
        votes[:, a] += p
        votes[:, b] += 1 - p
    return -votes


# ----------------------------------------------------------------------------- registry: name -> (kind, features, fn)
MODELS = {
    # classifiers
    "CLF  XGBoost (student config)": ("base", lambda a, b, c: run_classifier(xgb_clf, a, b, c)),
    "CLF  LightGBM": ("base", lambda a, b, c: run_classifier(lgbm_clf, a, b, c)),
    "CLF  CatBoost": ("base", lambda a, b, c: run_classifier(catboost_clf, a, b, c)),
    "CLF  HistGradientBoosting": ("base", lambda a, b, c: run_classifier(
        lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15,
                                               l2_regularization=1.0, random_state=0), a, b, c)),
    "CLF  Random forest": ("base", lambda a, b, c: run_classifier(
        lambda: RandomForestClassifier(500, min_samples_leaf=2, n_jobs=-1, random_state=0), a, b, c)),
    "CLF  Extra trees": ("base", lambda a, b, c: run_classifier(
        lambda: ExtraTreesClassifier(500, min_samples_leaf=2, n_jobs=-1, random_state=0), a, b, c)),
    "CLF  SVM (RBF)": ("eng", lambda a, b, c: run_classifier(
        lambda: scaled(SVC(C=10, gamma="scale", probability=True, random_state=0)), a, b, c)),
    "CLF  k-nearest neighbours": ("eng", lambda a, b, c: run_classifier(
        lambda: scaled(KNeighborsClassifier(n_neighbors=15, weights="distance")), a, b, c)),
    "CLF  Logistic regression": ("eng", lambda a, b, c: run_classifier(
        lambda: scaled(LogisticRegression(C=1.0, max_iter=3000)), a, b, c)),
    "CLF  Neural net (MLP)": ("eng", lambda a, b, c: run_classifier(
        lambda: scaled(MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=600, early_stopping=True,
                                     random_state=0)), a, b, c)),
    # runtime regressors
    "REG  Random forest": ("base", lambda a, b, c: run_regressor(
        lambda: RandomForestRegressor(400, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0), a, b, c)),
    "REG  CatBoost (multi-output)": ("base", lambda a, b, c: run_regressor(catboost_reg, a, b, c)),
    "REG  SVR (RBF)": ("eng", lambda a, b, c: run_regressor(
        lambda: scaled(MultiOutputRegressor(SVR(C=3, epsilon=0.1, gamma="scale"), n_jobs=-1)), a, b, c)),
    "REG  k-nearest neighbours": ("eng", lambda a, b, c: run_regressor(
        lambda: scaled(KNeighborsRegressor(n_neighbors=10, weights="distance")), a, b, c)),
    "REG  Ridge (linear)": ("eng", lambda a, b, c: run_regressor(lambda: scaled(Ridge(alpha=1.0)), a, b, c)),
    "REG  Neural net (MLP)": ("eng", lambda a, b, c: run_regressor(
        lambda: scaled(MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=800, early_stopping=True,
                                    random_state=0)), a, b, c)),
    # pairwise
    "PAIR LightGBM pairwise voting": ("base", run_pairwise),
}


def topk(T, scores, ks=(1, 3)):
    order = np.argsort(scores, axis=1, kind="stable")
    best = T.min(1, keepdims=True)
    r = np.arange(len(T))[:, None]
    out = {}
    for k in ks:
        top = order[:, :k]
        out[f"top{k}"] = 100 * (T[r, top] <= best).any(1).mean()
        out[f"top{k}_10pct"] = 100 * (T[r, top] <= best * 1.1).any(1).mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=["total", "bnb"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    models = {k: v for k, v in MODELS.items() if not a.only or any(o.lower() in k.lower() for o in a.only)}
    rows, timing = [], {}
    for target in a.targets:
        gdf, X, T, timed_out, algos, feats, _ = load_all(target)
        groups = gdf["dataset"].to_numpy()
        for rep in range(a.repeats):
            scores = {m: np.zeros(T.shape) for m in models}
            sbs = np.zeros(len(T), int)
            for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=groups):
                sbs[te] = T[tr].sum(0).argmin()
                for name, (fs, fn) in models.items():
                    t0 = time.time()
                    scores[name][te] = fn(X.iloc[tr][feats[fs]], T[tr], X.iloc[te][feats[fs]])
                    timing[name] = timing.get(name, 0) + time.time() - t0
            for name, s in scores.items():
                pick = s.argmin(1)
                r = evaluate(T, timed_out, pick, sbs)
                r = {k: r[k] for k in ("time_ratio", "gap_closed", "pred_timeouts")}
                r["gap_closed"] *= 100
                rows.append(dict(target=target, model=name, repeat=rep, **topk(T, s), **r))
            print(f"[{target}] repeat {rep} done", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "model_zoo_raw.csv", index=False)

    res = pd.DataFrame(rows)
    summ = res.groupby(["target", "model"]).mean(numeric_only=True).drop(columns="repeat")
    summ["fit_seconds_per_fold"] = [timing[m] / (len(a.targets) * a.repeats * 5) for _, m in summ.index]
    summ.to_csv(OUT / "model_zoo_summary.csv")
    pd.set_option("display.width", 230)
    for t in a.targets:
        print(f"\n=== target={t} (graph-grouped 5-fold CV x {a.repeats}); sorted by top-3")
        print(summ.loc[t].sort_values("top3", ascending=False).round(2).to_string())


if __name__ == "__main__":
    main()
