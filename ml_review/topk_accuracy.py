"""Top-k accuracy of runtime-prediction selectors vs the student's classifiers.

Top-k accuracy = share of instances where a fastest algorithm (ties count) is
among the model's k highest-ranked algorithms. Also reported "within 10%":
some algorithm in the top k is no more than 10% slower than the fastest.
Graph-grouped 5-fold CV x 3 repeats (same folds as the other scripts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold

from algo_selection_eval import expand_ties, student_weights
from runtime_selector import EPS, OUT, load_all

KS = (1, 3, 5)


def classifier_scores(clf, Xte, n_algos):
    prob = np.zeros((len(Xte), n_algos))
    prob[:, clf.classes_] = clf.predict_proba(Xte)
    return -prob  # lower = better, like predicted runtimes


def student_rf(Xtr, Ttr, Xte):
    Xe, y, w = expand_ties(Xtr, Ttr, student_weights(Ttr))  # raw W_TIME_RATIO weights
    clf = RandomForestClassifier(200, max_depth=20, min_samples_split=4, class_weight="balanced",
                                 n_jobs=-1, random_state=42).fit(Xe, y, sample_weight=w)
    return classifier_scores(clf, Xte, Ttr.shape[1])


def student_xgb(Xtr, Ttr, Xte):
    from xgboost import XGBClassifier
    Xe, y, w = expand_ties(Xtr, Ttr, np.log1p(student_weights(Ttr)))
    classes = np.unique(y)
    clf = XGBClassifier(objective="multi:softprob", n_estimators=400, max_depth=10, learning_rate=0.1,
                        subsample=0.8, random_state=42, n_jobs=-1, verbosity=0)
    clf.fit(Xe, np.searchsorted(classes, y), sample_weight=w)
    prob = np.zeros((len(Xte), Ttr.shape[1]))
    prob[:, classes] = clf.predict_proba(Xte)
    return -prob


def runtime_rf(Xtr, Ttr, Xte, rule):
    m = RandomForestRegressor(400, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0)
    m.fit(Xtr, np.log10(Ttr + EPS))
    pt = np.stack([t.predict(Xte.to_numpy()) for t in m.estimators_])
    return pt.mean(0) if rule == "mean_log" else (10.0 ** pt).mean(0)


def topk_metrics(T, scores):
    order = np.argsort(scores, axis=1, kind="stable")
    best = T.min(1, keepdims=True)
    exact = T <= best
    near = T <= best * 1.1
    r = np.arange(len(T))[:, None]
    out = {}
    for k in KS:
        top = order[:, :k]
        out[f"top{k}"] = 100 * exact[r, top].any(1).mean()
        out[f"top{k}_10pct"] = 100 * near[r, top].any(1).mean()
    return out


MODELS = {
    "Runtime RF, base feats (mean log)": ("base", lambda a, b, c: runtime_rf(a, b, c, "mean_log")),
    "Runtime RF, base feats (expected time)": ("base", lambda a, b, c: runtime_rf(a, b, c, "expected_time")),
    "Runtime RF, +probe feats (mean log)": ("probe", lambda a, b, c: runtime_rf(a, b, c, "mean_log")),
    "Student XGB classifier": ("base", student_xgb),
    "Student RF classifier": ("base", student_rf),
}

rows = []
for target in ("total", "bnb"):
    gdf, X, T, timed_out, algos, feats, _ = load_all(target)
    groups = gdf["dataset"].to_numpy()
    for rep in range(3):
        scores = {m: np.zeros(T.shape) for m in MODELS}
        for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=groups):
            for name, (fs, fn) in MODELS.items():
                scores[name][te] = fn(X.iloc[tr][feats[fs]], T[tr], X.iloc[te][feats[fs]])
        for name, s in scores.items():
            rows.append(dict(target=target, model=name, repeat=rep, **topk_metrics(T, s)))
        # reference: rank algorithms by training-fold total time (no features)
        sbs_scores = np.zeros(T.shape)
        for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=groups):
            sbs_scores[te] = T[tr].sum(0)
        rows.append(dict(target=target, model="No model: fixed ranking by total time", repeat=rep,
                         **topk_metrics(T, sbs_scores)))
        print(f"[{target}] repeat {rep} done", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "topk_raw.csv", index=False)
summ = res.groupby(["target", "model"]).mean(numeric_only=True).drop(columns="repeat")
sd = res.groupby(["target", "model"])["top3"].std().rename("top3_sd")
summ = summ.join(sd).sort_values(["target", "top3"], ascending=[False, False])
summ.to_csv(OUT / "topk_summary.csv")
pd.set_option("display.width", 220)
print(summ.round(1).to_string())
