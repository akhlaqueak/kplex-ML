"""Quick test of risk-aware and hybrid selectors (graph-grouped 5-fold CV)."""
import sys, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from algo_selection_eval import load, evaluate, expand_ties, student_weights, logT, STUDENT_FEATURES

target = sys.argv[1] if len(sys.argv) > 1 else "bnb"
df, X, T, to, algos, feats_eng = load(target)
groups = df["dataset"].to_numpy()
recs = []
for rep in range(3):
    P = {k: np.zeros(len(T), int) for k in ["reg_logmean", "reg_expected_time", "clf", "hybrid_top3", "hybrid_top5"]}
    sbs_row = np.zeros(len(T), int)
    for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=groups):
        sbs_row[te] = T[tr].sum(0).argmin()
        Xtr, Xte = X.iloc[tr][STUDENT_FEATURES], X.iloc[te][STUDENT_FEATURES]
        reg = RandomForestRegressor(500, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0).fit(Xtr, logT(T[tr]))
        per_tree = np.stack([t.predict(Xte.to_numpy()) for t in reg.estimators_])  # trees x rows x algos
        mean_log = per_tree.mean(0)
        exp_time = (10 ** per_tree).mean(0)
        P["reg_logmean"][te] = mean_log.argmin(1)
        P["reg_expected_time"][te] = exp_time.argmin(1)
        Xe, y, w = expand_ties(X.iloc[tr][feats_eng], T[tr], np.log1p(student_weights(T[tr])))
        clf = RandomForestClassifier(500, min_samples_leaf=2, n_jobs=-1, random_state=0).fit(Xe, y, sample_weight=w)
        prob = np.zeros((len(te), T.shape[1])); prob[:, clf.classes_] = clf.predict_proba(X.iloc[te][feats_eng])
        P["clf"][te] = prob.argmax(1)
        for kk in (3, 5):
            top = np.argsort(-prob, 1)[:, :kk]
            P[f"hybrid_top{kk}"][te] = top[np.arange(len(te)), np.take_along_axis(exp_time, top, 1).argmin(1)]
    for k, p in P.items():
        r = evaluate(T, to, p, sbs_row); r["model"] = k; recs.append(r)
res = pd.DataFrame(recs).groupby("model").mean().sort_values("time_ratio")
res[["acc", "acc_10pct", "acc_50pct", "gap_closed"]] *= 100
print(target); print(res.round(2).to_string())
