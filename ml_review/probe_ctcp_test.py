"""Quick test: add the CTCP reduction time (same code in all algorithms) as a probe feature."""
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from accuracy_study import CONFIGS, TRIVIAL_THRESH, acc_metrics, run_config
from runtime_selector import load_all, PROBE_ALGO, EPS

df = pd.read_excel("../kplex_ML_data.xlsx")
_, _, Tt, _, _, _, _ = load_all("total"); trivial = Tt.max(1) < TRIVIAL_THRESH
cfg = CONFIGS["Flat CatBoost, +heuristic probe"]
for target in ("total", "bnb"):
    gdf, X, T, _, algos, feats, _ = load_all(target)
    key = df.set_index(["dataset", "k"])
    idx = list(zip(gdf["dataset"], gdf["k_value"]))
    ct = pd.to_numeric(key.loc[idx, f"{PROBE_ALGO}-ctcp"].to_numpy(), errors="coerce")
    X = X.copy(); X["ctcp_log_time"] = np.log10(np.nan_to_num(ct, nan=1800) + EPS)
    groups = gdf["dataset"].to_numpy()
    for name, cols in (("probe", feats["probe"]), ("probe+ctcp", feats["probe"] + ["ctcp_log_time"])):
        res = []
        for rep in range(3):
            S = np.zeros(T.shape)
            for tr, te in GroupKFold(5, shuffle=True, random_state=rep).split(X, groups=groups):
                tr = tr[~trivial[tr]]
                S[te] = run_config(cfg, X[cols].iloc[tr], T[tr], X[cols].iloc[te], algos, groups[tr])
            res.append(acc_metrics(T, S, ~trivial))
        r = pd.DataFrame(res).mean()
        print(f"{target:5s} {name:11s} top1 {r.top1:.1f} top3 {r.top3:.1f} 10% {r.top1_10pct:.1f} ratio {r.time_ratio:.2f}", flush=True)
