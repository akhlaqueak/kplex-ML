# Task brief: extract new features to predict the fastest k-plex algorithm

This brief is for Claude Code running on UAB Cheaha (VS Code 1.98.2 Remote Tunnel).
It is self-contained. Read it fully before doing anything.

---

## 1. Background

The project uses machine learning to predict which of **24 maximum k-plex algorithms**
is fastest on a given (graph, k) instance.

- The 24 algorithms come from three choices:
  - **family** ∈ {kPEX, UMKP, UMKP+}
  - **orientation** ∈ {oriented, plain}
  - **branching** ∈ {S, R, P, B}
- They are named like `UMKP+-oriented-B` or `kPEX-S`.
- Time limit: 1,800 s. A value of `x` in the data means the run timed out.
- "Trivial" instances are those where every algorithm finishes in under 5 s on total time.
  There are 1,221 of 4,278 rows. They are excluded from training and from the headline metrics.

### Key files

All paths are relative to the repo root. The local Mac repo is synced to Cheaha with
`sync.sh` (rsync) into `~/kplex-ML/`.

| Path | What it is |
|---|---|
| `kplex_ML_data.xlsx` | Per-instance times for all 24 algos. Columns `<algo>-heur-kp`, `-mkp`, `-heur`, `-ctcp`, `-bnb`, `-total`. **`mkp` is the optimum: never use it as a feature (leakage).** |
| `naman files/kpex_ML_git-main/input_data/ml-data.csv` | Graph features per (dataset, k): n, m, d_avg, d_max, core_number, shell_sizes. Row `soc-youtube,k=20` is corrupt and dropped. |
| `ml_review/runtime_selector.py` | `load_all(target)` → `gdf, X, T, timed_out, algos, feats, probe_cost`. Builds engineered + "probe" features. |
| `ml_review/accuracy_study.py` | Main evaluation: 19 methods, graph-grouped CV, `acc_metrics`, `run_config`, `CONFIGS`. |
| `ml_review/probe_ctcp_test.py` | Small example of adding one feature column and re-evaluating the best model. **Copy this pattern.** |
| `UMKP/Graph.cpp`, `UMKP/KPlex_BB_matrix.h`, `UMKP/makefile` | UMKP solver source. Build with `make`. Run with `./UMKP <graph> <k> [-branching S-Br\|R-Br\|...] [-topCTCP true] [-twoHopG true]`. |
| `kPEX/`, `UMKPEX/`, `Maple-kPEX/`, ... | Other solver sources. Work out which directory and flags produce each of the 24 algorithms before relying on them. |

### Evaluation protocol (do not change; results must be comparable)

- `GroupKFold(5, shuffle=True, random_state=rep)` grouped by `dataset`, for rep = 0, 1, 2.
  All k-values of a graph stay in the same fold.
- Train on non-trivial rows only. Report on non-trivial test rows.
- Metrics:
  - **top1** — the fastest algorithm is predicted; ties count as correct.
  - **top3** — the fastest is in the model's top 3.
  - **top1_10pct** — the pick is within 10% of the fastest.
  - **time_ratio** — Σ time of pick ÷ Σ oracle time.

### Current best result

Model: flat CatBoost classifier, 800 iterations, depth 6, lr 0.05, with tie-expanded training
rows. Config name: `"Flat CatBoost, +heuristic probe"` in `accuracy_study.py`.

Features:
- 18 engineered graph features;
- heuristic probe features: lower bound `lb` = `heur-kp`, the heuristic time capped at 5 s,
  `ub = core_max + k`, `ub − lb`, `lb/ub`, etc.

Results on non-trivial instances:

| target | top1 | top3 | within 10% | time ratio |
|---|---:|---:|---:|---:|
| total | 35.5 | 62.1 | 68.0 | 3.15× |
| bnb   | 40.3 | 67.5 | 63.0 | 3.58× |

### What has already been tried, and should not be repeated

- **Hierarchical classifiers:**
  - family → branching
  - family → orient → branching
  - other orders
  - winner-only, product and greedy combinations
- **Other model approaches:**
  - factorized classifiers
  - stacking on predicted runtimes
  - one-vs-rest
  - LightGBM, XGBoost and random forest
  - per-algorithm runtime regression
  - pairwise voting
  - ensembles
- **The CTCP time as an extra feature:** no accuracy gain.

None of these beat flat CatBoost by more than about 1 point.

**Conclusion:** model choice is saturated. The only real gain came from features that describe
the instance after the heuristic. The next step is better features about the reduced problem
that branch-and-bound actually faces.

---

## 2. Goal

Extract new per-(graph, k) features that describe the problem branch-and-bound faces. Add them
to the best model and measure the change in top1, top3 and within-10% with the protocol above.

### Rules

- **Allowed:** features computable *before* choosing an algorithm. That means graph statistics,
  heuristic and reduction outputs, or short capped probe runs.
- **Probe cost:** record the wall time of any probe, so it can be added to the time ratio.
- **No leakage:** no feature may use `mkp`, any algorithm's final time, or anything produced
  by a full branch-and-bound run.
- **Use the real k:** features must be computed per (graph, k), because the reductions depend on k.
- **Stay on compute nodes:** don't run heavy jobs on a login node. Use SLURM (`sbatch`, array
  jobs) or the HPC Desktop job you're in.

---

## 3. Features to extract

The tiers are in priority order.

### Tier 1 — `--probe` mode in UMKP (highest value; implement first)

Add a command-line option `-probe true` to `UMKP/Graph.cpp`.

- **What it does:** `Graph::search()` runs normally through the heuristic, the degeneracy pass,
  the reductions and the subproblem extraction.
- **What it skips:** every `kplex_solver` branch-and-bound call.
- **What it prints:** one machine-readable line that starts with `PROBE,`, holding the
  dataset name, k and the features below. Then it returns.

Keep the default behaviour unchanged when `-probe` is absent. Build the probe binary under a
new name, such as `UMKP-probe`, so the existing binaries are not overwritten.

| Feature | Location in `UMKP/Graph.cpp` (approximate) |
|---|---|
| `lb` (`kplex.size()` after the heuristic), `max_degree`, heuristic time | after `heuristic_kplex_max_degree(10)` (~line 291) |
| `UB` from `degen(...)`, `UB − lb`, `lb/UB` | ~line 300 |
| `n`, `m`, density after `shrink_graph` + `ego_degen` | ~lines 316–317 |
| `n`, `m`, density after `CTPrune::core_truss_copruning` | ~line 330 |
| Whether the instance is solved by the reductions alone (`n <= kplex.size()`) | ~line 334 |
| Triangle statistics from `tri_cnt` after `oriented_triangle_counting`: total, max, mean, fraction of edges with `tri ≥ lb − 2K` | ~line 362 |
| **Subproblem distribution**, one value per vertex `u` processed in the main loop after `extract_subgraph_and_prune`. Compute statistics of the subgraph vertex count `ids_n` and edge count (`vp.size()`): count, total, max, mean, p50, p90, Σ ids_n², and density mean/min/max. The code already tracks `prune_cnt`, `max_n_prune`, `min_density_prune`, `total_density_prune` (printed ~lines 531–532); extend these. | main loop ~lines 430–490 |
| **Non-edge structure per subproblem.** Missing edges = `ids_n(ids_n−1)/2 − m_sub`. Also the mean non-neighbours per vertex, and the fraction of vertices with ≥ K non-neighbours, aggregated across subproblems. These relate to branching rules (S/R/P/B). | same loop |
| Subproblems eliminated and vertices removed by `subgraph_prune` | ~line 476 |
| Total probe wall time (needed for cost accounting) | end of probe |

**Caveat:** in a real run `lb` improves during the search, which shrinks later subproblems.
The probe uses the heuristic `lb` only. That's fine: it's consistent across instances.

**Orientation features.**
- Inspect how the oriented variants build subproblems: u plus its later neighbours in the
  degeneracy order, versus u plus all neighbours.
- If feasible, compute both subproblem sizes per vertex.
- Report the distributions of both, plus the ratio between them.

This targets the oriented-vs-plain decision.

### Tier 2 — Short trial runs of each family (higher cost, likely high value)

**Trial runs.** For one representative algorithm per family (kPEX, UMKP, UMKP+), run with a
hard time budget of **1 s**. Add a `-budget <sec>` option, or use `timeout`, but a clean
internal stop that still prints counters is better. Record:
- branch-and-bound nodes explored;
- subproblems completed out of the total;
- best size found;
- whether the run finished.

The node counter exists but is commented out (search `nodeCnt` in `Graph.cpp` and in
`KPlex_BB_matrix.h`).

**Root bounds.** For each family, record the upper bound at the root of the first few
(e.g. 20) subproblems. Families differ mainly in their bounding, so `root_bound − lb` per
family directly targets the family choice.

### Tier 3 — Graph statistics (no solver changes)

Compute these once per graph and reuse them for every k. Use Python with `igraph` or
`networkit`, or C++ for big graphs.

- **Degeneracy order:** out-degree distribution (max, p90, mean, Σ out-degree²).
- **Clustering:** global transitivity and average clustering coefficient. Also the transitivity
  of the max core.
- **Truss:** max truss number and the size of the top truss.
- **Two-hop neighbourhoods:** sizes for top-core vertices (mean, max).
- **Degree spread:** coefficient of variation, degree assortativity, and the number of
  connected components of the max core.

---

## 4. Plan of work

Confirm each step with the user before running large jobs.

1. **Explore and build.**
   - Find the graph files on Cheaha; ask the user if they aren't obvious.
   - Find which solver directory and flags correspond to each of the 24 algorithms.
   - Build `UMKP`. Run one small graph with a small k to confirm the output matches
     `kplex_ML_data.xlsx` (heur-kp, mkp).
2. **Implement Tier 1 `-probe`.**
   - Test it on 3–5 instances of different sizes.
   - Check that `lb` equals the xlsx `heur-kp`.
   - Check that the probe time is about `heur + ctcp + extraction` and far below the bnb time.
3. **Run it on all instances.**
   - Write a SLURM array script. Get the instance list from the (dataset, k) pairs in
     `kplex_ML_data.xlsx`.
   - Use one CPU per task, a time limit of about 10 min, and the `short` partition.
   - Write each instance's output into `ml_review/new_features/probe_raw/`.
   - Collect the output into `ml_review/new_features/probe_features.csv` with columns
     `dataset,k,<features...>,probe_time`.
4. **Evaluate.**
   - Write `ml_review/new_feature_eval.py` following the pattern of `probe_ctcp_test.py`.
   - Merge the new CSV on (dataset, k), using `gdf["dataset"]` and `gdf["k_value"]` from
     `load_all`.
   - Fill missing values with −1.
   - Compare, with the best CatBoost config, for both targets (total, bnb) over 3 repeats:
     - (a) current probe features
     - (b) + Tier 1
     - (c) + Tier 1 + Tier 3
     - (d) + Tier 2, if done
   - Report top1, top3, top1_10pct and time_ratio. Also report time_ratio with the probe
     cost added to every instance.
   - Report CatBoost feature importances for the new features.
5. **Do Tier 2, then Tier 3,** in the same way, if time permits and the user agrees.
6. **Report.** Write a table of results and short conclusions to
   `ml_review/results/new_features_report.md`.

---

## 5. Environment notes

- **Python:**
  - Try `module load Anaconda3`, then create an env with `pandas numpy scikit-learn catboost
    lightgbm xgboost openpyxl python-igraph`.
  - Pin nothing unusual.
  - Use `conda-forge` (`-c conda-forge --override-channels`). Don't accept Anaconda
    default-channel terms of service on the user's behalf.
- **Run times:** `accuracy_study.py`-style CatBoost runs take about 15 s per fold.
  A 3-repeat × 2-target comparison of 4 feature sets takes about 30–40 min. Run it inside a
  job, not on the login node.
- **Don't modify the student's files:** leave `naman files/` and the original data files
  read-only.
- **Keep changes isolated:**
  - Don't change existing solver behaviour or output format when `-probe` is not given.
  - Keep solver changes minimal and in clearly marked blocks.
