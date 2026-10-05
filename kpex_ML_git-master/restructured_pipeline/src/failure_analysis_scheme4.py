"""Build Scheme 4 or Scheme 5 family-specific enriched important-failure workbooks.

Uses canonical tol_0 important failures and the raw runtime table. Top-K ratios
follow the legacy mixed-analysis rule: for each selected Top-K, retain/evaluate
the best candidate by runtime and measure it against the 12-algorithm family
optimum. Rows whose 12-algorithm family optimum differs from the true 24-way
global optimum are highlighted yellow; no other highlighting is applied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from restructured_pipeline.src.features import (
    algorithms as ALL_ALGOS,
    load_excel_data,
    ORIENTED_ALGOS,
    NON_ORIENTED_ALGOS,
    B_ALGOS,
    OTHER_ALGOS,
)

TOP_K_VALUES = range(1, 7)
SPLITS = ("strat_y", "strat_y_K")
MIN_RATIO_THRESHOLD = 1.20
DEFAULT_PROJECT_CONFIG = Path("restructured_pipeline/configs/project.yaml")
YELLOW_FILL = PatternFill(fill_type="solid", start_color="FFF2CC", end_color="FFF2CC")


def norm_split(v):
    t = str(v).strip().lower()
    if t in {"y", "strat_y"}:
        return "strat_y"
    if t in {"y and k", "y_k", "y+k", "strat_y_k", "strat_yk"}:
        return "strat_y_K"
    return str(v).strip()


def safe_float(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def load_tol0(path: Path, scope: str, scheme: str = "scheme4") -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None)
    scheme_number = 4 if scheme == "scheme4" else 5
    token = f"scheme_{scheme_number}"
    names = [n for n in sheets if n.lower() == f"{token}_tol_0"]
    if not names:
        names = [n for n in sheets if n.lower().endswith("_tol_0") and token in n.lower()]
    if not names:
        raise ValueError(f"No {scheme} tol_0 sheet found in {path}; sheets={list(sheets)}")
    df = pd.concat([sheets[n] for n in names], ignore_index=True, sort=False)
    df["data_scope"] = scope
    split_col = next((c for c in ("split_mode", "strat_on", "stratified_on") if c in df.columns), None)
    if split_col is None:
        raise ValueError(f"No split metadata column in {path}")
    df["split_mode"] = df[split_col].map(norm_split)
    return df


def active_algorithms_for_family(family: str, scheme: str = "scheme4") -> list[str]:
    if scheme == "scheme4":
        if family == "oriented": algos = list(ORIENTED_ALGOS)
        elif family == "forward": algos = list(NON_ORIENTED_ALGOS)
        else: raise ValueError(f"Unsupported Scheme 4 family: {family!r}")
        expected = 12
    elif scheme == "scheme5":
        if family == "b": algos = list(B_ALGOS)
        elif family == "other": algos = list(OTHER_ALGOS)
        else: raise ValueError(f"Unsupported Scheme 5 family: {family!r}")
        expected = 6 if family == "b" else 18
    else:
        raise ValueError(f"Unsupported scheme: {scheme!r}")
    if len(algos) != expected:
        raise ValueError(f"Expected {expected} algorithms for {scheme}/{family}, found {len(algos)}: {algos}")
    return algos


def build_runtime_lookup(project_config_path: Path) -> pd.DataFrame:
    with open(project_config_path, "r", encoding="utf-8") as handle:
        project = yaml.safe_load(handle)
    data_cfg = project["data"]
    runtime_df = load_excel_data(
        data_cfg["file_path"],
        timeout_replacement=data_cfg["timeout_replacement"],
        y_cols=data_cfg["raw_target_columns"],
        sep=data_cfg["sep"],
        quotechar=data_cfg["quotechar"],
        engine=data_cfg["engine"],
        rename_columns=data_cfg["rename_columns"],
    )
    required = ["dataset", "k_value", *ALL_ALGOS]
    missing = [c for c in required if c not in runtime_df.columns]
    if missing:
        raise ValueError(f"Runtime source is missing columns: {missing}")
    lookup = runtime_df[required].copy()
    lookup["dataset"] = lookup["dataset"].astype(str).str.strip()
    lookup["k_value"] = pd.to_numeric(lookup["k_value"], errors="raise").astype(int)
    if lookup.duplicated(["dataset", "k_value"]).any():
        dup = lookup.loc[lookup.duplicated(["dataset", "k_value"], keep=False), ["dataset", "k_value"]]
        raise ValueError(f"Duplicate runtime keys: {dup.head(20).to_dict(orient='records')}")
    return lookup.set_index(["dataset", "k_value"])


def runtime_row_for(runtime_lookup: pd.DataFrame, dataset, k_value) -> pd.Series:
    key = (str(dataset).strip(), int(pd.to_numeric(k_value, errors="raise")))
    try:
        return runtime_lookup.loc[key]
    except KeyError as exc:
        raise KeyError(f"No runtime row for dataset={key[0]!r}, k_value={key[1]}") from exc


def resolve_fastest(runtime_row: pd.Series, algos: list[str]) -> tuple[str, float]:
    vals = [(safe_float(runtime_row.get(a)), a) for a in algos]
    vals = [(rt, a) for rt, a in vals if rt is not None]
    if not vals:
        raise ValueError("No valid runtimes found for requested algorithm family.")
    rt, algo = min(vals, key=lambda x: (x[0], x[1]))
    return algo, rt


def enrich_row(row, family_algos, runtime_lookup, scheme="scheme4"):
    runtime_row = runtime_row_for(runtime_lookup, row.get("dataset"), row.get("k_value"))
    family_fastest_algo, family_fastest_time = resolve_fastest(runtime_row, family_algos)
    global_fastest_algo, global_fastest_time = resolve_fastest(runtime_row, list(ALL_ALGOS))

    entries = []
    for order, algo in enumerate(family_algos):
        probability = safe_float(row.get(f"prob_{algo}"))
        runtime = safe_float(runtime_row.get(algo))
        ratio = runtime / family_fastest_time if runtime is not None and family_fastest_time else None
        entries.append((algo, probability, runtime, ratio, order))
    entries.sort(key=lambda x: (float("inf") if x[1] is None else -x[1], x[4]))

    rank = next((i for i, e in enumerate(entries, 1) if e[0] == family_fastest_algo), None)
    rec = {
        "dataset": row.get("dataset"),
        "k_value": int(pd.to_numeric(row.get("k_value"), errors="raise")),
        "scheme": "scheme 4" if scheme == "scheme4" else "scheme 5",
        "model": "XGB",
        "split_mode": row.get("split_mode"),
        "data_scope": row.get("data_scope"),
        "fastest_algo": family_fastest_algo,
        "fastest_time": round(family_fastest_time, 4),
        "global_fastest_algo": global_fastest_algo,
        "global_fastest_time": round(global_fastest_time, 4),
        "family_fastest_is_global": family_fastest_algo == global_fastest_algo,
        "family_optimum_vs_global_ratio": round(family_fastest_time / global_fastest_time, 4) if global_fastest_time else pd.NA,
        "fastest_algo_rank": rank,
        "fastest_algo_tuple": f"({family_fastest_time:.4f}, {family_fastest_algo!r}, {rank})",
    }

    for k in TOP_K_VALUES:
        candidates = [e for e in entries[:k] if e[3] is not None]
        best = min(candidates, key=lambda e: (e[3], e[4])) if candidates else None
        rec[f"top{k}_min_time_ratio"] = round(best[3], 4) if best else pd.NA
        rec[f"top{k}_candidate"] = (
            f"({best[2]:.4f}, {best[0]!r}, {best[1]:.4f})"
            if best and best[2] is not None and best[1] is not None else pd.NA
        )
        # Legacy rule: this row is still a failure at Top-K iff its selected
        # Top-K minimum runtime ratio remains above 1.20.
        rec[f"top{k}_still_failure"] = bool(best and best[3] > MIN_RATIO_THRESHOLD)

    rec["all_predictions_sorted_by_prob"] = "[" + ", ".join(
        f"({('None' if e[3] is None else f'{e[3]:.4f}')}, {e[0]!r}, "
        f"{('None' if e[2] is None else f'{e[2]:.4f}')}, "
        f"{('None' if e[1] is None else f'{e[1]:.4f}')})"
        for e in entries
    ) + "]"
    return rec


def build_unique_scope_view(df):
    w = df.copy()
    priority = {"test_data": 0, "entire_data": 1}
    w["_p"] = w["data_scope"].map(priority).fillna(9)
    test_keys = set(zip(
        w.loc[w.data_scope.eq("test_data"), "dataset"].astype(str),
        w.loc[w.data_scope.eq("test_data"), "k_value"],
    ))
    w["failure_origin"] = ["Test" if (str(d), k) in test_keys else "Train" for d, k in zip(w.dataset, w.k_value)]
    return (
        w.sort_values(["dataset", "k_value", "_p"], kind="stable")
        .drop_duplicates(["dataset", "k_value"], keep="first")
        .drop(columns="_p")
        .reset_index(drop=True)
    )


def build_legacy_topk_summary(unique: pd.DataFrame, family: str, split: str) -> pd.DataFrame:
    """Report remaining failures exactly like the old Top-K analysis."""
    rows = []
    initial = len(unique)
    for k in TOP_K_VALUES:
        ratio_col = f"top{k}_min_time_ratio"
        ratios = pd.to_numeric(unique[ratio_col], errors="coerce") if not unique.empty else pd.Series(dtype=float)
        remaining_mask = ratios > MIN_RATIO_THRESHOLD
        remaining = int(remaining_mask.sum())
        recovered = int(initial - remaining)
        rows.append({
            "family": family,
            "split_mode": split,
            "top_k": k,
            "initial_important_failures": initial,
            "remaining_failures_gt_1_2x": remaining,
            "recovered_le_1_2x": recovered,
            "recovery_percent": round(100.0 * recovered / initial, 2) if initial else 0.0,
            "filter_rule": f"top{k}_min_time_ratio > {MIN_RATIO_THRESHOLD:.2f}",
        })
    return pd.DataFrame(rows)


def highlight_family_global_mismatches(output_path: Path) -> int:
    """Highlight only rows where the family optimum is not the global 24-way optimum."""
    wb = load_workbook(output_path)
    highlighted = 0
    for ws in wb.worksheets:
        headers = [c.value for c in ws[1]]
        if "fastest_algo" not in headers or "global_fastest_algo" not in headers:
            continue
        family_col = headers.index("fastest_algo") + 1
        global_col = headers.index("global_fastest_algo") + 1
        for ri in range(2, ws.max_row + 1):
            family_algo = ws.cell(ri, family_col).value
            global_algo = ws.cell(ri, global_col).value
            if family_algo is not None and global_algo is not None and str(family_algo) != str(global_algo):
                for ci in range(1, ws.max_column + 1):
                    ws.cell(ri, ci).fill = YELLOW_FILL
                highlighted += 1
    wb.save(output_path)
    wb.close()
    return highlighted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scheme", choices=["scheme4", "scheme5"], default="scheme4")
    p.add_argument("--family", required=True, choices=["oriented", "forward", "b", "other"])
    p.add_argument("--entire-important-workbook", type=Path, required=True)
    p.add_argument("--test-important-workbook", type=Path, required=True)
    p.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    p.add_argument("--output-path", type=Path, required=True)
    a = p.parse_args()

    entire = load_tol0(a.entire_important_workbook, "entire_data", a.scheme)
    test = load_tol0(a.test_important_workbook, "test_data", a.scheme)
    source = pd.concat([entire, test], ignore_index=True, sort=False)
    algos = active_algorithms_for_family(a.family, a.scheme)
    runtime_lookup = build_runtime_lookup(a.project_config)

    frames = {}
    summary = []
    legacy_summaries = []
    for split in SPLITS:
        entire_sub = entire[entire.split_mode.eq(split)].copy()
        test_sub = test[test.split_mode.eq(split)].copy()
        sub = source[source.split_mode.eq(split)].copy()
        out = pd.DataFrame([enrich_row(r, algos, runtime_lookup, a.scheme) for _, r in sub.iterrows()]) if not sub.empty else pd.DataFrame()
        frames[split] = out
        unique = build_unique_scope_view(out) if not out.empty else out
        frames[f"{split}_unique"] = unique

        summary.append({
            "family": a.family,
            "split_mode": split,
            "entire_important_failures": len(entire_sub),
            "test_important_failures": len(test_sub),
            "all_scope_important_rows": len(out),
            "unique_important_failures": len(unique),
            "unique_test_failures": int(unique.failure_origin.eq("Test").sum()) if not unique.empty else 0,
            "unique_train_failures": int(unique.failure_origin.eq("Train").sum()) if not unique.empty else 0,
            "family_optimum_not_global": int((~unique["family_fastest_is_global"]).sum()) if not unique.empty else 0,
        })
        legacy_summaries.append(build_legacy_topk_summary(unique, a.family, split))

    legacy_topk_summary = pd.concat(legacy_summaries, ignore_index=True) if legacy_summaries else pd.DataFrame()

    a.output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(a.output_path, engine="openpyxl") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)
        legacy_topk_summary.to_excel(writer, sheet_name="topk_failure_summary", index=False)
        for split in SPLITS:
            frames[split].to_excel(writer, sheet_name=split, index=False)
            frames[f"{split}_unique"].to_excel(writer, sheet_name=f"{split}_unique", index=False)

    wb = load_workbook(a.output_path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
    wb.save(a.output_path)
    wb.close()

    highlighted = highlight_family_global_mismatches(a.output_path)
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nLegacy Top-K failure counts (ratio > 1.20):")
    print(legacy_topk_summary.to_string(index=False))
    print(f"Rows highlighted yellow (family optimum != global optimum): {highlighted}")
    print(f"Wrote: {a.output_path}")


if __name__ == "__main__":
    main()
