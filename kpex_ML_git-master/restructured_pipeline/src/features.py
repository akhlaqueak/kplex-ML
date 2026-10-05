"""Shared feature and preprocessing utilities for the restructured pipeline."""

from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd

y_columns = [
    "kPEX-S-bnb",
    "kPEX-R-bnb",
    "kPEX-P-bnb",
    "kPEX-B-bnb",
    "kPEX-oriented-S-bnb",
    "kPEX-oriented-R-bnb",
    "kPEX-oriented-P-bnb",
    "kPEX-oriented-B-bnb",
    "UMKP-S-bnb",
    "UMKP-R-bnb",
    "UMKP-P-bnb",
    "UMKP-B-bnb",
    "UMKP-oriented-S-bnb",
    "UMKP-oriented-R-bnb",
    "UMKP-oriented-P-bnb",
    "UMKP-oriented-B-bnb",
    "UMKP+-S-bnb",
    "UMKP+-R-bnb",
    "UMKP+-P-bnb",
    "UMKP+-B-bnb",
    "UMKP+-oriented-S-bnb",
    "UMKP+-oriented-R-bnb",
    "UMKP+-oriented-P-bnb",
    "UMKP+-oriented-B-bnb",
]

algorithms = list(y_columns)



feature_cols = [
    "k_value",
    "n",
    "m",
    "d_avg",
    "d_max",
    "core_max",
    "core_min",
    "shell_num",
    "shell_avgSize",
    "shell_maxSize",
    "shell_minSize",
    "shell_stdSize",
]

ALL_ALGOS = list(algorithms)
ALT_RB_ALGOS = [
    "UMKP+-S-bnb",
    "UMKP+-R-bnb",
    "UMKP+-P-bnb",
    "UMKP+-B-bnb",
    "UMKP+-oriented-S-bnb",
    "UMKP+-oriented-R-bnb",
    "UMKP+-oriented-P-bnb",
    "UMKP+-oriented-B-bnb",
]
NO_ALT_RB_ALGOS = [
    "UMKP-S-bnb",
    "UMKP-R-bnb",
    "UMKP-P-bnb",
    "UMKP-B-bnb",
    "UMKP-oriented-S-bnb",
    "UMKP-oriented-R-bnb",
    "UMKP-oriented-P-bnb",
    "UMKP-oriented-B-bnb",
]
DENSE_ALGOS = [
    "kPEX-S-bnb",
    "kPEX-R-bnb",
    "kPEX-P-bnb",
    "kPEX-B-bnb",
    "kPEX-oriented-S-bnb",
    "kPEX-oriented-R-bnb",
    "kPEX-oriented-P-bnb",
    "kPEX-oriented-B-bnb",
]
SPARSE_ALGOS = ALT_RB_ALGOS + NO_ALT_RB_ALGOS
ORIENTED_ALGOS = [algo for algo in ALL_ALGOS if "-oriented-" in algo]
NON_ORIENTED_ALGOS = [algo for algo in ALL_ALGOS if algo not in ORIENTED_ALGOS]
FORWARD_ALGOS = NON_ORIENTED_ALGOS
ROUTER_CLASSB_ALGOS = [algo for algo in ALL_ALGOS if algo.replace("-bnb", "").endswith("-B")]
ROUTER_CLASSS_ALGOS = [algo for algo in ALL_ALGOS if algo.replace("-bnb", "").endswith("-S")]
ROUTER_CLASSR_ALGOS = [algo for algo in ALL_ALGOS if algo.replace("-bnb", "").endswith("-R")]
ROUTER_CLASSP_ALGOS = [algo for algo in ALL_ALGOS if algo.replace("-bnb", "").endswith("-P")]
ROUTER_CLASS1_ALGOS = ROUTER_CLASSB_ALGOS
ROUTER_CLASS2_ALGOS = ROUTER_CLASSS_ALGOS
ROUTER_CLASS3_ALGOS = ROUTER_CLASSR_ALGOS
ROUTER_CLASS4_ALGOS = ROUTER_CLASSP_ALGOS
ROUTER4_TO_ALGOS = {
    "ClassB": ROUTER_CLASSB_ALGOS,
    "ClassS": ROUTER_CLASSS_ALGOS,
    "ClassR": ROUTER_CLASSR_ALGOS,
    "ClassP": ROUTER_CLASSP_ALGOS,
    # Backward-compatible aliases for older reports/artifacts.
    "Class1": ROUTER_CLASSB_ALGOS,
    "Class2": ROUTER_CLASSS_ALGOS,
    "Class3": ROUTER_CLASSR_ALGOS,
    "Class4": ROUTER_CLASSP_ALGOS,
}



W_TIME_RATIO = 1
W_TIME_DIFF_RELATIVE = 2
W_LOG_TIME_RATIO = 4
W_TIME_MIN = 8
W_LOG_TIME_MIN = 16
W_TIME_MIN_X_W_LOG_TIME_RATIO = W_TIME_MIN + W_LOG_TIME_RATIO
W_LOG_TIME_MIN_X_W_LOG_TIME_RATIO = W_LOG_TIME_MIN + W_LOG_TIME_RATIO
W_LOG_TIME_MIN_PLUS_W_LOG_TIME_RATIO = (W_LOG_TIME_MIN + W_LOG_TIME_RATIO) * 2

# k-th-min weighting schemes for Scheme 1 experiments.
KTH_MIN_WEIGHT_EPSILON = 1e-4
W_2MIN_X_LOG_2MIN_RATIO = "2min_x_log_2min_ratio"
W_3MIN_X_LOG_3MIN_RATIO = "3min_x_log_3min_ratio"
W_4MIN_X_LOG_4MIN_RATIO = "4min_x_log_4min_ratio"
W_5MIN_X_LOG_5MIN_RATIO = "5min_x_log_5min_ratio"
W_6MIN_X_LOG_6MIN_RATIO = "6min_x_log_6min_ratio"

KTH_MIN_WEIGHT_SCHEMES = {
    W_2MIN_X_LOG_2MIN_RATIO: 2,
    W_3MIN_X_LOG_3MIN_RATIO: 3,
    W_4MIN_X_LOG_4MIN_RATIO: 4,
    W_5MIN_X_LOG_5MIN_RATIO: 5,
    W_6MIN_X_LOG_6MIN_RATIO: 6,
}


def load_excel_data(
    file_path,
    timeout_replacement=1800,
    y_cols=None,
    sep=",",
    quotechar='"',
    engine="python",
    rename_columns=None,
):
    columns_to_replace = list(y_cols or y_columns)

    df = pd.read_csv(
        file_path,
        sep=sep,
        quotechar=quotechar,
        engine=engine,
    )

    if rename_columns is None:
        rename_columns = {"shell_sizes": "kcore_list", "core_number": "core_max", "k": "k_value"}
    df.rename(columns=rename_columns, inplace=True)

    runtime_cols = [col for col in columns_to_replace if col in df.columns]
    if runtime_cols:
        df[runtime_cols] = df[runtime_cols].replace("x", timeout_replacement)

        mask = (df[runtime_cols] == timeout_replacement).all(axis=1)
        df = df[~mask]

    cat_df = df
    for col in runtime_cols:
        if col in cat_df.columns:
            cat_df[col] = pd.to_numeric(cat_df[col], errors="coerce").astype("float")

    return cat_df


def time_ratio(vals, epsilon):
    max_val = vals.max()
    min_val = vals.min()
    if min_val != 0:
        return max(max_val / min_val, 1.0)
    return max(max_val / epsilon, 1.0)


def time_diff_relative(vals, epsilon):
    max_val = vals.max()
    min_val = vals.min()
    if min_val != 0:
        return (max_val - min_val) / min_val
    return max(max_val / epsilon, 0.0)


def time_min(vals):
    return vals.min()


def kth_min_x_log_kth_min_ratio(vals, k, epsilon):
    """
    weight = kth_min * log(
        max(kth_min / (min + epsilon), 1.0)
    )

    k=2 -> second-smallest runtime
    ...
    k=6 -> sixth-smallest runtime

    Zero/near ties receive weight 0.
    """

    values = pd.to_numeric(
        pd.Series(vals),
        errors="coerce",
    ).to_numpy(dtype=float)

    # ---------------------------------------------------------
    # Validate runtimes
    # ---------------------------------------------------------
    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite runtime encountered while "
            "calculating kth-min sample weight."
        )

    if len(values) < k:
        raise ValueError(
            f"Cannot calculate {k}-th minimum from "
            f"only {len(values)} runtimes."
        )

    sorted_vals = np.sort(values)

    min_val = float(sorted_vals[0])
    kth_min = float(sorted_vals[k - 1])

    # ---------------------------------------------------------
    # Special case:
    # kth minimum itself is zero.
    #
    # There is no positive runtime separation to weight.
    # Avoid 0 * log(0).
    # ---------------------------------------------------------
    if kth_min == 0.0:
        return 0.0

    denominator = min_val + epsilon

    if denominator <= 0:
        raise ValueError(
            f"Invalid denominator: "
            f"min={min_val}, "
            f"epsilon={epsilon}"
        )

    # ---------------------------------------------------------
    # Clamp ratio to >= 1.
    #
    # If kth_min <= min + epsilon, treat it as an
    # epsilon-level near tie and assign zero weight.
    # ---------------------------------------------------------
    ratio = max(
        kth_min / denominator,
        1.0,
    )

    weight = kth_min * np.log(ratio)

    # ---------------------------------------------------------
    # Final safety checks
    # ---------------------------------------------------------
    if not np.isfinite(weight):
        raise ValueError(
            "Non-finite sample weight generated: "
            f"k={k}, "
            f"min={min_val}, "
            f"kth_min={kth_min}, "
            f"epsilon={epsilon}, "
            f"ratio={ratio}, "
            f"weight={weight}"
        )

    if weight < 0:
        raise ValueError(
            "Negative sample weight generated: "
            f"k={k}, "
            f"min={min_val}, "
            f"kth_min={kth_min}, "
            f"epsilon={epsilon}, "
            f"weight={weight}"
        )

    return float(weight)


def compute_weight(row, scheme, epsilon):
    vals = row[y_columns]
    if scheme == W_TIME_RATIO:
        return time_ratio(vals, epsilon)
    if scheme == W_TIME_DIFF_RELATIVE:
        return time_diff_relative(vals, epsilon)
    if scheme == W_LOG_TIME_RATIO:
        return np.log(time_ratio(vals, epsilon))
    if scheme == W_TIME_MIN:
        return time_min(vals)
    if scheme == W_LOG_TIME_MIN:
        return np.log(1 + time_min(vals))
    if scheme == W_TIME_MIN_X_W_LOG_TIME_RATIO:
        return time_min(vals) * np.log(time_ratio(vals, epsilon))
    if scheme == W_LOG_TIME_MIN_X_W_LOG_TIME_RATIO:
        return np.log(1 + time_min(vals)) * np.log(time_ratio(vals, epsilon))
    if scheme == W_LOG_TIME_MIN_PLUS_W_LOG_TIME_RATIO:
        return np.log(1 + time_min(vals)) + np.log(time_ratio(vals, epsilon))
    if scheme in KTH_MIN_WEIGHT_SCHEMES:
        return kth_min_x_log_kth_min_ratio(
            vals, k=KTH_MIN_WEIGHT_SCHEMES[scheme], epsilon=KTH_MIN_WEIGHT_EPSILON # as we have very small runtime difference b/w top1 and top2
        )
    raise ValueError(f"Unknown scheme: {scheme}")


# def assign_weights(df, scheme, epsilon):
#     df["weight"] = df.apply(compute_weight, scheme=scheme, epsilon=epsilon, axis=1)

def assign_weights(df, scheme, epsilon):
    df["weight"] = df.apply(
        compute_weight,
        scheme=scheme,
        epsilon=epsilon,
        axis=1,
    )

    weights = pd.to_numeric(
        df["weight"],
        errors="coerce",
    )

    invalid = (
        ~np.isfinite(weights.to_numpy())
        | weights.lt(0).to_numpy()
    )

    if invalid.any():
        bad = df.loc[
            invalid,
            ["dataset", "k_value", "weight"]
        ].head(20)

        raise ValueError(
            "Invalid sample weights generated.\n"
            f"scheme={scheme!r}, epsilon={epsilon}\n"
            f"invalid_rows={int(invalid.sum())}\n"
            f"{bad.to_string(index=False)}"
        )

    df["weight"] = weights.astype(float)


def export_sorted(df, output_file="sorted_data.csv"):
    df_sorted = df.sort_values(by="weight", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(output_file, index=False)


def parse_kcore_list(s):
    if pd.isna(s):
        return None

    s = s.strip()
    s = re.sub(r"\)\s*\(", "),(", s)
    return ast.literal_eval(s)


def process_kcore_list_to_statistics(df, keep_col=False):
    shell_num = []
    core_min = []
    shell_avgSize = []
    shell_maxSize = []
    shell_minSize = []
    shell_stdSize = []

    df["kcore_list"] = df["kcore_list"].apply(parse_kcore_list)

    for lst in df["kcore_list"]:
        sizes = [size for (_, size) in lst]

        shell_num.append(len(lst))
        core_min.append(lst[0][0])
        shell_avgSize.append(np.mean(sizes))
        shell_maxSize.append(np.max(sizes))
        shell_minSize.append(np.min(sizes))
        shell_stdSize.append(np.std(sizes))

    df["shell_num"] = shell_num
    df["core_min"] = core_min
    df["shell_avgSize"] = shell_avgSize
    df["shell_maxSize"] = shell_maxSize
    df["shell_minSize"] = shell_minSize
    df["shell_stdSize"] = shell_stdSize

    if not keep_col:
        df.drop("kcore_list", axis=1, inplace=True)


def post_process_data(cat_df):
    int_cols = ["n", "m", "d_max", "core_max", "shell_maxSize", "shell_minSize"]

    for col in int_cols:
        if col in cat_df.columns:
            cat_df[col] = pd.to_numeric(cat_df[col], errors="coerce").astype("Int64")

    new_order = [
        "dataset",
        "k_value",
        "n",
        "m",
        "d_avg",
        "d_max",
        "core_max",
        "core_min",
        "shell_num",
        "shell_avgSize",
        "shell_maxSize",
        "shell_minSize",
        "shell_stdSize",
        "weight",
    ] + y_columns

    missing_cols = [col for col in new_order if col not in cat_df.columns]
    if missing_cols:
        print(
            "Warning: Columns missing in DataFrame and will be skipped in reordering: "
            f"{missing_cols}"
        )

    ordered_cols = [col for col in new_order if col in cat_df.columns]
    remaining_cols = [col for col in cat_df.columns if col not in ordered_cols]
    return cat_df[ordered_cols + remaining_cols]

def get_fastest_algorithm(row, algorithms=ALL_ALGOS):
    times = np.array([row[a] for a in algorithms], dtype=float)
    return algorithms[np.argmin(times)]



def assign_scheme4_targets(cat_df):
    """
    Add Scheme 4's two family-specific targets.

    Every original dataset/K row receives BOTH targets.
    """

    result = cat_df.copy()

    result["y_scheme4_oriented"] = result.apply(
        lambda row: get_fastest_algorithm(
            row,
            ORIENTED_ALGOS,
        ),
        axis=1,
    )

    result["y_scheme4_forward"] = result.apply(
        lambda row: get_fastest_algorithm(
            row,
            NON_ORIENTED_ALGOS,
        ),
        axis=1,
    )

    return result




def map_group(algo):
    if algo in DENSE_ALGOS:
        return "Dense"
    return "Sparse"


def map_sparse_subgroup(algo):
    if algo in ALT_RB_ALGOS:
        return "AltRB"
    if algo in NO_ALT_RB_ALGOS:
        return "No-AltRB"
    return None

def map_orientation(algo):
    if "-oriented-" in str(algo):
        return "Oriented"
    return "NonOriented"

def _strip_runtime_suffix(algo):
    algo = str(algo)
    if algo.endswith("-bnb"):
        return algo[:-4]
    return algo

def compute_group_weight(row, epsilon=1e-3):
    sparse_vals = np.array([row[a] for a in SPARSE_ALGOS], dtype=float)
    dense_vals = np.array([row[a] for a in DENSE_ALGOS], dtype=float)

    sparse_best = sparse_vals.min()
    dense_best = dense_vals.min()
    vals = np.array([sparse_best, dense_best])

    min_val = vals.min()
    max_val = vals.max()

    if min_val == 0:
        ratio = max(max_val / epsilon, 1.0)
    else:
        ratio = max(max_val / min_val, 1.0)

    return np.log(1 + min_val) + np.log(ratio)


def compute_sparse_weight(row, epsilon=1e-3):
    alt_best = min(float(row[a]) for a in ALT_RB_ALGOS)
    no_alt_best = min(float(row[a]) for a in NO_ALT_RB_ALGOS)
    vals = np.array([alt_best, no_alt_best], dtype=float)
    min_val = vals.min()
    max_val = vals.max()

    if min_val == 0:
        ratio = max(max_val / epsilon, 1.0)
    else:
        ratio = max(max_val / min_val, 1.0)

    return np.log(1 + min_val) + np.log(ratio)


def map_router4(algo):
    stripped = _strip_runtime_suffix(algo)
    if stripped.endswith("-B"):
        return "ClassB"
    if stripped.endswith("-S"):
        return "ClassS"
    if stripped.endswith("-R"):
        return "ClassR"
    if stripped.endswith("-P"):
        return "ClassP"
    raise ValueError(f"Unsupported algorithm for Scheme 3 router mapping: {algo}")

def compute_orientation_weight(row, epsilon=1e-3):
    oriented_best = min(float(row[a]) for a in ORIENTED_ALGOS)
    non_oriented_best = min(float(row[a]) for a in NON_ORIENTED_ALGOS)
    vals = np.array([oriented_best, non_oriented_best], dtype=float)

    min_val = vals.min()
    max_val = vals.max()

    if min_val == 0:
        ratio = max(max_val / epsilon, 1.0)
    else:
        ratio = max(max_val / min_val, 1.0)

    return np.log(1 + min_val) + np.log(ratio)


def compute_router4_weight(row, epsilon=1e-3):
    vals = np.array(
        [
            min(float(row[a]) for a in ROUTER_CLASSB_ALGOS),
            min(float(row[a]) for a in ROUTER_CLASSS_ALGOS),
            min(float(row[a]) for a in ROUTER_CLASSR_ALGOS),
            min(float(row[a]) for a in ROUTER_CLASSP_ALGOS),
        ],
        dtype=float,
    )

    min_val = vals.min()
    max_val = vals.max()

    if min_val == 0:
        ratio = max(max_val / epsilon, 1.0)
    else:
        ratio = max(max_val / min_val, 1.0)

    return np.log(1 + min_val) + np.log(ratio)


def assign_scheme2_targets_and_weights(cat_df, weight_col="weight"):
    cat_df["y_algo"] = cat_df.apply(lambda r: get_fastest_algorithm(r, ALL_ALGOS), axis=1)
    cat_df["y_group"] = cat_df["y_algo"].apply(map_group)
    cat_df["y_sparse_subgroup"] = cat_df["y_algo"].apply(map_sparse_subgroup)

    cat_df["weight_group"] = cat_df.apply(compute_group_weight, axis=1)
    cat_df["weight_sparse_subgroup"] = cat_df.apply(compute_sparse_weight, axis=1)
    cat_df["weight_altRB"] = np.where(cat_df["y_sparse_subgroup"] == "AltRB", cat_df[weight_col], 0)
    cat_df["weight_noAltRB"] = np.where(
        cat_df["y_sparse_subgroup"] == "No-AltRB",
        cat_df[weight_col],
        0,
    )
    cat_df["weight_dense"] = cat_df[weight_col]
    return cat_df


def assign_scheme3_targets_and_weights(cat_df, weight_col="weight"):
    cat_df["y_algo"] = cat_df.apply(lambda r: get_fastest_algorithm(r, ALL_ALGOS), axis=1)
    cat_df["y_group"] = cat_df["y_algo"].apply(map_group)
    cat_df["y_sparse_subgroup"] = cat_df["y_algo"].apply(map_sparse_subgroup)
    cat_df["y_orientation"] = cat_df["y_algo"].apply(map_orientation)
    cat_df["y_router4"] = cat_df["y_algo"].apply(map_router4)

    cat_df["weight_group"] = cat_df.apply(compute_group_weight, axis=1)
    cat_df["weight_sparse_subgroup"] = cat_df.apply(compute_sparse_weight, axis=1)
    cat_df["weight_orientation"] = cat_df.apply(compute_orientation_weight, axis=1)
    cat_df["weight_router4"] = cat_df.apply(compute_router4_weight, axis=1)
    return cat_df


__all__ = [
    "ALL_ALGOS",
    "ALT_RB_ALGOS",
    "DENSE_ALGOS",
    "NON_ORIENTED_ALGOS",
    "NO_ALT_RB_ALGOS",
    "ORIENTED_ALGOS",
    "FORWARD_ALGOS"
    "ROUTER4_TO_ALGOS",
    "ROUTER_CLASSB_ALGOS",
    "ROUTER_CLASSP_ALGOS",
    "ROUTER_CLASSR_ALGOS",
    "ROUTER_CLASSS_ALGOS",
    "ROUTER_CLASS1_ALGOS",
    "ROUTER_CLASS2_ALGOS",
    "ROUTER_CLASS3_ALGOS",
    "ROUTER_CLASS4_ALGOS",
    "SPARSE_ALGOS",
    "W_LOG_TIME_MIN",
    "W_LOG_TIME_MIN_PLUS_W_LOG_TIME_RATIO",
    "W_LOG_TIME_MIN_X_W_LOG_TIME_RATIO",
    "W_LOG_TIME_RATIO",
    "W_TIME_DIFF_RELATIVE",
    "W_TIME_MIN",
    "W_TIME_MIN_X_W_LOG_TIME_RATIO",
    "W_TIME_RATIO",
    "algorithms",
    "assign_scheme2_targets_and_weights",
    "assign_scheme3_targets_and_weights",
    "assign_weights",
    "compute_group_weight",
    "compute_orientation_weight",
    "compute_router4_weight",
    "compute_sparse_weight",
    "compute_weight",
    "feature_cols",
    "get_fastest_algorithm",
    "load_excel_data",
    "map_group",
    "map_orientation",
    "map_router4",
    "map_sparse_subgroup",
    "parse_kcore_list",
    "post_process_data",
    "process_kcore_list_to_statistics",
    "time_diff_relative",
    "time_min",
    "time_ratio",
    "y_columns",
]
