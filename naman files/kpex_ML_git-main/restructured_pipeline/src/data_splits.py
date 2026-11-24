"""Shared train/test split helpers for Scheme 1, Scheme 2, and Scheme 3."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.model_selection import train_test_split

from restructured_pipeline.src.features import algorithms

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

def _resolve_feature_cols(for_nn=False, kcore_list=None):
    if kcore_list:
        return feature_cols + ["kcore_list"]
    if for_nn:
        return feature_cols + ["fastest_time"]
    return list(feature_cols)


def _resolve_split_sizes(n_samples, test_size):
    if isinstance(test_size, float):
        n_test = math.ceil(test_size * n_samples)
    else:
        n_test = int(test_size)
    return n_samples - n_test, n_test


def _select_forced_train_rows(stratify_labels, source_row_indices, test_size):
    forced_mask = pd.Series(False, index=stratify_labels.index)
    forced_records = []
    forced_classes = set()

    while True:
        remaining_labels = stratify_labels[~forced_mask]
        if remaining_labels.empty:
            break

        class_counts = remaining_labels.value_counts(dropna=False)
        rare_classes = class_counts[class_counts < 2]

        if not rare_classes.empty:
            for class_label, class_count in rare_classes.sort_index().items():
                if class_label in forced_classes:
                    continue
                class_mask = (~forced_mask) & (stratify_labels == class_label)
                forced_mask |= class_mask
                forced_classes.add(class_label)
                forced_records.append(
                    {
                        "class_label": class_label,
                        "count": int(class_count),
                        "reason": "combined class has fewer than 2 rows",
                        "row_indices": source_row_indices[class_mask].tolist(),
                    }
                )
            continue

        n_train, n_test = _resolve_split_sizes(len(remaining_labels), test_size)
        max_supported_classes = min(n_train, n_test)
        if len(class_counts) <= max_supported_classes:
            break

        class_counts_df = class_counts.rename_axis("class_label").reset_index(name="count")
        class_counts_df["class_sort_key"] = class_counts_df["class_label"].astype(str)
        class_counts_df = class_counts_df.sort_values(
            ["count", "class_sort_key"],
            ascending=[True, True],
            kind="mergesort",
        )

        class_label = class_counts_df.iloc[0]["class_label"]
        class_count = int(class_counts_df.iloc[0]["count"])
        class_mask = (~forced_mask) & (stratify_labels == class_label)
        forced_mask |= class_mask
        forced_classes.add(class_label)
        forced_records.append(
            {
                "class_label": class_label,
                "count": class_count,
                "reason": (
                    "combined class removed to keep target+K stratification feasible "
                    f"(remaining_classes={len(class_counts)}, n_train={n_train}, n_test={n_test})"
                ),
                "row_indices": source_row_indices[class_mask].tolist(),
            }
        )

    return forced_mask, forced_records


def _train_test_split_k_aware(
    X,
    y,
    *extra_arrays,
    test_size,
    random_state,
    stratify_labels,
    context_label,
):
    positional_index = pd.RangeIndex(len(X))
    normalized_labels = pd.Series(list(stratify_labels), index=positional_index)
    source_row_indices = pd.Series(list(X.index), index=positional_index)

    forced_mask, forced_records = _select_forced_train_rows(
        normalized_labels,
        source_row_indices,
        test_size,
    )

    remaining_positions = positional_index[~forced_mask.to_numpy()]
    forced_positions = positional_index[forced_mask.to_numpy()]

    if len(remaining_positions) == 0:
        raise ValueError(
            f"{context_label}: unable to perform a K-aware split because all rows would need to be forced into training."
        )

    split_inputs = [
        X.iloc[remaining_positions],
        y.iloc[remaining_positions],
        *[array.iloc[remaining_positions] for array in extra_arrays],
    ]
    split_stratify_labels = normalized_labels.iloc[remaining_positions]

    split_result = train_test_split(
        *split_inputs,
        test_size=test_size,
        random_state=random_state,
        stratify=split_stratify_labels,
    )

    split_pairs = []
    for i in range(0, len(split_result), 2):
        split_pairs.append([split_result[i], split_result[i + 1]])

    if len(forced_positions) > 0:
        forced_arrays = [
            X.iloc[forced_positions],
            y.iloc[forced_positions],
            *[array.iloc[forced_positions] for array in extra_arrays],
        ]
        for pair, forced_array in zip(split_pairs, forced_arrays):
            pair[0] = pd.concat([pair[0], forced_array], axis=0)

    if forced_records:
        print(
            f"{context_label}: forced {int(forced_mask.sum())} row(s) into training "
            "before the K-aware split."
        )
        for record in forced_records:
            print(
                f"{context_label}: forced class {record['class_label']} "
                f"({record['count']} row(s)); reason={record['reason']}; "
                f"source_row_indices={record['row_indices']}"
            )
    else:
        print(f"{context_label}: forced 0 row(s) into training before the K-aware split.")

    print(
        f"{context_label}: completed true target+K stratification on "
        f"{split_stratify_labels.nunique(dropna=False)} combined class(es)."
    )
    print(
        f"{context_label}: final train/test sizes = "
        f"{len(split_pairs[0][0])}/{len(split_pairs[0][1])}."
    )

    return tuple(item for pair in split_pairs for item in pair)


def prepare_train_test_strat_Y(
    cat_df,
    weight_col="weight",
    test_size=0.2,
    random_state=42,
    for_nn=False,
    kcore_list=None,
    tie_tolerance=0.00,
):
    local_feature_cols = _resolve_feature_cols(for_nn=for_nn, kcore_list=kcore_list)

    expanded_rows = []
    source_row_indices = []
    num_tied_expansions = 0

    for _, row in cat_df.iterrows():
        values = [row[alg] for alg in algorithms]
        min_val = min(values)
        tied_algs = [alg for alg in algorithms if row[alg] <= min_val * (1 + tie_tolerance)]

        for alg in tied_algs:
            new_row = row[local_feature_cols].copy()
            new_row["target"] = alg
            new_row["weight"] = row[weight_col] / len(tied_algs)
            expanded_rows.append(new_row)
            source_row_indices.append(row.name)

        if len(tied_algs) > 1:
            num_tied_expansions += 1

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df.index = source_row_indices
    print(
        f"Expanded {num_tied_expansions} rows with near ties using tolerance "
        f"{tie_tolerance*100:.1f}%."
    )

    X = expanded_df.drop(["target", "weight"], axis=1)
    y = expanded_df["target"]
    weights = expanded_df["weight"]

    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X,
        y,
        weights,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, weights_train, weights_test


def prepare_train_test_strat_Y_K(
    cat_df,
    weight_col="weight",
    test_size=0.2,
    random_state=42,
    for_nn=False,
    kcore_list=None,
    tie_tolerance=0.00,
):
    local_feature_cols = _resolve_feature_cols(for_nn=for_nn, kcore_list=kcore_list)

    expanded_rows = []
    source_row_indices = []
    num_tied_expansions = 0

    for _, row in cat_df.iterrows():
        values = [row[alg] for alg in algorithms]
        min_val = min(values)
        tied_algs = [alg for alg in algorithms if row[alg] <= min_val * (1 + tie_tolerance)]

        for alg in tied_algs:
            new_row = row[local_feature_cols].copy()
            new_row["target"] = alg
            new_row["weight"] = row[weight_col] / len(tied_algs)
            expanded_rows.append(new_row)
            source_row_indices.append(row.name)

        if len(tied_algs) > 1:
            num_tied_expansions += 1

    expanded_df = pd.DataFrame(expanded_rows)
    expanded_df.index = source_row_indices
    print(
        f"Expanded {num_tied_expansions} rows with near ties using tolerance "
        f"{tie_tolerance*100:.1f}%."
    )

    expanded_df["stratify_var"] = (
        expanded_df["target"].astype(str) + "_k" + expanded_df["k_value"].astype(str)
    )

    X = expanded_df.drop(columns=["target", "weight", "stratify_var"])
    y = expanded_df["target"]
    weights = expanded_df["weight"]

    X_train, X_test, y_train, y_test, weights_train, weights_test = _train_test_split_k_aware(
        X,
        y,
        weights,
        test_size=test_size,
        random_state=random_state,
        stratify_labels=expanded_df["stratify_var"],
        context_label="prepare_train_test_strat_Y_K",
    )

    return X_train, X_test, y_train, y_test, weights_train, weights_test


def prepare_train_test_strat_Y_hier(
    cat_df,
    weight_col="weight",
    test_size=0.2,
    random_state=42,
    for_nn=False,
    kcore_list=None,
    tie_tolerance=0.00,
):
    local_feature_cols = _resolve_feature_cols(for_nn=for_nn, kcore_list=kcore_list)

    X = cat_df[local_feature_cols]
    y_algo = cat_df["y_algo"]
    y_group = cat_df["y_group"]
    y_sparse_subgroup = cat_df["y_sparse_subgroup"]

    weight_group = cat_df["weight_group"]
    weight_sparse = cat_df["weight_sparse_subgroup"]
    weight_altRB = cat_df["weight_altRB"]
    weight_noAltRB = cat_df["weight_noAltRB"]
    weight_dense = cat_df["weight_dense"]

    X_train, X_test, y_algo_train, y_algo_test = train_test_split(
        X,
        y_algo,
        test_size=test_size,
        random_state=random_state,
        stratify=y_algo,
    )

    idx_train = X_train.index
    idx_test = X_test.index

    return (
        X_train,
        X_test,
        y_algo_train,
        y_algo_test,
        y_group.loc[idx_train],
        y_group.loc[idx_test],
        y_sparse_subgroup.loc[idx_train],
        y_sparse_subgroup.loc[idx_test],
        weight_group.loc[idx_train],
        weight_group.loc[idx_test],
        weight_sparse.loc[idx_train],
        weight_sparse.loc[idx_test],
        weight_altRB.loc[idx_train],
        weight_altRB.loc[idx_test],
        weight_noAltRB.loc[idx_train],
        weight_noAltRB.loc[idx_test],
        weight_dense.loc[idx_train],
        weight_dense.loc[idx_test],
    )


def prepare_train_test_strat_Y_K_hier(
    cat_df,
    weight_col="weight",
    test_size=0.2,
    random_state=42,
    for_nn=False,
    kcore_list=None,
    tie_tolerance=0.00,
):
    local_feature_cols = _resolve_feature_cols(for_nn=for_nn, kcore_list=kcore_list)

    X = cat_df[local_feature_cols]
    y_algo = cat_df["y_algo"]
    y_group = cat_df["y_group"]
    y_sparse_subgroup = cat_df["y_sparse_subgroup"]

    weight_group = cat_df["weight_group"]
    weight_sparse = cat_df["weight_sparse_subgroup"]
    weight_altRB = cat_df["weight_altRB"]
    weight_noAltRB = cat_df["weight_noAltRB"]
    weight_dense = cat_df["weight_dense"]

    stratify_var = y_algo.astype(str) + "_k" + cat_df["k_value"].astype(str)

    X_train, X_test, y_algo_train, y_algo_test = _train_test_split_k_aware(
        X,
        y_algo,
        test_size=test_size,
        random_state=random_state,
        stratify_labels=stratify_var,
        context_label="prepare_train_test_strat_Y_K_hier",
    )

    idx_train = X_train.index
    idx_test = X_test.index

    return (
        X_train,
        X_test,
        y_algo_train,
        y_algo_test,
        y_group.loc[idx_train],
        y_group.loc[idx_test],
        y_sparse_subgroup.loc[idx_train],
        y_sparse_subgroup.loc[idx_test],
        weight_group.loc[idx_train],
        weight_group.loc[idx_test],
        weight_sparse.loc[idx_train],
        weight_sparse.loc[idx_test],
        weight_altRB.loc[idx_train],
        weight_altRB.loc[idx_test],
        weight_noAltRB.loc[idx_train],
        weight_noAltRB.loc[idx_test],
        weight_dense.loc[idx_train],
        weight_dense.loc[idx_test],
    )


def prepare_train_test_strat_Y_hier3(
    cat_df,
    weight_col="weight",
    test_size=0.2,
    random_state=42,
    for_nn=False,
    kcore_list=None,
    tie_tolerance=0.00,
):
    local_feature_cols = _resolve_feature_cols(for_nn=for_nn, kcore_list=kcore_list)

    X = cat_df[local_feature_cols]
    y_algo = cat_df["y_algo"]
    y_group = cat_df["y_group"]
    y_sparse_subgroup = cat_df["y_sparse_subgroup"]
    y_orientation = cat_df["y_orientation"]
    y_router4 = cat_df["y_router4"]

    weight_group = cat_df["weight_group"]
    weight_sparse = cat_df["weight_sparse_subgroup"]
    weight_orientation = cat_df["weight_orientation"]
    weight_router4 = cat_df["weight_router4"]

    X_train, X_test, y_algo_train, y_algo_test = train_test_split(
        X,
        y_algo,
        test_size=test_size,
        random_state=random_state,
        stratify=y_algo,
    )

    idx_train = X_train.index
    idx_test = X_test.index

    return (
        X_train,
        X_test,
        y_algo_train,
        y_algo_test,
        y_group.loc[idx_train],
        y_group.loc[idx_test],
        y_sparse_subgroup.loc[idx_train],
        y_sparse_subgroup.loc[idx_test],
        y_orientation.loc[idx_train],
        y_orientation.loc[idx_test],
        y_router4.loc[idx_train],
        y_router4.loc[idx_test],
        weight_group.loc[idx_train],
        weight_group.loc[idx_test],
        weight_sparse.loc[idx_train],
        weight_sparse.loc[idx_test],
        weight_orientation.loc[idx_train],
        weight_orientation.loc[idx_test],
        weight_router4.loc[idx_train],
        weight_router4.loc[idx_test],
    )


def prepare_train_test_strat_Y_K_hier3(
    cat_df,
    weight_col="weight",
    test_size=0.2,
    random_state=42,
    for_nn=False,
    kcore_list=None,
    tie_tolerance=0.00,
):
    local_feature_cols = _resolve_feature_cols(for_nn=for_nn, kcore_list=kcore_list)

    X = cat_df[local_feature_cols]
    y_algo = cat_df["y_algo"]
    y_group = cat_df["y_group"]
    y_sparse_subgroup = cat_df["y_sparse_subgroup"]
    y_orientation = cat_df["y_orientation"]
    y_router4 = cat_df["y_router4"]

    weight_group = cat_df["weight_group"]
    weight_sparse = cat_df["weight_sparse_subgroup"]
    weight_orientation = cat_df["weight_orientation"]
    weight_router4 = cat_df["weight_router4"]

    stratify_var = y_algo.astype(str) + "_k" + cat_df["k_value"].astype(str)

    X_train, X_test, y_algo_train, y_algo_test = _train_test_split_k_aware(
        X,
        y_algo,
        test_size=test_size,
        random_state=random_state,
        stratify_labels=stratify_var,
        context_label="prepare_train_test_strat_Y_K_hier3",
    )

    idx_train = X_train.index
    idx_test = X_test.index

    return (
        X_train,
        X_test,
        y_algo_train,
        y_algo_test,
        y_group.loc[idx_train],
        y_group.loc[idx_test],
        y_sparse_subgroup.loc[idx_train],
        y_sparse_subgroup.loc[idx_test],
        y_orientation.loc[idx_train],
        y_orientation.loc[idx_test],
        y_router4.loc[idx_train],
        y_router4.loc[idx_test],
        weight_group.loc[idx_train],
        weight_group.loc[idx_test],
        weight_sparse.loc[idx_train],
        weight_sparse.loc[idx_test],
        weight_orientation.loc[idx_train],
        weight_orientation.loc[idx_test],
        weight_router4.loc[idx_train],
        weight_router4.loc[idx_test],
    )


__all__ = [
    "algorithms",
    "feature_cols",
    "prepare_train_test_strat_Y",
    "prepare_train_test_strat_Y_K",
    "prepare_train_test_strat_Y_K_hier",
    "prepare_train_test_strat_Y_K_hier3",
    "prepare_train_test_strat_Y_hier",
    "prepare_train_test_strat_Y_hier3",
]
