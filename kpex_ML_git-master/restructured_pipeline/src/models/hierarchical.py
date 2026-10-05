"""Hierarchical model wrappers for Scheme 2 staged classification."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from ..data_splits import (
    algorithms,
    feature_cols,
    prepare_train_test_strat_Y_hier,
    prepare_train_test_strat_Y_K_hier,
)
from ..features import assign_scheme2_targets_and_weights
from ..metrics import (
    evaluate_algorithm_performance,
    evaluate_time_metrics,
    evaluate_time_metrics_with_tol,
    evaluate_with_tolerance,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


HIERARCHICAL_RF_PRESETS = {
    "group_clf": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 4,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "sparse_subgroup_clf": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 4,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "altRB_clf": {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 4,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "noAltRB_clf": {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 4,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "dense_clf": {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 4,
        "class_weight": "balanced",
        "random_state": 42,
    },
}

HIERARCHICAL_XGB_PRESETS = {
    "group_clf": {
        "objective": "binary:logistic",
        "n_estimators": 400,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "use_label_encoder": False,
        "random_state": 42,
    },
    "sparse_subgroup_clf": {
        "objective": "binary:logistic",
        "n_estimators": 400,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "use_label_encoder": False,
        "random_state": 42,
    },
    "altRB_clf": {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 400,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "use_label_encoder": False,
        "random_state": 42,
    },
    "noAltRB_clf": {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 400,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "use_label_encoder": False,
        "random_state": 42,
    },
    "dense_clf": {
        "objective": "multi:softprob",
        "num_class": 4,
        "n_estimators": 400,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "use_label_encoder": False,
        "random_state": 42,
    },
}


def get_fastest_algorithms(row):
    times = [row[alg] for alg in algorithms]
    min_time = min(times)
    return [alg for alg in algorithms if row[alg] == min_time]


class HierarchicalClassifier5:
    def __init__(
        self,
        group_clf,
        sparse_subgroup_clf,
        altRB_clf,
        noAltRB_clf,
        dense_clf,
        le_group=None,
        le_sparse=None,
        le_algo=None,
        tau_group=0.0,
        tau_sparse=0.0,
        tau_algo=0.0,
        fallback_label=None,
    ):
        self.group_clf = group_clf
        self.sparse_subgroup_clf = sparse_subgroup_clf
        self.altRB_clf = altRB_clf
        self.noAltRB_clf = noAltRB_clf
        self.dense_clf = dense_clf

        self.le_group = le_group
        self.le_sparse = le_sparse
        self.le_algo = le_algo

        self.tau_group = tau_group
        self.tau_sparse = tau_sparse
        self.tau_algo = tau_algo
        self.fallback_label = fallback_label

        self.alt_map = None
        self.noalt_map = None
        self.dense_map = None

        self.uses_label_encoding = all(
            encoder is not None for encoder in (self.le_group, self.le_sparse, self.le_algo)
        )
        self.classes_ = getattr(self.le_algo, "classes_", None) if self.uses_label_encoding else None

    def _reindex_labels(self, y_subset):
        unique = np.unique(y_subset)
        forward_map = {g: i for i, g in enumerate(unique)}
        inverse_map = {i: g for g, i in forward_map.items()}
        y_local = np.array([forward_map[g] for g in y_subset])
        return y_local, forward_map, inverse_map

    def _confident(self, probs, tau):
        idx = np.argmax(probs)
        return probs[idx] >= tau, idx

    def fit(
        self,
        X,
        y_group,
        y_sparse_subgroup,
        y_algo,
        weight_group=None,
        weight_sparse=None,
        weight_altRB=None,
        weight_noAltRB=None,
        weight_dense=None,
    ):
        X = np.asarray(X)
        y_group = np.asarray(y_group)
        y_sparse_subgroup = np.asarray(y_sparse_subgroup)
        y_algo = np.asarray(y_algo)

        if weight_group is not None:
            weight_group = np.asarray(weight_group)
        if weight_sparse is not None:
            weight_sparse = np.asarray(weight_sparse)
        if weight_altRB is not None:
            weight_altRB = np.asarray(weight_altRB)
        if weight_noAltRB is not None:
            weight_noAltRB = np.asarray(weight_noAltRB)
        if weight_dense is not None:
            weight_dense = np.asarray(weight_dense)

        if self.uses_label_encoding:
            self.group_clf.fit(X, y_group, sample_weight=weight_group)

            sparse_label = self.le_group.transform(["Sparse"])[0]
            sparse_idx = y_group == sparse_label
            self.sparse_subgroup_clf.fit(
                X[sparse_idx],
                y_sparse_subgroup[sparse_idx],
                sample_weight=None if weight_sparse is None else weight_sparse[sparse_idx],
            )

            alt_label = self.le_sparse.transform(["AltRB"])[0]
            alt_idx = y_sparse_subgroup == alt_label
            y_local, _, inverse_map = self._reindex_labels(y_algo[alt_idx])
            self.alt_map = inverse_map
            self.altRB_clf.fit(
                X[alt_idx],
                y_local,
                sample_weight=None if weight_altRB is None else weight_altRB[alt_idx],
            )

            no_label = self.le_sparse.transform(["No-AltRB"])[0]
            no_idx = y_sparse_subgroup == no_label
            y_local, _, inverse_map = self._reindex_labels(y_algo[no_idx])
            self.noalt_map = inverse_map
            self.noAltRB_clf.fit(
                X[no_idx],
                y_local,
                sample_weight=None if weight_noAltRB is None else weight_noAltRB[no_idx],
            )

            dense_label = self.le_group.transform(["Dense"])[0]
            dense_idx = y_group == dense_label
            y_local, _, inverse_map = self._reindex_labels(y_algo[dense_idx])
            self.dense_map = inverse_map
            self.dense_clf.fit(
                X[dense_idx],
                y_local,
                sample_weight=None if weight_dense is None else weight_dense[dense_idx],
            )

            self.classes_ = self.le_algo.classes_
            return self

        if weight_group is not None:
            self.group_clf.fit(X, y_group, sample_weight=weight_group)
        else:
            self.group_clf.fit(X, y_group)

        sparse_idx = y_group == "Sparse"
        if weight_sparse is not None:
            self.sparse_subgroup_clf.fit(
                X[sparse_idx],
                y_sparse_subgroup[sparse_idx],
                sample_weight=weight_sparse[sparse_idx],
            )
        else:
            self.sparse_subgroup_clf.fit(X[sparse_idx], y_sparse_subgroup[sparse_idx])

        alt_idx = y_sparse_subgroup == "AltRB"
        if weight_altRB is not None:
            self.altRB_clf.fit(X[alt_idx], y_algo[alt_idx], sample_weight=weight_altRB[alt_idx])
        else:
            self.altRB_clf.fit(X[alt_idx], y_algo[alt_idx])

        noalt_idx = y_sparse_subgroup == "No-AltRB"
        if weight_noAltRB is not None:
            self.noAltRB_clf.fit(
                X[noalt_idx],
                y_algo[noalt_idx],
                sample_weight=weight_noAltRB[noalt_idx],
            )
        else:
            self.noAltRB_clf.fit(X[noalt_idx], y_algo[noalt_idx])

        dense_idx = y_group == "Dense"
        if weight_dense is not None:
            self.dense_clf.fit(X[dense_idx], y_algo[dense_idx], sample_weight=weight_dense[dense_idx])
        else:
            self.dense_clf.fit(X[dense_idx], y_algo[dense_idx])

        self.classes_ = np.unique(y_algo)
        return self

    def predict(self, X):
        X = np.asarray(X)

        if self.uses_label_encoding:
            proba = self.predict_proba(X)
            pred_enc = np.argmax(proba, axis=1)
            return self.le_algo.inverse_transform(pred_enc)

        preds = []
        for x in X:
            x = np.asarray(x).reshape(1, -1)

            p_group = self.group_clf.predict_proba(x)[0]
            ok, idx = self._confident(p_group, self.tau_group)
            if not ok:
                preds.append(self.fallback_label)
                continue

            group = self.group_clf.classes_[idx]
            if group == "Dense":
                p_alg = self.dense_clf.predict_proba(x)[0]
                ok, idx = self._confident(p_alg, self.tau_algo)
                preds.append(self.dense_clf.classes_[idx] if ok else self.fallback_label)
                continue

            p_sparse = self.sparse_subgroup_clf.predict_proba(x)[0]
            ok, idx = self._confident(p_sparse, self.tau_sparse)
            if not ok:
                preds.append(self.fallback_label)
                continue

            sparse_group = self.sparse_subgroup_clf.classes_[idx]
            if sparse_group == "AltRB":
                p_alg = self.altRB_clf.predict_proba(x)[0]
                ok, idx = self._confident(p_alg, self.tau_algo)
                preds.append(self.altRB_clf.classes_[idx] if ok else self.fallback_label)
            else:
                p_alg = self.noAltRB_clf.predict_proba(x)[0]
                ok, idx = self._confident(p_alg, self.tau_algo)
                preds.append(self.noAltRB_clf.classes_[idx] if ok else self.fallback_label)

        return np.array(preds)

    def predict_proba(self, X):
        X = np.asarray(X)
        n_samples = len(X)

        if self.uses_label_encoding:
            n_classes = len(self.le_algo.classes_)
            proba = np.zeros((n_samples, n_classes))

            p_group = self.group_clf.predict_proba(X)
            idx_sparse = self.le_group.transform(["Sparse"])[0]
            idx_dense = self.le_group.transform(["Dense"])[0]
            idx_alt = self.le_sparse.transform(["AltRB"])[0]
            idx_no = self.le_sparse.transform(["No-AltRB"])[0]

            for i in range(n_samples):
                p_sparse = p_group[i, idx_sparse]
                p_dense = p_group[i, idx_dense]

                p_dense_leaf = self.dense_clf.predict_proba(X[i].reshape(1, -1))[0]
                for local_class, p_leaf in enumerate(p_dense_leaf):
                    proba[i, self.dense_map[local_class]] += p_dense * p_leaf

                p_sparse_sub = self.sparse_subgroup_clf.predict_proba(X[i].reshape(1, -1))[0]
                p_alt = p_sparse * p_sparse_sub[idx_alt]
                p_no = p_sparse * p_sparse_sub[idx_no]

                p_alt_leaf = self.altRB_clf.predict_proba(X[i].reshape(1, -1))[0]
                for local_class, p_leaf in enumerate(p_alt_leaf):
                    proba[i, self.alt_map[local_class]] += p_alt * p_leaf

                p_no_leaf = self.noAltRB_clf.predict_proba(X[i].reshape(1, -1))[0]
                for local_class, p_leaf in enumerate(p_no_leaf):
                    proba[i, self.noalt_map[local_class]] += p_no * p_leaf

            return proba

        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        p_group = self.group_clf.predict_proba(X)

        for i in range(n_samples):
            p_sparse = p_group[i, list(self.group_clf.classes_).index("Sparse")]
            p_dense = p_group[i, list(self.group_clf.classes_).index("Dense")]

            p_dense_leaf = self.dense_clf.predict_proba(X[i].reshape(1, -1))[0]
            for alg, p_leaf in zip(self.dense_clf.classes_, p_dense_leaf):
                proba[i, class_to_idx[alg]] += p_dense * p_leaf

            p_sparse_sub = self.sparse_subgroup_clf.predict_proba(X[i].reshape(1, -1))[0]
            idx_alt = list(self.sparse_subgroup_clf.classes_).index("AltRB")
            idx_no = list(self.sparse_subgroup_clf.classes_).index("No-AltRB")
            p_alt = p_sparse * p_sparse_sub[idx_alt]
            p_no = p_sparse * p_sparse_sub[idx_no]

            p_alt_leaf = self.altRB_clf.predict_proba(X[i].reshape(1, -1))[0]
            for alg, p_leaf in zip(self.altRB_clf.classes_, p_alt_leaf):
                proba[i, class_to_idx[alg]] += p_alt * p_leaf

            p_no_leaf = self.noAltRB_clf.predict_proba(X[i].reshape(1, -1))[0]
            for alg, p_leaf in zip(self.noAltRB_clf.classes_, p_no_leaf):
                proba[i, class_to_idx[alg]] += p_no * p_leaf

        return proba


def _get_hier_split_function(split_mode="strat_y"):
    if split_mode == "strat_y":
        return prepare_train_test_strat_Y_hier
    if split_mode == "strat_y_K":
        return prepare_train_test_strat_Y_K_hier
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def _ensure_scheme2_columns(cat_df, weight_col="weight"):
    required_cols = {
        "y_algo",
        "y_group",
        "y_sparse_subgroup",
        "weight_group",
        "weight_sparse_subgroup",
        "weight_altRB",
        "weight_noAltRB",
        "weight_dense",
    }
    if required_cols.issubset(cat_df.columns):
        return cat_df

    scheme2_df = cat_df.copy()
    return assign_scheme2_targets_and_weights(scheme2_df, weight_col=weight_col)


def _prepare_X_values(results_df, features, kcore_list=False, for_nn=False):
    if kcore_list:
        return results_df[features + ["kcore_list"]].values
    if for_nn:
        return results_df[features + ["fastest_time"]].values
    return results_df[features].values


def _transform_optional_labels(labels, le):
    labels = pd.Series(labels)
    out = np.full(len(labels), -1, dtype=int)
    mask = labels.notna()
    if mask.any():
        out[mask] = le.transform(labels.loc[mask])
    return out


def _resolve_stage_local_num_class(model_kwargs, y_group, y_sparse_subgroup, y_algo, le_group, le_sparse):
    """Align XGB leaf num_class with the labels present in the training split."""

    resolved = dict(model_kwargs)
    resolved["altRB_clf_kwargs"] = dict(resolved.get("altRB_clf_kwargs", {}))
    resolved["noAltRB_clf_kwargs"] = dict(resolved.get("noAltRB_clf_kwargs", {}))
    resolved["dense_clf_kwargs"] = dict(resolved.get("dense_clf_kwargs", {}))

    dense_label = le_group.transform(["Dense"])[0]
    alt_label = le_sparse.transform(["AltRB"])[0]
    noalt_label = le_sparse.transform(["No-AltRB"])[0]

    alt_mask = y_sparse_subgroup == alt_label
    noalt_mask = y_sparse_subgroup == noalt_label
    dense_mask = y_group == dense_label

    resolved["altRB_clf_kwargs"]["num_class"] = int(len(np.unique(y_algo[alt_mask])))
    resolved["noAltRB_clf_kwargs"]["num_class"] = int(len(np.unique(y_algo[noalt_mask])))
    resolved["dense_clf_kwargs"]["num_class"] = int(len(np.unique(y_algo[dense_mask])))
    return resolved


def _decode_leaf_classes(hclf5, clf, stage_name):
    if not hclf5.uses_label_encoding:
        return list(clf.classes_)

    if stage_name == "stage3":
        mapping = hclf5.alt_map
    elif stage_name == "stage4":
        mapping = hclf5.noalt_map
    elif stage_name == "stage5":
        mapping = hclf5.dense_map
    else:
        raise ValueError(f"Unknown stage_name: {stage_name}")

    return list(hclf5.le_algo.inverse_transform([mapping[i] for i in clf.classes_]))


def _add_stage_probabilities(results_df, clf, X_full):
    p_group = clf.group_clf.predict_proba(X_full)
    if clf.uses_label_encoding:
        group_labels = list(clf.le_group.inverse_transform(clf.group_clf.classes_))
    else:
        group_labels = list(clf.group_clf.classes_)
    for i, cls in enumerate(group_labels):
        results_df[f"prob_stage1_{cls}"] = p_group[:, i]

    sparse_mask = results_df["y_group"] == "Sparse"
    if sparse_mask.any():
        X_sparse = X_full[sparse_mask]
        p_sparse = clf.sparse_subgroup_clf.predict_proba(X_sparse)
        if clf.uses_label_encoding:
            sparse_labels = list(clf.le_sparse.inverse_transform(clf.sparse_subgroup_clf.classes_))
        else:
            sparse_labels = list(clf.sparse_subgroup_clf.classes_)
        for i, cls in enumerate(sparse_labels):
            results_df.loc[sparse_mask, f"prob_stage2_{cls}"] = p_sparse[:, i]

    alt_mask = results_df["y_sparse_subgroup"] == "AltRB"
    if alt_mask.any():
        X_alt = X_full[alt_mask]
        p_alt = clf.altRB_clf.predict_proba(X_alt)
        for i, cls in enumerate(_decode_leaf_classes(clf, clf.altRB_clf, "stage3")):
            results_df.loc[alt_mask, f"prob_stage3_{cls}"] = p_alt[:, i]

    noalt_mask = results_df["y_sparse_subgroup"] == "No-AltRB"
    if noalt_mask.any():
        X_noalt = X_full[noalt_mask]
        p_noalt = clf.noAltRB_clf.predict_proba(X_noalt)
        for i, cls in enumerate(_decode_leaf_classes(clf, clf.noAltRB_clf, "stage4")):
            results_df.loc[noalt_mask, f"prob_stage4_{cls}"] = p_noalt[:, i]

    dense_mask = results_df["y_group"] == "Dense"
    if dense_mask.any():
        X_dense = X_full[dense_mask]
        p_dense = clf.dense_clf.predict_proba(X_dense)
        for i, cls in enumerate(_decode_leaf_classes(clf, clf.dense_clf, "stage5")):
            results_df.loc[dense_mask, f"prob_stage5_{cls}"] = p_dense[:, i]

    return results_df


def prepare_train_test_data_hier(
    cat_df,
    split_mode="strat_y",
    weight_col="weight",
    test_size=0.2,
    random_state=42,
    for_nn=False,
    kcore_list=None,
    tie_tolerance=0.00,
):
    scheme2_df = _ensure_scheme2_columns(cat_df, weight_col=weight_col)
    split_function = _get_hier_split_function(split_mode)
    return split_function(
        scheme2_df,
        weight_col=weight_col,
        test_size=test_size,
        random_state=random_state,
        for_nn=for_nn,
        kcore_list=kcore_list,
        tie_tolerance=tie_tolerance,
    )


def create_label_encoders(y_group=None, y_sparse_subgroup=None, y_algo=None):
    le_group = LabelEncoder()
    le_sparse = LabelEncoder()
    le_algo = LabelEncoder()

    if y_group is not None:
        le_group.fit(pd.Series(y_group).dropna())
    if y_sparse_subgroup is not None:
        le_sparse.fit(pd.Series(y_sparse_subgroup).dropna())
    if y_algo is not None:
        le_algo.fit(pd.Series(y_algo).dropna())

    return le_group, le_sparse, le_algo


def build_hierarchical_random_forest_classifier(
    group_clf_kwargs=None,
    sparse_subgroup_clf_kwargs=None,
    altRB_clf_kwargs=None,
    noAltRB_clf_kwargs=None,
    dense_clf_kwargs=None,
    tau_group=0.0,
    tau_sparse=0.0,
    tau_algo=0.0,
    fallback_label="SAFE_DEFAULT_ALGO",
):
    group_params = dict(HIERARCHICAL_RF_PRESETS["group_clf"])
    group_params.update(group_clf_kwargs or {})

    sparse_params = dict(HIERARCHICAL_RF_PRESETS["sparse_subgroup_clf"])
    sparse_params.update(sparse_subgroup_clf_kwargs or {})

    alt_params = dict(HIERARCHICAL_RF_PRESETS["altRB_clf"])
    alt_params.update(altRB_clf_kwargs or {})

    noalt_params = dict(HIERARCHICAL_RF_PRESETS["noAltRB_clf"])
    noalt_params.update(noAltRB_clf_kwargs or {})

    dense_params = dict(HIERARCHICAL_RF_PRESETS["dense_clf"])
    dense_params.update(dense_clf_kwargs or {})

    return HierarchicalClassifier5(
        group_clf=RandomForestClassifier(**group_params),
        sparse_subgroup_clf=RandomForestClassifier(**sparse_params),
        altRB_clf=RandomForestClassifier(**alt_params),
        noAltRB_clf=RandomForestClassifier(**noalt_params),
        dense_clf=RandomForestClassifier(**dense_params),
        tau_group=tau_group,
        tau_sparse=tau_sparse,
        tau_algo=tau_algo,
        fallback_label=fallback_label,
    )


def build_hierarchical_xgb_classifier(
    group_clf_kwargs=None,
    sparse_subgroup_clf_kwargs=None,
    altRB_clf_kwargs=None,
    noAltRB_clf_kwargs=None,
    dense_clf_kwargs=None,
    le_group=None,
    le_sparse=None,
    le_algo=None,
    tau_group=0.0,
    tau_sparse=0.0,
    tau_algo=0.0,
    fallback_label="SAFE_DEFAULT_ALGO",
):
    if XGBClassifier is None:  # pragma: no cover - optional dependency
        raise ImportError("xgboost is required to build hierarchical XGB models.")

    group_params = dict(HIERARCHICAL_XGB_PRESETS["group_clf"])
    group_params.update(group_clf_kwargs or {})

    sparse_params = dict(HIERARCHICAL_XGB_PRESETS["sparse_subgroup_clf"])
    sparse_params.update(sparse_subgroup_clf_kwargs or {})

    alt_params = dict(HIERARCHICAL_XGB_PRESETS["altRB_clf"])
    alt_params.update(altRB_clf_kwargs or {})

    noalt_params = dict(HIERARCHICAL_XGB_PRESETS["noAltRB_clf"])
    noalt_params.update(noAltRB_clf_kwargs or {})

    dense_params = dict(HIERARCHICAL_XGB_PRESETS["dense_clf"])
    dense_params.update(dense_clf_kwargs or {})

    return HierarchicalClassifier5(
        group_clf=XGBClassifier(**group_params),
        sparse_subgroup_clf=XGBClassifier(**sparse_params),
        altRB_clf=XGBClassifier(**alt_params),
        noAltRB_clf=XGBClassifier(**noalt_params),
        dense_clf=XGBClassifier(**dense_params),
        le_group=le_group if le_group is not None else LabelEncoder(),
        le_sparse=le_sparse if le_sparse is not None else LabelEncoder(),
        le_algo=le_algo if le_algo is not None else LabelEncoder(),
        tau_group=tau_group,
        tau_sparse=tau_sparse,
        tau_algo=tau_algo,
        fallback_label=fallback_label,
    )


def fit_hierarchical_random_forest_classifier(
    cat_df,
    split_mode="strat_y",
    model_kwargs=None,
    split_kwargs=None,
):
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    split_kwargs = {} if split_kwargs is None else dict(split_kwargs)

    split_data_raw = prepare_train_test_data_hier(cat_df, split_mode=split_mode, **split_kwargs)
    (
        X_train,
        X_test,
        y_algo_train,
        y_algo_test,
        y_group_train,
        y_group_test,
        y_sparse_subgroup_train,
        y_sparse_subgroup_test,
        weight_group_train,
        weight_group_test,
        weight_sparse_train,
        weight_sparse_test,
        weight_altRB_train,
        weight_altRB_test,
        weight_noAltRB_train,
        weight_noAltRB_test,
        weight_dense_train,
        weight_dense_test,
    ) = split_data_raw

    clf = build_hierarchical_random_forest_classifier(**model_kwargs)
    clf.fit(
        X_train.values,
        y_group_train.values,
        y_sparse_subgroup_train.values,
        y_algo_train.values,
        weight_group=weight_group_train.values,
        weight_sparse=weight_sparse_train.values,
        weight_altRB=weight_altRB_train.values,
        weight_noAltRB=weight_noAltRB_train.values,
        weight_dense=weight_dense_train.values,
    )

    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_algo_train": y_algo_train,
        "y_algo_test": y_algo_test,
        "y_group_train": y_group_train,
        "y_group_test": y_group_test,
        "y_sparse_subgroup_train": y_sparse_subgroup_train,
        "y_sparse_subgroup_test": y_sparse_subgroup_test,
        "weight_group_train": weight_group_train,
        "weight_group_test": weight_group_test,
        "weight_sparse_train": weight_sparse_train,
        "weight_sparse_test": weight_sparse_test,
        "weight_altRB_train": weight_altRB_train,
        "weight_altRB_test": weight_altRB_test,
        "weight_noAltRB_train": weight_noAltRB_train,
        "weight_noAltRB_test": weight_noAltRB_test,
        "weight_dense_train": weight_dense_train,
        "weight_dense_test": weight_dense_test,
    }
    return clf, split_data


def fit_hierarchical_xgb_classifier(
    cat_df,
    split_mode="strat_y",
    model_kwargs=None,
    split_kwargs=None,
):
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    split_kwargs = {} if split_kwargs is None else dict(split_kwargs)

    split_data_raw = prepare_train_test_data_hier(cat_df, split_mode=split_mode, **split_kwargs)
    (
        X_train,
        X_test,
        y_algo_train,
        y_algo_test,
        y_group_train,
        y_group_test,
        y_sparse_subgroup_train,
        y_sparse_subgroup_test,
        weight_group_train,
        weight_group_test,
        weight_sparse_train,
        weight_sparse_test,
        weight_altRB_train,
        weight_altRB_test,
        weight_noAltRB_train,
        weight_noAltRB_test,
        weight_dense_train,
        weight_dense_test,
    ) = split_data_raw

    le_group, le_sparse, le_algo = create_label_encoders(
        y_group=y_group_train,
        y_sparse_subgroup=y_sparse_subgroup_train,
        y_algo=y_algo_train,
    )

    y_group_train_enc = le_group.transform(y_group_train)
    y_group_test_enc = le_group.transform(y_group_test)
    y_sparse_train_enc = _transform_optional_labels(y_sparse_subgroup_train, le_sparse)
    y_sparse_test_enc = _transform_optional_labels(y_sparse_subgroup_test, le_sparse)
    y_algo_train_enc = le_algo.transform(y_algo_train)
    y_algo_test_enc = le_algo.transform(y_algo_test)

    model_kwargs = _resolve_stage_local_num_class(
        model_kwargs,
        y_group_train_enc,
        y_sparse_train_enc,
        y_algo_train_enc,
        le_group,
        le_sparse,
    )

    clf = build_hierarchical_xgb_classifier(
        le_group=le_group,
        le_sparse=le_sparse,
        le_algo=le_algo,
        **model_kwargs,
    )
    clf.fit(
        X_train.values,
        y_group_train_enc,
        y_sparse_train_enc,
        y_algo_train_enc,
        weight_group=weight_group_train.values,
        weight_sparse=weight_sparse_train.values,
        weight_altRB=weight_altRB_train.values,
        weight_noAltRB=weight_noAltRB_train.values,
        weight_dense=weight_dense_train.values,
    )

    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_algo_train": y_algo_train,
        "y_algo_test": y_algo_test,
        "y_group_train": y_group_train,
        "y_group_test": y_group_test,
        "y_sparse_subgroup_train": y_sparse_subgroup_train,
        "y_sparse_subgroup_test": y_sparse_subgroup_test,
        "y_group_train_enc": y_group_train_enc,
        "y_group_test_enc": y_group_test_enc,
        "y_sparse_train_enc": y_sparse_train_enc,
        "y_sparse_test_enc": y_sparse_test_enc,
        "y_algo_train_enc": y_algo_train_enc,
        "y_algo_test_enc": y_algo_test_enc,
        "le_group": le_group,
        "le_sparse": le_sparse,
        "le_algo": le_algo,
        "weight_group_train": weight_group_train,
        "weight_group_test": weight_group_test,
        "weight_sparse_train": weight_sparse_train,
        "weight_sparse_test": weight_sparse_test,
        "weight_altRB_train": weight_altRB_train,
        "weight_altRB_test": weight_altRB_test,
        "weight_noAltRB_train": weight_noAltRB_train,
        "weight_noAltRB_test": weight_noAltRB_test,
        "weight_dense_train": weight_dense_train,
        "weight_dense_test": weight_dense_test,
    }
    return clf, split_data


def evaluate_hierarchical_stages(results_df, hclf5, X_full, algorithms):
    summary_rows = []
    per_class_reports = {}

    def clean_labels(y_true, y_pred):
        mask = pd.notnull(y_true) & pd.notnull(y_pred)
        return np.asarray(y_true)[mask], np.asarray(y_pred)[mask]

    def evaluate_stage(stage_name, y_true, y_pred, labels, enforce_order=False):
        y_true, y_pred = clean_labels(y_true, y_pred)
        label_order = labels if enforce_order else sorted(list(set(y_true) | set(y_pred)))

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        summary_rows.append(
            {
                "Stage": stage_name,
                "Samples": len(y_true),
                "Accuracy": acc,
                "Avg Precision": prec,
                "Avg Recall": rec,
                "Avg F1": f1,
            }
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=label_order,
            output_dict=True,
            zero_division=0,
        )
        per_class_reports[stage_name] = pd.DataFrame(report).T

        cm = confusion_matrix(y_true, y_pred, labels=label_order)
        disp = ConfusionMatrixDisplay(cm, display_labels=label_order)
        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(stage_name)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.close(fig)

    if hclf5.uses_label_encoding:
        y_true = results_df["y_group"].values
        y_pred = hclf5.le_group.inverse_transform(hclf5.group_clf.predict(X_full))
        evaluate_stage("Stage 1 (Sparse vs Dense)", y_true, y_pred, labels=["Sparse", "Dense"], enforce_order=True)

        mask = results_df["y_group"] == "Sparse"
        if mask.sum() > 0:
            y_true = results_df.loc[mask, "y_sparse_subgroup"].values
            y_pred = hclf5.le_sparse.inverse_transform(hclf5.sparse_subgroup_clf.predict(X_full[mask]))
            evaluate_stage("Stage 2 (AltRB vs No-AltRB)", y_true, y_pred, labels=["AltRB", "No-AltRB"], enforce_order=True)

        mask = results_df["y_sparse_subgroup"] == "AltRB"
        if mask.sum() > 0:
            y_true = results_df.loc[mask, "y_algo"].values
            y_pred_local = hclf5.altRB_clf.predict(X_full[mask])
            y_pred = hclf5.le_algo.inverse_transform(np.array([hclf5.alt_map[i] for i in y_pred_local]))
            evaluate_stage("Stage 3 (AltRB leaf)", y_true, y_pred, labels=sorted(set(y_true)))

        mask = results_df["y_sparse_subgroup"] == "No-AltRB"
        if mask.sum() > 0:
            y_true = results_df.loc[mask, "y_algo"].values
            y_pred_local = hclf5.noAltRB_clf.predict(X_full[mask])
            y_pred = hclf5.le_algo.inverse_transform(np.array([hclf5.noalt_map[i] for i in y_pred_local]))
            evaluate_stage("Stage 4 (No-AltRB leaf)", y_true, y_pred, labels=sorted(set(y_true)))

        mask = results_df["y_group"] == "Dense"
        if mask.sum() > 0:
            y_true = results_df.loc[mask, "y_algo"].values
            y_pred_local = hclf5.dense_clf.predict(X_full[mask])
            y_pred = hclf5.le_algo.inverse_transform(np.array([hclf5.dense_map[i] for i in y_pred_local]))
            evaluate_stage("Stage 5 (Dense leaf)", y_true, y_pred, labels=sorted(set(y_true)))
    else:
        y_true = results_df["y_group"].values
        y_pred = hclf5.group_clf.predict(X_full)
        evaluate_stage("Stage 1 (Sparse vs Dense)", y_true, y_pred, labels=["Sparse", "Dense"], enforce_order=True)

        mask = results_df["y_group"] == "Sparse"
        if mask.sum() > 0:
            y_true = results_df.loc[mask, "y_sparse_subgroup"].values
            y_pred = hclf5.sparse_subgroup_clf.predict(X_full[mask])
            evaluate_stage("Stage 2 (AltRB vs No-AltRB)", y_true, y_pred, labels=["AltRB", "No-AltRB"], enforce_order=True)

        mask = results_df["y_sparse_subgroup"] == "AltRB"
        if mask.sum() > 0:
            y_true = results_df.loc[mask, "y_algo"].values
            y_pred = hclf5.altRB_clf.predict(X_full[mask])
            evaluate_stage("Stage 3 (AltRB leaf)", y_true, y_pred, labels=list(hclf5.altRB_clf.classes_))

        mask = results_df["y_sparse_subgroup"] == "No-AltRB"
        if mask.sum() > 0:
            y_true = results_df.loc[mask, "y_algo"].values
            y_pred = hclf5.noAltRB_clf.predict(X_full[mask])
            evaluate_stage("Stage 4 (No-AltRB leaf)", y_true, y_pred, labels=list(hclf5.noAltRB_clf.classes_))

        mask = results_df["y_group"] == "Dense"
        if mask.sum() > 0:
            y_true = results_df.loc[mask, "y_algo"].values
            y_pred = hclf5.dense_clf.predict(X_full[mask])
            evaluate_stage("Stage 5 (Dense leaf)", y_true, y_pred, labels=list(hclf5.dense_clf.classes_))

    y_true = results_df["y_algo"].values
    y_pred = hclf5.predict(X_full)
    evaluate_stage("Final (End-to-End)", y_true, y_pred, labels=algorithms, enforce_order=True)

    summary_df = pd.DataFrame(summary_rows)
    print("\n==============================")
    print("Stage-wise Summary Metrics")
    print("==============================\n")
    print(summary_df)
    return summary_df, per_class_reports


def run_full_evaluation_hier5(
    cat_df: pd.DataFrame,
    clf,
    features: list,
    algorithms: list,
    le=None,
    drop_indices: list = None,
    epsilon: float = 1e-3,
    tol: float = 0.0,
    save_files: bool = False,
    output_dir: str = "./",
    file_suffix: str = "hierarchical_eval",
    kcore_list: bool = False,
    for_nn: bool = False,
):
    results_df = _ensure_scheme2_columns(cat_df).copy()
    results_df["fastest_algorithms"] = results_df.apply(get_fastest_algorithms, axis=1)

    X_full = _prepare_X_values(results_df, features, kcore_list=kcore_list, for_nn=for_nn)

    print("Predicting final outputs...")
    y_pred = clf.predict(X_full)
    if le is not None and np.asarray(y_pred).dtype.kind in {"i", "u"}:
        results_df["predicted"] = le.inverse_transform(y_pred)
    else:
        results_df["predicted"] = y_pred

    results_df["correct"] = results_df.apply(
        lambda row: row["predicted"] in row["fastest_algorithms"],
        axis=1,
    )

    proba_cols = []
    if hasattr(clf, "predict_proba"):
        print("Adding hierarchical leaf probabilities...")
        probas = clf.predict_proba(X_full)
        for i, cls in enumerate(clf.classes_):
            col = f"prob_{cls}"
            results_df[col] = probas[:, i]
            proba_cols.append(col)
        for cls in algorithms:
            col = f"prob_{cls}"
            if col not in results_df.columns:
                results_df[col] = 0.0
                proba_cols.append(col)

    print("Adding hierarchical stage probabilities...")
    results_df = _add_stage_probabilities(results_df, clf, X_full)

    print("\nClassification performance:")
    if tol > 0:
        out_dict_1 = evaluate_with_tolerance(results_df, algorithms, tol=tol)
    else:
        out_dict_1 = evaluate_algorithm_performance(results_df, algorithms)

    df_time = results_df
    if drop_indices:
        df_time = df_time.drop(index=drop_indices, errors="ignore")

    if tol > 0:
        out_dict_2 = evaluate_time_metrics_with_tol(df_time, algorithms, tol=tol, epsilon=epsilon)
    else:
        out_dict_2 = evaluate_time_metrics(df_time, algorithms, epsilon=epsilon)

    def get_top3(row):
        probs = {alg: row[f"prob_{alg}"] for alg in algorithms}
        return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]

    results_df["top3_predicted"] = results_df.apply(get_top3, axis=1)
    wrong_df = results_df[~results_df["correct"]].copy()

    if save_files:
        os.makedirs(output_dir, exist_ok=True)
        save_cols = (
            ["dataset", "k_value", "fastest_algorithms", "predicted", "correct", "top3_predicted"]
            + algorithms
            + proba_cols
            + [c for c in results_df.columns if c.startswith("prob_stage")]
        )

        full_path = os.path.join(output_dir, f"all_predictions_{file_suffix}.csv")
        results_df[save_cols].to_csv(full_path, index=False)
        print(f"\nSaved full results to: {full_path}")

        wrong_path = os.path.join(output_dir, f"wrong_predictions_{file_suffix}.csv")
        wrong_df[save_cols].to_csv(wrong_path, index=False)
        print(f"Saved wrong predictions to: {wrong_path}")

    return out_dict_1, out_dict_2, results_df, wrong_df


def save_stage_reports(report_obj, file_name, output_dir="./", to_excel=True):
    os.makedirs(output_dir, exist_ok=True)

    if to_excel:
        file_path = os.path.join(output_dir, f"{file_name}.xlsx")
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            if isinstance(report_obj, dict):
                for stage, df in report_obj.items():
                    safe_sheet_name = str(stage)

                    for invalid_char in [":", "\\", "/", "?", "*", "[", "]"]:
                        safe_sheet_name = safe_sheet_name.replace(invalid_char, "-")

                    safe_sheet_name = safe_sheet_name[:31]

                    base_sheet_name = safe_sheet_name
                    counter = 1

                    while safe_sheet_name in writer.book.sheetnames:
                        suffix = f"_{counter}"
                        safe_sheet_name = f"{base_sheet_name[:31 - len(suffix)]}{suffix}"
                        counter += 1

                    df.to_excel(writer, sheet_name=safe_sheet_name)
                    # df.to_excel(writer, sheet_name=stage[:31])
                    # safe_sheet_name = str(stage)

                    # for invalid_char in [":", "\\", "/", "?", "*", "[", "]"]:
                    #     safe_sheet_name = safe_sheet_name.replace(invalid_char, "-")

                    # safe_sheet_name = safe_sheet_name[:31]

                    # df.to_excel(writer, sheet_name=safe_sheet_name)
            elif isinstance(report_obj, pd.DataFrame):
                report_obj.to_excel(writer, sheet_name="Report")
            else:
                raise ValueError("report_obj must be dict or DataFrame")
        print(f"\nSaved stage report to: {file_path}")
        return file_path

    file_path = os.path.join(output_dir, f"{file_name}.csv")
    if isinstance(report_obj, dict):
        pd.concat(report_obj, names=["Stage"]).to_csv(file_path)
    elif isinstance(report_obj, pd.DataFrame):
        report_obj.to_csv(file_path)
    else:
        raise ValueError("report_obj must be dict or DataFrame")
    print(f"\nSaved stage report to: {file_path}")
    return file_path


__all__ = [
    "HIERARCHICAL_RF_PRESETS",
    "HIERARCHICAL_XGB_PRESETS",
    "HierarchicalClassifier5",
    "algorithms",
    "build_hierarchical_random_forest_classifier",
    "build_hierarchical_xgb_classifier",
    "create_label_encoders",
    "evaluate_hierarchical_stages",
    "feature_cols",
    "fit_hierarchical_random_forest_classifier",
    "fit_hierarchical_xgb_classifier",
    "get_fastest_algorithms",
    "prepare_train_test_data_hier",
    "run_full_evaluation_hier5",
    "save_stage_reports",
]
