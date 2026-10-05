"""Hierarchical model wrappers for Scheme 3 staged classification."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
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
    prepare_train_test_strat_Y_hier3,
    prepare_train_test_strat_Y_K_hier3,
)
from ..features import (ROUTER4_TO_ALGOS, assign_scheme3_targets_and_weights, algorithms as ACTIVE_ALGORITHMS,
    map_group,
    map_orientation,
    map_router4,
    map_sparse_subgroup,
    assign_scheme3_targets_and_weights)

from ..metrics import (
    evaluate_algorithm_performance,
    evaluate_time_metrics,
    evaluate_time_metrics_with_tol,
    evaluate_with_tolerance,
)
from .hierarchical import save_stage_reports

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


SCHEME3_ROUTER_LABELS = ["ClassB", "ClassS", "ClassR", "ClassP"]
SCHEME3_ORIENTATION_LABELS = ["Oriented", "NonOriented"]


SCHEME3_ROUTER_TO_ALGORITHMS = dict(ROUTER4_TO_ALGOS)

SCHEME3_RF_PRESETS = {
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
    "orientation_clf": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 4,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "router4_clf": {
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 4,
        "class_weight": "balanced",
        "random_state": 42,
    },
}

SCHEME3_XGB_PRESETS = {
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
    "orientation_clf": {
        "objective": "binary:logistic",
        "n_estimators": 400,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "use_label_encoder": False,
        "random_state": 42,
    },
    "router4_clf": {
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


class HierarchicalClassifier3:
    def __init__(
        self,
        group_clf,
        sparse_subgroup_clf,
        router4_clf,
        orientation_clf=None,
        le_group=None,
        le_sparse=None,
        le_orientation=None,
        le_router=None,
        le_algo=None,
        tau_group=0.0,
        tau_sparse=0.0,
        tau_orientation=0.0,
        tau_router=0.0,
        fallback_label=None,
    ):
        self.group_clf = group_clf
        self.sparse_subgroup_clf = sparse_subgroup_clf
        self.orientation_clf = orientation_clf

        # Branch-local Scheme 3 orientation classifiers.
        self.orientation_dense_clf = None
        self.orientation_altRB_clf = None
        self.orientation_noAltRB_clf = None

        self.router4_clf = router4_clf
        self.le_group = le_group
        self.le_sparse = le_sparse
        self.le_orientation = le_orientation
        self.le_router = le_router
        self.le_algo = le_algo
        self.tau_group = tau_group
        self.tau_sparse = tau_sparse
        self.tau_orientation = tau_orientation
        self.tau_router = tau_router
        self.fallback_label = fallback_label

        self.router_map = None

        self.uses_label_encoding = all(
            encoder is not None
            for encoder in (self.le_group, self.le_sparse, self.le_router, self.le_algo)
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

    def _has_orientation_stage(self):
        return getattr(self, "orientation_clf", None) is not None

    def _get_orientation_classifier(self, group_label, sparse_label=None):
        """
        Return the orientation classifier associated with the resolved
        Scheme 3 branch.
        """
        if group_label == "Dense":
            return self.orientation_dense_clf

        if group_label == "Sparse":
            if sparse_label == "AltRB":
                return self.orientation_altRB_clf
            if sparse_label == "No-AltRB":
                return self.orientation_noAltRB_clf

        return None

    def _orientation_branch_masks(self, y_group, y_sparse_subgroup):
        """
        Build mutually exclusive masks for branch-local orientation training.
        """
        if self.uses_label_encoding:
            dense_label = self.le_group.transform(["Dense"])[0]
            sparse_label = self.le_group.transform(["Sparse"])[0]
            alt_label = self.le_sparse.transform(["AltRB"])[0]
            noalt_label = self.le_sparse.transform(["No-AltRB"])[0]

            dense_idx = y_group == dense_label
            alt_idx = (y_group == sparse_label) & (y_sparse_subgroup == alt_label)
            noalt_idx = (y_group == sparse_label) & (y_sparse_subgroup == noalt_label)
        else:
            dense_idx = y_group == "Dense"
            alt_idx = (y_group == "Sparse") & (y_sparse_subgroup == "AltRB")
            noalt_idx = (y_group == "Sparse") & (y_sparse_subgroup == "No-AltRB")

        return {
            "Dense": dense_idx,
            "AltRB": alt_idx,
            "No-AltRB": noalt_idx,
        }    

    def _resolve_final_algorithm(
        self,
        router_label,
        group_label=None,
        sparse_label=None,
        orientation_label="NonOriented",
    ):
        if router_label == "Class4" and not self._has_orientation_stage():
            return "kPEX-B-bnb"

        suffix = {
            "ClassB": "B",
            "ClassS": "S",
            "ClassR": "R",
            "ClassP": "P",
            "Class1": "S",
            "Class2": "R",
            "Class3": "P",
            "Class4": "B",
        }[router_label]

        if group_label == "Dense":
            family = "kPEX"
        elif sparse_label == "AltRB":
            family = "UMKP+"
        elif sparse_label == "No-AltRB":
            family = "UMKP"
        else:
            return self.fallback_label

        if orientation_label == "Oriented":
            return f"{family}-oriented-{suffix}-bnb"
        if orientation_label == "NonOriented":
            return f"{family}-{suffix}-bnb"
        return self.fallback_label

    def fit(
        self,
        X,
        y_group,
        y_sparse_subgroup,
        y_orientation,
        y_router4,
        y_algo,
        weight_group=None,
        weight_sparse=None,
        weight_orientation=None,
        weight_router4=None,
    ):
        X = np.asarray(X)
        y_group = np.asarray(y_group)
        y_sparse_subgroup = np.asarray(y_sparse_subgroup)
        y_orientation = np.asarray(y_orientation)
        y_router4 = np.asarray(y_router4)
        y_algo = np.asarray(y_algo)

        if weight_group is not None:
            weight_group = np.asarray(weight_group)
        if weight_sparse is not None:
            weight_sparse = np.asarray(weight_sparse)
        if weight_orientation is not None:
            weight_orientation = np.asarray(weight_orientation)
        if weight_router4 is not None:
            weight_router4 = np.asarray(weight_router4)

        if self.uses_label_encoding:
            self.group_clf.fit(X, y_group, sample_weight=weight_group)

            sparse_label = self.le_group.transform(["Sparse"])[0]
            sparse_idx = y_group == sparse_label
            self.sparse_subgroup_clf.fit(
                X[sparse_idx],
                y_sparse_subgroup[sparse_idx],
                sample_weight=None if weight_sparse is None else weight_sparse[sparse_idx],
            )

            if self._has_orientation_stage():
                orientation_masks = self._orientation_branch_masks(
                    y_group,
                    y_sparse_subgroup,
                )

                self.orientation_dense_clf = clone(self.orientation_clf)
                self.orientation_altRB_clf = clone(self.orientation_clf)
                self.orientation_noAltRB_clf = clone(self.orientation_clf)

                branch_classifiers = {
                    "Dense": self.orientation_dense_clf,
                    "AltRB": self.orientation_altRB_clf,
                    "No-AltRB": self.orientation_noAltRB_clf,
                }

                for branch_name, branch_clf in branch_classifiers.items():
                    branch_idx = orientation_masks[branch_name]

                    if not np.any(branch_idx):
                        raise ValueError(
                            f"Scheme 3 orientation branch '{branch_name}' "
                            "contains no training rows."
                        )

                    branch_targets = y_orientation[branch_idx]

                    if len(np.unique(branch_targets)) < 2:
                        raise ValueError(
                            f"Scheme 3 orientation branch '{branch_name}' "
                            "does not contain both orientation classes."
                        )

                    branch_clf.fit(
                        X[branch_idx],
                        branch_targets,
                        sample_weight=(
                            None
                            if weight_orientation is None
                            else weight_orientation[branch_idx]
                        ),
                    )

            y_router_local, _, inverse_map = self._reindex_labels(y_router4)
            self.router_map = inverse_map
            self.router4_clf.fit(
                X,
                y_router_local,
                sample_weight=weight_router4,
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

        if self._has_orientation_stage():
            orientation_masks = self._orientation_branch_masks(
                y_group,
                y_sparse_subgroup,
            )

            self.orientation_dense_clf = clone(self.orientation_clf)
            self.orientation_altRB_clf = clone(self.orientation_clf)
            self.orientation_noAltRB_clf = clone(self.orientation_clf)

            branch_classifiers = {
                "Dense": self.orientation_dense_clf,
                "AltRB": self.orientation_altRB_clf,
                "No-AltRB": self.orientation_noAltRB_clf,
            }

            for branch_name, branch_clf in branch_classifiers.items():
                branch_idx = orientation_masks[branch_name]

                if not np.any(branch_idx):
                    raise ValueError(
                        f"Scheme 3 orientation branch '{branch_name}' "
                        "contains no training rows."
                    )

                branch_targets = y_orientation[branch_idx]

                if len(np.unique(branch_targets)) < 2:
                    raise ValueError(
                        f"Scheme 3 orientation branch '{branch_name}' "
                        "does not contain both orientation classes."
                    )

                if weight_orientation is not None:
                    branch_clf.fit(
                        X[branch_idx],
                        branch_targets,
                        sample_weight=weight_orientation[branch_idx],
                    )
                else:
                    branch_clf.fit(
                        X[branch_idx],
                        branch_targets,
                    )

        if weight_router4 is not None:
            self.router4_clf.fit(X, y_router4, sample_weight=weight_router4)
        else:
            self.router4_clf.fit(X, y_router4)

        self.classes_ = np.unique(y_algo)
        return self

    def _get_stage_class_maps(self):
        if self.uses_label_encoding:
            group_labels = list(self.le_group.inverse_transform(self.group_clf.classes_))
            sparse_labels = list(self.le_sparse.inverse_transform(self.sparse_subgroup_clf.classes_))
            orientation_labels = (
                list(self.le_orientation.classes_)
                if self._has_orientation_stage() and self.le_orientation is not None
                else []
            )
            router_labels = list(
                self.le_router.inverse_transform([self.router_map[i] for i in self.router4_clf.classes_])
            )
        else:
            group_labels = list(self.group_clf.classes_)
            sparse_labels = list(self.sparse_subgroup_clf.classes_)
            orientation_labels = (
                list(SCHEME3_ORIENTATION_LABELS)
                if self._has_orientation_stage()
                else []
            )
            router_labels = list(self.router4_clf.classes_)

        return (
            {label: i for i, label in enumerate(group_labels)},
            {label: i for i, label in enumerate(sparse_labels)},
            {label: i for i, label in enumerate(orientation_labels)},
            {label: i for i, label in enumerate(router_labels)},
        )

    def predict(self, X):
        X = np.asarray(X)

        if self.uses_label_encoding:
            proba = self.predict_proba(X)
            pred_enc = np.argmax(proba, axis=1)
            return self.le_algo.inverse_transform(pred_enc)

        preds = []
        group_idx_map, sparse_idx_map, orientation_idx_map, router_idx_map = (
            self._get_stage_class_maps()
        )

        for x in X:
            x = np.asarray(x).reshape(1, -1)

            # Router4 remains global.
            p_router = self.router4_clf.predict_proba(x)[0]
            ok, idx = self._confident(p_router, self.tau_router)
            if not ok:
                preds.append(self.fallback_label)
                continue

            router_label = list(router_idx_map.keys())[
                list(router_idx_map.values()).index(idx)
            ]

            if router_label == "Class4" and not self._has_orientation_stage():
                preds.append("kPEX-B-bnb")
                continue

            # Stage 1: resolve Dense vs Sparse first.
            p_group = self.group_clf.predict_proba(x)[0]
            ok, idx = self._confident(p_group, self.tau_group)
            if not ok:
                preds.append(self.fallback_label)
                continue

            group_label = list(group_idx_map.keys())[
                list(group_idx_map.values()).index(idx)
            ]

            sparse_label = None

            # Stage 2 is only required for Sparse.
            if group_label == "Sparse":
                p_sparse = self.sparse_subgroup_clf.predict_proba(x)[0]
                ok, idx = self._confident(p_sparse, self.tau_sparse)
                if not ok:
                    preds.append(self.fallback_label)
                    continue

                sparse_label = list(sparse_idx_map.keys())[
                    list(sparse_idx_map.values()).index(idx)
                ]

            # Stage 3: choose the orientation classifier belonging to
            # the branch resolved above.
            orientation_label = "NonOriented"

            if self._has_orientation_stage():
                orientation_clf = self._get_orientation_classifier(
                    group_label,
                    sparse_label,
                )

                if orientation_clf is None:
                    preds.append(self.fallback_label)
                    continue

                p_orientation = orientation_clf.predict_proba(x)[0]
                ok, idx = self._confident(
                    p_orientation,
                    self.tau_orientation,
                )
                if not ok:
                    preds.append(self.fallback_label)
                    continue

                if self.uses_label_encoding:
                    orientation_label = self.le_orientation.inverse_transform(
                        [orientation_clf.classes_[idx]]
                    )[0]
                else:
                    orientation_label = orientation_clf.classes_[idx]

            preds.append(
                self._resolve_final_algorithm(
                    router_label,
                    group_label=group_label,
                    sparse_label=sparse_label,
                    orientation_label=orientation_label,
                )
            )

        return np.array(preds)

    def predict_proba(self, X):
        X = np.asarray(X)
        n_samples = len(X)
        algo_to_idx = {alg: i for i, alg in enumerate(self.classes_)}
        proba = np.zeros((n_samples, len(self.classes_)))

        group_idx_map, sparse_idx_map, orientation_idx_map, router_idx_map = (
            self._get_stage_class_maps()
        )

        p_group = self.group_clf.predict_proba(X)

        # Router4 deliberately remains global.
        p_router = self.router4_clf.predict_proba(X)

        p_orientation_by_branch = {}

        if self._has_orientation_stage():
            p_orientation_by_branch = {
                "Dense": self.orientation_dense_clf.predict_proba(X),
                "AltRB": self.orientation_altRB_clf.predict_proba(X),
                "No-AltRB": self.orientation_noAltRB_clf.predict_proba(X),
            }

        def class_probability(probabilities, idx_map, label):
            return probabilities[idx_map[label]] if label in idx_map else 0.0

        for i in range(n_samples):
            p_dense = (
                p_group[i, group_idx_map["Dense"]]
                if "Dense" in group_idx_map
                else 0.0
            )
            p_sparse = (
                p_group[i, group_idx_map["Sparse"]]
                if "Sparse" in group_idx_map
                else 0.0
            )

            p_sparse_sub = self.sparse_subgroup_clf.predict_proba(
                X[i].reshape(1, -1)
            )[0]

            p_alt_conditional = (
                p_sparse_sub[sparse_idx_map["AltRB"]]
                if "AltRB" in sparse_idx_map
                else 0.0
            )
            p_no_conditional = (
                p_sparse_sub[sparse_idx_map["No-AltRB"]]
                if "No-AltRB" in sparse_idx_map
                else 0.0
            )

            p_alt = p_sparse * p_alt_conditional
            p_no = p_sparse * p_no_conditional

            if not self._has_orientation_stage():
                class1 = (
                    p_router[i, router_idx_map["Class1"]]
                    if "Class1" in router_idx_map
                    else 0.0
                )
                class2 = (
                    p_router[i, router_idx_map["Class2"]]
                    if "Class2" in router_idx_map
                    else 0.0
                )
                class3 = (
                    p_router[i, router_idx_map["Class3"]]
                    if "Class3" in router_idx_map
                    else 0.0
                )
                class4 = (
                    p_router[i, router_idx_map["Class4"]]
                    if "Class4" in router_idx_map
                    else 0.0
                )

                contributions = {
                    "UMKP-S-bnb": p_no * class1,
                    "UMKP-R-bnb": p_no * class2,
                    "UMKP-P-bnb": p_no * class3,
                    "UMKP+-S-bnb": p_alt * class1,
                    "UMKP+-R-bnb": p_alt * class2,
                    "UMKP+-P-bnb": p_alt * class3,
                    "kPEX-S-bnb": p_dense * class1,
                    "kPEX-R-bnb": p_dense * class2,
                    "kPEX-P-bnb": p_dense * class3,
                    "kPEX-B-bnb": class4,
                }

            else:
                contributions = {}

                for alg in ACTIVE_ALGORITHMS:
                    group_label = map_group(alg)
                    sparse_label = map_sparse_subgroup(alg)
                    orientation_label = map_orientation(alg)
                    router_label = map_router4(alg)

                    if group_label == "Dense":
                        group_probability = p_dense
                        branch_name = "Dense"
                        sparse_probability = 1.0

                    elif sparse_label == "AltRB":
                        group_probability = p_sparse
                        branch_name = "AltRB"
                        sparse_probability = p_alt_conditional

                    elif sparse_label == "No-AltRB":
                        group_probability = p_sparse
                        branch_name = "No-AltRB"
                        sparse_probability = p_no_conditional

                    else:
                        contributions[alg] = 0.0
                        continue

                    branch_orientation_clf = self._get_orientation_classifier(
                        group_label,
                        sparse_label,
                    )

                    branch_orientation_probs = p_orientation_by_branch[
                        branch_name
                    ][i]

                    if self.uses_label_encoding:
                        branch_orientation_labels = list(
                            self.le_orientation.inverse_transform(
                                branch_orientation_clf.classes_
                            )
                        )
                    else:
                        branch_orientation_labels = list(
                            branch_orientation_clf.classes_
                        )

                    branch_orientation_idx_map = {
                        label: j
                        for j, label in enumerate(branch_orientation_labels)
                    }

                    orientation_probability = class_probability(
                        branch_orientation_probs,
                        branch_orientation_idx_map,
                        orientation_label,
                    )

                    router_probability = class_probability(
                        p_router[i],
                        router_idx_map,
                        router_label,
                    )

                    contributions[alg] = (
                        group_probability
                        * sparse_probability
                        * orientation_probability
                        * router_probability
                    )

            for alg, value in contributions.items():
                if alg in algo_to_idx:
                    proba[i, algo_to_idx[alg]] += value

        return proba


def build_hierarchical_scheme3_random_forest_classifier(
    group_clf_kwargs=None,
    sparse_subgroup_clf_kwargs=None,
    orientation_clf_kwargs=None,
    router4_clf_kwargs=None,
    tau_group=0.0,
    tau_sparse=0.0,
    tau_orientation=0.0,
    tau_router=0.0,
    fallback_label="SAFE_DEFAULT_ALGO",
):
    group_params = dict(SCHEME3_RF_PRESETS["group_clf"])
    group_params.update(group_clf_kwargs or {})

    sparse_params = dict(SCHEME3_RF_PRESETS["sparse_subgroup_clf"])
    sparse_params.update(sparse_subgroup_clf_kwargs or {})

    orientation_params = dict(SCHEME3_RF_PRESETS["orientation_clf"])
    orientation_params.update(orientation_clf_kwargs or {})

    router_params = dict(SCHEME3_RF_PRESETS["router4_clf"])
    router_params.update(router4_clf_kwargs or {})

    return HierarchicalClassifier3(
        group_clf=RandomForestClassifier(**group_params),
        sparse_subgroup_clf=RandomForestClassifier(**sparse_params),
        orientation_clf=RandomForestClassifier(**orientation_params),
        router4_clf=RandomForestClassifier(**router_params),
        tau_group=tau_group,
        tau_sparse=tau_sparse,
        tau_orientation=tau_orientation,
        tau_router=tau_router,
        fallback_label=fallback_label,
    )


def build_hierarchical_scheme3_xgb_classifier(
    group_clf_kwargs=None,
    sparse_subgroup_clf_kwargs=None,
    orientation_clf_kwargs=None,
    router4_clf_kwargs=None,
    le_group=None,
    le_sparse=None,
    le_orientation=None,
    le_router=None,
    le_algo=None,
    tau_group=0.0,
    tau_sparse=0.0,
    tau_orientation=0.0,
    tau_router=0.0,
    fallback_label="SAFE_DEFAULT_ALGO",
):
    if XGBClassifier is None:  # pragma: no cover - optional dependency
        raise ImportError("xgboost is required to build Scheme 3 XGB models.")

    group_params = dict(SCHEME3_XGB_PRESETS["group_clf"])
    group_params.update(group_clf_kwargs or {})

    sparse_params = dict(SCHEME3_XGB_PRESETS["sparse_subgroup_clf"])
    sparse_params.update(sparse_subgroup_clf_kwargs or {})

    orientation_params = dict(SCHEME3_XGB_PRESETS["orientation_clf"])
    orientation_params.update(orientation_clf_kwargs or {})

    router_params = dict(SCHEME3_XGB_PRESETS["router4_clf"])
    router_params.update(router4_clf_kwargs or {})

    return HierarchicalClassifier3(
        group_clf=XGBClassifier(**group_params),
        sparse_subgroup_clf=XGBClassifier(**sparse_params),
        orientation_clf=XGBClassifier(**orientation_params),
        router4_clf=XGBClassifier(**router_params),
        le_group=le_group,
        le_sparse=le_sparse,
        le_orientation=le_orientation,
        le_router=le_router,
        le_algo=le_algo,
        tau_group=tau_group,
        tau_sparse=tau_sparse,
        tau_orientation=tau_orientation,
        tau_router=tau_router,
        fallback_label=fallback_label,
    )


def _get_hier3_split_function(split_mode="strat_y"):
    if split_mode == "strat_y":
        return prepare_train_test_strat_Y_hier3
    if split_mode == "strat_y_K":
        return prepare_train_test_strat_Y_K_hier3
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def _ensure_scheme3_columns(cat_df, weight_col="weight"):
    required_cols = {
        "y_algo",
        "y_group",
        "y_sparse_subgroup",
        "y_orientation",
        "y_router4",
        "weight_group",
        "weight_sparse_subgroup",
        "weight_orientation",
        "weight_router4",
    }
    if required_cols.issubset(cat_df.columns):
        return cat_df

    scheme3_df = cat_df.copy()
    return assign_scheme3_targets_and_weights(scheme3_df, weight_col=weight_col)


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


def _resolve_router_num_class(model_kwargs, y_router4):
    resolved = dict(model_kwargs)
    resolved["router4_clf_kwargs"] = dict(resolved.get("router4_clf_kwargs", {}))
    resolved["router4_clf_kwargs"]["num_class"] = int(len(np.unique(y_router4)))
    return resolved


def _decode_router_predictions(hclf3, y_pred_local):
    if not hclf3.uses_label_encoding:
        return y_pred_local
    global_router = np.array([hclf3.router_map[i] for i in y_pred_local])
    return hclf3.le_router.inverse_transform(global_router)


def create_label_encoders_scheme3(
    y_group=None,
    y_sparse_subgroup=None,
    y_orientation=None,
    y_router4=None,
    y_algo=None,
):
    le_group = LabelEncoder()
    le_sparse = LabelEncoder()
    le_orientation = LabelEncoder()
    le_router = LabelEncoder()
    le_algo = LabelEncoder()

    if y_group is not None:
        le_group.fit(pd.Series(y_group).dropna())
    if y_sparse_subgroup is not None:
        le_sparse.fit(pd.Series(y_sparse_subgroup).dropna())
    if y_orientation is not None:
        le_orientation.fit(pd.Series(y_orientation).dropna())
    if y_router4 is not None:
        le_router.fit(pd.Series(y_router4).dropna())
    if y_algo is not None:
        le_algo.fit(pd.Series(y_algo).dropna())

    return le_group, le_sparse, le_orientation, le_router, le_algo


def prepare_train_test_data_hier3(
    cat_df,
    split_mode="strat_y",
    weight_col="weight",
    test_size=0.2,
    random_state=42,
    for_nn=False,
    kcore_list=None,
    tie_tolerance=0.00,
):
    scheme3_df = _ensure_scheme3_columns(cat_df, weight_col=weight_col)
    split_function = _get_hier3_split_function(split_mode)
    return split_function(
        scheme3_df,
        weight_col=weight_col,
        test_size=test_size,
        random_state=random_state,
        for_nn=for_nn,
        kcore_list=kcore_list,
        tie_tolerance=tie_tolerance,
    )
def fit_hierarchical_scheme3_random_forest_classifier(
    cat_df,
    split_mode="strat_y",
    model_kwargs=None,
    split_kwargs=None,
):
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    split_kwargs = {} if split_kwargs is None else dict(split_kwargs)

    split_data_raw = prepare_train_test_data_hier3(cat_df, split_mode=split_mode, **split_kwargs)
    (
        X_train,
        X_test,
        y_algo_train,
        y_algo_test,
        y_group_train,
        y_group_test,
        y_sparse_subgroup_train,
        y_sparse_subgroup_test,
        y_orientation_train,
        y_orientation_test,
        y_router4_train,
        y_router4_test,
        weight_group_train,
        weight_group_test,
        weight_sparse_train,
        weight_sparse_test,
        weight_orientation_train,
        weight_orientation_test,
        weight_router4_train,
        weight_router4_test,
    ) = split_data_raw

    clf = build_hierarchical_scheme3_random_forest_classifier(**model_kwargs)
    clf.fit(
        X_train.values,
        y_group_train.values,
        y_sparse_subgroup_train.values,
        y_orientation_train.values,
        y_router4_train.values,
        y_algo_train.values,
        weight_group=weight_group_train.values,
        weight_sparse=weight_sparse_train.values,
        weight_orientation=weight_orientation_train.values,
        weight_router4=weight_router4_train.values,
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
        "y_orientation_train": y_orientation_train,
        "y_orientation_test": y_orientation_test,
        "y_router4_train": y_router4_train,
        "y_router4_test": y_router4_test,
        "weight_group_train": weight_group_train,
        "weight_group_test": weight_group_test,
        "weight_sparse_train": weight_sparse_train,
        "weight_sparse_test": weight_sparse_test,
        "weight_orientation_train": weight_orientation_train,
        "weight_orientation_test": weight_orientation_test,
        "weight_router4_train": weight_router4_train,
        "weight_router4_test": weight_router4_test,
    }
    return clf, split_data


def fit_hierarchical_scheme3_xgb_classifier(
    cat_df,
    split_mode="strat_y",
    model_kwargs=None,
    split_kwargs=None,
):
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    split_kwargs = {} if split_kwargs is None else dict(split_kwargs)

    split_data_raw = prepare_train_test_data_hier3(cat_df, split_mode=split_mode, **split_kwargs)
    (
        X_train,
        X_test,
        y_algo_train,
        y_algo_test,
        y_group_train,
        y_group_test,
        y_sparse_subgroup_train,
        y_sparse_subgroup_test,
        y_orientation_train,
        y_orientation_test,
        y_router4_train,
        y_router4_test,
        weight_group_train,
        weight_group_test,
        weight_sparse_train,
        weight_sparse_test,
        weight_orientation_train,
        weight_orientation_test,
        weight_router4_train,
        weight_router4_test,
    ) = split_data_raw

    le_group, le_sparse, le_orientation, le_router, le_algo = create_label_encoders_scheme3(
        y_group=y_group_train,
        y_sparse_subgroup=y_sparse_subgroup_train,
        y_orientation=y_orientation_train,
        y_router4=y_router4_train,
        y_algo=y_algo_train,
    )

    y_group_train_enc = le_group.transform(y_group_train)
    y_group_test_enc = le_group.transform(y_group_test)
    y_sparse_train_enc = _transform_optional_labels(y_sparse_subgroup_train, le_sparse)
    y_sparse_test_enc = _transform_optional_labels(y_sparse_subgroup_test, le_sparse)
    y_orientation_train_enc = le_orientation.transform(y_orientation_train)
    y_orientation_test_enc = le_orientation.transform(y_orientation_test)
    y_router4_train_enc = le_router.transform(y_router4_train)
    y_router4_test_enc = le_router.transform(y_router4_test)
    y_algo_train_enc = le_algo.transform(y_algo_train)
    y_algo_test_enc = le_algo.transform(y_algo_test)

    model_kwargs = _resolve_router_num_class(model_kwargs, y_router4_train_enc)

    clf = build_hierarchical_scheme3_xgb_classifier(
        le_group=le_group,
        le_sparse=le_sparse,
        le_orientation=le_orientation,
        le_router=le_router,
        le_algo=le_algo,
        **model_kwargs,
    )
    clf.fit(
        X_train.values,
        y_group_train_enc,
        y_sparse_train_enc,
        y_orientation_train_enc,
        y_router4_train_enc,
        y_algo_train_enc,
        weight_group=weight_group_train.values,
        weight_sparse=weight_sparse_train.values,
        weight_orientation=weight_orientation_train.values,
        weight_router4=weight_router4_train.values,
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
        "y_orientation_train": y_orientation_train,
        "y_orientation_test": y_orientation_test,
        "y_router4_train": y_router4_train,
        "y_router4_test": y_router4_test,
        "y_group_train_enc": y_group_train_enc,
        "y_group_test_enc": y_group_test_enc,
        "y_sparse_train_enc": y_sparse_train_enc,
        "y_sparse_test_enc": y_sparse_test_enc,
        "y_orientation_train_enc": y_orientation_train_enc,
        "y_orientation_test_enc": y_orientation_test_enc,
        "y_router4_train_enc": y_router4_train_enc,
        "y_router4_test_enc": y_router4_test_enc,
        "y_algo_train_enc": y_algo_train_enc,
        "y_algo_test_enc": y_algo_test_enc,
        "le_group": le_group,
        "le_sparse": le_sparse,
        "le_orientation": le_orientation,
        "le_router": le_router,
        "le_algo": le_algo,
        "weight_group_train": weight_group_train,
        "weight_group_test": weight_group_test,
        "weight_sparse_train": weight_sparse_train,
        "weight_sparse_test": weight_sparse_test,
        "weight_orientation_train": weight_orientation_train,
        "weight_orientation_test": weight_orientation_test,
        "weight_router4_train": weight_router4_train,
        "weight_router4_test": weight_router4_test,
    }
    return clf, split_data

def evaluate_hierarchical_scheme3_stages(results_df, hclf3, X_full, algorithms):
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

    if hclf3.uses_label_encoding:
        y_true = results_df["y_group"].values
        y_pred = hclf3.le_group.inverse_transform(hclf3.group_clf.predict(X_full))
        evaluate_stage("Stage 1 (Sparse vs Dense)", y_true, y_pred, labels=["Sparse", "Dense"], enforce_order=True)

        mask = results_df["y_group"] == "Sparse"
        if mask.sum() > 0:
            y_true = results_df.loc[mask, "y_sparse_subgroup"].values
            y_pred = hclf3.le_sparse.inverse_transform(hclf3.sparse_subgroup_clf.predict(X_full[mask]))
            evaluate_stage("Stage 2 (AltRB vs No-AltRB)", y_true, y_pred, labels=["AltRB", "No-AltRB"], enforce_order=True)

        if hclf3._has_orientation_stage():
            orientation_branches = {
                "Dense": results_df["y_group"].eq("Dense").to_numpy(),
                "Sparse-AltRB": (
                    results_df["y_group"].eq("Sparse")
                    & results_df["y_sparse_subgroup"].eq("AltRB")
                ).to_numpy(),
                "Sparse-NoAltRB": (
                    results_df["y_group"].eq("Sparse")
                    & results_df["y_sparse_subgroup"].eq("No-AltRB")
                ).to_numpy(),
            }

            orientation_classifiers = {
                "Dense": hclf3.orientation_dense_clf,
                "Sparse-AltRB": hclf3.orientation_altRB_clf,
                "Sparse-NoAltRB": hclf3.orientation_noAltRB_clf,
            }

            for branch_name, branch_mask in orientation_branches.items():
                if not branch_mask.any():
                    continue

                branch_clf = orientation_classifiers[branch_name]

                y_true = results_df.loc[
                    branch_mask,
                    "y_orientation",
                ].values

                y_pred = hclf3.le_orientation.inverse_transform(
                    branch_clf.predict(X_full[branch_mask])
                )

                evaluate_stage(
                    f"Stage 3 ({branch_name} Orientation)",
                    y_true,
                    y_pred,
                    labels=SCHEME3_ORIENTATION_LABELS,
                    enforce_order=True,
                )

        y_true = results_df["y_router4"].values
        y_pred_local = hclf3.router4_clf.predict(X_full)
        y_pred = _decode_router_predictions(hclf3, y_pred_local)
        router_stage_name = "Stage 4 (B-S-R-P router)" if hclf3._has_orientation_stage() else "Stage 3 (4-class router)"
        evaluate_stage(router_stage_name, y_true, y_pred, labels=SCHEME3_ROUTER_LABELS, enforce_order=True)
    else:
        y_true = results_df["y_group"].values
        y_pred = hclf3.group_clf.predict(X_full)
        evaluate_stage("Stage 1 (Sparse vs Dense)", y_true, y_pred, labels=["Sparse", "Dense"], enforce_order=True)

        mask = results_df["y_group"] == "Sparse"
        if mask.sum() > 0:
            y_true = results_df.loc[mask, "y_sparse_subgroup"].values
            y_pred = hclf3.sparse_subgroup_clf.predict(X_full[mask])
            evaluate_stage("Stage 2 (AltRB vs No-AltRB)", y_true, y_pred, labels=["AltRB", "No-AltRB"], enforce_order=True)

        if hclf3._has_orientation_stage():
            orientation_branches = {
                "Dense": results_df["y_group"].eq("Dense").to_numpy(),
                "Sparse-AltRB": (
                    results_df["y_group"].eq("Sparse")
                    & results_df["y_sparse_subgroup"].eq("AltRB")
                ).to_numpy(),
                "Sparse-NoAltRB": (
                    results_df["y_group"].eq("Sparse")
                    & results_df["y_sparse_subgroup"].eq("No-AltRB")
                ).to_numpy(),
            }

            orientation_classifiers = {
                "Dense": hclf3.orientation_dense_clf,
                "Sparse-AltRB": hclf3.orientation_altRB_clf,
                "Sparse-NoAltRB": hclf3.orientation_noAltRB_clf,
            }

            for branch_name, branch_mask in orientation_branches.items():
                if not branch_mask.any():
                    continue

                branch_clf = orientation_classifiers[branch_name]

                y_true = results_df.loc[
                    branch_mask,
                    "y_orientation",
                ].values

                y_pred = branch_clf.predict(X_full[branch_mask])

                evaluate_stage(
                    f"Stage 3 ({branch_name}: Oriented vs NonOriented)",
                    y_true,
                    y_pred,
                    labels=SCHEME3_ORIENTATION_LABELS,
                    enforce_order=True,
                )

        y_true = results_df["y_router4"].values
        y_pred = hclf3.router4_clf.predict(X_full)
        router_stage_name = "Stage 4 (B-S-R-P router)" if hclf3._has_orientation_stage() else "Stage 3 (4-class router)"
        evaluate_stage(router_stage_name, y_true, y_pred, labels=SCHEME3_ROUTER_LABELS, enforce_order=True)

    y_true = results_df["y_algo"].values
    y_pred = hclf3.predict(X_full)
    evaluate_stage("Final (End-to-End)", y_true, y_pred, labels=algorithms, enforce_order=True)

    summary_df = pd.DataFrame(summary_rows)
    print("\n==============================")
    print("Scheme 3 Stage-wise Summary Metrics")
    print("==============================\n")
    print(summary_df)
    return summary_df, per_class_reports

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

    if clf._has_orientation_stage():
        branch_specs = [
            (
                "Dense",
                results_df["y_group"].eq("Dense"),
                clf.orientation_dense_clf,
            ),
            (
                "AltRB",
                results_df["y_group"].eq("Sparse")
                & results_df["y_sparse_subgroup"].eq("AltRB"),
                clf.orientation_altRB_clf,
            ),
            (
                "NoAltRB",
                results_df["y_group"].eq("Sparse")
                & results_df["y_sparse_subgroup"].eq("No-AltRB"),
                clf.orientation_noAltRB_clf,
            ),
        ]

        for branch_name, branch_mask, branch_clf in branch_specs:
            if not branch_mask.any():
                continue

            X_branch = X_full[branch_mask]
            p_orientation = branch_clf.predict_proba(X_branch)

            if clf.uses_label_encoding:
                orientation_labels = list(
                    clf.le_orientation.inverse_transform(
                        branch_clf.classes_
                    )
                )
            else:
                orientation_labels = list(branch_clf.classes_)

            for i, cls in enumerate(orientation_labels):
                col = f"prob_stage3_{branch_name}_{cls}"
                results_df.loc[branch_mask, col] = p_orientation[:, i]

        for branch_name in ("Dense", "AltRB", "NoAltRB"):
            for cls in SCHEME3_ORIENTATION_LABELS:
                col = f"prob_stage3_{branch_name}_{cls}"
                if col not in results_df.columns:
                    results_df[col] = 0.0
                else:
                    results_df[col] = results_df[col].fillna(0.0)

    p_router = clf.router4_clf.predict_proba(X_full)
    if clf.uses_label_encoding:
        router_labels = list(
            clf.le_router.inverse_transform([clf.router_map[i] for i in clf.router4_clf.classes_])
        )
    else:
        router_labels = list(clf.router4_clf.classes_)
    router_stage_prefix = "prob_stage4" if clf._has_orientation_stage() else "prob_stage3"
    for i, cls in enumerate(router_labels):
        results_df[f"{router_stage_prefix}_{cls}"] = p_router[:, i]

    for cls in SCHEME3_ROUTER_LABELS:
        col = f"{router_stage_prefix}_{cls}"
        if col not in results_df.columns:
            results_df[col] = 0.0

    return results_df


def run_full_evaluation_scheme3(
    cat_df,
    clf,
    features,
    algorithms,
    le=None,
    drop_indices=None,
    epsilon=1e-3,
    tol=0.0,
    save_files=False,
    output_dir="./",
    file_suffix="scheme3_eval",
    kcore_list=False,
    for_nn=False,
):
    results_df = _ensure_scheme3_columns(cat_df).copy()
    results_df["fastest_algorithms"] = results_df.apply(get_fastest_algorithms, axis=1)

    X_full = _prepare_X_values(results_df, features, kcore_list=kcore_list, for_nn=for_nn)

    print("Predicting Scheme 3 final outputs...")
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
        print("Adding Scheme 3 hierarchical leaf probabilities...")
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

    print("Adding Scheme 3 stage probabilities...")
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


__all__ = [
    "HierarchicalClassifier3",
    "SCHEME3_RF_PRESETS",
    "SCHEME3_ROUTER_LABELS",
    "SCHEME3_ROUTER_TO_ALGORITHMS",
    "SCHEME3_XGB_PRESETS",
    "algorithms",
    "build_hierarchical_scheme3_random_forest_classifier",
    "build_hierarchical_scheme3_xgb_classifier",
    "create_label_encoders_scheme3",
    "evaluate_hierarchical_scheme3_stages",
    "feature_cols",
    "fit_hierarchical_scheme3_random_forest_classifier",
    "fit_hierarchical_scheme3_xgb_classifier",
    "get_fastest_algorithms",
    "prepare_train_test_data_hier3",
    "run_full_evaluation_scheme3",
    "save_stage_reports",
]