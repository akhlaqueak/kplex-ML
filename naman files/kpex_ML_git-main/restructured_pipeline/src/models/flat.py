"""Flat model wrappers for Scheme 1 multiclass classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from ..data_splits import (
    algorithms,
    prepare_train_test_strat_Y,
    prepare_train_test_strat_Y_K,
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None


RF_PRESETS = {
    "rf_strat_y": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 4,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "rf_strat_y_K": {
        "n_estimators": 400,
        "max_depth": 20,
        "min_samples_split": 2,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "rf_holdout_strat_y": {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 4,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "rf_holdout_strat_y_K": {
        "n_estimators": 400,
        "max_depth": 20,
        "min_samples_split": 2,
        "class_weight": "balanced",
        "random_state": 42,
    },
}

XGB_PRESETS = {
    "xgb_strat_y": {
        "objective": "multi:softprob",
        "n_estimators": 400,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "use_label_encoder": False,
        "random_state": 42,
    },
    "xgb_strat_y_K": {
        "objective": "multi:softprob",
        "n_estimators": 400,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "use_label_encoder": False,
        "random_state": 42,
    },
    "xgb_holdout_strat_y": {
        "objective": "multi:softprob",
        "n_estimators": 200,
        "max_depth": 10,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "random_state": 42,
    },
    "xgb_holdout_strat_y_K": {
        "objective": "multi:softprob",
        "n_estimators": 150,
        "max_depth": 15,
        "learning_rate": 0.75,
        "use_label_encoder": False,
        "random_state": 42,
    },
}


def _get_split_function(split_mode="strat_y"):
    if split_mode == "strat_y":
        return prepare_train_test_strat_Y
    if split_mode == "strat_y_K":
        return prepare_train_test_strat_Y_K
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def transform_sample_weight(sample_weight, mode="raw"):
    if sample_weight is None:
        return None

    if mode in (None, "raw"):
        return sample_weight

    sample_weight = np.asarray(sample_weight, dtype=float)

    if mode == "log1p":
        return np.log1p(sample_weight)

    raise ValueError(f"Unsupported sample_weight transform: {mode}")


def build_random_forest_classifier(model_name="rf_strat_y", **overrides):
    params = dict(RF_PRESETS.get(model_name, RF_PRESETS["rf_strat_y"]))
    params.update(overrides)
    return RandomForestClassifier(**params)


def build_xgb_classifier(model_name="xgb_strat_y", num_class=None, **overrides):
    if XGBClassifier is None:  # pragma: no cover - optional dependency
        raise ImportError("xgboost is required to build XGBClassifier models.")

    params = dict(XGB_PRESETS.get(model_name, XGB_PRESETS["xgb_strat_y"]))
    params["num_class"] = len(algorithms) if num_class is None else num_class
    params.update(overrides)
    return XGBClassifier(**params)


def prepare_train_test_data(
    cat_df,
    split_mode="strat_y",
    weight_col="weight",
    test_size=0.2,
    random_state=42,
    for_nn=False,
    kcore_list=None,
    tie_tolerance=0.00,
):
    split_function = _get_split_function(split_mode)
    return split_function(
        cat_df,
        weight_col=weight_col,
        test_size=test_size,
        random_state=random_state,
        for_nn=for_nn,
        kcore_list=kcore_list,
        tie_tolerance=tie_tolerance,
    )


def fit_random_forest_classifier(
    cat_df,
    model_name="rf_strat_y",
    split_mode="strat_y",
    model_kwargs=None,
    split_kwargs=None,
):
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    split_kwargs = {} if split_kwargs is None else dict(split_kwargs)

    X_train, X_test, y_train, y_test, weights_train, weights_test = prepare_train_test_data(
        cat_df,
        split_mode=split_mode,
        **split_kwargs,
    )

    clf = build_random_forest_classifier(model_name=model_name, **model_kwargs)
    clf.fit(X_train, y_train, sample_weight=weights_train)

    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "weights_train": weights_train,
        "weights_test": weights_test,
    }
    return clf, split_data


def fit_xgb_classifier(
    cat_df,
    model_name="xgb_strat_y",
    split_mode="strat_y",
    model_kwargs=None,
    split_kwargs=None,
    sample_weight_transform_mode="raw",
):
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    split_kwargs = {} if split_kwargs is None else dict(split_kwargs)

    X_train, X_test, y_train_str, y_test_str, weights_train, weights_test = prepare_train_test_data(
        cat_df,
        split_mode=split_mode,
        **split_kwargs,
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_test = le.transform(y_test_str)

    num_class = model_kwargs.pop("num_class", len(algorithms))
    clf = build_xgb_classifier(model_name=model_name, num_class=num_class, **model_kwargs)
    transformed_weights_train = transform_sample_weight(
        weights_train,
        mode=sample_weight_transform_mode,
    )
    clf.fit(X_train, y_train, sample_weight=transformed_weights_train)

    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_str": y_train_str,
        "y_test_str": y_test_str,
        "y_train": y_train,
        "y_test": y_test,
        "weights_train": transformed_weights_train,
        "weights_test": weights_test,
        "raw_weights_train": weights_train,
    }
    return clf, le, split_data

__all__ = [
    "RF_PRESETS",
    "XGB_PRESETS",
    "algorithms",
    "build_random_forest_classifier",
    "build_xgb_classifier",
    "fit_random_forest_classifier",
    "fit_xgb_classifier",
    "prepare_train_test_data",
    "transform_sample_weight",
]
