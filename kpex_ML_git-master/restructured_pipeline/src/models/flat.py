"""Generic flat model wrappers for Scheme 1, Scheme 4, and Scheme 5 classification.

Scheme 1 uses the complete algorithm vocabulary by default.
Scheme 4/Scheme 5 reuse the same flat-model pipeline with an explicit algorithm subset
(e.g. ORIENTED_ALGOS or NON_ORIENTED_ALGOS) supplied through
split_kwargs['algorithm_cols'].
"""


from __future__ import annotations

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

def _resolve_algorithm_cols(algorithm_cols=None):
    """Resolve and validate the active flat-classification vocabulary."""
    active_algorithms = list(algorithms) if algorithm_cols is None else list(algorithm_cols)
    if not active_algorithms:
        raise ValueError("algorithm_cols must contain at least one algorithm.")
    if len(active_algorithms) != len(set(active_algorithms)):
        raise ValueError("algorithm_cols contains duplicate algorithm names.")
    return active_algorithms


def _validate_prepared_split(X_train, X_test, y_train, y_test, active_algorithms):
    """Validate source-row isolation and active-vocabulary membership."""
    overlap = set(X_train.index) & set(X_test.index)
    if overlap:
        raise ValueError(
            "Flat-model train/test source-row leakage detected after preparation: "
            f"{sorted(overlap)[:20]}"
        )
    if X_test.index.duplicated().any():
        duplicate_indices = X_test.index[X_test.index.duplicated(keep=False)].unique().tolist()
        raise ValueError(
            "Flat-model held-out X_test contains duplicate original source rows: "
            f"{duplicate_indices[:20]}"
        )

    active_set = set(active_algorithms)
    unexpected_train = set(pd.Series(y_train).dropna().astype(str).unique()) - active_set
    unexpected_test = set(pd.Series(y_test).dropna().astype(str).unique()) - active_set
    if unexpected_train:
        raise ValueError(
            "Training targets contain algorithms outside the active vocabulary: "
            f"{sorted(unexpected_train)}"
        )
    if unexpected_test:
        raise ValueError(
            "Test targets contain algorithms outside the active vocabulary: "
            f"{sorted(unexpected_test)}"
        )





def build_random_forest_classifier(model_name="rf_strat_y", **overrides):
    params = dict(RF_PRESETS.get(model_name, RF_PRESETS["rf_strat_y"]))
    params.update(overrides)
    return RandomForestClassifier(**params)


def build_xgb_classifier(model_name="xgb_strat_y", num_class=None, **overrides):
    if XGBClassifier is None:  # pragma: no cover - optional dependency
        raise ImportError("xgboost is required to build XGBClassifier models.")

    params = dict(XGB_PRESETS.get(model_name, XGB_PRESETS["xgb_strat_y"]))
    params["num_class"] = len(algorithms) if num_class is None else int(num_class)
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
    algorithm_cols=None
):
    active_algorithms = _resolve_algorithm_cols(algorithm_cols)
    split_function = _get_split_function(split_mode)
    return split_function(
        cat_df,
        weight_col=weight_col,
        test_size=test_size,
        random_state=random_state,
        for_nn=for_nn,
        kcore_list=kcore_list,
        tie_tolerance=tie_tolerance,
        algorithm_cols=active_algorithms,
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

    active_algorithms = _resolve_algorithm_cols(split_kwargs.get("algorithm_cols"))

    X_train, X_test, y_train, y_test, weights_train, weights_test = prepare_train_test_data(
        cat_df,
        split_mode=split_mode,
        **split_kwargs,
    )

    _validate_prepared_split(X_train, X_test, y_train, y_test, active_algorithms)


    clf = build_random_forest_classifier(model_name=model_name, **model_kwargs)
    clf.fit(X_train, y_train, sample_weight=weights_train)

    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "weights_train": weights_train,
        "weights_test": weights_test,
        "algorithms": active_algorithms,
        "num_class": len(active_algorithms),

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
    active_algorithms = _resolve_algorithm_cols(split_kwargs.get("algorithm_cols"))


    X_train, X_test, y_train_str, y_test_str, weights_train, weights_test = prepare_train_test_data(
        cat_df,
        split_mode=split_mode,
        **split_kwargs,
    )

    _validate_prepared_split(X_train, X_test, y_train_str, y_test_str, active_algorithms)


    # Fit on the complete ACTIVE vocabulary, not only observed training labels.
    # Scheme 1 -> 24 classes; Scheme 4 family -> 12 classes; Scheme 5 family -> 6 or 18 classes.

    le = LabelEncoder()
    le.fit((active_algorithms))
    y_train = le.transform(y_train_str)
    y_test = le.transform(y_test_str)

    if sample_weight_transform_mode == "raw":
        fit_weights = weights_train
    elif sample_weight_transform_mode == "log1p":
        import numpy as np
        fit_weights = pd.Series(
            np.log1p(pd.to_numeric(weights_train, errors="raise").to_numpy()),
            index=weights_train.index,
        )
    else:
        raise ValueError(
            "Unsupported sample_weight_transform_mode: "
            f"{sample_weight_transform_mode!r}"
        )

    expected_num_class = len(active_algorithms)
    num_class = int(model_kwargs.pop("num_class", expected_num_class))
    if num_class != expected_num_class:
        raise ValueError(
            "XGBoost num_class does not match the active algorithm vocabulary: "
            f"num_class={num_class}, active_algorithms={expected_num_class}."
        )


    clf = build_xgb_classifier(model_name=model_name, num_class=num_class, **model_kwargs)
    clf.fit(X_train, y_train, sample_weight=fit_weights)

    if len(le.classes_) != expected_num_class or set(le.classes_) != set(active_algorithms):
        raise ValueError("LabelEncoder classes do not match the active algorithm vocabulary.")


    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train_str": y_train_str,
        "y_test_str": y_test_str,
        "y_train": y_train,
        "y_test": y_test,
        "weights_train": weights_train,
        "weights_test": weights_test,
        "algorithms": active_algorithms,
        "num_class": expected_num_class,
    }
    return clf, le, split_data


def get_feature_importances(clf, X_train):
    if not hasattr(clf, "feature_importances_"):
        raise AttributeError(f"{clf.__class__.__name__} does not expose feature_importances_.")

    return pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)


__all__ = [
    "RF_PRESETS",
    "XGB_PRESETS",
    "algorithms",
    "build_random_forest_classifier",
    "build_xgb_classifier",
    "fit_random_forest_classifier",
    "fit_xgb_classifier",
    "get_feature_importances",
    "prepare_train_test_data",
]
