"""Shared filename and compare-metadata parsing helpers."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


SCHEME_LABELS = {
    "scheme1": "scheme 1",
    "scheme2": "scheme 2",
    "scheme3": "scheme 3",
}
MODEL_LABELS = {
    "random_forest": "RF",
    "xgboost": "XGB",
}
STRAT_ON_LABELS = {
    "y": "y",
    "y_K": "y and K",
    "yK": "y and K",
}
SCHEME_MODEL_PATTERN = re.compile(r"(scheme[123])_(random_forest|xgboost)")
SPLIT_METADATA_PATTERN = re.compile(r"strat_(y(?:_?K)?)(?:_exp_\d{2})?(?=_|\.|$)")
COMPARE_METADATA_PATTERN = re.compile(r"_(?:inference_)?(?:entire_data|test_data)(?:_|\.|$)")


def infer_scheme_model(path: str | Path) -> tuple[str, str]:
    path_obj = Path(path)
    text = path_obj.as_posix()

    match = SCHEME_MODEL_PATTERN.search(text)
    if match:
        return SCHEME_LABELS[match.group(1)], MODEL_LABELS[match.group(2)]

    parts = path_obj.parts
    scheme_token = next((part for part in parts if part in SCHEME_LABELS), None)
    model_token = next((part for part in parts if part in MODEL_LABELS), None)

    if scheme_token is None:
        raise ValueError(f"Could not infer scheme from {path}")
    if model_token is None:
        raise ValueError(f"Could not infer model from {path}")

    return SCHEME_LABELS[scheme_token], MODEL_LABELS[model_token]


def infer_stratified_on(path: str | Path) -> str:
    name = Path(path).name if isinstance(path, Path) else Path(str(path)).name
    if "strat_y_K" in name:
        return "strat_y_K"
    if "strat_yK" in name:
        return "strat_y_K"
    if "strat_y" in name:
        return "strat_y"
    raise ValueError(f"Could not infer split mode from {path}")


def infer_scheme_model_strat(path: str | Path) -> tuple[str, str, str]:
    scheme, model = infer_scheme_model(path)
    stratified_on = infer_stratified_on(path)
    return scheme, model, stratified_on


def infer_strategy_from_filename(filename: str) -> str:
    if "_with_holdout_" in filename:
        return "split_and_expand"
    if "_without_holdout_" in filename:
        return "expand_and_split"
    if "predict_inference_test_data" in filename or "eval_inference_test_data" in filename:
        return "split_and_expand"
    return "expand_and_split"


def has_compare_metadata(filename: str) -> bool:
    return bool(
        COMPARE_METADATA_PATTERN.search(filename)
        or "_with_holdout_" in filename
        or "_without_holdout_" in filename
        or "predict_inference_test_data" in filename
        or "eval_inference_test_data" in filename
    )


def has_split_metadata(filename: str) -> bool:
    return bool(SPLIT_METADATA_PATTERN.search(filename))


def derive_scheme_and_model_columns(
    df: pd.DataFrame,
    *,
    filename_col: str = "filename",
    source_model_col: str = "source_model",
) -> pd.DataFrame:
    normalized = df.copy()
    metadata_source = normalized.get(
        filename_col,
        pd.Series([""] * len(normalized), index=normalized.index),
    ).fillna("")
    if source_model_col in normalized.columns:
        metadata_source = metadata_source.where(
            metadata_source.astype(str).str.len() > 0,
            normalized[source_model_col].fillna("").astype(str),
        )

    extracted = metadata_source.astype(str).str.extract(SCHEME_MODEL_PATTERN)
    if source_model_col in normalized.columns:
        source_extracted = normalized[source_model_col].fillna("").astype(str).str.extract(SCHEME_MODEL_PATTERN)
        missing_scheme = extracted[0].isna()
        missing_model = extracted[1].isna()
        extracted.loc[missing_scheme, 0] = source_extracted.loc[missing_scheme, 0]
        extracted.loc[missing_model, 1] = source_extracted.loc[missing_model, 1]
    normalized["scheme"] = extracted[0].map(SCHEME_LABELS).fillna("")
    normalized["model"] = extracted[1].map(MODEL_LABELS).fillna("")
    return normalized


def extract_compare_filename_info(
    df: pd.DataFrame,
    *,
    filename_col: str = "filename",
) -> pd.DataFrame:
    normalized = df.copy()
    filename_series = normalized[filename_col]

    normalized["inference_on"] = filename_series.str.contains(
        r"_(?:inference_)?entire_data(?:_|\.|$)",
        regex=True,
    ).map({True: "on entire dataset", False: "on 20% test set"})

    strat_match = filename_series.str.extract(SPLIT_METADATA_PATTERN)
    normalized["strat_on"] = strat_match[0].map(STRAT_ON_LABELS).fillna("")

    exp_match = filename_series.str.extract(r"_exp_(\d{2})")
    normalized["exp_num"] = exp_match[0].fillna("00")

    return normalized
