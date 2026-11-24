"""Lightweight sanity checks for shared parsing and metadata helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from restructured_pipeline.src.compare_metadata_utils import (
    derive_scheme_and_model_columns,
    extract_compare_filename_info,
    infer_scheme_model_strat,
)
from restructured_pipeline.src.reporting_utils import (
    parse_nullable_literal,
    parse_report_literal,
    parse_saved_run_literal,
    resolve_runtime_column,
    round_or_none,
)


def main() -> None:
    assert round_or_none(1.23456) == 1.2346
    assert round_or_none(None) is None

    assert resolve_runtime_column(["fastest_time"], "fastest_runtime", "fastest_time") == "fastest_time"

    assert parse_saved_run_literal("('a', 1)") == ("a", 1)
    assert parse_saved_run_literal("not_a_literal") == "not_a_literal"
    assert parse_nullable_literal("('a', 1)") == ("a", 1)
    assert parse_nullable_literal("not_a_literal") is None
    assert parse_report_literal("[(algo, 0.5000, 1.2500)]") == [("algo", 0.5, 1.25)]

    scheme, model, strat = infer_scheme_model_strat(
        Path("wrong_predictions_scheme1_random_forest_strat_y_predict_inference_entire_data.xlsx")
    )
    assert (scheme, model, strat) == ("scheme 1", "RF", "strat_y")

    metadata_df = extract_compare_filename_info(
        pd.DataFrame({"filename": ["wrong_predictions_scheme2_xgboost_strat_y_K_predict_inference_entire_data.xlsx"]})
    )
    assert metadata_df.loc[0, "inference_on"] == "on entire dataset"
    assert metadata_df.loc[0, "strat_on"] == "y and K"

    scheme_df = derive_scheme_and_model_columns(
        pd.DataFrame({"filename": ["wrong_predictions_scheme3_random_forest_strat_y.xlsx"]})
    )
    assert scheme_df.loc[0, "scheme"] == "scheme 3"
    assert scheme_df.loc[0, "model"] == "RF"

    print("helper_sanity_checks_ok")


if __name__ == "__main__":
    main()
