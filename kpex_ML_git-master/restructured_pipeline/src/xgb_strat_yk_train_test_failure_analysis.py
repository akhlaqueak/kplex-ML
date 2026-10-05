"""Analyze Scheme 1 / Scheme 3 XGB important failures and mixed
combination train/test origins for both strat_y and strat_y_K.

Inputs
------
1. Current enriched important-failure workbook.
2. XGB Scheme 1 + Scheme 3 mixed workbook for strat_y.
3. XGB Scheme 1 + Scheme 3 mixed workbook for strat_y_K.
4. Saved entire-data and test-data wrong-prediction exports.

Outputs
-------
1. One consolidated train/test analysis workbook.
2. A train-highlighted copy of the strat_y mixed workbook.
3. A train-highlighted copy of the strat_y_K mixed workbook.

The original input workbooks are never modified.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill


if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


from restructured_pipeline.src.compare_wrong_predictions import (
    autosize_columns,
)

from restructured_pipeline.src.failure_analysis_counts_and_train_test_summary import (
    DEFAULT_ENTIRE_RAW_GLOB,
    DEFAULT_IMPORTANT_CANDIDATES,
    DEFAULT_TEST_RAW_GLOB,
    MERGE_COLUMNS,
    annotate_important_partitions,
    load_important_filtered_rows,
    prepare_unfiltered_frames,
    resolve_existing_path,
)


# ============================================================================
# Constants
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROJECT_CONFIG_PATH = Path(
    "restructured_pipeline/configs/project.yaml"
)

DEFAULT_MIXED_STRAT_Y_PATH = Path(
    "restructured_pipeline/outputs/reports/"
    "mixed_failure_case_combinations_xgb_strat_y_05sep2026.xlsx"
)

DEFAULT_MIXED_STRAT_YK_PATH = Path(
    "restructured_pipeline/outputs/reports/"
    "mixed_failure_case_combinations_xgb_strat_y_K_05sep2026.xlsx"
)

DEFAULT_HIGHLIGHTED_STRAT_Y_PATH = Path(
    "restructured_pipeline/outputs/reports/"
    "mixed_failure_case_combinations_xgb_strat_y_05sep2026_"
    "train_highlighted.xlsx"
)

DEFAULT_HIGHLIGHTED_STRAT_YK_PATH = Path(
    "restructured_pipeline/outputs/reports/"
    "mixed_failure_case_combinations_xgb_strat_y_K_05sep2026_"
    "train_highlighted.xlsx"
)

DEFAULT_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/"
    "xgb_strat_y_and_strat_yK_train_test_failure_counts_05sep2026.xlsx"
)


XGB_MODEL = "XGB"

SUPPORTED_SPLITS = (
    "strat_y",
    "strat_y_K",
)


IMPORTANT_SHEET = "important_failure_counts"
MIXED_COUNTS_SHEET = "mixed_combo_train_test_counts"
NOTES_SHEET = "analysis_notes"

HIGHLIGHT_SUMMARY_SHEET = "train_test_failure_summary"


TRAIN_HIGHLIGHT_FILL = PatternFill(
    fill_type="solid",
    start_color="FFF9C4",
    end_color="FFF9C4",
)


# Sheets in the mixed workbook that are not combination sheets.
NON_COMBINATION_SHEETS = {
    "summary",
    "shared_failure_summary",
    HIGHLIGHT_SUMMARY_SHEET,
}


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--important-workbook",
        type=Path,
        default=None,
        help=(
            "Current enriched Scheme 1 / Scheme 3 "
            "failure-analysis workbook."
        ),
    )

    parser.add_argument(
        "--mixed-workbook-strat-y",
        type=Path,
        default=DEFAULT_MIXED_STRAT_Y_PATH,
        help="Mixed XGB Scheme 1 + Scheme 3 workbook for strat_y.",
    )

    parser.add_argument(
        "--mixed-workbook-strat-y-k",
        type=Path,
        default=DEFAULT_MIXED_STRAT_YK_PATH,
        help="Mixed XGB Scheme 1 + Scheme 3 workbook for strat_y_K.",
    )

    parser.add_argument(
        "--project-config-path",
        type=Path,
        default=DEFAULT_PROJECT_CONFIG_PATH,
    )

    parser.add_argument(
        "--entire-raw-glob",
        default=DEFAULT_ENTIRE_RAW_GLOB,
    )

    parser.add_argument(
        "--test-raw-glob",
        default=DEFAULT_TEST_RAW_GLOB,
    )

    parser.add_argument(
        "--analysis-output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--highlighted-strat-y-output-path",
        type=Path,
        default=DEFAULT_HIGHLIGHTED_STRAT_Y_PATH,
    )

    parser.add_argument(
        "--highlighted-strat-y-k-output-path",
        type=Path,
        default=DEFAULT_HIGHLIGHTED_STRAT_YK_PATH,
    )

    return parser.parse_args()


# ============================================================================
# Path helpers
# ============================================================================

def resolve_repo_path(path: Path) -> Path:

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def require_file(path: Path, label: str) -> Path:

    resolved = resolve_repo_path(path)

    if not resolved.exists():
        raise FileNotFoundError(
            f"{label} not found: {resolved}"
        )

    return resolved


# ============================================================================
# Important-failure summary
# ============================================================================

def build_important_summary(
    important_rows: pd.DataFrame,
) -> pd.DataFrame:

    filtered = important_rows[
        important_rows["model"].eq(XGB_MODEL)
        & important_rows["split_mode"].isin(SUPPORTED_SPLITS)
    ].copy()

    grouped = (
        filtered
        .groupby(
            [
                "scheme",
                "model",
                "split_mode",
            ],
            dropna=False,
        )["partition"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    for column in (
        "train",
        "test",
        "unknown",
    ):
        if column not in grouped.columns:
            grouped[column] = 0

    grouped["total_failures"] = (
        grouped["train"]
        + grouped["test"]
        + grouped["unknown"]
    )

    grouped = grouped.rename(
        columns={
            "scheme": "Scheme",
            "model": "Model",
            "split_mode": "Split",
            "train": "Training Failures",
            "test": "Testing Failures",
            "unknown": "Unknown Origin",
            "total_failures": "Total Failures",
        }
    )

    return (
        grouped[
            [
                "Scheme",
                "Model",
                "Split",
                "Training Failures",
                "Testing Failures",
                "Unknown Origin",
                "Total Failures",
            ]
        ]
        .sort_values(
            [
                "Split",
                "Scheme",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


# ============================================================================
# Mixed-workbook loading
# ============================================================================

def load_mixed_combination_rows(
    mixed_workbook: Path,
    split_mode: str,
) -> pd.DataFrame:
    """
    Load every actual combination sheet from one mixed workbook.

    The current mixed workbooks already contain:
        scheme
        model
        split_mode
        data_scope
        source_scheme
        source_rule

    Therefore no old master-workbook metadata recovery is needed.
    """

    workbook = pd.ExcelFile(
        mixed_workbook
    )

    frames: list[pd.DataFrame] = []

    for sheet_name in workbook.sheet_names:

        if sheet_name in NON_COMBINATION_SHEETS:
            continue

        df = pd.read_excel(
            mixed_workbook,
            sheet_name=sheet_name,
        ).copy()

        if df.empty:
            continue

        required = {
            "dataset",
            "k_value",
            "scheme",
            "model",
            "split_mode",
            "data_scope",
            "source_scheme",
            "source_rule",
            *MERGE_COLUMNS,
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Mixed sheet {sheet_name!r} "
                f"is missing required columns: "
                f"{sorted(missing)}"
            )

        # -----------------------------------------------------
        # Validate this workbook really represents the expected
        # experiment.
        # -----------------------------------------------------

        if not df["model"].eq(XGB_MODEL).all():
            raise ValueError(
                f"{sheet_name!r} contains non-XGB rows."
            )

        if not df["scheme"].isin(
            [
                "scheme 1",
                "scheme 3",
            ]
        ).all():
            raise ValueError(
                f"{sheet_name!r} contains rows outside "
                "Scheme 1 / Scheme 3."
            )

        if not df["split_mode"].eq(
            split_mode
        ).all():
            unexpected = (
                df.loc[
                    ~df["split_mode"].eq(split_mode),
                    "split_mode",
                ]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                f"{sheet_name!r} contains unexpected "
                f"split modes: {unexpected}"
            )

        if not df["data_scope"].eq(
            "entire_data"
        ).all():
            raise ValueError(
                f"{sheet_name!r} contains rows outside "
                "data_scope='entire_data'."
            )

        df.insert(
            0,
            "combination_sheet",
            sheet_name,
        )

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    return combined


# ============================================================================
# Train/test partitioning of mixed rows
# ============================================================================

def annotate_mixed_partitions(
    mixed_rows: pd.DataFrame,
    test_df: pd.DataFrame,
    split_mode: str,
) -> pd.DataFrame:
    """
    Determine whether each mixed failure row belongs to the held-out
    test partition.

    Mixed rows are already entire-data failures.

    test:
        matching failure exists in saved test-data wrong predictions

    train:
        entire-data failure does not appear in saved test wrong predictions
    """

    if mixed_rows.empty:
        return mixed_rows.copy()

    required = {
        "scheme",
        "model",
        "split_mode",
        *MERGE_COLUMNS,
    }

    missing_mixed = (
        required - set(mixed_rows.columns)
    )

    if missing_mixed:
        raise ValueError(
            "Mixed rows missing required columns: "
            f"{sorted(missing_mixed)}"
        )

    missing_test = (
        required - set(test_df.columns)
    )

    if missing_test:
        raise ValueError(
            "Test wrong-prediction rows missing required columns: "
            f"{sorted(missing_test)}"
        )

    # Only the matching XGB/split experiment is relevant.
    test_lookup_source = test_df[
        test_df["model"].eq(XGB_MODEL)
        & test_df["split_mode"].eq(split_mode)
        & test_df["scheme"].isin(
            [
                "scheme 1",
                "scheme 3",
            ]
        )
    ].copy()

    keys = [
        "scheme",
        "model",
        "split_mode",
        *MERGE_COLUMNS,
    ]

    test_lookup = (
        test_lookup_source[
            keys
        ]
        .drop_duplicates()
        .assign(_is_test=True)
    )

    annotated = mixed_rows.merge(
        test_lookup,
        on=keys,
        how="left",
    )

    annotated["partition"] = (
        annotated["_is_test"]
        .eq(True)
        .map(
            {
                True: "test",
                False: "train",
            }
        )
    )

    return annotated.drop(
        columns=["_is_test"]
    )


# ============================================================================
# Mixed summary
# ============================================================================

def build_mixed_summary(
    mixed_rows: pd.DataFrame,
    split_mode: str,
) -> pd.DataFrame:

    output_columns = [
        "Split",
        "Combination Sheet",
        "Source Rule",
        "Scheme Pair",
        "Training Failures",
        "Testing Failures",
        "Unknown Origin",
        "Total Failures",
        "Dominant Origin",
    ]

    if mixed_rows.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    rows = []

    for (
        sheet_name,
        group,
    ) in mixed_rows.groupby(
        "combination_sheet",
        dropna=False,
    ):

        source_rule = (
            str(group["source_rule"].iloc[0])
            if "source_rule" in group.columns
            else str(sheet_name)
        )

        train_count = int(
            group["partition"]
            .eq("train")
            .sum()
        )

        test_count = int(
            group["partition"]
            .eq("test")
            .sum()
        )

        unknown_count = int(
            group["partition"]
            .eq("unknown")
            .sum()
        )

        total_count = int(
            len(group)
        )

        if (
            test_count > train_count
            and test_count > unknown_count
        ):
            dominant = "testing"

        elif (
            train_count > test_count
            and train_count > unknown_count
        ):
            dominant = "training"

        elif (
            unknown_count
            > max(
                train_count,
                test_count,
            )
        ):
            dominant = "unknown"

        else:
            dominant = "balanced"

        rows.append(
            {
                "Split":
                    split_mode,

                "Combination Sheet":
                    sheet_name,

                "Source Rule":
                    source_rule,

                "Scheme Pair":
                    "scheme 1 + scheme 3",

                "Training Failures":
                    train_count,

                "Testing Failures":
                    test_count,

                "Unknown Origin":
                    unknown_count,

                "Total Failures":
                    total_count,

                "Dominant Origin":
                    dominant,
            }
        )

    return (
        pd.DataFrame(rows)
        [output_columns]
        .sort_values(
            [
                "Split",
                "Combination Sheet",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )


# ============================================================================
# Notes
# ============================================================================

def build_notes(
    important_workbook: Path,
    mixed_strat_y_workbook: Path,
    mixed_strat_yk_workbook: Path,
    highlighted_strat_y_path: Path,
    highlighted_strat_yk_path: Path,
    entire_raw_glob: str,
    test_raw_glob: str,
    important_unknown: int,
    mixed_unknown: int,
) -> pd.DataFrame:

    rows = [
        (
            "important_workbook_used",
            str(important_workbook),
        ),
        (
            "mixed_strat_y_workbook_used",
            str(mixed_strat_y_workbook),
        ),
        (
            "mixed_strat_y_K_workbook_used",
            str(mixed_strat_yk_workbook),
        ),
        (
            "highlighted_strat_y_output",
            str(highlighted_strat_y_path),
        ),
        (
            "highlighted_strat_y_K_output",
            str(highlighted_strat_yk_path),
        ),
        (
            "entire_wrong_prediction_glob",
            entire_raw_glob,
        ),
        (
            "test_wrong_prediction_glob",
            test_raw_glob,
        ),
        (
            "important_failure_filter",
            (
                "model == XGB and split_mode in "
                "{strat_y, strat_y_K}"
            ),
        ),
        (
            "mixed_analysis",
            (
                "Scheme 1 + Scheme 3; XGB only; "
                "strat_y and strat_y_K analyzed separately"
            ),
        ),
        (
            "train_test_origin_rule",
            (
                "test = mixed/important entire-data failure matched "
                "to saved predict_inference_test_data wrong row; "
                "train = entire-data failure not present in test "
                "wrong rows"
            ),
        ),
        (
            "important_unknown_origin_rows",
            str(important_unknown),
        ),
        (
            "mixed_unknown_origin_rows",
            str(mixed_unknown),
        ),
        (
            "mixed_training_highlight_rule",
            (
                "light yellow fill on full row only when "
                "partition == train"
            ),
        ),
        (
            "original_workbooks_modified",
            "no",
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "note",
            "value",
        ],
    )


# ============================================================================
# Consolidated analysis workbook
# ============================================================================

def write_analysis_workbook(
    output_path: Path,
    important_summary: pd.DataFrame,
    mixed_summary: pd.DataFrame,
    notes_df: pd.DataFrame,
) -> None:

    output_path = resolve_repo_path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        important_summary.to_excel(
            writer,
            sheet_name=IMPORTANT_SHEET,
            index=False,
        )

        mixed_summary.to_excel(
            writer,
            sheet_name=MIXED_COUNTS_SHEET,
            index=False,
        )

        notes_df.to_excel(
            writer,
            sheet_name=NOTES_SHEET,
            index=False,
        )

    workbook = load_workbook(
        output_path
    )

    for sheet_name in workbook.sheetnames:

        ws = workbook[
            sheet_name
        ]

        ws.freeze_panes = "A2"

        ws.auto_filter.ref = (
            ws.dimensions
        )

        autosize_columns(ws)

    workbook.save(
        output_path
    )

    workbook.close()


# ============================================================================
# Highlight training failures in mixed workbook copy
# ============================================================================

def write_highlighted_copy(
    source_workbook: Path,
    output_workbook: Path,
    mixed_rows: pd.DataFrame,
    mixed_summary: pd.DataFrame,
) -> None:

    source_workbook = resolve_repo_path(
        source_workbook
    )

    output_workbook = resolve_repo_path(
        output_workbook
    )

    output_workbook.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source_workbook,
        output_workbook,
    )

    workbook = load_workbook(
        output_workbook
    )

    # ---------------------------------------------------------
    # Add/replace train-test summary.
    # ---------------------------------------------------------

    if (
        HIGHLIGHT_SUMMARY_SHEET
        in workbook.sheetnames
    ):
        del workbook[
            HIGHLIGHT_SUMMARY_SHEET
        ]

    summary_ws = workbook.create_sheet(
        HIGHLIGHT_SUMMARY_SHEET,
        0,
    )

    summary_rows = [
        mixed_summary.columns.tolist(),
        *mixed_summary.values.tolist(),
    ]

    for (
        row_idx,
        row_values,
    ) in enumerate(
        summary_rows,
        start=1,
    ):

        for (
            col_idx,
            value,
        ) in enumerate(
            row_values,
            start=1,
        ):

            summary_ws.cell(
                row=row_idx,
                column=col_idx,
                value=value,
            )

    # ---------------------------------------------------------
    # Build training keys separately for every combination.
    # ---------------------------------------------------------

    training_keys_by_sheet: dict[
        str,
        set[tuple[object, ...]],
    ] = {}

    key_columns = [
        "dataset",
        "k_value",
        "scheme",
        "split_mode",
        "fastest_algo_tuple",
        "source_rule",
    ]

    if not mixed_rows.empty:

        for (
            sheet_name,
            group,
        ) in mixed_rows.groupby(
            "combination_sheet",
            dropna=False,
        ):

            train_group = group[
                group["partition"].eq(
                    "train"
                )
            ].copy()

            key_set = {
                tuple(
                    row[column]
                    for column in key_columns
                )
                for _, row
                in train_group[
                    key_columns
                ].iterrows()
            }

            training_keys_by_sheet[
                str(sheet_name)
            ] = key_set

    # ---------------------------------------------------------
    # Highlight train rows.
    # ---------------------------------------------------------

    for (
        sheet_name,
        key_set,
    ) in training_keys_by_sheet.items():

        if (
            sheet_name
            not in workbook.sheetnames
            or not key_set
        ):
            continue

        ws = workbook[
            sheet_name
        ]

        header = [
            ws.cell(
                row=1,
                column=col_idx,
            ).value
            for col_idx in range(
                1,
                ws.max_column + 1,
            )
        ]

        col_index = {
            str(name): idx + 1
            for idx, name
            in enumerate(header)
        }

        missing = [
            name
            for name in key_columns
            if name not in col_index
        ]

        if missing:
            raise ValueError(
                f"Cannot highlight training rows "
                f"in {sheet_name!r}; "
                f"missing columns: {missing}"
            )

        for row_idx in range(
            2,
            ws.max_row + 1,
        ):

            key = tuple(
                ws.cell(
                    row=row_idx,
                    column=col_index[name],
                ).value
                for name in key_columns
            )

            if key not in key_set:
                continue

            for col_idx in range(
                1,
                ws.max_column + 1,
            ):

                ws.cell(
                    row=row_idx,
                    column=col_idx,
                ).fill = (
                    TRAIN_HIGHLIGHT_FILL
                )

    # ---------------------------------------------------------
    # Formatting.
    # ---------------------------------------------------------

    for sheet_name in workbook.sheetnames:

        ws = workbook[
            sheet_name
        ]

        ws.freeze_panes = "A2"

        ws.auto_filter.ref = (
            ws.dimensions
        )

        autosize_columns(ws)

    workbook.save(
        output_workbook
    )

    workbook.close()


# ============================================================================
# Validation
# ============================================================================

def validate(
    important_summary: pd.DataFrame,
    mixed_summary: pd.DataFrame,
    highlighted_strat_y_path: Path,
    highlighted_strat_yk_path: Path,
    mixed_rows: pd.DataFrame,
) -> dict[str, object]:

    important_invalid = important_summary[
        (
            important_summary[
                "Training Failures"
            ]
            + important_summary[
                "Testing Failures"
            ]
            + important_summary[
                "Unknown Origin"
            ]
        )
        != important_summary[
            "Total Failures"
        ]
    ]

    if not important_invalid.empty:
        raise ValueError(
            "Important summary has invalid totals."
        )

    mixed_invalid = mixed_summary[
        (
            mixed_summary[
                "Training Failures"
            ]
            + mixed_summary[
                "Testing Failures"
            ]
            + mixed_summary[
                "Unknown Origin"
            ]
        )
        != mixed_summary[
            "Total Failures"
        ]
    ]

    if not mixed_invalid.empty:
        raise ValueError(
            "Mixed summary has invalid totals."
        )

    for (
        label,
        path,
    ) in (
        (
            "strat_y highlighted workbook",
            highlighted_strat_y_path,
        ),
        (
            "strat_y_K highlighted workbook",
            highlighted_strat_yk_path,
        ),
    ):

        resolved = resolve_repo_path(
            path
        )

        if not resolved.exists():
            raise FileNotFoundError(
                f"{label} was not created: "
                f"{resolved}"
            )

    split_values = set(
        mixed_summary[
            "Split"
        ].dropna()
    )

    expected_splits = set(
        SUPPORTED_SPLITS
    )

    if (
        not mixed_summary.empty
        and split_values
        != expected_splits
    ):
        raise ValueError(
            "Mixed summary does not contain "
            "both expected split modes. "
            f"Expected={expected_splits}, "
            f"found={split_values}"
        )

    return {
        "important_total":
            int(
                important_summary[
                    "Total Failures"
                ].sum()
            ),

        "mixed_total":
            int(
                mixed_summary[
                    "Total Failures"
                ].sum()
            ),

        "mixed_training_rows":
            int(
                mixed_rows[
                    "partition"
                ].eq("train").sum()
            ),

        "mixed_testing_rows":
            int(
                mixed_rows[
                    "partition"
                ].eq("test").sum()
            ),

        "mixed_unknown_rows":
            int(
                mixed_rows[
                    "partition"
                ].eq("unknown").sum()
            ),

        "splits_analyzed":
            sorted(
                mixed_summary[
                    "Split"
                ].dropna().unique()
            ),
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve inputs.
    # ------------------------------------------------------------------

    important_workbook = (
        resolve_existing_path(
            args.important_workbook,
            DEFAULT_IMPORTANT_CANDIDATES,
            "important workbook",
        )
    )

    mixed_strat_y = require_file(
        args.mixed_workbook_strat_y,
        "strat_y mixed workbook",
    )

    mixed_strat_yk = require_file(
        args.mixed_workbook_strat_y_k,
        "strat_y_K mixed workbook",
    )

    highlighted_strat_y = (
        resolve_repo_path(
            args.highlighted_strat_y_output_path
        )
    )

    highlighted_strat_yk = (
        resolve_repo_path(
            args.highlighted_strat_y_k_output_path
        )
    )

    analysis_output = resolve_repo_path(
        args.analysis_output_path
    )

    # ------------------------------------------------------------------
    # Load current important failures.
    #
    # load_important_filtered_rows() uses entire_data important failures.
    # ------------------------------------------------------------------

    filtered_source_rows = (
        load_important_filtered_rows(
            important_workbook
        )
    )

    # ------------------------------------------------------------------
    # Load current entire/test wrong-prediction exports.
    # ------------------------------------------------------------------

    (
        entire_unfiltered_df,
        test_unfiltered_df,
    ) = prepare_unfiltered_frames(
        args.project_config_path,
        args.entire_raw_glob,
        args.test_raw_glob,
    )

    # ------------------------------------------------------------------
    # Important failures:
    #
    # determine whether each entire-data important failure came from
    # the train or test partition.
    # ------------------------------------------------------------------

    important_rows = (
        annotate_important_partitions(
            filtered_source_rows,
            entire_unfiltered_df,
            test_unfiltered_df,
        )
    )

    important_rows = important_rows[
        important_rows[
            "model"
        ].eq(XGB_MODEL)
        & important_rows[
            "split_mode"
        ].isin(SUPPORTED_SPLITS)
        & important_rows[
            "scheme"
        ].isin(
            [
                "scheme 1",
                "scheme 3",
            ]
        )
    ].copy()

    important_summary = (
        build_important_summary(
            important_rows
        )
    )

    important_unknown = int(
        important_rows[
            "partition"
        ].eq("unknown").sum()
    )

    # ------------------------------------------------------------------
    # strat_y mixed combinations.
    # ------------------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "Mixed analysis: XGB / strat_y"
    )
    print(
        "============================================================"
    )

    mixed_y_rows = (
        load_mixed_combination_rows(
            mixed_strat_y,
            "strat_y",
        )
    )

    mixed_y_rows = (
        annotate_mixed_partitions(
            mixed_y_rows,
            test_unfiltered_df,
            "strat_y",
        )
    )

    mixed_y_summary = (
        build_mixed_summary(
            mixed_y_rows,
            "strat_y",
        )
    )

    # ------------------------------------------------------------------
    # strat_y_K mixed combinations.
    # ------------------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "Mixed analysis: XGB / strat_y_K"
    )
    print(
        "============================================================"
    )

    mixed_yk_rows = (
        load_mixed_combination_rows(
            mixed_strat_yk,
            "strat_y_K",
        )
    )

    mixed_yk_rows = (
        annotate_mixed_partitions(
            mixed_yk_rows,
            test_unfiltered_df,
            "strat_y_K",
        )
    )

    mixed_yk_summary = (
        build_mixed_summary(
            mixed_yk_rows,
            "strat_y_K",
        )
    )

    # ------------------------------------------------------------------
    # Combine both mixed summaries.
    # ------------------------------------------------------------------

    mixed_summary = pd.concat(
        [
            mixed_y_summary,
            mixed_yk_summary,
        ],
        ignore_index=True,
        sort=False,
    )

    mixed_rows = pd.concat(
        [
            mixed_y_rows,
            mixed_yk_rows,
        ],
        ignore_index=True,
        sort=False,
    )

    mixed_unknown = int(
        mixed_rows[
            "partition"
        ].eq("unknown").sum()
    )

    # ------------------------------------------------------------------
    # Notes.
    # ------------------------------------------------------------------

    notes_df = build_notes(
        important_workbook,
        mixed_strat_y,
        mixed_strat_yk,
        highlighted_strat_y,
        highlighted_strat_yk,
        args.entire_raw_glob,
        args.test_raw_glob,
        important_unknown,
        mixed_unknown,
    )

    # ------------------------------------------------------------------
    # Consolidated analysis workbook.
    # ------------------------------------------------------------------

    write_analysis_workbook(
        analysis_output,
        important_summary,
        mixed_summary,
        notes_df,
    )

    # ------------------------------------------------------------------
    # Highlight train failures in a COPY of each mixed workbook.
    # ------------------------------------------------------------------

    write_highlighted_copy(
        mixed_strat_y,
        highlighted_strat_y,
        mixed_y_rows,
        mixed_y_summary,
    )

    write_highlighted_copy(
        mixed_strat_yk,
        highlighted_strat_yk,
        mixed_yk_rows,
        mixed_yk_summary,
    )

    # ------------------------------------------------------------------
    # Validate.
    # ------------------------------------------------------------------

    validation = validate(
        important_summary,
        mixed_summary,
        highlighted_strat_y,
        highlighted_strat_yk,
        mixed_rows,
    )

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "XGB TRAIN/TEST FAILURE ANALYSIS COMPLETE"
    )
    print(
        "============================================================"
    )

    print(
        f"Analysis workbook:\n"
        f"  {analysis_output}"
    )

    print()

    print(
        "Highlighted strat_y mixed workbook:\n"
        f"  {highlighted_strat_y}"
    )

    print()

    print(
        "Highlighted strat_y_K mixed workbook:\n"
        f"  {highlighted_strat_yk}"
    )

    print()

    print(
        "Important failures summary:"
    )

    print(
        important_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Mixed combination summary:"
    )

    print(
        mixed_summary.to_string(
            index=False
        )
    )

    print()

    print(
        f"Validation: {validation}"
    )


if __name__ == "__main__":
    main()