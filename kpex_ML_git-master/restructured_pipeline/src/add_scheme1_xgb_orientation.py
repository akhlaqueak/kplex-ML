"""Extract Scheme 1 XGB failure cases and classify fastest algorithms.

Reads only the Scheme 1 XGB sheets from the failure-analysis workbook:

    scheme_1_XGB_strat_y
    scheme_1_XGB_strat_y_K

Adds:

    algorithm_orientation

Classification rule:
    fastest algorithm contains "oriented" -> "Oriented"
    otherwise                             -> "Forward"

All other sheets are discarded from the output workbook.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


DEFAULT_INPUT = Path(
    "restructured_pipeline/outputs/reports/"
    "failure_analysis_summary_scheme1_scheme3_09sep2026.xlsx"
)

DEFAULT_OUTPUT = Path(
    "restructured_pipeline/outputs/reports/"
    "failure_analysis_scheme1_xgb_orientation_09sep2026.xlsx"
)

SHEETS = [
    "scheme_1_XGB_strat_y",
    "scheme_1_XGB_strat_y_K",
]

NEW_COLUMN = "algorithm_orientation"

ALL_PREDICTIONS_COLUMN = "all_predictions_sorted_by_prob"

SUBLIST_RANK_COLUMN = "fastest_rank_within_sublist"
SUBLIST_TUPLE_COLUMN = "fastest_tuple_within_sublist"


def derive_algorithm_families(
    prediction_values,
) -> tuple[set[str], set[str]]:
    """
    Derive the Oriented and Forward algorithm families directly
    from the 24 algorithm names in all_predictions_sorted_by_prob.

    Rule:
        algorithm name contains "oriented" -> Oriented
        otherwise                          -> Forward

    Matching is case-insensitive.

    The function strictly validates that the resulting partition is
    exactly 12 Oriented + 12 Forward algorithms.
    """

    all_algorithms: set[str] = set()

    for value in prediction_values:

        predictions = parse_all_predictions(value)

        for _, algorithm, _, _ in predictions:
            all_algorithms.add(
                str(algorithm).strip()
            )

    if len(all_algorithms) != 24:
        raise ValueError(
            "Expected exactly 24 unique algorithms in "
            "all_predictions_sorted_by_prob, but found "
            f"{len(all_algorithms)}:\n"
            f"{sorted(all_algorithms)}"
        )

    oriented_algorithms = {
        algorithm
        for algorithm in all_algorithms
        if "oriented" in algorithm.lower()
    }

    forward_algorithms = (
        all_algorithms
        - oriented_algorithms
    )

    print()
    print("=== ALGORITHM FAMILY DISCOVERY ===")

    print(
        f"Total algorithms:    "
        f"{len(all_algorithms)}"
    )

    print(
        f"Oriented algorithms: "
        f"{len(oriented_algorithms)}"
    )

    for algorithm in sorted(
        oriented_algorithms
    ):
        print(
            f"  [Oriented] {algorithm}"
        )

    print(
        f"Forward algorithms:  "
        f"{len(forward_algorithms)}"
    )

    for algorithm in sorted(
        forward_algorithms
    ):
        print(
            f"  [Forward]  {algorithm}"
        )

    print(
        "=================================="
    )
    print()

    if len(oriented_algorithms) != 12:
        raise ValueError(
            "Automatic family detection expected exactly "
            "12 Oriented algorithms, but found "
            f"{len(oriented_algorithms)}.\n"
            "The rule is: algorithm name contains "
            "'oriented' (case-insensitive)."
        )

    if len(forward_algorithms) != 12:
        raise ValueError(
            "Automatic family detection expected exactly "
            "12 Forward algorithms, but found "
            f"{len(forward_algorithms)}."
        )

    return (
        oriented_algorithms,
        forward_algorithms,
    )


def find_fastest_algorithm_column(df: pd.DataFrame) -> str:
    """
    Resolve the column containing the fastest algorithm.

    The failure-analysis workflow may expose this information using
    slightly different column names, so check the known possibilities.
    """

    candidates = [
        "fastest_algo",
        "fastest_algorithm",
        "fastest_algorithms",
        "fastest_algo_tuple",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not find the fastest-algorithm column.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def classify_orientation(value) -> str:
    """
    Classify the fastest algorithm.

    Any occurrence of 'oriented', regardless of case, is classified
    as Oriented. Everything else is Forward.
    """

    if pd.isna(value):
        return "Forward"

    text = str(value).strip().lower()

    if "oriented" in text:
        return "Oriented"

    return "Forward"

def deduplicate_train_test_failures(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse duplicate entire_data/test_data representations
    of the same underlying failure.

    Within one Scheme 1 XGB experiment, failure identity is:

        dataset + k_value

    Rules
    -----
    - Appears in test_data (and normally entire_data):
          keep one row and label origin = Test
    - Appears only in entire_data:
          keep one row and label origin = Train

    data_scope is removed from the returned dataframe.
    """

    required = {
        "dataset",
        "k_value",
        "data_scope",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Cannot deduplicate failure rows; "
            f"missing columns: {sorted(missing)}"
        )

    working = df.copy()

    # Normalize values.
    working["dataset"] = (
        working["dataset"]
        .astype(str)
        .str.strip()
    )

    working["k_value"] = pd.to_numeric(
        working["k_value"],
        errors="raise",
    )

    working["data_scope"] = (
        working["data_scope"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Determine whether each failure appears in test_data.
    # ---------------------------------------------------------

    test_keys = set(
        zip(
            working.loc[
                working["data_scope"].eq("test_data"),
                "dataset",
            ],
            working.loc[
                working["data_scope"].eq("test_data"),
                "k_value",
            ],
        )
    )

    working["failure_origin"] = [
        (
            "Test"
            if (dataset, k_value) in test_keys
            else "Train"
        )
        for dataset, k_value in zip(
            working["dataset"],
            working["k_value"],
        )
    ]

    # ---------------------------------------------------------
    # Prefer test_data representation when both scopes exist.
    # ---------------------------------------------------------

    scope_priority = {
        "test_data": 0,
        "entire_data": 1,
    }

    working["_scope_priority"] = (
        working["data_scope"]
        .map(scope_priority)
        .fillna(99)
    )

    working = working.sort_values(
        [
            "dataset",
            "k_value",
            "_scope_priority",
        ],
        kind="stable",
    )

    rows_before = len(working)

    working = (
        working
        .drop_duplicates(
            subset=[
                "dataset",
                "k_value",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    rows_after = len(working)

    print(
        f"  rows before scope deduplication: "
        f"{rows_before}"
    )

    print(
        f"  unique failure cases:            "
        f"{rows_after}"
    )

    print(
        f"  duplicate scope rows removed:    "
        f"{rows_before - rows_after}"
    )

    print(
        f"  Test failures: "
        f"{working['failure_origin'].eq('Test').sum()}"
    )

    print(
        f"  Train failures: "
        f"{working['failure_origin'].eq('Train').sum()}"
    )

    # data_scope is no longer needed.
    working = working.drop(
        columns=[
            "data_scope",
            "_scope_priority",
        ],
        errors="ignore",
    )

    return working

def parse_all_predictions(value) -> list[tuple]:
    """
    Parse all_predictions_sorted_by_prob.

    Expected format:

        [
            (time_ratio, algorithm, runtime, probability),
            ...
        ]

    The input is already ordered by descending model probability.
    That ordering is preserved.
    """

    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, list):
        parsed = value
    else:
        text = str(value).strip()

        if not text:
            return []

        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                "Could not parse all_predictions_sorted_by_prob:\n"
                f"{value}"
            ) from exc

    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            "all_predictions_sorted_by_prob must contain "
            "a list/tuple of prediction tuples."
        )

    output = []

    for item in parsed:

        if not isinstance(item, (list, tuple)):
            raise ValueError(
                f"Invalid prediction entry: {item!r}"
            )

        if len(item) < 4:
            raise ValueError(
                "Prediction tuple must contain at least four "
                "elements: "
                "(time_ratio, algorithm, runtime, probability). "
                f"Found: {item!r}"
            )

        time_ratio = item[0]
        algorithm = str(item[1]).strip()
        runtime = item[2]
        probability = item[3]

        output.append(
            (
                time_ratio,
                algorithm,
                runtime,
                probability,
            )
        )

    return output

def extract_fastest_algorithm(
    row: pd.Series,
    fastest_column: str,
) -> str:
    """
    Return the true fastest algorithm name from the row.
    """

    value = row.get(fastest_column)

    if value is None or (
        isinstance(value, float)
        and pd.isna(value)
    ):
        raise ValueError(
            "Missing fastest algorithm value."
        )

    # Plain fastest_algo / fastest_algorithm column.
    if fastest_column in {
        "fastest_algo",
        "fastest_algorithm",
    }:
        return str(value).strip()

    # Tuple-like representation.
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(
                value.strip()
            )
        except (ValueError, SyntaxError):
            parsed = None
    else:
        parsed = value

    if isinstance(parsed, (list, tuple)):

        # Your fastest tuple convention places the algorithm
        # in the second position.
        if len(parsed) >= 2:
            return str(parsed[1]).strip()

    # Last-resort string representation.
    return str(value).strip()

def calculate_fastest_family_rank(
    row: pd.Series,
    fastest_column: str,
    oriented_algorithms: set[str],
    forward_algorithms: set[str],
) -> pd.Series:
    """
    Find the true fastest algorithm's probability rank within
    its corresponding 12-algorithm sublist.

    all_predictions_sorted_by_prob contains 24 tuples already
    sorted by descending model probability:

        (time_ratio, algorithm, runtime, probability)

    For an Oriented failure:
        keep only the 12 Oriented algorithms.

    For a Forward failure:
        keep only the 12 Forward algorithms.

    The original probability ordering is preserved.

    Outputs:
        fastest_rank_within_sublist
        fastest_tuple_within_sublist

    Tuple format:
        (runtime, fastest_algorithm, rank_within_sublist)
    """

    fastest_algorithm = extract_fastest_algorithm(
        row,
        fastest_column,
    )

    orientation = row.get(NEW_COLUMN)

    predictions = parse_all_predictions(
        row.get(ALL_PREDICTIONS_COLUMN)
    )

    if len(predictions) != 24:
        raise ValueError(
            f"Expected 24 predictions for "
            f"dataset={row.get('dataset')}, "
            f"k={row.get('k_value')}; "
            f"found {len(predictions)}."
        )

    # ---------------------------------------------------------
    # Select the correct 12-algorithm family.
    # ---------------------------------------------------------

    if orientation == "Oriented":
        family_algorithms = oriented_algorithms

    elif orientation == "Forward":
        family_algorithms = forward_algorithms

    else:
        raise ValueError(
            f"Unknown orientation: {orientation!r}"
        )

    # ---------------------------------------------------------
    # all_predictions_sorted_by_prob is already sorted by
    # descending probability.
    #
    # Filtering it therefore gives the probability ranking
    # within the corresponding 12-algorithm family.
    # ---------------------------------------------------------

    family_predictions = [
        prediction
        for prediction in predictions
        if prediction[1] in family_algorithms
    ]

    if len(family_predictions) != 12:
        raise ValueError(
            f"Expected exactly 12 {orientation} algorithms for "
            f"dataset={row.get('dataset')}, "
            f"k={row.get('k_value')}; "
            f"found {len(family_predictions)}."
        )

    # ---------------------------------------------------------
    # Find the TRUE fastest algorithm within that probability-
    # ordered family.
    # ---------------------------------------------------------

    for rank, prediction in enumerate(
        family_predictions,
        start=1,
    ):
        _, algorithm, runtime, _ = prediction

        if algorithm != fastest_algorithm:
            continue

        return pd.Series(
            {
                SUBLIST_RANK_COLUMN: rank,

                SUBLIST_TUPLE_COLUMN: (
                    f"({float(runtime):.4f}, "
                    f"{algorithm!r}, "
                    f"{rank})"
                ),
            }
        )

    raise ValueError(
        f"Fastest algorithm {fastest_algorithm!r} "
        f"was not found in the {orientation} sublist "
        f"for dataset={row.get('dataset')}, "
        f"k={row.get('k_value')}."
    )

def process_sheet(
    input_path: Path,
    sheet_name: str,
) -> pd.DataFrame:

    print(f"Reading: {sheet_name}")

    df = pd.read_excel(
        input_path,
        sheet_name=sheet_name,
    )

    # Collapse test_data + entire_data representations
    # of the same dataset/K failure.
    df = deduplicate_train_test_failures(df)

    fastest_column = find_fastest_algorithm_column(df)

    print(
        f"  fastest algorithm column: "
        f"{fastest_column}"
    )

    # Put the new column immediately after the fastest-algorithm column.
    if NEW_COLUMN in df.columns:
        df = df.drop(columns=[NEW_COLUMN])

    insert_position = (
        df.columns.get_loc(fastest_column) + 1
    )

    df.insert(
        insert_position,
        NEW_COLUMN,
        df[fastest_column].apply(
            classify_orientation
        ),
    )
    # ---------------------------------------------------------
    # Calculate probability rank of the true fastest algorithm
    # within its corresponding 12-algorithm family.
    # ---------------------------------------------------------

    if ALL_PREDICTIONS_COLUMN not in df.columns:
        raise ValueError(
            f"Required column {ALL_PREDICTIONS_COLUMN!r} "
            f"is missing from {sheet_name}."
        )


    # Remove derived columns if processing an existing output again.
    for column in (
        SUBLIST_RANK_COLUMN,
        SUBLIST_TUPLE_COLUMN,
    ):
        if column in df.columns:
            df = df.drop(columns=[column])

    # Discover the 12 + 12 algorithm families.
    oriented_algorithms, forward_algorithms = (
        derive_algorithm_families(
            df[ALL_PREDICTIONS_COLUMN]
        )
    )

    # Calculate the true fastest algorithm's probability rank
    # within the appropriate 12-algorithm family.
    family_rank_df = df.apply(
        lambda row: calculate_fastest_family_rank(
            row=row,
            fastest_column=fastest_column,
            oriented_algorithms=oriented_algorithms,
            forward_algorithms=forward_algorithms,
        ),
        axis=1,
    )

    # Insert immediately after algorithm_orientation.
    insert_position = (
        df.columns.get_loc(NEW_COLUMN) + 1
    )

    df.insert(
        insert_position,
        SUBLIST_RANK_COLUMN,
        family_rank_df[SUBLIST_RANK_COLUMN],
    )

    df.insert(
        insert_position + 1,
        SUBLIST_TUPLE_COLUMN,
        family_rank_df[SUBLIST_TUPLE_COLUMN],
    )
        
    oriented_count = int(
        df[NEW_COLUMN].eq("Oriented").sum()
    )

    forward_count = int(
        df[NEW_COLUMN].eq("Forward").sum()
    )

    print(f"  total:    {len(df)}")
    print(f"  Oriented: {oriented_count}")
    print(f"  Forward:  {forward_count}")

    return df


def build_summary(
    sheet_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    rows = []

    for sheet_name, df in sheet_frames.items():

        if sheet_name.endswith("_strat_y_K"):
            split_mode = "strat_y_K"
        else:
            split_mode = "strat_y"

        total = len(df)

        test_mask = (
            df["failure_origin"].eq("Test")
        )

        train_mask = (
            df["failure_origin"].eq("Train")
        )

        oriented_mask = (
            df[NEW_COLUMN].eq("Oriented")
        )

        forward_mask = (
            df[NEW_COLUMN].eq("Forward")
        )

        rows.append(
            {
                "scheme": "scheme 1",
                "model": "XGB",
                "split_mode": split_mode,

                "unique_failure_cases":
                    total,

                "test_failures":
                    int(test_mask.sum()),

                "train_failures":
                    int(train_mask.sum()),

                "oriented_failures":
                    int(oriented_mask.sum()),

                "forward_failures":
                    int(forward_mask.sum()),

                "test_oriented":
                    int(
                        (
                            test_mask
                            & oriented_mask
                        ).sum()
                    ),

                "test_forward":
                    int(
                        (
                            test_mask
                            & forward_mask
                        ).sum()
                    ),

                "train_oriented":
                    int(
                        (
                            train_mask
                            & oriented_mask
                        ).sum()
                    ),

                "train_forward":
                    int(
                        (
                            train_mask
                            & forward_mask
                        ).sum()
                    ),

                "oriented_percent":
                    round(
                        100.0
                        * oriented_mask.sum()
                        / total,
                        2,
                    )
                    if total
                    else 0.0,
            }
        )

    return pd.DataFrame(rows)


def format_workbook(
    output_path: Path,
) -> None:

    workbook = load_workbook(output_path)

    for worksheet in workbook.worksheets:

        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = (
            worksheet.dimensions
        )

        for column_cells in worksheet.columns:

            values = [
                str(cell.value)
                for cell in column_cells
                if cell.value is not None
            ]

            max_length = max(
                (len(value) for value in values),
                default=10,
            )

            worksheet.column_dimensions[
                column_cells[0].column_letter
            ].width = min(
                max(max_length + 2, 12),
                45,
            )

    workbook.save(output_path)
    workbook.close()


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract Scheme 1 XGB failure sheets and classify "
            "the fastest algorithm as Oriented or Forward."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input failure-analysis workbook.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output Scheme 1 XGB workbook.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input workbook not found: {args.input}"
        )

    available_sheets = pd.ExcelFile(
        args.input
    ).sheet_names

    missing = [
        sheet
        for sheet in SHEETS
        if sheet not in available_sheets
    ]

    if missing:
        raise ValueError(
            f"Required Scheme 1 XGB sheets are missing: {missing}\n"
            f"Available sheets: {available_sheets}"
        )

    sheet_frames = {}

    print()
    print("=" * 60)
    print("SCHEME 1 XGB ORIENTATION ANALYSIS")
    print("=" * 60)

    for sheet_name in SHEETS:

        sheet_frames[sheet_name] = (
            process_sheet(
                args.input,
                sheet_name,
            )
        )

    summary_df = build_summary(
        sheet_frames
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        args.output,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        for sheet_name, df in sheet_frames.items():

            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

    format_workbook(
        args.output
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Output workbook:\n"
        f"  {args.output}"
    )


if __name__ == "__main__":
    main()