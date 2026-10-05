"""Population-normalized train/test wrong-prediction and important-failure analysis.

For each Scheme x Model x Split configuration, report:

    - actual training population size
    - actual held-out test population size
    - all wrong predictions in the training partition
    - all wrong predictions in the test partition
    - important failures in the training partition
    - important failures in the test partition
    - normalized wrong-prediction rates
    - normalized important-failure rates
    - test/train important-failure-rate multiplier

Important failures are taken from the existing enriched failure-analysis
workflow. This program does NOT redefine the important-failure criterion.

The program uses saved train/test indices as the authoritative source of
partition membership.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font


# ============================================================================
# Repository import setup
# ============================================================================

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
    FAILURE_ID_COLUMNS,
    deduplicate_failure_rows,
    load_important_filtered_rows,
    prepare_unfiltered_frames,
    resolve_existing_path,
)


# ============================================================================
# Constants
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROJECT_CONFIG = Path(
    "restructured_pipeline/configs/project.yaml"
)

DEFAULT_SPLITS_ROOT = Path(
    "restructured_pipeline/data/splits"
)

DEFAULT_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/"
    "failure_rate_population_analysis.xlsx"
)


ACTIVE_SCHEMES = (
    "scheme 1",
    "scheme 3",
)

ACTIVE_MODELS = (
    "RF",
    "XGB",
)

ACTIVE_SPLITS = (
    "strat_y",
    "strat_y_K",
)


SUMMARY_SHEET = "population_failure_summary"
DETAIL_SHEET = "population_failure_details"
VALIDATION_SHEET = "validation"
NOTES_SHEET = "analysis_notes"


# ============================================================================
# Experiment definitions
# ============================================================================

EXPERIMENTS = [
    {
        "scheme": "scheme 1",
        "model": "RF",
        "split_mode": "strat_y",
        "scheme_dir": "scheme1",
        "model_dir": "random_forest",
        "run_prefix": "scheme1_random_forest_rf_strat_y",
    },
    {
        "scheme": "scheme 1",
        "model": "RF",
        "split_mode": "strat_y_K",
        "scheme_dir": "scheme1",
        "model_dir": "random_forest",
        "run_prefix": "scheme1_random_forest_rf_strat_y_K",
    },
    {
        "scheme": "scheme 1",
        "model": "XGB",
        "split_mode": "strat_y",
        "scheme_dir": "scheme1",
        "model_dir": "xgboost",
        "run_prefix": "scheme1_xgboost_xgb_strat_y",
    },
    {
        "scheme": "scheme 1",
        "model": "XGB",
        "split_mode": "strat_y_K",
        "scheme_dir": "scheme1",
        "model_dir": "xgboost",
        "run_prefix": "scheme1_xgboost_xgb_strat_y_K",
    },
    {
        "scheme": "scheme 3",
        "model": "RF",
        "split_mode": "strat_y",
        "scheme_dir": "scheme3",
        "model_dir": "random_forest",
        "run_prefix": "scheme3_random_forest_strat_y",
    },
    {
        "scheme": "scheme 3",
        "model": "RF",
        "split_mode": "strat_y_K",
        "scheme_dir": "scheme3",
        "model_dir": "random_forest",
        "run_prefix": "scheme3_random_forest_strat_y_K",
    },
    {
        "scheme": "scheme 3",
        "model": "XGB",
        "split_mode": "strat_y",
        "scheme_dir": "scheme3",
        "model_dir": "xgboost",
        "run_prefix": "scheme3_xgboost_strat_y",
    },
    {
        "scheme": "scheme 3",
        "model": "XGB",
        "split_mode": "strat_y_K",
        "scheme_dir": "scheme3",
        "model_dir": "xgboost",
        "run_prefix": "scheme3_xgboost_strat_y_K",
    },
]


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
            "Current enriched important-failure workbook, e.g. "
            "failure_analysis_summary_scheme1_scheme3_05sep2026.xlsx"
        ),
    )

    parser.add_argument(
        "--run-tag",
        required=True,
        help=(
            "Training/prediction run tag used in saved split-index "
            "and wrong-prediction filenames, e.g. 05sep2026."
        ),
    )

    parser.add_argument(
        "--splits-root",
        type=Path,
        default=DEFAULT_SPLITS_ROOT,
        help="Root directory containing saved train/test index CSV files.",
    )

    parser.add_argument(
        "--project-config-path",
        type=Path,
        default=DEFAULT_PROJECT_CONFIG,
    )

    parser.add_argument(
        "--entire-raw-glob",
        default=None,
        help=(
            "Optional explicit entire-data wrong-prediction glob. "
            "If omitted, a run-tag-specific glob is generated."
        ),
    )

    parser.add_argument(
        "--test-raw-glob",
        default=None,
        help=(
            "Optional explicit test-data wrong-prediction glob. "
            "If omitted, a run-tag-specific glob is generated."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    return parser.parse_args()


# ============================================================================
# Path helpers
# ============================================================================

def resolve_repo_path(path: Path) -> Path:

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def resolve_split_index_path(
    splits_root: Path,
    experiment: dict[str, str],
    run_tag: str,
    partition: str,
) -> Path:

    if partition not in {
        "train",
        "test",
    }:
        raise ValueError(
            f"Unsupported partition: {partition}"
        )

    run_name = (
        f"{experiment['run_prefix']}_"
        f"{run_tag}"
    )

    filename = (
        f"{partition}_indices_"
        f"{run_name}.csv"
    )

    return (
        splits_root
        / experiment["scheme_dir"]
        / experiment["model_dir"]
        / filename
    )


# ============================================================================
# Saved split membership
# ============================================================================

def load_index_file(
    path: Path,
    label: str,
) -> dict[str, object]:
    """Load a saved split-index file at original-source-row granularity.

    Scheme 1 may contain repeated source_row_index values because a raw row can
    be expanded into multiple supervised observations when several algorithms
    are tied. Those repetitions are valid within one partition. For population
    analysis, however, each original source row counts once.
    """

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    df = pd.read_csv(path)
    if "source_row_index" not in df.columns:
        raise ValueError(f"{path} does not contain 'source_row_index'.")

    indices = pd.to_numeric(df["source_row_index"], errors="raise").astype(int)
    expanded_row_count = int(len(indices))
    unique_indices = set(indices.tolist())
    unique_row_count = int(len(unique_indices))
    duplicate_expansion_count = expanded_row_count - unique_row_count

    duplicate_source_rows = (
        indices[indices.duplicated(keep=False)]
        .value_counts()
        .sort_values(ascending=False)
    )

    print(
        f"{label}: expanded_rows={expanded_row_count}, "
        f"unique_source_rows={unique_row_count}, "
        f"extra_expanded_rows={duplicate_expansion_count}"
    )

    if not duplicate_source_rows.empty:
        print(
            f"{label}: {len(duplicate_source_rows)} source rows occur multiple "
            "times because of tie expansion."
        )
        print(f"{label}: most repeated source rows:")
        for source_idx, count in duplicate_source_rows.head(10).items():
            print(
                f"  source_row_index={int(source_idx)} "
                f"occurrences={int(count)}"
            )

    return {
        "indices": unique_indices,
        "expanded_row_count": expanded_row_count,
        "unique_row_count": unique_row_count,
        "duplicate_expansion_count": int(duplicate_expansion_count),
        "duplicate_source_row_count": int(len(duplicate_source_rows)),
    }


def load_experiment_membership(
    splits_root: Path,
    experiment: dict[str, str],
    run_tag: str,
) -> dict[str, object]:
    """Load authoritative train/test membership and detect source-row leakage.

    Repeated indices *within* train or test are collapsed to unique original
    rows. An original row appearing in both partitions is not collapsed away:
    it is treated as data leakage and aborts the analysis.
    """

    train_path = resolve_split_index_path(
        splits_root, experiment, run_tag, "train"
    )
    test_path = resolve_split_index_path(
        splits_root, experiment, run_tag, "test"
    )

    train_info = load_index_file(train_path, "training-index file")
    test_info = load_index_file(test_path, "test-index file")

    train_indices = train_info["indices"]
    test_indices = test_info["indices"]
    overlap = train_indices & test_indices

    if overlap:
        overlap_examples = sorted(overlap)[:25]
        raise ValueError(
            "\nDATA LEAKAGE DETECTED.\n"
            f"Experiment: {experiment['scheme']} / {experiment['model']} / "
            f"{experiment['split_mode']}\n"
            f"{len(overlap)} original source rows occur in BOTH train and test "
            "after collapsing tie-expanded rows.\n"
            f"Examples: {overlap_examples}\n\n"
            "The population-normalized train/test analysis is not valid for "
            "this run. Do not remove the overlap manually; fix the split so "
            "original rows are partitioned before tie expansion, retrain, and "
            "regenerate inference/failure outputs."
        )

    train_size = len(train_indices)
    test_size = len(test_indices)
    total_size = train_size + test_size

    print(
        f"Unique original population: train={train_size}, "
        f"test={test_size}, total={total_size}"
    )

    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_size": int(train_size),
        "test_size": int(test_size),
        "total_size": int(total_size),
        "train_expanded_size": int(train_info["expanded_row_count"]),
        "test_expanded_size": int(test_info["expanded_row_count"]),
        "train_extra_expanded_rows": int(train_info["duplicate_expansion_count"]),
        "test_extra_expanded_rows": int(test_info["duplicate_expansion_count"]),
        "train_duplicate_source_rows": int(train_info["duplicate_source_row_count"]),
        "test_duplicate_source_rows": int(test_info["duplicate_source_row_count"]),
        "train_test_source_overlap": 0,
        "train_indices_path": str(train_path),
        "test_indices_path": str(test_path),
    }


# ============================================================================
# Map dataset/K to authoritative source_row_index
# ============================================================================

def build_dataset_k_to_source_index(
    project_config_path: Path,
) -> pd.DataFrame:
    """
    Use the same raw/preprocessed dataset ordering used by the pipeline to
    establish:

        dataset + k_value -> source_row_index

    We import the training loader rather than independently rebuilding the
    dataset.
    """

    from restructured_pipeline.src.train import (
        load_and_preprocess_data,
        load_yaml_config,
    )

    project_config = load_yaml_config(
        project_config_path
    )

    # Scheme 1 is sufficient for dataset/K/source-index identity.
    data = load_and_preprocess_data(
        project_config,
        "scheme1",
    )

    required = {
        "dataset",
        "k_value",
        "source_row_index",
    }

    missing = required - set(
        data.columns
    )

    if missing:
        raise ValueError(
            "Preprocessed dataset missing columns: "
            f"{sorted(missing)}"
        )

    lookup = data[
        [
            "dataset",
            "k_value",
            "source_row_index",
        ]
    ].copy()

    lookup["dataset"] = (
        lookup["dataset"]
        .astype(str)
    )

    lookup["k_value"] = pd.to_numeric(
        lookup["k_value"],
        errors="raise",
    ).astype(int)

    lookup["source_row_index"] = pd.to_numeric(
        lookup["source_row_index"],
        errors="raise",
    ).astype(int)

    duplicated = lookup.duplicated(
        subset=[
            "dataset",
            "k_value",
        ],
        keep=False,
    )

    if duplicated.any():

        examples = (
            lookup.loc[
                duplicated,
                [
                    "dataset",
                    "k_value",
                    "source_row_index",
                ],
            ]
            .head(20)
            .to_dict(
                orient="records"
            )
        )

        raise ValueError(
            "dataset + k_value is not unique in "
            "the source dataset. Cannot safely use "
            "it to recover split membership. "
            f"Examples: {examples}"
        )

    return lookup


def attach_source_row_index(
    df: pd.DataFrame,
    source_lookup: pd.DataFrame,
    label: str,
) -> pd.DataFrame:

    working = df.copy()

    if "source_row_index" in working.columns:

        source_index = pd.to_numeric(
            working[
                "source_row_index"
            ],
            errors="coerce",
        )

        if source_index.notna().all():

            working[
                "source_row_index"
            ] = source_index.astype(int)

            return working

    required = {
        "dataset",
        "k_value",
    }

    missing = required - set(
        working.columns
    )

    if missing:
        raise ValueError(
            f"{label} cannot recover "
            "source_row_index; missing "
            f"{sorted(missing)}"
        )

    working["dataset"] = (
        working["dataset"]
        .astype(str)
    )

    working["k_value"] = pd.to_numeric(
        working["k_value"],
        errors="raise",
    ).astype(int)

    # Remove an incomplete pre-existing column before merge.
    if "source_row_index" in working.columns:
        working = working.drop(
            columns=[
                "source_row_index"
            ]
        )

    working = working.merge(
        source_lookup,
        on=[
            "dataset",
            "k_value",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_index = (
        working[
            "source_row_index"
        ].isna()
    )

    if missing_index.any():

        examples = (
            working.loc[
                missing_index,
                [
                    "dataset",
                    "k_value",
                ],
            ]
            .head(20)
            .to_dict(
                orient="records"
            )
        )

        raise ValueError(
            f"Could not recover source_row_index "
            f"for {int(missing_index.sum())} "
            f"{label} rows. "
            f"Examples: {examples}"
        )

    working[
        "source_row_index"
    ] = pd.to_numeric(
        working[
            "source_row_index"
        ],
        errors="raise",
    ).astype(int)

    return working


# ============================================================================
# Partition assignment using saved indices
# ============================================================================

def assign_partition_from_indices(
    df: pd.DataFrame,
    train_indices: set[int],
    test_indices: set[int],
    label: str,
) -> pd.DataFrame:

    working = df.copy()

    if "source_row_index" not in working.columns:
        raise ValueError(
            f"{label} missing source_row_index."
        )

    def resolve_partition(
        source_index: int,
    ) -> str:

        if source_index in train_indices:
            return "train"

        if source_index in test_indices:
            return "test"

        return "unknown"

    working["partition"] = (
        working[
            "source_row_index"
        ]
        .astype(int)
        .map(resolve_partition)
    )

    return working


# ============================================================================
# Experiment filtering
# ============================================================================

def select_experiment_rows(
    df: pd.DataFrame,
    experiment: dict[str, str],
) -> pd.DataFrame:

    return df[
        df["scheme"].eq(
            experiment["scheme"]
        )
        & df["model"].eq(
            experiment["model"]
        )
        & df["split_mode"].eq(
            experiment["split_mode"]
        )
    ].copy()


# ============================================================================
# Rates
# ============================================================================

def safe_rate(
    numerator: int,
    denominator: int,
) -> float | None:

    if denominator <= 0:
        return None

    return float(
        numerator / denominator
    )


def safe_multiplier(
    test_rate: float | None,
    train_rate: float | None,
) -> float | None:

    if (
        test_rate is None
        or train_rate is None
    ):
        return None

    if train_rate == 0:

        if test_rate == 0:
            return 1.0

        return float("inf")

    return float(
        test_rate / train_rate
    )


# ============================================================================
# Main analysis
# ============================================================================

def build_population_failure_analysis(
    important_df: pd.DataFrame,
    entire_wrong_df: pd.DataFrame,
    test_wrong_df: pd.DataFrame,
    source_lookup: pd.DataFrame,
    splits_root: Path,
    run_tag: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    summary_rows = []
    detail_frames = []
    validation_rows = []

    # Important workbook contains entire-data important failures only
    # after load_important_filtered_rows().
    important_df = attach_source_row_index(
        important_df,
        source_lookup,
        "important-failure",
    )

    entire_wrong_df = attach_source_row_index(
        entire_wrong_df,
        source_lookup,
        "entire-data wrong-prediction",
    )

    test_wrong_df = attach_source_row_index(
        test_wrong_df,
        source_lookup,
        "test-data wrong-prediction",
    )

    for experiment in EXPERIMENTS:

        scheme = experiment["scheme"]
        model = experiment["model"]
        split_mode = experiment[
            "split_mode"
        ]

        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            f"{scheme} / {model} / "
            f"{split_mode}"
        )

        membership = (
            load_experiment_membership(
                splits_root,
                experiment,
                run_tag,
            )
        )

        train_indices = membership[
            "train_indices"
        ]

        test_indices = membership[
            "test_indices"
        ]

        train_size = int(
            membership["train_size"]
        )

        test_size = int(
            membership["test_size"]
        )

        total_size = int(
            membership["total_size"]
        )

        # -----------------------------------------------------
        # ALL WRONG PREDICTIONS
        #
        # Use entire-data wrong predictions as canonical source,
        # then assign them directly to saved train/test indices.
        # -----------------------------------------------------

        wrong_rows = (
            select_experiment_rows(
                entire_wrong_df,
                experiment,
            )
        )

        wrong_rows = (
            assign_partition_from_indices(
                wrong_rows,
                train_indices,
                test_indices,
                "wrong-prediction",
            )
        )

        # -----------------------------------------------------
        # IMPORTANT FAILURES
        # -----------------------------------------------------

        important_rows = (
            select_experiment_rows(
                important_df,
                experiment,
            )
        )

        important_rows = (
            assign_partition_from_indices(
                important_rows,
                train_indices,
                test_indices,
                "important-failure",
            )
        )

        # -----------------------------------------------------
        # Counts
        # -----------------------------------------------------

        train_wrong = int(
            wrong_rows[
                "partition"
            ].eq("train").sum()
        )

        test_wrong = int(
            wrong_rows[
                "partition"
            ].eq("test").sum()
        )

        unknown_wrong = int(
            wrong_rows[
                "partition"
            ].eq("unknown").sum()
        )

        total_wrong = int(
            len(wrong_rows)
        )

        train_important = int(
            important_rows[
                "partition"
            ].eq("train").sum()
        )

        test_important = int(
            important_rows[
                "partition"
            ].eq("test").sum()
        )

        unknown_important = int(
            important_rows[
                "partition"
            ].eq("unknown").sum()
        )

        total_important = int(
            len(important_rows)
        )

        # -----------------------------------------------------
        # Normalized rates
        # -----------------------------------------------------

        train_wrong_rate = safe_rate(
            train_wrong,
            train_size,
        )

        test_wrong_rate = safe_rate(
            test_wrong,
            test_size,
        )

        train_important_rate = safe_rate(
            train_important,
            train_size,
        )

        test_important_rate = safe_rate(
            test_important,
            test_size,
        )

        important_multiplier = (
            safe_multiplier(
                test_important_rate,
                train_important_rate,
            )
        )

        wrong_multiplier = (
            safe_multiplier(
                test_wrong_rate,
                train_wrong_rate,
            )
        )

        # -----------------------------------------------------
        # Important among wrong predictions
        # -----------------------------------------------------

        train_important_given_wrong = (
            safe_rate(
                train_important,
                train_wrong,
            )
        )

        test_important_given_wrong = (
            safe_rate(
                test_important,
                test_wrong,
            )
        )

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        if (
            train_wrong
            + test_wrong
            + unknown_wrong
            != total_wrong
        ):
            raise ValueError(
                f"Wrong-prediction count mismatch "
                f"for {scheme}/{model}/{split_mode}."
            )

        if (
            train_important
            + test_important
            + unknown_important
            != total_important
        ):
            raise ValueError(
                f"Important-failure count mismatch "
                f"for {scheme}/{model}/{split_mode}."
            )

        if train_important > train_wrong:
            raise ValueError(
                f"Train important failures exceed "
                f"train wrong predictions for "
                f"{scheme}/{model}/{split_mode}."
            )

        if test_important > test_wrong:
            raise ValueError(
                f"Test important failures exceed "
                f"test wrong predictions for "
                f"{scheme}/{model}/{split_mode}."
            )

        if train_wrong > train_size:
            raise ValueError(
                f"Train wrong predictions exceed "
                f"train population for "
                f"{scheme}/{model}/{split_mode}."
            )

        if test_wrong > test_size:
            raise ValueError(
                f"Test wrong predictions exceed "
                f"test population for "
                f"{scheme}/{model}/{split_mode}."
            )

        # -----------------------------------------------------
        # Summary row
        # -----------------------------------------------------

        summary_rows.append(
            {
                "Scheme":
                    scheme,

                "Model":
                    model,

                "Split":
                    split_mode,

                "Train Population":
                    train_size,

                "Test Population":
                    test_size,

                "Total Population":
                    total_size,

                "Train Wrong Predictions":
                    train_wrong,

                "Test Wrong Predictions":
                    test_wrong,

                "Total Wrong Predictions":
                    total_wrong,

                "Train Important Failures":
                    train_important,

                "Test Important Failures":
                    test_important,

                "Total Important Failures":
                    total_important,

                "Train Wrong Rate":
                    train_wrong_rate,

                "Test Wrong Rate":
                    test_wrong_rate,

                "Test/Train Wrong Rate Multiplier":
                    wrong_multiplier,

                "Train Important Failure Rate":
                    train_important_rate,

                "Test Important Failure Rate":
                    test_important_rate,

                "Test/Train Important Failure Rate Multiplier":
                    important_multiplier,

                "Train Important / Wrong":
                    train_important_given_wrong,

                "Test Important / Wrong":
                    test_important_given_wrong,

                "Unknown Wrong Predictions":
                    unknown_wrong,

                "Unknown Important Failures":
                    unknown_important,
            }
        )

        # -----------------------------------------------------
        # Detailed important rows
        # -----------------------------------------------------

        if not important_rows.empty:

            detail = (
                important_rows.copy()
            )

            detail.insert(
                0,
                "analysis_split",
                split_mode,
            )

            detail.insert(
                0,
                "analysis_model",
                model,
            )

            detail.insert(
                0,
                "analysis_scheme",
                scheme,
            )

            detail_frames.append(
                detail
            )

        # -----------------------------------------------------
        # Validation row
        # -----------------------------------------------------

        validation_rows.append(
            {
                "Scheme":
                    scheme,

                "Model":
                    model,

                "Split":
                    split_mode,

                "Train Index Count":
                    train_size,

                "Test Index Count":
                    test_size,

                "Index Total":
                    total_size,

                "Train Expanded Rows":
                    membership["train_expanded_size"],

                "Test Expanded Rows":
                    membership["test_expanded_size"],

                "Train Unique Source Rows":
                    train_size,

                "Test Unique Source Rows":
                    test_size,

                "Train Extra Tie-Expanded Rows":
                    membership["train_extra_expanded_rows"],

                "Test Extra Tie-Expanded Rows":
                    membership["test_extra_expanded_rows"],

                "Train Duplicate Source Rows":
                    membership["train_duplicate_source_rows"],

                "Test Duplicate Source Rows":
                    membership["test_duplicate_source_rows"],

                "Train/Test Source Overlap":
                    membership["train_test_source_overlap"],

                "Wrong Unknown":
                    unknown_wrong,

                "Important Unknown":
                    unknown_important,

                "Train+Test Wrong = Total":
                    (
                        train_wrong
                        + test_wrong
                        + unknown_wrong
                        == total_wrong
                    ),

                "Train+Test Important = Total":
                    (
                        train_important
                        + test_important
                        + unknown_important
                        == total_important
                    ),

                "Important <= Wrong":
                    (
                        train_important
                        <= train_wrong
                        and test_important
                        <= test_wrong
                    ),

                "Train Indices":
                    membership[
                        "train_indices_path"
                    ],

                "Test Indices":
                    membership[
                        "test_indices_path"
                    ],
            }
        )

        print(
            f"Population: "
            f"train={train_size}, "
            f"test={test_size}, "
            f"total={total_size}"
        )

        print(
            f"Wrong predictions: "
            f"train={train_wrong}, "
            f"test={test_wrong}, "
            f"total={total_wrong}"
        )

        print(
            f"Important failures: "
            f"train={train_important}, "
            f"test={test_important}, "
            f"total={total_important}"
        )

        print(
            "Important failure rates: "
            f"train="
            f"{train_important_rate:.4%}"
            if train_important_rate
            is not None
            else "train=NA"
        )

        print(
            "Test important failure rate: "
            f"{test_important_rate:.4%}"
            if test_important_rate
            is not None
            else "NA"
        )

        print(
            "Test/train important-failure "
            f"rate multiplier: "
            f"{important_multiplier}"
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    detail_df = (
        pd.concat(
            detail_frames,
            ignore_index=True,
            sort=False,
        )
        if detail_frames
        else pd.DataFrame()
    )

    validation_df = pd.DataFrame(
        validation_rows
    )

    return (
        summary_df,
        detail_df,
        validation_df,
    )


# ============================================================================
# Notes
# ============================================================================

def build_notes(
    important_workbook: Path,
    run_tag: str,
    entire_raw_glob: str,
    test_raw_glob: str,
) -> pd.DataFrame:

    rows = [
        (
            "important_workbook",
            str(important_workbook),
        ),
        (
            "run_tag",
            run_tag,
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
            "population_source",
            (
                "Saved train_indices and test_indices "
                "CSV files generated during model training"
            ),
        ),
        (
            "partition_assignment",
            (
                "source_row_index is matched directly "
                "against the saved train/test index sets"
            ),
        ),
        (
            "population_granularity",
            (
                "Population sizes count unique original source_row_index values; "
                "repeated Scheme 1 tie-expanded rows within a partition count once"
            ),
        ),
        (
            "leakage_validation",
            (
                "Analysis aborts if any original source_row_index occurs in both "
                "saved train and test index files"
            ),
        ),
        (
            "wrong_prediction_source",
            (
                "Entire-data wrong-prediction exports; "
                "rows are assigned to train/test by saved indices"
            ),
        ),
        (
            "important_failure_source",
            (
                "Current enriched important-failure workbook; "
                "the important-failure criterion is inherited "
                "from the upstream workflow"
            ),
        ),
        (
            "important_failure_definition",
            (
                "Wrong prediction retained by upstream important-failure "
                "filter: runtime > 5 seconds and runtime ratio > 1.20"
            ),
        ),
        (
            "train_wrong_rate",
            "train wrong predictions / train population",
        ),
        (
            "test_wrong_rate",
            "test wrong predictions / test population",
        ),
        (
            "train_important_failure_rate",
            "train important failures / train population",
        ),
        (
            "test_important_failure_rate",
            "test important failures / test population",
        ),
        (
            "important_failure_rate_multiplier",
            (
                "test important-failure rate / "
                "train important-failure rate"
            ),
        ),
        (
            "important_given_wrong",
            (
                "important failures / all wrong predictions "
                "within the same partition"
            ),
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
# Excel output
# ============================================================================

def write_workbook(
    output_path: Path,
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    notes_df: pd.DataFrame,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name=SUMMARY_SHEET,
            index=False,
        )

        detail_df.to_excel(
            writer,
            sheet_name=DETAIL_SHEET,
            index=False,
        )

        validation_df.to_excel(
            writer,
            sheet_name=VALIDATION_SHEET,
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

    # ---------------------------------------------------------
    # Format all worksheets.
    # ---------------------------------------------------------

    for sheet_name in workbook.sheetnames:

        ws = workbook[
            sheet_name
        ]

        ws.freeze_panes = "A2"

        ws.auto_filter.ref = (
            ws.dimensions
        )

        autosize_columns(
            ws
        )

        for cell in ws[1]:
            cell.font = Font(
                bold=True
            )

    # ---------------------------------------------------------
    # Percentage formatting on summary.
    # ---------------------------------------------------------

    ws = workbook[
        SUMMARY_SHEET
    ]

    headers = {
        cell.value:
            cell.column
        for cell in ws[1]
    }

    percentage_columns = [
        "Train Wrong Rate",
        "Test Wrong Rate",
        "Train Important Failure Rate",
        "Test Important Failure Rate",
        "Train Important / Wrong",
        "Test Important / Wrong",
    ]

    for column_name in percentage_columns:

        if column_name not in headers:
            continue

        column_idx = headers[
            column_name
        ]

        for row_idx in range(
            2,
            ws.max_row + 1,
        ):

            ws.cell(
                row=row_idx,
                column=column_idx,
            ).number_format = "0.00%"

    multiplier_columns = [
        "Test/Train Wrong Rate Multiplier",
        (
            "Test/Train Important Failure "
            "Rate Multiplier"
        ),
    ]

    for column_name in multiplier_columns:

        if column_name not in headers:
            continue

        column_idx = headers[
            column_name
        ]

        for row_idx in range(
            2,
            ws.max_row + 1,
        ):

            ws.cell(
                row=row_idx,
                column=column_idx,
            ).number_format = "0.00x"

    workbook.save(
        output_path
    )

    workbook.close()


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    args = parse_args()

    important_workbook = (
        resolve_existing_path(
            args.important_workbook,
            DEFAULT_IMPORTANT_CANDIDATES,
            "important workbook",
        )
    )

    splits_root = resolve_repo_path(
        args.splits_root
    )

    project_config_path = (
        resolve_repo_path(
            args.project_config_path
        )
    )

    output_path = resolve_repo_path(
        args.output_path
    )

    # ---------------------------------------------------------
    # Restrict wrong-prediction discovery to this exact run.
    # ---------------------------------------------------------

    entire_raw_glob = (
        args.entire_raw_glob
        or (
            "restructured_pipeline/outputs/"
            "wrong_cases/*/*/"
            f"wrong_predictions_*{args.run_tag}*"
            "_predict_inference_entire_data.xlsx"
        )
    )

    test_raw_glob = (
        args.test_raw_glob
        or (
            "restructured_pipeline/outputs/"
            "wrong_cases/*/*/"
            f"wrong_predictions_*{args.run_tag}*"
            "_predict_inference_test_data.xlsx"
        )
    )

    print(
        "============================================================"
    )

    print(
        "POPULATION-NORMALIZED FAILURE ANALYSIS"
    )

    print(
        "============================================================"
    )

    print(
        f"Run tag: {args.run_tag}"
    )

    print(
        f"Important workbook: "
        f"{important_workbook}"
    )

    print(
        f"Splits root: "
        f"{splits_root}"
    )

    print(
        f"Entire wrong glob: "
        f"{entire_raw_glob}"
    )

    print(
        f"Test wrong glob: "
        f"{test_raw_glob}"
    )

    # ---------------------------------------------------------
    # Important failures.
    #
    # This loader returns canonical entire_data important rows.
    # ---------------------------------------------------------

    important_df = (
        load_important_filtered_rows(
            important_workbook
        )
    )

    # ---------------------------------------------------------
    # Raw wrong-prediction exports.
    # ---------------------------------------------------------

    (
        entire_wrong_df,
        test_wrong_df,
    ) = prepare_unfiltered_frames(
        args.project_config_path,
        entire_raw_glob,
        test_raw_glob,
    )

    # ---------------------------------------------------------
    # Deduplicate within each source.
    # ---------------------------------------------------------

    (
        important_df,
        important_dedupe,
    ) = deduplicate_failure_rows(
        important_df,
        FAILURE_ID_COLUMNS,
    )

    (
        entire_wrong_df,
        entire_wrong_dedupe,
    ) = deduplicate_failure_rows(
        entire_wrong_df,
        FAILURE_ID_COLUMNS,
    )

    (
        test_wrong_df,
        test_wrong_dedupe,
    ) = deduplicate_failure_rows(
        test_wrong_df,
        FAILURE_ID_COLUMNS,
    )

    print()
    print(
        f"Important dedupe: "
        f"{important_dedupe}"
    )

    print(
        f"Entire wrong dedupe: "
        f"{entire_wrong_dedupe}"
    )

    print(
        f"Test wrong dedupe: "
        f"{test_wrong_dedupe}"
    )

    # ---------------------------------------------------------
    # Build authoritative dataset/K -> source index lookup.
    # ---------------------------------------------------------

    source_lookup = (
        build_dataset_k_to_source_index(
            project_config_path
        )
    )

    # ---------------------------------------------------------
    # Analysis.
    # ---------------------------------------------------------

    (
        summary_df,
        detail_df,
        validation_df,
    ) = build_population_failure_analysis(
        important_df,
        entire_wrong_df,
        test_wrong_df,
        source_lookup,
        splits_root,
        args.run_tag,
    )

    notes_df = build_notes(
        important_workbook,
        args.run_tag,
        entire_raw_glob,
        test_raw_glob,
    )

    # Add dedupe audit to notes.
    dedupe_notes = pd.DataFrame(
        [
            (
                "important_dedupe",
                str(important_dedupe),
            ),
            (
                "entire_wrong_dedupe",
                str(entire_wrong_dedupe),
            ),
            (
                "test_wrong_dedupe",
                str(test_wrong_dedupe),
            ),
        ],
        columns=[
            "note",
            "value",
        ],
    )

    notes_df = pd.concat(
        [
            notes_df,
            dedupe_notes,
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Write.
    # ---------------------------------------------------------

    write_workbook(
        output_path,
        summary_df,
        detail_df,
        validation_df,
        notes_df,
    )

    # ---------------------------------------------------------
    # Final validation.
    # ---------------------------------------------------------

    if (
        validation_df[
            "Wrong Unknown"
        ].sum()
        != 0
    ):
        raise ValueError(
            "Unknown wrong-prediction partition "
            "rows remain. Check saved split indices."
        )

    if (
        validation_df[
            "Important Unknown"
        ].sum()
        != 0
    ):
        raise ValueError(
            "Unknown important-failure partition "
            "rows remain. Check saved split indices."
        )

    print()
    print(
        "============================================================"
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "============================================================"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Output workbook:\n"
        f"  {output_path}"
    )


if __name__ == "__main__":
    main()