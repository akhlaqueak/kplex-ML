"""Unified failure-count, train/test, duplicate-audit, and mixed-combination summary workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from restructured_pipeline.src.compare_wrong_predictions import autosize_columns
from restructured_pipeline.src.failure_analysis_summary import (
    DEFAULT_PROJECT_CONFIG_PATH,
    FAILURE_ANALYSIS_SHEET_ORDER,
    _build_runtime_lookup,
)
from restructured_pipeline.src.failure_analysis_unfiltered_summary import load_unfiltered_records


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMPORTANT_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/failure_analysis_summary_scheme1_scheme3_04sep2026.xlsx"),
    Path("restructured_pipeline/outputs/reports/failure_analysis_summary_all_scheme_entire_data.xlsx"),
]
DEFAULT_MIXED_WORKBOOK_PATH = Path(
    "restructured_pipeline/outputs/reports/mixed_failure_case_combinations_xgb_only_corrected.xlsx"
)
DEFAULT_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/failure_train_test_split_count_summary.xlsx"
)
DEFAULT_ENTIRE_RAW_GLOB = (
    "restructured_pipeline/outputs/wrong_cases/*/*/"
    "wrong_predictions_*_predict_inference_entire_data.xlsx"
)
DEFAULT_TEST_RAW_GLOB = (
    "restructured_pipeline/outputs/wrong_cases/*/*/"
    "wrong_predictions_*_predict_inference_test_data.xlsx"
)

MERGE_COLUMNS = [
    "dataset", "k_value", "fastest_algo_tuple",
    "top1_candidate", "top2_candidate", "top3_candidate",
    "top1_min_time_ratio", "top2_min_time_ratio", "top3_min_time_ratio",
]
FAILURE_ID_COLUMNS = ["dataset", "k_value", "scheme", "model", "split_mode"]
ACTIVE_SCHEMES = {"scheme 1", "scheme 3"}
DUPLICATE_FILL = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

IMPORTANT_COUNTS_SHEET = "important_failure_counts"
ALL_WRONG_COUNTS_SHEET = "all_wrong_predictions_counts"
MIXED_COUNTS_SHEET = "mixed_combo_train_test_counts"
IMPORTANT_BY_SCHEME_MODEL_SHEET = "important_by_scheme_model"
IMPORTANT_BY_SPLIT_SCHEME_SHEET = "important_by_split_scheme"
ALL_WRONG_BY_SCHEME_MODEL_SHEET = "all_wrong_by_scheme_model"
ALL_WRONG_BY_SPLIT_SCHEME_SHEET = "all_wrong_by_split_scheme"
MASTER_SHEET = "important_failure_master"
DUPLICATE_SUMMARY_SHEET = "duplicate_failure_summary"
NOTES_SHEET = "analysis_notes"
PIVOT_SHEET = "pivot_summary"
OUTPUT_COUNT_COLUMNS = ["Scheme", "Model", "Split", "Training Failures", "Testing Failures", "Total Failures"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--important-workbook", type=Path, default=None)
    parser.add_argument("--mixed-workbook", type=Path, default=DEFAULT_MIXED_WORKBOOK_PATH)
    parser.add_argument("--project-config-path", type=Path, default=DEFAULT_PROJECT_CONFIG_PATH)

    # Preferred: explicitly select the experiment generation for each scheme.
    parser.add_argument("--scheme1-entire-raw-glob", default=None)
    parser.add_argument("--scheme1-test-raw-glob", default=None)
    parser.add_argument("--scheme3-entire-raw-glob", default=None)
    parser.add_argument("--scheme3-test-raw-glob", default=None)

    # Backward-compatible single-glob mode. Use only when both schemes share
    # exactly the same filename convention and generation.
    parser.add_argument("--entire-raw-glob", default=None)
    parser.add_argument("--test-raw-glob", default=None)

    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--skip-mixed", action="store_true", help="Skip mixed-combination analysis.")
    parser.add_argument(
        "--highlighted-input-copy",
        type=Path,
        default=None,
        help="Optional copy of the important-failure input workbook with duplicate rows highlighted yellow.",
    )
    return parser.parse_args()

def prepare_unfiltered_frames_by_scheme(
    project_config_path: Path,
    scheme1_entire_glob: str,
    scheme1_test_glob: str,
    scheme3_entire_glob: str,
    scheme3_test_glob: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Scheme 1 and Scheme 3 wrong cases from explicitly selected globs."""
    runtime_lookup = _build_runtime_lookup(REPO_ROOT / project_config_path)

    def load_one(pattern: str, expected_scheme: str, scope: str) -> pd.DataFrame:
        if not pattern:
            raise ValueError(f"Missing raw glob for {expected_scheme}/{scope}.")
        print(f"Loading {expected_scheme}/{scope}: {pattern}")
        df = load_unfiltered_records(pattern, runtime_lookup).copy()
        df = df.rename(columns={"_scheme": "scheme", "stratified_on": "split_mode"})
        required = {"scheme", "model", "split_mode", "dataset", "k_value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{expected_scheme}/{scope} missing columns: {sorted(missing)}")
        df["split_mode"] = df["split_mode"].map(normalize_split_mode)
        df = df[df["scheme"].eq(expected_scheme)].copy()
        if df.empty:
            raise ValueError(
                f"No rows for {expected_scheme}/{scope} matched {pattern!r}. "
                "Check the actual wrong-case filenames and glob."
            )
        print(f"  loaded rows={len(df)}")
        return df

    s1_entire = load_one(scheme1_entire_glob, "scheme 1", "entire_data")
    s1_test = load_one(scheme1_test_glob, "scheme 1", "test_data")
    s3_entire = load_one(scheme3_entire_glob, "scheme 3", "entire_data")
    s3_test = load_one(scheme3_test_glob, "scheme 3", "test_data")

    entire_df = pd.concat([s1_entire, s3_entire], ignore_index=True, sort=False)
    test_df = pd.concat([s1_test, s3_test], ignore_index=True, sort=False)

    for label, frame in (("Entire-data", entire_df), ("Test-data", test_df)):
        print(f"\n{label} experiment counts:")
        print(
            frame.groupby(["scheme", "model", "split_mode"], dropna=False)
            .size().reset_index(name="rows")
            .sort_values(["scheme", "model", "split_mode"], kind="stable")
            .to_string(index=False)
        )
    return entire_df, test_df

def resolve_existing_path(explicit: Path | None, candidates: list[Path], label: str) -> Path:
    if explicit is not None:
        resolved = REPO_ROOT / explicit if not explicit.is_absolute() else explicit
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"{label} not found: {resolved}")
    for candidate in candidates:
        resolved = REPO_ROOT / candidate
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Could not resolve {label} from candidates: {candidates}")


def normalize_split_mode(value: object) -> str:
    text = "" if value is None else str(value).strip()
    lowered = text.lower()
    if lowered in {"strat_y", "y"}:
        return "strat_y"
    if lowered in {"strat_y_k", "strat_yk", "y and k", "y_k", "y+k"}:
        return "strat_y_K"
    return text or "unknown"


def load_important_filtered_rows(workbook_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for expected_scheme, expected_model, expected_split_mode, sheet_name in FAILURE_ANALYSIS_SHEET_ORDER:
        df = pd.read_excel(workbook_path, sheet_name=sheet_name).copy()
        if df.empty:
            continue
        required = {"scheme", "model", "split_mode", "data_scope"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Sheet {sheet_name!r} missing provenance columns: {sorted(missing)}")
        if (~df["scheme"].eq(expected_scheme)).any():
            raise ValueError(f"Unexpected scheme values in {sheet_name}")
        if (~df["model"].eq(expected_model)).any():
            raise ValueError(f"Unexpected model values in {sheet_name}")
        if (~df["split_mode"].eq(expected_split_mode)).any():
            raise ValueError(f"Unexpected split_mode values in {sheet_name}")
        frames.append(df)
    if not frames:
        raise ValueError(f"No important-failure rows loaded from {workbook_path}")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    entire_only = combined[combined["data_scope"].eq("entire_data")].copy()
    if entire_only.empty:
        raise ValueError("Important workbook contains no data_scope='entire_data' rows.")
    print(f"Important source rows: all={len(combined)}, entire-used={len(entire_only)}, test-scope-excluded={combined['data_scope'].eq('test_data').sum()}")
    return entire_only.reset_index(drop=True)


def prepare_unfiltered_frames(
    project_config_path: Path,
    entire_raw_glob: str,
    test_raw_glob: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print()
    print("=== RAW WRONG-PREDICTION INPUTS ===")
    print(f"Entire glob: {entire_raw_glob}")
    print(f"Test glob:   {test_raw_glob}")

    # ---------------------------------------------------------
    # 1. Build runtime lookup
    # ---------------------------------------------------------

    runtime_lookup = _build_runtime_lookup(
        REPO_ROOT / project_config_path
    )

    # ---------------------------------------------------------
    # 2. LOAD DATA FIRST
    # ---------------------------------------------------------

    entire_df = load_unfiltered_records(
        entire_raw_glob,
        runtime_lookup,
    ).copy()

    test_df = load_unfiltered_records(
        test_raw_glob,
        runtime_lookup,
    ).copy()

    print()
    print(
        f"Raw rows loaded: "
        f"entire={len(entire_df)}, "
        f"test={len(test_df)}"
    )

    # ---------------------------------------------------------
    # 3. Normalize metadata column names
    # ---------------------------------------------------------

    entire_df = entire_df.rename(
        columns={
            "_scheme": "scheme",
            "stratified_on": "split_mode",
        }
    )

    test_df = test_df.rename(
        columns={
            "_scheme": "scheme",
            "stratified_on": "split_mode",
        }
    )

    # ---------------------------------------------------------
    # 4. Validate required metadata
    # ---------------------------------------------------------

    required_columns = {
        "scheme",
        "model",
        "split_mode",
    }

    for name, frame in (
        ("entire", entire_df),
        ("test", test_df),
    ):
        missing = (
            required_columns
            - set(frame.columns)
        )

        if missing:
            raise ValueError(
                f"{name} wrong-prediction data "
                f"is missing metadata columns: "
                f"{sorted(missing)}"
            )

    # ---------------------------------------------------------
    # 5. Normalize split labels
    # ---------------------------------------------------------

    entire_df["split_mode"] = (
        entire_df["split_mode"]
        .map(normalize_split_mode)
    )

    test_df["split_mode"] = (
        test_df["split_mode"]
        .map(normalize_split_mode)
    )

    # ---------------------------------------------------------
    # 6. Keep only active schemes
    # ---------------------------------------------------------

    entire_df = entire_df[
        entire_df["scheme"].isin(
            ACTIVE_SCHEMES
        )
    ].copy()

    test_df = test_df[
        test_df["scheme"].isin(
            ACTIVE_SCHEMES
        )
    ].copy()

    # ---------------------------------------------------------
    # 7. NOW print diagnostics
    # ---------------------------------------------------------

    print()
    print("Entire-data experiment counts:")

    if entire_df.empty:
        print("  NO ROWS FOUND")
    else:
        print(
            entire_df.groupby(
                [
                    "scheme",
                    "model",
                    "split_mode",
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="rows"
            )
            .sort_values(
                [
                    "scheme",
                    "model",
                    "split_mode",
                ],
                kind="stable",
            )
            .to_string(
                index=False
            )
        )

    print()
    print("Test-data experiment counts:")

    if test_df.empty:
        print("  NO ROWS FOUND")
    else:
        print(
            test_df.groupby(
                [
                    "scheme",
                    "model",
                    "split_mode",
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="rows"
            )
            .sort_values(
                [
                    "scheme",
                    "model",
                    "split_mode",
                ],
                kind="stable",
            )
            .to_string(
                index=False
            )
        )

    print()
    print(
        f"Rows after scheme filtering: "
        f"entire={len(entire_df)}, "
        f"test={len(test_df)}"
    )

    print(
        "====================================="
    )
    print()

    return entire_df, test_df


def analyze_failure_duplicates(df: pd.DataFrame, key_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [c for c in key_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot analyze duplicates; missing columns: {missing}")
    annotated = df.copy()
    annotated["duplicate_count"] = annotated.groupby(key_columns, dropna=False)[key_columns[0]].transform("size")
    annotated["is_duplicate_failure"] = annotated["duplicate_count"].gt(1)
    duplicate_summary = (
        annotated[annotated["is_duplicate_failure"]]
        .groupby(key_columns, dropna=False)
        .size().reset_index(name="duplicate_count")
        .sort_values(["duplicate_count", *key_columns], ascending=[False, *([True] * len(key_columns))], kind="stable")
        .reset_index(drop=True)
    )
    return annotated, duplicate_summary


def deduplicate_failure_rows(df: pd.DataFrame, key_columns: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    annotated, summary = analyze_failure_duplicates(df, key_columns)
    deduped = annotated.drop_duplicates(subset=key_columns, keep="first").reset_index(drop=True)
    stats = {
        "rows_before": int(len(annotated)),
        "rows_after": int(len(deduped)),
        "duplicate_rows_removed": int(len(annotated) - len(deduped)),
        "duplicate_groups": int(len(summary)),
    }
    return deduped, stats


def annotate_important_partitions(
    filtered_df: pd.DataFrame,
    entire_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign each important entire-data failure to the train or held-out
    test portion.

    Failure identity is based on the original experiment instance:

        scheme
        model
        split_mode
        dataset
        k_value

    Derived prediction fields such as top-k candidates, probabilities,
    fastest_algo_tuple, and time ratios are deliberately NOT merge keys.

    Rule:
        - every important failure must exist in the corresponding
          entire-data wrong-prediction output;
        - if the same experiment/dataset/K also occurs in the held-out
          test wrong-prediction output -> test;
        - otherwise -> train.

    NOTE:
    This method is suitable after the corrected Scheme 1 split, where
    original source rows cannot occur in both train and test.
    """

    keys = [
        "scheme",
        "model",
        "split_mode",
        "dataset",
        "k_value",
    ]

    # ---------------------------------------------------------
    # Validate columns
    # ---------------------------------------------------------

    for name, frame in (
        ("important", filtered_df),
        ("entire", entire_df),
        ("test", test_df),
    ):
        missing = [
            column
            for column in keys
            if column not in frame.columns
        ]

        if missing:
            raise ValueError(
                f"{name} source missing identity columns: "
                f"{missing}"
            )

    important = filtered_df.copy()
    entire = entire_df.copy()
    test = test_df.copy()

    # ---------------------------------------------------------
    # Normalize identity dtypes
    # ---------------------------------------------------------

    for frame in (
        important,
        entire,
        test,
    ):
        frame["scheme"] = (
            frame["scheme"]
            .astype(str)
            .str.strip()
        )

        frame["model"] = (
            frame["model"]
            .astype(str)
            .str.strip()
        )

        frame["split_mode"] = (
            frame["split_mode"]
            .map(normalize_split_mode)
        )

        frame["dataset"] = (
            frame["dataset"]
            .astype(str)
            .str.strip()
        )

        frame["k_value"] = pd.to_numeric(
            frame["k_value"],
            errors="raise",
        ).astype(int)

    # ---------------------------------------------------------
    # Entire-data lookup
    # ---------------------------------------------------------

    entire_lookup = (
        entire[
            keys
        ]
        .drop_duplicates()
        .assign(
            _in_entire=True
        )
    )

    important = important.merge(
        entire_lookup,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    unmatched_entire = (
        important["_in_entire"]
        .isna()
    )

    if unmatched_entire.any():

        unmatched = important.loc[
            unmatched_entire,
            keys,
        ].copy()

        print()
        print(
            "IMPORTANT FAILURES NOT FOUND IN "
            "ENTIRE-DATA WRONG PREDICTIONS"
        )

        print(
            unmatched.head(25).to_string(
                index=False
            )
        )

        print()

        print(
            "Unmatched counts by experiment:"
        )

        print(
            unmatched.groupby(
                [
                    "scheme",
                    "model",
                    "split_mode",
                ],
                dropna=False,
            )
            .size()
            .reset_index(
                name="rows"
            )
            .to_string(
                index=False
            )
        )

        raise ValueError(
            f"Could not match "
            f"{int(unmatched_entire.sum())} "
            "important failures to the corresponding "
            "entire-data wrong-prediction output using "
            "scheme/model/split/dataset/k identity. "
            "This usually means the important workbook "
            "and raw wrong-prediction files come from "
            "different experiment generations/run tags."
        )

    # ---------------------------------------------------------
    # Held-out test lookup
    # ---------------------------------------------------------

    test_lookup = (
        test[
            keys
        ]
        .drop_duplicates()
        .assign(
            _in_test=True
        )
    )

    important = important.merge(
        test_lookup,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    important["partition"] = (
        important["_in_test"]
        .eq(True)
        .map(
            {
                True: "test",
                False: "train",
            }
        )
    )

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    diagnostics = (
        important.groupby(
            [
                "scheme",
                "model",
                "split_mode",
                "partition",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="rows"
        )
    )

    print()
    print(
        "Important-failure partition diagnostics:"
    )

    print(
        diagnostics.to_string(
            index=False
        )
    )

    return important.drop(
        columns=[
            "_in_entire",
            "_in_test",
        ]
    )


def build_partition_counts(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["split_mode", "scheme", "model"], dropna=False)["partition"].value_counts().unstack(fill_value=0).reset_index()
    for column in ("train", "test", "unknown"):
        if column not in grouped.columns:
            grouped[column] = 0
    grouped = grouped.rename(columns={"train": "train_failures", "test": "test_failures", "unknown": "unknown_partition"})
    grouped["total_failures"] = grouped["train_failures"] + grouped["test_failures"] + grouped["unknown_partition"]
    return grouped[["split_mode", "scheme", "model", "train_failures", "test_failures", "unknown_partition", "total_failures"]].sort_values(["scheme", "model", "split_mode"], kind="stable").reset_index(drop=True)


def build_all_wrong_partitioned(entire_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Partition wrong predictions by instance identity, not count subtraction."""
    keys = FAILURE_ID_COLUMNS
    entire_unique = entire_df[keys].drop_duplicates().copy()
    test_unique = test_df[keys].drop_duplicates().copy()

    test_membership = test_unique.assign(_in_test=True)
    marked = entire_unique.merge(test_membership, on=keys, how="left", validate="one_to_one")
    marked["partition"] = marked["_in_test"].eq(True).map({True: "test", False: "train"})

    # Every test wrong prediction must also be a wrong prediction under entire-data inference.
    entire_membership = entire_unique.assign(_in_entire=True)
    test_check = test_unique.merge(entire_membership, on=keys, how="left", validate="one_to_one")
    missing_test = test_check["_in_entire"].isna()
    if missing_test.any():
        examples = test_check.loc[missing_test, keys].head(25)
        raise ValueError(
            f"{int(missing_test.sum())} test wrong predictions are absent from entire-data wrong predictions. "
            f"Examples:\n{examples.to_string(index=False)}"
        )

    grouped = (
        marked.groupby(["split_mode", "scheme", "model"], dropna=False)["partition"]
        .value_counts().unstack(fill_value=0).reset_index()
    )
    for column in ("train", "test"):
        if column not in grouped.columns:
            grouped[column] = 0
    grouped = grouped.rename(columns={"train": "train_failures", "test": "test_failures"})
    grouped["unknown_partition"] = 0
    grouped["total_failures"] = grouped["train_failures"] + grouped["test_failures"]
    return grouped[["split_mode", "scheme", "model", "train_failures", "test_failures", "unknown_partition", "total_failures"]].sort_values(
        ["scheme", "model", "split_mode"], kind="stable"
    ).reset_index(drop=True)

def build_aggregate_sheet(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_cols, dropna=False)[["train_failures", "test_failures", "unknown_partition", "total_failures"]].sum().reset_index().sort_values(group_cols, kind="stable").reset_index(drop=True)


def format_count_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.rename(columns={"scheme": "Scheme", "model": "Model", "split_mode": "Split", "train_failures": "Training Failures", "test_failures": "Testing Failures", "total_failures": "Total Failures"}).copy()
    return formatted[OUTPUT_COUNT_COLUMNS]


def annotate_mixed_combo_rows(mixed_workbook: Path, entire_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    workbook = pd.ExcelFile(mixed_workbook)
    summary_sheet = "failure case summary" if "failure case summary" in workbook.sheet_names else "summary"
    excluded = {summary_sheet, "shared_failure_summary"}
    frames = []
    for sheet_name in workbook.sheet_names:
        if sheet_name in excluded:
            continue
        df = pd.read_excel(mixed_workbook, sheet_name=sheet_name).copy()
        if df.empty:
            continue
        df.insert(0, "combination_sheet", sheet_name)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if combined.empty:
        return combined, summary_sheet
    xgb_entire = entire_df[entire_df["model"].eq("XGB")].copy()
    xgb_test = test_df[test_df["model"].eq("XGB")].copy()
    lookup = xgb_entire[["scheme", "split_mode", *MERGE_COLUMNS]].drop_duplicates()
    merged = combined.merge(lookup, on=["scheme", *MERGE_COLUMNS], how="left")
    if merged["split_mode"].isna().any():
        raise ValueError(f"Could not recover split_mode for {merged['split_mode'].isna().sum()} mixed rows.")
    test_lookup = xgb_test[["scheme", "split_mode", *MERGE_COLUMNS]].drop_duplicates().assign(_is_test=True)
    merged = merged.merge(test_lookup, on=["scheme", "split_mode", *MERGE_COLUMNS], how="left")
    merged["partition"] = merged["_is_test"].eq(True).map({True: "test", False: "train"})
    return merged.drop(columns=["_is_test"]), summary_sheet


def build_mixed_combo_summary(mixed_rows: pd.DataFrame) -> pd.DataFrame:
    if mixed_rows.empty:
        return pd.DataFrame(columns=["Combination Sheet", "Source Rule", "Scheme Pair", "Training Failures", "Testing Failures", "Total Failures", "Notes"])
    rows = []
    for sheet_name, group in mixed_rows.groupby("combination_sheet", dropna=False):
        source_rule = str(group["source_rule"].iloc[0]) if "source_rule" in group.columns else str(sheet_name)
        pair = "scheme 1 + scheme 3" if "scheme3" in str(sheet_name) else "scheme 1 + scheme 2"
        train = int(group["partition"].eq("train").sum())
        test = int(group["partition"].eq("test").sum())
        splits = ",".join(sorted(group["split_mode"].dropna().astype(str).unique()))
        rows.append({"Combination Sheet": sheet_name, "Source Rule": source_rule, "Scheme Pair": pair, "Training Failures": train, "Testing Failures": test, "Total Failures": int(len(group)), "Notes": f"splits={splits or 'unknown'}"})
    return pd.DataFrame(rows).sort_values("Combination Sheet", kind="stable").reset_index(drop=True)


def build_pivot_summary(important_counts: pd.DataFrame, all_wrong_counts: pd.DataFrame) -> pd.DataFrame:
    important = important_counts.copy(); important.insert(0, "source_type", "important_failures")
    all_wrong = all_wrong_counts.copy(); all_wrong.insert(0, "source_type", "all_wrong_predictions")
    return pd.concat([important, all_wrong], ignore_index=True, sort=False)


def highlight_duplicate_rows(worksheet, flag_column: str = "is_duplicate_failure") -> int:
    headers = [cell.value for cell in worksheet[1]]
    if flag_column not in headers:
        return 0
    flag_idx = headers.index(flag_column) + 1
    count = 0
    for row_idx in range(2, worksheet.max_row + 1):
        value = worksheet.cell(row=row_idx, column=flag_idx).value
        if not (value is True or str(value).strip().lower() in {"true", "1", "yes"}):
            continue
        for col_idx in range(1, worksheet.max_column + 1):
            worksheet.cell(row=row_idx, column=col_idx).fill = DUPLICATE_FILL
        count += 1
    return count


def validate_outputs(important_counts: pd.DataFrame, all_wrong_counts: pd.DataFrame, unique_important_rows: pd.DataFrame) -> dict[str, object]:
    for name, frame in (("important", important_counts), ("all_wrong", all_wrong_counts)):
        invalid = frame[frame["total_failures"] != frame["train_failures"] + frame["test_failures"] + frame["unknown_partition"]]
        if not invalid.empty:
            raise ValueError(f"{name} counts have invalid totals.")
    if int(important_counts["total_failures"].sum()) != len(unique_important_rows):
        raise ValueError("Important counts do not equal deduplicated important rows.")
    comparison = important_counts.merge(all_wrong_counts, on=["split_mode", "scheme", "model"], how="left", suffixes=("_important", "_all"))
    if (comparison["total_failures_all"] < comparison["total_failures_important"]).any():
        raise ValueError("All-wrong count is smaller than important count for at least one group.")
    return {"important_unique_total": int(len(unique_important_rows)), "all_wrong_total": int(all_wrong_counts["total_failures"].sum())}


def write_workbook(output_path: Path, important_counts: pd.DataFrame, all_wrong_counts: pd.DataFrame,
                   important_by_scheme_model: pd.DataFrame, important_by_split_scheme: pd.DataFrame,
                   all_wrong_by_scheme_model: pd.DataFrame, all_wrong_by_split_scheme: pd.DataFrame,
                   important_master: pd.DataFrame, duplicate_summary: pd.DataFrame,
                   mixed_summary: pd.DataFrame, notes_df: pd.DataFrame, pivot_df: pd.DataFrame) -> None:
    output_path = REPO_ROOT / output_path if not output_path.is_absolute() else output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        format_count_table(important_counts).to_excel(writer, sheet_name=IMPORTANT_COUNTS_SHEET, index=False)
        format_count_table(all_wrong_counts).to_excel(writer, sheet_name=ALL_WRONG_COUNTS_SHEET, index=False)
        important_by_scheme_model.to_excel(writer, sheet_name=IMPORTANT_BY_SCHEME_MODEL_SHEET, index=False)
        important_by_split_scheme.to_excel(writer, sheet_name=IMPORTANT_BY_SPLIT_SCHEME_SHEET, index=False)
        all_wrong_by_scheme_model.to_excel(writer, sheet_name=ALL_WRONG_BY_SCHEME_MODEL_SHEET, index=False)
        all_wrong_by_split_scheme.to_excel(writer, sheet_name=ALL_WRONG_BY_SPLIT_SCHEME_SHEET, index=False)
        important_master.to_excel(writer, sheet_name=MASTER_SHEET, index=False)
        duplicate_summary.to_excel(writer, sheet_name=DUPLICATE_SUMMARY_SHEET, index=False)
        mixed_summary.to_excel(writer, sheet_name=MIXED_COUNTS_SHEET, index=False)
        notes_df.to_excel(writer, sheet_name=NOTES_SHEET, index=False)
        pivot_df.to_excel(writer, sheet_name=PIVOT_SHEET, index=False)
    workbook = load_workbook(output_path)
    highlighted = highlight_duplicate_rows(workbook[MASTER_SHEET])
    for sheet_name in workbook.sheetnames:
        ws = workbook[sheet_name]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        autosize_columns(ws)
    workbook.save(output_path)
    workbook.close()
    print(f"Duplicate master rows highlighted yellow: {highlighted}")

def _sortable_k_value(value):
    """Return a stable numeric-first sort key for k_value."""
    try:
        if pd.isna(value):
            return (1, float("inf"))

        return (0, float(value))

    except (TypeError, ValueError):
        return (1, str(value))

def create_duplicate_highlighted_input_copy(
    input_workbook: Path,
    output_workbook: Path,
) -> dict[str, object]:
    """
    Create a copy of the enriched failure-analysis workbook and highlight
    failures that occur in BOTH entire_data and test_data.

    Overlap identity:
        dataset
        k_value
        scheme
        model
        split_mode

    data_scope is deliberately NOT part of the identity.

    A group is highlighted only when it contains:
        entire_data
        AND
        test_data

    Rows are sorted by:
        dataset
        k_value
        data_scope

    with entire_data shown before test_data.
    """
    import shutil

    output_workbook.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        input_workbook,
        output_workbook,
    )

    workbook = load_workbook(output_workbook)

    yellow_fill = PatternFill(
        start_color="FFF2CC",
        end_color="FFF2CC",
        fill_type="solid",
    )

    no_fill = PatternFill(fill_type=None)

    # IMPORTANT:
    # data_scope intentionally excluded.
    overlap_key_columns = [
        "dataset",
        "k_value",
        "scheme",
        "model",
        "split_mode",
    ]

    experiment_sheets = [
        sheet_name
        for _, _, _, sheet_name
        in FAILURE_ANALYSIS_SHEET_ORDER
    ]

    total_rows_highlighted = 0
    total_overlap_groups = 0

    per_sheet_stats = {}

    for sheet_name in experiment_sheets:

        if sheet_name not in workbook.sheetnames:
            continue

        worksheet = workbook[sheet_name]

        headers = [
            cell.value
            for cell in worksheet[1]
        ]

        required_columns = [
            *overlap_key_columns,
            "data_scope",
        ]

        missing = [
            column
            for column in required_columns
            if column not in headers
        ]

        if missing:
            raise ValueError(
                f"Cannot analyze scope overlap in "
                f"{sheet_name!r}; missing columns: "
                f"{missing}"
            )

        # --------------------------------------------------
        # Read rows.
        # --------------------------------------------------

        row_records = []

        for row_idx in range(
            2,
            worksheet.max_row + 1,
        ):
            record = {
                header: worksheet.cell(
                    row=row_idx,
                    column=col_idx,
                ).value
                for col_idx, header
                in enumerate(headers, start=1)
            }

            row_records.append(record)

        # --------------------------------------------------
        # Sort:
        #
        # dataset
        # k_value
        # entire_data before test_data
        # --------------------------------------------------

        scope_order = {
            "entire_data": 0,
            "test_data": 1,
        }

        row_records.sort(
            key=lambda record: (
                str(
                    record.get("dataset") or ""
                ).casefold(),
                _sortable_k_value(
                    record.get("k_value")
                ),
                scope_order.get(
                    str(
                        record.get("data_scope")
                    ),
                    99,
                ),
            )
        )

        # --------------------------------------------------
        # Rewrite sorted rows.
        # --------------------------------------------------

        for output_row_idx, record in enumerate(
            row_records,
            start=2,
        ):
            for col_idx, header in enumerate(
                headers,
                start=1,
            ):
                cell = worksheet.cell(
                    row=output_row_idx,
                    column=col_idx,
                )

                cell.value = record.get(header)

                # Remove old fill before applying our
                # deterministic overlap highlighting.
                cell.fill = no_fill

        # --------------------------------------------------
        # Build groups WITHOUT data_scope.
        # --------------------------------------------------

        key_indices = {
            column:
                headers.index(column) + 1
            for column in overlap_key_columns
        }

        scope_idx = (
            headers.index("data_scope") + 1
        )

        groups = {}

        for row_idx in range(
            2,
            worksheet.max_row + 1,
        ):
            key = tuple(
                worksheet.cell(
                    row=row_idx,
                    column=key_indices[column],
                ).value
                for column in overlap_key_columns
            )

            scope = worksheet.cell(
                row=row_idx,
                column=scope_idx,
            ).value

            scope = (
                ""
                if scope is None
                else str(scope).strip()
            )

            group = groups.setdefault(
                key,
                {
                    "rows": [],
                    "scopes": set(),
                },
            )

            group["rows"].append(row_idx)

            if scope:
                group["scopes"].add(scope)

        # --------------------------------------------------
        # Highlight ONLY cross-scope overlaps.
        # --------------------------------------------------

        sheet_overlap_groups = 0
        sheet_rows_highlighted = 0

        for group in groups.values():

            scopes = group["scopes"]

            if not {
                "entire_data",
                "test_data",
            }.issubset(scopes):
                continue

            sheet_overlap_groups += 1

            for row_idx in group["rows"]:

                for col_idx in range(
                    1,
                    worksheet.max_column + 1,
                ):
                    worksheet.cell(
                        row=row_idx,
                        column=col_idx,
                    ).fill = yellow_fill

                sheet_rows_highlighted += 1

        per_sheet_stats[sheet_name] = {
            "overlap_groups":
                sheet_overlap_groups,
            "rows_highlighted":
                sheet_rows_highlighted,
        }

        total_overlap_groups += (
            sheet_overlap_groups
        )

        total_rows_highlighted += (
            sheet_rows_highlighted
        )

        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = (
            worksheet.dimensions
        )

        autosize_columns(worksheet)

    workbook.save(output_workbook)
    workbook.close()

    return {
        "total_scope_overlap_groups":
            total_overlap_groups,
        "total_rows_highlighted":
            total_rows_highlighted,
        "per_sheet":
            per_sheet_stats,
    }

def main() -> None:
    args = parse_args()
    important_workbook = resolve_existing_path(args.important_workbook, DEFAULT_IMPORTANT_CANDIDATES, "important workbook")

    if args.highlighted_input_copy is not None:

        highlighted_path = (
            REPO_ROOT / args.highlighted_input_copy
            if not args.highlighted_input_copy.is_absolute()
            else args.highlighted_input_copy
        )

        duplicate_highlight_stats = (
            create_duplicate_highlighted_input_copy(
                important_workbook,
                highlighted_path,
            )
        )

        print(
            "\nHighlighted duplicate input workbook:"
            f"\n  {highlighted_path}"
        )

        print(
            "Duplicate highlighting statistics:"
        )

        for key, value in (
            duplicate_highlight_stats.items()
        ):
            print(f"  {key}: {value}")

    filtered_source_rows = load_important_filtered_rows(important_workbook)

    explicit_scheme_mode = all([
        args.scheme1_entire_raw_glob,
        args.scheme1_test_raw_glob,
        args.scheme3_entire_raw_glob,
        args.scheme3_test_raw_glob,
    ])
    legacy_mode = bool(args.entire_raw_glob and args.test_raw_glob)

    if explicit_scheme_mode:
        entire_df, test_df = prepare_unfiltered_frames_by_scheme(
            args.project_config_path,
            args.scheme1_entire_raw_glob,
            args.scheme1_test_raw_glob,
            args.scheme3_entire_raw_glob,
            args.scheme3_test_raw_glob,
        )
        raw_input_notes = [
            ("scheme1_entire_wrong_glob", args.scheme1_entire_raw_glob),
            ("scheme1_test_wrong_glob", args.scheme1_test_raw_glob),
            ("scheme3_entire_wrong_glob", args.scheme3_entire_raw_glob),
            ("scheme3_test_wrong_glob", args.scheme3_test_raw_glob),
        ]
    elif legacy_mode:
        print("WARNING: using legacy single-glob mode. Explicit per-scheme globs are safer.")
        entire_df, test_df = prepare_unfiltered_frames(
            args.project_config_path, args.entire_raw_glob, args.test_raw_glob
        )
        raw_input_notes = [
            ("entire_wrong_prediction_glob", args.entire_raw_glob),
            ("test_wrong_prediction_glob", args.test_raw_glob),
        ]
    else:
        raise ValueError(
            "Provide either all four per-scheme raw globs "
            "(--scheme1-entire-raw-glob, --scheme1-test-raw-glob, "
            "--scheme3-entire-raw-glob, --scheme3-test-raw-glob) "
            "or both legacy --entire-raw-glob/--test-raw-glob."
        )

    # Audit important failures before deduplication; preserve this version in the master sheet.
    important_master, duplicate_summary = analyze_failure_duplicates(filtered_source_rows, FAILURE_ID_COLUMNS)
    unique_important, important_dedupe = deduplicate_failure_rows(filtered_source_rows, FAILURE_ID_COLUMNS)

    # Deduplicate entire and test raw wrong predictions independently. A row present once in each scope is intentional.
    entire_df, entire_dedupe = deduplicate_failure_rows(entire_df, FAILURE_ID_COLUMNS)
    test_df, test_dedupe = deduplicate_failure_rows(test_df, FAILURE_ID_COLUMNS)

    important_rows = annotate_important_partitions(unique_important, entire_df, test_df)
    important_counts = build_partition_counts(important_rows)
    all_wrong_counts = build_all_wrong_partitioned(entire_df, test_df)

    important_by_scheme_model = build_aggregate_sheet(important_counts, ["scheme", "model"])
    important_by_split_scheme = build_aggregate_sheet(important_counts, ["split_mode", "scheme"])
    all_wrong_by_scheme_model = build_aggregate_sheet(all_wrong_counts, ["scheme", "model"])
    all_wrong_by_split_scheme = build_aggregate_sheet(all_wrong_counts, ["split_mode", "scheme"])

    mixed_summary = pd.DataFrame(columns=["Combination Sheet", "Source Rule", "Scheme Pair", "Training Failures", "Testing Failures", "Total Failures", "Notes"])
    mixed_source = "skipped"
    if not args.skip_mixed:
        mixed_path = REPO_ROOT / args.mixed_workbook if not args.mixed_workbook.is_absolute() else args.mixed_workbook
        if mixed_path.exists():
            mixed_rows, mixed_source = annotate_mixed_combo_rows(mixed_path, entire_df, test_df)
            mixed_summary = build_mixed_combo_summary(mixed_rows)
        else:
            print(f"WARNING: mixed workbook not found; mixed analysis skipped: {mixed_path}")
            mixed_source = "missing"

    pivot_df = build_pivot_summary(format_count_table(important_counts), format_count_table(all_wrong_counts))
    notes_df = pd.DataFrame([
        ("important_workbook_used", str(important_workbook)),
        *raw_input_notes,
        ("duplicate_key", ", ".join(FAILURE_ID_COLUMNS)),
        ("important_dedupe", str(important_dedupe)),
        ("entire_wrong_dedupe", str(entire_dedupe)),
        ("test_wrong_dedupe", str(test_dedupe)),
        ("mixed_summary_source", mixed_source),
        ("train_test_rule", "test = matched test wrong prediction; train = entire wrong prediction not present in test wrong predictions"),
    ], columns=["note", "value"])

    write_workbook(args.output_path, important_counts, all_wrong_counts,
                   important_by_scheme_model, important_by_split_scheme,
                   all_wrong_by_scheme_model, all_wrong_by_split_scheme,
                   important_master, duplicate_summary, mixed_summary, notes_df, pivot_df)

    validation = validate_outputs(important_counts, all_wrong_counts, unique_important)
    print(f"Output workbook: {REPO_ROOT / args.output_path}")
    print(f"Important dedupe: {important_dedupe}")
    print(f"Entire wrong dedupe: {entire_dedupe}")
    print(f"Test wrong dedupe: {test_dedupe}")
    print("Important failure counts:")
    print(format_count_table(important_counts).to_string(index=False))
    print("All wrong prediction counts:")
    print(format_count_table(all_wrong_counts).to_string(index=False))
    print("Mixed combination counts:")
    print(mixed_summary.to_string(index=False))
    print(f"Validation: {validation}")


if __name__ == "__main__":
    main()