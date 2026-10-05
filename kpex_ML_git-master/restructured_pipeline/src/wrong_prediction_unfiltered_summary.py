"""Build an unfiltered wrong-prediction summary workbook from existing exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_GLOB = "restructured_pipeline/outputs/wrong_cases/*/*/wrong_predictions_*_predict_inference_entire_data.xlsx"
DEFAULT_FILTERED_WORKBOOK_PATH = Path(
    "restructured_pipeline/outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data.xlsx"
)
DEFAULT_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/wrong_prediction_unfiltered_summary.xlsx"
)
SUMMARY_SHEET = "summary"
DETAIL_SHEET_ORDER = [
    ("scheme 1", "RF", "scheme_1_RF"),
    ("scheme 1", "XGB", "scheme_1_XGB"),
    ("scheme 2", "RF", "scheme_2_RF"),
    ("scheme 2", "XGB", "scheme_2_XGB"),
    ("scheme 3", "RF", "scheme_3_RF"),
    ("scheme 3", "XGB", "scheme_3_XGB"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-glob",
        default=DEFAULT_RAW_GLOB,
        help="Glob used to discover existing entire-dataset wrong-prediction workbooks.",
    )
    parser.add_argument(
        "--filtered-workbook-path",
        type=Path,
        default=DEFAULT_FILTERED_WORKBOOK_PATH,
        help="Existing important-failure workbook used for count comparison.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination for the unfiltered wrong-prediction summary workbook.",
    )
    return parser.parse_args()


def detect_scheme_model(path: Path) -> tuple[str, str]:
    parts = path.as_posix()
    if "scheme1" in parts:
        scheme = "scheme 1"
    elif "scheme2" in parts:
        scheme = "scheme 2"
    elif "scheme3" in parts:
        scheme = "scheme 3"
    else:
        raise ValueError(f"Could not infer scheme from {path}")

    if "xgboost" in parts:
        model = "XGB"
    elif "random_forest" in parts:
        model = "RF"
    else:
        raise ValueError(f"Could not infer model from {path}")

    return scheme, model


def detect_stratified_on(path: Path) -> str:
    name = path.name
    if "strat_y_K" in name:
        return "strat_y_K"
    if "strat_y" in name:
        return "strat_y"
    raise ValueError(f"Could not infer split mode from {path}")


def load_raw_wrong_predictions(raw_glob: str) -> pd.DataFrame:
    raw_paths = sorted(REPO_ROOT.glob(raw_glob))
    if not raw_paths:
        raise FileNotFoundError(f"No wrong-prediction workbooks matched glob: {raw_glob}")

    frames: list[pd.DataFrame] = []
    for path in raw_paths:
        df = pd.read_excel(path)
        scheme, model = detect_scheme_model(path)
        stratified_on = detect_stratified_on(path)
        insert_at = 0
        df.insert(insert_at, "source_workbook", path.name)
        df.insert(insert_at, "stratified_on", stratified_on)
        df.insert(insert_at, "model", model)
        df.insert(insert_at, "scheme", scheme)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def load_filtered_counts(filtered_workbook_path: Path) -> pd.DataFrame:
    filtered_workbook_path = REPO_ROOT / filtered_workbook_path
    if not filtered_workbook_path.exists():
        return pd.DataFrame(
            columns=[
                "scheme",
                "model",
                "stratified_on",
                "important_failure_count",
            ]
        )

    filtered_df = pd.read_excel(filtered_workbook_path, sheet_name="on entire dataset")
    filtered_df = filtered_df.copy()
    filtered_df["stratified_on"] = filtered_df["strat_on"].map(
        {
            "y": "strat_y",
            "y and K": "strat_y_K",
        }
    ).fillna(filtered_df["strat_on"].astype(str))

    grouped = (
        filtered_df.groupby(["scheme", "model", "stratified_on"], dropna=False)
        .size()
        .reset_index(name="important_failure_count")
    )
    return grouped


def build_summary(raw_df: pd.DataFrame, filtered_counts_df: pd.DataFrame) -> pd.DataFrame:
    raw_counts = (
        raw_df.groupby(["scheme", "model", "stratified_on"], dropna=False)
        .size()
        .reset_index(name="wrong_prediction_count")
    )
    summary = raw_counts.merge(
        filtered_counts_df,
        on=["scheme", "model", "stratified_on"],
        how="left",
    )
    summary["important_failure_count"] = summary["important_failure_count"].fillna(0).astype(int)
    summary["filtered_out_count"] = summary["wrong_prediction_count"] - summary["important_failure_count"]
    summary["filtered_out_percentage"] = (
        100.0 * summary["filtered_out_count"] / summary["wrong_prediction_count"]
    ).round(2)

    sort_key = {key: index for index, key in enumerate((sheet[:2] for sheet in DETAIL_SHEET_ORDER))}
    summary["_sort_key"] = summary.apply(lambda row: sort_key[(row["scheme"], row["model"])], axis=1)
    summary["_strat_key"] = summary["stratified_on"].map({"strat_y": 0, "strat_y_K": 1}).fillna(99)
    summary = summary.sort_values(["_sort_key", "_strat_key"]).drop(columns=["_sort_key", "_strat_key"])
    return summary


def autosize_excel_columns(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    for index, column_name in enumerate(df.columns, start=1):
        values = [column_name, *df[column_name].tolist()]
        width = min(max(len(str(value)) for value in values) + 2, 50)
        worksheet.column_dimensions[worksheet.cell(row=1, column=index).column_letter].width = width


def write_workbook(raw_df: pd.DataFrame, summary_df: pd.DataFrame, output_path: Path) -> None:
    output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name=SUMMARY_SHEET, index=False)
        autosize_excel_columns(writer, SUMMARY_SHEET, summary_df)

        for scheme, model, sheet_name in DETAIL_SHEET_ORDER:
            sheet_df = raw_df[(raw_df["scheme"] == scheme) & (raw_df["model"] == model)].copy()
            sheet_df = sheet_df.sort_values(["stratified_on", "dataset", "k_value", "source_row_index"])
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
            autosize_excel_columns(writer, sheet_name, sheet_df)


def validate_workbook(raw_df: pd.DataFrame, summary_df: pd.DataFrame) -> dict[str, object]:
    raw_low_runtime_mask = (raw_df["fastest_time"] < 5) & (raw_df["predicted_time"] < 5)
    raw_within_ratio_mask = raw_df["predicted_time"] <= (1.2 * raw_df["fastest_time"])

    per_sheet_counts = {}
    for scheme, model, sheet_name in DETAIL_SHEET_ORDER:
        count = int(((raw_df["scheme"] == scheme) & (raw_df["model"] == model)).sum())
        summary_count = int(
            summary_df[(summary_df["scheme"] == scheme) & (summary_df["model"] == model)]["wrong_prediction_count"].sum()
        )
        if count != summary_count:
            raise ValueError(
                f"Summary mismatch for {sheet_name}: detail rows={count}, summary rows={summary_count}"
            )
        per_sheet_counts[sheet_name] = count

    if raw_df["stratified_on"].isna().any():
        raise ValueError("Found missing stratified_on values in raw wrong-prediction rows")

    return {
        "raw_low_runtime_rows_present": int(raw_low_runtime_mask.sum()),
        "raw_within_ratio_rows_present": int(raw_within_ratio_mask.sum()),
        "sheet_row_counts": per_sheet_counts,
    }


def main() -> None:
    args = parse_args()
    raw_df = load_raw_wrong_predictions(args.raw_glob)
    filtered_counts_df = load_filtered_counts(args.filtered_workbook_path)
    summary_df = build_summary(raw_df, filtered_counts_df)
    write_workbook(raw_df, summary_df, args.output_path)
    validation = validate_workbook(raw_df, summary_df)

    print(f"Wrote {args.output_path}")
    print("Summary counts:")
    print(summary_df.to_string(index=False))
    print("Validation:")
    for key, value in validation.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
