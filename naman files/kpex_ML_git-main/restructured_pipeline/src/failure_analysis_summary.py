"""Build failure summary workbooks and enrich important-failure artifacts."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd
import yaml
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from restructured_pipeline.src.compare_wrong_predictions import (
    autosize_columns,
)
from restructured_pipeline.src.failure_case_filter import annotate_failure_cases
from restructured_pipeline.src.features import algorithms as ALL_ALGOS, load_excel_data
from restructured_pipeline.src.reporting_utils import (
    round_or_none as shared_round_or_none,
    _safe_float as shared_safe_float,
    warn_deprecated_path_usage,
)


DEFAULT_ENTIRE_IMPORTANT_PATH = Path(
    "restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_report_all_entire_data_in_one_sheet.xlsx"
)
DEFAULT_ENTIRE_IMPORTANT_CANDIDATES = [
    DEFAULT_ENTIRE_IMPORTANT_PATH,
    Path("restructured_pipeline/outputs/reports/failure_case_analysis_cleaned.xlsx"),
    Path("restructured_pipeline/outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data_fixed.xlsx"),
]
DEFAULT_ENTIRE_BASE_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data.xlsx"),
    Path("restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_scheme_entire_data_pre_consistency_fix.xlsx"),
    Path("restructured_pipeline/outputs/reports/comparison_wrong_predictions_all_schema_entire_data.xlsx"),
]
DEFAULT_TEST_COMPARE_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/test_data/compare_wrong_predictions_test_data.xlsx"),
    Path("restructured_pipeline/outputs/reports/compare_wrong_predictions_test_data.xlsx"),
]
DEFAULT_OUTPUT_PATH = Path("restructured_pipeline/outputs/reports/failure_analysis_summary_entire_vs_test.xlsx")
DEFAULT_FAILURE_ANALYSIS_WORKBOOK_OUTPUT_PATH = Path(
    "restructured_pipeline/outputs/reports/failure_analysis_summary_all_scheme_entire_data.xlsx"
)
DEFAULT_PROJECT_CONFIG_PATH = Path("restructured_pipeline/configs/project.yaml")
DEFAULT_FAILURE_ANALYSIS_INPUT_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/failure_case_analysis_important_failures_all_scheme_entire_data.xlsx"),
    Path("restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"),
    Path("restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_important_failures.xlsx"),
]

DEFAULT_ENTIRE_IMPORTANT_FAILURES_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/entire_data/compare_wrong_predictions_tolerance_important_failures_entire_data.xlsx"),
    Path("restructured_pipeline/outputs/reports/compare_wrong_predictions_tolerance_important_failures.xlsx"),
]
DEFAULT_TEST_IMPORTANT_FAILURES_CANDIDATES = [
    Path("restructured_pipeline/outputs/reports/test_data/compare_wrong_predictions_tolerance_important_failures_test_data.xlsx"),
]
DEFAULT_FILTER_SUMMARY_SHEET = "filter_summary"
DEFAULT_ENRICHMENT_MODE = "enrich"

SCHEMES = ("scheme 1", "scheme 2", "scheme 3")
MODELS = ("RF", "XGB")
SEVERITY_LABELS = (">100%", "20–100%")
FAILURE_ANALYSIS_SHEET_ORDER = [
    ("scheme 1", "RF", "scheme_1_RF"),
    ("scheme 1", "XGB", "scheme_1_XGB"),
    ("scheme 2", "RF", "scheme_2_RF"),
    ("scheme 2", "XGB", "scheme_2_XGB"),
    ("scheme 3", "RF", "scheme_3_RF"),
    ("scheme 3", "XGB", "scheme_3_XGB"),
]
TOP_K_VALUES = tuple(range(1, 7))
FORBIDDEN_ANALYSIS_COLUMNS = {
    "inference_on",
    "with_holdout",
    "strat_on",
    "scheme",
    "source_model",
    "failure_case_filter_reason",
    "failure_severity",
    "fastest_algo",
    "fastest_time",
    "predicted_algo",
    "predicted_time",
    "predicted_probability",
    "time_gap",
    "time_gap_ratio",
    "tolerance",
    "split_scope",
    "source_sheet",
    "actual_time_ratio",
    "high_confidence_wrong_prediction_flag",
}


def resolve_existing_path(candidates: list[Path], label: str) -> Path:
    canonical_path = candidates[0]
    for index, candidate in enumerate(candidates):
        if candidate.exists():
            if index > 0:
                warn_deprecated_path_usage(
                    candidate,
                    canonical_path,
                    label,
                )
            return candidate
    raise FileNotFoundError(f"Could not resolve {label} from candidates: {candidates}")


def resolve_explicit_or_existing_path(
    explicit_path: str | Path | None,
    candidates: list[Path],
    label: str,
) -> Path:
    if explicit_path:
        resolved = Path(explicit_path)
        deprecated_candidates = candidates[1:]
        if resolved in deprecated_candidates:
            warn_deprecated_path_usage(
                resolved,
                candidates[0],
                label,
            )
        return resolved
    return resolve_existing_path(candidates, label)


def percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(100.0 * numerator / denominator):.1f}%"


def normalize_severity_label(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return text.replace("â€“", "–")


def resolve_failure_severity_mask(df: pd.DataFrame, severity_label: str) -> pd.Series:
    return df["failure_severity"].map(normalize_severity_label).eq(severity_label)


def load_entire_important(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="on entire dataset")


def load_entire_base(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="on entire dataset")


def load_test_compare(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="on 20% test set")


def compute_important_subset_from_compare(df: pd.DataFrame) -> pd.DataFrame:
    important_df, _ = annotate_failure_cases(
        df,
        small_runtime_threshold=5.0,
        slowdown_ratio_threshold=1.2,
    )
    return important_df


def build_scheme_summary(base_df: pd.DataFrame, important_df: pd.DataFrame, scheme: str) -> dict[str, pd.DataFrame]:
    base_scheme = base_df[base_df["scheme"] == scheme].copy()
    important_scheme = important_df[important_df["scheme"] == scheme].copy()

    total_rows = int(len(base_scheme))
    important_rows = int(len(important_scheme))
    filtered_out = total_rows - important_rows

    overall = pd.DataFrame(
        [
            ("Total rows", total_rows),
            ("Important failures", important_rows),
            ("Important %", percent(important_rows, total_rows)),
            ("Filtered out", filtered_out),
            ("Filtered out %", percent(filtered_out, total_rows)),
        ],
        columns=["Metric", "Value"],
    )

    severity_rows = []
    for severity in SEVERITY_LABELS:
        count = int(resolve_failure_severity_mask(important_scheme, severity).sum())
        severity_rows.append((severity, count, percent(count, important_rows)))
    severity = pd.DataFrame(severity_rows, columns=["Severity", "Count", "%"])

    model_rows = []
    model_severity_rows = []
    for model in MODELS:
        base_model = base_scheme[base_scheme["model"] == model]
        important_model = important_scheme[important_scheme["model"] == model]
        model_total = int(len(base_model))
        model_important = int(len(important_model))
        model_rows.append((model, model_total, model_important, percent(model_important, model_total)))
        for severity_label in SEVERITY_LABELS:
            count = int(resolve_failure_severity_mask(important_model, severity_label).sum())
            model_severity_rows.append((model, severity_label, count, percent(count, model_important)))

    model_breakdown = pd.DataFrame(model_rows, columns=["Model", "Total", "Important", "%"])
    model_severity = pd.DataFrame(model_severity_rows, columns=["Model", "Severity", "Count", "%"])

    return {
        "overall": overall,
        "severity": severity,
        "model_breakdown": model_breakdown,
        "model_severity": model_severity,
        "total_rows": total_rows,
        "important_rows": important_rows,
    }


def write_section(worksheet, start_row: int, title: str, dataframe: pd.DataFrame) -> int:
    worksheet.cell(row=start_row, column=1, value=title).font = Font(bold=True)
    header_row = start_row + 1
    for column_index, column_name in enumerate(dataframe.columns, start=1):
        cell = worksheet.cell(row=header_row, column=column_index, value=column_name)
        cell.font = Font(bold=True)
    for row_offset, values in enumerate(dataframe.itertuples(index=False), start=1):
        for column_index, value in enumerate(values, start=1):
            worksheet.cell(row=header_row + row_offset, column=column_index, value=value)
    return header_row + len(dataframe) + 2


def autosize(worksheet) -> None:
    for column_cells in worksheet.columns:
        max_width = max((len(str(cell.value)) for cell in column_cells if cell.value is not None), default=0)
        worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_width + 2, 12), 32)


def build_workbook(
    entire_base_df: pd.DataFrame,
    entire_important_df: pd.DataFrame,
    test_base_df: pd.DataFrame,
    test_important_df: pd.DataFrame,
    output_path: Path,
) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)

    overview_rows = []
    datasets = [
        ("entire_data", entire_base_df, entire_important_df),
        ("test_data", test_base_df, test_important_df),
    ]

    for scheme in SCHEMES:
        scheme_number = scheme.replace("scheme ", "scheme_")
        for dataset_label, base_df, important_df in datasets:
            sheet_name = f"{scheme_number}_{dataset_label}"
            worksheet = workbook.create_sheet(title=sheet_name[:31])
            summary = build_scheme_summary(base_df, important_df, scheme)

            row = 1
            row = write_section(worksheet, row, "Section 1: Overall", summary["overall"])
            row = write_section(worksheet, row, "Section 2: Severity breakdown", summary["severity"])
            row = write_section(worksheet, row, "Section 3: Model breakdown", summary["model_breakdown"])
            write_section(worksheet, row, "Section 4: Model severity breakdown", summary["model_severity"])
            autosize(worksheet)

            overview_rows.append(
                (
                    scheme,
                    dataset_label,
                    summary["total_rows"],
                    summary["important_rows"],
                    percent(summary["important_rows"], summary["total_rows"]),
                )
            )

    overview = workbook.create_sheet(title="summary_overview")
    overview_df = pd.DataFrame(overview_rows, columns=["Scheme", "Dataset", "Total", "Important", "%"])
    write_section(overview, 1, "Summary Overview", overview_df)
    autosize(overview)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    workbook.close()


def validate_counts(
    entire_base_df: pd.DataFrame,
    entire_important_df: pd.DataFrame,
    test_base_df: pd.DataFrame,
    test_important_df: pd.DataFrame,
) -> None:
    for scheme in SCHEMES:
        entire_total = int((entire_base_df["scheme"] == scheme).sum())
        entire_important = int((entire_important_df["scheme"] == scheme).sum())
        test_total = int((test_base_df["scheme"] == scheme).sum())
        test_important = int((test_important_df["scheme"] == scheme).sum())

        if entire_important > entire_total:
            raise ValueError(f"Entire-data counts invalid for {scheme}: important exceeds total.")
        if test_important > test_total:
            raise ValueError(f"Test-data counts invalid for {scheme}: important exceeds total.")

        for model in MODELS:
            model_entire_total = int(((entire_base_df["scheme"] == scheme) & (entire_base_df["model"] == model)).sum())
            model_entire_important = int(
                ((entire_important_df["scheme"] == scheme) & (entire_important_df["model"] == model)).sum()
            )
            model_test_total = int(((test_base_df["scheme"] == scheme) & (test_base_df["model"] == model)).sum())
            model_test_important = int(
                ((test_important_df["scheme"] == scheme) & (test_important_df["model"] == model)).sum()
            )
            if model_entire_important > model_entire_total:
                raise ValueError(f"Entire-data model counts invalid for {scheme}/{model}.")
            if model_test_important > model_test_total:
                raise ValueError(f"Test-data model counts invalid for {scheme}/{model}.")


def _safe_float(value):
    return shared_safe_float(value)


def _round_or_none(value: float | None) -> float | None:
    return shared_round_or_none(value)


def _format_float(value: float | None) -> str:
    rounded = _round_or_none(value)
    return "None" if rounded is None else f"{rounded:.4f}"


def _format_candidate_tuple(entry: dict[str, object] | None):
    if entry is None:
        return pd.NA
    return (
        f"({_format_float(_safe_float(entry.get('predicted_time')))}, "
        f"{entry['algo']!r}, "
        f"{_format_float(_safe_float(entry.get('probability')))})"
    )


def _format_fastest_tuple(fastest_time: float | None, fastest_algo: str | None, rank: int | None):
    if fastest_algo is None:
        return pd.NA
    return f"({_format_float(fastest_time)}, {fastest_algo!r}, {rank if rank is not None else 'None'})"


def _format_all_predictions(entries: list[dict[str, object]]):
    formatted_entries = [
        (
            f"({_format_float(_safe_float(entry.get('time_ratio')))}, "
            f"{entry['algo']!r}, "
            f"{_format_float(_safe_float(entry.get('predicted_time')))}, "
            f"{_format_float(_safe_float(entry.get('probability')))})"
        )
        for entry in entries
    ]
    return "[" + ", ".join(formatted_entries) + "]"


def _build_runtime_lookup(project_config_path: Path) -> pd.DataFrame:
    with open(project_config_path, "r", encoding="utf-8") as handle:
        project_config = yaml.safe_load(handle)

    data_cfg = project_config["data"]
    runtime_df = load_excel_data(
        data_cfg["file_path"],
        timeout_replacement=data_cfg["timeout_replacement"],
        y_cols=data_cfg["raw_target_columns"],
        sep=data_cfg["sep"],
        quotechar=data_cfg["quotechar"],
        engine=data_cfg["engine"],
        rename_columns=data_cfg["rename_columns"],
    )
    lookup = runtime_df[["dataset", "k_value", *ALL_ALGOS]].copy()
    lookup["dataset"] = lookup["dataset"].astype(str)
    lookup["k_value"] = pd.to_numeric(lookup["k_value"], errors="raise").astype(int)
    if lookup.duplicated(subset=["dataset", "k_value"]).any():
        duplicates = lookup.loc[lookup.duplicated(subset=["dataset", "k_value"], keep=False), ["dataset", "k_value"]]
        raise ValueError(f"Duplicate runtime lookup keys detected: {duplicates.head(10).to_dict(orient='records')}")
    return lookup.set_index(["dataset", "k_value"])


def _resolve_runtime_row(runtime_lookup: pd.DataFrame, dataset, k_value) -> pd.Series:
    dataset_key = str(dataset)
    k_key = int(pd.to_numeric(pd.Series([k_value]), errors="raise").iloc[0])
    try:
        return runtime_lookup.loc[(dataset_key, k_key)]
    except KeyError as exc:
        raise KeyError(f"Missing runtime lookup row for dataset={dataset_key!r}, k_value={k_key}.") from exc


def _build_sorted_prediction_entries(row: pd.Series, runtime_row: pd.Series) -> list[dict[str, object]]:
    fastest_time = _safe_float(row.get("fastest_time", row.get("fastest_runtime")))
    entries: list[dict[str, object]] = []
    for algo_order, algo in enumerate(ALL_ALGOS):
        probability = _safe_float(row.get(f"prob_{algo}"))
        predicted_time = _safe_float(runtime_row.get(algo))
        time_ratio = None
        if predicted_time is not None and fastest_time not in (None, 0):
            time_ratio = predicted_time / fastest_time
        entries.append(
            {
                "algo": algo,
                "probability": probability,
                "predicted_time": predicted_time,
                "time_ratio": time_ratio,
                "algo_order": algo_order,
            }
        )

    entries.sort(
        key=lambda entry: (
            float("-inf") if entry["probability"] is None else -float(entry["probability"]),
            entry["algo_order"],
        )
    )
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank
    return entries


def _select_best_topk_entry(entries: list[dict[str, object]], top_k: int) -> dict[str, object] | None:
    candidates = [entry for entry in entries[:top_k] if _safe_float(entry.get("time_ratio")) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda entry: (float(entry["time_ratio"]), int(entry["rank"])))


def _build_failure_analysis_row(row: pd.Series, runtime_lookup: pd.DataFrame) -> dict[str, object]:
    runtime_row = _resolve_runtime_row(runtime_lookup, row.get("dataset"), row.get("k_value"))
    sorted_entries = _build_sorted_prediction_entries(row, runtime_row)

    fastest_algo = row.get("fastest_algo")
    fastest_time = _safe_float(row.get("fastest_time", row.get("fastest_runtime")))
    predicted_algo = row.get("predicted_algo")
    predicted_time = _safe_float(row.get("predicted_time"))
    predicted_probability = _safe_float(row.get("predicted_probability"))
    actual_time_ratio = None
    if predicted_time is not None and fastest_time not in (None, 0):
        actual_time_ratio = predicted_time / fastest_time

    fastest_rank = next((int(entry["rank"]) for entry in sorted_entries if entry["algo"] == fastest_algo), None)

    record = {
        "dataset": row.get("dataset"),
        "k_value": int(pd.to_numeric(pd.Series([row.get('k_value')]), errors="raise").iloc[0]),
        "model": row.get("model"),
        "fastest_algo_rank": fastest_rank,
        "fastest_algo_tuple": _format_fastest_tuple(fastest_time, fastest_algo, fastest_rank),
    }

    for top_k in TOP_K_VALUES:
        best_entry = _select_best_topk_entry(sorted_entries, top_k)
        record[f"top{top_k}_min_time_ratio"] = _round_or_none(
            _safe_float(best_entry.get("time_ratio")) if best_entry is not None else None
        )
        record[f"top{top_k}_candidate"] = _format_candidate_tuple(best_entry)

    record["all_predictions_sorted_by_prob"] = _format_all_predictions(sorted_entries)
    return record


def _build_failure_analysis_sheet(df: pd.DataFrame, runtime_lookup: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        columns = [
            "dataset",
            "k_value",
            "model",
            "fastest_algo_rank",
            "fastest_algo_tuple",
        ]
        for top_k in TOP_K_VALUES:
            columns.extend([f"top{top_k}_min_time_ratio", f"top{top_k}_candidate"])
        columns.append("all_predictions_sorted_by_prob")
        return pd.DataFrame(columns=columns)

    records = [_build_failure_analysis_row(row, runtime_lookup) for _, row in df.iterrows()]
    sheet_df = pd.DataFrame(records)
    return sheet_df.sort_values(
        by=["top1_min_time_ratio", "dataset", "k_value", "fastest_algo_rank"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _load_entire_failure_dataframe(workbook_path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(workbook_path, sheet_name=None)
    if "on entire dataset" in sheets:
        return sheets["on entire dataset"].copy()
    for sheet_name, sheet_df in sheets.items():
        if sheet_name != DEFAULT_FILTER_SUMMARY_SHEET:
            return sheet_df.copy()
    raise ValueError(f"No usable entire-dataset sheet found in {workbook_path}.")


def _write_failure_analysis_workbook(sheet_frames: dict[str, pd.DataFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name in [sheet_name for _, _, sheet_name in FAILURE_ANALYSIS_SHEET_ORDER]:
            sheet_frames[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

    workbook = load_workbook(output_path)
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        autosize_columns(worksheet)
    workbook.save(output_path)
    workbook.close()


def _validate_failure_analysis_workbook(
    output_path: Path,
    sheet_frames: dict[str, pd.DataFrame],
) -> dict[str, object]:
    workbook = load_workbook(output_path)
    actual_sheet_names = workbook.sheetnames
    expected_sheet_names = [sheet_name for _, _, sheet_name in FAILURE_ANALYSIS_SHEET_ORDER]
    if actual_sheet_names != expected_sheet_names:
        raise ValueError(f"Failure analysis workbook sheet names mismatch: expected {expected_sheet_names}, found {actual_sheet_names}.")

    highlighted_cells = 0
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        for row in worksheet.iter_rows():
            for cell in row:
                if cell.fill.fill_type not in (None, "none"):
                    highlighted_cells += 1
    workbook.close()
    if highlighted_cells != 0:
        raise ValueError("Failure analysis workbook unexpectedly contains highlighted cells.")

    read_back_sheets = pd.read_excel(output_path, sheet_name=None)
    removed_column_violations: dict[str, list[str]] = {}
    tuple_validation_rows = 0
    for sheet_name in expected_sheet_names:
        exported_df = read_back_sheets[sheet_name]
        forbidden_found = [
            column
            for column in exported_df.columns
            if column in FORBIDDEN_ANALYSIS_COLUMNS or str(column).startswith("prob_")
        ]
        if forbidden_found:
            removed_column_violations[sheet_name] = forbidden_found

        for _, row in exported_df.iterrows():
            if pd.isna(row.get("all_predictions_sorted_by_prob")):
                continue
            all_predictions = ast.literal_eval(str(row["all_predictions_sorted_by_prob"]))
            fastest_tuple = ast.literal_eval(str(row["fastest_algo_tuple"]))
            fastest_rank = int(fastest_tuple[2]) if fastest_tuple[2] != "None" else None
            actual_fastest_rank = next(
                (idx for idx, entry in enumerate(all_predictions, start=1) if entry[1] == fastest_tuple[1]),
                None,
            )
            if fastest_rank != actual_fastest_rank:
                raise ValueError(f"fastest_algo rank mismatch in sheet {sheet_name} for dataset={row['dataset']}, k={row['k_value']}.")

            for top_k in TOP_K_VALUES:
                prefix = all_predictions[:top_k]
                _, best_entry = min(
                    enumerate(prefix, start=1),
                    key=lambda item: (float(item[1][0]), item[0]),
                )
                exported_ratio = _safe_float(row[f"top{top_k}_min_time_ratio"])
                if exported_ratio is None or abs(exported_ratio - float(best_entry[0])) > 1e-9:
                    raise ValueError(f"top{top_k}_min_time_ratio mismatch in sheet {sheet_name} for dataset={row['dataset']}, k={row['k_value']}.")
                exported_candidate = ast.literal_eval(str(row[f"top{top_k}_candidate"]))
                expected_candidate = (round(float(best_entry[2]), 4), best_entry[1], round(float(best_entry[3]), 4))
                if exported_candidate != expected_candidate:
                    raise ValueError(f"top{top_k}_candidate mismatch in sheet {sheet_name} for dataset={row['dataset']}, k={row['k_value']}.")
            tuple_validation_rows += 1

    if removed_column_violations:
        raise ValueError(f"Forbidden columns found in failure analysis workbook: {removed_column_violations}")

    return {
        "sheet_count": len(expected_sheet_names),
        "sheet_names": expected_sheet_names,
        "highlighted_cells": highlighted_cells,
        "validated_rows": tuple_validation_rows,
        "row_counts": {sheet_name: int(len(sheet_frames[sheet_name])) for sheet_name in expected_sheet_names},
    }


def run_summary_mode(args) -> None:
    entire_important_arg = Path(args.entire_important_workbook)
    if entire_important_arg.exists():
        entire_important_path = resolve_explicit_or_existing_path(
            entire_important_arg,
            DEFAULT_ENTIRE_IMPORTANT_CANDIDATES,
            "entire-data important-failures workbook for summary mode",
        )
    else:
        entire_important_path = resolve_existing_path(
            DEFAULT_ENTIRE_IMPORTANT_CANDIDATES,
            "entire-data important-failures workbook for summary mode",
        )
    entire_base_path = resolve_explicit_or_existing_path(
        args.entire_base_workbook,
        DEFAULT_ENTIRE_BASE_CANDIDATES,
        "entire-data base compare workbook",
    )
    test_compare_path = resolve_explicit_or_existing_path(
        args.test_compare_workbook,
        DEFAULT_TEST_COMPARE_CANDIDATES,
        "test-data compare workbook",
    )
    output_path = Path(args.output_path)

    entire_important_df = load_entire_important(entire_important_path)
    entire_base_df = load_entire_base(entire_base_path)
    test_base_df = load_test_compare(test_compare_path)
    test_important_df = compute_important_subset_from_compare(test_base_df)

    validate_counts(entire_base_df, entire_important_df, test_base_df, test_important_df)
    build_workbook(entire_base_df, entire_important_df, test_base_df, test_important_df, output_path)

    print(f"Wrote failure summary workbook to: {output_path}")
    print(f"Entire data source: important={entire_important_path}, base={entire_base_path}")
    print(f"Test data source: compare={test_compare_path}")
    for scheme in SCHEMES:
        print(
            f"{scheme}: "
            f"entire total={int((entire_base_df['scheme'] == scheme).sum())}, "
            f"entire important={int((entire_important_df['scheme'] == scheme).sum())}, "
            f"test total={int((test_base_df['scheme'] == scheme).sum())}, "
            f"test important={int((test_important_df['scheme'] == scheme).sum())}"
        )


def run_enrichment_mode(args) -> None:
    input_path = resolve_explicit_or_existing_path(
        args.failure_workbook,
        DEFAULT_FAILURE_ANALYSIS_INPUT_CANDIDATES,
        "entire-data failure-analysis input workbook",
    )
    output_path = Path(args.enriched_failure_workbook_output or DEFAULT_FAILURE_ANALYSIS_WORKBOOK_OUTPUT_PATH)
    runtime_lookup = _build_runtime_lookup(Path(args.project_config))
    source_df = _load_entire_failure_dataframe(input_path)

    sheet_frames: dict[str, pd.DataFrame] = {}
    for scheme_name, model_name, sheet_name in FAILURE_ANALYSIS_SHEET_ORDER:
        subset = source_df[(source_df["scheme"] == scheme_name) & (source_df["model"] == model_name)].copy()
        sheet_frames[sheet_name] = _build_failure_analysis_sheet(subset, runtime_lookup)

    _write_failure_analysis_workbook(sheet_frames, output_path)
    validation = _validate_failure_analysis_workbook(output_path, sheet_frames)

    print(f"Failure analysis workbook: {output_path}")
    print(f"Source workbook: {input_path}")
    print(
        "Validation: "
        f"sheet_count={validation['sheet_count']}, "
        f"sheet_names={validation['sheet_names']}, "
        f"highlighted_cells={validation['highlighted_cells']}, "
        f"validated_rows={validation['validated_rows']}, "
        f"row_counts={validation['row_counts']}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build failure summary workbooks and enrich failure-analysis outputs.")
    parser.add_argument(
        "--mode",
        default=DEFAULT_ENRICHMENT_MODE,
        choices=["summary", "enrich"],
        help="Run the original entire-vs-test summary workbook flow or build the six-sheet entire-data failure-analysis workbook.",
    )
    parser.add_argument("--entire-important-workbook", default=str(DEFAULT_ENTIRE_IMPORTANT_PATH))
    parser.add_argument("--entire-base-workbook", default=None)
    parser.add_argument("--test-compare-workbook", default=None)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--failure-workbook",
        default=None,
        help="Entire-data important-failures workbook to enrich in --mode enrich.",
    )
    parser.add_argument(
        "--enriched-failure-workbook-output",
        default=str(DEFAULT_FAILURE_ANALYSIS_WORKBOOK_OUTPUT_PATH),
        help="Output path for the six-sheet entire-data failure-analysis workbook.",
    )
    parser.add_argument(
        "--project-config",
        default=str(DEFAULT_PROJECT_CONFIG_PATH),
        help="Project config used to resolve the raw dataset for runtime lookups in the failure-analysis workbook.",
    )
    return parser


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "summary":
        run_summary_mode(args)
        return

    run_enrichment_mode(args)


if __name__ == "__main__":
    main()
