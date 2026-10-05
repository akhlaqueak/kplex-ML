"""Generate consolidated tolerance workbooks from saved experiment outputs.

Two input modes are supported.

1. Legacy saved-run mode
   ---------------------
   Uses run_evaluation() exactly as the original implementation did.

   Example:
       python restructured_pipeline/src/generate_saved_run_tolerance_reports.py \
           --run-tag 03sep2026

2. Existing-predictions mode
   -------------------------
   Reads already-generated all_predictions_*.csv files directly.
   No model is loaded, no inference is run, and prediction files are not
   modified.

   This mode supports separate model/training and downstream-analysis tags.

   Example:
       python restructured_pipeline/src/generate_saved_run_tolerance_reports.py \
           --scheme scheme1 \
           --model-tag 08sep2026 \
           --analysis-tag 09sep2026 \
           --use-existing-predictions

Tolerance metrics are consolidated for 0%, 10%, 20%, and 50%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ---------------------------------------------------------------------
# Allow direct execution from repository root.
# ---------------------------------------------------------------------

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


from restructured_pipeline.src.evaluate import run_evaluation
from restructured_pipeline.src.tolerance_reports import (
    write_tolerance_workbook,
)
from restructured_pipeline.src.train import load_yaml_config


PROJECT_CONFIG = "restructured_pipeline/configs/project.yaml"
SCHEME1_CONFIG = "restructured_pipeline/configs/models_scheme1.yaml"
SCHEME3_CONFIG = "restructured_pipeline/configs/models_scheme3.yaml"
SCHEME4_CONFIG = "restructured_pipeline/configs/models_scheme4.yaml"
SCHEME5_CONFIG = "restructured_pipeline/configs/models_scheme5.yaml"

from restructured_pipeline.src.features import ORIENTED_ALGOS, NON_ORIENTED_ALGOS, B_ALGOS, OTHER_ALGOS

DEFAULT_TOLERANCES = [0.0, 0.1, 0.2, 0.5]

SUPPORTED_SCHEMES = (
    "scheme1",
    "scheme3",
    "scheme4",
    "scheme5",
)

SUPPORTED_FAMILIES = (
    "random_forest",
    "xgboost",
)

SUPPORTED_SPLITS = (
    "strat_y",
    "strat_y_K",
)

SUPPORTED_INFERENCE_MODES = (
    "inference_test_set",
    "inference_entire_data",
)


# =====================================================================
# Shared helpers
# =====================================================================


def get_variant(
    scheme: str,
    family: str,
    split_mode: str,
) -> str | None:
    """Return the Scheme 1 variant name.

    Scheme 3 does not use a variant.
    """

    if scheme != "scheme1":
        return None

    prefix = (
        "rf"
        if family == "random_forest"
        else "xgb"
    )

    return f"{prefix}_{split_mode}"


def get_scope(
    inference_mode: str,
) -> str:
    """Convert inference mode to the short workbook scope label."""

    if inference_mode == "inference_test_set":
        return "test_data"

    if inference_mode == "inference_entire_data":
        return "entire_data"

    raise ValueError(
        f"Unsupported inference mode: {inference_mode}"
    )


def get_model_config(
    scheme: str,
) -> str:
    if scheme == "scheme1":
        return SCHEME1_CONFIG

    if scheme == "scheme3":
        return SCHEME3_CONFIG

    if scheme == "scheme4":
        return SCHEME4_CONFIG

    if scheme == "scheme5":
        return SCHEME5_CONFIG

    raise ValueError(
        f"Unsupported scheme: {scheme}"
    )


def build_saved_run_name(
    scheme: str,
    family: str,
    split_mode: str,
    run_tag: str,
) -> str:
    """Build the legacy training/saved-run name."""

    if scheme == "scheme1":
        prefix = (
            "rf"
            if family == "random_forest"
            else "xgb"
        )

        return (
            f"{scheme}_{family}_"
            f"{prefix}_{split_mode}_{run_tag}"
        )

    return (
        f"{scheme}_{family}_"
        f"{split_mode}_{run_tag}"
    )


# =====================================================================
# Legacy evaluation path
# =====================================================================


def collect_record_from_saved_run(
    scheme: str,
    family: str,
    split_mode: str,
    run_name: str,
    model_config: str,
    inference_mode: str,
) -> dict:
    """Load results using the original run_evaluation workflow."""

    variant = get_variant(
        scheme,
        family,
        split_mode,
    )

    result = run_evaluation(
        scheme=scheme,
        model_family=family,
        variant=variant,
        split_mode=split_mode,
        inference_mode=inference_mode,
        run_name=run_name,
        project_config_path=PROJECT_CONFIG,
        model_config_path=model_config,
        save_files=False,
        tol=0.0,
    )

    scope = get_scope(
        inference_mode
    )

    return {
        "sheet_name":
            f"{split_mode}_{scope}",

        "scheme":
            scheme,

        "model_family":
            family,

        "split_mode":
            split_mode,

        "inference_mode":
            scope,

        "results_df":
            result["results_df"],
    }


# =====================================================================
# Existing prediction CSV path
# =====================================================================


def build_prediction_filename(
    scheme: str,
    family: str,
    split_mode: str,
    model_tag: str,
    analysis_tag: str,
    inference_mode: str,
    scheme4_family: str | None = None,
    scheme5_family: str | None = None,
) -> str:
    """Build the expected all_predictions CSV filename.

    Examples
    --------
    Scheme 1:

        all_predictions_scheme1_xgboost_strat_y_K_
        08sep2026_09sep2026_predict_inference_entire_data.csv

    Scheme 3:

        all_predictions_scheme3_xgboost_strat_y_K_
        03sep2026_09sep2026_predict_inference_entire_data.csv
    """

    if inference_mode == "inference_test_set":
        suffix = "predict_inference_test_data"

    elif inference_mode == "inference_entire_data":
        suffix = "predict_inference_entire_data"

    else:
        raise ValueError(
            f"Unsupported inference mode: {inference_mode}"
        )

    if scheme == "scheme1":

        prefix = (
            "rf"
            if family == "random_forest"
            else "xgb"
        )

        variant = f"{prefix}_{split_mode}"

        return (
            f"all_predictions_"
            f"{scheme}_{family}_{variant}_"
            f"{model_tag}_{analysis_tag}_"
            f"{suffix}.csv"
        )

    if scheme == "scheme3":
        prefix = (
            "rf"
            if family == "random_forest"
            else "xgb"
        )

        variant = f"{prefix}_{split_mode}"
        return (
            f"all_predictions_"
            f"{scheme}_{family}_{variant}_"
            f"{model_tag}_{analysis_tag}_"
            f"{suffix}.csv"
        )


    if scheme == "scheme4":
        if scheme4_family not in {"oriented", "forward"}:
            raise ValueError("Scheme 4 requires scheme4_family='oriented' or 'forward'.")
        variant = f"xgb_{split_mode}"
        return (
            f"all_predictions_scheme4_{scheme4_family}_xgboost_{variant}_"
            f"{model_tag}_{analysis_tag}_{suffix}.csv"
        )

    if scheme == "scheme5":
        if scheme5_family not in {"b", "other"}:
            raise ValueError("Scheme 5 requires scheme5_family='b' or 'other'.")
        prefix = "rf" if family == "random_forest" else "xgb"
        variant = f"{prefix}_{split_mode}"
        return (
            f"all_predictions_scheme5_{scheme5_family}_{family}_{variant}_"
            f"{model_tag}_{analysis_tag}_{suffix}.csv"
        )

    raise ValueError(
        f"Unsupported scheme: {scheme}"
    )


def resolve_prediction_path(
    project: dict,
    scheme: str,
    family: str,
    split_mode: str,
    model_tag: str,
    analysis_tag: str,
    inference_mode: str,
    scheme4_family: str | None = None,
    scheme5_family: str | None = None,
) -> Path:
    """Resolve exactly one existing prediction CSV."""

    predictions_root = Path(
        project["paths"]["pipeline"]["predictions_dir"]
    )

    filename = build_prediction_filename(
        scheme=scheme,
        family=family,
        split_mode=split_mode,
        model_tag=model_tag,
        analysis_tag=analysis_tag,
        inference_mode=inference_mode,
        scheme4_family=scheme4_family,
        scheme5_family=scheme5_family,
    )

    if scheme == "scheme4":
        path = predictions_root / scheme / scheme4_family / family / filename
    elif scheme == "scheme5":
        path = predictions_root / scheme / scheme5_family / family / filename
    else:
        path = predictions_root / scheme / family / filename

    if not path.exists():
        raise FileNotFoundError(
            "\nExisting prediction CSV was not found.\n"
            f"Scheme:       {scheme}\n"
            f"Model:        {family}\n"
            f"Split:        {split_mode}\n"
            f"Model tag:    {model_tag}\n"
            f"Analysis tag: {analysis_tag}\n"
            f"Inference:    {inference_mode}\n"
            f"Expected:     {path}\n"
        )

    return path


def validate_prediction_dataframe(
    results_df: pd.DataFrame,
    prediction_path: Path,
    algorithms: list[str],
) -> None:
    """Perform lightweight validation before tolerance calculation."""

    if results_df.empty:
        raise ValueError(
            f"Prediction CSV is empty: {prediction_path}"
        )

    required_columns = {
        "predicted",
        *algorithms,
    }

    missing = (
        required_columns
        - set(results_df.columns)
    )

    if missing:
        raise ValueError(
            f"Prediction CSV {prediction_path} is missing "
            f"columns required for tolerance evaluation: "
            f"{sorted(missing)}"
        )

    unknown_predictions = (
        set(
            results_df["predicted"]
            .dropna()
            .astype(str)
        )
        - set(algorithms)
    )

    if unknown_predictions:
        raise ValueError(
            f"Prediction CSV {prediction_path} contains "
            f"predicted algorithms outside the configured "
            f"algorithm list: "
            f"{sorted(unknown_predictions)}"
        )


def collect_record_from_prediction_csv(
    project: dict,
    algorithms: list[str],
    scheme: str,
    family: str,
    split_mode: str,
    model_tag: str,
    analysis_tag: str,
    inference_mode: str,
    scheme4_family: str | None = None,
    scheme5_family: str | None = None,
) -> dict:
    """Read an existing prediction CSV without running evaluation."""

    prediction_path = resolve_prediction_path(
        project=project,
        scheme=scheme,
        family=family,
        split_mode=split_mode,
        model_tag=model_tag,
        analysis_tag=analysis_tag,
        inference_mode=inference_mode,
        scheme4_family=scheme4_family,
        scheme5_family=scheme5_family,
    )

    print(
        f"Reading existing predictions:\n"
        f"  {prediction_path}"
    )

    results_df = pd.read_csv(
        prediction_path
    )

    validate_prediction_dataframe(
        results_df,
        prediction_path,
        algorithms,
    )

    scope = get_scope(
        inference_mode
    )

    return {
        "sheet_name":
            f"{split_mode}_{scope}",

        "scheme":
            scheme,

        "model_family":
            family,

        "scheme4_family":
            scheme4_family,

        "scheme5_family":
            scheme5_family,

        "split_mode":
            split_mode,

        "inference_mode":
            scope,

        "prediction_path":
            str(prediction_path),

        "results_df":
            results_df,
    }


# =====================================================================
# Job selection
# =====================================================================


def build_jobs(
    requested_scheme: str,
) -> list[tuple[str, str, str, str | None]]:
    """Return scheme/model/subfamily jobs requested by the CLI."""

    all_jobs = [
        ("scheme1", "random_forest", SCHEME1_CONFIG, None, None),
        ("scheme1", "xgboost", SCHEME1_CONFIG, None, None),
        ("scheme3", "random_forest", SCHEME3_CONFIG, None, None),
        ("scheme3", "xgboost", SCHEME3_CONFIG, None, None),
        ("scheme4", "xgboost", SCHEME4_CONFIG, "oriented", None),
        ("scheme4", "xgboost", SCHEME4_CONFIG, "forward", None),
        ("scheme5", "random_forest", SCHEME5_CONFIG, None, "b"),
        ("scheme5", "xgboost", SCHEME5_CONFIG, None, "b"),
        ("scheme5", "random_forest", SCHEME5_CONFIG, None, "other"),
        ("scheme5", "xgboost", SCHEME5_CONFIG, None, "other"),
    ]

    if requested_scheme == "all":
        return all_jobs
    return [job for job in all_jobs if job[0] == requested_scheme]


def resolve_model_tag(
    scheme: str,
    args,
) -> str:
    """Resolve the training/model provenance tag for a scheme."""

    # Scheme-specific tags take precedence.
    if scheme == "scheme1":
        if args.scheme1_model_tag:
            return args.scheme1_model_tag

    if scheme == "scheme3":
        if args.scheme3_model_tag:
            return args.scheme3_model_tag

    if scheme == "scheme4":
        if args.scheme4_model_tag:
            return args.scheme4_model_tag

    if scheme == "scheme5":
        if args.scheme5_model_tag:
            return args.scheme5_model_tag

    # Generic model tag comes next.
    if args.model_tag:
        return args.model_tag

    # For backwards compatibility, fall back to run-tag.
    return args.run_tag


# =====================================================================
# Scheme 5 algorithm-level classification metrics
# =====================================================================

def build_algorithm_level_metric_row(results_df, algorithms, split_mode, model_family, scope):
    """Compute ordinary multiclass metrics over the active Scheme 5 algorithm vocabulary."""
    if "predicted" not in results_df.columns:
        raise ValueError("Prediction dataframe is missing 'predicted'.")

    # Canonical true label: first active algorithm (in configured order) attaining
    # the minimum runtime. This matches the single-label classification view used
    # for precision/recall/F1; tie-aware runtime analyses remain unchanged elsewhere.
    def canonical_true(row):
        best = min(float(row[a]) for a in algorithms)
        return next(a for a in algorithms if float(row[a]) == best)

    y_true = results_df.apply(canonical_true, axis=1)
    y_pred = results_df["predicted"].astype(str)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=algorithms, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=algorithms, average="weighted", zero_division=0
    )
    return {
        "split_mode": split_mode,
        "model": "RF" if model_family == "random_forest" else "XGB",
        "scope": scope,
        "rows": int(len(results_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }


def write_scheme5_algorithm_metrics(metric_rows, model_tag, analysis_tag):
    """Write one report workbook with independent B6 and Other18 summary sheets."""
    if not metric_rows:
        return None
    output_dir = Path("restructured_pipeline/outputs/reports/scheme5")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"scheme5_algorithm_level_metrics_{model_tag}_{analysis_tag}.xlsx"
    all_rows = pd.DataFrame(metric_rows)
    order = [
        "split_mode", "model", "scope", "rows", "accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_precision", "weighted_recall", "weighted_f1",
    ]
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for family, sheet in (("b", "B6"), ("other", "Other18")):
            frame = all_rows.loc[all_rows["scheme5_family"].eq(family), order].copy()
            frame["_split_order"] = frame["split_mode"].map({"strat_y": 0, "strat_y_K": 1})
            frame["_model_order"] = frame["model"].map({"RF": 0, "XGB": 1})
            frame["_scope_order"] = frame["scope"].map({"test_data": 0, "entire_data": 1})
            frame = frame.sort_values(["_split_order", "_model_order", "_scope_order"], kind="stable")
            frame = frame.drop(columns=["_split_order", "_model_order", "_scope_order"])
            frame.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Wrote Scheme 5 algorithm-level metrics: {output_path}")
    return output_path


# =====================================================================
# CLI
# =====================================================================


def build_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Generate consolidated tolerance workbooks from "
            "saved runs or existing all_predictions CSV files."
        )
    )

    # -------------------------------------------------------------
    # Original argument.
    #
    # Kept for full backwards compatibility.
    # -------------------------------------------------------------

    parser.add_argument(
        "--run-tag",
        default="03sep2026",
        help=(
            "Legacy saved-run tag. In legacy mode this is used "
            "to construct saved run names. It is also the fallback "
            "model tag in existing-predictions mode."
        ),
    )

    # -------------------------------------------------------------
    # Optional scheme restriction.
    # -------------------------------------------------------------

    parser.add_argument(
        "--scheme",
        choices=[
            "all",
            "scheme1",
            "scheme3",
            "scheme4",
            "scheme5",
        ],
        default="all",
        help=(
            "Generate reports for all supported schemes or restrict "
            "generation to Scheme 1, Scheme 3, or Scheme 4."
        ),
    )

    # -------------------------------------------------------------
    # New mode.
    # -------------------------------------------------------------

    parser.add_argument(
        "--use-existing-predictions",
        action="store_true",
        help=(
            "Read existing all_predictions CSV files directly. "
            "No model loading, inference, or prediction generation "
            "is performed."
        ),
    )

    # -------------------------------------------------------------
    # Generic model/analysis provenance.
    # -------------------------------------------------------------

    parser.add_argument(
        "--model-tag",
        default=None,
        help=(
            "Training/model provenance tag used when resolving "
            "existing prediction filenames."
        ),
    )

    parser.add_argument(
        "--analysis-tag",
        default=None,
        help=(
            "Downstream prediction/analysis tag used in existing "
            "prediction filenames, e.g. 09sep2026."
        ),
    )

    # -------------------------------------------------------------
    # Scheme-specific model tags.
    #
    # These allow a future all-scheme command such as:
    #
    #   Scheme 1 model = 08sep
    #   Scheme 3 model = 03sep
    #   analysis       = 09sep
    # -------------------------------------------------------------

    parser.add_argument(
        "--scheme1-model-tag",
        default=None,
        help=(
            "Optional Scheme 1-specific model provenance tag. "
            "Overrides --model-tag for Scheme 1."
        ),
    )

    parser.add_argument(
        "--scheme3-model-tag",
        default=None,
        help=(
            "Optional Scheme 3-specific model provenance tag. "
            "Overrides --model-tag for Scheme 3."
        ),
    )

    parser.add_argument(
        "--scheme4-model-tag", default=None, help="Optional Scheme 4 training/model provenance tag."
    )
    parser.add_argument(
        "--scheme5-model-tag", default=None, help="Optional Scheme 5 training/model provenance tag."
    )

    return parser


# =====================================================================
# Main
# =====================================================================


def main():

    parser = build_arg_parser()
    args = parser.parse_args()

    project = load_yaml_config(
        PROJECT_CONFIG
    )

    default_algorithms = list(
        project["targets"]["scheme1"]["algorithms"]
    )

    epsilon = project[
        "evaluation"
    ]["epsilon"]

    tolerances = list(
        DEFAULT_TOLERANCES
    )

    jobs = build_jobs(
        args.scheme
    )

    # -------------------------------------------------------------
    # Validate new mode.
    # -------------------------------------------------------------

    if args.use_existing_predictions:

        if not args.analysis_tag:
            parser.error(
                "--analysis-tag is required when "
                "--use-existing-predictions is used."
            )

        print(
            "\n"
            "============================================================\n"
            "EXISTING-PREDICTIONS MODE\n"
            "============================================================"
        )

        print(
            "No model will be loaded."
        )

        print(
            "No inference will be performed."
        )

        print(
            "Existing prediction CSV files will only be read."
        )

        print(
            f"Analysis tag: {args.analysis_tag}"
        )

    else:

        print(
            "\n"
            "============================================================\n"
            "LEGACY SAVED-RUN MODE\n"
            "============================================================"
        )

        print(
            f"Run tag: {args.run_tag}"
        )

    scheme5_algorithm_metric_rows = []

    # -------------------------------------------------------------
    # Generate one tolerance workbook per scheme/model family,
    # preserving the behavior of the original implementation.
    # -------------------------------------------------------------

    for scheme, family, config, scheme4_family, scheme5_family in jobs:

        records = []

        if scheme == "scheme4":
            active_algorithms = list(ORIENTED_ALGOS if scheme4_family == "oriented" else NON_ORIENTED_ALGOS)
            if len(active_algorithms) != 12:
                raise ValueError(f"Scheme 4 {scheme4_family} must contain exactly 12 algorithms; found {len(active_algorithms)}.")
        elif scheme == "scheme5":
            active_algorithms = list(B_ALGOS if scheme5_family == "b" else OTHER_ALGOS)
            expected = 6 if scheme5_family == "b" else 18
            if len(active_algorithms) != expected:
                raise ValueError(f"Scheme 5 {scheme5_family} must contain exactly {expected} algorithms; found {len(active_algorithms)}.")
        else:
            active_algorithms = default_algorithms

        model_tag = resolve_model_tag(
            scheme,
            args,
        )

        print()
        print(
            "============================================================"
        )

        print(
            f"{scheme} / {(scheme4_family or scheme5_family) + ' / ' if (scheme4_family or scheme5_family) else ''}{family}"
        )

        if args.use_existing_predictions:

            print(
                f"Model tag:    {model_tag}"
            )

            print(
                f"Analysis tag: {args.analysis_tag}"
            )

        print(
            "============================================================"
        )

        for split_mode in SUPPORTED_SPLITS:

            if args.use_existing_predictions:

                for inference_mode in (
                    SUPPORTED_INFERENCE_MODES
                ):

                    records.append(
                        collect_record_from_prediction_csv(
                            project=project,
                            algorithms=active_algorithms,
                            scheme=scheme,
                            family=family,
                            split_mode=split_mode,
                            model_tag=model_tag,
                            analysis_tag=args.analysis_tag,
                            inference_mode=inference_mode,
                            scheme4_family=scheme4_family,
                            scheme5_family=scheme5_family,
                        )
                    )

            else:

                run_name = build_saved_run_name(
                    scheme=scheme,
                    family=family,
                    split_mode=split_mode,
                    run_tag=args.run_tag,
                )

                for inference_mode in (
                    SUPPORTED_INFERENCE_MODES
                ):

                    print(
                        f"Loading "
                        f"{run_name} / "
                        f"{inference_mode}"
                    )

                    records.append(
                        collect_record_from_saved_run(
                            scheme=scheme,
                            family=family,
                            split_mode=split_mode,
                            run_name=run_name,
                            model_config=config,
                            inference_mode=inference_mode,
                        )
                    )

        # ---------------------------------------------------------
        # Output location.
        #
        # Preserve original naming in legacy mode.
        #
        # In existing-predictions mode, include both provenance
        # tags so the workbook is self-describing.
        # ---------------------------------------------------------

        output_dir = (
            Path(
                project[
                    "paths"
                ][
                    "pipeline"
                ][
                    "metrics_dir"
                ]
            )
            / scheme
            / ((scheme4_family if scheme == "scheme4" else scheme5_family) if scheme in {"scheme4", "scheme5"} else family)
            / (family if scheme in {"scheme4", "scheme5"} else "")
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if args.use_existing_predictions:

            output_name = (
                "model_performance_for_all_exp_tolerance_"
                f"{model_tag}_"
                f"{args.analysis_tag}.xlsx"
            )

        else:

            output_name = (
                "model_performance_for_all_exp_tolerance_"
                f"{args.run_tag}.xlsx"
            )

        output_path = (
            output_dir
            / output_name
        )

        print()
        print(
            f"Writing tolerance workbook:\n"
            f"  {output_path}"
        )

        write_tolerance_workbook(
            experiment_records=records,
            tolerance_values=tolerances,
            output_path=output_path,
            algorithms=active_algorithms,
            epsilon=epsilon,
        )

        if scheme == "scheme5" and args.use_existing_predictions:
            for record in records:
                scheme5_algorithm_metric_rows.append({
                    "scheme5_family": scheme5_family,
                    **build_algorithm_level_metric_row(
                        record["results_df"], active_algorithms,
                        record["split_mode"], family, record["inference_mode"],
                    ),
                })

        print(
            f"Wrote: {output_path}"
        )


    if args.use_existing_predictions and args.scheme in {"scheme5", "all"}:
        scheme5_tag = resolve_model_tag("scheme5", args)
        write_scheme5_algorithm_metrics(
            scheme5_algorithm_metric_rows, scheme5_tag, args.analysis_tag
        )


if __name__ == "__main__":
    main()