"""Standalone Scheme 1 inference: raw instance(s) -> best algorithm, using a saved .pkl model.

No ground-truth runtime columns are required. Reuses the same feature-engineering
functions as training/evaluation (features.py) and the same single-row cleanup
used by preprocess_scheme1_instance.py, so the feature vector fed to the model is
identical to what the model was trained/evaluated on.

Input file columns expected (CSV or XLSX), one row per instance to predict:
    dataset, k (or k_value), n, m, d_avg, d_max, core_number (or core_max),
    shell_sizes (or kcore_list)

Run:
    python restructured_pipeline/src/predict_scheme1_from_pkl.py \
        --input-file my_instances.csv \
        --model-path model_scheme1_xgboost_xgb_strat_y_weight_log_time_min_plus_log_time_ratio_23sep2026.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from restructured_pipeline.src.preprocess_scheme1_instance import (
    load_project_config,
    standardize_columns,
    engineer_features,
    DEFAULT_PROJECT_CONFIG,
)
from restructured_pipeline.src.features import (
    feature_cols as DEFAULT_FEATURE_COLS,
    algorithms as DEFAULT_ALGORITHMS,
)


def read_input_rows(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("Input must be .csv, .xlsx, or .xls.")


def load_artifact(model_path: Path) -> dict:
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    if not isinstance(artifact, dict) or "classifier" not in artifact:
        raise ValueError(f"Unexpected model artifact format at {model_path}: {type(artifact)}")
    return artifact


def predict_instances(input_file: Path, model_path: Path, project_config_path: Path, top_k: int = 3) -> pd.DataFrame:
    project = load_project_config(project_config_path)

    raw_df = read_input_rows(input_file)
    df = standardize_columns(raw_df, project)
    df = engineer_features(df)  # adds core_min/shell_* and validates feature_cols

    artifact = load_artifact(model_path)
    clf = artifact["classifier"]
    le = artifact.get("label_encoder")
    features = artifact.get("feature_cols") or list(DEFAULT_FEATURE_COLS)
    algorithms = artifact.get("algorithms") or list(DEFAULT_ALGORITHMS)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Preprocessed instance(s) missing model features: {missing}")

    X = df[features]
    pred_idx = clf.predict(X)
    predicted = le.inverse_transform(pred_idx) if le is not None else pred_idx

    out = pd.DataFrame(index=df.index)
    if "dataset" in df.columns:
        out["dataset"] = df["dataset"]
    if "k_value" in df.columns:
        out["k_value"] = df["k_value"]
    out["predicted_best_algorithm"] = predicted

    if hasattr(clf, "predict_proba"):
        class_labels = list(le.classes_) if le is not None else list(clf.classes_)
        probas = clf.predict_proba(X)
        for row_pos, row_idx in enumerate(df.index):
            ranked = sorted(zip(class_labels, probas[row_pos]), key=lambda x: x[1], reverse=True)
            out.loc[row_idx, "top_k_predictions"] = str(ranked[:top_k])

    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True, help="CSV/XLSX with one or more raw instances.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the saved .pkl model artifact.")
    parser.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-file", type=Path, default=None, help="Optional CSV path to also save the predictions.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    result = predict_instances(args.input_file, args.model_path, args.project_config, top_k=args.top_k)
    print(result.to_string(index=False))
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_file, index=False)
        print(f"\nWrote: {args.output_file}")


if __name__ == "__main__":
    main()
