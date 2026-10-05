"""Preprocess one raw instance for a trained Scheme 1 XGBoost model."""
''' Run command :
python restructured_pipeline/src/preprocess_scheme1_instance.py   
--input-file D:\\Studies\\Research\\Da_Project\\kpex_ML_git\\input_data\\ml-data_1_row.xlsx   
--output-file processed_row.csv
'''
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from restructured_pipeline.src.features import (
    algorithms as ALL_ALGOS,
    feature_cols as SCHEME1_FEATURE_COLS,
    process_kcore_list_to_statistics,
    post_process_data,
)

DEFAULT_PROJECT_CONFIG = Path("restructured_pipeline/configs/project.yaml")

def load_project_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_one_input_row(path: Path, project: dict) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        cfg = project.get("data", {})
        df = pd.read_csv(path, sep=cfg.get("sep", ","), quotechar=cfg.get("quotechar", '"'), engine=cfg.get("engine", "python"))
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError("Input must be .csv, .xlsx, or .xls.")
    if len(df) != 1:
        raise ValueError(f"Expected exactly one data row, found {len(df)}.")
    return df.copy()

def standardize_columns(df: pd.DataFrame, project: dict) -> pd.DataFrame:
    out = df.copy()
    defaults = {"shell_sizes": "kcore_list", "core_number": "core_max", "k": "k_value"}
    configured = project.get("data", {}).get("rename_columns") or {}
    out.rename(columns={**defaults, **configured}, inplace=True)
    return out

def normalize_runtimes(df: pd.DataFrame, project: dict, require_all: bool) -> pd.DataFrame:
    out = df.copy()
    timeout = project.get("data", {}).get("timeout_replacement", 1800)
    present = [a for a in ALL_ALGOS if a in out.columns]
    missing = [a for a in ALL_ALGOS if a not in out.columns]
    if require_all and missing:
        raise ValueError(f"Evaluation mode requires all 24 runtime columns. Missing: {missing}")
    if present:
        out[present] = out[present].replace("x", timeout)
        for col in present:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        bad = [c for c in present if out[c].isna().any() or not np.isfinite(out[c].to_numpy(dtype=float)).all()]
        if bad:
            raise ValueError(f"Invalid runtime values after conversion: {bad}")
        if require_all and out[present].eq(float(timeout)).all(axis=1).iloc[0]:
            raise ValueError(f"All algorithms timed out ({timeout}); canonical preprocessing would remove this row.")
    return out

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    engineered = {"core_min","shell_num","shell_avgSize","shell_maxSize","shell_minSize","shell_stdSize"}
    if "kcore_list" in out.columns:
        process_kcore_list_to_statistics(out, keep_col=False)
    else:
        missing = sorted(engineered - set(out.columns))
        if missing:
            raise ValueError("No kcore_list/shell_sizes and missing engineered features: " + ", ".join(missing))
    out = post_process_data(out)
    missing = [c for c in SCHEME1_FEATURE_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing Scheme 1 features: {missing}")
    for col in SCHEME1_FEATURE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    bad = [c for c in SCHEME1_FEATURE_COLS if out[c].isna().any() or not np.isfinite(out[c].to_numpy(dtype=float)).all()]
    if bad:
        raise ValueError(f"Invalid Scheme 1 feature values: {bad}")
    return out

def preprocess_instance(input_file: Path, output_file: Path, project_config: Path, mode: str) -> pd.DataFrame:
    project = load_project_config(project_config)
    df = read_one_input_row(input_file, project)
    df = standardize_columns(df, project)
    df = normalize_runtimes(df, project, require_all=(mode == "evaluate"))
    df = engineer_features(df)

    if mode == "predict":
        result = df[list(SCHEME1_FEATURE_COLS)].copy()
    else:
        columns = (["dataset"] if "dataset" in df.columns else []) + list(SCHEME1_FEATURE_COLS) + list(ALL_ALGOS)
        result = df[columns].copy()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    print(f"Wrote: {output_file}")
    print("Feature order:", list(SCHEME1_FEATURE_COLS))
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-file", type=Path, required=True)
    p.add_argument("--output-file", type=Path, required=True)
    p.add_argument("--mode", choices=("predict", "evaluate"), default="predict")
    p.add_argument("--project-config", type=Path, default=DEFAULT_PROJECT_CONFIG)
    a = p.parse_args()
    preprocess_instance(a.input_file, a.output_file, a.project_config, a.mode)

if __name__ == "__main__":
    main()
