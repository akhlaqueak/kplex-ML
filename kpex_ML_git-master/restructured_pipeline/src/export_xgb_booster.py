"""Export a saved Scheme 1 XGBoost .pkl artifact to a plain booster + metadata pair
that UMKPEX/MLPredictor.h can load directly through the XGBoost C API, with no
Python/sklearn dependency at inference time.

Run once per trained model:
    python restructured_pipeline/src/export_xgb_booster.py \
        --model-path model_scheme1_xgboost_xgb_strat_y_weight_log_time_min_plus_log_time_ratio_23sep2026.pkl \
        --output-dir UMKPEX/ml_export

Writes:
    <output-dir>/model.json      -- the raw XGBoost booster (XGBoosterLoadModel-compatible)
    <output-dir>/model_meta.txt  -- two lines: "classes:<label1>,<label2>,..." in the
                                    model's class-index order, and
                                    "feature_cols:<name1>,<name2>,..." in the exact
                                    column order the booster was trained on.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def export_booster(model_path: Path, output_dir: Path) -> None:
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    clf = artifact["classifier"]
    booster = clf.get_booster() if hasattr(clf, "get_booster") else clf

    le = artifact.get("label_encoder")
    classes = list(le.classes_) if le is not None else list(getattr(clf, "classes_", []))
    feature_cols = list(artifact.get("feature_cols") or [])

    if not classes:
        raise ValueError("Could not resolve class labels (no label_encoder/classes_ on the artifact).")
    if not feature_cols:
        raise ValueError("Artifact has no feature_cols; cannot guarantee the C++ side uses the same column order.")

    output_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(output_dir / "model.json"))

    with open(output_dir / "model_meta.txt", "w") as f:
        f.write("classes:" + ",".join(classes) + "\n")
        f.write("feature_cols:" + ",".join(feature_cols) + "\n")

    print(f"Wrote: {output_dir / 'model.json'}")
    print(f"Wrote: {output_dir / 'model_meta.txt'}")
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Feature order ({len(feature_cols)}): {feature_cols}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main():
    args = build_arg_parser().parse_args()
    export_booster(args.model_path, args.output_dir)


if __name__ == "__main__":
    main()
