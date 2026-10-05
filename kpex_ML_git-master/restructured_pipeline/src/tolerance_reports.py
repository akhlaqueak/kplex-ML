"""Tolerance-based workbook exporters for experiment-matrix runs."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from restructured_pipeline.src.evaluate import compute_top_k_metrics
from restructured_pipeline.src.metrics import evaluate_time_metrics_with_tol, evaluate_with_tolerance
from restructured_pipeline.src.write_utils import consolidate_experiments, write_metrics_to_excel_file

def _resolve_record_algorithms(record, default_algorithms):
    active = record.get("algorithms")
    if active is None:
        active = default_algorithms
    active = list(active)
    if not active:
        raise ValueError(f"No algorithms resolved for {record.get('sheet_name')!r}.")
    return active

def build_tolerance_records(results_df, algorithms, tolerance_values, epsilon=1e-3):
    records=[]
    for tol in tolerance_values:
        try: cls_metrics=evaluate_with_tolerance(results_df, algorithms, tol=tol)
        except ValueError: cls_metrics=None
        time_metrics=evaluate_time_metrics_with_tol(results_df, algorithms, tol=tol, epsilon=epsilon)
        top_k_metrics=compute_top_k_metrics(results_df, algorithms, tol=tol)
        records.append({"tolerance":tol,"accuracy_with_tolerance":None if cls_metrics is None else cls_metrics["accuracy"],"time_ratio_with_tolerance":time_metrics["time_ratio"],"total_time_pred_with_tolerance":time_metrics["total_time_pred"],"total_time_opt_with_tolerance":time_metrics["total_time_opt"],"top1_accuracy":top_k_metrics["top1_accuracy"],"top2_accuracy":top_k_metrics["top2_accuracy"],"top3_accuracy":top_k_metrics["top3_accuracy"],"top1_accuracy_with_tolerance":top_k_metrics["top1_accuracy_with_tolerance"],"top2_accuracy_with_tolerance":top_k_metrics["top2_accuracy_with_tolerance"],"top3_accuracy_with_tolerance":top_k_metrics["top3_accuracy_with_tolerance"]})#"avg_overhead_per_case_with_tolerance":time_metrics["avg_overhead_per_case"]
    return records

def build_tolerance_sheet_name(experiment_sheet_name,tol): return f"{experiment_sheet_name}_tol_{int(round(tol*100))}"

def write_tolerance_workbook(experiment_records,tolerance_values,output_path,algorithms=None,epsilon=1e-3,summary_sheet_name="tolerance_eval_all_exp"):
    output_path=Path(output_path); output_path.parent.mkdir(parents=True,exist_ok=True)
    if output_path.exists(): output_path.unlink()
    summary_rows=[]
    for record in experiment_records:
        active_algorithms=_resolve_record_algorithms(record, algorithms)
        tolerance_records=build_tolerance_records(record["results_df"],active_algorithms,tolerance_values,epsilon=epsilon)
        for row in tolerance_records:
            sheet_name=build_tolerance_sheet_name(record["sheet_name"],row["tolerance"])
            try: cls_metrics=evaluate_with_tolerance(record["results_df"],active_algorithms,tol=row["tolerance"])
            except ValueError: cls_metrics={"accuracy":row["accuracy_with_tolerance"],"precision_per_alg":{},"recall_per_alg":{}}
            time_metrics=evaluate_time_metrics_with_tol(record["results_df"],active_algorithms,tol=row["tolerance"],epsilon=epsilon)
            write_metrics_to_excel_file(cls_metrics,time_metrics,str(output_path),sheet_name=sheet_name)
            summary_rows.append({"experiment_sheet":record["sheet_name"],"scheme":record["scheme"],"scheme4_family":record.get("scheme4_family"),"scheme5_family":record.get("scheme5_family"),"model_family":record["model_family"],"split_mode":record["split_mode"],"inference_mode":record["inference_mode"],"algorithm_count":len(active_algorithms),**row})
        consolidate_experiments(str(output_path),prefix=f"{record['sheet_name']}_tol_",master_name=record["sheet_name"],delete_source_sheets=True)
    with pd.ExcelWriter(output_path,engine="openpyxl",mode="a",if_sheet_exists="replace") as writer:
        pd.DataFrame(summary_rows).to_excel(writer,sheet_name=summary_sheet_name,index=False)
    return str(output_path)

__all__=["build_tolerance_records","build_tolerance_sheet_name","write_tolerance_workbook"]
