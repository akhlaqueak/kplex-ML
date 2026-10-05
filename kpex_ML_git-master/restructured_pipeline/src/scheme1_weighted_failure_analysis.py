"""Combine both Scheme 1 weighted XGB splits into one failure-analysis workbook."""
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

from restructured_pipeline.src.failure_analysis_summary import (
    _build_runtime_lookup,
    _load_failure_dataframe,
    _build_failure_analysis_sheet,
    normalize_failure_analysis_metadata,
)
from restructured_pipeline.src.compare_wrong_predictions import autosize_columns

LIGHT_YELLOW = PatternFill(fill_type='solid', start_color='FFF2CC', end_color='FFF2CC')


def _load_important(path: Path, scope: str, split_mode: str) -> pd.DataFrame:
    df = normalize_failure_analysis_metadata(_load_failure_dataframe(path, scope, 0))
    return df[(df['scheme'].eq('scheme 1')) & (df['model'].eq('XGB')) & (df['split_mode'].eq(split_mode))].copy()


def _count_excel_rows(path: Path) -> int:
    sheets = pd.read_excel(path, sheet_name=None)
    frames = [df for df in sheets.values() if not df.empty]
    if not frames:
        return 0
    # Prediction wrong-case files normally contain one data sheet.
    return len(frames[0])


def _highlight_scope_overlap(ws) -> int:
    headers = [c.value for c in ws[1]]
    required = {'dataset', 'k_value', 'data_scope'}
    if not required.issubset(headers):
        return 0
    dcol = headers.index('dataset') + 1
    kcol = headers.index('k_value') + 1
    scol = headers.index('data_scope') + 1
    groups = {}
    for r in range(2, ws.max_row + 1):
        key = (str(ws.cell(r, dcol).value), str(ws.cell(r, kcol).value))
        groups.setdefault(key, []).append((r, str(ws.cell(r, scol).value)))
    highlighted = 0
    for rows in groups.values():
        scopes = {scope for _, scope in rows}
        if {'test_data', 'entire_data'}.issubset(scopes):
            for r, _ in rows:
                for c in range(1, ws.max_column + 1):
                    ws.cell(r, c).fill = LIGHT_YELLOW
                highlighted += 1
    return highlighted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for split_slug in ('strat-y', 'strat-y-k'):
        p.add_argument(f'--{split_slug}-test-important-workbook', type=Path, required=True)
        p.add_argument(f'--{split_slug}-entire-important-workbook', type=Path, required=True)
        p.add_argument(f'--{split_slug}-test-predictions', type=Path, required=True)
        p.add_argument(f'--{split_slug}-entire-predictions', type=Path, required=True)
        p.add_argument(f'--{split_slug}-test-wrong', type=Path, required=True)
        p.add_argument(f'--{split_slug}-entire-wrong', type=Path, required=True)
    p.add_argument('--weight-scheme', required=True)
    p.add_argument('--output-path', type=Path, required=True)
    p.add_argument('--project-config', type=Path, default=Path('restructured_pipeline/configs/project.yaml'))
    a = p.parse_args()

    runtime_lookup = _build_runtime_lookup(a.project_config)
    frames = {}
    summary_rows = []

    specs = [
        ('strat_y', a.strat_y_test_important_workbook, a.strat_y_entire_important_workbook,
         a.strat_y_test_predictions, a.strat_y_entire_predictions, a.strat_y_test_wrong, a.strat_y_entire_wrong),
        ('strat_y_K', a.strat_y_k_test_important_workbook, a.strat_y_k_entire_important_workbook,
         a.strat_y_k_test_predictions, a.strat_y_k_entire_predictions, a.strat_y_k_test_wrong, a.strat_y_k_entire_wrong),
    ]

    for split_mode, test_imp, entire_imp, test_pred, entire_pred, test_wrong, entire_wrong in specs:
        for path in (test_imp, entire_imp, test_pred, entire_pred, test_wrong, entire_wrong):
            if not path.exists():
                raise FileNotFoundError(path)

        test = _load_important(test_imp, 'test_data', split_mode)
        entire = _load_important(entire_imp, 'entire_data', split_mode)
        source = pd.concat([test, entire], ignore_index=True, sort=False)
        detail = _build_failure_analysis_sheet(source, runtime_lookup)
        frames[split_mode] = detail

        for scope, pred_path, wrong_path, important_df in [
            ('test_data', test_pred, test_wrong, test),
            ('entire_data', entire_pred, entire_wrong, entire),
        ]:
            total = len(pd.read_csv(pred_path))
            wrong = _count_excel_rows(wrong_path)
            important = len(important_df)
            summary_rows.append({
                'Stratification': split_mode,
                'Model': 'XGB',
                'Evaluation Scope': scope,
                'Total Predictions': total,
                'Wrong Predictions': wrong,
                'Important Failures': important,
                'Important / Wrong (%)': round(100 * important / wrong, 2) if wrong else 0.0,
                'Important / All (%)': round(100 * important / total, 2) if total else 0.0,
                'scheme': 'scheme 1',
                'weight_scheme': a.weight_scheme,
                'all_scope_important_rows': len(detail),
            })

    summary = pd.DataFrame(summary_rows)
    a.output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(a.output_path, engine='openpyxl') as w:
        summary.to_excel(w, sheet_name='summary', index=False)
        # Excel sheet names are limited to 31 characters.
        frames['strat_y'].to_excel(w, sheet_name='important_strat_y_all_scopes', index=False)
        frames['strat_y_K'].to_excel(w, sheet_name='important_strat_y_K_all_scopes', index=False)

    wb = load_workbook(a.output_path)
    highlighted = 0
    for ws in wb.worksheets:
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = ws.dimensions
        autosize_columns(ws)
        if ws.title.startswith('important_'):
            highlighted += _highlight_scope_overlap(ws)
    wb.save(a.output_path)
    wb.close()
    print(summary.to_string(index=False))
    print(f'Rows highlighted light yellow (same dataset/K in test and entire scopes): {highlighted}')
    print(f'Wrote: {a.output_path}')


if __name__ == '__main__':
    main()
