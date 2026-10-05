"""Build one tolerance workbook containing both Scheme 1 weighted XGB split strategies."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from restructured_pipeline.src.features import algorithms as ALL_ALGOS
from restructured_pipeline.src.tolerance_reports import write_tolerance_workbook
from restructured_pipeline.src.train import load_yaml_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--strat-y-test-predictions', type=Path, required=True)
    p.add_argument('--strat-y-entire-predictions', type=Path, required=True)
    p.add_argument('--strat-y-k-test-predictions', type=Path, required=True)
    p.add_argument('--strat-y-k-entire-predictions', type=Path, required=True)
    p.add_argument('--weight-scheme', required=True)
    p.add_argument('--output-path', type=Path, required=True)
    p.add_argument('--project-config', default='restructured_pipeline/configs/project.yaml')
    a = p.parse_args()

    jobs = [
        ('strat_y', 'test_data', a.strat_y_test_predictions),
        ('strat_y', 'entire_data', a.strat_y_entire_predictions),
        ('strat_y_K', 'test_data', a.strat_y_k_test_predictions),
        ('strat_y_K', 'entire_data', a.strat_y_k_entire_predictions),
    ]
    for _, _, path in jobs:
        if not path.exists():
            raise FileNotFoundError(path)

    project = load_yaml_config(a.project_config)
    epsilon = project['evaluation']['epsilon']
    records = []
    for split_mode, scope, path in jobs:
        records.append({
            'sheet_name': f'{split_mode}_{scope}',
            'scheme': 'scheme1',
            'model_family': 'xgboost',
            'split_mode': split_mode,
            'inference_mode': scope,
            'algorithms': list(ALL_ALGOS),
            'results_df': pd.read_csv(path),
        })

    a.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_tolerance_workbook(
        records,
        [0.0, 0.1, 0.2, 0.5],
        a.output_path,
        algorithms=list(ALL_ALGOS),
        epsilon=epsilon,
        summary_sheet_name='tolerance_eval_all_exp',
    )
    print(f'Weight scheme: {a.weight_scheme}')
    print(f'Wrote: {a.output_path}')


if __name__ == '__main__':
    main()
