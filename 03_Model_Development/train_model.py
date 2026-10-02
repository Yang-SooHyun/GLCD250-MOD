"""Train one configuration or execute a Cartesian hyperparameter grid."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import pandas as pd
import torch
from functions import read_config, grid_candidates, train_run, summarize_runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('l3', 'insitu', 'manifest', 'config', 'grid'):
        p.add_argument('--'+name)
    p.add_argument('--select-only', action='store_true', help='Rank executed runs in --output by validation score')
    p.add_argument('--list-only', action='store_true')
    p.add_argument('--run-index', type=int)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='data/output/training')
    a = p.parse_args()
    if a.select_only:
        summarize_runs(Path(a.output))
        return
    config = read_config(a.config)
    runs = grid_candidates(json.loads(Path(a.grid).read_text())) if a.grid else [
        {k: config['hyperparams'][k] for k in ('bs', 'md', 'nh', 'nl', 'lr', 'dr', 'fd')}]
    if a.run_index is not None and not 0 <= a.run_index < len(runs):
        p.error('run-index outside candidate range')
    indexed = list(enumerate(runs)) if a.run_index is None else [(a.run_index, runs[a.run_index])]
    if a.list_only:
        print(pd.DataFrame([{'run_index': i, **hp} for i, hp in indexed]).to_string(index=False))
        print(f'{len(indexed)} planned configurations; no training performed.')
        return
    if not all((a.l3, a.insitu, a.manifest)) or a.epochs < 1:
        p.error('Provide --l3, --insitu, --manifest and positive --epochs')
    for index, hp in indexed:
        result = train_run(config, hp, a, Path(a.output)/f'run_{index:04d}')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
