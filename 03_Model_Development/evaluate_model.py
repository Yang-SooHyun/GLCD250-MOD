"""Export in-situ split predictions; optionally evaluate all L3 splits."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
import torch
from functions import (load_model, insitu_loaders, _insitu_bag_loss,
                       _split_70_10_20, _make_seq_array, chla_metrics, evaluation_seed)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'insitu', 'manifest', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--l3', help='Optional curated L3-TD parquet')
    p.add_argument('--seed', type=int, help='Must match checkpoint split seed; normally inferred automatically')
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    model, scalers, columns, checkpoint = load_model(a.checkpoint, a.device)
    split_seed = evaluation_seed(checkpoint, a.seed) if a.l3 else None
    loaders, manifest = insitu_loaders(pd.read_parquet(a.insitu), a.manifest, columns, scalers)
    out.mkdir(parents=True)
    results = []
    with torch.inference_mode():
        for split, loader in loaders.items():
            frames = []
            for bag in loader:
                _, predicted, observed = _insitu_bag_loss(model, bag, torch.nn.MSELoss(), a.device)
                frames.append(pd.DataFrame({'observation_id': bag[-1].numpy(),
                    'observed_chla': np.power(10., observed.cpu().numpy().ravel()),
                    'pred_chla': np.power(10., predicted.cpu().numpy().ravel())}))
            frame = pd.concat(frames, ignore_index=True).merge(
                manifest[['observation_id', 'Hylak_id']], on='observation_id', validate='one_to_one')
            frame.to_parquet(out/f'insitu_{split}.parquet', index=False)
            results.append({'dataset': 'insitu', 'split': split,
                            **chla_metrics(frame.observed_chla, frame.pred_chla)})
        if a.l3:
            for split, frame in zip(('train', 'validation', 'test'), _split_70_10_20(pd.read_parquet(a.l3), split_seed)):
                predictions = []
                for start in range(0, len(frame), 4096):
                    part = frame.iloc[start:start+4096]
                    seq = torch.from_numpy(_make_seq_array(part, columns['seq_input'])).to(a.device)
                    aux = torch.from_numpy(scalers['aux_input'].transform(
                        part[columns['aux_input']].to_numpy()).astype(np.float32)).to(a.device)
                    _, pred = model(seq, aux)
                    predictions.append(np.power(10., scalers['target'].inverse_transform(pred.cpu().numpy()).ravel()))
                results.append({'dataset': 'L3', 'split': split,
                                **chla_metrics(frame['Chl-a'], np.concatenate(predictions))})
    pd.DataFrame(results).to_csv(out/'metrics.csv', index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == '__main__':
    main()
