"""Predict Chl-a for the Application Dataset (AD) using row-balanced Parquet inference."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import heapq
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from functions import load_model, predict_frame, sha256


def make_manifest(root, nshards):
    if nshards < 1:
        raise ValueError('shards must be positive')
    entries = [(pq.ParquetFile(path).metadata.num_rows, path.relative_to(root).as_posix())
               for path in sorted(root.rglob('Hylak_id_*.parquet'))]
    if not entries:
        raise ValueError('No application files found')
    heap, rows = [(0, i) for i in range(nshards)], []
    heapq.heapify(heap)
    for count, relative in sorted(entries, key=lambda x: (-x[0], x[1])):
        load, index = heapq.heappop(heap)
        rows.append({'path': relative, 'rows': count, 'shard': index})
        heapq.heappush(heap, (load+count, index))
    return pd.DataFrame(rows)


def apply_file(source, destination, bundle, device, batch_size, model_hash):
    if destination.exists():
        meta = pq.ParquetFile(destination).schema_arrow.metadata or {}
        if meta.get(b'checkpoint_sha256') != model_hash.encode():
            raise ValueError(f'Existing output uses another model: {destination}')
        if pq.ParquetFile(destination).metadata.num_rows != pq.ParquetFile(source).metadata.num_rows:
            raise ValueError(f'Existing output row count differs: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
    writer = None
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=65536):
            frame = batch.to_pandas()
            raw, log = predict_frame(frame, *bundle[:3], device=device, batch_size=batch_size)
            frame = frame.drop(columns=['pred_Chl-a', 'pred_log_Chl-a'], errors='ignore')
            frame['pred_chla'], frame['pred_log10_chla'] = raw, log
            table = pa.Table.from_pandas(frame, preserve_index=False)
            metadata = dict(table.schema.metadata or {})
            metadata[b'checkpoint_sha256'] = model_hash.encode()
            table = table.replace_schema_metadata(metadata)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression='snappy')
            writer.write_table(table)
        if writer is None:
            raise ValueError(f'Empty application file: {source}')
        writer.close(); writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output')
    p.add_argument('--checkpoint')
    p.add_argument('--expected-sha256')
    p.add_argument('--manifest', required=True)
    p.add_argument('--plan', action='store_true')
    p.add_argument('--shards', type=int, default=10)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    root = Path(a.input).resolve()
    if a.plan:
        target = Path(a.manifest)
        if target.exists():
            raise FileExistsError(target)
        manifest = make_manifest(root, a.shards)
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_parquet(target, index=False)
        print(manifest.groupby('shard')['rows'].sum())
        return
    if not a.output or not a.checkpoint or not 0 <= a.shard_index < a.shards:
        p.error('Provide checkpoint/output and a valid shard index')
    output = Path(a.output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError('Input and output must be separate, non-nested directories')
    manifest = pd.read_parquet(a.manifest)
    if manifest.path.duplicated().any() or not manifest.shard.between(0, a.shards-1).all():
        raise ValueError('Invalid manifest')
    bundle = load_model(a.checkpoint, a.device, a.expected_sha256)
    fingerprint = sha256(a.checkpoint)
    for row in manifest[manifest.shard == a.shard_index].itertuples(index=False):
        relative = Path(row.path)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Manifest paths must be relative and remain inside input root')
        source = root/relative
        if pq.ParquetFile(source).metadata.num_rows != row.rows:
            raise ValueError(f'Input changed since planning: {source}')
        apply_file(source, output/relative, bundle, a.device, a.batch_size, fingerprint)
        print(relative, row.rows, flush=True)


if __name__ == '__main__':
    main()
