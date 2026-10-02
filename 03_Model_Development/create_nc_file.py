"""Export prediction Parquets as lake-wise NetCDF; no upload packaging."""
import sys
sys.dont_write_bytecode = True

import os
import time
import re
import numpy as np
import pandas as pd
import netCDF4
import pyarrow.parquet as pq
from pathlib import Path

def write_predictions(temp_nc, paths, mask, dates, xs, ys, lake_id):
    """Write bounded time slabs; never allocate a lake's full 25-year cube.

    Small cubes fit in one slab. Large inputs must be chronologically ordered;
    duplicates retain the last value, as in the original S05.
    """
    mapping = mask[['lon', 'lat', 'sinu_x_GQ', 'sinu_y_GQ']].drop_duplicates()
    if mapping.duplicated(['lon', 'lat']).any():
        raise ValueError(f'Ambiguous geographic to sinusoidal mapping: lake {lake_id}')
    lookup = pd.MultiIndex.from_frame(mapping[['lon', 'lat']])
    mx = np.searchsorted(xs, mapping.sinu_x_GQ.to_numpy())
    my = np.searchsorted(ys, mapping.sinu_y_GQ.to_numpy())
    slab_days = min(len(dates), max(1, (64 * 1024**2) // (len(xs) * len(ys) * 4)))
    slab_start, slab = None, None
    read_rows = finite_rows = 0
    last_log = time.monotonic()
    start_day = dates[0].to_datetime64().astype('datetime64[D]')
    with netCDF4.Dataset(temp_nc, 'a') as nc:
        chunks = (min(slab_days, 32), min(len(ys), 128), min(len(xs), 128))
        var = nc.createVariable('chlor_a', 'f4', ('time', 'YDim_MODIS_Grid_2D', 'XDim_MODIS_Grid_2D'),
                                zlib=True, complevel=2, shuffle=True,
                                fill_value=np.float32(-999), chunksizes=chunks)
        var.setncatts({'standard_name': 'mass_concentration_of_chlorophyll_a_in_sea_water',
                      'long_name': 'Chlorophyll-a Concentration', 'units': 'mg m-3',
                      'coverage_content_type': 'physicalMeasurement',
                      'grid_mapping': 'MODIS_Sinusoidal_Tiling_System'})
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=262144,
                    columns=['date', 'lon', 'lat', 'Hylak_id', 'pred_chla']):
                frame = batch.to_pandas()
                if not frame.Hylak_id.eq(lake_id).all():
                    raise ValueError(f'Wrong lake inside {path}')
                pos = lookup.get_indexer(pd.MultiIndex.from_frame(frame[['lon', 'lat']]))
                ti = (pd.to_datetime(frame.date).to_numpy().astype('datetime64[D]') - start_day).astype('int64')
                if (pos < 0).any() or (ti < 0).any() or (ti >= len(dates)).any():
                    raise ValueError(f'Unmatched coordinates/date in {path}; no rows silently dropped')
                values = frame.pred_chla.to_numpy(dtype=np.float32)
                if np.isinf(values).any() or (values[np.isfinite(values)] <= 0).any():
                    raise ValueError(f'Invalid finite Chl-a in {path}')
                read_rows += len(frame)
                finite_rows += int(np.isfinite(values).sum())
                values = np.where(np.isfinite(values), values, np.float32(-999))
                blocks = ti // slab_days * slab_days
                if np.any(np.diff(blocks) < 0):
                    raise ValueError(f'Unsorted time slabs in {path}')
                boundaries = np.r_[0, np.flatnonzero(np.diff(blocks)) + 1, len(blocks)]
                for lo, hi in zip(boundaries[:-1], boundaries[1:]):
                    new_start = int(blocks[lo])
                    if slab_start is None or new_start != slab_start:
                        if slab_start is not None:
                            if new_start < slab_start:
                                raise ValueError(f'Time order regression in {path}')
                            var[slab_start:slab_start+len(slab)] = slab
                        slab_start = new_start
                        slab = np.full((min(slab_days, len(dates)-slab_start), len(ys), len(xs)),
                                       -999, dtype=np.float32)
                    # Make last-row-wins duplicate handling explicit.
                    flat = ((ti[lo:hi]-slab_start)*len(ys)+my[pos[lo:hi]])*len(xs)+mx[pos[lo:hi]]
                    _, rev = np.unique(flat[::-1], return_index=True)
                    keep = len(flat)-1-rev
                    slab.reshape(-1)[flat[keep]] = values[lo:hi][keep]
                if time.monotonic() - last_log >= 60:
                    print(f'Lake {lake_id:07}: {read_rows:,} prediction rows converted', flush=True)
                    last_log = time.monotonic()
        if slab_start is not None:
            var[slab_start:slab_start+len(slab)] = slab
        nc.setncattr('source_prediction_rows', read_rows)
        nc.setncattr('source_finite_prediction_rows', finite_rows)
        nc.setncattr('conversion_complete', 1)
    return read_rows

def export_lake(path, mask, output_root, model_config=None):
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    prediction_hash = metadata.get(b'checkpoint_sha256', b'').decode('ascii')
    if model_config is not None:
        expected_hash = model_config.get('checkpoint_sha256')
        if not prediction_hash or not expected_hash or prediction_hash != expected_hash:
            raise ValueError(f'Prediction/config checkpoint SHA-256 mismatch or missing hash: {path}')
        # Validate provenance before creating any output directories or files.
        model_epoch = int(model_config['selected_epoch'])
        model_run = model_config['run_name']
    match = re.fullmatch(r"Hylak_id_(\d{7})(?:_(\d{4}))?", Path(path).stem)
    if not match:
        raise ValueError(f"Unexpected input filename: {path}")
    lake = int(match.group(1))
    year = int(match.group(2)) if match.group(2) else None
    if (lake == 1) != (year is not None):
        raise ValueError("Only Lake 1 must use annual input files")
    start, end = pd.Timestamp("2000-02-24"), pd.Timestamp("2024-12-31")
    if year is not None:
        start, end = max(start, pd.Timestamp(year,1,1)), min(end,pd.Timestamp(year,12,31))
    if start > end:
        raise ValueError("Year outside release period")
    dates = pd.date_range(start, end)
    suffix = f'_{year:04d}' if year is not None else ''
    target = Path(output_root)/f'{lake:07d}'/f'GLCD250-MOD_{lake:07d}{suffix}.nc'
    if target.exists():
        raise FileExistsError(target)
    if mask.empty or not mask.Hylak_id.eq(lake).all():
        raise ValueError("Missing or mismatched lake mask")
    xs = np.sort(mask.sinu_x_GQ.unique()).astype(float)
    ys = np.sort(mask.sinu_y_GQ.unique()).astype(float)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    try:
        with netCDF4.Dataset(temporary, "w") as ds:
            dims = ("time", "YDim_MODIS_Grid_2D", "XDim_MODIS_Grid_2D")
            for name, values in zip(dims, (dates, ys, xs)):
                ds.createDimension(name, len(values))
                coord = ds.createVariable(name, "i8" if name == "time" else "f8", (name,))
                if name == "time":
                    coord.units = "days since 2000-02-24"
                    coord.calendar = "gregorian"
                    coord.standard_name = "time"
                    coord.axis = "T"
                    coord[:] = (dates-pd.Timestamp("2000-02-24")).days
                else:
                    coord.units = "m"
                    coord.axis = "Y" if name.startswith("Y") else "X"
                    coord.standard_name = "projection_y_coordinate" if name.startswith("Y") else "projection_x_coordinate"
                    coord[:] = values
            projection = ds.createVariable("MODIS_Sinusoidal_Tiling_System", "i4")
            projection.assignValue(0)
            projection.setncatts(dict(grid_mapping_name="sinusoidal",
                longitude_of_projection_origin=0.0, false_easting=0.0, false_northing=0.0,
                earth_radius=6371007.181))
            ds.setncatts(dict(
                title="GLCD250-MOD: Global Lakes Chlorophyll-a Daily 250 m based on MODIS",
                Conventions="CF-1.8, ACDD-1.3", Hylak_id=lake, id=target.stem,
                spatial_resolution="Nominal 250 m MODIS sinusoidal grid; geographic spacing is not fixed in degrees.",
                time_coverage_start=start.strftime("%Y-%m-%dT00:00:00Z"),
                time_coverage_end=end.strftime("%Y-%m-%dT00:00:00Z"),
                time_coverage_duration=f"P{(end-start).days}D", time_coverage_resolution="P1D",
                geospatial_lat_min=float(mask.lat.min()), geospatial_lat_max=float(mask.lat.max()),
                geospatial_lon_min=float(mask.lon.min()), geospatial_lon_max=float(mask.lon.max()),
                creator_email="ykcha@uos.ac.kr", institution="University of Seoul",
                license="CC-BY-4.0", platform="Terra", instrument="MODIS",
                summary="Available daily lake chlorophyll-a estimates from the transformer-based hierarchical model; missing predictions use -999."))
            if prediction_hash:
                ds.model_checkpoint_sha256 = prediction_hash
            if model_config is not None:
                ds.model_epoch = model_epoch
                ds.model_run = model_run
        write_predictions(temporary, [Path(path)], mask, dates, xs, ys, lake)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import re
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output', 'mask'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--model-config', help='Optional provenance JSON for the predictions being converted')
    p.add_argument('--lake-id', type=int)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    source = Path(a.input)
    paths = [source] if source.is_file() else sorted(source.rglob('Hylak_id_*.parquet'))
    if a.lake_id is not None:
        paths = [f for f in paths if int(f.stem.split('_')[2]) == a.lake_id]
    if a.limit:
        paths = paths[:a.limit]
    if not paths:
        raise ValueError('No matching prediction files')
    mask = pd.read_parquet(a.mask)
    mask = mask.rename(columns={c: c.replace('_GQ','') for c in ('lat_GQ','lon_GQ')
                                if c in mask and c.replace('_GQ','') not in mask})
    groups = mask.groupby('Hylak_id').indices
    config = json.loads(Path(a.model_config).read_text()) if a.model_config else None
    for path in paths:
        lake = int(path.stem.split('_')[2])
        if lake not in groups:
            raise ValueError(f'No mask for lake {lake}')
        print(export_lake(path, mask.iloc[groups[lake]], a.output, config), flush=True)


if __name__ == '__main__':
    main()
