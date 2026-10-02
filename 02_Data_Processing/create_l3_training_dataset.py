"""Aggregate the AD and match recalibrated Terra L3 to construct the L3-TD."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflow_utils import configure_paths
from workflow_utils import add_rrs_bio_optical_features
from workflow_utils import L3_RRS_VARS
from workflow_utils import record_event, atomic_parquet, require_matching_files
import os
import gc
import glob
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Configuration settings for file paths and pixel coverage filtering
CONFIG = {
    "MASK_250M_PATH": "data/inventory/df_masking_250m.parquet",
    "MASK_4_6KM_PATH": "data/inventory/df_masking_4_6km.parquet",
    "MASK_TOTAL_PATH": "data/inventory/df_masking_total.parquet",
    "INPUT_250M_DIR": "data/output/250m_resolution_dataset_daily",
    "INPUT_TERRA_4KM_DIR": "data/intermediate/DB_pixels_TERRA_4km_recalibrated",
    "OUTPUT_MOD09_4KM_DIR": "data/output/MOD09_4km_daily",
    "OUTPUT_FINAL_DIR": "data/output/4.6km_resolution_dataset",
    "MIN_COUNT_RATIO": 0.5,
}

# Define variable groups for data processing and output
INFO_VARS = ["date", "lon", "lat", "Hylak_id", "count", "max_count", "count_ratio"]
SR_VARS = ["SR_645", "SR_859", "SR_469", "SR_555", "SR_1240", "SR_1640", "SR_2130"]
SR_TO_RRS_VARS = ["SR_645_Rrs", "SR_859_Rrs", "SR_469_Rrs", "SR_555_Rrs", "SR_1240_Rrs", "SR_1640_Rrs", "SR_2130_Rrs"]
INDEX_VARS = ["Ratio_blue", "Ratio_blue_SR_Rrs", "FAI", "FAI_SR_Rrs", "NDVI", "NDVI_SR_Rrs"]
RRS_VARS = L3_RRS_VARS
BIO_OPT_VARS = ["Rrs_blue", "Rrs_green", "MBR", "X", "X2", "X3", "X4"]
CHL_VARS = ["log_Chl-a", "Chl-a", "Chl-a_re", "log_Chl-a_re"]
SEASONALITY_VARS = ["DOY_sin", "DOY_cos", "lat_amp", "season_sin", "season_cos"]

OUTPUT_COLUMNS = INFO_VARS + ["NASA_Chl-a"] + SR_VARS + SR_TO_RRS_VARS + INDEX_VARS + RRS_VARS + BIO_OPT_VARS + CHL_VARS + SEASONALITY_VARS


def audit_path(config):
    return Path(config['OUTPUT_FINAL_DIR']) / 'processing_report.jsonl'


def save_daily(frame, path, config, reason='processed'):
    # An explicit empty output marks a successfully processed date with no match.
    frame = frame.reindex(columns=OUTPUT_COLUMNS)
    atomic_parquet(frame, path)
    record_event(audit_path(config), reason, file=Path(path).name, rows=len(frame))

# Load masking tables for both 250m and 4.6km resolutions
def load_masking_tables(config):
    df_masking_250m = pd.read_parquet(config["MASK_250M_PATH"])
    df_masking_4_6km = pd.read_parquet(config["MASK_4_6KM_PATH"])
    return df_masking_250m, df_masking_4_6km

# Perform spatial matching between resolutions using cKDTree
def build_masking_total(df_masking_250m, df_masking_4_6km):
    matched_list = []
    common_lakes = sorted(set(df_masking_250m["Hylak_id"].unique()) & set(df_masking_4_6km["Hylak_id"].unique()))

    for hylak_id in common_lakes:
        sub_250m = df_masking_250m[df_masking_250m["Hylak_id"] == hylak_id].copy()
        sub_4_6km = df_masking_4_6km[df_masking_4_6km["Hylak_id"] == hylak_id].copy()

        if len(sub_250m) == 0 or len(sub_4_6km) == 0:
            continue

        tree_4_6km = cKDTree(sub_4_6km[["lon", "lat"]].values)
        _, idx_4_6km = tree_4_6km.query(sub_250m[["lon_GQ", "lat_GQ"]].values)

        sub_250m["lon_4_6km"] = sub_4_6km.iloc[idx_4_6km]["lon"].values
        sub_250m["lat_4_6km"] = sub_4_6km.iloc[idx_4_6km]["lat"].values
        matched_list.append(sub_250m)

    if not matched_list:
        return pd.DataFrame()

    df_masking_total = pd.concat(matched_list, ignore_index=True)
    float_cols = [
        "sinu_x_GQ", "sinu_y_GQ",
        "sinu_x_GA", "sinu_y_GA",
        "sinu_x_QA", "sinu_y_QA",
        "lat_GQ", "lon_GQ",
        "lat_4_6km", "lon_4_6km",
    ]
    existing_float_cols = [col for col in float_cols if col in df_masking_total.columns]
    df_masking_total[existing_float_cols] = df_masking_total[existing_float_cols].astype("float32")
    df_masking_total["Hylak_id"] = df_masking_total["Hylak_id"].astype("int32")
    return df_masking_total

# Manage loading of existing total mask or building a new one
def load_or_build_masking_total(config):
    if os.path.exists(config["MASK_TOTAL_PATH"]):
        return pd.read_parquet(config["MASK_TOTAL_PATH"])

    df_masking_250m, df_masking_4_6km = load_masking_tables(config)
    df_masking_total = build_masking_total(df_masking_250m, df_masking_4_6km)

    os.makedirs(os.path.dirname(config["MASK_TOTAL_PATH"]), exist_ok=True)
    df_masking_total.to_parquet(config["MASK_TOTAL_PATH"], index=False)
    return df_masking_total

# Create dictionary of yearly Terra data file paths
def load_terra_yearly_files(input_dir):
    yearly_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    yearly_dict = {}
    for path in yearly_files:
        year = os.path.basename(path).split("_")[-1].split(".")[0]
        yearly_dict[year] = path
    return yearly_dict

# Count the maximum available 250m pixels for each 4.6km L3 pixel
def build_l3_pixel_count_reference(df_masking_total):
    return (
        df_masking_total
        .groupby(["Hylak_id", "lon_4_6km", "lat_4_6km"])
        .size()
        .reset_index(name="max_count")
    )

# Calculate solar and seasonal variables based on date and latitude
def seasonality(df):
    df = df.copy()
    date = pd.to_datetime(df["date"])
    days_in_year = date.dt.is_leap_year.map({True: 366.0, False: 365.0})
    doy = date.dt.dayofyear.astype(np.float32)

    is_southern = df["lat"] <= 0
    doy_shifted = doy.copy()
    doy_shifted[is_southern] = doy_shifted[is_southern] + days_in_year[is_southern] / 2
    doy_shifted = doy_shifted % days_in_year

    df["DOY_sin"] = np.sin(2 * np.pi * doy_shifted / days_in_year)
    df["DOY_cos"] = np.cos(2 * np.pi * doy_shifted / days_in_year)
    df["lat_amp"] = np.sin(np.deg2rad(np.abs(df["lat"].values))).astype(np.float32)
    df["season_sin"] = (df["lat_amp"] * df["DOY_sin"]).astype(np.float32)
    df["season_cos"] = (df["lat_amp"] * df["DOY_cos"]).astype(np.float32)
    return df

# Process daily 250m data into aggregated 4.6km dataset with bio-optical variables
def build_daily_4_6km_dataset(config, df_masking_total):
    input_250m_dir = config["INPUT_250M_DIR"]
    input_terra_dir = config["INPUT_TERRA_4KM_DIR"]
    output_4km_dir = config["OUTPUT_MOD09_4KM_DIR"]

    os.makedirs(output_4km_dir, exist_ok=True)
    count_reference = build_l3_pixel_count_reference(df_masking_total)
    terra_yearly_dict = load_terra_yearly_files(input_terra_dir)
    years = sorted([year for year in os.listdir(input_250m_dir) if os.path.isdir(os.path.join(input_250m_dir, year))])

    for year in years:
        if year not in terra_yearly_dict:
            record_event(audit_path(config), 'missing_terra_year', year=year)
            raise FileNotFoundError(f'Missing Terra L3 data for {year}; see {audit_path(config)}')

        terra_4km = pd.read_parquet(terra_yearly_dict[year])
        l3_value_vars = ["NASA_Chl-a", "Chl-a", "Chl-a_re", "log_Chl-a_re"]
        terra_columns = ["date", "lon", "lat", "Hylak_id"] + [col for col in l3_value_vars if col in terra_4km.columns] + RRS_VARS
        terra_4km = terra_4km[terra_columns]

        months = sorted([month for month in os.listdir(os.path.join(input_250m_dir, year)) if os.path.isdir(os.path.join(input_250m_dir, year, month))])

        for month in months:
            folder = os.path.join(input_250m_dir, year, month)
            files = sorted(p.name for p in Path(folder).glob('A*.parquet'))

            for file in files:
                date_value = datetime.strptime(file.split(".")[0][1:], "%Y%j")
                terra_4km_daily = terra_4km[terra_4km["date"] == date_value]

                file_path_250m = os.path.join(folder, file)
                file_path_4km = os.path.join(output_4km_dir, file)
                if os.path.exists(file_path_4km):
                    record_event(audit_path(config), 'existing_daily_output', file=file)
                    continue

                try:
                    df = pd.read_parquet(file_path_250m).reset_index(drop=True)
                except Exception as exc:
                    record_event(audit_path(config), 'read_error', file=file_path_250m, error=str(exc))
                    raise RuntimeError(f'Cannot read {file_path_250m}') from exc

                df = df.merge(
                    df_masking_total.rename(columns={"lat_GQ": "lat", "lon_GQ": "lon"}),
                    on=[
                        "Hylak_id",
                        "sinu_x_GQ", "sinu_y_GQ",
                        "sinu_x_GA", "sinu_y_GA",
                        "sinu_x_QA", "sinu_y_QA",
                        "lat", "lon",
                    ],
                    how="inner",
                )

                if df.empty:
                    save_daily(pd.DataFrame(), file_path_4km, config, 'no_spatial_match')
                    continue

                grouped = df.groupby(["date", "Hylak_id", "lon_4_6km", "lat_4_6km"])
                df_out = grouped.mean(numeric_only=True).reset_index()
                df_out["count"] = grouped.size().values

                drop_cols = [col for col in ["lon", "lat"] if col in df_out.columns]
                df_out = df_out.drop(columns=drop_cols).rename(columns={"lon_4_6km": "lon", "lat_4_6km": "lat"})
                df_out = df_out.merge(
                    count_reference.rename(columns={"lon_4_6km": "lon", "lat_4_6km": "lat"}),
                    on=["Hylak_id", "lon", "lat"],
                    how="left",
                    validate="many_to_one",
                )
                if df_out["max_count"].isna().any():
                    raise ValueError(f"Missing 250m pixel count reference for {file}")
                df_out["count_ratio"] = df_out["count"] / df_out["max_count"]
                df_out = df_out.merge(terra_4km_daily, on=["date", "Hylak_id", "lon", "lat"], how="inner")
                if df_out.empty:
                    save_daily(pd.DataFrame(), file_path_4km, config, 'no_same_day_l3_match')
                    continue

                if "NASA_Chl-a" not in df_out.columns:
                    df_out["NASA_Chl-a"] = df_out["Chl-a"]
                df_out = add_rrs_bio_optical_features(df_out)
                df_out = seasonality(df_out)

                save_daily(df_out[OUTPUT_COLUMNS], file_path_4km, config)

                del df, grouped, df_out
                gc.collect()

# Count the number of processed files available for each year
def count_daily_files_by_year(input_250m_dir):
    counting_250m = {}
    for year in sorted(os.listdir(input_250m_dir)):
        year_path = os.path.join(input_250m_dir, year)
        if not os.path.isdir(year_path):
            continue

        count = 0
        for month in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month)
            if not os.path.isdir(month_path):
                continue
            count += len(list(Path(month_path).glob('A*.parquet')))

        counting_250m[year] = count
    return counting_250m

# Compile yearly parquet files from daily data filtered by 250m pixel coverage
def build_yearly_4_6km_resolution_dataset(config):
    mod09_4km_dir = config["OUTPUT_MOD09_4KM_DIR"]
    final_dir = config["OUTPUT_FINAL_DIR"]
    min_count_ratio = config["MIN_COUNT_RATIO"]
    counting_250m = count_daily_files_by_year(config["INPUT_250M_DIR"])

    os.makedirs(final_dir, exist_ok=True)
    all_files_in_mod09_4km = sorted(os.listdir(mod09_4km_dir))

    for year in sorted(counting_250m.keys()):
        files = [name for name in all_files_in_mod09_4km if name.startswith(f"A{year}") and name.endswith('.parquet')]
        expected = [p.name for p in (Path(config['INPUT_250M_DIR']) / year).rglob('A*.parquet')]
        if len(expected) != len(set(expected)):
            raise ValueError(f'Duplicate input dates in {year}')
        require_matching_files(expected, files, audit_path(config), year=year)

        year_path = os.path.join(final_dir, f"4.6km_resolution_dataset_{year}.parquet")
        if os.path.exists(year_path):
            continue

        dfs = []
        for file in files:
            df = pd.read_parquet(os.path.join(mod09_4km_dir, file))
            df_filtered = df[df["count_ratio"] >= min_count_ratio]
            if not df_filtered.empty:
                dfs.append(df_filtered)

        df_year = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=OUTPUT_COLUMNS)
        atomic_parquet(df_year, year_path)
        record_event(audit_path(config), 'year_completed', year=year, rows=len(df_year), daily_files=len(files))
        del df_year

        del dfs
        gc.collect()

# Merge all yearly threshold files into a single final master dataset
def build_final_4_6km_resolution_dataset(config):
    final_dir = config["OUTPUT_FINAL_DIR"]
    final_path = os.path.join(final_dir, "4.6km_resolution_dataset.parquet")

    dfs = []
    for file in sorted(os.listdir(final_dir)):
        path = os.path.join(final_dir, file)
        if os.path.isdir(path):
            continue
        if os.path.basename(path) == "4.6km_resolution_dataset.parquet":
            continue
        if not file.startswith("4.6km_resolution_dataset_") or not file.endswith('.parquet'):
            continue

        df = pd.read_parquet(path)
        dfs.append(df)

    if not dfs:
        raise ValueError('No yearly L3 tables; master dataset was not created')
    final_db = pd.concat(dfs, ignore_index=True)
    if final_db.empty:
        raise ValueError('All yearly L3 tables are empty; inspect processing_report.jsonl')
    atomic_parquet(final_db, final_path)
    record_event(audit_path(config), 'master_completed', rows=len(final_db), years=len(dfs))
    del final_db

    del dfs
    gc.collect()

def main():
    configure_paths(CONFIG)
    df_masking_total = load_or_build_masking_total(CONFIG)
    build_daily_4_6km_dataset(CONFIG, df_masking_total)
    build_yearly_4_6km_resolution_dataset(CONFIG)
    build_final_4_6km_resolution_dataset(CONFIG)


if __name__ == "__main__":
    main()
