
import os
import gc
import glob
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# Configuration settings for file paths and thresholds
CONFIG = {
    "MASK_250M_PATH": "data/inventory/df_masking_250m.parquet",
    "MASK_4_6KM_PATH": "data/inventory/df_masking_4_6km.parquet",
    "MASK_TOTAL_PATH": "data/inventory/df_masking_total.parquet",
    "INPUT_250M_DIR": "data/output/250m_daily",
    "INPUT_TERRA_4KM_DIR": "data/intermediate/DB_pixels_TERRA_4km",
    "OUTPUT_MOD09_4KM_DIR": "data/output/MOD09_4km_daily",
    "OUTPUT_FINAL_DIR": "data/output/4.6km_resolution_dataset",
    "THRESHOLD": 250,
}

# Define variable groups for data processing and output
INFO_VARS = ["date", "lon", "lat", "Hylak_id", "count"]
SR_VARS = ["SR_645", "SR_859", "SR_469", "SR_555", "SR_1240", "SR_1640", "SR_2130"]
SR_TO_RRS_VARS = ["SR_645_Rrs", "SR_859_Rrs", "SR_469_Rrs", "SR_555_Rrs", "SR_1240_Rrs", "SR_1640_Rrs", "SR_2130_Rrs"]
INDEX_VARS = ["Ratio_blue", "Ratio_blue_SR_Rrs", "FAI", "FAI_SR_Rrs", "NDVI", "NDVI_SR_Rrs"]
RRS_VARS = ["Rrs_443", "Rrs_469", "Rrs_488", "Rrs_547", "Rrs_555", "Rrs_645"]
BIO_OPT_VARS = ["Rrs_blue", "ratio", "R", "R2", "R3", "R4"]
CHL_VARS = ["log_Chl-a", "Chl-a"]
SEASONALITY_VARS = ["DOY_sin", "DOY_cos", "lat_amp", "season_sin", "season_cos"]

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
    terra_yearly_dict = load_terra_yearly_files(input_terra_dir)
    years = sorted([year for year in os.listdir(input_250m_dir) if os.path.isdir(os.path.join(input_250m_dir, year))])

    for year in years:
        if year not in terra_yearly_dict:
            continue

        terra_4km = pd.read_parquet(terra_yearly_dict[year])
        terra_4km = terra_4km[["date", "lon", "lat", "Hylak_id", "Rrs_443", "Rrs_469", "Rrs_488", "Rrs_547", "Rrs_555", "Rrs_645", "Chl-a"]]

        months = sorted([month for month in os.listdir(os.path.join(input_250m_dir, year)) if os.path.isdir(os.path.join(input_250m_dir, year, month))])

        for month in months:
            folder = os.path.join(input_250m_dir, year, month)
            files = sorted(os.listdir(folder))

            for file in files:
                date_value = datetime.strptime(file.split(".")[0][1:], "%Y%j")
                terra_4km_daily = terra_4km[terra_4km["date"] == date_value]

                file_path_250m = os.path.join(folder, file)
                file_path_4km = os.path.join(output_4km_dir, file)
                if os.path.exists(file_path_4km):
                    continue

                try:
                    df = pd.read_parquet(file_path_250m).reset_index(drop=True)
                except Exception:
                    continue

                df = df.merge(
                    df_masking_total,
                    on=[
                        "Hylak_id",
                        "sinu_x_GQ", "sinu_y_GQ",
                        "sinu_x_GA", "sinu_y_GA",
                        "sinu_x_QA", "sinu_y_QA",
                        "lat_GQ", "lon_GQ",
                    ],
                    how="inner",
                )

                if df.empty:
                    continue

                grouped = df.groupby(["date", "Hylak_id", "lon_4_6km", "lat_4_6km"])
                df_out = grouped.mean(numeric_only=True).reset_index()
                df_out["count"] = grouped.size().values

                drop_cols = [col for col in ["lon", "lat"] if col in df_out.columns]
                df_out = df_out.drop(columns=drop_cols).rename(columns={"lon_4_6km": "lon", "lat_4_6km": "lat"})
                df_out = df_out.merge(terra_4km_daily, on=["date", "Hylak_id", "lon", "lat"], how="inner")
                if df_out.empty:
                    continue

                df_out["Rrs_blue"] = df_out[["Rrs_443", "Rrs_488"]].max(axis=1)
                df_out["ratio"] = df_out["Rrs_blue"] / df_out["Rrs_547"]
                df_out["R"] = np.log10(df_out["ratio"])
                df_out["R2"] = df_out["R"] ** 2
                df_out["R3"] = df_out["R"] ** 3
                df_out["R4"] = df_out["R"] ** 4
                df_out["log_Chl-a"] = np.log(1 + df_out["Chl-a"])
                df_out = seasonality(df_out)

                df_out = df_out[INFO_VARS + SR_VARS + SR_TO_RRS_VARS + INDEX_VARS + RRS_VARS + BIO_OPT_VARS + CHL_VARS + SEASONALITY_VARS]
                df_out.to_parquet(file_path_4km, compression="snappy")

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
            count += len([f for f in os.listdir(month_path) if os.path.isfile(os.path.join(month_path, f))])

        counting_250m[year] = count
    return counting_250m

# Compile yearly parquet files from daily data filtered by pixel count threshold
def build_yearly_4_6km_resolution_dataset(config):
    mod09_4km_dir = config["OUTPUT_MOD09_4KM_DIR"]
    final_dir = config["OUTPUT_FINAL_DIR"]
    threshold = config["THRESHOLD"]
    counting_250m = count_daily_files_by_year(config["INPUT_250M_DIR"])

    os.makedirs(final_dir, exist_ok=True)
    all_files_in_mod09_4km = sorted(os.listdir(mod09_4km_dir))

    for year in sorted(counting_250m.keys()):
        files = [name for name in all_files_in_mod09_4km if name.startswith(f"A{year}")]
        if len(files) != counting_250m[year]:
            continue

        year_path = os.path.join(final_dir, f"4.6km_resolution_dataset_{year}.parquet")
        if os.path.exists(year_path):
            continue

        dfs = []
        for file in files:
            df = pd.read_parquet(os.path.join(mod09_4km_dir, file))
            df_filtered = df[df["count"] > threshold]
            if not df_filtered.empty:
                dfs.append(df_filtered)

        if dfs:
            df_year = pd.concat(dfs).reset_index(drop=True)
            df_year.to_parquet(year_path, compression="snappy")
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
        if not file.startswith("4.6km_resolution_"):
            continue

        df = pd.read_parquet(path)
        dfs.append(df)

    if dfs:
        final_db = pd.concat(dfs).reset_index(drop=True)
        final_db.to_parquet(final_path, compression="snappy")
        del final_db

    del dfs
    gc.collect()

# Execution sequence
df_masking_total = load_or_build_masking_total(CONFIG)
build_daily_4_6km_dataset(CONFIG, df_masking_total)
build_yearly_4_6km_resolution_dataset(CONFIG)
build_final_4_6km_resolution_dataset(CONFIG)
