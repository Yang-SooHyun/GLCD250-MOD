
import os
import glob
import gc
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from joblib import Parallel, delayed

# Configuration for file paths, directories, and parallel processing settings
CONFIG = {
    "GQ_MASK_PATH": "data/inventory/df_masking_GQ.parquet",
    "GA_MASK_PATH": "data/inventory/df_masking_GA.parquet",
    "QA_MASK_PATH": "data/inventory/df_masking_QA.parquet",
    "OUTPUT_MASK_250m_PATH": "data/inventory/df_masking_250m.parquet",
    "RAW_GQ_DIR": "data/intermediate/GQ_processed_dataframe",
    "RAW_GA_DIR": "data/intermediate/GA_processed_dataframe",
    "RAW_QA_DIR": "data/intermediate/QA_processed_dataframe",
    "PROC_GQ_DIR": "data/intermediate/Processed/GQ",
    "PROC_GA_DIR": "data/intermediate/Processed/GA",
    "PROC_QA_DIR": "data/intermediate/Processed/QA",
    "FINAL_DIR": "data/output/250m_resolution_dataset_daily",
    "APPLY_OUTPUT_DIR": "data/output/250m_resolution_dataset_lake",
    "N_JOBS": 10,
}

# Calculate Modified Normalized Difference Water Index (MNDWI)
def mndwi(df):
    df = df.copy()
    df["MNDWI"] = (df["Refl4"] - df["Refl6"]) / (df["Refl4"] + df["Refl6"] + 1e-10)
    return df

# Calculate Normalized Difference Snow Index (NDSI)
def ndsi(df):
    df = df.copy()
    df["NDSI"] = (df["Refl4"] - df["Refl6"]) / (df["Refl4"] + df["Refl6"] + 1e-10)
    return df

# Check for good quality pixels in 500m bands using QC flags
def is_good_band3456(df, col="QC_GA"):
    qc = df[col].astype("int32").values
    b3 = (qc >> 10) & 0b1111
    b4 = (qc >> 14) & 0b1111
    b5 = (qc >> 18) & 0b1111
    b6 = (qc >> 22) & 0b1111
    good = (b3 == 0) & (b4 == 0) & (b5 == 0) & (b6 == 0)
    return df.loc[good].copy()

# Convert surface reflectance to remote sensing reflectance (SR_Rrs)
def sr_to_rrs(df):
    df = df.copy()
    refl_cols = ["Refl1", "Refl2", "Refl3", "Refl4", "Refl5", "Refl6", "Refl7"]
    df = df[(df[refl_cols] > 0).all(axis=1)]

    min_values = df[["Refl2", "Refl5", "Refl6", "Refl7"]].min(axis=1)
    for col in refl_cols:
        df[f"{col}_Rrs"] = (df[col] - min_values) / np.pi

    rrs_cols = [f"{c}_Rrs" for c in refl_cols]
    df = df[(df[rrs_cols] >= 0).all(axis=1)]
    return df

# Compute spectral indices like FAI, NDVI, and blue ratios
def add_band_features(df):
    df = df.copy()
    df["FAI"] = df["Refl2"] - (df["Refl1"] + (df["Refl5"] - df["Refl1"]) * (859 - 645) / (1240 - 645))
    df["FAI_SR_Rrs"] = df["Refl2_Rrs"] - (df["Refl1_Rrs"] + (df["Refl5_Rrs"] - df["Refl1_Rrs"]) * (859 - 645) / (1240 - 645))
    df["Ratio_blue"] = df["Refl3"] / df["Refl4"]
    df["Ratio_blue_SR_Rrs"] = df["Refl3_Rrs"] / df["Refl4_Rrs"]
    df["NDVI"] = (df["Refl2"] - df["Refl1"]) / (df["Refl2"] + df["Refl1"] + 1e-10)
    df["NDVI_SR_Rrs"] = (df["Refl2_Rrs"] - df["Refl1_Rrs"]) / (df["Refl2_Rrs"] + df["Refl1_Rrs"] + 1e-10)
    return df

# Calculate solar and seasonal variables based on date and latitude
def seasonality(df):
    df = df.copy()
    date = df["date"]
    days_in_year = date.dt.is_leap_year.map({True: 366.0, False: 365.0})
    doy = date.dt.dayofyear.astype(np.float32)

    is_southern = df["lat_GQ"] <= 0
    doy_shifted = doy.copy()
    doy_shifted[is_southern] = doy_shifted[is_southern] + days_in_year[is_southern] / 2
    doy_shifted = doy_shifted % days_in_year

    df["DOY_sin"] = np.sin(2 * np.pi * doy_shifted / days_in_year)
    df["DOY_cos"] = np.cos(2 * np.pi * doy_shifted / days_in_year)
    df["lat_amp"] = np.sin(np.deg2rad(np.abs(df["lat_GQ"].values))).astype(np.float32)
    df["season_sin"] = (df["lat_amp"] * df["DOY_sin"]).astype(np.float32)
    df["season_cos"] = (df["lat_amp"] * df["DOY_cos"]).astype(np.float32)
    return df

# Decode 16-bit QA flags for atmospheric and cloud conditions
def decode_qa(value):
    state = format(int(value), "016b")
    return {
        "cloud_state": int(state[-2:], 2),
        "cloud_shadow": int(state[-3]),
        "land_water": int(state[-6:-3], 2),
        "aerosol_quantity": int(state[-8:-6], 2),
        "cirrus": int(state[-10:-8], 2),
        "internal_cloud": int(state[-11]),
        "internal_fire": int(state[-12]),
        "mod35_snow_ice": int(state[-13]),
        "adjacent_cloud": int(state[-14]),
        "salt_pan": int(state[-15]),
        "internal_snow": int(state[-16]),
    }

# Filter valid pixels based on decoded cloud and surface quality
def is_valid_qa(qa_dict):
    return int(
        qa_dict["cloud_state"] == 0
        and qa_dict["cloud_shadow"] == 0
        and qa_dict["aerosol_quantity"] != 3
        and qa_dict["cirrus"] != 3
        and qa_dict["internal_cloud"] == 0
        and qa_dict["internal_fire"] == 0
        and qa_dict["mod35_snow_ice"] == 0
        and qa_dict["salt_pan"] == 0
        and qa_dict["internal_snow"] == 0
    )

# Extract unique date key from the file name
def date_key_from_filename(filepath):
    return os.path.basename(filepath).split("_")[0]

# Convert MODIS day-of-year key to calendar year and month
def doy_to_year_month(date_key):
    year = date_key[1:5]
    doy = int(date_key[5:8])
    date_obj = datetime(int(year), 1, 1) + timedelta(days=doy - 1)
    return year, f"{date_obj.month:02d}"

# Match and merge coordinates across 250m, 500m, and 1km masking tables
def create_250m_masking_table(config):
    df_gq = pd.read_parquet(config["GQ_MASK_PATH"])
    df_ga = pd.read_parquet(config["GA_MASK_PATH"])
    df_qa = pd.read_parquet(config["QA_MASK_PATH"])

    matched_data = []
    for hylak_id in df_gq["Hylak_id"].unique():
        sub_gq = df_gq[df_gq["Hylak_id"] == hylak_id].copy()
        sub_ga = df_ga[df_ga["Hylak_id"] == hylak_id]
        sub_qa = df_qa[df_qa["Hylak_id"] == hylak_id]

        if len(sub_ga) > 0:
            tree_ga = cKDTree(sub_ga[["sinu_x_GA", "sinu_y_GA"]].values)
            _, idx_ga = tree_ga.query(sub_gq[["sinu_x_GQ", "sinu_y_GQ"]].values)
            sub_gq["sinu_x_GA"] = sub_ga.iloc[idx_ga]["sinu_x_GA"].values
            sub_gq["sinu_y_GA"] = sub_ga.iloc[idx_ga]["sinu_y_GA"].values
        else:
            sub_gq["sinu_x_GA"] = np.nan
            sub_gq["sinu_y_GA"] = np.nan

        if len(sub_qa) > 0:
            tree_qa = cKDTree(sub_qa[["sinu_x_QA", "sinu_y_QA"]].values)
            _, idx_qa = tree_qa.query(sub_gq[["sinu_x_GQ", "sinu_y_GQ"]].values)
            sub_gq["sinu_x_QA"] = sub_qa.iloc[idx_qa]["sinu_x_QA"].values
            sub_gq["sinu_y_QA"] = sub_qa.iloc[idx_qa]["sinu_y_QA"].values
        else:
            sub_gq["sinu_x_QA"] = np.nan
            sub_gq["sinu_y_QA"] = np.nan

        matched_data.append(sub_gq)

    df_masking_250m = pd.concat(matched_data, ignore_index=True)
    float_cols = [
        "sinu_x_GQ", "sinu_y_GQ",
        "sinu_x_GA", "sinu_y_GA",
        "sinu_x_QA", "sinu_y_QA",
        "lat_GQ", "lon_GQ",
    ]
    df_masking_250m[float_cols] = df_masking_250m[float_cols].astype("float32")
    df_masking_250m["Hylak_id"] = df_masking_250m["Hylak_id"].astype("int32")
    df_masking_250m.to_parquet(config["OUTPUT_MASK_250m_PATH"], index=False)
    return df_masking_250m

# Preprocess 250m (GQ) files by filtering quality and reflectance
def preprocess_gq(path, config):
    try:
        df = pd.read_parquet(path).dropna()
        df = df[df["QC_GQ"] == 4096.0]
        df = df[(df["Refl1"] >= 0) & (df["Refl2"] >= 0)]

        out_dir = os.path.join(config["PROC_GQ_DIR"], *path.replace("\\", "/").split("/")[-3:-1])
        os.makedirs(out_dir, exist_ok=True)
        out_name = "_".join(os.path.basename(path).split(".")[1:4]) + ".parquet"
        df.drop(columns=["QC_GQ"]).to_parquet(os.path.join(out_dir, out_name), compression="snappy")
        return True
    except Exception:
        return False

# Preprocess 500m (GA) files by filtering water pixels and band quality
def preprocess_ga(path, config):
    try:
        df = pd.read_parquet(
            path,
            columns=["date", "sinu_x_GA", "sinu_y_GA", "Hylak_id", "Refl3", "Refl4", "Refl5", "Refl6", "Refl7", "QC_GA"],
        ).dropna()
        df = mndwi(df)
        df = is_good_band3456(df)
        df = df[df["MNDWI"] > 0]

        out_dir = os.path.join(config["PROC_GA_DIR"], *path.replace("\\", "/").split("/")[-3:-1])
        os.makedirs(out_dir, exist_ok=True)
        out_name = "_".join(os.path.basename(path).split(".")[1:4]) + ".parquet"
        df.drop(columns=["MNDWI", "QC_GA"]).to_parquet(os.path.join(out_dir, out_name), compression="snappy")
        return True
    except Exception:
        return False

# Preprocess 1km (QA) state files by validating atmospheric flags
def preprocess_qa(path, config):
    try:
        df = pd.read_parquet(path).dropna()
        df["valid_qa"] = df["QA"].apply(lambda x: is_valid_qa(decode_qa(x)))
        df = df[df["valid_qa"] == 1].drop(columns=["valid_qa"])

        out_dir = os.path.join(config["PROC_QA_DIR"], *path.replace("\\", "/").split("/")[-3:-1])
        os.makedirs(out_dir, exist_ok=True)
        out_name = "_".join(os.path.basename(path).split(".")[1:4]) + ".parquet"
        df.to_parquet(os.path.join(out_dir, out_name), compression="snappy")
        return True
    except Exception:
        return False

# Aggregate and merge preprocessed products for a single day
def process_day_final(date_key, gq_dict, ga_dict, qa_dict, df_masking, config):
    try:
        dfs = {"GQ": [], "GA": [], "QA": []}
        for src, d in zip(["GQ", "GA", "QA"], [gq_dict, ga_dict, qa_dict]):
            for p in d[date_key]:
                if os.path.getsize(p) >= 100:
                    dfs[src].append(pd.read_parquet(p))
            if not dfs[src]:
                return None
            dfs[src] = pd.concat(dfs[src], ignore_index=True)

        df_base = dfs["GQ"].merge(df_masking, on=["sinu_x_GQ", "sinu_y_GQ", "Hylak_id"], how="inner")
        df_total = df_base.merge(dfs["GA"], on=["date", "sinu_x_GA", "sinu_y_GA", "Hylak_id"], how="inner")
        df_total = df_total.merge(
            dfs["QA"][["date", "sinu_x_QA", "sinu_y_QA", "QA"]],
            on=["date", "sinu_x_QA", "sinu_y_QA"],
            how="inner",
        )

        df_total = ndsi(df_total)
        df_total = df_total[~((df_total["NDSI"] >= 0.4) & (df_total["Refl2"] >= 0.11) & (df_total["Refl4"] >= 0.10))]
        df_total = sr_to_rrs(df_total)
        df_total = add_band_features(df_total)
        df_total = seasonality(df_total)

        rename_map = {
            "lat_GQ": "lat",
            "lon_GQ": "lon",
            "Refl1": "SR_645",
            "Refl2": "SR_859",
            "Refl3": "SR_469",
            "Refl4": "SR_555",
            "Refl5": "SR_1240",
            "Refl6": "SR_1640",
            "Refl7": "SR_2130",
            "Refl1_Rrs": "SR_645_Rrs",
            "Refl2_Rrs": "SR_859_Rrs",
            "Refl3_Rrs": "SR_469_Rrs",
            "Refl4_Rrs": "SR_555_Rrs",
            "Refl5_Rrs": "SR_1240_Rrs",
            "Refl6_Rrs": "SR_1640_Rrs",
            "Refl7_Rrs": "SR_2130_Rrs",
        }

        df_total = df_total.rename(columns=rename_map)

        keep_cols = [
            "date", "lat", "lon", "Hylak_id",
            "lat_GQ", "lon_GQ",
            "sinu_x_GQ", "sinu_y_GQ",
            "sinu_x_GA", "sinu_y_GA",
            "sinu_x_QA", "sinu_y_QA",
            "QA",
            "SR_645", "SR_859", "SR_469", "SR_555", "SR_1240", "SR_1640", "SR_2130",
            "SR_645_Rrs", "SR_859_Rrs", "SR_469_Rrs", "SR_555_Rrs", "SR_1240_Rrs", "SR_1640_Rrs", "SR_2130_Rrs",
            "FAI", "FAI_SR_Rrs", "Ratio_blue", "Ratio_blue_SR_Rrs", "NDVI", "NDVI_SR_Rrs",
            "DOY_sin", "DOY_cos", "lat_amp", "season_sin", "season_cos",
        ]

        df_final = df_total[keep_cols].drop_duplicates(
            subset=["date", "lat", "lon", "Hylak_id", "sinu_x_GQ", "sinu_y_GQ", "sinu_x_GA", "sinu_y_GA", "sinu_x_QA", "sinu_y_QA"]
        )

        year, month = doy_to_year_month(date_key)
        out_path = os.path.join(config["FINAL_DIR"], year, month, f"{date_key}.parquet")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_final.to_parquet(out_path, compression="snappy")
        return date_key
    except Exception:
        return None

# Organize integrated daily files into individual lake-based parquet files
def build_250m_resolution_dataset_by_lake(config):
    input_dir = config["FINAL_DIR"]
    output_dir = config["APPLY_OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    daily_files = sorted(glob.glob(os.path.join(input_dir, "**", "*.parquet"), recursive=True))
    lake_data = {}

    for path in daily_files:
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue

        if "Hylak_id" not in df.columns:
            continue

        for hylak_id, group in df.groupby("Hylak_id"):
            hylak_id = int(hylak_id)
            lake_data.setdefault(hylak_id, []).append(group)

        del df
        gc.collect()

    for hylak_id, dfs in lake_data.items():
        df_lake = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(output_dir, f"Hylak_id_{hylak_id:07}.parquet")
        df_lake.to_parquet(out_path, compression="snappy")
        del df_lake
        gc.collect()

# Execution sequence
if not os.path.exists(CONFIG["OUTPUT_MASK_250m_PATH"]):
    df_masking_250m = create_250m_masking_table(CONFIG)
else:
    df_masking_250m = pd.read_parquet(CONFIG["OUTPUT_MASK_250m_PATH"])

gq_raw = glob.glob(os.path.join(CONFIG["RAW_GQ_DIR"], "**/*.parquet"), recursive=True)
ga_raw = glob.glob(os.path.join(CONFIG["RAW_GA_DIR"], "**/*.parquet"), recursive=True)
qa_raw = glob.glob(os.path.join(CONFIG["RAW_QA_DIR"], "**/*.parquet"), recursive=True)

Parallel(n_jobs=CONFIG["N_JOBS"])(delayed(preprocess_gq)(p, CONFIG) for p in gq_raw)
Parallel(n_jobs=CONFIG["N_JOBS"])(delayed(preprocess_ga)(p, CONFIG) for p in ga_raw)
Parallel(n_jobs=CONFIG["N_JOBS"])(delayed(preprocess_qa)(p, CONFIG) for p in qa_raw)

proc_gq = glob.glob(os.path.join(CONFIG["PROC_GQ_DIR"], "**/*.parquet"), recursive=True)
proc_ga = glob.glob(os.path.join(CONFIG["PROC_GA_DIR"], "**/*.parquet"), recursive=True)
proc_qa = glob.glob(os.path.join(CONFIG["PROC_QA_DIR"], "**/*.parquet"), recursive=True)

gq_dict, ga_dict, qa_dict = defaultdict(list), defaultdict(list), defaultdict(list)
for p in proc_gq:
    gq_dict[date_key_from_filename(p)].append(p)
for p in proc_ga:
    ga_dict[date_key_from_filename(p)].append(p)
for p in proc_qa:
    qa_dict[date_key_from_filename(p)].append(p)

common_dates = sorted(set(gq_dict) & set(ga_dict) & set(qa_dict))
Parallel(n_jobs=CONFIG["N_JOBS"])(
    delayed(process_day_final)(dk, gq_dict, ga_dict, qa_dict, df_masking_250m, CONFIG) for dk in common_dates
)

build_250m_resolution_dataset_by_lake(CONFIG)
