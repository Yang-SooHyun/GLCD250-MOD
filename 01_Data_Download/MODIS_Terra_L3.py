
import os
import gc
import subprocess
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import xarray as xr
import rioxarray
import geopandas as gpd

# Configuration for paths, product names, and download settings
CONFIG = {
    "token": "your_token",
    "base_save_dir": "data/raw/Terra_L3",
    "url_dir": "data/url_lists",
    "products": ["CHL", "Rrs_443", "Rrs_469", "Rrs_488", "Rrs_547", "Rrs_555", "Rrs_645"],
    "chl_folder": "data/TERRA",
    "rrs_base_dir": "data/Terra_Rrs",
    "shapefile_path": "data/globallakes.shp",
    "mask_output_path": "data/inventory/df_masking_4_6km.parquet",
    "chl_db_dir": "data/intermediate/DB_pixels_Chl_4km",
    "rrs_db_base_dir": "data/intermediate",
    "terra_db_dir": "data/intermediate/DB_pixels_TERRA_4km",
    "max_connection_per_server": "4",
    "split": "4",
    "max_tries": "3",
    "retry_wait": "2",
    "timeout": "30",
}

# Download Level-3 NetCDF files from NASA using aria2c
def download_l3_files(config):
    for product in config["products"]:
        url_list_path = os.path.join(config["url_dir"], f"MODIS_Terra_L3_{product}_Urls.txt")
        if not os.path.exists(url_list_path):
            continue

        with open(url_list_path, "r") as f:
            urls = [line.strip() for line in f if line.strip()]

        save_dir = os.path.join(config["base_save_dir"], product)
        os.makedirs(save_dir, exist_ok=True)

        for url in urls:
            filename = url.split("/")[-1]
            cmd = [
                "aria2c",
                "--header", f"Authorization: Bearer {config['token']}",
                "--dir", save_dir,
                "--out", filename,
                f"--max-connection-per-server={config['max_connection_per_server']}",
                f"--split={config['split']}",
                f"--max-tries={config['max_tries']}",
                f"--retry-wait={config['retry_wait']}",
                f"--timeout={config['timeout']}",
                "--continue=true",
                "--quiet=true",
                url,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Extract and mask pixels for a single lake polygon using rioxarray
def mask_chl_file(args):
    data, gdf_one = args
    try:
        clipped = data.rio.clip(gdf_one.geometry, gdf_one.crs, drop=True, all_touched=True)
        data_values = clipped.values
        lat_values = clipped.coords["lat"].values
        lon_values = clipped.coords["lon"].values

        clipped_df = pd.DataFrame(data_values, index=pd.Index(lat_values), columns=pd.Index(lon_values))
        mask = clipped_df.stack().reset_index().dropna().iloc[:, :2]
        mask.columns = ["lat", "lon"]
        mask["Hylak_id"] = gdf_one["Hylak_id"].iloc[0]
        return mask
    except rioxarray.exceptions.NoDataInBounds:
        return pd.DataFrame(columns=["lat", "lon", "Hylak_id"])

# Scan folder for files and extract dates from filenames
def load_chl_file_info(chl_folder):
    filenames = sorted(os.listdir(chl_folder))
    filepaths = [os.path.join(chl_folder, name) for name in filenames]
    dates = [pd.to_datetime(name.split(".")[1]).strftime("%Y%m%d") for name in filenames]
    return filenames, filepaths, dates

# Create a spatial template from a NetCDF file for masking
def build_chl_mask_template(chl_path):
    ds = xr.open_dataset(chl_path)
    data = ds["chlor_a"].rio.write_crs("EPSG:4326")
    data[:] = 1
    ds.close()
    return data

# Parallel process all lakes to create a master pixel-to-lake mask
def build_chl_mask(data, lake_gdf, output_path):
    if os.path.exists(output_path):
        return pd.read_parquet(output_path)

    args = [(data, lake_gdf.iloc[[idx]]) for idx in range(lake_gdf.shape[0])]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(mask_chl_file, args))

    mask_df = pd.concat(results).reset_index(drop=True)
    mask_df.columns = ["lat", "lon", "Hylak_id"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask_df.to_parquet(output_path, index=False)
    return mask_df

# Group dates into yearly lists for chunked processing
def build_yearly_chunks(dates):
    year_list = sorted(set(pd.to_datetime(dates).year))
    yearly_chunks = []
    for year in year_list:
        indices = [idx for idx, date in enumerate(dates) if pd.to_datetime(date).year == year]
        yearly_chunks.append(indices)
    return year_list, yearly_chunks

# Extract pixel values for a variable and save as yearly parquet files
def build_yearly_variable_db(var_name, folder_path, dates, year_list, yearly_chunks, mask, output_dir, file_tag=None):
    os.makedirs(output_dir, exist_ok=True)
    if file_tag is None:
        file_tag = var_name

    already_exist = [
        int(name.split(".")[0].split("_")[-1])
        for name in sorted(os.listdir(output_dir))
        if name.endswith(".parquet")
    ]

    filenames = sorted(os.listdir(folder_path))
    filepaths = [os.path.join(folder_path, name) for name in filenames]

    for chunk_idx, year in enumerate(year_list):
        if year in already_exist:
            continue

        df_yearly = []
        chunk = yearly_chunks[chunk_idx]

        for file_idx in chunk:
            path = filepaths[file_idx]
            date = dates[file_idx]

            ds = xr.open_dataset(path)
            data = ds[var_name].rio.write_crs("EPSG:4326")
            data_values = data.values
            lat_values = data.coords["lat"].values
            lon_values = data.coords["lon"].values

            data_df = pd.DataFrame(data_values, index=pd.Index(lat_values), columns=pd.Index(lon_values))
            data_df = data_df.stack().reset_index()
            data_df.columns = ["lat", "lon", date]

            result = mask.merge(data_df, how="left")
            if file_idx == chunk[0]:
                df_yearly.append(result)
            else:
                df_yearly.append(result.iloc[:, -1])

            ds.close()
            del data, data_df, result
            gc.collect()

        yearly_result = pd.concat(df_yearly, axis=1)
        yearly_result = yearly_result.sort_values("Hylak_id").reset_index(drop=True)
        yearly_result.to_parquet(os.path.join(output_dir, f"DB_pixels_{file_tag}_4km_{year}.parquet"), compression="snappy")
        del yearly_result, df_yearly
        gc.collect()

# Combine separate variable files into a final long-format daily database
def build_final_terra_db(year_list, rrs_list, config):
    os.makedirs(config["terra_db_dir"], exist_ok=True)

    id_vars = ["lat", "lon", "Hylak_id"]
    already_exist = [
        int(name.split(".")[0].split("_")[-1])
        for name in sorted(os.listdir(config["terra_db_dir"]))
        if name.endswith(".parquet")
    ]

    for year in year_list:
        if year in already_exist:
            continue

        dfs = {}
        dfs["Chl-a"] = pd.read_parquet(os.path.join(config["chl_db_dir"], f"DB_pixels_Chl_4km_{year}.parquet"))

        for rrs_name in rrs_list:
            dfs[rrs_name] = pd.read_parquet(
                os.path.join(config["rrs_db_base_dir"], f"DB_pixels_{rrs_name}_4km", f"DB_pixels_{rrs_name}_4km_{year}.parquet")
            )

        date_columns = [col for col in dfs["Chl-a"].columns if col not in id_vars]
        df_concat = []

        for date in date_columns:
            dfs_long = []
            for var_name, df in dfs.items():
                df_long = df.melt(id_vars=id_vars, value_vars=date, var_name="date", value_name=var_name)
                df_long = df_long.set_index(["date", "Hylak_id", "lat", "lon"])
                dfs_long.append(df_long)

            daily_df = pd.concat(dfs_long, axis=1).reset_index()
            daily_df = daily_df.dropna()
            daily_df["date"] = pd.to_datetime(daily_df["date"])
            df_concat.append(daily_df)

        result = pd.concat(df_concat)
        result = result.sort_values(["date", "Hylak_id", "lat", "lon"]).reset_index(drop=True)
        result.to_parquet(os.path.join(config["terra_db_dir"], f"DB_pixels_TERRA_4km_{year}.parquet"), compression="snappy")
        del dfs, df_concat, result
        gc.collect()

# Execution sequence
lake_gdf = gpd.read_file(CONFIG["shapefile_path"])
lake_gdf = lake_gdf[["Hylak_id", "geometry"]].copy()

_, filepaths_chl, dates_chl = load_chl_file_info(CONFIG["chl_folder"])
year_list, yearly_chunks = build_yearly_chunks(dates_chl)

chl_template = build_chl_mask_template(filepaths_chl[0])
chl_mask = build_chl_mask(chl_template, lake_gdf, CONFIG["mask_output_path"])

build_yearly_variable_db(
    var_name="chlor_a",
    folder_path=CONFIG["chl_folder"],
    dates=dates_chl,
    year_list=year_list,
    yearly_chunks=yearly_chunks,
    mask=chl_mask,
    output_dir=CONFIG["chl_db_dir"],
    file_tag="Chl",
)

rrs_list = ["Rrs_443", "Rrs_469", "Rrs_488", "Rrs_547", "Rrs_555", "Rrs_645"]
for rrs_name in rrs_list:
    build_yearly_variable_db(
        var_name=rrs_name,
        folder_path=os.path.join(CONFIG["rrs_base_dir"], rrs_name),
        dates=dates_chl,
        year_list=year_list,
        yearly_chunks=yearly_chunks,
        mask=chl_mask,
        output_dir=os.path.join(CONFIG["rrs_db_base_dir"], f"DB_pixels_{rrs_name}_4km"),
    )

build_final_terra_db(year_list, rrs_list, CONFIG)
