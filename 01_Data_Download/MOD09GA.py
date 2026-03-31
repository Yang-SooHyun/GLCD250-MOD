import os
import gc
import glob
import time
import pickle
import subprocess
import re
from threading import Lock
from datetime import datetime

import netCDF4 as nc
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from bs4 import BeautifulSoup
import requests
from joblib import Parallel, delayed

# Configuration for NASA Earthdata credentials, date ranges, and processing paths
CONFIG = {
    "token": "your_token",
    "start_date": "2000-02-24",
    "end_date": "2024-12-31",
    "base_url": "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD09GA/",
    "inventory_dir": "data/inventory",
    "download_dir": "data/raw/GA",
    "meshgrid_dir": "data/intermediate/hv_meshgrid",
    "masked_meshgrid_dir_GA": "data/intermediate/hv_meshgrid_masked_GA",
    "masked_meshgrid_dir_QA": "data/intermediate/hv_meshgrid_masked_QA",
    "processed_name_dir_GA": "data/intermediate/GA_processed_file_name",
    "processed_name_dir_QA": "data/intermediate/QA_processed_file_name",
    "processed_df_dir_GA": "data/intermediate/GA_processed_dataframe",
    "processed_df_dir_QA": "data/intermediate/QA_processed_dataframe",
    "shapefile_path": "data/globallakes.shp",
    "n_jobs": 4,
    "max_connection_per_server": "4",
    "split": "4",
    "max_tries": "3",
    "retry_wait": "2",
    "timeout": "30",
}

# Scrape NASA LADSWEB archive to collect all available HDF file URLs for the date range
def collect_ga_urls(output_dir, start_date, end_date, base_url):
    os.makedirs(output_dir, exist_ok=True)
    target_date_range = [d.strftime("%Y/%j") for d in pd.date_range(start_date, end_date)]
    scraped_urls = set()

    for date_path in target_date_range:
        directory_url = base_url + date_path
        try:
            response = requests.get(directory_url, timeout=20)
            response.raise_for_status()
        except Exception:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        hdf_files = [a["href"] for a in soup.find_all("a", href=True) if a["href"].endswith(".hdf")]
        scraped_urls.update(f"{directory_url}/{fname}" for fname in hdf_files)

    output_path = os.path.join(output_dir, "MOD09GA_urls.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(list(scraped_urls), f)
    return output_path

# Identify dates where all required global tiles (hv) are present
def find_complete_dates(inventory_dir):
    pkl_path = os.path.join(inventory_dir, "MOD09GA_urls.pkl")
    with open(pkl_path, "rb") as f:
        download_urls = pickle.load(f)

    keys = [".".join(url.split("/")[-1].split(".")[1:3]) for url in download_urls]
    df_metadata = pd.DataFrame(
        {
            "Date": [k.split(".")[0] for k in keys],
            "hv": [k.split(".")[1] for k in keys],
        }
    )

    tile_counts = df_metadata.groupby("Date")["hv"].count().reset_index()
    total_tiles = df_metadata["hv"].nunique()

    complete_dates = sorted(set(tile_counts[tile_counts["hv"] == total_tiles]["Date"]))
    hv_list = sorted(df_metadata["hv"].unique())

    with open(os.path.join(inventory_dir, "download_urls_GA.pkl"), "wb") as f:
        pickle.dump(download_urls, f)

    with open(os.path.join(inventory_dir, "hv_list_GA.pkl"), "wb") as f:
        pickle.dump(hv_list, f)

    return download_urls, hv_list, complete_dates

# Download all tiles for a specific example date using aria2c
def download_example_date(token, download_urls, example_date, base_save_dir, product_name, settings):
    filtered_urls = [url for url in download_urls if example_date in url]
    save_dir = os.path.join(base_save_dir, product_name)
    os.makedirs(save_dir, exist_ok=True)

    for url in filtered_urls:
        filename = url.split("/")[-1]
        cmd = [
            "aria2c",
            "--header", f"Authorization: Bearer {token}",
            "--dir", save_dir,
            "--out", filename,
            f"--max-connection-per-server={settings['max_connection_per_server']}",
            f"--split={settings['split']}",
            f"--max-tries={settings['max_tries']}",
            f"--retry-wait={settings['retry_wait']}",
            f"--timeout={settings['timeout']}",
            "--continue=true",
            "--quiet=true",
            url,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Create 500m (GA) and 1km (QA) Sinusoidal coordinate grids for each tile
def build_ga_meshgrid(base_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    hdf_paths = glob.glob(os.path.join(base_dir, "*GA*/*.hdf"), recursive=True)

    for path in hdf_paths:
        with nc.Dataset(path, "r") as ds:
            metadata_text = ds.getncattr("StructMetadata.0")

        ul_match = re.search(r"UpperLeftPointMtrs=\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)", metadata_text)
        lr_match = re.search(r"LowerRightMtrs=\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)", metadata_text)
        if not (ul_match and lr_match):
            continue

        ul_x, ul_y = map(float, ul_match.groups())
        lr_x, lr_y = map(float, lr_match.groups())
        hv = os.path.basename(path).split(".")[2]

        x_ga = np.linspace(ul_x, lr_x, 2400)
        y_ga = np.linspace(ul_y, lr_y, 2400)
        xv_ga, yv_ga = np.meshgrid(x_ga, y_ga)

        x_qa = np.linspace(ul_x, lr_x, 1200)
        y_qa = np.linspace(ul_y, lr_y, 1200)
        xv_qa, yv_qa = np.meshgrid(x_qa, y_qa)

        np.savez(os.path.join(output_dir, f"{hv}_GA.npz"), xv=xv_ga, yv=yv_ga)
        np.savez(os.path.join(output_dir, f"{hv}_QA.npz"), xv=xv_qa, yv=yv_qa)

# Intersect grid points with lake polygons to identify pixels of interest
def process_masking(shapefile_path, base_dir, meshgrid_dir, data_type):
    gdf = gpd.read_file(shapefile_path)
    boundary_gdf = gdf.sort_values(by="Lake_area", ascending=False)[["geometry", "Hylak_id"]].copy()

    sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    wgs84 = pyproj.Proj("EPSG:4326")
    transformer = pyproj.Transformer.from_proj(sinu, wgs84, always_xy=True)

    dfs_masking = []
    hdf_paths = glob.glob(os.path.join(base_dir, "*GA*/*.hdf"), recursive=True)

    for path in hdf_paths:
        hv = os.path.basename(path).split(".")[2]
        mesh_path = os.path.join(meshgrid_dir, f"{hv}_{data_type}.npz")
        if not os.path.exists(mesh_path):
            continue

        data = np.load(mesh_path)
        xv = data["xv"]
        yv = data["yv"]
        lon, lat = transformer.transform(xv, yv)

        point_gdf = gpd.GeoDataFrame(
            {
                f"lat_{data_type}": lat.ravel(),
                f"lon_{data_type}": lon.ravel(),
                f"sinu_x_{data_type}": xv.ravel(),
                f"sinu_y_{data_type}": yv.ravel(),
            },
            geometry=gpd.points_from_xy(lon.ravel(), lat.ravel()),
            crs="EPSG:4326",
        )

        point_gdf = point_gdf[(point_gdf[f"lat_{data_type}"] >= -56) & (point_gdf[f"lat_{data_type}"] <= 84)]
        clipped_gdf = gpd.sjoin(
            point_gdf,
            boundary_gdf,
            how="inner",
            predicate="intersects",
        ).drop(columns=["index_right", "geometry"])
        dfs_masking.append(clipped_gdf)

    if not dfs_masking:
        return pd.DataFrame(columns=[f"lat_{data_type}", f"lon_{data_type}", f"sinu_x_{data_type}", f"sinu_y_{data_type}", "Hylak_id"])

    df_masking = pd.concat(dfs_masking, ignore_index=True)
    df_masking = df_masking.drop_duplicates(subset=[f"sinu_x_{data_type}", f"sinu_y_{data_type}", "Hylak_id"], keep="first")
    return df_masking

# Generate binary mask arrays and lake ID maps for each global tile
def create_hv_masked(df_masking, meshgrid_dir, output_dir, data_type, n_jobs=4):
    os.makedirs(output_dir, exist_ok=True)
    coords = df_masking[[f"sinu_x_{data_type}", f"sinu_y_{data_type}"]].drop_duplicates().to_numpy()
    files = [fname for fname in os.listdir(meshgrid_dir) if fname.endswith(f"_{data_type}.npz") and "_masked" not in fname]
    hv_list = sorted([fname.split("_")[0] for fname in files])

    def process_one_hv(hv_name):
        data = np.load(os.path.join(meshgrid_dir, f"{hv_name}_{data_type}.npz"))
        xv = data["xv"].ravel()
        yv = data["yv"].ravel()
        xy = np.column_stack((xv, yv))

        xmin, xmax = xv.min(), xv.max()
        ymin, ymax = yv.min(), yv.max()

        coords_cut = coords[
            (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax) &
            (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)
        ]

        if coords_cut.size == 0:
            mask = np.zeros(xy.shape[0], dtype=bool)
        else:
            xy_c = np.ascontiguousarray(xy)
            coords_c = np.ascontiguousarray(coords_cut)
            xy_view = xy_c.view([("x", xy_c.dtype), ("y", xy_c.dtype)])
            coords_view = coords_c.view([("x", coords_c.dtype), ("y", coords_c.dtype)])
            mask = np.isin(xy_view, coords_view).reshape(-1)

        df_hv = df_masking[
            (df_masking[f"sinu_x_{data_type}"] >= xmin) & (df_masking[f"sinu_x_{data_type}"] <= xmax) &
            (df_masking[f"sinu_y_{data_type}"] >= ymin) & (df_masking[f"sinu_y_{data_type}"] <= ymax)
        ]

        coord_to_id = {
            (getattr(row, f"sinu_x_{data_type}"), getattr(row, f"sinu_y_{data_type}")): row.Hylak_id
            for row in df_hv.itertuples()
        }

        hylak_id = np.array([coord_to_id.get((x, y), -1) for x, y in zip(xv, yv)], dtype=np.int32)
        save_path = os.path.join(output_dir, f"{hv_name}_{data_type}_masked.npz")
        np.savez(save_path, xv=xv.astype("float32"), yv=yv.astype("float32"), hylak_id=hylak_id, mask=mask.astype(bool))

    Parallel(n_jobs=n_jobs, backend="threading")(delayed(process_one_hv)(hv) for hv in hv_list)

# Extract 500m reflectance bands (3-7) from HDF and save masked pixels to Parquet
def file_process_ga(url, raw_nc_file_dir, hv_name, xv, yv, hylak_id, mask, processed_df_dir):
    filename = url.split("/")[-1]
    filepath = os.path.join(raw_nc_file_dir, filename)
    max_wait_time = 180

    start_time = time.time()
    while not os.path.exists(filepath):
        if time.time() - start_time > max_wait_time:
            return False
        time.sleep(1)

    start_time = time.time()
    while os.path.exists(filepath + ".crdownload") or os.path.exists(filepath + ".aria2"):
        if time.time() - start_time > max_wait_time:
            return False
        time.sleep(1)

    ds = None
    for _ in range(3):
        try:
            ds = nc.Dataset(filepath, "r")
            break
        except Exception:
            time.sleep(2)

    if ds is None:
        return False

    try:
        date_str = filename.split(".")[1][1:]
        date = datetime.strptime(date_str, "%Y%j")
        year = date.year
        month = date.month

        output_dir = os.path.join(processed_df_dir, str(year), f"{month:02d}")
        os.makedirs(output_dir, exist_ok=True)

        refl3_cpu = ds.variables["sur_refl_b03_1"][:].astype("float32")
        refl4_cpu = ds.variables["sur_refl_b04_1"][:].astype("float32")
        refl5_cpu = ds.variables["sur_refl_b05_1"][:].astype("float32")
        refl6_cpu = ds.variables["sur_refl_b06_1"][:].astype("float32")
        refl7_cpu = ds.variables["sur_refl_b07_1"][:].astype("float32")
        qc_cpu = ds.variables["QC_500m_1"][:]

        if refl3_cpu.size != mask.size:
            ds.close()
            return False

        for arr in [refl3_cpu, refl4_cpu, refl5_cpu, refl6_cpu, refl7_cpu]:
            arr[arr <= 0] = np.nan

        refl3_flat = refl3_cpu.flatten().astype("<f4")
        refl4_flat = refl4_cpu.flatten().astype("<f4")
        refl5_flat = refl5_cpu.flatten().astype("<f4")
        refl6_flat = refl6_cpu.flatten().astype("<f4")
        refl7_flat = refl7_cpu.flatten().astype("<f4")
        qc_flat = qc_cpu.flatten().astype("<u2")

        valid_count = int(mask.sum())
        results = {
            "date": [date] * valid_count,
            "hv_tile": [hv_name] * valid_count,
            "sinu_x_GA": xv[mask],
            "sinu_y_GA": yv[mask],
            "Hylak_id": hylak_id[mask],
            "Refl3": refl3_flat[mask],
            "Refl4": refl4_flat[mask],
            "Refl5": refl5_flat[mask],
            "Refl6": refl6_flat[mask],
            "Refl7": refl7_flat[mask],
            "QC_GA": qc_flat[mask],
        }

        if valid_count > 0:
            df = pd.DataFrame(results)
            base_filename = filename.replace(".hdf", "")
            output_filename = os.path.join(output_dir, f"{base_filename}.parquet")
            df.to_parquet(output_filename, engine="pyarrow", compression="snappy")

        ds.close()
        del refl3_flat, refl4_flat, refl5_flat, refl6_flat, refl7_flat, qc_flat
        gc.collect()
        return True
    except Exception:
        try:
            if ds is not None:
                ds.close()
        except Exception:
            pass
        return False

# Extract 1km State QA flags from HDF and save masked pixels to Parquet
def file_process_qa(url, raw_nc_file_dir, xv, yv, mask, processed_df_dir, lock):
    filename = url.split("/")[-1]
    filepath = os.path.join(raw_nc_file_dir, filename)
    max_wait_time = 180
    start_time = time.time()
    aria2_path = filepath + ".aria2"

    while True:
        if time.time() - start_time > max_wait_time:
            return False
        if os.path.exists(filepath) and not os.path.exists(aria2_path):
            time.sleep(0.5)
            break
        time.sleep(2)

    qa_cpu = None
    for _ in range(3):
        try:
            with lock:
                with nc.Dataset(filepath, "r") as ds:
                    qa_cpu = np.array(ds.variables["state_1km_1"][:], dtype=np.int16)
            break
        except Exception:
            time.sleep(2)

    if qa_cpu is None:
        return False

    try:
        date_str = filename.split(".")[1][1:]
        date = datetime.strptime(date_str, "%Y%j")
        year = date.year
        month = date.month

        output_dir = os.path.join(processed_df_dir, str(year), f"{month:02d}")
        os.makedirs(output_dir, exist_ok=True)

        qa_flat = qa_cpu.flatten().astype("<u2")
        results = {
            "date": [date] * int(mask.sum()),
            "sinu_x_QA": xv[mask],
            "sinu_y_QA": yv[mask],
            "QA": qa_flat[mask],
        }

        if len(results["sinu_x_QA"]) > 0:
            df = pd.DataFrame(results)
            base_filename = filename.replace(".hdf", "")
            output_filename = os.path.join(output_dir, f"{base_filename}.parquet")
            df.to_parquet(output_filename, engine="pyarrow", compression="snappy")

        del qa_flat
        gc.collect()
        return True
    except Exception:
        return False

# Orchestrate the per-tile download and processing loop for the entire inventory
def download_and_process_by_hv(token, hv_list, download_urls_ga, ga_masked_dir, qa_masked_dir, raw_dir, ga_processed_name_dir, qa_processed_name_dir, ga_processed_df_dir, qa_processed_df_dir, settings, lock):
    completed_hv = []
    for hv in hv_list:
        processed_file_name_dir = os.path.join(ga_processed_name_dir, hv)
        if os.path.exists(processed_file_name_dir):
            completed_hv.append(hv)

    hv_filtered = [hv for hv in hv_list if hv not in completed_hv[:-1]]

    for hv in hv_list:
        if hv not in hv_filtered:
            continue

        processed_file_name_dir_ga = os.path.join(ga_processed_name_dir, hv)
        processed_file_name_dir_qa = os.path.join(qa_processed_name_dir, hv)
        os.makedirs(processed_file_name_dir_ga, exist_ok=True)
        os.makedirs(processed_file_name_dir_qa, exist_ok=True)

        raw_nc_file_dir = os.path.join(raw_dir, hv)
        os.makedirs(raw_nc_file_dir, exist_ok=True)
        save_dir = os.path.abspath(raw_nc_file_dir)

        ga_coord_file = os.path.join(ga_masked_dir, f"{hv}_GA_masked.npz")
        qa_coord_file = os.path.join(qa_masked_dir, f"{hv}_QA_masked.npz")
        if not os.path.exists(ga_coord_file) or not os.path.exists(qa_coord_file):
            continue

        try:
            ga_coord_data = np.load(ga_coord_file)
            qa_coord_data = np.load(qa_coord_file)
        except Exception:
            continue

        xv_ga = ga_coord_data["xv"].flatten().astype("float32")
        yv_ga = ga_coord_data["yv"].flatten().astype("float32")
        hylak_id_ga = ga_coord_data["hylak_id"].flatten().astype("int32")
        mask_ga = ga_coord_data["mask"].flatten()

        xv_qa = qa_coord_data["xv"].flatten().astype("float32")
        yv_qa = qa_coord_data["yv"].flatten().astype("float32")
        mask_qa = qa_coord_data["mask"].flatten()

        urls = download_urls_ga[hv]
        url_filenames = [os.path.basename(url) for url in urls]
        processed_file_names = os.listdir(processed_file_name_dir_ga)
        remaining_filenames = list(set(url_filenames) - set(processed_file_names))
        remaining_urls = [url for url in urls if os.path.basename(url) in remaining_filenames]

        for url in remaining_urls:
            filename = url.split("/")[-1]
            save_path = os.path.join(save_dir, filename)

            cmd = [
                "aria2c",
                "--header", f"Authorization: Bearer {token}",
                "--dir", save_dir,
                "--out", filename,
                f"--max-connection-per-server={settings['max_connection_per_server']}",
                f"--split={settings['split']}",
                f"--max-tries={settings['max_tries']}",
                f"--retry-wait={settings['retry_wait']}",
                f"--timeout={settings['timeout']}",
                "--continue=true",
                "--quiet=true",
                url,
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            download_started = False
            for _ in range(20):
                time.sleep(0.5)
                if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                    download_started = True
                    break

            if not download_started:
                continue

            success_ga = file_process_ga(
                url=url,
                raw_nc_file_dir=raw_nc_file_dir,
                hv_name=hv,
                xv=xv_ga,
                yv=yv_ga,
                hylak_id=hylak_id_ga,
                mask=mask_ga,
                processed_df_dir=ga_processed_df_dir,
            )
            if not success_ga:
                continue

            success_qa = file_process_qa(
                url=url,
                raw_nc_file_dir=raw_nc_file_dir,
                xv=xv_qa,
                yv=yv_qa,
                mask=mask_qa,
                processed_df_dir=qa_processed_df_dir,
                lock=lock,
            )
            if not success_qa:
                continue

            if os.path.exists(save_path):
                os.remove(save_path)

            with open(os.path.join(processed_file_name_dir_ga, filename), "w") as f:
                f.write("")
            with open(os.path.join(processed_file_name_dir_qa, filename), "w") as f:
                f.write("")

# Create a mapping of MODIS tile names (hv) to their corresponding file URLs
def build_hv_url_dict(download_urls):
    hv_dict = {}
    for url in download_urls:
        hv = url.split("/")[-1].split(".")[2]
        hv_dict.setdefault(hv, []).append(url)
    return hv_dict

# Execution sequence
inventory_dir = CONFIG["inventory_dir"]
os.makedirs(inventory_dir, exist_ok=True)

mod09ga_urls_path = os.path.join(inventory_dir, "MOD09GA_urls.pkl")
if not os.path.exists(mod09ga_urls_path):
    collect_ga_urls(
        output_dir=inventory_dir,
        start_date=CONFIG["start_date"],
        end_date=CONFIG["end_date"],
        base_url=CONFIG["base_url"],
    )

download_urls, hv_list, complete_dates = find_complete_dates(inventory_dir)
if not complete_dates:
    raise ValueError("No complete dates found.")

example_date = complete_dates[0]
download_example_date(
    token=CONFIG["token"],
    download_urls=download_urls,
    example_date=example_date,
    base_save_dir=CONFIG["download_dir"],
    product_name="GA",
    settings=CONFIG,
)

build_ga_meshgrid(
    base_dir=CONFIG["download_dir"],
    output_dir=CONFIG["meshgrid_dir"],
)

df_masking_ga = process_masking(
    shapefile_path=CONFIG["shapefile_path"],
    base_dir=CONFIG["download_dir"],
    meshgrid_dir=CONFIG["meshgrid_dir"],
    data_type="GA",
)
df_masking_ga.to_parquet(os.path.join(inventory_dir, "df_masking_GA.parquet"), index=False)

df_masking_qa = process_masking(
    shapefile_path=CONFIG["shapefile_path"],
    base_dir=CONFIG["download_dir"],
    meshgrid_dir=CONFIG["meshgrid_dir"],
    data_type="QA",
)
df_masking_qa.to_parquet(os.path.join(inventory_dir, "df_masking_QA.parquet"), index=False)

create_hv_masked(
    df_masking=df_masking_ga,
    meshgrid_dir=CONFIG["meshgrid_dir"],
    output_dir=CONFIG["masked_meshgrid_dir_GA"],
    data_type="GA",
    n_jobs=CONFIG["n_jobs"],
)

create_hv_masked(
    df_masking=df_masking_qa,
    meshgrid_dir=CONFIG["meshgrid_dir"],
    output_dir=CONFIG["masked_meshgrid_dir_QA"],
    data_type="QA",
    n_jobs=CONFIG["n_jobs"],
)

download_urls_ga = build_hv_url_dict(download_urls)
lock = Lock()

download_and_process_by_hv(
    token=CONFIG["token"],
    hv_list=hv_list,
    download_urls_ga=download_urls_ga,
    ga_masked_dir=CONFIG["masked_meshgrid_dir_GA"],
    qa_masked_dir=CONFIG["masked_meshgrid_dir_QA"],
    raw_dir=CONFIG["download_dir"],
    ga_processed_name_dir=CONFIG["processed_name_dir_GA"],
    qa_processed_name_dir=CONFIG["processed_name_dir_QA"],
    ga_processed_df_dir=CONFIG["processed_df_dir_GA"],
    qa_processed_df_dir=CONFIG["processed_df_dir_QA"],
    settings=CONFIG,
    lock=lock,
)
