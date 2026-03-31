import os
import gc
import glob
import time
import pickle
import subprocess
import re
from datetime import datetime

import netCDF4 as nc
import numpy as np
import pandas as pd
import geopandas as gpd
import pyproj
from bs4 import BeautifulSoup
import requests
from joblib import Parallel, delayed

# Configuration for NASA Earthdata token, dates, and directory paths
CONFIG = {
    "token": "your_token",
    "start_date": "2000-02-24",
    "end_date": "2024-12-31",
    "base_url": "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MOD09GQ/",
    "inventory_dir": "data/inventory",
    "download_dir": "data/raw/GQ",
    "processed_name_dir": "data/intermediate/GQ_processed_file_name",
    "processed_df_dir": "data/intermediate/GQ_processed_dataframe",
    "meshgrid_dir": "data/intermediate/hv_meshgrid_GQ",
    "masked_meshgrid_dir": "data/intermediate/hv_meshgrid_masked_GQ",
    "shapefile_path": "data/globallakes.shp",
    "n_jobs": 4,
    "max_connection_per_server": "4",
    "split": "4",
    "max_tries": "3",
    "retry_wait": "2",
    "timeout": "30",
}

# Scrape the NASA LADSWEB archive to collect HDF file URLs for the specified date range
def collect_gq_urls(output_dir, start_date, end_date, base_url):
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

    output_path = os.path.join(output_dir, "MOD09GQ_urls.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(list(scraped_urls), f)
    return output_path

# Filter dates where all required MODIS tiles (hv) are available in the archive
def find_complete_dates(inventory_dir):
    pkl_path = os.path.join(inventory_dir, "MOD09GQ_urls.pkl")
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

    with open(os.path.join(inventory_dir, "download_urls_GQ.pkl"), "wb") as f:
        pickle.dump(download_urls, f)

    with open(os.path.join(inventory_dir, "hv_list_GQ.pkl"), "wb") as f:
        pickle.dump(hv_list, f)

    return download_urls, hv_list, complete_dates

# Download all tiles for a specific date using aria2c with multi-connection settings
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

# Generate coordinate grids (Sinusoidal meters) from HDF metadata for each tile
def build_gq_meshgrid(base_dir, output_dir, resolution=4800):
    os.makedirs(output_dir, exist_ok=True)
    hdf_paths = glob.glob(os.path.join(base_dir, "*GQ*/*.hdf"), recursive=True)

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

        x = np.linspace(ul_x, lr_x, resolution)
        y = np.linspace(ul_y, lr_y, resolution)
        xv, yv = np.meshgrid(x, y)

        save_path = os.path.join(output_dir, f"{hv}_GQ.npz")
        np.savez(save_path, xv=xv, yv=yv)

# Intersect coordinate grids with a shapefile to identify pixels located within lakes
def process_masking_gq(shapefile_path, base_dir, meshgrid_dir):
    gdf = gpd.read_file(shapefile_path)
    boundary_gdf = gdf.sort_values(by="Lake_area", ascending=False)[["geometry", "Hylak_id"]].copy()

    sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
    wgs84 = pyproj.Proj("EPSG:4326")
    transformer = pyproj.Transformer.from_proj(sinu, wgs84, always_xy=True)

    dfs_masking = []
    hdf_paths = glob.glob(os.path.join(base_dir, "*GQ*/*.hdf"), recursive=True)

    for path in hdf_paths:
        hv = os.path.basename(path).split(".")[2]
        mesh_path = os.path.join(meshgrid_dir, f"{hv}_GQ.npz")
        if not os.path.exists(mesh_path):
            continue

        data = np.load(mesh_path)
        xv = data["xv"]
        yv = data["yv"]
        lon, lat = transformer.transform(xv, yv)

        point_gdf = gpd.GeoDataFrame(
            {
                "lat_GQ": lat.ravel(),
                "lon_GQ": lon.ravel(),
                "sinu_x_GQ": xv.ravel(),
                "sinu_y_GQ": yv.ravel(),
            },
            geometry=gpd.points_from_xy(lon.ravel(), lat.ravel()),
            crs="EPSG:4326",
        )

        point_gdf = point_gdf[(point_gdf["lat_GQ"] >= -56) & (point_gdf["lat_GQ"] <= 84)]

        clipped_gdf = gpd.sjoin(
            point_gdf,
            boundary_gdf,
            how="inner",
            predicate="intersects",
        ).drop(columns=["index_right", "geometry"])

        dfs_masking.append(clipped_gdf)

    if not dfs_masking:
        return pd.DataFrame(columns=["lat_GQ", "lon_GQ", "sinu_x_GQ", "sinu_y_GQ", "Hylak_id"])

    df_masking = pd.concat(dfs_masking, ignore_index=True)
    df_masking = df_masking.drop_duplicates(subset=["sinu_x_GQ", "sinu_y_GQ", "Hylak_id"], keep="first")
    return df_masking

# Create binary mask files and lake-ID arrays for each tile in parallel
def create_hv_masked(df_masking, meshgrid_dir, output_dir, n_jobs=4):
    os.makedirs(output_dir, exist_ok=True)
    coords = df_masking[["sinu_x_GQ", "sinu_y_GQ"]].drop_duplicates().to_numpy()
    files = [fname for fname in os.listdir(meshgrid_dir) if fname.endswith(".npz") and "_masked" not in fname]
    hv_list = sorted([fname.split("_")[0] for fname in files])

    def process_one_hv(hv_name):
        data = np.load(os.path.join(meshgrid_dir, f"{hv_name}_GQ.npz"))
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
            (df_masking["sinu_x_GQ"] >= xmin) & (df_masking["sinu_x_GQ"] <= xmax) &
            (df_masking["sinu_y_GQ"] >= ymin) & (df_masking["sinu_y_GQ"] <= ymax)
        ]
        coord_to_id = {(row.sinu_x_GQ, row.sinu_y_GQ): row.Hylak_id for row in df_hv.itertuples()}
        hylak_id = np.array([coord_to_id.get((x, y), -1) for x, y in zip(xv, yv)], dtype=np.int32)

        save_path = os.path.join(output_dir, f"{hv_name}_GQ_masked.npz")
        np.savez(
            save_path,
            xv=xv.astype("float32"),
            yv=yv.astype("float32"),
            hylak_id=hylak_id,
            mask=mask.astype(bool),
        )

    Parallel(n_jobs=n_jobs, backend="threading")(delayed(process_one_hv)(hv) for hv in hv_list)

# Extract band 1, 2, and QC data from HDF and save masked lake pixels as Parquet
def file_process(url, raw_nc_file_dir, hv_name, xv, yv, hylak_id, mask, processed_hv_name_dir, processed_df_dir):
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

        refl1_cpu = ds.variables["sur_refl_b01_1"][:].astype("float32")
        refl2_cpu = ds.variables["sur_refl_b02_1"][:].astype("float32")
        qc_cpu = ds.variables["QC_250m_1"][:]

        if refl1_cpu.size != mask.size:
            ds.close()
            return False

        refl1_cpu[refl1_cpu <= 0] = np.nan
        refl2_cpu[refl2_cpu <= 0] = np.nan

        refl1_flat = refl1_cpu.flatten().astype("<f4")
        refl2_flat = refl2_cpu.flatten().astype("<f4")
        qc_flat = qc_cpu.flatten().astype("<u2")

        valid_count = int(mask.sum())
        results = {
            "date": [date] * valid_count,
            "hv_tile": [hv_name] * valid_count,
            "sinu_x_GQ": xv[mask],
            "sinu_y_GQ": yv[mask],
            "Hylak_id": hylak_id[mask],
            "Refl1": refl1_flat[mask],
            "Refl2": refl2_flat[mask],
            "QC_GQ": qc_flat[mask],
        }

        if valid_count > 0:
            df = pd.DataFrame(results)
            base_filename = filename.replace(".hdf", "")
            output_filename = os.path.join(output_dir, f"{base_filename}.parquet")
            df.to_parquet(output_filename, engine="pyarrow", compression="snappy")
            del df, results

        ds.close()
        if os.path.exists(filepath):
            os.remove(filepath)

        with open(os.path.join(processed_hv_name_dir, filename), "w") as f:
            f.write("")

        del refl1_flat, refl2_flat, qc_flat
        gc.collect()
        return True
    except Exception:
        try:
            if ds is not None:
                ds.close()
        except Exception:
            pass
        return False

# Manage the sequential download and extraction of HDF tiles for each global region
def download_and_process_by_hv(token, hv_list, download_urls_gq, masked_dir, raw_dir, processed_name_dir, processed_df_dir, settings):
    completed_hv = []
    for hv in hv_list:
        processed_file_name_dir = os.path.join(processed_name_dir, hv)
        if os.path.exists(processed_file_name_dir):
            completed_hv.append(hv)

    hv_filtered = [hv for hv in hv_list if hv not in completed_hv[:-1]]

    for hv in hv_list:
        if hv not in hv_filtered:
            continue

        processed_file_name_dir = os.path.join(processed_name_dir, hv)
        os.makedirs(processed_file_name_dir, exist_ok=True)

        raw_nc_file_dir = os.path.join(raw_dir, hv)
        os.makedirs(raw_nc_file_dir, exist_ok=True)
        save_dir = os.path.abspath(raw_nc_file_dir)

        coord_file = os.path.join(masked_dir, f"{hv}_GQ_masked.npz")
        if not os.path.exists(coord_file):
            continue

        try:
            coord_data = np.load(coord_file)
        except Exception:
            continue

        xv = coord_data["xv"].flatten().astype("float32")
        yv = coord_data["yv"].flatten().astype("float32")
        hylak_id = coord_data["hylak_id"].flatten().astype("int32")
        mask = coord_data["mask"].flatten()

        urls = download_urls_gq[hv]
        url_filenames = [os.path.basename(url) for url in urls]
        processed_file_names = os.listdir(processed_file_name_dir)
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

            file_process(
                url,
                raw_nc_file_dir,
                hv,
                xv,
                yv,
                hylak_id,
                mask,
                processed_file_name_dir,
                processed_df_dir,
            )

# Create a dictionary mapping MODIS tile names (hv) to their archive URLs
def build_hv_url_dict(download_urls):
    hv_dict = {}
    for url in download_urls:
        hv = url.split("/")[-1].split(".")[2]
        hv_dict.setdefault(hv, []).append(url)
    return hv_dict

# Execution sequence
inventory_dir = CONFIG["inventory_dir"]
os.makedirs(inventory_dir, exist_ok=True)

mod09gq_urls_path = os.path.join(inventory_dir, "MOD09GQ_urls.pkl")
if not os.path.exists(mod09gq_urls_path):
    collect_gq_urls(
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
    product_name="GQ",
    settings=CONFIG,
)

build_gq_meshgrid(
    base_dir=CONFIG["download_dir"],
    output_dir=CONFIG["meshgrid_dir"],
)

df_masking_gq = process_masking_gq(
    shapefile_path=CONFIG["shapefile_path"],
    base_dir=CONFIG["download_dir"],
    meshgrid_dir=CONFIG["meshgrid_dir"],
)
df_masking_path = os.path.join(inventory_dir, "df_masking_GQ.parquet")
df_masking_gq.to_parquet(df_masking_path, index=False)

create_hv_masked(
    df_masking=df_masking_gq,
    meshgrid_dir=CONFIG["meshgrid_dir"],
    output_dir=CONFIG["masked_meshgrid_dir"],
    n_jobs=CONFIG["n_jobs"],
)

download_urls_gq = build_hv_url_dict(download_urls)
download_and_process_by_hv(
    token=CONFIG["token"],
    hv_list=hv_list,
    download_urls_gq=download_urls_gq,
    masked_dir=CONFIG["masked_meshgrid_dir"],
    raw_dir=CONFIG["download_dir"],
    processed_name_dir=CONFIG["processed_name_dir"],
    processed_df_dir=CONFIG["processed_df_dir"],
    settings=CONFIG,
)
