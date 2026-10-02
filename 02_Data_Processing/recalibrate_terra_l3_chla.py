"""Match in-situ data, run the manuscript's 960-candidate recalibration, and apply it."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workflow_utils import configure_paths, atomic_parquet, L3_RRS_VARS

import argparse
import gc
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score


CONFIG = {
    "terra_db_dir": "data/intermediate/DB_pixels_TERRA_4km",
    "recalibrated_db_dir": "data/intermediate/DB_pixels_TERRA_4km_recalibrated",
    "matched_path": "data/intermediate/L3_in_situ_matched.parquet",
    "summary_path": "data/intermediate/L3_Chl-a_recalibration_summary.csv",
    "selected_config_path": "data/intermediate/L3_Chl-a_recalibration_selected.json",
    "insitu_path": "data/L3_in-situ_matched.parquet",
}

TARGET_COL = "insitu_Chl-a"
NASA_CHLA_COL = "Chl-a"
MATCHED_NASA_CHLA_COL = "L3_Chl-a"
RECAL_CHLA_COL = "Chl-a_re"
TARGET_MAX = 200.0
EPS = 1e-12
INVALID_OBJECTIVE = 1e12
MIN_VALIDATION_OBSERVATIONS = 20
STABILITY_MBR_RANGE = (0.01, 30.0)
SEARCH_METHOD = "shared_mbr_rma_r2_quantile_lolov_v1"

OCX_CONFIGS = {
    "OC3M": [0.26294, -2.64669, 1.28364, 1.08209, -1.76828],
    "OC4M": [0.27015, -2.47936, 1.53752, -0.13967, -0.66166],
    "OC5M": [0.42919, -4.88411, 9.57678, -9.24289, 2.51916],
    "OC6M": [1.22914, -4.99423, 5.64706, -3.53426, 0.69266],
}

MAX_NUMERATOR_GROUPS = [
    ("max_Rrs_443_Rrs_488", ["Rrs_443", "Rrs_488"]),
    ("max_Rrs_412_Rrs_443_Rrs_488", ["Rrs_412", "Rrs_443", "Rrs_488"]),
    ("max_Rrs_412_Rrs_443_Rrs_488_Rrs_531", ["Rrs_412", "Rrs_443", "Rrs_488", "Rrs_531"]),
]

DENOMINATOR_GROUPS = [
    ("Rrs_547", ["Rrs_547"], "single"),
    ("Rrs_555", ["Rrs_555"], "single"),
    ("mean_Rrs_547_Rrs_667", ["Rrs_547", "Rrs_667"], "mean"),
    ("mean_Rrs_555_Rrs_667", ["Rrs_555", "Rrs_667"], "mean"),
]

INITIAL_THRESHOLD_PAIRS = [(1.0, 3.0), (1.0, 5.0), (3.0, 5.0), (3.0, 10.0), (5.0, 10.0)]


def _rrs_extract(expr):
    parts = str(expr).split("_")
    return [f"Rrs_{parts[i + 1]}" for i, part in enumerate(parts[:-1]) if part == "Rrs"]


def component_series(df, expr):
    if expr in df.columns:
        return pd.to_numeric(df[expr], errors="coerce")
    cols = _rrs_extract(expr)
    if not cols:
        raise ValueError(f"Cannot parse Rrs component: {expr}")
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing Rrs columns: {missing}")
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    if expr.startswith("max_"):
        return values.max(axis=1, skipna=False)
    if expr.startswith("mean_"):
        return values.mean(axis=1, skipna=False)
    return values[cols[0]]


def ratio_series(df, ratio_col):
    if "/" not in ratio_col:
        raise ValueError(f"Invalid ratio name: {ratio_col}")
    numerator, denominator = ratio_col.split("/", 1)
    den = component_series(df, denominator)
    return component_series(df, numerator) / den.where(den != 0, np.nan)


def build_ratio_columns(df):
    df = df.copy()
    ratio_cols = []
    for numerator_name, numerator_cols in MAX_NUMERATOR_GROUPS:
        df[numerator_name] = df[numerator_cols].apply(pd.to_numeric, errors="coerce").max(axis=1, skipna=False)
    for denominator_name, denominator_cols, mode in DENOMINATOR_GROUPS:
        values = df[denominator_cols].apply(pd.to_numeric, errors="coerce")
        df[denominator_name] = values.mean(axis=1, skipna=False) if mode == "mean" else values.iloc[:, 0]
    for numerator_name, _ in MAX_NUMERATOR_GROUPS:
        for denominator_name, _, _ in DENOMINATOR_GROUPS:
            ratio_col = f"{numerator_name}/{denominator_name}"
            df[ratio_col] = ratio_series(df, ratio_col)
            ratio_cols.append(ratio_col)
    return df, ratio_cols


def ocx_chla(mbr, coefs, clip=True):
    mbr = np.asarray(mbr, dtype=float)
    x = np.log10(np.maximum(mbr, EPS))
    log_chla = sum(float(coefs[i]) * x**i for i in range(5))
    # Application retains the released model's numerical clipping; fitting and
    # stability checks evaluate unbounded curves so clipping cannot hide overflow.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return 10 ** (np.clip(log_chla, -5.0, 5.0) if clip else log_chla)


def unpack_params(params):
    params = np.asarray(params, dtype=float)
    if params.shape != (12,):
        raise ValueError("Expected 12 recalibration parameters")
    return params[:5], params[5:10], float(params[10]), float(params[11])


def predict_recalibrated(df, params, low_ratio_col, high_ratio_col, clip=True):
    low_coefs, high_coefs, t1, t2 = unpack_params(params)
    if not (0.1 < t1 < t2 <= 50):
        return None
    low_ratio = pd.to_numeric(df[low_ratio_col], errors="coerce").to_numpy(dtype=float)
    high_ratio = pd.to_numeric(df[high_ratio_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(low_ratio) & np.isfinite(high_ratio) & (low_ratio > 0) & (high_ratio > 0)
    out = np.full(len(df), np.nan, dtype=float)
    if not valid.any():
        return out
    chl_low = ocx_chla(low_ratio[valid], low_coefs, clip=clip)
    chl_high = ocx_chla(high_ratio[valid], high_coefs, clip=clip)
    basis = chl_low
    values = np.empty_like(chl_low)
    use_low = basis < t1
    use_high = basis > t2
    use_blend = ~(use_low | use_high)
    values[use_low] = chl_low[use_low]
    values[use_high] = chl_high[use_high]
    weight_high = (basis[use_blend] - t1) / (t2 - t1)
    values[use_blend] = ((1.0 - weight_high) * chl_low[use_blend]
                         + weight_high * chl_high[use_blend])
    out[valid] = values
    return out


def metrics_log(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted) & (observed > 0) & (predicted > 0)
    observed, predicted = observed[valid], predicted[valid]
    empty = {"N": len(observed), "RMSE_log": np.nan, "R2": np.nan,
             "Pearson_r_log": np.nan, "Slope": np.nan, "Bias_log": np.nan,
             "RMA_slope": np.nan, "RMA_intercept": np.nan,
             "Quantile_RMSE_log": np.nan}
    if len(observed) < 2:
        return empty
    log_true, log_pred = np.log10(observed), np.log10(predicted)
    residual = log_pred - log_true
    std_true, std_pred = np.std(log_true), np.std(log_pred)
    if std_true <= EPS or std_pred <= EPS:
        return empty
    r = float(np.corrcoef(log_true, log_pred)[0, 1])
    rma_slope = float(np.sign(r) * std_pred / std_true)
    rma_intercept = float(np.mean(log_pred) - rma_slope * np.mean(log_true))
    slope = float(r * std_pred / std_true)
    return {
        "N": len(observed),
        "RMSE_log": float(np.sqrt(np.mean(residual**2))),
        "R2": float(r2_score(log_true, log_pred)),
        "Pearson_r_log": r,
        "Slope": float(slope),
        "RMA_slope": rma_slope,
        "RMA_intercept": rma_intercept,
        "Bias_log": float(np.mean(residual)),
        "Quantile_RMSE_log": float(np.sqrt(np.mean((np.sort(log_pred) - np.sort(log_true)) ** 2))),
    }


def objective_from_metrics(metric):
    """Equal-weight sum in log10 space; R2 is the coefficient of determination."""
    needed = [metric["RMA_slope"], metric["RMA_intercept"],
              metric["R2"], metric["Quantile_RMSE_log"]]
    if not np.isfinite(needed).all():
        return INVALID_OBJECTIVE
    return (abs(metric["RMA_slope"] - 1.0) + abs(metric["RMA_intercept"])
            + (1.0 - metric["R2"]) + metric["Quantile_RMSE_log"])


def objective(params, df, low_ratio_col, high_ratio_col):
    pred = predict_recalibrated(df, params, low_ratio_col, high_ratio_col, clip=False)
    if pred is None or np.any(~np.isfinite(pred)) or np.any(pred <= 0):
        return INVALID_OBJECTIVE
    metric = metrics_log(df[TARGET_COL].to_numpy(dtype=float), pred)
    return objective_from_metrics(metric)


def fit_candidate(df, low_ratio_col, high_ratio_col, low_init, high_init, t1_init, t2_init, maxiter):
    init = np.array(list(low_init) + list(high_init) + [t1_init, t2_init], dtype=float)
    return minimize(
        objective, init, args=(df, low_ratio_col, high_ratio_col), method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1e-8, "fatol": 1e-8, "disp": False},
    )


def shape_diagnostics(params):
    """Check unbounded curves on the evaluated MBR grid, without shape heuristics."""
    low_coefs, high_coefs, t1, t2 = unpack_params(params)
    if not (0.1 < t1 < t2 <= 50):
        return {"passed_shape_filter": False, "invalid_thresholds": True,
                "combined_curve_max": np.nan}
    x_line = np.geomspace(*STABILITY_MBR_RANGE, 1000)
    chl_low = ocx_chla(x_line, low_coefs, clip=False)
    chl_high = ocx_chla(x_line, high_coefs, clip=False)
    combined = np.empty_like(chl_low)
    use_low, use_high = chl_low < t1, chl_low > t2
    use_blend = ~(use_low | use_high)
    combined[use_low], combined[use_high] = chl_low[use_low], chl_high[use_high]
    weight_high = (chl_low[use_blend] - t1) / (t2 - t1)
    combined[use_blend] = ((1.0 - weight_high) * chl_low[use_blend]
                           + weight_high * chl_high[use_blend])

    curves = np.concatenate([chl_low, chl_high, combined])
    passed = bool(np.isfinite(curves).all() and (curves > 0).all())
    return {
        "passed_shape_filter": passed,
        "invalid_thresholds": False,
        "nonfinite_or_nonpositive_curve": not passed,
        "stability_mbr_min": STABILITY_MBR_RANGE[0],
        "stability_mbr_max": STABILITY_MBR_RANGE[1],
        "combined_curve_max": float(np.max(combined)) if passed else np.nan,
    }


def _normalise_matched_schema(frame):
    frame = frame.copy()
    if MATCHED_NASA_CHLA_COL in frame.columns and NASA_CHLA_COL not in frame.columns:
        frame[NASA_CHLA_COL] = frame[MATCHED_NASA_CHLA_COL]
    if "L3_date" in frame.columns:
        frame["L3_date"] = pd.to_datetime(frame["L3_date"], errors="coerce").dt.normalize()
    return frame


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _l3_grid_indices(lat, lon):
    """Containing cell in the global 4320 x 8640 Terra L3 4-km mapped grid.

    Cells span 1/24 degree; boundaries belong to the cell to the south/east.
    Computing cell indices avoids nearest-valid-pixel substitution and tolerates
    float32 rounding of the stored pixel-centre coordinates.
    """
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if (not np.isfinite(lat).all() or not np.isfinite(lon).all()
            or (np.abs(lat) > 90).any() or (np.abs(lon) > 180).any()):
        raise ValueError('L3 matching requires finite geographic coordinates in degrees')
    row = np.minimum(np.floor((90.0 - lat) * 24).astype(np.int64), 4319)
    col = np.floor(((lon + 180.0) % 360.0) * 24).astype(np.int64)
    return row, col


def match_insitu_to_l3(insitu_path, terra_dir, output_path, max_date_difference_days=1,
                       max_distance_km=None):
    insitu = pd.read_parquet(insitu_path).copy()
    required = ["Hylak_id", "date", "lat", "lon", TARGET_COL]
    missing = [col for col in required if col not in insitu.columns]
    if missing:
        raise KeyError(f"Missing in-situ columns: {missing}")
    insitu["date"] = pd.to_datetime(insitu["date"], errors="coerce").dt.normalize()
    already_matched = all(col in insitu.columns for col in L3_RRS_VARS) and (
        NASA_CHLA_COL in insitu.columns or MATCHED_NASA_CHLA_COL in insitu.columns)
    if already_matched:
        matched = _normalise_matched_schema(insitu)
        atomic_parquet(matched, output_path)
        return matched

    if max_date_difference_days < 0:
        raise ValueError("max_date_difference_days must be non-negative")
    terra_dir = Path(terra_dir)
    parts = []
    for year, insitu_year in insitu.groupby(insitu["date"].dt.year):
        adjacent = []
        for candidate_year in range(int(year) - 1, int(year) + 2):
            path = terra_dir / f"DB_pixels_TERRA_4km_{candidate_year}.parquet"
            if path.exists():
                adjacent.append(pd.read_parquet(path))
        if not adjacent:
            continue
        l3 = pd.concat(adjacent, ignore_index=True)
        l3["date"] = pd.to_datetime(l3["date"], errors="coerce").dt.normalize()
        keep = ["date", "Hylak_id", "lat", "lon", NASA_CHLA_COL] + L3_RRS_VARS
        missing_l3 = [col for col in keep if col not in l3.columns]
        if missing_l3:
            raise KeyError(f"Missing Terra L3 columns: {missing_l3}")
        l3 = l3[keep].dropna(subset=["date", "Hylak_id", "lat", "lon"])
        l3 = l3[l3["Hylak_id"].isin(insitu_year["Hylak_id"].unique())]
        valid = np.isfinite(l3[[NASA_CHLA_COL] + L3_RRS_VARS].to_numpy(dtype=float)).all(axis=1)
        l3 = l3[valid & (l3[NASA_CHLA_COL] > 0)].copy()
        l3['_grid_row'], l3['_grid_col'] = _l3_grid_indices(l3['lat'], l3['lon'])
        for _, obs in insitu_year.iterrows():
            if pd.isna(obs['lat']) or pd.isna(obs['lon']):
                continue
            grid_row, grid_col = _l3_grid_indices(obs['lat'], obs['lon'])
            candidates = l3[(l3["Hylak_id"] == obs["Hylak_id"])
                            & (l3['_grid_row'] == grid_row)
                            & (l3['_grid_col'] == grid_col)].copy()
            if candidates.empty or pd.isna(obs["date"]):
                continue
            candidates["_day_delta"] = (candidates["date"] - obs["date"]).dt.days.abs()
            candidates = candidates[candidates["_day_delta"] <= max_date_difference_days]
            if candidates.empty:
                continue
            # Absolute lag first; the earlier date wins a +/- lag tie.
            candidates = candidates.sort_values(
                ["_day_delta", "date", "lat", "lon"], kind="mergesort")
            selected = candidates.iloc[0]
            distance = float(_haversine_km(float(obs["lat"]), float(obs["lon"]),
                                           float(selected["lat"]), float(selected["lon"])))
            if max_distance_km is not None and distance > max_distance_km:
                continue
            record = obs.to_dict()
            record.update({
                "L3_date": selected["date"], "L3_lat": selected["lat"],
                "L3_lon": selected["lon"], "L3_distance_km": distance,
                MATCHED_NASA_CHLA_COL: selected[NASA_CHLA_COL],
                NASA_CHLA_COL: selected[NASA_CHLA_COL],
            })
            record.update({col: selected[col] for col in L3_RRS_VARS})
            parts.append(record)
        del l3, adjacent
        gc.collect()
    if not parts:
        raise ValueError("No in-situ rows could be matched to Terra L3 data")
    matched = pd.DataFrame(parts)
    atomic_parquet(matched, output_path)
    return matched


def prepare_matched_data(df):
    df = _normalise_matched_schema(df)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[np.isfinite(df[TARGET_COL]) & (df[TARGET_COL] > 0) & (df[TARGET_COL] < TARGET_MAX)].copy()
    missing = [col for col in L3_RRS_VARS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing matched Rrs columns: {missing}")
    df, ratio_cols = build_ratio_columns(df)
    return df.replace([np.inf, -np.inf], np.nan).reset_index(drop=True), ratio_cols


def candidate_rows(df, low_ratio_col, high_ratio_col):
    subset = df.dropna(subset=[TARGET_COL, low_ratio_col, high_ratio_col]).copy()
    return subset[(subset[low_ratio_col] > 0) & (subset[high_ratio_col] > 0)].reset_index(drop=True)


def validation_lake_ids(df, max_lakes=None):
    counts = df.groupby("Hylak_id").size()
    lake_ids = sorted(counts[counts >= MIN_VALIDATION_OBSERVATIONS].index.tolist())
    if max_lakes is not None:
        if max_lakes < 1:
            raise ValueError("max_lakes must be positive")
        lake_ids = lake_ids[:max_lakes]
    return lake_ids


def make_lolov_splits(df, max_lakes=None, lake_ids=None):
    if lake_ids is None:
        lake_ids = validation_lake_ids(df, max_lakes)
    for lake_id in lake_ids:
        train, validation = df[df["Hylak_id"] != lake_id], df[df["Hylak_id"] == lake_id]
        if train.empty or len(validation) < MIN_VALIDATION_OBSERVATIONS:
            raise ValueError(f"Lake {lake_id} lacks sufficient valid candidate match-ups")
        # Smaller lakes are retained in every training fold.
        yield lake_id, train, validation


def _all_jobs(ratio_cols):
    for ratio_col in ratio_cols:
        for low_name, low_init in OCX_CONFIGS.items():
            for high_name, high_init in OCX_CONFIGS.items():
                for t1_init, t2_init in INITIAL_THRESHOLD_PAIRS:
                    yield (ratio_col, ratio_col, low_name, high_name,
                           low_init, high_init, t1_init, t2_init)


def run_lolov_search(df, output_path, max_jobs=None, max_lakes=None, maxiter=20000,
                     full_search=False):
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    df, ratio_cols = prepare_matched_data(df)
    lake_ids = validation_lake_ids(df, max_lakes)
    if not lake_ids:
        raise ValueError("LOLOV requires lakes with at least 20 match-up observations")
    jobs = list(_all_jobs(ratio_cols))
    if max_jobs is None and not full_search:
        raise ValueError(
            f"The full search contains {len(jobs):,} candidates and many LOLOV fits. "
            "Use --max-jobs for a trial or explicitly pass --full-search."
        )
    if max_jobs is not None:
        if max_jobs < 1:
            raise ValueError("max_jobs must be positive")
        jobs = jobs[:max_jobs]
    print(f"Searching {len(jobs):,} recalibration candidates; "
          f"LOLOV lakes ({len(lake_ids)}): {lake_ids}", flush=True)

    rows = []
    for job_i, job in enumerate(jobs):
        low_ratio_col, high_ratio_col, low_name, high_name, low_init, high_init, t1_init, t2_init = job
        data = candidate_rows(df, low_ratio_col, high_ratio_col)
        # Use the same eligible lakes for every candidate to keep scores comparable.
        counts = data.groupby("Hylak_id").size()
        if any(counts.get(lake_id, 0) < MIN_VALIDATION_OBSERVATIONS for lake_id in lake_ids):
            print(f"job={job_i} skipped: insufficient valid validation samples", flush=True)
            continue

        # Stage 1: full-data fit, followed by numerical stability screening.
        result_all = fit_candidate(data, low_ratio_col, high_ratio_col, low_init, high_init,
                                   t1_init, t2_init, maxiter)
        if (not result_all.success or not np.isfinite(result_all.fun)
                or result_all.fun >= INVALID_OBJECTIVE):
            print(f"job={job_i} skipped: unsuccessful all-data optimization", flush=True)
            continue
        shape = shape_diagnostics(result_all.x)
        if not shape["passed_shape_filter"]:
            print(f"job={job_i} skipped: numerically unstable full-data curve", flush=True)
            continue
        pred_all = predict_recalibrated(data, result_all.x, low_ratio_col, high_ratio_col,
                                       clip=False)
        all_metrics = metrics_log(data[TARGET_COL].to_numpy(dtype=float), pred_all)

        # Stage 2: refit from each candidate's original initialization in each fold.
        fold_metrics = []
        fold_objectives = []
        fold_failed = False
        for _, train, validation in make_lolov_splits(data, lake_ids=lake_ids):
            result = fit_candidate(train, low_ratio_col, high_ratio_col, low_init, high_init,
                                   t1_init, t2_init, maxiter)
            if (not result.success or not np.isfinite(result.fun)
                    or result.fun >= INVALID_OBJECTIVE):
                fold_failed = True
                break
            pred = predict_recalibrated(validation, result.x, low_ratio_col, high_ratio_col,
                                       clip=False)
            if pred is None or not np.isfinite(pred).all() or np.any(pred <= 0):
                fold_failed = True
                break
            metric = metrics_log(validation[TARGET_COL].to_numpy(dtype=float), pred)
            fold_objective = objective_from_metrics(metric)
            if not np.isfinite(fold_objective) or fold_objective >= INVALID_OBJECTIVE:
                fold_failed = True
                break
            fold_metrics.append(metric)
            fold_objectives.append(fold_objective)
        if fold_failed or not fold_metrics:
            print(f"job={job_i} skipped: unsuccessful LOLOV optimization", flush=True)
            continue

        low_opt, high_opt, t1_opt, t2_opt = unpack_params(result_all.x)
        row = {
            "search_method": SEARCH_METHOD,
            "job_i": job_i, "N": len(data), "lakes": data["Hylak_id"].nunique(),
            "validation_lake_ids": json.dumps([int(lake_id) for lake_id in lake_ids]),
            "validation_lakes": len(fold_metrics),
            "trial_search": max_jobs is not None or max_lakes is not None,
            "maxiter": maxiter,
            "low_ratio_col": low_ratio_col, "high_ratio_col": high_ratio_col,
            "low_init_model": low_name, "high_init_model": high_name,
            "init_t1": t1_init, "init_t2": t2_init,
            "mean_val_objective": float(np.mean(fold_objectives)),
            "mean_val_RMSE_log": float(np.nanmean([m["RMSE_log"] for m in fold_metrics])),
            "mean_val_R2": float(np.nanmean([m["R2"] for m in fold_metrics])),
            "all_data_RMSE_log": all_metrics["RMSE_log"], "all_data_R2": all_metrics["R2"],
            "all_data_Pearson_r_log": all_metrics["Pearson_r_log"],
            "all_data_RMA_slope": all_metrics["RMA_slope"],
            "all_data_RMA_intercept": all_metrics["RMA_intercept"],
            "all_data_Slope": all_metrics["Slope"], "all_data_opt_t1_param": t1_opt,
            "all_data_opt_t2_param": t2_opt, "optimizer_success": bool(result_all.success),
            "optimizer_nit": int(result_all.nit), "optimizer_fun": float(result_all.fun),
            **shape,
        }
        for i, value in enumerate(low_opt):
            row[f"all_data_opt_low_a{i}"] = float(value)
        for i, value in enumerate(high_opt):
            row[f"all_data_opt_high_a{i}"] = float(value)
        rows.append(row)
        print(f"job={job_i} mean_val_objective={row['mean_val_objective']:.4f} "
              f"passed={shape['passed_shape_filter']}", flush=True)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["mean_val_objective", "job_i"], kind="mergesort")
    atomic_parquet(summary, Path(output_path).with_suffix(".parquet"))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def select_recalibration(summary_path, rank=1):
    summary = pd.read_csv(summary_path)
    if summary.empty:
        raise ValueError(f"Empty recalibration summary: {summary_path}")
    if ("search_method" not in summary or "mean_val_objective" not in summary
            or not summary["search_method"].eq(SEARCH_METHOD).all()):
        raise ValueError("Re-run the recalibration search: this summary uses an older method")
    passed_values = summary["passed_shape_filter"]
    passed_mask = (passed_values if passed_values.dtype == bool else
                   passed_values.astype(str).str.lower().eq("true"))
    passed = summary[passed_mask].copy()
    if passed.empty:
        raise ValueError("No recalibration candidate passed numerical stability screening")
    passed = passed[np.isfinite(passed["mean_val_objective"])
                    & (passed["mean_val_objective"] < INVALID_OBJECTIVE)]
    candidates = passed.sort_values(["mean_val_objective", "job_i"],
                                    kind="mergesort").reset_index(drop=True)
    if rank < 1 or rank > len(candidates):
        raise ValueError(f"rank must be between 1 and {len(candidates)}")
    row = candidates.iloc[rank - 1]
    params = np.array([row[f"all_data_opt_low_a{i}"] for i in range(5)]
                      + [row[f"all_data_opt_high_a{i}"] for i in range(5)]
                      + [row["all_data_opt_t1_param"], row["all_data_opt_t2_param"]], dtype=float)
    return row, params


def calibration_from_row(row):
    return {
        "low_ratio_col": str(row["low_ratio_col"]),
        "high_ratio_col": str(row["high_ratio_col"]),
        "low_coefs": [float(row[f"all_data_opt_low_a{i}"]) for i in range(5)],
        "high_coefs": [float(row[f"all_data_opt_high_a{i}"]) for i in range(5)],
        "t1": float(row["all_data_opt_t1_param"]), "t2": float(row["all_data_opt_t2_param"]),
        "transition_basis": "low", "target_filter": f"0 < {TARGET_COL} < {TARGET_MAX:g}",
    }


def save_calibration_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_calibration_config(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = payload.get("ocx_config", payload)
    required = {"low_ratio_col", "high_ratio_col", "low_coefs", "high_coefs", "t1", "t2"}
    missing = required - set(config)
    if missing:
        raise KeyError(f"Calibration config is missing: {sorted(missing)}")
    if len(config["low_coefs"]) != 5 or len(config["high_coefs"]) != 5:
        raise ValueError("Each OCx coefficient set must contain five values")
    if config.get("transition_basis", "low") != "low":
        raise ValueError("Only transition_basis='low' is supported")
    return config


def apply_recalibration_to_l3(source_root, output_root, config):
    source_root, output_root = Path(source_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    low_ratio_col, high_ratio_col = config["low_ratio_col"], config["high_ratio_col"]
    params = np.asarray(config["low_coefs"] + config["high_coefs"]
                        + [config["t1"], config["t2"]], dtype=float)
    for source in sorted(source_root.glob("DB_pixels_TERRA_4km_*.parquet")):
        destination = output_root / source.name
        if destination.exists():
            raise FileExistsError(destination)
        frame = pd.read_parquet(source)
        frame, _ = build_ratio_columns(frame)
        if "NASA_Chl-a" not in frame.columns and NASA_CHLA_COL in frame.columns:
            frame["NASA_Chl-a"] = frame[NASA_CHLA_COL]
        frame[RECAL_CHLA_COL] = predict_recalibrated(frame, params, low_ratio_col, high_ratio_col)
        frame[NASA_CHLA_COL] = frame[RECAL_CHLA_COL]
        frame["log_Chl-a"] = np.log10(frame[NASA_CHLA_COL].where(frame[NASA_CHLA_COL] > 0, np.nan))
        frame["log_Chl-a_re"] = frame["log_Chl-a"]
        frame = frame.dropna(subset=[RECAL_CHLA_COL, "log_Chl-a"]).reset_index(drop=True)
        atomic_parquet(frame, destination)
        print(f"{source.name}: {len(frame)} rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--insitu", help="Raw or already L3-matched in-situ parquet")
    parser.add_argument("--matched-output")
    parser.add_argument("--terra-dir")
    parser.add_argument("--summary")
    parser.add_argument("--selected-config-output")
    parser.add_argument("--calibration-config", help="JSON or model_config.json containing ocx_config")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--full-search", action="store_true",
                        help="Explicitly allow all 960 shared-MBR search candidates")
    parser.add_argument("--output")
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--max-lakes", type=int,
                        help="Trial only: limit eligible LOLOV lakes (each has >=20 match-ups)")
    parser.add_argument("--maxiter", type=int, default=20000)
    parser.add_argument("--max-date-difference-days", type=int, default=1)
    parser.add_argument("--max-distance-km", type=float)
    args = parser.parse_args()

    configure_paths(CONFIG)
    terra_dir = args.terra_dir or CONFIG["terra_db_dir"]
    matched_output = args.matched_output or CONFIG["matched_path"]
    summary_path = args.summary or CONFIG["summary_path"]
    selected_config_output = args.selected_config_output or CONFIG["selected_config_path"]

    matched = None
    if args.insitu or args.search:
        matched = match_insitu_to_l3(
            args.insitu or CONFIG["insitu_path"], terra_dir, matched_output,
            max_date_difference_days=args.max_date_difference_days,
            max_distance_km=args.max_distance_km,
        )

    summary = None
    selected_config = load_calibration_config(args.calibration_config) if args.calibration_config else None
    if args.search:
        summary = run_lolov_search(
            matched, summary_path, max_jobs=args.max_jobs, max_lakes=args.max_lakes,
            maxiter=args.maxiter, full_search=args.full_search,
        )
        if summary.empty:
            raise ValueError("No recalibration candidates were fitted successfully")
        row, _ = select_recalibration(summary_path, rank=args.rank)
        selected_config = calibration_from_row(row)
        save_calibration_config(selected_config, selected_config_output)

    if args.apply:
        if selected_config is None:
            row, _ = select_recalibration(summary_path, rank=args.rank)
            selected_config = calibration_from_row(row)
            save_calibration_config(selected_config, selected_config_output)
        apply_recalibration_to_l3(
            terra_dir, args.output or CONFIG["recalibrated_db_dir"], selected_config)

    if not (args.insitu or args.search or args.apply):
        parser.error("Choose at least one action: --insitu, --search, or --apply")


if __name__ == "__main__":
    main()
