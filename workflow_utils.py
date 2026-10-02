"""Shared feature generation and I/O helpers for the MODIS workflow."""
import sys
sys.dont_write_bytecode = True


# features
import numpy as np
import pandas as pd

L3_RRS_BANDS = ["412", "443", "469", "488", "531", "547", "555", "645", "667", "678"]
L3_RRS_VARS = [f"Rrs_{band}" for band in L3_RRS_BANDS]
L3_RECALIBRATION_RRS_VARS = ["Rrs_412", "Rrs_443", "Rrs_488", "Rrs_547"]
L3_RECALIBRATION_LOW_COEFS = np.array([
    -0.14471456898707660,
    -4.49295453250624810,
    1.56446770620056608,
    -0.13256524097893285,
    -0.28815879834898661,
], dtype=float)
L3_RECALIBRATION_HIGH_COEFS = np.array([
    0.21383900650409210,
    -2.47587295479457481,
    1.91209390524842782,
    1.17778471481358893,
    -1.62506612850538179,
], dtype=float)
L3_RECALIBRATION_T1 = 0.79779644838836417
L3_RECALIBRATION_T2 = 3.73704908174786876

def add_sr_derived_rrs(df):

    SR_cols = ["SR_469", "SR_555", "SR_645", "SR_859", "SR_1240", "SR_1640", "SR_2130"]
    min_values = df[["SR_859", "SR_1240", "SR_1640", "SR_2130"]].min(axis=1)

    for col in SR_cols:
        df[f'{col}_Rrs'] = (df[col] - min_values) / np.pi
        
    return df

def add_spectral_indices(df):

    # Ratio_blue: 469nm / 555nm
    df['Ratio_blue']        = df['SR_469']     / df['SR_555']
    df['Ratio_blue_SR_Rrs'] = df['SR_469_Rrs'] / df['SR_555_Rrs']

    # FAI: 859nm - {645nm + (1240nm - 645nm) * (859-645)/(1240-645) }
    df['FAI']        = df['SR_859']     - (df['SR_645']     + (df['SR_1240']     - df['SR_645'])     * (859 - 645) / (1240 - 645))
    df['FAI_SR_Rrs'] = df['SR_859_Rrs'] - (df['SR_645_Rrs'] + (df['SR_1240_Rrs'] - df['SR_645_Rrs']) * (859 - 645) / (1240 - 645))

    # NDVI: (859nm - 645nm) / (859nm + 645nm)
    df['NDVI']        = (df['SR_859']     - df['SR_645'])     / (df['SR_859']     + df['SR_645'])
    df['NDVI_SR_Rrs'] = (df['SR_859_Rrs'] - df['SR_645_Rrs']) / (df['SR_859_Rrs'] + df['SR_645_Rrs'])
    
    return df

def add_seasonal_features(df):
    # Seasonality Adjustment
    date = df['date']

    # Account for leap years (366 days) vs. common years (365 days)
    days_in_year = date.dt.is_leap_year.map({True: 366.0, False: 365.0})
    doy = date.dt.dayofyear.astype(np.float32)

    # Shift DOY for Southern Hemisphere by 6 months to align global seasonality
    is_southern = df['lat'] <= 0
    doy_shifted = doy.copy()
    doy_shifted[is_southern] = doy_shifted[is_southern] + days_in_year[is_southern] / 2
    doy_shifted = doy_shifted % days_in_year

    # Cyclical DOY features and latitudinal amplitude interaction
    df['DOY_sin'] = np.sin(2 * np.pi * doy_shifted / days_in_year)
    df['DOY_cos'] = np.cos(2 * np.pi * doy_shifted / days_in_year)
    df['lat_amp'] = np.sin(np.deg2rad(np.abs(df['lat'].values))).astype(np.float32)
    df['season_sin'] = (df['lat_amp'] * df['DOY_sin']).astype(np.float32)
    df['season_cos'] = (df['lat_amp'] * df['DOY_cos']).astype(np.float32)
    
    return df

def add_l3_chla_recalibration(df):

    def _safe_ratio(numerator, denominator):
        return numerator / denominator.where(denominator != 0, np.nan)

    def _safe_log10(values):
        return np.log10(values.where(values > 0, np.nan))

    def _ocx_chla(x, coefs):
        log_chla = (
            coefs[0]
            + coefs[1] * x
            + coefs[2] * x**2
            + coefs[3] * x**3
            + coefs[4] * x**4
        )
        return 10 ** np.clip(log_chla, -5.0, 5.0)

    missing = [col for col in L3_RECALIBRATION_RRS_VARS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing L3 Rrs columns for Chl-a recalibration: {missing}")

    df = df.copy()
    if "Chl-a" in df.columns and "NASA_Chl-a" not in df.columns:
        df["NASA_Chl-a"] = df["Chl-a"]

    df["Rrs_blue"] = df[["Rrs_412", "Rrs_443", "Rrs_488"]].max(axis=1)
    df["Rrs_green"] = df["Rrs_547"]
    df["MBR"] = _safe_ratio(df["Rrs_blue"], df["Rrs_green"])
    df["X"] = _safe_log10(df["MBR"])
    df["X2"] = df["X"] ** 2
    df["X3"] = df["X"] ** 3
    df["X4"] = df["X"] ** 4
    df["ratio"] = df["MBR"]
    df["R"] = df["X"]
    df["R2"] = df["X2"]
    df["R3"] = df["X3"]
    df["R4"] = df["X4"]

    df["OCx_Chl-a_low"] = _ocx_chla(df["X"], L3_RECALIBRATION_LOW_COEFS)
    df["OCx_Chl-a_high"] = _ocx_chla(df["X"], L3_RECALIBRATION_HIGH_COEFS)

    df["Chl-a_re"] = np.nan
    valid = (
        np.isfinite(df["OCx_Chl-a_low"]) &
        np.isfinite(df["OCx_Chl-a_high"]) &
        (df["MBR"] > 0)
    )
    use_low = valid & (df["OCx_Chl-a_low"] < L3_RECALIBRATION_T1)
    use_high = valid & (df["OCx_Chl-a_low"] > L3_RECALIBRATION_T2)
    use_blend = valid & ~(use_low | use_high)

    df["OCx_blend_source"] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[use_low, "OCx_blend_source"] = "low"
    df.loc[use_high, "OCx_blend_source"] = "high"
    df.loc[use_blend, "OCx_blend_source"] = "blend"

    df.loc[use_low, "Chl-a_re"] = df.loc[use_low, "OCx_Chl-a_low"]
    df.loc[use_high, "Chl-a_re"] = df.loc[use_high, "OCx_Chl-a_high"]

    weight_high = (df.loc[use_blend, "OCx_Chl-a_low"] - L3_RECALIBRATION_T1) / (
        L3_RECALIBRATION_T2 - L3_RECALIBRATION_T1
    )
    df.loc[use_blend, "Chl-a_re"] = (
        (1.0 - weight_high) * df.loc[use_blend, "OCx_Chl-a_low"]
        + weight_high * df.loc[use_blend, "OCx_Chl-a_high"]
    )

    df["Chl-a"] = df["Chl-a_re"]
    df["log_Chl-a"] = _safe_log10(df["Chl-a_re"])
    df["log_Chl-a_re"] = df["log_Chl-a"]

    return df

def add_rrs_bio_optical_features(df):
    """Add OCx diagnostics without replacing an upstream recalibration.

    The recalibration search/application step is authoritative. Re-applying the
    fixed published formula here would silently overwrite a newly selected
    calibration during construction of the 4.6 km training table.
    """
    required = ["Rrs_412", "Rrs_443", "Rrs_488", "Rrs_547", "Chl-a_re"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            "Expected recalibrated Terra L3 input; missing columns: "
            f"{missing}. Run recalibrate_terra_l3_chla.py first."
        )

    df = df.copy()
    df["Rrs_blue"] = df[["Rrs_412", "Rrs_443", "Rrs_488"]].max(axis=1)
    df["Rrs_green"] = df["Rrs_547"]
    df["MBR"] = df["Rrs_blue"] / df["Rrs_green"].where(df["Rrs_green"] != 0, np.nan)
    df["X"] = np.log10(df["MBR"].where(df["MBR"] > 0, np.nan))
    df["X2"] = df["X"] ** 2
    df["X3"] = df["X"] ** 3
    df["X4"] = df["X"] ** 4
    df["Chl-a"] = pd.to_numeric(df["Chl-a_re"], errors="coerce")
    df["log_Chl-a"] = np.log10(df["Chl-a"].where(df["Chl-a"] > 0, np.nan))
    df["log_Chl-a_re"] = df["log_Chl-a"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required + ["MBR", "X", "log_Chl-a"]).reset_index(drop=True)
    return df



# settings
import os
from pathlib import Path


def configure_paths(config):
    root = Path(os.environ.get('GLCD_DATA_DIR', Path(__file__).resolve().parent/'data')).resolve()
    for key, value in list(config.items()):
        if isinstance(value, str) and value.startswith('data/'):
            config[key] = str(root / value[5:])
    if 'shapefile_path' in config and os.environ.get('GLCD_LAKE_SHP'):
        config['shapefile_path'] = os.environ['GLCD_LAKE_SHP']
    return config


# processing
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def record_event(report, event, **details):
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                                 'event': event, **details}, default=str) + '\n')


def atomic_parquet(frame, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
    try:
        frame.to_parquet(temporary, index=False, compression='snappy')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def require_matching_files(expected, actual, report, **context):
    """Compare names, not only counts (equal counts can hide different dates)."""
    missing, extra = sorted(set(expected)-set(actual)), sorted(set(actual)-set(expected))
    if missing or extra:
        record_event(report, 'file_coverage_error', missing=missing, extra=extra, **context)
        raise ValueError(f'File coverage mismatch: {len(missing)} missing, '
                         f'{len(extra)} unexpected; see {report}')


# downloads
import subprocess


def run_download(command):
    # CalledProcessError includes the full command, including the bearer token.
    # Check the return code ourselves and emit only non-sensitive information.
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        raise RuntimeError('Could not start aria2c; check its installation') from None
    if result.returncode:
        raise RuntimeError(f'Download failed (aria2c exit {result.returncode}); '
                           'check access, network and remote availability. Command/credentials omitted.')
