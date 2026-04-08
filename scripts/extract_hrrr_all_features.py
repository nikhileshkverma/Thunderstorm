"""
HRRR Feature Extraction Script v2
===================================
Extracts ALL "EXTRACTED FEATURES" from HRRR grib2 files as specified by Waylon Collins.
Fixed for cfgrib==0.9.15.1 — removes unsupported backend_kwargs.

Requirements:
    pip install cfgrib xarray numpy pandas eccodes

Usage:
    python extract_hrrr_all_features_v2.py
    Output: ExtractedFeatures_ALL_<date>.csv
"""

import cfgrib
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION — Edit these values as needed
# ============================================================
HRRR_BASE      = Path("/Storage03/nverma1/HRRR/2024")
TARGET_DATE    = "20240102"          # YYYYMMDD
CYCLE          = 0                   # UTC cycle hour (00)
FORECAST_HOURS = [6, 12, 18, 24, 30, 36]
LAT            =  28.51052
LON            = -97.5052
OUTPUT_CSV     = f"ExtractedFeatures_ALL_{TARGET_DATE}.csv"
# ============================================================


def build_filepath(base, date, cycle, fh):
    return base / f"hrrr.t{cycle:02d}z.wrfnatf{fh:02d}_{date}.grib2"


def nearest_idx(lats, lons, tlat, tlon):
    dist = (lats - tlat)**2 + (lons - tlon)**2
    return np.unravel_index(np.argmin(dist), dist.shape)


def fix_lons(lons):
    return np.where(lons > 180, lons - 360, lons)


def open_grib(filepath, filter_keys):
    """
    Open grib2 with cfgrib 0.9.15.1 compatible API.
    Returns list of datasets or [].
    """
    try:
        return cfgrib.open_datasets(str(filepath), filter_by_keys=filter_keys)
    except Exception as e:
        return []


def get_point(da, lats, lons):
    """Extract scalar value at nearest grid point."""
    lons = fix_lons(lons)
    li, lj = nearest_idx(lats, lons, LAT, LON)
    vals = da.values
    if vals.ndim == 2:
        return float(vals[li, lj])
    elif vals.ndim == 3:
        return float(vals[0, li, lj])
    return np.nan


def extract_2d(filepath, shortName, typeOfLevel, level=None):
    """Extract a single 2D field."""
    fk = {"shortName": shortName, "typeOfLevel": typeOfLevel}
    if level is not None:
        fk["level"] = level
    for ds in open_grib(filepath, fk):
        for v in ds.data_vars:
            da = ds[v]
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            return get_point(da, lats, lons)
    return np.nan


def extract_layer(filepath, shortName, typeOfLevel, topLevel, bottomLevel):
    """Extract a layer-averaged or layer-specific 2D field."""
    fk = {"shortName": shortName, "typeOfLevel": typeOfLevel,
          "topLevel": topLevel, "bottomLevel": bottomLevel}
    for ds in open_grib(filepath, fk):
        for v in ds.data_vars:
            da = ds[v]
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            return get_point(da, lats, lons)

    # Fallback: open without level filter and match via coords
    fk2 = {"shortName": shortName, "typeOfLevel": typeOfLevel}
    for ds in open_grib(filepath, fk2):
        for v in ds.data_vars:
            da = ds[v]
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            # check if level coord matches
            for coord_name in ["topLevel", "heightAboveGroundLayer"]:
                if coord_name in da.coords:
                    coord_val = float(da.coords[coord_name])
                    if int(coord_val) == topLevel:
                        return get_point(da, lats, lons)
            # if only one result returned, trust it
            return get_point(da, lats, lons)
    return np.nan


def extract_hybrid_profile(filepath, shortName, levels):
    """
    Extract vertical profile at hybrid levels.
    Returns dict: {shortName_hybrid_NN: value}
    """
    results = {}
    fk = {"shortName": shortName, "typeOfLevel": "hybridLevel"}

    # Strategy 1: open all levels at once (cfgrib stacks them)
    for ds in open_grib(filepath, fk):
        for v in ds.data_vars:
            da = ds[v]
            lats = ds["latitude"].values
            lons = ds["longitude"].values
            lons_fixed = fix_lons(lons)
            li, lj = nearest_idx(lats, lons_fixed, LAT, LON)

            # find hybrid level coordinate
            lev_coord = None
            for cname in ["hybrid", "hybridLevel", "level"]:
                if cname in ds.coords:
                    lev_coord = cname
                    break

            if lev_coord and da.values.ndim == 3:
                all_levs = ds.coords[lev_coord].values
                for lev in levels:
                    idx_arr = np.where(all_levs == lev)[0]
                    if len(idx_arr) > 0:
                        val = float(da.values[idx_arr[0], li, lj])
                        results[f"{shortName}_L{lev:02d}"] = val
            elif da.values.ndim == 2:
                # single level in this dataset — check what level it is
                if lev_coord:
                    lev_val = int(float(ds.coords[lev_coord]))
                    if lev_val in levels:
                        results[f"{shortName}_L{lev_val:02d}"] = float(da.values[li, lj])
            break  # only need first matching variable in dataset

    if len(results) == len(levels):
        return results

    # Strategy 2: open level by level (slower but reliable)
    for lev in levels:
        if f"{shortName}_L{lev:02d}" in results:
            continue
        fk2 = {"shortName": shortName, "typeOfLevel": "hybridLevel", "level": lev}
        for ds in open_grib(filepath, fk2):
            for v in ds.data_vars:
                da = ds[v]
                lats = ds["latitude"].values
                lons = ds["longitude"].values
                lons_fixed = fix_lons(lons)
                li, lj = nearest_idx(lats, lons_fixed, LAT, LON)
                vals = da.values
                if vals.ndim == 2:
                    results[f"{shortName}_L{lev:02d}"] = float(vals[li, lj])
                elif vals.ndim == 3:
                    results[f"{shortName}_L{lev:02d}"] = float(vals[0, li, lj])
                break
            break

    # Fill missing with NaN
    for lev in levels:
        key = f"{shortName}_L{lev:02d}"
        if key not in results:
            results[key] = np.nan

    return results


def process_fh(filepath, fh):
    print(f"\n  FH={fh:02d} | {filepath.name}")
    row = {
        "Date": TARGET_DATE,
        "Cycle_UTC": f"{CYCLE:02d}",
        "ForecastHour": fh,
        "Latitude": LAT,
        "Longitude": LON,
    }

    if not filepath.exists():
        print(f"  [ERROR] File not found: {filepath}")
        return row

    # ----------------------------------------------------------------
    # 1. HYBRID LEVEL PROFILES (levels 1-50)
    # ----------------------------------------------------------------
    print("    Hybrid 1-50: PRES, TMP, SPFH, UGRD, VGRD, VVEL, TKE ...")
    hybrid50_map = {
        "PRES_hybrid": "pres",
        "TMP_hybrid":  "t",
        "SPFH_hybrid": "q",
        "UGRD_hybrid": "u",
        "VGRD_hybrid": "v",
        "VVEL_hybrid": "w",
        "TKE_hybrid":  "tke",
    }
    for label_prefix, sn in hybrid50_map.items():
        profile = extract_hybrid_profile(filepath, sn, range(1, 51))
        for k, v in profile.items():
            lev = k.split("_L")[1]
            row[f"{label_prefix}_L{lev}"] = v

    # ----------------------------------------------------------------
    # 2. HYBRID LEVEL PROFILES (levels 1-20) — Particulate Matter
    # ----------------------------------------------------------------
    print("    Hybrid 1-20: PMTF, PMTC ...")
    for label_prefix, sn in [("PMTF_hybrid", "pmtf"), ("PMTC_hybrid", "pmtc")]:
        profile = extract_hybrid_profile(filepath, sn, range(1, 21))
        for k, v in profile.items():
            lev = k.split("_L")[1]
            row[f"{label_prefix}_L{lev}"] = v

    # ----------------------------------------------------------------
    # 3. SURFACE VARIABLES
    # ----------------------------------------------------------------
    print("    Surface variables ...")
    row["CAPE_surface"]         = extract_2d(filepath, "cape",  "surface")
    row["CIN_surface"]          = extract_2d(filepath, "cin",   "surface")
    row["FRICV_surface"]        = extract_2d(filepath, "fricv", "surface")
    row["SHTFL_surface"]        = extract_2d(filepath, "shtfl", "surface")
    row["LHTFL_surface"]        = extract_2d(filepath, "lhtfl", "surface")
    row["PRES_surface"]         = extract_2d(filepath, "sp",    "surface")
    row["HPBL_surface"]         = extract_2d(filepath, "hpbl",  "surface")
    row["CNWAT_surface"]        = extract_2d(filepath, "cnwat", "surface")
    row["SFCR_surface"]         = extract_2d(filepath, "fsr",   "surface")
    row["VGTYP_surface"]        = extract_2d(filepath, "vtype", "surface")
    # Soil moisture / moisture availability
    row["SOILW_0m_below"]       = extract_2d(filepath, "vsw",   "depthBelowLandLayer", 0)
    row["MSTAV_surface"]        = extract_2d(filepath, "mstav", "depthBelowLandLayer")
    if np.isnan(row["MSTAV_surface"]):
        row["MSTAV_surface"]    = extract_2d(filepath, "mstav", "surface")

    # ----------------------------------------------------------------
    # 4. 2 m ABOVE GROUND
    # ----------------------------------------------------------------
    print("    2m variables ...")
    row["TMP_2m"]               = extract_2d(filepath, "2t",  "heightAboveGround", 2)
    row["POT_2m"]               = extract_2d(filepath, "pt",  "heightAboveGround", 2)
    row["DPT_2m"]               = extract_2d(filepath, "2d",  "heightAboveGround", 2)
    row["RH_2m"]                = extract_2d(filepath, "2r",  "heightAboveGround", 2)
    row["SPFH_2m"]              = extract_2d(filepath, "q",   "heightAboveGround", 2)

    # ----------------------------------------------------------------
    # 5. 10 m ABOVE GROUND
    # ----------------------------------------------------------------
    print("    10m variables ...")
    row["UGRD_10m"]             = extract_2d(filepath, "10u",   "heightAboveGround", 10)
    row["VGRD_10m"]             = extract_2d(filepath, "10v",   "heightAboveGround", 10)
    row["WIND_10m_maxPastHour"] = extract_2d(filepath, "10si",  "heightAboveGround", 10)
    row["MAXUWU_10m"]           = extract_2d(filepath, "maxuw", "heightAboveGround", 10)
    row["MAXVWV_10m"]           = extract_2d(filepath, "maxvw", "heightAboveGround", 10)

    # ----------------------------------------------------------------
    # 6. ENTIRE ATMOSPHERE
    # ----------------------------------------------------------------
    print("    Entire atmosphere variables ...")
    row["PWAT_entireAtmos"]     = extract_2d(filepath, "pwat", "atmosphereSingleLayer")
    row["LTNG_entireAtmos"]     = extract_2d(filepath, "ltng", "atmosphereSingleLayer")
    row["AOTK_entireAtmos"]     = extract_2d(filepath, "aotk", "atmosphereSingleLayer")
    row["RHPW_entireAtmos"]     = extract_2d(filepath, "rhpw", "atmosphereSingleLayer")
    # Fallback typeOfLevel variants for PWAT/LTNG
    for key, sn in [("PWAT_entireAtmos","pwat"), ("LTNG_entireAtmos","ltng"),
                    ("AOTK_entireAtmos","aotk"), ("RHPW_entireAtmos","rhpw")]:
        if np.isnan(row[key]):
            for tol in ["atmosphere", "atmosphereLayer", "atmosphereColumn"]:
                v = extract_2d(filepath, sn, tol)
                if not np.isnan(v):
                    row[key] = v
                    break

    # ----------------------------------------------------------------
    # 7. MEAN SEA LEVEL
    # ----------------------------------------------------------------
    print("    MSL pressure ...")
    row["MSLMAM_msl"]           = extract_2d(filepath, "mslet", "meanSea")
    if np.isnan(row["MSLMAM_msl"]):
        row["MSLMAM_msl"]       = extract_2d(filepath, "prmsl", "meanSea")

    # ----------------------------------------------------------------
    # 8. 0.5-0.8 SIGMA LAYER — DZDT (Vertical Velocity Geometric)
    # ----------------------------------------------------------------
    print("    DZDT sigma layer ...")
    dzdt = np.nan
    for tol in ["sigmav1", "sigmaLayer", "sigma", "hybridLevel"]:
        dzdt = extract_2d(filepath, "dzdt", tol)
        if not np.isnan(dzdt):
            break
    row["DZDT_05_08sigma"]      = dzdt

    # ----------------------------------------------------------------
    # 9. LAYER-BASED: RELV (Relative Vorticity)
    # ----------------------------------------------------------------
    print("    RELV layers ...")
    row["RELV_2000_0m_AGL"]     = extract_layer(filepath, "absv",
                                                 "heightAboveGroundLayer", 2000, 0)
    row["RELV_1000_0m_AGL"]     = extract_layer(filepath, "absv",
                                                 "heightAboveGroundLayer", 1000, 0)
    # Some HRRR versions store RELV as "relv" shortName
    if np.isnan(row["RELV_2000_0m_AGL"]):
        row["RELV_2000_0m_AGL"] = extract_layer(filepath, "relv",
                                                 "heightAboveGroundLayer", 2000, 0)
    if np.isnan(row["RELV_1000_0m_AGL"]):
        row["RELV_1000_0m_AGL"] = extract_layer(filepath, "relv",
                                                 "heightAboveGroundLayer", 1000, 0)

    # ----------------------------------------------------------------
    # 10. LAYER-BASED: VUCSH / VVCSH (0-1000 m shear)
    # ----------------------------------------------------------------
    print("    Shear (VUCSH, VVCSH) ...")
    row["VUCSH_0_1000m"]        = extract_layer(filepath, "vucsh",
                                                 "heightAboveGroundLayer", 1000, 0)
    row["VVCSH_0_1000m"]        = extract_layer(filepath, "vvcsh",
                                                 "heightAboveGroundLayer", 1000, 0)

    return row


def main():
    print("=" * 65)
    print("HRRR ALL-FEATURES EXTRACTION  (cfgrib 0.9.15.1 compatible)")
    print(f"  Date     : {TARGET_DATE}")
    print(f"  Cycle    : {CYCLE:02d} UTC")
    print(f"  FH list  : {FORECAST_HOURS}")
    print(f"  Lat/Lon  : {LAT}, {LON}")
    print(f"  Base dir : {HRRR_BASE}")
    print("=" * 65)

    rows = []
    for fh in FORECAST_HOURS:
        fp = build_filepath(HRRR_BASE, TARGET_DATE, CYCLE, fh)
        rows.append(process_fh(fp, fh))

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_CSV, index=False)
    total_cols = len(df.columns)

    # Report NaN columns — use feature cols only (skip metadata cols at start)
    feature_df = df.iloc[:, 5:]
    nan_mask = feature_df.isnull().all()
    nan_cols = feature_df.columns[nan_mask].tolist()
    non_nan = feature_df.iloc[0].notna().sum()

    if nan_cols:
        print(f"\n  [INFO] {len(nan_cols)} columns are all-NaN across all forecast hours "
              f"(variable may not exist in these files or uses a different shortName):")
        for c in nan_cols:
            print(f"    - {c}")

    print(f"\n✅ Done. Saved: {OUTPUT_CSV}")
    print(f"   Rows: {len(df)} | Total columns: {total_cols} | "
          f"Non-NaN feature cols (FH=06): {non_nan} | All-NaN cols: {len(nan_cols)}")


if __name__ == "__main__":
    main()