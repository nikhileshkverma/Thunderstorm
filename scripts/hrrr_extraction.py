"""
WAYLON EXACT SPECIFICATION EXTRACTION
Aligned strictly to Page 3
"""

import pygrib
import pandas as pd
import numpy as np
from pathlib import Path

YEAR = 2024
MONTH = 1
DAY = 2
CYCLE = "00"
FORECAST_HOURS = [6,12,18,24,30,36]

TARGET_LAT = 28.51052
TARGET_LON = -97.5052

HRRR_BASE = Path("/Storage03/nverma1/HRRR/2024")
OUTPUT_FILE = "ExtractedFeatures_FULL_20240102_WAYLON.csv"


def nearest_idx(grb):
    vals, lats, lons = grb.data()
    dist = (lats-TARGET_LAT)**2 + (lons-TARGET_LON)**2
    return np.unravel_index(np.argmin(dist), dist.shape)


def safe(grb, idx):
    val = grb.values[idx]
    return np.nan if np.ma.is_masked(val) else float(val)


all_rows = []

for fh in FORECAST_HOURS:

    file_path = HRRR_BASE / f"hrrr.t{CYCLE}z.wrfnatf{fh:02d}_{YEAR}{MONTH:02d}{DAY:02d}.grib2"
    if not file_path.exists():
        print("Missing:", file_path)
        continue

    print("Processing:", file_path)

    row = {
        "cycle_time": f"{YEAR}{MONTH:02d}{DAY:02d}T{CYCLE}Z",
        "forecast_hour": fh,
        "latitude": TARGET_LAT,
        "longitude": TARGET_LON
    }

    grbs = pygrib.open(str(file_path))

    for grb in grbs:

        try:
            idx = nearest_idx(grb)
            val = safe(grb, idx)
        except:
            continue

        name = grb.name
        level_type = grb.typeOfLevel
        level = getattr(grb, "level", None)

        # ================= SURFACE =================
        if level_type == "surface":

            if name == "Convective available potential energy":
                row["SBCAPE"] = val

            elif name == "Convective inhibition":
                row["SBCIN"] = val

            elif name == "Friction velocity":
                row["FRICV"] = val

            elif name == "Sensible heat net flux":
                row["SHTFL"] = val

            elif name == "Latent heat net flux":
                row["LHTFL"] = val

            elif name == "Plant canopy surface water":
                row["CNWAT"] = val

            elif name == "Surface roughness":
                row["SFCR"] = val

            elif name == "Vegetation type":
                row["VGTYP"] = val

            elif name == "Planetary boundary layer height":
                row["HPBL"] = val

            elif name == "Pressure":
                row["PRES_SFC"] = val

        # ================= 2m =================
        if level_type == "heightAboveGround" and level == 2:

            if name == "Temperature":
                row["TMP_2m"] = val

            elif name == "Potential temperature":
                row["POT_2m"] = val

            elif name == "Dew point temperature":
                row["DPT_2m"] = val

            elif name == "Relative humidity":
                row["RH_2m"] = val

            elif name == "Specific humidity":
                row["SPFH_2m"] = val

        # ================= 10m =================
        if level_type == "heightAboveGround" and level == 10:

            if name == "U component of wind":
                row["UGRD_10m"] = val

            elif name == "V component of wind":
                row["VGRD_10m"] = val

            elif name == "Wind speed":
                row["WIND_10m"] = val

            elif "Maximum" in name and "U component" in name:
                row["MAXUWU"] = val

            elif "Maximum" in name and "V component" in name:
                row["MAXVWV"] = val

        # ================= ENTIRE ATMOS =================
        if level_type == "atmosphereSingleLayer":

            if name == "Precipitable water":
                row["PWAT"] = val

            elif name == "Lightning":
                row["LTNG"] = val

            elif name == "Aerosol optical thickness":
                row["AOTK"] = val

            elif name == "Relative humidity with respect to precipitable water":
                row["RHPW"] = val

        # ================= MSL =================
        if level_type == "meanSea":
            if name == "Mean sea level pressure":
                row["MSLMA"] = val

        # ================= SOIL =================
        if level_type == "depthBelowLandLayer":
            if name == "Volumetric soil moisture content":
                row["SOILW"] = val

        if level_type == "depthBelowLand":
            if name == "Moisture availability":
                row["MSTAV"] = val

        # ================= HYBRID =================
        if level_type == "hybrid" and level:

            if name == "Pressure":
                row[f"PRES_{level}"] = val

            elif name == "Temperature":
                row[f"TMP_{level}"] = val

            elif name == "Specific humidity":
                row[f"SPFH_{level}"] = val

            elif name == "U component of wind":
                row[f"UGRD_{level}"] = val

            elif name == "V component of wind":
                row[f"VGRD_{level}"] = val

            elif name == "Vertical velocity":
                row[f"VVEL_{level}"] = val

            elif name == "Turbulent kinetic energy":
                row[f"TKE_{level}"] = val

    grbs.close()
    all_rows.append(row)

df = pd.DataFrame(all_rows)
df.to_csv(OUTPUT_FILE, index=False)

print("WAYLON EXACT extraction complete.")