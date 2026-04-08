"""
Location: 28.51052, -97.5052
Date: 1 Jan 2024
Cycle: 00Z
Forecast Hours: 06,12,18,24,30,36

Output: ExtractedFeatures.csv
"""

import pygrib
import pandas as pd
import numpy as np
from pathlib import Path

# ====================================================
# CONFIGURATION
# ====================================================

YEAR = 2024
MONTH = 1
DAY = 1
CYCLE = "00"
FORECAST_HOURS = [6, 12, 18, 24, 30, 36]

TARGET_LAT = 28.51052
TARGET_LON = -97.5052

HRRR_BASE = Path("/Storage03/nverma1/HRRR/2024")
OUTPUT_FILE = "ExtractedFeatures.csv"

# ====================================================
# Helper: Nearest Grid Point (NO interpolation)
# ====================================================

def get_nearest_index(grb):
    values, lats, lons = grb.data()
    dist = (lats - TARGET_LAT)**2 + (lons - TARGET_LON)**2
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return idx

# ====================================================
# Safe value extraction
# ====================================================

def safe_value(grb, idx):
    raw_val = grb.values[idx]
    if np.ma.is_masked(raw_val):
        return np.nan
    return float(raw_val)

# ====================================================
# Extraction
# ====================================================

all_rows = []

for fh in FORECAST_HOURS:

    file_path = HRRR_BASE / f"hrrr.t{CYCLE}z.wrfnatf{fh:02d}_{YEAR}{MONTH:02d}{DAY:02d}.grib2"

    if not file_path.exists():
        print("Missing:", file_path)
        continue

    print("Processing:", file_path)

    # Initialize full row with NaNs FIRST
    row = {
        "cycle_time": f"{YEAR}{MONTH:02d}{DAY:02d}T{CYCLE}Z",
        "forecast_hour": fh,
        "latitude": TARGET_LAT,
        "longitude": TARGET_LON,

        # Surface variables
        "SBCAPE": np.nan,
        "CIN": np.nan,
        "T_2m": np.nan,
        "DPT_2m": np.nan,
        "PRES_SFC": np.nan,
        "PWAT": np.nan,
        "UGRD_10m": np.nan,
        "VGRD_10m": np.nan
    }

    # Pre-create ALL hybrid variables
    for lvl in range(1, 51):
        row[f"TMP_{lvl}"] = np.nan
        row[f"SPFH_{lvl}"] = np.nan
        row[f"UGRD_{lvl}"] = np.nan
        row[f"VGRD_{lvl}"] = np.nan
        row[f"VVEL_{lvl}"] = np.nan

    grbs = pygrib.open(str(file_path))

    for grb in grbs:

        try:
            idx = get_nearest_index(grb)
            value = safe_value(grb, idx)
        except:
            continue

        short = grb.shortName
        level = getattr(grb, "level", None)
        type_level = grb.typeOfLevel

        # ================================
        # SURFACE VARIABLES
        # ================================

        if short == "cape" and type_level == "surface":
            row["SBCAPE"] = value

        elif short == "cin" and type_level == "surface":
            row["CIN"] = value

        elif short == "2t":
            row["T_2m"] = value

        elif short == "2d":
            row["DPT_2m"] = value

        elif short == "sp":
            row["PRES_SFC"] = value

        elif short == "pwat":
            row["PWAT"] = value

        elif short == "10u":
            row["UGRD_10m"] = value

        elif short == "10v":
            row["VGRD_10m"] = value

        # ================================
        # HYBRID LEVEL VARIABLES
        # ================================

        elif type_level == "hybrid" and 1 <= level <= 50:

            if short == "t":
                row[f"TMP_{level}"] = value

            elif short == "q":
                row[f"SPFH_{level}"] = value

            elif short == "u":
                row[f"UGRD_{level}"] = value

            elif short == "v":
                row[f"VGRD_{level}"] = value

            elif short == "w":
                row[f"VVEL_{level}"] = value

    grbs.close()
    all_rows.append(row)

# ====================================================
# Save
# ====================================================

df = pd.DataFrame(all_rows)

# Order columns
meta_cols = ["cycle_time", "forecast_hour", "latitude", "longitude"]

surface_cols = [
    "SBCAPE", "CIN", "DPT_2m", "T_2m",
    "PRES_SFC", "PWAT", "UGRD_10m", "VGRD_10m"
]

vertical_cols = []
for lvl in range(1, 51):
    for prefix in ["TMP", "SPFH", "UGRD", "VGRD", "VVEL"]:
        vertical_cols.append(f"{prefix}_{lvl}")

df = df[meta_cols + surface_cols + vertical_cols]

df.to_csv(OUTPUT_FILE, index=False)

print("\nExtraction Complete.")
print("Saved:", OUTPUT_FILE)