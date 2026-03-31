"""
HRRR Official Waylon Feature Extraction (STRUCTURED FORMAT)
Full Vertical Preservation
All Cycles + Forecast Hours
Author: Nikhilesh Verma
"""

import pygrib
import pandas as pd
from pathlib import Path
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

YEAR = 2024
MONTH = 1
DAY = 1

FORECAST_HOURS = [0, 6, 12, 18, 24, 30, 36]
CYCLES = ["00", "06", "12", "18"]

HRRR_BASE = Path("/Storage03/nverma1/HRRR")

TARGET_LAT = 28.5
TARGET_LON = -97.5

OUTPUT_FILE = f"hrrr_waylon_features_STRUCTURED_{YEAR}{MONTH:02d}{DAY:02d}.csv"

# ============================================================
# FEATURE SHORT NAMES (Clean Column Naming)
# ============================================================

FEATURE_SHORT_NAMES = {
    "Temperature": "T",
    "Relative humidity": "RH",
    "Specific humidity": "Q",
    "Geopotential Height": "HGT",
    "Convective available potential energy": "CAPE",
    "Convective inhibition": "CIN",
    "U component of wind": "U",
    "V component of wind": "V",
    "Vertical velocity": "W",
    "Updraft helicity": "UH",
    "2 metre dewpoint temperature": "DPT2m",
    "Precipitable water": "PW",
    "Maximum/Composite radar reflectivity": "REFL",
    "Total precipitation": "PRECIP",
    "Convective precipitation": "CPRECIP",
    "2 metre temperature": "T2m",
    "10 metre U wind component": "U10m",
    "10 metre V wind component": "V10m",
    "Surface pressure": "PSFC"
}

RAW_FEATURES = list(FEATURE_SHORT_NAMES.keys())

# ============================================================
# FIND NEAREST GRID POINT
# ============================================================

def find_grid_index(grb):
    _, lats, lons = grb.data()
    dist = (lats - TARGET_LAT)**2 + (lons - TARGET_LON)**2
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return idx, lats[idx], lons[idx]

# ============================================================
# EXTRACTION LOOP (STRUCTURED FORMAT)
# ============================================================

records = []

for cycle in CYCLES:

    cycle_time = f"{YEAR}{MONTH:02d}{DAY:02d}T{cycle}Z"

    for fh in FORECAST_HOURS:

        grib_file = (
            HRRR_BASE /
            str(YEAR) /
            f"hrrr.t{cycle}z.wrfnatf{fh:02d}_{YEAR}{MONTH:02d}{DAY:02d}.grib2"
        )

        if not grib_file.exists():
            print("Missing:", grib_file)
            continue

        print("Processing:", grib_file)

        grbs = pygrib.open(str(grib_file))

        # Create ONE row per cycle + forecast hour
        row = {
            "cycle_time": cycle_time,
            "forecast_hour": fh
        }

        lat_actual = None
        lon_actual = None

        for grb in grbs:

            if grb.name not in RAW_FEATURES:
                continue

            short_name = FEATURE_SHORT_NAMES[grb.name]

            level = getattr(grb, "level", None)

            idx, lat_actual, lon_actual = find_grid_index(grb)
            value = float(grb.values[idx])

            # Add lat/lon only once
            row["latitude"] = lat_actual
            row["longitude"] = lon_actual

            # Vertical level variable
            if level is not None and isinstance(level, (int, float)):
                col_name = f"{short_name}_lvl{int(level)}"
            else:
                col_name = short_name

            row[col_name] = value

        grbs.close()

        records.append(row)

# ============================================================
# SAVE OUTPUT
# ============================================================

df = pd.DataFrame(records)

# Sort columns nicely
fixed_cols = ["cycle_time", "forecast_hour", "latitude", "longitude"]
other_cols = sorted([c for c in df.columns if c not in fixed_cols])
df = df[fixed_cols + other_cols]

df.to_csv(OUTPUT_FILE, index=False)

print("\nExtraction Complete.")
print("Saved:", OUTPUT_FILE)
print("Total Rows:", len(df))