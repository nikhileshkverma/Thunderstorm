"""
Build HRRR feature file using sliding-window logic
Region: South Texas
Month: January 2024
Grid: HRRR native 3 km (NO resampling)

Author: Nikhilesh Verma
"""

import numpy as np
import pandas as pd
import pygrib
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# 1. PATHS (REDFISH)
# ============================================================

HRRR_DIR = Path("/Storage03/nverma1/HRRR_DATA/2024")
OUT_DIR  = Path("/Storage03/nverma1/lightning_project/output/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. DOMAIN: SOUTH TEXAS
# ============================================================

LAT_MIN, LAT_MAX = 25.0, 31.0
LON_MIN, LON_MAX = -100.0, -93.0

BOX_SIZE = 8     # 8 × 3 km ≈ 24 km
STRIDE   = 1     # slide by 3 km

DOY_START = 1
DOY_END   = 31

# ============================================================
# 3. HRRR VARIABLES TO EXTRACT
# ============================================================

HRRR_VARS = {
    "t2m":  {"name": "2 metre temperature"},
    "cape": {"name": "Convective available potential energy"},
    "cin":  {"name": "Convective inhibition"},
    "refd": {"shortName": "refd"},
    "u10":  {"name": "10 metre U wind component"},
    "v10":  {"name": "10 metre V wind component"},
}

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def extract_grid(grib_file):
    """Extract lat/lon grid and South Texas mask."""
    with pygrib.open(grib_file) as grbs:
        grb = grbs[1]
        _, lats, lons = grb.data()

    mask = (
        (lats >= LAT_MIN) & (lats <= LAT_MAX) &
        (lons >= LON_MIN) & (lons <= LON_MAX)
    )
    return lats, lons, mask


def sliding_windows(mask):
    """Generate sliding-window indices."""
    ny, nx = mask.shape
    windows = []

    for i in range(0, ny - BOX_SIZE, STRIDE):
        for j in range(0, nx - BOX_SIZE, STRIDE):
            if mask[i:i+BOX_SIZE, j:j+BOX_SIZE].any():
                windows.append((i, j))

    return windows


def load_variable(grbs, var_cfg):
    """Safely load HRRR variable."""
    try:
        if "name" in var_cfg:
            grb = grbs.select(name=var_cfg["name"])[0]
        else:
            grb = grbs.select(shortName=var_cfg["shortName"])[0]
        data, _, _ = grb.data()
        return data
    except Exception:
        return None


# ============================================================
# 5. MAIN LOOP
# ============================================================

rows = []

for doy in range(DOY_START, DOY_END + 1):
    doy_str = f"{doy:03d}"

    date_obj = datetime(2024, 1, 1) + timedelta(days=doy - 1)
    date_str = date_obj.strftime("%Y%m%d")

    print(f"\nProcessing DOY {doy_str} ({date_str})")

    hrrr_files = sorted(HRRR_DIR.glob(f"*{date_str}*.grib2"))
    print(f"  → HRRR files: {len(hrrr_files)}")

    if not hrrr_files:
        continue

    # Use first file to define grid
    lats, lons, mask = extract_grid(hrrr_files[0])
    windows = sliding_windows(mask)

    for grib_file in hrrr_files:
        with pygrib.open(grib_file) as grbs:

            var_data = {}
            for key, cfg in HRRR_VARS.items():
                var_data[key] = load_variable(grbs, cfg)

            if any(v is None for v in var_data.values()):
                continue

            for (i, j) in windows:
                record = {
                    "year": 2024,
                    "doy": doy,
                    "date": int(date_str),
                    "box_i": i,
                    "box_j": j,
                }

                for key, arr in var_data.items():
                    box = arr[i:i+BOX_SIZE, j:j+BOX_SIZE]
                    record[f"{key}_mean"] = float(np.mean(box))
                    record[f"{key}_max"]  = float(np.max(box))
                    record[f"{key}_std"]  = float(np.std(box))

                rows.append(record)

# ============================================================
# 6. SAVE OUTPUT
# ============================================================

df = pd.DataFrame(rows)
out_file = OUT_DIR / "hrrr_features_january_2024_south_texas.csv"
df.to_csv(out_file, index=False)

print("\nDONE")
print("Saved:", out_file)
print("Total samples:", len(df))