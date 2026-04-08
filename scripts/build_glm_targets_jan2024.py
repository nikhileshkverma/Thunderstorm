"""
Build GLM lightning targets using sliding-window logic
Region: South Texas
Month: January 2024
Grid: HRRR native 3 km (NO resampling)

Author: Nikhilesh Verma
"""

import numpy as np
import pandas as pd
import xarray as xr
import pygrib
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# 1. PATHS (MATCH REDFISH EXACTLY)
# ============================================================

HRRR_DIR = Path("/Storage03/nverma1/HRRR/2024")
GLM_DIR  = Path("/Storage03/nverma1/GOES16_GLM/glm16_2024")
OUT_DIR  = Path("/Storage03/nverma1/lightning_project/output/targets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. DOMAIN: SOUTH TEXAS
# ============================================================

LAT_MIN, LAT_MAX = 25.0, 31.0
LON_MIN, LON_MAX = -100.0, -93.0

BOX_SIZE = 8    # 8 × 3 km ≈ 24 km
STRIDE   = 1    # slide by 3 km

DOY_START = 1
DOY_END   = 31

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def doy_to_yyyymmdd(year, doy):
    """Convert day-of-year to YYYYMMDD string."""
    date = datetime(year, 1, 1) + timedelta(days=doy - 1)
    return date.strftime("%Y%m%d")


def extract_hrrr_grid(grib_file):
    """Extract HRRR lat/lon grid and South Texas mask."""
    with pygrib.open(grib_file) as grbs:
        grb = grbs[1]   # any variable works for grid definition
        _, lats, lons = grb.data()

    mask = (
        (lats >= LAT_MIN) & (lats <= LAT_MAX) &
        (lons >= LON_MIN) & (lons <= LON_MAX)
    )

    return lats, lons, mask


def sliding_windows(mask):
    """Generate sliding-window indices over South Texas."""
    ny, nx = mask.shape
    windows = []

    for i in range(0, ny - BOX_SIZE + 1, STRIDE):
        for j in range(0, nx - BOX_SIZE + 1, STRIDE):
            if mask[i:i+BOX_SIZE, j:j+BOX_SIZE].any():
                windows.append((i, j))

    return windows


def load_glm_flashes(nc_file):
    """Load GLM flash latitude and longitude."""
    with xr.open_dataset(nc_file) as ds:
        lats = ds["flash_lat"].values
        lons = ds["flash_lon"].values
    return lats, lons


# ============================================================
# 4. MAIN PROCESSING LOOP
# ============================================================

rows = []

for doy in range(DOY_START, DOY_END + 1):
    doy_str = f"{doy:03d}"
    date_str = doy_to_yyyymmdd(2024, doy)

    print(f"\nProcessing DOY {doy_str} ({date_str})")

    # --------------------------------------------------------
    # GLM FILES (DOY / DOY / hour / *.nc)
    # --------------------------------------------------------

    day_root = GLM_DIR / doy_str / doy_str
    if not day_root.exists():
        print("  → Missing GLM day folder")
        continue

    glm_files = []
    for hour in range(24):
        hour_dir = day_root / f"{hour:02d}"
        if hour_dir.exists():
            glm_files.extend(sorted(hour_dir.glob("*.nc")))

    if not glm_files:
        print("  → No GLM files found")
        continue

    print(f"  → GLM files: {len(glm_files)}")

    # --------------------------------------------------------
    # HRRR FILES (YYYYMMDD-based filenames)
    # --------------------------------------------------------

    hrrr_files = sorted(HRRR_DIR.glob(f"*_{date_str}.grib2"))
    if not hrrr_files:
        print(f"  → No HRRR files for {date_str}")
        continue

    print(f"  → HRRR files: {len(hrrr_files)}")

    # Use ONE HRRR file to define native grid
    lats, lons, mask = extract_hrrr_grid(hrrr_files[0])
    windows = sliding_windows(mask)

    # --------------------------------------------------------
    # LOAD ALL GLM FLASHES FOR THE DAY
    # --------------------------------------------------------

    flash_lats = []
    flash_lons = []

    for nc in glm_files:
        fl, fn = load_glm_flashes(nc)
        if fl.size > 0:
            flash_lats.append(fl)
            flash_lons.append(fn)

    if not flash_lats:
        print("  → No lightning flashes")
        continue

    flash_lats = np.concatenate(flash_lats)
    flash_lons = np.concatenate(flash_lons)

    # --------------------------------------------------------
    # SLIDING-WINDOW LIGHTNING COUNTS
    # --------------------------------------------------------

    for (i, j) in windows:
        lat_box = lats[i:i+BOX_SIZE, j:j+BOX_SIZE]
        lon_box = lons[i:i+BOX_SIZE, j:j+BOX_SIZE]

        count = np.sum(
            (flash_lats >= lat_box.min()) &
            (flash_lats <= lat_box.max()) &
            (flash_lons >= lon_box.min()) &
            (flash_lons <= lon_box.max())
        )

        rows.append({
            "year": 2024,
            "doy": doy,
            "date": date_str,
            "box_i": i,
            "box_j": j,
            "lightning_count": int(count)
        })

# ============================================================
# 5. SAVE OUTPUT
# ============================================================

df = pd.DataFrame(rows)

out_file = OUT_DIR / "glm_targets_january_2024_south_texas.csv"
df.to_csv(out_file, index=False)

print("\nDONE")
print("Saved:", out_file)
print("Total samples:", len(df))