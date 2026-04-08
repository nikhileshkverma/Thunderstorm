#!/usr/bin/env python3
"""
================================================================================
01_extract_features.py — Extract HRRR features for all dates in config range
================================================================================
For each date in [DATASET_START, DATASET_END] and each cycle in HRRR_CYCLES,
extracts all forecast hours and saves one CSV per (date, cycle) to FEATURES_DIR.

Run:
    cd /Storage03/nverma1/lightning_project/src
    source /Storage03/nverma1/lightning_project/scripts/venv/bin/activate
    python 01_extract_features.py

Output:
    data/features/features_{YYYYMMDD}_{CC}Z.csv   (one per date×cycle)
================================================================================
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    HRRR_BASE, HRRR_CYCLES, HRRR_FORECAST_HOURS,
    DATASET_START, DATASET_END, REGIONS, FEATURES_DIR, WGRIB2
)
from utils.hrrr_extractor import extract_one_file, compute_derived

# =============================================================================
def hrrr_file(year: int, date_str: str, cycle: str, fh: int) -> Path:
    return (HRRR_BASE / str(year)
            / f"hrrr.t{cycle}z.wrfnatf{fh:02d}_{date_str}.grib2")

# Column ordering for output CSV
def derived_col_order():
    cols = []
    for lvl in range(1, 51): cols.append(f"POT_{lvl}")
    cols.append("EPT_2m")
    for lvl in range(1, 51): cols.append(f"EPT_{lvl}")
    for lvl in range(1, 51): cols.append(f"RH_{lvl}")
    for lvl in range(1, 51): cols.append(f"DPT_{lvl}")
    for lvl in range(1, 50): cols.append(f"VTMPLR_{lvl}{lvl+1}")
    for lvl in range(1, 50): cols.append(f"VS_{lvl}{lvl+1}")
    cols += ["BowenRatio_Mean", "ZL_MIN", "ZL_MAX", "diff_LFC_PSFC"]
    return cols

# =============================================================================
def main():
    if not WGRIB2.exists():
        print(f"ERROR: wgrib2 not found at {WGRIB2}")
        sys.exit(1)

    # Date range
    current = DATASET_START
    all_dates = []
    while current <= DATASET_END:
        all_dates.append(current)
        current += timedelta(days=1)

    print(f"\n{'='*65}")
    print(f"  STEP 1: HRRR FEATURE EXTRACTION")
    print(f"  Dates:  {DATASET_START} → {DATASET_END}  ({len(all_dates)} days)")
    print(f"  Cycles: {HRRR_CYCLES}")
    print(f"  FH:     {HRRR_FORECAST_HOURS}")
    print(f"  Regions: {[r['name'] for r in REGIONS]}")
    print(f"{'='*65}\n")

    total_jobs = len(all_dates) * len(HRRR_CYCLES)
    done = skipped = 0

    for d in tqdm(all_dates, desc="Dates"):
        date_str = d.strftime("%Y%m%d")
        year     = d.year

        for cycle in HRRR_CYCLES:
            cycle_str = f"{date_str}T{cycle}Z"
            out_file  = FEATURES_DIR / f"features_{date_str}_{cycle}Z.csv"

            if out_file.exists():
                skipped += 1
                continue   # resume-friendly: skip already extracted

            rows = []

            for region in REGIONS:
                lat = region["lat"]
                lon = region["lon"]

                for fh in HRRR_FORECAST_HOURS:
                    fp = hrrr_file(year, date_str, cycle, fh)
                    if not fp.exists():
                        continue

                    row = extract_one_file(fp, cycle_str, cycle, fh, lat, lon)
                    row = compute_derived(row)
                    row["region"] = region["name"]
                    rows.append(row)

            if not rows:
                continue

            df = pd.DataFrame(rows)

            # Enforce column order
            from utils.hrrr_extractor import build_empty_row
            base_cols    = list(build_empty_row(cycle_str, 0, lat, lon).keys())
            derived_cols = [c for c in derived_col_order() if c in df.columns]
            meta_cols    = ["region"]
            all_cols     = (["region"] + base_cols +
                            [c for c in derived_cols if c not in base_cols])
            df = df[[c for c in all_cols if c in df.columns]]

            df.to_csv(out_file, index=False)
            done += 1

    print(f"\n{'='*65}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Written: {done}  |  Skipped (already exist): {skipped}")
    print(f"  Output:  {FEATURES_DIR}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
