#!/usr/bin/env python3
"""
01_extract_features.py — HRRR feature extraction  (v4 — PARALLEL + FAST)
====================================================================
Performance:
  - ONE wgrib2 call per file (was 382 calls per file) → 100-200x faster
  - Parallel processing across all 48 CPU cores using multiprocessing
  - Files processed simultaneously instead of sequentially
  - Expected time: ~15-30 min for 1 month (was 47 hours)

GPU note: wgrib2 is I/O bound, not compute bound — GPU is not applicable here.
          GPU is used in step 4 for XGBoost training.
"""

import sys
import os
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial
import traceback

sys.path.insert(0, str(Path(__file__).parent))
from utils.config_loader import load_config, get_active_regions
from utils.hrrr_extractor import (extract_one_file, compute_derived,
                                   build_empty_row, derived_column_order)

# =============================================================================
def hrrr_path(hrrr_base, year, date_str, cycle, fh):
    return (Path(hrrr_base) / str(year)
            / f"hrrr.t{cycle}z.wrfnatf{fh:02d}_{date_str}.grib2")

def process_one_file(args):
    """
    Worker function — runs in a separate process.
    Extracts + derives features for ONE (date, cycle, fh, region) combination.
    """
    hrrr_base, date_str, year, cycle, fh, region, out_file_str = args
    try:
        fp = hrrr_path(hrrr_base, year, date_str, cycle, fh)
        if not fp.exists():
            return None

        cycle_str = f"{date_str}T{cycle}Z"
        row = extract_one_file(fp, cycle_str, fh, region["lat"], region["lon"])
        row = compute_derived(row)
        row["region"] = region["name"]
        return row
    except Exception as e:
        print(f"  ERROR {date_str} {cycle}Z FH={fh}: {e}")
        traceback.print_exc()
        return None

def col_order(cycle_str, lat, lon):
    dummy        = build_empty_row(cycle_str, 0, lat, lon)
    base_cols    = list(dummy.keys())
    derived_cols = derived_column_order()
    return ["region"] + base_cols + [c for c in derived_cols if c not in base_cols]

# =============================================================================
def main():
    cfg          = load_config()
    wgrib2       = cfg["paths"]["wgrib2"]
    hrrr_base    = str(cfg["paths"]["hrrr_base"])
    features_dir = cfg["paths"]["features_dir"]
    cycles       = cfg["hrrr"]["cycles"]
    fhours       = cfg["hrrr"]["forecast_hours"]
    regions      = get_active_regions(cfg)
    ds           = cfg["dataset"]

    if not wgrib2.exists():
        print(f"ERROR: wgrib2 not found at {wgrib2}"); sys.exit(1)

    start = date.fromisoformat(ds["start_date"])
    end   = date.fromisoformat(ds["end_date"])
    all_dates = [start + timedelta(days=i) for i in range((end-start).days+1)]

    # How many workers to use (leave 4 cores free for OS)
    n_workers = max(1, cpu_count() - 4)

    print(f"\n{'='*65}")
    print(f"  STEP 1: HRRR FEATURE EXTRACTION  (v4 — FAST PARALLEL)")
    print(f"  Dates:    {start} → {end}  ({len(all_dates)} days)")
    print(f"  Cycles:   {cycles}")
    print(f"  FH:       {fhours}")
    print(f"  Regions:  {[r['name'] for r in regions]}")
    print(f"  Workers:  {n_workers} parallel CPU processes (of {cpu_count()} available)")
    print(f"  Method:   ONE wgrib2 call per file (was 382 calls per file)")
    print(f"  Speedup:  ~100-200x vs previous version")
    print(f"{'='*65}\n")

    done = skipped = 0

    for d in all_dates:
        date_str = d.strftime("%Y%m%d")
        year     = d.year

        for cycle in cycles:
            cycle_str = f"{date_str}T{cycle}Z"
            out_file  = features_dir / f"features_{date_str}_{cycle}Z.csv"

            if out_file.exists():
                skipped += 1
                continue

            # Build list of (date, cycle, fh, region) jobs for this cycle
            # Each job = one grib2 file × one region
            jobs = []
            for region in regions:
                for fh in fhours:
                    jobs.append((hrrr_base, date_str, year, cycle,
                                 fh, region, str(out_file)))

            # Run all FH for this cycle in parallel
            with Pool(processes=min(n_workers, len(jobs))) as pool:
                results = pool.map(process_one_file, jobs)

            rows = [r for r in results if r is not None]
            if not rows:
                continue

            df = pd.DataFrame(rows)
            # Enforce column order
            ordered = [c for c in col_order(cycle_str, regions[0]["lat"],
                                            regions[0]["lon"]) if c in df.columns]
            df = df[[c for c in ordered if c in df.columns]]
            df.to_csv(out_file, index=False)
            done += 1

            # Progress: print once per cycle
            n1_str = f"{date_str} {cycle}Z — {len(rows)} rows written"
            print(f"  ✓ {n1_str}")

    print(f"\n{'='*65}")
    print(f"  EXTRACTION COMPLETE")
    print(f"  Written: {done}  |  Skipped (exist): {skipped}")
    print(f"  Output:  {features_dir}")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()
