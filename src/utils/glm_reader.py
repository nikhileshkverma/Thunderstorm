"""
================================================================================
utils/glm_reader.py — GOES-16 GLM lightning reader and labeler  (v2)
================================================================================
v2 fixes: path auto-detection, diagnostic function, masked array handling
================================================================================
"""

import math
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GLM_BASE, LIGHTNING_WINDOW_HOURS, BOX_HALF_DEG, LIGHTNING_FLASH_THRESHOLD

try:
    import netCDF4 as nc
    NC4_OK = True
except ImportError:
    NC4_OK = False
    print("WARNING: netCDF4 not installed. Run: pip install netCDF4")


def _glm_hour_dir(year: int, jday: int, hour: int) -> Optional[Path]:
    """Try all known GLM directory structures; return first that exists."""
    candidates = [
        # Structure A: confirmed from server — JJJ folder appears TWICE
        GLM_BASE / f"glm16_{year}" / f"{jday:03d}" / f"{jday:03d}" / f"{hour:02d}",
        # Structure B: JJJ once
        GLM_BASE / f"glm16_{year}" / f"{jday:03d}" / f"{hour:02d}",
        # Structure C: flat year/hour
        GLM_BASE / f"glm16_{year}" / f"{hour:02d}",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _parse_glm_start_time(filename: str) -> Optional[datetime]:
    """Parse start time from OR_GLM-L2-LCFA_G16_s{YYYYDDDHHMMSS}_... filename."""
    try:
        s_part = filename.split("_s")[1][:13]
        year   = int(s_part[0:4])
        jday   = int(s_part[4:7])
        hour   = int(s_part[7:9])
        minute = int(s_part[9:11])
        second = int(s_part[11:13])
        base   = datetime(year, 1, 1)
        return base + timedelta(days=jday - 1, hours=hour,
                                minutes=minute, seconds=second)
    except Exception:
        return None


def glm_files_for_window(valid_time: datetime,
                          window_hours: float = LIGHTNING_WINDOW_HOURS) -> List[Path]:
    """Return sorted list of GLM .nc files in [valid_time ± window_hours]."""
    t_start = valid_time - timedelta(hours=window_hours)
    t_end   = valid_time + timedelta(hours=window_hours)

    files = []
    t = (t_start - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    while t <= t_end + timedelta(hours=1):
        glm_dir = _glm_hour_dir(t.year, t.timetuple().tm_yday, t.hour)
        if glm_dir is not None:
            for f in sorted(glm_dir.glob("OR_GLM-L2-LCFA_G16_s*.nc")):
                ft = _parse_glm_start_time(f.name)
                if ft is not None and t_start <= ft <= t_end:
                    files.append(f)
        t += timedelta(hours=1)

    return sorted(set(files))


def check_lightning(valid_time: datetime,
                    center_lat: float, center_lon: float,
                    box_half_deg: float = BOX_HALF_DEG,
                    window_hours: float = LIGHTNING_WINDOW_HOURS,
                    flash_threshold: int = LIGHTNING_FLASH_THRESHOLD) -> int:
    """
    Returns: 1=lightning, 0=no lightning, -1=GLM data unavailable
    """
    if not NC4_OK:
        return -1

    files = glm_files_for_window(valid_time, window_hours)
    if not files:
        return -1

    lat_min = center_lat - box_half_deg
    lat_max = center_lat + box_half_deg
    lon_min = center_lon - box_half_deg
    lon_max = center_lon + box_half_deg

    total_flashes = 0
    for f in files:
        try:
            with nc.Dataset(str(f), "r") as ds:
                if "flash_lat" not in ds.variables:
                    continue
                flash_lat = np.ma.filled(ds.variables["flash_lat"][:], np.nan)
                flash_lon = np.ma.filled(ds.variables["flash_lon"][:], np.nan)
                in_box = (
                    (flash_lat >= lat_min) & (flash_lat <= lat_max) &
                    (flash_lon >= lon_min) & (flash_lon <= lon_max)
                )
                total_flashes += int(np.nansum(in_box))
                if total_flashes >= flash_threshold:
                    return 1
        except Exception:
            continue

    return 1 if total_flashes >= flash_threshold else 0


def compute_valid_time(date_str: str, cycle: str, fh: int) -> datetime:
    base = datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H")
    return base + timedelta(hours=fh)


def label_dataframe(df, center_lat: float, center_lon: float, verbose: bool = False):
    """Add 'valid_time' and 'label' columns to a features DataFrame."""
    import pandas as pd

    labels = []; valid_times = []
    for _, row in df.iterrows():
        ct_str  = str(row["cycle_time"])
        date_s  = ct_str[:8]
        cycle_s = ct_str[9:11]
        fh      = int(row["forecast_hour"])
        vt      = compute_valid_time(date_s, cycle_s, fh)
        lbl     = check_lightning(vt, center_lat, center_lon)
        valid_times.append(vt)
        labels.append(lbl)
        if verbose:
            status = {1: "LIGHTNING", 0: "no-lightning", -1: "DATA-MISSING"}
            print(f"  {ct_str} FH={fh:2d}  valid={vt.strftime('%Y-%m-%d %HZ')}  "
                  f"→ {status.get(lbl,'?')}")

    df = df.copy()
    df.insert(4, "valid_time", valid_times)
    df.insert(5, "label",      labels)
    return df


def run_glm_diagnostic(test_date: str = "20240102", test_hour: int = 12):
    """
    Run on the server to verify GLM paths before Step 2.

    Usage:
        cd /Storage03/nverma1/lightning_project/src
        python -c "
        from utils.glm_reader import run_glm_diagnostic
        run_glm_diagnostic('20240102', 12)
        "
    """
    dt   = datetime.strptime(test_date, "%Y%m%d").replace(hour=test_hour)
    year = dt.year
    jday = dt.timetuple().tm_yday
    hour = dt.hour

    print(f"\n{'='*60}")
    print(f"GLM DIAGNOSTIC — {test_date} {test_hour:02d}Z  (Julian day {jday})")
    print(f"GLM_BASE = {GLM_BASE}")
    print(f"{'='*60}")

    year_dir = GLM_BASE / f"glm16_{year}"
    print(f"\nYear dir exists: {year_dir.exists()}  ({year_dir})")
    if year_dir.exists():
        subdirs = sorted([d.name for d in year_dir.iterdir() if d.is_dir()])
        print(f"  Subfolders (first 5): {subdirs[:5]}")

    jday_dir = year_dir / f"{jday:03d}"
    print(f"\nJulian day dir exists: {jday_dir.exists()}  ({jday_dir})")
    if jday_dir.exists():
        subdirs = sorted([d.name for d in jday_dir.iterdir() if d.is_dir()])
        print(f"  Subfolders: {subdirs}")

    print(f"\nCandidate hour directories:")
    candidates = [
        year_dir / f"{jday:03d}" / f"{jday:03d}" / f"{hour:02d}",
        year_dir / f"{jday:03d}" / f"{hour:02d}",
        year_dir / f"{hour:02d}",
    ]
    for p in candidates:
        exists = p.exists()
        count  = len(list(p.glob("*.nc"))) if exists else 0
        print(f"  {'✓' if exists else '✗'}  {p}  → {count} files")

    print(f"\nFull recursive search for any .nc file under {year_dir}:")
    if year_dir.exists():
        found = list(year_dir.rglob("OR_GLM-L2-LCFA_G16_s*.nc"))[:5]
        if found:
            for f in found:
                print(f"  FOUND: {f.relative_to(year_dir)}")
                ft = _parse_glm_start_time(f.name)
                print(f"         Parsed time: {ft}")
        else:
            print("  *** NO GLM FILES FOUND — check GLM_BASE in config.py ***")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_glm_diagnostic()