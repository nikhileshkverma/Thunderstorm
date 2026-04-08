"""
utils/glm_reader.py — GOES-16 GLM lightning reader and labeler  (v2)
Confirmed GLM path: /Storage03/nverma1/GOES16_GLM/glm16_{YYYY}/{JJJ}/{JJJ}/{HH}/
"""
import math
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config

try:
    import netCDF4 as nc
    NC4_OK = True
except ImportError:
    NC4_OK = False

def _cfg(): return load_config()

def _glm_hour_dir(year: int, jday: int, hour: int) -> Optional[Path]:
    base = _cfg()["paths"]["glm_base"]
    candidates = [
        base / f"glm16_{year}" / f"{jday:03d}" / f"{jday:03d}" / f"{hour:02d}",
        base / f"glm16_{year}" / f"{jday:03d}" / f"{hour:02d}",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def _parse_glm_start_time(filename: str) -> Optional[datetime]:
    try:
        s = filename.split("_s")[1][:13]
        year=int(s[0:4]); jday=int(s[4:7]); hour=int(s[7:9])
        minute=int(s[9:11]); second=int(s[11:13])
        return datetime(year,1,1) + timedelta(days=jday-1, hours=hour,
                                              minutes=minute, seconds=second)
    except Exception:
        return None

def glm_files_for_window(valid_time: datetime, window_hours: float = None) -> List[Path]:
    if window_hours is None:
        window_hours = _cfg()["labeling"]["window_hours"]
    t_start = valid_time - timedelta(hours=window_hours)
    t_end   = valid_time + timedelta(hours=window_hours)
    files = []
    t = (t_start - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    while t <= t_end + timedelta(hours=1):
        d = _glm_hour_dir(t.year, t.timetuple().tm_yday, t.hour)
        if d:
            for f in sorted(d.glob("OR_GLM-L2-LCFA_G16_s*.nc")):
                ft = _parse_glm_start_time(f.name)
                if ft and t_start <= ft <= t_end:
                    files.append(f)
        t += timedelta(hours=1)
    return sorted(set(files))

def check_lightning(valid_time: datetime, center_lat: float, center_lon: float,
                    box_half_deg: float = None, window_hours: float = None,
                    flash_threshold: int = None) -> int:
    cfg = _cfg()["labeling"]
    if box_half_deg is None:   box_half_deg   = cfg["box_half_deg"]
    if window_hours is None:   window_hours   = cfg["window_hours"]
    if flash_threshold is None: flash_threshold = cfg["flash_threshold"]

    if not NC4_OK:
        return -1
    files = glm_files_for_window(valid_time, window_hours)
    if not files:
        return -1

    lat_min = center_lat - box_half_deg; lat_max = center_lat + box_half_deg
    lon_min = center_lon - box_half_deg; lon_max = center_lon + box_half_deg
    total = 0
    for f in files:
        try:
            with nc.Dataset(str(f), "r") as ds:
                if "flash_lat" not in ds.variables: continue
                flat = np.ma.filled(ds.variables["flash_lat"][:], np.nan)
                flon = np.ma.filled(ds.variables["flash_lon"][:], np.nan)
                in_box = ((flat>=lat_min)&(flat<=lat_max)&
                          (flon>=lon_min)&(flon<=lon_max))
                total += int(np.nansum(in_box))
                if total >= flash_threshold: return 1
        except Exception:
            continue
    return 1 if total >= flash_threshold else 0

def compute_valid_time(date_str: str, cycle: str, fh: int) -> datetime:
    base = datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H")
    return base + timedelta(hours=fh)

def label_dataframe(df, center_lat: float, center_lon: float, verbose: bool = False):
    import pandas as pd
    labels = []; valid_times = []
    for _, row in df.iterrows():
        ct = str(row["cycle_time"])
        vt = compute_valid_time(ct[:8], ct[9:11], int(row["forecast_hour"]))
        lbl = check_lightning(vt, center_lat, center_lon)
        valid_times.append(vt); labels.append(lbl)
        if verbose:
            s = {1:"LIGHTNING",0:"no-lightning",-1:"DATA-MISSING"}
            print(f"  {ct} FH={int(row['forecast_hour']):2d}  valid={vt.strftime('%Y-%m-%d %HZ')}  → {s.get(lbl,'?')}")
    df = df.copy()
    df.insert(4, "valid_time", valid_times)
    df.insert(5, "label",      labels)
    return df
