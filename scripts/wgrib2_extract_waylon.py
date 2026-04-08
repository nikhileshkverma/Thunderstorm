#!/usr/bin/env python3
"""
================================================================================
HRRR Feature Extraction  —  FINAL v3
================================================================================
Project : Deep Learning Thunderstorm Model for the CONUS
Authors : Verma, Collins, Kamangir, King, Tissot  |  3 December 2025

CHANGES FROM v2 → v3 (based on server diagnostic output 11 Mar 2026):
  - CONFIRMED file naming: hrrr.t00z.wrfnatf{FH:02d}_{YYYYMMDD}.grib2
  - CONFIRMED hybrid level strings: "1 hybrid level", "2 hybrid level", etc.
  - CONFIRMED surface level string: "surface"
  - CONFIRMED 2m level string: "2 m above ground"
  - FIXED: MAXUW / MAXVWV were NaN in all 6 rows
    Root cause: these variables appear with a time-range suffix in wgrib2 output
    e.g. "MAXUW:10 m above ground:1-0 hour max fcst" — the variable token
    itself IS "MAXUW" but the level string may not simply be "10 m above ground"
    Fix: catch MAXUW/MAXVW at ANY level (they appear only once per file)

Waylon Collins email corrections (Feb 25-28, 2026) — all still applied:
  FIX 1 — SBCAPE  : CAPE at "surface" (0-SFC) ONLY
  FIX 2 — SBCIN   : CIN  at "surface" (0-SFC) ONLY
  FIX 3 — WIND_10m: WIND at "10 m above ground" — NOT GUST
  FIX 4 — MSLMA   : Correct token (PDF typo was "MSLMAM")

Waylon DEGRIB spot-check values for FH24 (lat=28.51052, lon=-97.5052):
  SBCAPE   = 0.0       (0-SFC ONLY — NOT 2.6 from 0-3000-HTGL)
  SBCIN    = 0.0       (0-SFC ONLY — NOT -8.0 from 25500-0-SPDL)
  WIND_10m = 4.47433   (WIND — NOT 6.78825 GUST)
  MSLMA    = 101730.0  Pa
================================================================================
"""

import subprocess
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Tuple

# ============================================================
#  CONFIG
# ============================================================
YEAR           = 2024
MONTH          = 1
DAY            = 2
CYCLE          = "00"
FORECAST_HOURS = [6, 12, 18, 24, 30, 36]

LAT = 28.51052
LON = -97.5052

HRRR_BASE   = Path("/Storage03/nverma1/HRRR/2024")
OUTPUT_FILE = "ExtractedFeatures_FULL_20240102_WAYLON_FINAL_1.csv"

VERBOSE = False   # Set True to debug individual line matching

# ============================================================
#  PRE-FLIGHT: confirm wgrib2 is available
# ============================================================
r = subprocess.run(["which", "wgrib2"], capture_output=True, text=True)
if r.returncode != 0:
    # Try explicit path used on redfish server
    r2 = subprocess.run(["/home/nverma1/bin/wgrib2", "--version"],
                        capture_output=True, text=True)
    if r2.returncode == 0:
        WGRIB2 = "/home/nverma1/bin/wgrib2"
    else:
        print("FATAL: wgrib2 not found.", file=sys.stderr)
        sys.exit(1)
else:
    WGRIB2 = r.stdout.strip()
print(f"wgrib2: {WGRIB2}")

# ============================================================
#  VARIABLES TO EXTRACT
#  GUST is intentionally absent — Waylon confirmed WIND ≠ GUST
# ============================================================
TARGET_VARS = {
    # Hybrid multi-level
    "PRES", "TMP", "SPFH", "UGRD", "VGRD", "VVEL", "TKE",
    "PMTF", "PMTC",
    # Single-level
    "DZDT", "RELV", "LTNG",
    "POT", "DPT", "RH",
    "WIND",             # 10-m wind speed — NOT GUST
    "MAXUW", "MAXVW", # hourly max 10-m wind components
    "FRICV", "SHTFL", "LHTFL",
    "CAPE", "CIN",      # surface only → SBCAPE / SBCIN
    "PWAT", "AOTK", "RHPW",
    "VUCSH", "VVCSH",
    "HPBL",
    "MSLMA",
    "SOILW", "MSTAV",
    "CNWAT", "SFCR", "VGTYP",
    # GUST is NOT here
}

# ============================================================
#  SURFACE LEVEL STRINGS — confirmed from diagnostic output
# ============================================================
SURFACE_LEVELS = {"surface", "0-sfc", "0 m above ground"}

def is_surface(level: str) -> bool:
    return level.strip().lower() in SURFACE_LEVELS

# ============================================================
#  WGRIB2 RUNNER
# ============================================================
def run_wgrib2(file_path: Path) -> List[str]:
    cmd = [WGRIB2, str(file_path), "-s", "-lon", str(LON), str(LAT)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [WARN] wgrib2 error: {r.stderr[:200]}", file=sys.stderr)
    return r.stdout.splitlines()

# ============================================================
#  LINE PARSER
#  Format: rec:byte:d=YYYYMMDDHH:VAR:LEVEL:FORECAST:...:val=VALUE
#  parts[3] = variable name
#  parts[4] = level string
# ============================================================
def parse_line(line: str):
    if "val=" not in line:
        return None
    try:
        val = float(line.split("val=")[-1].strip())
    except ValueError:
        return None
    parts = line.split(":")
    if len(parts) < 6:
        return None
    var   = parts[3].strip()
    level = parts[4].strip()
    if var not in TARGET_VARS:
        return None
    return var, level, val

# ============================================================
#  HYBRID LEVEL NUMBER DETECTION
#  Confirmed format from diagnostic: "1 hybrid level", "2 hybrid level"
# ============================================================
def hybrid_level_num(level: str) -> int:
    if "hybrid" not in level.lower():
        return -1
    for tok in level.split():
        if tok.isdigit():
            return int(tok)
    return -1

# ============================================================
#  FILE LOCATOR
#  Confirmed naming from server: hrrr.t00z.wrfnatf{FH:02d}_{YYYYMMDD}.grib2
# ============================================================
def find_file(fh: int) -> Optional[Path]:
    date_str = f"{YEAR}{MONTH:02d}{DAY:02d}"
    candidates = [
        # PRIMARY — confirmed naming convention from server
        HRRR_BASE / f"hrrr.t{CYCLE}z.wrfnatf{fh:02d}_{date_str}.grib2",
        # Fallbacks
        HRRR_BASE / date_str / f"hrrr.t{CYCLE}z.wrfnatf{fh:02d}.grib2",
        HRRR_BASE / f"hrrr.t{CYCLE}z.wrfnatf{fh:02d}.grib2",
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        print(f"  [SKIP FH={fh:02d}] File not found. Tried:", file=sys.stderr)
        for c in candidates:
            print(f"    {c}", file=sys.stderr)
    return found

# ============================================================
#  BUILD EMPTY ROW — all 426 required columns
# ============================================================
def build_empty_row(fh: int) -> dict:
    row = {
        "cycle_time":    f"{YEAR}{MONTH:02d}{DAY:02d}T{CYCLE}Z",
        "forecast_hour": fh,
        "latitude":      LAT,
        "longitude":     LON,
    }
    for lvl in range(1, 51):
        for v in ["PRES", "TMP", "SPFH", "UGRD", "VGRD", "VVEL", "TKE"]:
            row[f"{v}_{lvl}"] = None
    for lvl in range(1, 21):
        row[f"PMTF_{lvl}"] = None
        row[f"PMTC_{lvl}"] = None
    for col in [
        "DZDT", "RELV_2000", "RELV_1000",
        "LTNG", "PWAT", "AOTK", "RHPW",
        "TMP_2m", "POT_2m", "DPT_2m", "RH_2m", "SPFH_2m",
        "UGRD_10m", "VGRD_10m", "WIND_10m", "MAXUW", "MAXVW",
        "FRICV", "SHTFL", "LHTFL",
        "SBCAPE", "SBCIN",
        "HPBL", "PRES_SFC", "CNWAT", "SFCR", "VGTYP",
        "VUCSH", "VVCSH",
        "MSLMA",
        "SOILW", "MSTAV",
    ]:
        row[col] = None
    return row

# ============================================================
#  MAIN EXTRACTION
# ============================================================
all_rows = []

for fh in tqdm(FORECAST_HOURS, desc="Forecast Hours"):

    file_path = find_file(fh)
    if file_path is None:
        continue

    print(f"  Processing FH={fh:02d}: {file_path.name}")
    lines = run_wgrib2(file_path)
    row   = build_empty_row(fh)

    for line in lines:
        parsed = parse_line(line)
        if parsed is None:
            continue
        var, level, val = parsed
        level_lo = level.lower()

        # ── 1. HYBRID SIGMA LEVELS ────────────────────────────────────
        lvl = hybrid_level_num(level)
        if lvl != -1:
            if 1 <= lvl <= 50 and var in ("PRES","TMP","SPFH","UGRD","VGRD","VVEL","TKE"):
                col = f"{var}_{lvl}"
                if row[col] is None:
                    row[col] = val
            if 1 <= lvl <= 20 and var in ("PMTF","PMTC"):
                col = f"{var}_{lvl}"
                if row[col] is None:
                    row[col] = val
            continue   # never matches surface/2m/10m

        # ── 2. DZDT  —  0.5-0.8 sigma layer ─────────────────────────
        if var == "DZDT":
            if "sigma" in level_lo and "0.5" in level and "0.8" in level:
                row["DZDT"] = val

        # ── 3. RELV  —  2000-0m and 1000-0m above ground ────────────
        elif var == "RELV":
            if "2000" in level and "above ground" in level_lo:
                row["RELV_2000"] = val
            elif "1000" in level and "above ground" in level_lo:
                row["RELV_1000"] = val

        # ── 4. SBCAPE  —  surface ONLY ───────────────────────────────
        # Waylon: 0-SFC = 0.0 (correct), 0-3000-HTGL = 2.6 (wrong)
        elif var == "CAPE":
            if is_surface(level):
                row["SBCAPE"] = val
            elif VERBOSE:
                print(f"    [SKIP CAPE layer: {level}]")

        # ── 5. SBCIN  —  surface ONLY ────────────────────────────────
        # Waylon: 0-SFC = 0.0 (correct), 25500-0-SPDL = -8.0 (wrong)
        elif var == "CIN":
            if is_surface(level):
                row["SBCIN"] = val
            elif VERBOSE:
                print(f"    [SKIP CIN layer: {level}]")

        # ── 6. 2 m above ground ───────────────────────────────────────
        elif "2 m above ground" in level_lo:
            if   var == "TMP":  row["TMP_2m"]  = val
            elif var == "POT":  row["POT_2m"]  = val
            elif var == "DPT":  row["DPT_2m"]  = val
            elif var == "RH":   row["RH_2m"]   = val
            elif var == "SPFH": row["SPFH_2m"] = val

        # ── 7. 10 m above ground ──────────────────────────────────────
        # WIND only — GUST is excluded from TARGET_VARS entirely
        elif "10 m above ground" in level_lo:
            if   var == "UGRD": row["UGRD_10m"] = val
            elif var == "VGRD": row["VGRD_10m"] = val
            elif var == "WIND": row["WIND_10m"] = val   # NOT GUST

        # ── 8. MAXUW / MAXVW ────────────────────────────────────────
        # FIX v3: catch at ANY level — they appear only once per file.
        # In some HRRR versions the level is "10 m above ground" but in
        # others it carries a time-range suffix making level_lo != simple
        # "10 m above ground". Catching at any level is safe because
        # MAXUW/MAXVW are unique tokens — no other variable shares them.
        elif var == "MAXUW":
            if row["MAXUW"] is None:
                row["MAXUW"] = val
                if VERBOSE:
                    print(f"    [MAXUW matched at level: {level}]")
        elif var == "MAXVW":
            if row["MAXVW"] is None:
                row["MAXVW"] = val
                if VERBOSE:
                    print(f"    [MAXVW matched at level: {level}]")

        # ── 9. Wind shear 0-1000 m above ground ──────────────────────
        elif var == "VUCSH":
            if "1000" in level and "above ground" in level_lo:
                row["VUCSH"] = val
        elif var == "VVCSH":
            if "1000" in level and "above ground" in level_lo:
                row["VVCSH"] = val

        # ── 10. Entire-atmosphere scalars ─────────────────────────────
        elif var == "LTNG":  row["LTNG"] = val
        elif var == "PWAT":  row["PWAT"] = val
        elif var == "AOTK":  row["AOTK"] = val
        elif var == "RHPW":  row["RHPW"] = val

        # ── 11. Surface fluxes and misc surface vars ──────────────────
        elif var == "FRICV": row["FRICV"] = val
        elif var == "SHTFL": row["SHTFL"] = val
        elif var == "LHTFL": row["LHTFL"] = val
        elif var == "HPBL":  row["HPBL"]  = val
        elif var == "CNWAT": row["CNWAT"] = val
        elif var == "SFCR":  row["SFCR"]  = val
        elif var == "VGTYP": row["VGTYP"] = val

        # ── 12. Surface pressure (PRES at surface only) ───────────────
        elif var == "PRES":
            if is_surface(level):
                row["PRES_SFC"] = val

        # ── 13. MSLMA — PDF typo "MSLMAM" corrected by Waylon ────────
        elif var == "MSLMA":
            row["MSLMA"] = val

        # ── 14. Soil moisture ─────────────────────────────────────────
        elif var == "SOILW":
            row["SOILW"] = val
        elif var == "MSTAV":
            row["MSTAV"] = val

    all_rows.append(row)

# ============================================================
#  SAVE — enforcing exact column order
# ============================================================
df = pd.DataFrame(all_rows)

ordered_cols = (
    ["cycle_time", "forecast_hour", "latitude", "longitude"]
    + [f"{v}_{i}" for v in ["PRES","TMP","SPFH","UGRD","VGRD","VVEL","TKE"]
                  for i in range(1, 51)]
    + [f"PMTF_{i}" for i in range(1, 21)]
    + [f"PMTC_{i}" for i in range(1, 21)]
    + ["DZDT", "RELV_2000", "RELV_1000",
       "LTNG", "PWAT", "AOTK", "RHPW",
       "TMP_2m", "POT_2m", "DPT_2m", "RH_2m", "SPFH_2m",
       "UGRD_10m", "VGRD_10m", "WIND_10m", "MAXUW", "MAXVW",
       "FRICV", "SHTFL", "LHTFL", "SBCAPE", "SBCIN",
       "HPBL", "PRES_SFC", "CNWAT", "SFCR", "VGTYP",
       "VUCSH", "VVCSH",
       "MSLMA",
       "SOILW", "MSTAV"]
)
ordered_cols = [c for c in ordered_cols if c in df.columns]
df = df[ordered_cols]
df.to_csv(OUTPUT_FILE, index=False)

# ============================================================
#  VERIFICATION REPORT
# ============================================================
SEP = "=" * 70
print(f"\n{SEP}")
print(f"EXTRACTION COMPLETE  →  {OUTPUT_FILE}")
print(f"Rows: {len(df)}   |   Columns: {len(df.columns)}")
print(SEP)

expected_cols = 4 + 7*50 + 2*20 + 32  # = 426
col_status = "✓ OK" if len(df.columns) == expected_cols else f"✗ MISMATCH (expected {expected_cols})"
print(f"Column count: {len(df.columns)} — {col_status}")

# Null check — single-value columns only
hybrid_pfx = tuple(f"{v}_" for v in
    ["PRES","TMP","SPFH","UGRD","VGRD","VVEL","TKE","PMTF","PMTC"])
meta = {"cycle_time","forecast_hour","latitude","longitude"}
single_cols = [c for c in df.columns if not c.startswith(hybrid_pfx) and c not in meta]

null_counts = df[single_cols].isnull().sum()
bad = null_counts[null_counts > 0]
print(f"\nNull counts per non-hybrid column:")
if bad.empty:
    print("  ✓ ALL columns fully populated — zero NaN values")
else:
    for col, cnt in bad.items():
        fhs = df.loc[df[col].isnull(), "forecast_hour"].tolist()
        print(f"  ✗ {col:15s}  NaN in {cnt}/{len(df)} rows  (FH: {fhs})")

# Waylon DEGRIB spot-check for FH=24
print("\nWaylon DEGRIB spot-check (FH=24):")
fh24 = df[df["forecast_hour"] == 24]
if fh24.empty:
    print("  [FH24 row not present]")
else:
    checks = [
        ("SBCAPE",   0.0,       "0-SFC → correct=0.0, wrong=2.6 (0-3000-HTGL)"),
        ("SBCIN",    0.0,       "0-SFC → correct=0.0, wrong=-8.0 (25500-0-SPDL)"),
        ("WIND_10m", 4.47433,   "WIND  → correct=4.47433, wrong=6.78825 (GUST)"),
        ("MSLMA",    101730.0,  "MSLMA [Pa]"),
    ]
    for col, expected_val, note in checks:
        actual = fh24[col].values[0] if col in fh24 else "MISSING"
        if actual == "MISSING" or actual is None:
            sym = "✗"
        elif abs(float(actual) - expected_val) < 0.1:
            sym = "✓"
        else:
            sym = "✗"
        print(f"  {sym} {col:12s}  got={actual}  expected≈{expected_val}  ({note})")

print(SEP)