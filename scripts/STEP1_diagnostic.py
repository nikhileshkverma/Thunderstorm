#!/usr/bin/env python3
"""
STEP 1 — RUN THIS FIRST ON YOUR SERVER
Dumps the exact wgrib2 token names and level strings for FH24.
Send the output back to Claude so missing columns can be fixed.

Usage:
    python3 STEP1_diagnostic.py > diagnostic_output.txt 2>&1
    # Then share diagnostic_output.txt
"""
import subprocess, sys
from pathlib import Path

YEAR=2024; MONTH=1; DAY=2; CYCLE="00"; FH=24
LAT=28.51052; LON=-97.5052
HRRR_BASE = Path("/Storage03/nverma1/HRRR/2024")
DATE_STR   = f"{YEAR}{MONTH:02d}{DAY:02d}"

# ── Check wgrib2 ──────────────────────────────────────────────────────
r = subprocess.run(["which","wgrib2"], capture_output=True, text=True)
if r.returncode != 0:
    print("FATAL: wgrib2 not found in PATH. Cannot proceed.")
    sys.exit(1)
print(f"wgrib2 path : {r.stdout.strip()}")
r2 = subprocess.run(["wgrib2","--version"], capture_output=True, text=True)
print(f"wgrib2 ver  : {(r2.stdout or r2.stderr).splitlines()[0]}")

# ── Find FH24 file ────────────────────────────────────────────────────
candidates = [
    HRRR_BASE / f"hrrr.t{CYCLE}z.wrfnatf{FH:02d}_{DATE_STR}.grib2",
    HRRR_BASE / DATE_STR / f"hrrr.t{CYCLE}z.wrfnatf{FH:02d}.grib2",
    HRRR_BASE / f"hrrr.t{CYCLE}z.wrfnatf{FH:02d}.grib2",
]
grib = next((p for p in candidates if p.exists()), None)
if grib is None:
    print("FATAL: FH24 grib2 file not found. Tried:")
    for c in candidates: print(f"  {c}")
    sys.exit(1)
print(f"File        : {grib}\n")

# ── Pull full inventory with point value ──────────────────────────────
print("Running wgrib2 -s -lon (may take ~30s)...")
r = subprocess.run(
    ["wgrib2", str(grib), "-s", "-lon", str(LON), str(LAT)],
    capture_output=True, text=True
)
lines = r.stdout.splitlines()
print(f"Total records in file: {len(lines)}\n")

# ── Print entries for EVERY variable Waylon needs ────────────────────
WATCH = [
    "PRES","TMP","SPFH","UGRD","VGRD","VVEL","TKE",
    "PMTF","PMTC",
    "DZDT","RELV",
    "LTNG","PWAT","AOTK","RHPW",
    "POT","DPT","RH",
    "WIND","GUST","MAXUWU","MAXVWV",
    "FRICV","SHTFL","LHTFL",
    "CAPE","CIN",
    "HPBL","CNWAT","SFCR","VGTYP",
    "VUCSH","VVCSH",
    "MSLMA","MSLMAM",
    "SOILW","MSTAV",
]

print("="*70)
print("EXACT wgrib2 TOKENS AND LEVEL STRINGS FOR KEY VARIABLES")
print("="*70)
for var in WATCH:
    matches = [l for l in lines if f":{var}:" in l]
    if matches:
        print(f"\n{var}  ({len(matches)} record(s) found):")
        for m in matches:
            parts   = m.split(":")
            level   = parts[4] if len(parts) > 4 else "?"
            val_str = m.split("val=")[-1] if "val=" in m else "no val"
            print(f"   level=[{level}]   val={val_str}")
    else:
        print(f"\n{var}:  *** NOT FOUND IN FILE ***")

print("\n" + "="*70)
print("DONE — share this output so Claude can fix any remaining NaN columns")
print("="*70)