#!/usr/bin/env python3
"""
================================================================================
DERIVED FEATURES CALCULATION  —  v2
================================================================================
Project : Deep Learning Thunderstorm Model for the CONUS
Authors : Verma, Collins, Kamangir, King, Tissot

PURPOSE
-------
Step 1: Extract HRRR features for cycle 12Z 1 January 2025,
        forecast hours 6, 12, 18, 24, 30, 36
        at lat=28.51052, lon=-97.5052  (SAME case as verified extraction)

Step 2: Calculate ALL center-point derived features per Waylon Collins email
        (16 March 2026) and instructions PDF:
          POT_1..50
          EPT_1..50, EPT_2m
          RH_1..50
          DPT_1..50
          VTMPLR_12 .. VTMPLR_4950  (49 layers)
          VS_12 .. VS_4950           (49 layers)
          BowenRatio_Mean
          diff_LFC_PSFC              (NaN expected: SBCAPE=0 for all FH)
          ZL_MIN, ZL_MAX             (same value at single point)

        SKIPPED per Waylon email:
          MXR  (point 1)
          GMI features  (point 3)
          Box-region statistics  (point 4)

Waylon corrections (still applied from previous work):
  - SBCAPE: surface layer only
  - SBCIN:  surface layer only
  - WIND_10m: WIND not GUST
  - MSLMA: correct token name

OUTPUT
------
  DerivedFeatures_20250101_12Z.csv
  (6 rows × [426 extracted + 303 derived] = 729 columns)

DEPENDENCIES
------------
  pip install pandas numpy metpy tqdm
================================================================================
"""

import subprocess
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List

# ── try metpy ────────────────────────────────────────────────────────────────
try:
    import metpy.calc as mpcalc
    from metpy.units import units as munits
    METPY_OK = True
except ImportError:
    METPY_OK = False
    print("WARNING: metpy not installed. diff_LFC_PSFC will be NaN.")
    print("         Install with:  pip install metpy")

# =============================================================================
#  CONFIG — update these for your server paths
# =============================================================================
YEAR           = 2025                              # FIX 1: 2024 (same as verified extraction)
MONTH          = 1
DAY            = 1                              # FIX 1: 2 Jan 2024 (same as verified extraction)
CYCLE          = "12"                           # FIX 1: cycle 00Z
FORECAST_HOURS = [6, 12, 18, 24, 30, 36]       # FIX 1: no FH=21 for this case

LAT = 28.51052
LON = -97.5052

HRRR_BASE   = Path("/Storage03/nverma1/HRRR/2025")   # FIX 1: 2024 directory
OUTPUT_FILE  = "DerivedFeatures_20250101_12Z_v2.csv"     # FIX 1: correct filename

WGRIB2      = Path("/home/nverma1/bin/wgrib2")
VERBOSE     = False

YYYYMMDD    = f"{YEAR}{MONTH:02d}{DAY:02d}"

# =============================================================================
#  PHYSICAL CONSTANTS
# =============================================================================
R_DRY   = 287.0    # J kg-1 K-1  dry air gas constant
G       = 9.8      # m s-2       gravitational acceleration
CPD     = 1005.7   # J kg-1 K-1  specific heat dry air (constant pressure)
KAPPA   = 0.4      # von Kármán constant

# =============================================================================
#  FILE FINDER
# =============================================================================
def find_file(fh: int) -> Optional[Path]:
    fname = f"hrrr.t{CYCLE}z.wrfnatf{fh:02d}_{YYYYMMDD}.grib2"
    p = HRRR_BASE / fname
    if p.exists():
        return p
    print(f"  WARNING: file not found: {p}")
    return None

# =============================================================================
#  WGRIB2 HELPERS
# =============================================================================
def wgrib2_inventory(file_path: Path) -> List[str]:
    result = subprocess.run(
        [str(WGRIB2), str(file_path), "-s"],
        capture_output=True, text=True
    )
    return result.stdout.splitlines()

def wgrib2_extract(file_path: Path, record_num: int) -> Optional[float]:
    """Extract a single value nearest to LAT/LON."""
    result = subprocess.run(
        [str(WGRIB2), str(file_path),
         "-d", str(record_num),
         "-lon", str(LON), str(LAT)],
        capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        if "val=" in line:
            try:
                return float(line.split("val=")[1].split()[0])
            except Exception:
                return None
    return None

def hybrid_level_num(level_str: str) -> int:
    """Return hybrid level number (1-50) or -1 if not a hybrid level."""
    s = level_str.strip().lower()
    if s.endswith("hybrid level"):
        try:
            return int(s.split()[0])
        except ValueError:
            return -1
    return -1

# =============================================================================
#  BUILD EMPTY ROW
# =============================================================================
def build_empty_row() -> dict:
    row = {
        "cycle_time":    f"{YEAR}{MONTH:02d}{DAY:02d}T{CYCLE}Z",
        "forecast_hour": None,
        "latitude":      LAT,
        "longitude":     LON,
    }
    # Interleaved hybrid vars (levels 1-50): PRES TMP SPFH UGRD VGRD VVEL TKE
    for lvl in range(1, 51):
        for v in ["PRES", "TMP", "SPFH", "UGRD", "VGRD", "VVEL", "TKE"]:
            row[f"{v}_{lvl}"] = None
    # PMTF/PMTC interleaved (levels 1-20)
    for lvl in range(1, 21):
        row[f"PMTF_{lvl}"] = None
        row[f"PMTC_{lvl}"] = None
    # Single-value extracted features
    for col in [
        "DZDT", "RELV_2000", "RELV_1000",
        "LTNG", "PWAT", "AOTK", "RHPW",
        "TMP_2m", "POT_2m", "DPT_2m", "RH_2m", "SPFH_2m",
        "UGRD_10m", "VGRD_10m", "WIND_10m", "MAXUWU", "MAXVWV",
        "FRICV", "SHTFL", "LHTFL",
        "SBCAPE", "SBCIN",
        "HPBL", "PRES_SFC", "CNWAT", "SFCR", "VGTYP",
        "VUCSH", "VVCSH",
        "MSLMA",
        "SOILW", "MSTAV",
    ]:
        row[col] = None
    return row

# =============================================================================
#  EXTRACTION — same logic as v5 script (verified working)
# =============================================================================
def extract_one_file(file_path: Path, fh: int) -> dict:
    row = build_empty_row()
    row["forecast_hour"] = fh

    lines = wgrib2_inventory(file_path)

    for line in lines:
        parts = line.split(":")
        if len(parts) < 5:
            continue
        rec   = int(parts[0])
        var   = parts[3].strip()
        level = parts[4].strip()
        level_lo = level.lower()

        # ── A. HYBRID LEVEL VARIABLES ──────────────────────────────────────
        lvl = hybrid_level_num(level)
        if lvl != -1:
            if 1 <= lvl <= 50 and var in ("PRES", "TMP", "SPFH", "UGRD", "VGRD", "VVEL", "TKE"):
                col = f"{var}_{lvl}"
                if row[col] is None:
                    row[col] = wgrib2_extract(file_path, rec)
            if 1 <= lvl <= 20 and var in ("PMTF", "PMTC"):
                col = f"{var}_{lvl}"
                if row[col] is None:
                    row[col] = wgrib2_extract(file_path, rec)
            continue

        # ── B. DZDT — 0.5-0.8 sigma layer ──────────────────────────────────
        if var == "DZDT":
            if "sigma" in level_lo and "0.5" in level and "0.8" in level:
                if row["DZDT"] is None:
                    row["DZDT"] = wgrib2_extract(file_path, rec)

        # ── C. RELV — 2000m and 1000m above ground ─────────────────────────
        elif var == "RELV":
            if "2000" in level and "above ground" in level_lo:
                if row["RELV_2000"] is None:
                    row["RELV_2000"] = wgrib2_extract(file_path, rec)
            elif "1000" in level and "above ground" in level_lo:
                if row["RELV_1000"] is None:
                    row["RELV_1000"] = wgrib2_extract(file_path, rec)

        # ── D. SBCAPE — surface ONLY ────────────────────────────────────────
        elif var == "CAPE":
            if level.strip().lower() == "surface":
                if row["SBCAPE"] is None:
                    row["SBCAPE"] = wgrib2_extract(file_path, rec)

        # ── E. SBCIN — surface ONLY ─────────────────────────────────────────
        elif var == "CIN":
            if level.strip().lower() == "surface":
                if row["SBCIN"] is None:
                    row["SBCIN"] = wgrib2_extract(file_path, rec)

        # ── F. LTNG — entire atmosphere ─────────────────────────────────────
        elif var == "LTNG":
            if "entire atmosphere" in level_lo:
                if row["LTNG"] is None:
                    row["LTNG"] = wgrib2_extract(file_path, rec)

        # ── G. 2m variables ─────────────────────────────────────────────────
        elif var == "TMP" and "2 m above ground" in level_lo:
            if row["TMP_2m"] is None:
                row["TMP_2m"] = wgrib2_extract(file_path, rec)
        elif var == "POT" and "2 m above ground" in level_lo:
            if row["POT_2m"] is None:
                row["POT_2m"] = wgrib2_extract(file_path, rec)
        elif var == "DPT" and "2 m above ground" in level_lo:
            if row["DPT_2m"] is None:
                row["DPT_2m"] = wgrib2_extract(file_path, rec)
        elif var == "RH" and "2 m above ground" in level_lo:
            if row["RH_2m"] is None:
                row["RH_2m"] = wgrib2_extract(file_path, rec)
        elif var == "SPFH" and "2 m above ground" in level_lo:
            if row["SPFH_2m"] is None:
                row["SPFH_2m"] = wgrib2_extract(file_path, rec)

        # ── H. 10m variables ────────────────────────────────────────────────
        elif var == "UGRD" and "10 m above ground" in level_lo:
            if row["UGRD_10m"] is None:
                row["UGRD_10m"] = wgrib2_extract(file_path, rec)
        elif var == "VGRD" and "10 m above ground" in level_lo:
            if row["VGRD_10m"] is None:
                row["VGRD_10m"] = wgrib2_extract(file_path, rec)
        elif var == "WIND" and "10 m above ground" in level_lo:
            if row["WIND_10m"] is None:
                row["WIND_10m"] = wgrib2_extract(file_path, rec)
        elif var == "MAXUWU" and row["MAXUWU"] is None:
            row["MAXUWU"] = wgrib2_extract(file_path, rec)
        elif var == "MAXVWV" and row["MAXVWV"] is None:
            row["MAXVWV"] = wgrib2_extract(file_path, rec)

        # ── I. Surface variables ─────────────────────────────────────────────
        elif var == "FRICV" and "surface" in level_lo:
            if row["FRICV"] is None:
                row["FRICV"] = wgrib2_extract(file_path, rec)
        elif var == "SHTFL" and "surface" in level_lo:
            if row["SHTFL"] is None:
                row["SHTFL"] = wgrib2_extract(file_path, rec)
        elif var == "LHTFL" and "surface" in level_lo:
            if row["LHTFL"] is None:
                row["LHTFL"] = wgrib2_extract(file_path, rec)
        elif var == "HPBL" and "surface" in level_lo:
            if row["HPBL"] is None:
                row["HPBL"] = wgrib2_extract(file_path, rec)
        elif var == "PRES" and level.strip().lower() == "surface":
            if row["PRES_SFC"] is None:
                row["PRES_SFC"] = wgrib2_extract(file_path, rec)
        elif var == "CNWAT" and "surface" in level_lo:
            if row["CNWAT"] is None:
                row["CNWAT"] = wgrib2_extract(file_path, rec)
        elif var == "SFCR" and "surface" in level_lo:
            if row["SFCR"] is None:
                row["SFCR"] = wgrib2_extract(file_path, rec)
        elif var == "VGTYP" and "surface" in level_lo:
            if row["VGTYP"] is None:
                row["VGTYP"] = wgrib2_extract(file_path, rec)

        # ── J. Other single-value variables ──────────────────────────────────
        elif var == "PWAT" and "entire atmosphere" in level_lo:
            if row["PWAT"] is None:
                row["PWAT"] = wgrib2_extract(file_path, rec)
        elif var == "AOTK" and "entire atmosphere" in level_lo:
            if row["AOTK"] is None:
                row["AOTK"] = wgrib2_extract(file_path, rec)
        elif var == "RHPW" and "entire atmosphere" in level_lo:
            if row["RHPW"] is None:
                row["RHPW"] = wgrib2_extract(file_path, rec)
        elif var == "VUCSH" and "0-1000" in level:
            if row["VUCSH"] is None:
                row["VUCSH"] = wgrib2_extract(file_path, rec)
        elif var == "VVCSH" and "0-1000" in level:
            if row["VVCSH"] is None:
                row["VVCSH"] = wgrib2_extract(file_path, rec)
        elif var == "MSLMA":
            if row["MSLMA"] is None:
                row["MSLMA"] = wgrib2_extract(file_path, rec)
        elif var == "SOILW":
            if row["SOILW"] is None:
                row["SOILW"] = wgrib2_extract(file_path, rec)
        elif var == "MSTAV":
            if row["MSTAV"] is None:
                row["MSTAV"] = wgrib2_extract(file_path, rec)

    return row


# =============================================================================
#  DERIVED FEATURE CALCULATIONS (single-point, per Waylon PDF equations)
# =============================================================================

def calc_mixing_ratio(q: float) -> float:
    """r = q / (1 - q)  [kg/kg → kg/kg, used as g/g in equations]"""
    return q / (1.0 - q)


def calc_POT(T: float, P: float, P_sfc: float) -> float:
    """Potential temperature θ = T * (P_SFC / P)^0.2854  [K]"""
    return T * (P_sfc / P) ** 0.2854


def calc_TL(T_sfc: float, RH: float) -> float:
    """
    Temperature at Lifted Condensation Level (LCL) [K]
    TL = 1 / (1/(T_sfc - 55) - ln(RH/100)/2840) + 55
    T_sfc = TMP at 2m above ground [K]
    RH    = RH at 2m above ground [%]
    """
    rh_frac = max(RH / 100.0, 1e-10)
    return 1.0 / (1.0 / (T_sfc - 55.0) - math.log(rh_frac) / 2840.0) + 55.0


def calc_EPT(theta: float, r: float, TK: float, T_sfc_2m: float, RH_2m: float) -> float:
    """
    Equivalent Potential Temperature θE [K]
    θE = θ * exp( L*r / (cpd * TL) * (1 + k*r) )
    k = 0.81, cpd = 1005.7 J/(kg·K)
    L = 2.6975e6 - 2554.5*(TK - 273.15)  [J/kg]
    TL = temperature at LCL  [K]
    """
    L  = 2.6975e6 - 2554.5 * (TK - 273.15)
    TL = calc_TL(T_sfc_2m, RH_2m)
    k  = 0.81
    return theta * math.exp(L * r / (CPD * TL) * (1.0 + k * r))


def calc_RH_hybrid(q: float, P_Pa: float, T_K: float) -> float:
    """
    RH at hybrid level [%]
    Steps: r → e → e_s → RH = e/e_s * 100
    P in hPa = P_Pa / 100
    """
    r  = calc_mixing_ratio(q)
    p_hPa = P_Pa / 100.0
    e  = r * p_hPa / (0.622 + r)                          # vapor pressure [mb]
    TC = T_K - 273.15
    es = 6.112 * math.exp(17.67 * TC / (TC + 243.5))      # sat vapor pressure [mb]
    return (e / es) * 100.0


def calc_DPT_hybrid(RH_frac: float, T_K: float) -> float:
    """
    Dew Point Temperature at hybrid level [°C]
    DPT = A*(ln(RH) + B*T/(A+T)) / (B - ln(RH) - B*T/(A+T))
    A=237, B=7.5; RH as fraction [0,1]; T in °C
    Returns DPT in °C (Waylon's equation uses natural log)
    NOTE: convert to K for output if needed — kept in °C here per PDF
    """
    A = 237.0
    B = 7.5
    rh = max(RH_frac, 1e-10)
    T  = T_K - 273.15
    ln_rh  = math.log(rh)            # natural log of RH fraction
    BT_ApT = B * T / (A + T)
    numerator   = A * (ln_rh + BT_ApT)
    denominator = B - ln_rh - BT_ApT
    return numerator / denominator   # [°C]


def calc_DeltaZ(T1: float, T2: float, P1: float, P2: float) -> float:
    """
    Layer thickness ΔZ [m] using hypsometric equation
    ΔZ = R/g * (T1+T2)/2 * ln(P1/P2)
    subscript 1 = lower level (higher P), 2 = upper level (lower P)
    """
    return (R_DRY / G) * ((T1 + T2) / 2.0) * math.log(P1 / P2)


def calc_VTMPLR(T1: float, T2: float, q1: float, q2: float,
                P1: float, P2: float) -> float:
    """
    Virtual Temperature Lapse Rate [10^-3 K/m]
    Tv = (1 + 0.61*r) * T
    VTMPLR = (Tv2 - Tv1) / ΔZ   [then multiply by 1000 for 10^-3 units]
    subscript 1 = lower level
    """
    r1  = calc_mixing_ratio(q1)
    r2  = calc_mixing_ratio(q2)
    Tv1 = (1.0 + 0.61 * r1) * T1
    Tv2 = (1.0 + 0.61 * r2) * T2
    dZ  = calc_DeltaZ(T1, T2, P1, P2)
    if dZ == 0:
        return float("nan")
    return ((Tv2 - Tv1) / dZ) * 1e3   # [10^-3 K/m]


def calc_VS(U1: float, U2: float, V1: float, V2: float,
            T1: float, T2: float, P1: float, P2: float) -> float:
    """
    Vertical wind shear [10^-3 s^-1]
    VS = sqrt(|U1-U2|^2 + |V1-V2|^2) / ΔZ
    subscript 1 = lower level
    """
    dU  = abs(U1 - U2)
    dV  = abs(V1 - V2)
    dZ  = calc_DeltaZ(T1, T2, P1, P2)
    if dZ == 0:
        return float("nan")
    return (math.sqrt(dU**2 + dV**2) / dZ) * 1e3   # [10^-3 s^-1]


def calc_BowenRatio(SHTFL: float, LHTFL: float) -> float:
    """Bowen Ratio = SHTFL / LHTFL"""
    if LHTFL == 0:
        return float("nan")
    return SHTFL / LHTFL


def calc_ZL(HPBL: float, FRICV: float, POT_2m: float, SHTFL: float) -> float:
    """
    Atmospheric stability parameter -Z/L
    L = (u*)^3 / (κ * (g/θ0) * w'θ')
    where w'θ' = SHTFL [W/m^2] (kinematic heat flux proxy)
    -Z/L = -HPBL / L
    """
    u_star = FRICV
    theta0 = POT_2m
    w_theta = SHTFL
    if w_theta == 0 or u_star == 0:
        return float("nan")
    L = (u_star ** 3) / (KAPPA * (G / theta0) * w_theta)
    if L == 0:
        return float("nan")
    return -HPBL / L


def calc_diff_LFC_PSFC(row: dict) -> float:
    """
    diff_LFC_PSFC = LFC_pressure [hPa] - PRES_SFC [hPa]
    Uses MetPy metpy.calc.lfc with which='top'
    pressure array = [PRES_SFC/100, PRES_1/100, ..., PRES_50/100]
    temperature array = [TMP_2m-273.15, TMP_1-273.15, ..., TMP_50-273.15]
    dew_point array = [DPT_2m(°C), DPT_1(°C), ..., DPT_50(°C)]

    NOTE: In stable winter atmospheres (e.g. Jan 1), no LFC may exist.
    MetPy returns a NaN pressure in this case. diff_LFC_PSFC = NaN is the
    physically correct result: no LFC means no free convection.
    """
    if not METPY_OK:
        return float("nan")

    try:
        # Build pressure array [hPa], surface first (highest pressure)
        pres = [row["PRES_SFC"] / 100.0]
        tmp  = [row["TMP_2m"] - 273.15]
        # DPT_2m extracted in K → convert to °C
        dpt  = [row["DPT_2m"] - 273.15]

        for lvl in range(1, 51):
            pres.append(row[f"PRES_{lvl}"] / 100.0)
            tmp.append(row[f"TMP_{lvl}"] - 273.15)
            # DPT at hybrid levels: calc_DPT_hybrid returns °C directly
            dpt_val = row.get(f"DPT_{lvl}", float("nan"))
            dpt.append(dpt_val if dpt_val is not None else float("nan"))

        # Sort descending by pressure (surface = highest P first)
        triples = sorted(zip(pres, tmp, dpt), key=lambda x: -x[0])
        pres_s = [t[0] for t in triples]
        tmp_s  = [t[1] for t in triples]
        dpt_s  = [t[2] for t in triples]

        # Remove rows with any NaN value
        valid = [(p, t, d) for p, t, d in zip(pres_s, tmp_s, dpt_s)
                 if not any(math.isnan(v) if isinstance(v, float) else False
                            for v in (p, t, d))]
        if len(valid) < 3:
            return float("nan")

        pres_s = [v[0] for v in valid]
        tmp_s  = [v[1] for v in valid]
        dpt_s  = [v[2] for v in valid]

        import numpy as np
        p_arr  = np.array(pres_s) * munits.hPa
        t_arr  = np.array(tmp_s)  * munits.degC
        dp_arr = np.array(dpt_s)  * munits.degC

        lfc_p, _ = mpcalc.lfc(p_arr, t_arr, dp_arr, which='top')

        # MetPy returns NaN pressure when no LFC exists (stable atmosphere)
        lfc_mag = lfc_p.to(munits.hPa).magnitude
        if math.isnan(float(lfc_mag)):
            # Physically correct: no LFC in stable winter profile
            return float("nan")

        psfc_hPa = row["PRES_SFC"] / 100.0
        return float(lfc_mag - psfc_hPa)

    except Exception as e:
        print(f"    LFC calc error (FH={row.get('forecast_hour','?')}): {type(e).__name__}: {e}")
        return float("nan")


# =============================================================================
#  ADD ALL DERIVED FEATURES TO A ROW DICT
# =============================================================================
def compute_derived(row: dict) -> dict:
    """
    Given an extracted row dict, compute all center-point derived features
    and add them to the dict. Returns the enriched dict.
    """
    P_sfc  = row["PRES_SFC"]
    T_2m   = row["TMP_2m"]
    RH_2m  = row["RH_2m"]       # [%]

    # ── 1. POT at hybrid levels 1-50 ─────────────────────────────────────────
    for lvl in range(1, 51):
        T = row[f"TMP_{lvl}"]
        P = row[f"PRES_{lvl}"]
        row[f"POT_{lvl}"] = calc_POT(T, P, P_sfc) if (T and P and P_sfc) else float("nan")

    # ── 2. RH at hybrid levels 1-50 ──────────────────────────────────────────
    for lvl in range(1, 51):
        q = row[f"SPFH_{lvl}"]
        P = row[f"PRES_{lvl}"]
        T = row[f"TMP_{lvl}"]
        row[f"RH_{lvl}"] = calc_RH_hybrid(q, P, T) if (q and P and T) else float("nan")

    # ── 3. DPT at hybrid levels 1-50 ─────────────────────────────────────────
    #    DPT equation uses RH as fraction and T in °C; result in °C
    for lvl in range(1, 51):
        rh_pct = row[f"RH_{lvl}"]
        T      = row[f"TMP_{lvl}"]
        if rh_pct is not None and T is not None and not math.isnan(rh_pct):
            rh_frac = max(rh_pct / 100.0, 1e-10)
            row[f"DPT_{lvl}"] = calc_DPT_hybrid(rh_frac, T)
        else:
            row[f"DPT_{lvl}"] = float("nan")

    # ── 4. EPT at hybrid levels 1-50 ─────────────────────────────────────────
    for lvl in range(1, 51):
        theta = row.get(f"POT_{lvl}", float("nan"))
        q     = row[f"SPFH_{lvl}"]
        TK    = row[f"TMP_{lvl}"]
        if None not in (theta, q, TK) and not math.isnan(theta):
            r = calc_mixing_ratio(q)
            row[f"EPT_{lvl}"] = calc_EPT(theta, r, TK, T_2m, RH_2m)
        else:
            row[f"EPT_{lvl}"] = float("nan")

    # ── 5. EPT at 2m above ground ─────────────────────────────────────────────
    theta_2m = row["POT_2m"]
    q_2m     = row["SPFH_2m"]
    TK_2m    = row["TMP_2m"]
    if None not in (theta_2m, q_2m, TK_2m):
        r_2m = calc_mixing_ratio(q_2m)
        row["EPT_2m"] = calc_EPT(theta_2m, r_2m, TK_2m, T_2m, RH_2m)
    else:
        row["EPT_2m"] = float("nan")

    # ── 6. VTMPLR — virtual temp lapse rate for 49 layers ────────────────────
    for lvl in range(1, 50):
        T1 = row[f"TMP_{lvl}"];    T2 = row[f"TMP_{lvl+1}"]
        q1 = row[f"SPFH_{lvl}"];   q2 = row[f"SPFH_{lvl+1}"]
        P1 = row[f"PRES_{lvl}"];   P2 = row[f"PRES_{lvl+1}"]
        if None not in (T1, T2, q1, q2, P1, P2):
            row[f"VTMPLR_{lvl}{lvl+1}"] = calc_VTMPLR(T1, T2, q1, q2, P1, P2)
        else:
            row[f"VTMPLR_{lvl}{lvl+1}"] = float("nan")

    # ── 7. VS — vertical wind shear for 49 layers ────────────────────────────
    for lvl in range(1, 50):
        U1 = row[f"UGRD_{lvl}"];   U2 = row[f"UGRD_{lvl+1}"]
        V1 = row[f"VGRD_{lvl}"];   V2 = row[f"VGRD_{lvl+1}"]
        T1 = row[f"TMP_{lvl}"];    T2 = row[f"TMP_{lvl+1}"]
        P1 = row[f"PRES_{lvl}"];   P2 = row[f"PRES_{lvl+1}"]
        if None not in (U1, U2, V1, V2, T1, T2, P1, P2):
            row[f"VS_{lvl}{lvl+1}"] = calc_VS(U1, U2, V1, V2, T1, T2, P1, P2)
        else:
            row[f"VS_{lvl}{lvl+1}"] = float("nan")

    # ── 8. BowenRatio_Mean (single point = just the ratio) ───────────────────
    # FIX v2: use explicit None check — 'if (SHTFL and LHTFL)' falsely triggers
    # when either value is 0.0 (common in winter), causing spurious NaN.
    # LHTFL=0 → true division-by-zero → NaN (physically correct).
    # SHTFL=0 is valid (zero sensible flux) → ratio = 0, not NaN.
    SHTFL = row["SHTFL"]
    LHTFL = row["LHTFL"]
    if SHTFL is None or LHTFL is None:
        row["BowenRatio_Mean"] = float("nan")
    else:
        row["BowenRatio_Mean"] = calc_BowenRatio(SHTFL, LHTFL)

    # ── 9. ZL_MIN and ZL_MAX (same value at single point) ────────────────────
    HPBL  = row["HPBL"]
    FRICV = row["FRICV"]
    POT2m = row["POT_2m"]
    SHTFL = row["SHTFL"]
    if None not in (HPBL, FRICV, POT2m, SHTFL):
        zl = calc_ZL(HPBL, FRICV, POT2m, SHTFL)
    else:
        zl = float("nan")
    row["ZL_MIN"] = zl
    row["ZL_MAX"] = zl

    # ── 10. diff_LFC_PSFC — requires DPT hybrid levels (calc'd above) ────────
    row["diff_LFC_PSFC"] = calc_diff_LFC_PSFC(row)

    return row


# =============================================================================
#  VERIFICATION HELPER
# =============================================================================
def spot_check_derived(df: pd.DataFrame):
    """Print a summary check of derived features for FH=24."""
    fh24 = df[df["forecast_hour"] == 24]
    if fh24.empty:
        print("  No FH=24 row found for spot check.")
        return

    r = fh24.iloc[0]
    print("\n── DERIVED FEATURE SPOT-CHECK (FH=24) ──────────────────────────────")

    checks = [
        ("POT_1",         r.get("POT_1"),
         "θ = T1*(PRES_SFC/PRES_1)^0.2854"),
        ("EPT_2m",        r.get("EPT_2m"),       "θE at 2m"),
        ("RH_1",          r.get("RH_1"),          "RH at hybrid level 1 [%]"),
        ("DPT_1",         r.get("DPT_1"),         "DPT at hybrid level 1 [°C]"),
        ("VTMPLR_12",     r.get("VTMPLR_12"),     "Virtual Temp Lapse Rate lyr 1-2 [10-3 K/m]"),
        ("VS_12",         r.get("VS_12"),          "Vert Wind Shear lyr 1-2 [10-3 s-1]"),
        ("BowenRatio_Mean", r.get("BowenRatio_Mean"),
         f"SHTFL({r.get('SHTFL')})/LHTFL({r.get('LHTFL')}) — NaN if LHTFL=0"),
        ("ZL_MIN",        r.get("ZL_MIN"),         "-Z/L stability"),
        ("ZL_MAX",        r.get("ZL_MAX"),         "-Z/L (same as ZL_MIN at single point)"),
        ("diff_LFC_PSFC", r.get("diff_LFC_PSFC"),
         "LFC-PSFC [hPa] — NaN=physically correct for stable winter profile"),
    ]
    for name, val, note in checks:
        status = "✓" if val is not None and not (isinstance(val, float) and math.isnan(val)) else "✗ NaN"
        print(f"  {status} {name:20s} = {val}  ({note})")

    # Count NaNs across all derived columns
    derived_cols = [c for c in df.columns if c.startswith(("POT_", "EPT_", "RH_", "DPT_",
                                                             "VTMPLR_", "VS_",
                                                             "BowenRatio", "ZL_", "diff_LFC"))]
    nan_count = sum(1 for c in derived_cols if pd.isna(r.get(c)))
    print(f"\n  Total derived columns: {len(derived_cols)}")
    print(f"  NaN derived values in FH=24: {nan_count}")


# =============================================================================
#  MAIN
# =============================================================================
def main():
    print(f"\nwgrib2: {WGRIB2}")
    if not WGRIB2.exists():
        print(f"ERROR: wgrib2 not found at {WGRIB2}")
        sys.exit(1)

    all_rows = []

    for fh in tqdm(FORECAST_HOURS, desc="Forecast Hours"):
        tqdm.write(f"  Processing FH={fh:02d}: hrrr.t{CYCLE}z.wrfnatf{fh:02d}_{YYYYMMDD}.grib2")

        fp = find_file(fh)
        if fp is None:
            continue

        # Step 1: Extract raw HRRR variables
        row = extract_one_file(fp, fh)

        # Step 2: Compute all derived features
        row = compute_derived(row)

        all_rows.append(row)

    if not all_rows:
        print("ERROR: No rows extracted.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)

    # ── Column ordering: extracted then derived ───────────────────────────────
    # Extracted columns (from build_empty_row key order)
    extracted_base = list(build_empty_row().keys())

    # Derived columns in logical order
    derived_order = []
    for lvl in range(1, 51):
        derived_order.append(f"POT_{lvl}")
    derived_order.append("EPT_2m")
    for lvl in range(1, 51):
        derived_order.append(f"EPT_{lvl}")
    for lvl in range(1, 51):
        derived_order.append(f"RH_{lvl}")
    for lvl in range(1, 51):
        derived_order.append(f"DPT_{lvl}")
    for lvl in range(1, 50):
        derived_order.append(f"VTMPLR_{lvl}{lvl+1}")
    for lvl in range(1, 50):
        derived_order.append(f"VS_{lvl}{lvl+1}")
    derived_order += ["BowenRatio_Mean", "ZL_MIN", "ZL_MAX", "diff_LFC_PSFC"]

    # Reorder
    all_cols = extracted_base + [c for c in derived_order if c in df.columns]
    df = df[[c for c in all_cols if c in df.columns]]

    df.to_csv(OUTPUT_FILE, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"COMPLETE  →  {OUTPUT_FILE}")
    print(f"Rows: {len(df)}   |   Columns: {len(df.columns)}")
    print(f"  Extracted features: {len(extracted_base)}")
    print(f"  Derived features:   {len(derived_order)}")
    print(f"{'='*70}")

    # Check for NaN in extracted single-value columns
    single_val = ["DZDT","RELV_2000","RELV_1000","LTNG","PWAT","AOTK","RHPW",
                  "TMP_2m","POT_2m","DPT_2m","RH_2m","SPFH_2m",
                  "UGRD_10m","VGRD_10m","WIND_10m",
                  "FRICV","SHTFL","LHTFL","SBCAPE","SBCIN",
                  "HPBL","PRES_SFC","CNWAT","SFCR","VGTYP",
                  "VUCSH","VVCSH","MSLMA","SOILW","MSTAV"]

    print("Null check — extracted single-value columns:")
    has_null = False
    for col in single_val:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                fhs = df[df[col].isna()]["forecast_hour"].tolist()
                print(f"  ✗ {col:<15s} NaN in {n}/6 rows (FH: {fhs})")
                has_null = True
    if not has_null:
        print("  ✓ ALL extracted columns fully populated — zero NaN values")

    # ── BowenRatio NaN report ─────────────────────────────────────────────────
    print("BowenRatio and flux values per row:")
    for _, row_s in df.iterrows():
        fh   = row_s["forecast_hour"]
        sh   = row_s.get("SHTFL", float("nan"))
        lh   = row_s.get("LHTFL", float("nan"))
        br   = row_s.get("BowenRatio_Mean", float("nan"))
        note = "NaN: LHTFL=0 (correct for stable/night)" if (pd.isna(br) and lh == 0) else ""
        print(f"  FH={fh:2d}: SHTFL={sh:8.3f}  LHTFL={lh:8.3f}  BowenRatio={br}  {note}")

    # ── diff_LFC_PSFC NaN report ──────────────────────────────────────────────
    lfc_nan = df["diff_LFC_PSFC"].isna().sum() if "diff_LFC_PSFC" in df.columns else 0
    if lfc_nan > 0:
        print(f"\ndiff_LFC_PSFC NaN in {lfc_nan}/6 rows.")
        print("  This is PHYSICALLY CORRECT for stable winter profiles where no")
        print("  Level of Free Convection exists. For this date/location SBCAPE=0")
        print("  for all forecast hours → no free convection → LFC does not exist.")
        print("  Will inform Waylon: diff_LFC_PSFC=NaN is expected for this case.")

    spot_check_derived(df)
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()