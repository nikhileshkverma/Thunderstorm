"""
================================================================================
utils/hrrr_extractor.py — HRRR extraction + derived feature functions
================================================================================
Extracted from calc_derived_features_v2.py (fully verified against Waylon
DEGRIB spot-check values on 2024-01-02).

Called by 01_extract_features.py — do not run directly.
================================================================================
"""

import subprocess
import math
from pathlib import Path
from typing import Optional, List

# Import constants from config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import R_DRY, G, CPD, KAPPA, WGRIB2

# =============================================================================
#  WGRIB2 HELPERS
# =============================================================================
def wgrib2_inventory(file_path: Path) -> List[str]:
    result = subprocess.run(
        [str(WGRIB2), str(file_path), "-s"],
        capture_output=True, text=True
    )
    return result.stdout.splitlines()


def wgrib2_extract(file_path: Path, record_num: int,
                   lat: float, lon: float) -> Optional[float]:
    """Extract the nearest grid-point value to (lat, lon)."""
    result = subprocess.run(
        [str(WGRIB2), str(file_path), "-d", str(record_num),
         "-lon", str(lon), str(lat)],
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
#  EMPTY ROW BUILDER
# =============================================================================
def build_empty_row(cycle_str: str, fh: int, lat: float, lon: float) -> dict:
    """Build an ordered dict with all 426 extracted feature slots set to None."""
    row = {
        "cycle_time":    cycle_str,
        "forecast_hour": fh,
        "latitude":      lat,
        "longitude":     lon,
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
        "UGRD_10m", "VGRD_10m", "WIND_10m", "MAXUWU", "MAXVWV",
        "FRICV", "SHTFL", "LHTFL",
        "SBCAPE", "SBCIN",
        "HPBL", "PRES_SFC", "CNWAT", "SFCR", "VGTYP",
        "VUCSH", "VVCSH", "MSLMA", "SOILW", "MSTAV",
    ]:
        row[col] = None
    return row

# =============================================================================
#  EXTRACTION FROM ONE GRIB2 FILE
# =============================================================================
def extract_one_file(file_path: Path, cycle_str: str, cycle_code: str,
                     fh: int, lat: float, lon: float) -> dict:
    """Extract all HRRR variables from one grib2 file at the given lat/lon."""
    row = build_empty_row(cycle_str, fh, lat, lon)
    lines = wgrib2_inventory(file_path)

    for line in lines:
        parts = line.split(":")
        if len(parts) < 5:
            continue
        rec      = int(parts[0])
        var      = parts[3].strip()
        level    = parts[4].strip()
        level_lo = level.lower()

        def get():
            return wgrib2_extract(file_path, rec, lat, lon)

        # ── Hybrid level variables ────────────────────────────────────────────
        lvl = hybrid_level_num(level)
        if lvl != -1:
            if 1 <= lvl <= 50 and var in ("PRES","TMP","SPFH","UGRD","VGRD","VVEL","TKE"):
                col = f"{var}_{lvl}"
                if row[col] is None:
                    row[col] = get()
            if 1 <= lvl <= 20 and var in ("PMTF", "PMTC"):
                col = f"{var}_{lvl}"
                if row[col] is None:
                    row[col] = get()
            continue

        # ── DZDT — 0.5-0.8 sigma layer ───────────────────────────────────────
        if var == "DZDT":
            if "sigma" in level_lo and "0.5" in level and "0.8" in level:
                if row["DZDT"] is None:
                    row["DZDT"] = get()

        # ── RELV — 2000m and 1000m above ground ──────────────────────────────
        elif var == "RELV":
            if "2000" in level and "above ground" in level_lo:
                if row["RELV_2000"] is None: row["RELV_2000"] = get()
            elif "1000" in level and "above ground" in level_lo:
                if row["RELV_1000"] is None: row["RELV_1000"] = get()

        # ── SBCAPE / SBCIN — surface ONLY ────────────────────────────────────
        elif var == "CAPE" and level.strip().lower() == "surface":
            if row["SBCAPE"] is None: row["SBCAPE"] = get()
        elif var == "CIN" and level.strip().lower() == "surface":
            if row["SBCIN"] is None: row["SBCIN"] = get()

        # ── LTNG — entire atmosphere ──────────────────────────────────────────
        elif var == "LTNG" and "entire atmosphere" in level_lo:
            if row["LTNG"] is None: row["LTNG"] = get()

        # ── 2m variables ─────────────────────────────────────────────────────
        elif var == "TMP"  and "2 m above ground" in level_lo:
            if row["TMP_2m"]  is None: row["TMP_2m"]  = get()
        elif var == "POT"  and "2 m above ground" in level_lo:
            if row["POT_2m"]  is None: row["POT_2m"]  = get()
        elif var == "DPT"  and "2 m above ground" in level_lo:
            if row["DPT_2m"]  is None: row["DPT_2m"]  = get()
        elif var == "RH"   and "2 m above ground" in level_lo:
            if row["RH_2m"]   is None: row["RH_2m"]   = get()
        elif var == "SPFH" and "2 m above ground" in level_lo:
            if row["SPFH_2m"] is None: row["SPFH_2m"] = get()

        # ── 10m variables ─────────────────────────────────────────────────────
        elif var == "UGRD" and "10 m above ground" in level_lo:
            if row["UGRD_10m"] is None: row["UGRD_10m"] = get()
        elif var == "VGRD" and "10 m above ground" in level_lo:
            if row["VGRD_10m"] is None: row["VGRD_10m"] = get()
        elif var == "WIND" and "10 m above ground" in level_lo:
            if row["WIND_10m"] is None: row["WIND_10m"] = get()
        elif var == "MAXUWU" and row["MAXUWU"] is None:
            row["MAXUWU"] = get()
        elif var == "MAXVWV" and row["MAXVWV"] is None:
            row["MAXVWV"] = get()

        # ── Surface variables ─────────────────────────────────────────────────
        elif var == "FRICV" and "surface" in level_lo:
            if row["FRICV"] is None: row["FRICV"] = get()
        elif var == "SHTFL" and "surface" in level_lo:
            if row["SHTFL"] is None: row["SHTFL"] = get()
        elif var == "LHTFL" and "surface" in level_lo:
            if row["LHTFL"] is None: row["LHTFL"] = get()
        elif var == "HPBL"  and "surface" in level_lo:
            if row["HPBL"]  is None: row["HPBL"]  = get()
        elif var == "PRES"  and level.strip().lower() == "surface":
            if row["PRES_SFC"] is None: row["PRES_SFC"] = get()
        elif var == "CNWAT" and "surface" in level_lo:
            if row["CNWAT"] is None: row["CNWAT"] = get()
        elif var == "SFCR"  and "surface" in level_lo:
            if row["SFCR"]  is None: row["SFCR"]  = get()
        elif var == "VGTYP" and "surface" in level_lo:
            if row["VGTYP"] is None: row["VGTYP"] = get()

        # ── Whole-atmosphere / other ──────────────────────────────────────────
        elif var == "PWAT" and "entire atmosphere" in level_lo:
            if row["PWAT"] is None: row["PWAT"] = get()
        elif var == "AOTK" and "entire atmosphere" in level_lo:
            if row["AOTK"] is None: row["AOTK"] = get()
        elif var == "RHPW" and "entire atmosphere" in level_lo:
            if row["RHPW"] is None: row["RHPW"] = get()
        elif var == "VUCSH" and "0-1000" in level:
            if row["VUCSH"] is None: row["VUCSH"] = get()
        elif var == "VVCSH" and "0-1000" in level:
            if row["VVCSH"] is None: row["VVCSH"] = get()
        elif var == "MSLMA":
            if row["MSLMA"] is None: row["MSLMA"] = get()
        elif var == "SOILW":
            if row["SOILW"] is None: row["SOILW"] = get()
        elif var == "MSTAV":
            if row["MSTAV"] is None: row["MSTAV"] = get()

    return row

# =============================================================================
#  DERIVED FEATURE MATH  (all verified against Waylon PDF equations)
# =============================================================================
def _mr(q: float) -> float:
    """Mixing ratio r = q/(1-q)  [kg/kg]"""
    return q / (1.0 - q)

def calc_POT(T: float, P: float, P_sfc: float) -> float:
    """θ = T*(PSFC/P)^0.2854  [K]"""
    return T * (P_sfc / P) ** 0.2854

def _calc_TL(T_sfc: float, RH_pct: float) -> float:
    """Temperature at LCL [K].  T_sfc=TMP_2m [K], RH_pct in %."""
    rh = max(RH_pct / 100.0, 1e-10)
    return 1.0 / (1.0 / (T_sfc - 55.0) - math.log(rh) / 2840.0) + 55.0

def calc_EPT(theta: float, q: float, TK: float,
             T_2m: float, RH_2m: float) -> float:
    """θE = θ * exp(L*r/(cpd*TL) * (1+0.81*r))  [K]"""
    r  = _mr(q)
    L  = 2.6975e6 - 2554.5 * (TK - 273.15)
    TL = _calc_TL(T_2m, RH_2m)
    return theta * math.exp(L * r / (CPD * TL) * (1.0 + 0.81 * r))

def calc_RH_hybrid(q: float, P_Pa: float, T_K: float) -> float:
    """RH at hybrid level [%] — 4-step: r→e→es→RH"""
    r     = _mr(q)
    p_hPa = P_Pa / 100.0
    e     = r * p_hPa / (0.622 + r)
    TC    = T_K - 273.15
    es    = 6.112 * math.exp(17.67 * TC / (TC + 243.5))
    return (e / es) * 100.0

def calc_DPT_hybrid(RH_frac: float, T_K: float) -> float:
    """DPT [°C] via Magnus formula. RH_frac in [0,1]."""
    A = 237.0; B = 7.5
    rh   = max(RH_frac, 1e-10)
    T    = T_K - 273.15
    ln_r = math.log(rh)
    BT   = B * T / (A + T)
    return A * (ln_r + BT) / (B - ln_r - BT)

def _dZ(T1: float, T2: float, P1: float, P2: float) -> float:
    """Layer thickness ΔZ [m] (hypsometric). subscript 1=lower level."""
    return (R_DRY / G) * ((T1 + T2) / 2.0) * math.log(P1 / P2)

def calc_VTMPLR(T1: float, T2: float, q1: float, q2: float,
                P1: float, P2: float) -> float:
    """Virtual temp lapse rate [10^-3 K/m]. subscript 1=lower level."""
    Tv1 = (1.0 + 0.61 * _mr(q1)) * T1
    Tv2 = (1.0 + 0.61 * _mr(q2)) * T2
    dz  = _dZ(T1, T2, P1, P2)
    return ((Tv2 - Tv1) / dz) * 1e3 if dz != 0 else float("nan")

def calc_VS(U1: float, U2: float, V1: float, V2: float,
            T1: float, T2: float, P1: float, P2: float) -> float:
    """Vertical wind shear [10^-3 s^-1]. subscript 1=lower level."""
    dz = _dZ(T1, T2, P1, P2)
    return (math.sqrt((U1-U2)**2 + (V1-V2)**2) / dz) * 1e3 if dz != 0 else float("nan")

def calc_BowenRatio(SHTFL: float, LHTFL: float) -> float:
    """Bowen Ratio = SHTFL/LHTFL. NaN if LHTFL=0."""
    return SHTFL / LHTFL if LHTFL != 0 else float("nan")

def calc_ZL(HPBL: float, FRICV: float, POT_2m: float, SHTFL: float) -> float:
    """-Z/L atmospheric stability parameter."""
    if SHTFL == 0 or FRICV == 0:
        return float("nan")
    L = (FRICV**3) / (KAPPA * (G / POT_2m) * SHTFL)
    return (-HPBL / L) if L != 0 else float("nan")

def calc_diff_LFC_PSFC(row: dict) -> float:
    """
    diff_LFC_PSFC = LFC [hPa] - PSFC [hPa]
    Uses MetPy metpy.calc.lfc(which='top').
    Returns NaN when no LFC exists (stable atmosphere, SBCAPE=0).
    """
    try:
        import numpy as np
        import metpy.calc as mpcalc
        from metpy.units import units as munits

        pres = [row["PRES_SFC"] / 100.0]
        tmp  = [row["TMP_2m"] - 273.15]
        dpt  = [row["DPT_2m"] - 273.15]

        for lvl in range(1, 51):
            pres.append(row[f"PRES_{lvl}"] / 100.0)
            tmp.append(row[f"TMP_{lvl}"] - 273.15)
            dv = row.get(f"DPT_{lvl}", float("nan"))
            dpt.append(dv if dv is not None else float("nan"))

        triples = sorted(zip(pres, tmp, dpt), key=lambda x: -x[0])
        valid   = [(p, t, d) for p, t, d in triples
                   if not any(math.isnan(v) if isinstance(v, float) else False
                              for v in (p, t, d))]
        if len(valid) < 3:
            return float("nan")

        p_arr  = np.array([v[0] for v in valid]) * munits.hPa
        t_arr  = np.array([v[1] for v in valid]) * munits.degC
        dp_arr = np.array([v[2] for v in valid]) * munits.degC

        lfc_p, _ = mpcalc.lfc(p_arr, t_arr, dp_arr, which="top")
        lfc_mag  = lfc_p.to(munits.hPa).magnitude
        if math.isnan(float(lfc_mag)):
            return float("nan")
        return float(lfc_mag - row["PRES_SFC"] / 100.0)

    except Exception:
        return float("nan")

# =============================================================================
#  COMPUTE ALL DERIVED FEATURES FOR ONE ROW DICT
# =============================================================================
def compute_derived(row: dict) -> dict:
    """Add all center-point derived features to row dict. Returns enriched row."""
    P_sfc = row["PRES_SFC"]
    T_2m  = row["TMP_2m"]
    RH_2m = row["RH_2m"]

    # 1. POT_1..50
    for lvl in range(1, 51):
        T = row[f"TMP_{lvl}"]; P = row[f"PRES_{lvl}"]
        row[f"POT_{lvl}"] = (calc_POT(T, P, P_sfc)
                              if (T and P and P_sfc) else float("nan"))

    # 2. RH_1..50
    for lvl in range(1, 51):
        q = row[f"SPFH_{lvl}"]; P = row[f"PRES_{lvl}"]; T = row[f"TMP_{lvl}"]
        row[f"RH_{lvl}"] = (calc_RH_hybrid(q, P, T)
                             if (q and P and T) else float("nan"))

    # 3. DPT_1..50  [°C]
    for lvl in range(1, 51):
        rh = row[f"RH_{lvl}"]; T = row[f"TMP_{lvl}"]
        if rh is not None and T is not None and not math.isnan(rh):
            row[f"DPT_{lvl}"] = calc_DPT_hybrid(max(rh / 100.0, 1e-10), T)
        else:
            row[f"DPT_{lvl}"] = float("nan")

    # 4. EPT_1..50
    for lvl in range(1, 51):
        theta = row.get(f"POT_{lvl}", float("nan"))
        q = row[f"SPFH_{lvl}"]; TK = row[f"TMP_{lvl}"]
        if None not in (theta, q, TK) and not math.isnan(theta):
            row[f"EPT_{lvl}"] = calc_EPT(theta, q, TK, T_2m, RH_2m)
        else:
            row[f"EPT_{lvl}"] = float("nan")

    # 5. EPT_2m
    th2m = row["POT_2m"]; q2m = row["SPFH_2m"]; TK2m = row["TMP_2m"]
    row["EPT_2m"] = (calc_EPT(th2m, q2m, TK2m, T_2m, RH_2m)
                     if None not in (th2m, q2m, TK2m) else float("nan"))

    # 6. VTMPLR_12 .. VTMPLR_4950  (49 layers)
    for lvl in range(1, 50):
        T1 = row[f"TMP_{lvl}"];   T2 = row[f"TMP_{lvl+1}"]
        q1 = row[f"SPFH_{lvl}"];  q2 = row[f"SPFH_{lvl+1}"]
        P1 = row[f"PRES_{lvl}"];  P2 = row[f"PRES_{lvl+1}"]
        row[f"VTMPLR_{lvl}{lvl+1}"] = (
            calc_VTMPLR(T1, T2, q1, q2, P1, P2)
            if None not in (T1, T2, q1, q2, P1, P2) else float("nan"))

    # 7. VS_12 .. VS_4950  (49 layers)
    for lvl in range(1, 50):
        U1 = row[f"UGRD_{lvl}"];  U2 = row[f"UGRD_{lvl+1}"]
        V1 = row[f"VGRD_{lvl}"];  V2 = row[f"VGRD_{lvl+1}"]
        T1 = row[f"TMP_{lvl}"];   T2 = row[f"TMP_{lvl+1}"]
        P1 = row[f"PRES_{lvl}"];  P2 = row[f"PRES_{lvl+1}"]
        row[f"VS_{lvl}{lvl+1}"] = (
            calc_VS(U1, U2, V1, V2, T1, T2, P1, P2)
            if None not in (U1, U2, V1, V2, T1, T2, P1, P2) else float("nan"))

    # 8. BowenRatio_Mean
    SH = row["SHTFL"]; LH = row["LHTFL"]
    row["BowenRatio_Mean"] = (calc_BowenRatio(SH, LH)
                               if (SH is not None and LH is not None)
                               else float("nan"))

    # 9. ZL_MIN, ZL_MAX  (same value at single center point)
    HPBL  = row["HPBL"];  FRICV = row["FRICV"]
    POT2m = row["POT_2m"]; SHTFL = row["SHTFL"]
    zl = (calc_ZL(HPBL, FRICV, POT2m, SHTFL)
          if None not in (HPBL, FRICV, POT2m, SHTFL) else float("nan"))
    row["ZL_MIN"] = zl
    row["ZL_MAX"] = zl

    # 10. diff_LFC_PSFC
    row["diff_LFC_PSFC"] = calc_diff_LFC_PSFC(row)

    return row
