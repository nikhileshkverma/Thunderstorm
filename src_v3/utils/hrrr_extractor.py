"""
================================================================================
utils/hrrr_extractor.py — HRRR extraction + ALL derived features  (v4 FAST)
================================================================================
v4 PERFORMANCE OVERHAUL:
  ONE wgrib2 call per file (was 382 calls per file)
  All values extracted in a single pass: wgrib2 file -s -lon LON LAT
  This parses the val= token directly from the inventory line
  Expected speedup: 100-200x vs v3

v3 equation corrections still applied (Waylon March 25 2026):
  A1. POT:     constant P0=100,000 Pa
  A2. DPT:     new kPa vapor-pressure equation
  A3. VTMPLR:  Tv-based ΔZ
  A4. VS_LYR:  Tv-based ΔZ
  A5. -Z/L:    includes air density ρ and cpd
  B1. T_LCL:   exposed as output feature
  B2. LCL:     height of LCL [m]
  C.  diff_LFC_PSFC removed, SNOWC_MEAN removed
================================================================================
"""

import subprocess
import math
from pathlib import Path
from typing import Optional, List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_config

_cfg_cache = None
def _cfg():
    global _cfg_cache
    if _cfg_cache is None:
        _cfg_cache = load_config()
    return _cfg_cache

# =============================================================================
#  CORE: ONE wgrib2 CALL PER FILE — parse val= from inventory line
# =============================================================================
def extract_all_values_single_pass(file_path: Path,
                                   lat: float, lon: float) -> Dict[str, float]:
    """
    Call wgrib2 ONCE with -s -lon LON LAT.
    Returns dict: {record_key: value}
    where record_key = "VAR:LEVEL" string for matching.

    wgrib2 output format with -lon:
      1:0:d=....:VAR:LEVEL:...:lon=X,lat=Y,val=VALUE
    """
    wgrib2 = str(_cfg()["paths"]["wgrib2"])
    result = subprocess.run(
        [wgrib2, str(file_path), "-s", "-lon", str(lon), str(lat)],
        capture_output=True, text=True
    )

    values = {}
    for line in result.stdout.splitlines():
        if "val=" not in line:
            continue
        parts = line.split(":")
        if len(parts) < 5:
            continue
        var   = parts[3].strip()
        level = parts[4].strip()
        try:
            val = float(line.split("val=")[-1].strip().split()[0])
        except (ValueError, IndexError):
            continue
        # Store by "VAR|||LEVEL" — triple pipe to avoid collision with colons in level strings
        key = f"{var}|||{level}"
        if key not in values:           # keep first occurrence only
            values[key] = val
    return values

def _get(values: dict, var: str, level: str) -> Optional[float]:
    """Look up a value from the single-pass dict."""
    return values.get(f"{var}|||{level}")

def _get_any_level(values: dict, var: str) -> Optional[float]:
    """Return first value for this var regardless of level."""
    for k, v in values.items():
        if k.startswith(f"{var}|||"):
            return v
    return None

def hybrid_level_num(level_str: str) -> int:
    s = level_str.strip().lower()
    if s.endswith("hybrid level"):
        try:
            return int(s.split()[0])
        except ValueError:
            return -1
    return -1

# =============================================================================
#  EMPTY ROW — exact Waylon naming (v3 column names kept)
# =============================================================================
def build_empty_row(cycle_str: str, fh: int, lat: float, lon: float) -> dict:
    row = {
        "cycle_time":    cycle_str,
        "forecast_hour": fh,
        "latitude":      lat,
        "longitude":     lon,
    }
    for lvl in range(1, 51):
        for v in ["PRES","TMP","SPFH","UGRD","VGRD","VVEL","TKE"]:
            row[f"{v}_{lvl}"] = None
    for lvl in range(1, 21):
        row[f"PMTF_{lvl}"] = None
        row[f"PMTC_{lvl}"] = None
    for col in [
        "DZDT",
        "RELV_2000", "RELV_1000",
        "LTNG", "PWAT", "AOTK", "RHPW",
        "TMP_2m", "POT_2m", "DPT_2m", "RH_2m", "SPFH_2m",
        "UGRD_10m", "VGRD_10m", "WIND_10m",
        "MAXUWU_10m", "MAXVWV_10m",
        "FRICV", "SHTFL", "LHTFL",
        "SBCAPE", "SBCIN",
        "HPBL", "PRES_SFC",
        "CNWAT", "SFCR", "VGTYP",
        "VUCSH_1000m", "VVCSH_1000m",
        "MSLMA", "SOILW", "MSTAV",
    ]:
        row[col] = None
    return row

# =============================================================================
#  EXTRACTION — ONE call, fill all 426 columns
# =============================================================================
def extract_one_file(file_path: Path, cycle_str: str, fh: int,
                     lat: float, lon: float) -> dict:
    """Extract all HRRR variables in a SINGLE wgrib2 call."""
    row = build_empty_row(cycle_str, fh, lat, lon)

    # ONE call per file
    vals = extract_all_values_single_pass(file_path, lat, lon)

    for key, val in vals.items():
        var, level = key.split("|||", 1)
        level_lo   = level.lower()

        # ── Hybrid level variables ────────────────────────────────────────────
        lvl = hybrid_level_num(level)
        if lvl != -1:
            if 1 <= lvl <= 50 and var in ("PRES","TMP","SPFH","UGRD","VGRD","VVEL","TKE"):
                col = f"{var}_{lvl}"
                if row.get(col) is None:
                    row[col] = val
            if 1 <= lvl <= 20 and var in ("PMTF","PMTC"):
                col = f"{var}_{lvl}"
                if row.get(col) is None:
                    row[col] = val
            continue

        # ── DZDT ─────────────────────────────────────────────────────────────
        if var == "DZDT" and "sigma" in level_lo and "0.5" in level and "0.8" in level:
            if row["DZDT"] is None: row["DZDT"] = val

        # ── RELV ─────────────────────────────────────────────────────────────
        elif var == "RELV":
            if "2000" in level and "above ground" in level_lo:
                if row["RELV_2000"] is None: row["RELV_2000"] = val
            elif "1000" in level and "above ground" in level_lo:
                if row["RELV_1000"] is None: row["RELV_1000"] = val

        # ── SBCAPE / SBCIN — surface ONLY ────────────────────────────────────
        elif var == "CAPE" and level.strip().lower() == "surface":
            if row["SBCAPE"] is None: row["SBCAPE"] = val
        elif var == "CIN" and level.strip().lower() == "surface":
            if row["SBCIN"] is None: row["SBCIN"] = val

        # ── LTNG ─────────────────────────────────────────────────────────────
        elif var == "LTNG" and "entire atmosphere" in level_lo:
            if row["LTNG"] is None: row["LTNG"] = val

        # ── 2m variables ─────────────────────────────────────────────────────
        elif var == "TMP"  and "2 m above ground" in level_lo:
            if row["TMP_2m"]  is None: row["TMP_2m"]  = val
        elif var == "POT"  and "2 m above ground" in level_lo:
            if row["POT_2m"]  is None: row["POT_2m"]  = val
        elif var == "DPT"  and "2 m above ground" in level_lo:
            if row["DPT_2m"]  is None: row["DPT_2m"]  = val
        elif var == "RH"   and "2 m above ground" in level_lo:
            if row["RH_2m"]   is None: row["RH_2m"]   = val
        elif var == "SPFH" and "2 m above ground" in level_lo:
            if row["SPFH_2m"] is None: row["SPFH_2m"] = val

        # ── 10m variables ─────────────────────────────────────────────────────
        elif var == "UGRD" and "10 m above ground" in level_lo:
            if row["UGRD_10m"] is None: row["UGRD_10m"] = val
        elif var == "VGRD" and "10 m above ground" in level_lo:
            if row["VGRD_10m"] is None: row["VGRD_10m"] = val
        elif var == "WIND" and "10 m above ground" in level_lo:
            if row["WIND_10m"] is None: row["WIND_10m"] = val
        elif var == "MAXUWU":
            if row["MAXUWU_10m"] is None: row["MAXUWU_10m"] = val
        elif var == "MAXVWV":
            if row["MAXVWV_10m"] is None: row["MAXVWV_10m"] = val

        # ── Surface variables ─────────────────────────────────────────────────
        elif var == "FRICV" and "surface" in level_lo:
            if row["FRICV"] is None: row["FRICV"] = val
        elif var == "SHTFL" and "surface" in level_lo:
            if row["SHTFL"] is None: row["SHTFL"] = val
        elif var == "LHTFL" and "surface" in level_lo:
            if row["LHTFL"] is None: row["LHTFL"] = val
        elif var == "HPBL"  and "surface" in level_lo:
            if row["HPBL"]  is None: row["HPBL"]  = val
        elif var == "PRES"  and level.strip().lower() == "surface":
            if row["PRES_SFC"] is None: row["PRES_SFC"] = val
        elif var == "CNWAT" and "surface" in level_lo:
            if row["CNWAT"] is None: row["CNWAT"] = val
        elif var == "SFCR"  and "surface" in level_lo:
            if row["SFCR"]  is None: row["SFCR"]  = val
        elif var == "VGTYP" and "surface" in level_lo:
            if row["VGTYP"] is None: row["VGTYP"] = val

        # ── Atmosphere/column ─────────────────────────────────────────────────
        elif var == "PWAT" and "entire atmosphere" in level_lo:
            if row["PWAT"] is None: row["PWAT"] = val
        elif var == "AOTK" and "entire atmosphere" in level_lo:
            if row["AOTK"] is None: row["AOTK"] = val
        elif var == "RHPW" and "entire atmosphere" in level_lo:
            if row["RHPW"] is None: row["RHPW"] = val

        # ── Wind shear ────────────────────────────────────────────────────────
        elif var == "VUCSH" and "0-1000" in level:
            if row["VUCSH_1000m"] is None: row["VUCSH_1000m"] = val
        elif var == "VVCSH" and "0-1000" in level:
            if row["VVCSH_1000m"] is None: row["VVCSH_1000m"] = val

        # ── Others ────────────────────────────────────────────────────────────
        elif var == "MSLMA":
            if row["MSLMA"] is None: row["MSLMA"] = val
        elif var == "SOILW":
            if row["SOILW"] is None: row["SOILW"] = val
        elif var == "MSTAV":
            if row["MSTAV"] is None: row["MSTAV"] = val

    return row


# =============================================================================
#  DERIVED FEATURES (v3 equations — all Waylon corrections applied)
# =============================================================================
def _safe(v) -> bool:
    if v is None: return False
    try: return not math.isnan(float(v))
    except: return False

def _mr(q): return q / (1.0 - q)
def _tv(T, q): return T * (1.0 + 0.61 * _mr(q))

def calc_POT(T, P):
    """θ = T*(100000/P)^0.2854  — P0=100,000 Pa constant (Waylon correction)"""
    return T * (100000.0 / P) ** 0.2854

def calc_T_LCL(T_sfc, RH_pct):
    """T_LCL [K] = 1/(1/(T_sfc-55) - ln(RH/100)/2840) + 55"""
    rh = max(RH_pct / 100.0, 1e-10)
    return 1.0 / (1.0/(T_sfc - 55.0) - math.log(rh)/2840.0) + 55.0

def calc_LCL(T_sfc, T_LCL):
    """LCL height [m] = (T_sfc - T_LCL) / 0.00982  (use T not Tv per Waylon)"""
    return (T_sfc - T_LCL) / 0.00982

def calc_EPT(theta, q, TK, T_2m, RH_2m):
    """θE [K]"""
    r  = _mr(q)
    L  = 2.6975e6 - 2554.5*(TK - 273.15)
    TL = calc_T_LCL(T_2m, RH_2m)
    return theta * math.exp(L*r/(1005.7*TL) * (1.0 + 0.81*r))

def calc_RH_hybrid(q, P_Pa, T_K):
    """RH [%] at hybrid level"""
    r    = _mr(q)
    p_hPa = P_Pa / 100.0
    e    = r * p_hPa / (0.622 + r)
    TC   = T_K - 273.15
    es   = 6.112 * math.exp(17.67*TC/(TC + 243.5))
    return (e/es)*100.0

def calc_DPT_hybrid(q, P_Pa):
    """DPT [°C] — new kPa equation (Waylon correction)"""
    r     = _mr(q)
    p_hPa = P_Pa / 100.0
    e_hPa = r * p_hPa / (0.622 + r)
    e_kPa = e_hPa / 10.0
    if e_kPa <= 0: return float("nan")
    ln_e  = math.log(e_kPa / 0.6112)
    denom = 17.67 - ln_e
    if abs(denom) < 1e-10: return float("nan")
    return 243.5 * ln_e / denom

def _dZ_tv(Tv1, Tv2, P1, P2):
    """ΔZ [m] using virtual temperature (Waylon correction)"""
    return (287.0/9.8) * ((Tv1+Tv2)/2.0) * math.log(P1/P2)

def calc_VTMPLR(T1, T2, q1, q2, P1, P2):
    """Virtual temp lapse rate [10^-3 K/m] — Tv-based ΔZ"""
    Tv1 = _tv(T1,q1); Tv2 = _tv(T2,q2)
    dz  = _dZ_tv(Tv1, Tv2, P1, P2)
    return ((Tv2-Tv1)/dz)*1e3 if dz != 0 else float("nan")

def calc_VS(U1, U2, V1, V2, T1, T2, q1, q2, P1, P2):
    """Vertical wind shear [10^-3 s^-1] — Tv-based ΔZ"""
    Tv1 = _tv(T1,q1); Tv2 = _tv(T2,q2)
    dz  = _dZ_tv(Tv1, Tv2, P1, P2)
    return (math.sqrt((U1-U2)**2+(V1-V2)**2)/dz)*1e3 if dz != 0 else float("nan")

def calc_BowenRatio(SHTFL, LHTFL):
    return SHTFL/LHTFL if LHTFL != 0 else float("nan")

def calc_ZL(HPBL, FRICV, POT_2m, SHTFL, PRES_SFC):
    """-Z/L with air density ρ and cpd (Waylon correction)"""
    if not all(_safe(x) for x in (HPBL,FRICV,POT_2m,SHTFL,PRES_SFC)): return float("nan")
    if PRES_SFC<=0 or FRICV==0 or SHTFL==0 or POT_2m<=0: return float("nan")
    rho   = PRES_SFC / (287.0 * POT_2m)
    H     = SHTFL / (rho * 1005.7)
    denom = 0.4 * 9.8 * H
    if abs(denom) < 1e-15: return float("nan")
    L = -(FRICV**3) * POT_2m / denom
    return (-HPBL/L) if L != 0 else float("nan")

def compute_derived(row: dict) -> dict:
    """Compute all derived features. All Waylon v3 corrections applied."""
    T_2m = row.get("TMP_2m"); RH_2m = row.get("RH_2m")

    # POT_1..50 (P0=100000 constant)
    for lvl in range(1,51):
        T=row.get(f"TMP_{lvl}"); P=row.get(f"PRES_{lvl}")
        row[f"POT_{lvl}"] = calc_POT(T,P) if (_safe(T) and _safe(P)) else float("nan")

    # RH_1..50
    for lvl in range(1,51):
        q=row.get(f"SPFH_{lvl}"); P=row.get(f"PRES_{lvl}"); T=row.get(f"TMP_{lvl}")
        row[f"RH_{lvl}"] = calc_RH_hybrid(q,P,T) if all(_safe(x) for x in (q,P,T)) else float("nan")

    # DPT_1..50 (new kPa equation)
    for lvl in range(1,51):
        q=row.get(f"SPFH_{lvl}"); P=row.get(f"PRES_{lvl}")
        row[f"DPT_{lvl}"] = calc_DPT_hybrid(q,P) if (_safe(q) and _safe(P)) else float("nan")

    # T_LCL and LCL (new features)
    T_LCL = float("nan")
    if _safe(T_2m) and _safe(RH_2m):
        T_LCL = calc_T_LCL(T_2m, RH_2m)
    row["T_LCL"] = T_LCL
    row["LCL"]   = calc_LCL(T_2m, T_LCL) if (_safe(T_2m) and _safe(T_LCL)) else float("nan")

    # EPT_1..50 and EPT_2m
    for lvl in range(1,51):
        theta=row.get(f"POT_{lvl}",float("nan")); q=row.get(f"SPFH_{lvl}"); TK=row.get(f"TMP_{lvl}")
        if all(_safe(x) for x in (theta,q,TK,T_2m,RH_2m)):
            row[f"EPT_{lvl}"] = calc_EPT(theta,q,TK,T_2m,RH_2m)
        else:
            row[f"EPT_{lvl}"] = float("nan")
    th2m=row.get("POT_2m"); q2m=row.get("SPFH_2m"); TK2m=row.get("TMP_2m")
    row["EPT_2m"] = calc_EPT(th2m,q2m,TK2m,T_2m,RH_2m) if all(_safe(x) for x in (th2m,q2m,TK2m,T_2m,RH_2m)) else float("nan")

    # VTMPLR_12..4950 (Tv-based ΔZ)
    for lvl in range(1,50):
        T1=row.get(f"TMP_{lvl}"); T2=row.get(f"TMP_{lvl+1}")
        q1=row.get(f"SPFH_{lvl}"); q2=row.get(f"SPFH_{lvl+1}")
        P1=row.get(f"PRES_{lvl}"); P2=row.get(f"PRES_{lvl+1}")
        row[f"VTMPLR_{lvl}{lvl+1}"] = (calc_VTMPLR(T1,T2,q1,q2,P1,P2)
            if all(_safe(x) for x in (T1,T2,q1,q2,P1,P2)) else float("nan"))

    # VS_12..4950 (Tv-based ΔZ)
    for lvl in range(1,50):
        U1=row.get(f"UGRD_{lvl}"); U2=row.get(f"UGRD_{lvl+1}")
        V1=row.get(f"VGRD_{lvl}"); V2=row.get(f"VGRD_{lvl+1}")
        T1=row.get(f"TMP_{lvl}"); T2=row.get(f"TMP_{lvl+1}")
        q1=row.get(f"SPFH_{lvl}"); q2=row.get(f"SPFH_{lvl+1}")
        P1=row.get(f"PRES_{lvl}"); P2=row.get(f"PRES_{lvl+1}")
        row[f"VS_{lvl}{lvl+1}"] = (calc_VS(U1,U2,V1,V2,T1,T2,q1,q2,P1,P2)
            if all(_safe(x) for x in (U1,U2,V1,V2,T1,T2,q1,q2,P1,P2)) else float("nan"))

    # BowenRatio, ZL_MIN, ZL_MAX
    SH=row.get("SHTFL"); LH=row.get("LHTFL")
    row["BowenRatio_Mean"] = calc_BowenRatio(SH,LH) if (SH is not None and LH is not None) else float("nan")
    zl = calc_ZL(row.get("HPBL"), row.get("FRICV"), row.get("POT_2m"),
                 row.get("SHTFL"), row.get("PRES_SFC"))
    row["ZL_MIN"] = zl
    row["ZL_MAX"] = zl
    return row

def derived_column_order() -> List[str]:
    cols = [f"POT_{i}" for i in range(1,51)]
    cols += ["T_LCL","LCL","EPT_2m"]
    cols += [f"EPT_{i}" for i in range(1,51)]
    cols += [f"RH_{i}"  for i in range(1,51)]
    cols += [f"DPT_{i}" for i in range(1,51)]
    cols += [f"VTMPLR_{i}{i+1}" for i in range(1,50)]
    cols += [f"VS_{i}{i+1}"     for i in range(1,50)]
    cols += ["BowenRatio_Mean","ZL_MIN","ZL_MAX"]
    return cols
