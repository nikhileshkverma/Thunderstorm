import pandas as pd
import numpy as np

INPUT_FILE = "hrrr_waylon_features_STRUCTURED_20240101.csv"
OUTPUT_FILE = "hrrr_engineered_ML_READY_20240101.csv"

df = pd.read_csv(INPUT_FILE)

# ==========================================================
# Helper: Safe column detection
# ==========================================================
def get_cols(keyword):
    return sorted([c for c in df.columns if keyword in c])

# ==========================================================
# TEMPERATURE FEATURES
# ==========================================================
temp_cols = get_cols("Temperature_")

if len(temp_cols) >= 3:
    df["mean_temperature"] = df[temp_cols].mean(axis=1)

    # Use first few as "low", middle as "mid"
    n = len(temp_cols)
    low = temp_cols[: max(2, n // 5)]
    mid = temp_cols[n // 3 : n // 2]

    if len(low) >= 2:
        df["low_level_mean_temp"] = df[low].mean(axis=1)
        df["lapse_rate_low"] = df[low[0]] - df[low[-1]]

    if len(mid) >= 2:
        df["mid_level_mean_temp"] = df[mid].mean(axis=1)
        df["lapse_rate_mid"] = df[mid[0]] - df[mid[-1]]

else:
    print("⚠ Temperature vertical columns not sufficient")

# ==========================================================
# WIND FEATURES
# ==========================================================
u_cols = get_cols("U_component_of_wind_")
v_cols = get_cols("V_component_of_wind_")

if len(u_cols) > 0 and len(v_cols) > 0:

    wind_speeds = []

    for u, v in zip(u_cols, v_cols):
        wind_speeds.append(np.sqrt(df[u]**2 + df[v]**2))

    wind_stack = np.vstack(wind_speeds).T

    df["mean_wind_speed"] = wind_stack.mean(axis=1)
    df["max_wind_speed"] = wind_stack.max(axis=1)

    # Safe shear calculation
    if len(u_cols) >= 2:
        df["bulk_shear"] = np.sqrt(
            (df[u_cols[-1]] - df[u_cols[0]])**2 +
            (df[v_cols[-1]] - df[v_cols[0]])**2
        )

# ==========================================================
# MOISTURE FEATURES
# ==========================================================
q_cols = get_cols("Specific_humidity_")
rh_cols = get_cols("Relative_humidity_")

if len(q_cols) > 0:
    df["mean_specific_humidity"] = df[q_cols].mean(axis=1)

if len(rh_cols) > 0:
    df["mean_RH"] = df[rh_cols].mean(axis=1)

if "Precipitable_water" in df.columns:
    df["PW"] = df["Precipitable_water"]

# ==========================================================
# VERTICAL MOTION
# ==========================================================
w_cols = get_cols("Vertical_velocity_")

if len(w_cols) > 0:
    df["max_updraft"] = df[w_cols].max(axis=1)
    df["mean_updraft"] = df[w_cols].mean(axis=1)
    df["updraft_depth"] = (df[w_cols] > 1.0).sum(axis=1)

# ==========================================================
# REFLECTIVITY
# ==========================================================
if "Maximum_Composite_radar_reflectivity" in df.columns:
    df["max_reflectivity"] = df["Maximum_Composite_radar_reflectivity"]
    df["convective_flag"] = (df["max_reflectivity"] >= 35).astype(int)

# ==========================================================
# SURFACE FEATURES
# ==========================================================
if "Surface_pressure" in df.columns:
    df["surface_pressure"] = df["Surface_pressure"]

if "2_metre_temperature" in df.columns:
    df["t2m"] = df["2_metre_temperature"]

# ==========================================================
# TIME ENCODING
# ==========================================================
if "cycle_time" in df.columns:
    df["hour"] = df["cycle_time"].str[9:11].astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# ==========================================================
# DROP RAW VERTICAL COLUMNS (ML VERSION)
# ==========================================================
drop_cols = temp_cols + u_cols + v_cols + q_cols + rh_cols + w_cols
df_clean = df.drop(columns=drop_cols, errors="ignore")

df_clean.to_csv(OUTPUT_FILE, index=False)

print("✅ ML-Ready Engineering Complete")
print("Saved:", OUTPUT_FILE)