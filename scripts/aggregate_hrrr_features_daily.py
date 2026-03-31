import pandas as pd
from pathlib import Path

FEATURE_FILE = Path(
    "/Storage03/nverma1/lightning_project/output/features/"
    "hrrr_features_january_2024_south_texas.csv"
)

OUT_FILE = Path(
    "/Storage03/nverma1/lightning_project/output/features/"
    "hrrr_features_january_2024_daily.csv"
)

print("Loading HRRR features...")
df = pd.read_csv(FEATURE_FILE)

# Identify feature columns (exclude keys)
key_cols = ["year", "doy", "date", "box_i", "box_j"]
feature_cols = [c for c in df.columns if c not in key_cols]

print("Aggregating to daily features...")

daily = (
    df
    .groupby(key_cols)[feature_cols]
    .agg(["mean", "max"])
)

# Flatten column names
daily.columns = [
    f"{var}_{stat}" for var, stat in daily.columns
]

daily = daily.reset_index()

daily.to_csv(OUT_FILE, index=False)

print("DONE")
print("Saved:", OUT_FILE)
print("Rows:", len(daily))