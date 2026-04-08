import pandas as pd
from pathlib import Path

FEATURE_FILE = Path(
    "/Storage03/nverma1/lightning_project/output/features/"
    "hrrr_features_january_2024_daily.csv"
)

TARGET_FILE = Path(
    "/Storage03/nverma1/lightning_project/output/targets/"
    "glm_targets_january_2024_south_texas.csv"
)

OUT_FILE = Path(
    "/Storage03/nverma1/lightning_project/output/"
    "dataset_january_2024_south_texas.csv"
)

print("Loading features...")
X = pd.read_csv(FEATURE_FILE)

print("Loading targets...")
Y = pd.read_csv(TARGET_FILE)

print("Merging X and Y...")
data = X.merge(
    Y,
    on=["year", "doy", "date", "box_i", "box_j"],
    how="inner"
)

print("Final dataset size:", len(data))

# Basic sanity checks
print("\nLightning stats:")
print(data["lightning_count"].describe())

data.to_csv(OUT_FILE, index=False)

print("\nDONE")
print("Saved:", OUT_FILE)