"""
HRRR 24km Grid + Lightning Visualization
January 2024 – South Texas
Final Clean Scientific Version
Author: Nikhilesh Verma
"""

import pandas as pd
import numpy as np
import pygrib
import folium
from pathlib import Path

# ============================================================
# 1. LOAD LIGHTNING DATA
# ============================================================

DATA_PATH = "/Storage03/nverma1/lightning_project/output/dataset_january_2024_south_texas.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Pick strongest lightning day
daily = df.groupby("date")["lightning_count"].sum()
selected_date = daily.sort_values(ascending=False).index[0]

print("Selected lightning day:", selected_date)

df_day = df[df["date"] == selected_date]

print("Non-zero lightning boxes:", (df_day["lightning_count"] > 0).sum())

# ============================================================
# 2. LOAD HRRR GRID (ONLY GRID, NOT REFLECTIVITY)
# ============================================================

HRRR_DIR = Path("/Storage03/nverma1/HRRR_DATA/2024")
hrrr_file = list(HRRR_DIR.rglob("*.grib2"))[0]

print("Using HRRR file:", hrrr_file)

with pygrib.open(str(hrrr_file)) as grbs:
    grb = grbs[1]   # just extract lat/lon grid
    _, lats, lons = grb.data()

# ============================================================
# 3. CREATE MAP
# ============================================================

m = folium.Map(
    location=[28.5, -97.5],
    zoom_start=7,
    tiles="cartodbpositron"
)

# ============================================================
# 4. DRAW 24km HRRR GRID (8x8 native cells)
# ============================================================

BOX_SIZE = 8   # 3km × 8 ≈ 24km

print("Drawing 24km HRRR grid...")

for i in range(0, lats.shape[0] - BOX_SIZE, BOX_SIZE):
    for j in range(0, lats.shape[1] - BOX_SIZE, BOX_SIZE):

        lat_center = lats[i:i+BOX_SIZE, j:j+BOX_SIZE].mean()
        lon_center = lons[i:i+BOX_SIZE, j:j+BOX_SIZE].mean()

        # South Texas bounding box
        if not (25 <= lat_center <= 31 and -100 <= lon_center <= -93):
            continue

        lat_min = lats[i:i+BOX_SIZE, j:j+BOX_SIZE].min()
        lat_max = lats[i:i+BOX_SIZE, j:j+BOX_SIZE].max()
        lon_min = lons[i:i+BOX_SIZE, j:j+BOX_SIZE].min()
        lon_max = lons[i:i+BOX_SIZE, j:j+BOX_SIZE].max()

        folium.Rectangle(
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            color="blue",
            weight=1,
            dash_array="4,4",
            fill=False
        ).add_to(m)

# ============================================================
# 5. ADD LIGHTNING (ONLY STRONG CELLS)
# ============================================================

print("Adding strong lightning markers...")

for _, row in df_day.iterrows():

    if row["lightning_count"] < 50:   # 🔥 Only strong cells
        continue

    i = int(row["box_i"])
    j = int(row["box_j"])

    lat_center = lats[i:i+BOX_SIZE, j:j+BOX_SIZE].mean()
    lon_center = lons[i:i+BOX_SIZE, j:j+BOX_SIZE].mean()

    folium.CircleMarker(
        location=[lat_center, lon_center],
        radius=5,
        color="red",
        fill=True,
        fill_opacity=0.85,
        popup=f"Flashes: {row['lightning_count']}"
    ).add_to(m)

# ============================================================
# 6. SAVE OUTPUT
# ============================================================

out_file = f"lightning_hrrr_24km_clean_{selected_date}.html"
m.save(out_file)

print("Saved:", out_file)
print("Visualization complete.")