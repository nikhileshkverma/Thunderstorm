#!/usr/bin/env python3
"""
================================================================================
02_label_lightning.py — Attach binary lightning labels from GOES-16 GLM
================================================================================
For each features CSV in FEATURES_DIR, queries GLM data and adds:
  - valid_time  : forecast valid time (UTC)
  - label       : 1=lightning occurred, 0=no lightning, -1=data missing

Run AFTER 01_extract_features.py:
    python 02_label_lightning.py

Output:
    data/labels/labels_{YYYYMMDD}_{CC}Z.csv
================================================================================
"""

import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import FEATURES_DIR, LABELS_DIR, REGIONS
from utils.glm_reader import label_dataframe

def main():
    feature_files = sorted(FEATURES_DIR.glob("features_*.csv"))

    if not feature_files:
        print(f"ERROR: No feature files found in {FEATURES_DIR}")
        print("       Run 01_extract_features.py first.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  STEP 2: LIGHTNING LABELING")
    print(f"  Feature files found: {len(feature_files)}")
    print(f"  GLM data path: from config.GLM_BASE")
    print(f"  Lightning window: ±{2} hours around valid time")
    print(f"{'='*65}\n")

    done = skipped = missing = 0

    for ff in tqdm(feature_files, desc="Labeling"):
        tag      = ff.stem.replace("features_", "")    # YYYYMMDD_CCZ
        out_file = LABELS_DIR / f"labels_{tag}.csv"

        if out_file.exists():
            skipped += 1
            continue

        df = pd.read_csv(ff)
        if df.empty:
            continue

        # Label each row for each region
        labeled_dfs = []
        for region in REGIONS:
            region_df = df[df["region"] == region["name"]].copy() \
                        if "region" in df.columns else df.copy()
            if region_df.empty:
                continue

            labeled = label_dataframe(
                region_df,
                center_lat=region["lat"],
                center_lon=region["lon"],
                verbose=False
            )
            labeled_dfs.append(labeled)

        if not labeled_dfs:
            continue

        combined = pd.concat(labeled_dfs, ignore_index=True)

        # Report data availability
        n_missing = (combined["label"] == -1).sum()
        if n_missing > 0:
            missing += n_missing
            tqdm.write(f"  WARNING: {n_missing} rows have no GLM data in {ff.name}")

        combined.to_csv(out_file, index=False)
        done += 1

    # Summary statistics
    all_label_files = sorted(LABELS_DIR.glob("labels_*.csv"))
    if all_label_files:
        all_labels = pd.concat(
            [pd.read_csv(f) for f in all_label_files], ignore_index=True
        )
        n_total     = len(all_labels)
        n_lightning = (all_labels["label"] == 1).sum()
        n_no_light  = (all_labels["label"] == 0).sum()
        n_missing_t = (all_labels["label"] == -1).sum()

        print(f"\n{'='*65}")
        print(f"  LABELING COMPLETE")
        print(f"  Written: {done}  |  Skipped: {skipped}")
        print(f"  Total rows:      {n_total}")
        print(f"  Lightning  (1):  {n_lightning}  ({100*n_lightning/max(n_total,1):.1f}%)")
        print(f"  No-light   (0):  {n_no_light}  ({100*n_no_light/max(n_total,1):.1f}%)")
        print(f"  Data missing(-1): {n_missing_t}")
        print(f"  Output: {LABELS_DIR}")
        print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
