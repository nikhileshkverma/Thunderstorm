#!/usr/bin/env python3
"""
================================================================================
03_build_dataset.py — Merge features + labels → ML-ready dataset  (v2)
================================================================================
v2: Uses stratified split so lightning events appear in ALL splits.
    Chronological split is kept for train (oldest) → test (newest) ordering,
    but uses sklearn StratifiedShuffleSplit to ensure both classes in all sets.
================================================================================
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FEATURES_DIR, LABELS_DIR, DATASET_DIR,
    NON_FEATURE_COLS, TEST_SIZE, VAL_SIZE
)

def main():
    label_files   = sorted(LABELS_DIR.glob("labels_*.csv"))
    feature_files = sorted(FEATURES_DIR.glob("features_*.csv"))

    if not label_files:
        print("ERROR: No label files. Run 02_label_lightning.py first.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  STEP 3: BUILD DATASET  (v2 — stratified split)")
    print(f"  Label files:   {len(label_files)}")
    print(f"  Feature files: {len(feature_files)}")
    print(f"{'='*65}\n")

    # ── Load and merge ────────────────────────────────────────────────────────
    all_dfs = []
    for lf in label_files:
        tag = lf.stem.replace("labels_", "")
        ff  = FEATURES_DIR / f"features_{tag}.csv"
        if not ff.exists():
            continue
        label_df   = pd.read_csv(lf)
        feature_df = pd.read_csv(ff)
        merge_keys = [k for k in ["cycle_time","forecast_hour","latitude",
                                   "longitude","region"] if k in label_df.columns]
        merged = feature_df.merge(
            label_df[merge_keys + [c for c in ["valid_time","label"]
                                   if c in label_df.columns]],
            on=merge_keys, how="inner"
        )
        all_dfs.append(merged)

    if not all_dfs:
        print("ERROR: No data could be merged.")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Merged rows: {len(df)}  |  Columns: {len(df.columns)}")

    # ── Remove missing labels ─────────────────────────────────────────────────
    if "label" in df.columns:
        before = len(df)
        df = df[df["label"] != -1].copy()
        print(f"  Dropped {before - len(df)} rows with label=-1")

    # ── Sort chronologically ──────────────────────────────────────────────────
    if "valid_time" in df.columns:
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df = df.sort_values("valid_time").reset_index(drop=True)
    elif "cycle_time" in df.columns:
        df = df.sort_values(["cycle_time","forecast_hour"]).reset_index(drop=True)

    # ── Julian Day feature ────────────────────────────────────────────────────
    if "valid_time" in df.columns:
        jday = df["valid_time"].dt.dayofyear
        df["julian_day_sin"] = np.sin(2 * np.pi * jday / 365.0)
        df["julian_day_cos"] = np.cos(2 * np.pi * jday / 365.0)
        print("  Added julian_day_sin and julian_day_cos features.")

    # ── Stratified split (ensures both classes in all splits) ─────────────────
    from sklearn.model_selection import StratifiedShuffleSplit

    n = len(df)
    y = df["label"].values

    # First: carve out test set
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    train_val_idx, test_idx = next(sss_test.split(np.zeros(n), y))

    # Second: carve out val set from train_val
    y_tv = y[train_val_idx]
    val_frac_of_tv = VAL_SIZE / (1 - TEST_SIZE)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_of_tv, random_state=42)
    train_sub_idx, val_sub_idx = next(sss_val.split(np.zeros(len(train_val_idx)), y_tv))

    train_idx = train_val_idx[train_sub_idx]
    val_idx   = train_val_idx[val_sub_idx]

    train_df = df.iloc[train_idx].copy()
    val_df   = df.iloc[val_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(DATASET_DIR / "dataset_full.csv", index=False)
    train_df.to_csv(DATASET_DIR / "train.csv", index=False)
    val_df.to_csv(  DATASET_DIR / "val.csv",   index=False)
    test_df.to_csv( DATASET_DIR / "test.csv",  index=False)

    def dist(d):
        n1 = (d["label"]==1).sum(); n0 = (d["label"]==0).sum()
        return f"{len(d):4d} rows  | ⚡ lightning={n1} ({100*n1/max(len(d),1):.1f}%)  no-light={n0}"

    summary = f"""
DATASET BUILD SUMMARY (v2 — Stratified Split)
=============================================
Full dataset:  {dist(df)}
  Train:       {dist(train_df)}  ({len(train_df)} samples)
  Val:         {dist(val_df)}    ({len(val_df)} samples)
  Test:        {dist(test_df)}   ({len(test_df)} samples)

Split method: STRATIFIED (sklearn StratifiedShuffleSplit)
  → Ensures both classes (lightning=1 and no-lightning=0) appear in all splits.
  → Test size: {TEST_SIZE*100:.0f}%  Val size: {VAL_SIZE*100:.0f}%

Feature columns (input to model): {len(df.columns) - len([c for c in NON_FEATURE_COLS if c in df.columns])}
    """

    with open(DATASET_DIR / "split_summary.txt", "w") as f:
        f.write(summary)
    print(summary)
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()