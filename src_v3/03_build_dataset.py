#!/usr/bin/env python3
"""
03_build_dataset.py — Merge features+labels → ML-ready dataset  (v2)
Uses stratified split. Reads all settings from config.yaml.
Output: data/dataset/train.csv, val.csv, test.csv, dataset_full.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils.config_loader import load_config

def main():
    cfg          = load_config()
    features_dir = cfg["paths"]["features_dir"]
    labels_dir   = cfg["paths"]["labels_dir"]
    dataset_dir  = cfg["paths"]["dataset_dir"]
    ds           = cfg["dataset"]

    lfiles = sorted(labels_dir.glob("labels_*.csv"))
    if not lfiles:
        print("ERROR: No label files. Run step 2 first."); sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  STEP 3: BUILD DATASET  (stratified split)")
    print(f"  Label files: {len(lfiles)}")
    print(f"{'='*65}\n")

    all_dfs = []
    for lf in lfiles:
        tag = lf.stem.replace("labels_","")
        ff  = features_dir / f"features_{tag}.csv"
        if not ff.exists(): continue
        ldf = pd.read_csv(lf); fdf = pd.read_csv(ff)
        keys = [k for k in ["cycle_time","forecast_hour","latitude","longitude","region"]
                if k in ldf.columns]
        merged = fdf.merge(
            ldf[keys + [c for c in ["valid_time","label"] if c in ldf.columns]],
            on=keys, how="inner")
        all_dfs.append(merged)

    if not all_dfs:
        print("ERROR: No data merged."); sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Merged: {len(df)} rows, {len(df.columns)} columns")

    before = len(df)
    df = df[df["label"] != -1].copy()
    print(f"  Dropped {before-len(df)} missing-label rows")

    # Sort chronologically
    if "valid_time" in df.columns:
        df["valid_time"] = pd.to_datetime(df["valid_time"])
        df = df.sort_values("valid_time").reset_index(drop=True)

    # Julian Day sin/cos (per Waylon PDF page 14)
    if "valid_time" in df.columns:
        jday = df["valid_time"].dt.dayofyear
        df["julian_day_sin"] = np.sin(2*np.pi*jday/365.0)
        df["julian_day_cos"] = np.cos(2*np.pi*jday/365.0)
        print("  Added julian_day_sin and julian_day_cos")

    # Stratified split
    from sklearn.model_selection import StratifiedShuffleSplit
    n = len(df); y = df["label"].values

    sss1 = StratifiedShuffleSplit(1, test_size=ds["test_size"], random_state=ds["random_state"])
    tv_idx, test_idx = next(sss1.split(np.zeros(n), y))

    val_frac = ds["val_size"] / (1 - ds["test_size"])
    sss2 = StratifiedShuffleSplit(1, test_size=val_frac, random_state=ds["random_state"])
    y_tv = y[tv_idx]
    tr_sub, val_sub = next(sss2.split(np.zeros(len(tv_idx)), y_tv))
    train_idx = tv_idx[tr_sub]; val_idx = tv_idx[val_sub]

    train_df = df.iloc[train_idx].copy()
    val_df   = df.iloc[val_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    df.to_csv(dataset_dir / "dataset_full.csv", index=False)
    train_df.to_csv(dataset_dir / "train.csv", index=False)
    val_df.to_csv(  dataset_dir / "val.csv",   index=False)
    test_df.to_csv( dataset_dir / "test.csv",  index=False)

    def dist(d):
        n1=(d["label"]==1).sum(); n0=(d["label"]==0).sum()
        return f"{len(d)} rows | lightning={n1}({100*n1/max(len(d),1):.1f}%) no-light={n0}"

    summary = f"""
DATASET BUILD SUMMARY (v2 — Stratified Split)
Full:  {dist(df)}
Train: {dist(train_df)}
Val:   {dist(val_df)}
Test:  {dist(test_df)}
Features: {len(df.columns)}
"""
    print(summary)
    with open(dataset_dir/"split_summary.txt","w") as f:
        f.write(summary)
    print(f"  Saved to {dataset_dir}\n")

if __name__ == "__main__":
    main()
