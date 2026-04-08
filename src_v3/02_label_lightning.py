#!/usr/bin/env python3
"""
02_label_lightning.py — Attach binary lightning labels from GOES-16 GLM
Run AFTER 01_extract_features.py
Output: data/labels/labels_{YYYYMMDD}_{CC}Z.csv
"""
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent))
from utils.config_loader import load_config, get_active_regions
from utils.glm_reader import label_dataframe

def main():
    cfg          = load_config()
    features_dir = cfg["paths"]["features_dir"]
    labels_dir   = cfg["paths"]["labels_dir"]
    regions      = get_active_regions(cfg)
    ffiles       = sorted(features_dir.glob("features_*.csv"))

    if not ffiles:
        print("ERROR: No feature files. Run step 1 first."); sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  STEP 2: LIGHTNING LABELING")
    print(f"  Feature files: {len(ffiles)}")
    print(f"  GLM base: {cfg['paths']['glm_base']}")
    print(f"  Window: ±{cfg['labeling']['window_hours']}h  Box: ±{cfg['labeling']['box_half_deg']}°")
    print(f"{'='*65}\n")

    done = skipped = 0
    for ff in tqdm(ffiles, desc="Labeling"):
        tag     = ff.stem.replace("features_","")
        out_file = labels_dir / f"labels_{tag}.csv"
        if out_file.exists():
            skipped += 1; continue

        df = pd.read_csv(ff)
        if df.empty: continue

        labeled_dfs = []
        for region in regions:
            rdf = df[df["region"]==region["name"]].copy() if "region" in df.columns else df.copy()
            if rdf.empty: continue
            labeled = label_dataframe(rdf, region["lat"], region["lon"], verbose=False)
            labeled_dfs.append(labeled)

        if not labeled_dfs: continue
        combined = pd.concat(labeled_dfs, ignore_index=True)
        n_miss = (combined["label"]==-1).sum()
        if n_miss: tqdm.write(f"  WARNING: {n_miss} rows missing GLM in {ff.name}")
        combined.to_csv(out_file, index=False)
        done += 1

    # Summary
    all_lf = sorted(labels_dir.glob("labels_*.csv"))
    if all_lf:
        all_labels = pd.concat([pd.read_csv(f) for f in all_lf], ignore_index=True)
        n = len(all_labels)
        n1 = (all_labels["label"]==1).sum(); n0 = (all_labels["label"]==0).sum()
        nm = (all_labels["label"]==-1).sum()
        print(f"\n  LABELING COMPLETE: written={done}  skipped={skipped}")
        print(f"  Total={n}  Lightning={n1}({100*n1/max(n,1):.1f}%)  No-light={n0}  Missing={nm}\n")

if __name__ == "__main__":
    main()
