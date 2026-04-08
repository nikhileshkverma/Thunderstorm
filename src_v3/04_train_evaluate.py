#!/usr/bin/env python3
"""
04_train_evaluate.py — Train RF + XGBoost  (v3 — GPU accelerated)
======================================================================
Hardware usage:
  - RandomForest: n_jobs=-1 → uses all 48 CPU cores
  - XGBoost:      device='cuda' → uses Tesla V100 GPU (32GB VRAM)
                  Falls back to CPU if GPU not available
  - Feature importance stability: runs N times across multiple cores
"""

import sys, json, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils.config_loader import load_config
from utils.metrics import (compute_all_metrics, print_metrics_table,
                            track_feature_importance)

def load_split(name, dataset_dir):
    p = dataset_dir / f"{name}.csv"
    if not p.exists():
        print(f"ERROR: {p} not found. Run step 3 first."); sys.exit(1)
    return pd.read_csv(p)

def get_Xy(df, cfg, feature_names=None):
    drop = [c for c in cfg["non_feature_cols"]+["label","valid_time","region"]
            if c in df.columns]
    y = df["label"].values.astype(int) if "label" in df.columns else None
    X = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
    if feature_names is not None:
        for col in feature_names:
            if col not in X.columns: X[col] = np.nan
        X = X[feature_names]
    return X.fillna(X.median()), y

def find_best_threshold(y, prob, cfg):
    best_score=-1; best_thresh=0.5; metric=cfg["threshold_metric"]
    rng = np.arange(cfg["threshold_search_min"],
                    cfg["threshold_search_max"],
                    cfg["threshold_search_step"])
    for t in rng:
        m = compute_all_metrics(y, (prob>=t).astype(int))
        if m[metric] > best_score:
            best_score=m[metric]; best_thresh=t
    return round(float(best_thresh),2), round(float(best_score),4)

def check_gpu():
    """Check if CUDA GPU is available for XGBoost."""
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            gpus = r.stdout.strip().split('\n')
            print(f"  GPUs found: {gpus}")
            return True
    except Exception:
        pass
    return False

def main():
    cfg         = load_config()
    dataset_dir = cfg["paths"]["dataset_dir"]
    models_dir  = cfg["paths"]["models_dir"]
    results_dir = cfg["paths"]["results_dir"]
    rf_cfg      = cfg["models"]["random_forest"]
    xgb_cfg     = cfg["models"]["xgboost"]
    eval_cfg    = cfg["evaluation"]
    n_runs      = rf_cfg.get("importance_runs", 10)

    print(f"\n{'='*65}")
    print(f"  STEP 4: TRAIN + EVALUATE  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    from multiprocessing import cpu_count
    print(f"  CPU cores available: {cpu_count()}")
    gpu_ok = check_gpu()
    print(f"  GPU available: {'YES — XGBoost will use Tesla V100' if gpu_ok else 'NO — using CPU'}")
    print(f"{'='*65}\n")

    train_df = load_split("train", dataset_dir)
    val_df   = load_split("val",   dataset_dir)
    test_df  = load_split("test",  dataset_dir)

    X_train, y_train = get_Xy(train_df, cfg)
    feature_names    = list(X_train.columns)
    X_val,  y_val    = get_Xy(val_df,  cfg, feature_names)
    X_test, y_test   = get_Xy(test_df, cfg, feature_names)

    print(f"  Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")
    print(f"  Features: {len(feature_names)}  |  Train lightning rate: {y_train.mean():.1%}")

    if len(X_train)==0 or len(feature_names)==0:
        print("  ERROR: Empty training set."); sys.exit(1)

    json.dump(feature_names, open(models_dir/"feature_names.json","w"))
    all_metrics = {}

    # ── RANDOM FOREST — uses all 48 CPU cores ─────────────────────────────────
    print(f"\n── Random Forest  (n_jobs=-1 → {cpu_count()} cores) ──────────────")
    from sklearn.ensemble import RandomForestClassifier
    rf_params = {
        "n_estimators":    rf_cfg["n_estimators"],
        "max_depth":       None if rf_cfg.get("max_depth") in (None,"null") else rf_cfg["max_depth"],
        "min_samples_leaf":rf_cfg["min_samples_leaf"],
        "n_jobs":          -1,   # ALL cores
        "random_state":    rf_cfg["random_state"],
        "class_weight":    rf_cfg["class_weight"],
    }
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)

    rf_thresh,_ = find_best_threshold(y_val, rf.predict_proba(X_val)[:,1], eval_cfg)
    print(f"  Best threshold (max {eval_cfg['threshold_metric']} on val): {rf_thresh}")

    for sname, X_s, y_s in [("train",X_train,y_train),("val",X_val,y_val),("test",X_test,y_test)]:
        prob = rf.predict_proba(X_s)[:,1]
        m    = compute_all_metrics(y_s, (prob>=rf_thresh).astype(int), prob, threshold=rf_thresh)
        print_metrics_table(m, f"RF {sname.upper()} | "
            f"n_est={rf_params['n_estimators']} min_leaf={rf_params['min_samples_leaf']} "
            f"class_weight={rf_params['class_weight']}")
        all_metrics[f"RF_{sname}"] = m

    print(f"\n  Running feature importance {n_runs}× (stability tracking)...")
    fi_stable = track_feature_importance(rf, X_train, y_train, feature_names, n_runs=n_runs)
    fi_stable.to_csv(results_dir/"feature_importance_rf_stable.csv", index=False)
    pd.DataFrame({"feature":feature_names,"importance":rf.feature_importances_})\
      .sort_values("importance",ascending=False)\
      .to_csv(results_dir/"feature_importance_rf.csv", index=False)

    print("  Top 10 RF features (mean_imp | top5_count/10 runs):")
    print(fi_stable.head(10)[["feature","mean_importance","top5_count","top10_count"]].to_string(index=False))

    pickle.dump({"model":rf,"threshold":rf_thresh,"feature_names":feature_names,
                 "params":rf_params,"train_shape":X_train.shape},
                open(models_dir/"rf_model.pkl","wb"))

    # ── XGBOOST — uses Tesla V100 GPU if available ─────────────────────────────
    print(f"\n── XGBoost  ({'GPU: Tesla V100' if gpu_ok else 'CPU mode'}) ─────────────")
    try:
        from xgboost import XGBClassifier
        n_neg=int(np.sum(y_train==0)); n_pos=int(np.sum(y_train==1))

        # GPU params: device='cuda' for XGBoost >= 2.0
        xgb_params = {
            "n_estimators":     xgb_cfg["n_estimators"],
            "max_depth":        xgb_cfg["max_depth"],
            "learning_rate":    xgb_cfg["learning_rate"],
            "subsample":        xgb_cfg["subsample"],
            "colsample_bytree": xgb_cfg["colsample_bytree"],
            "eval_metric":      xgb_cfg["eval_metric"],
            "random_state":     xgb_cfg["random_state"],
            "n_jobs":           -1,
            "scale_pos_weight": n_neg / max(n_pos,1),
        }
        if gpu_ok:
            xgb_params["device"] = "cuda"
            print("  Using GPU: device=cuda (Tesla V100 32GB)")
        else:
            xgb_params["tree_method"] = "hist"
            print("  Using CPU: tree_method=hist")

        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=50)

        xgb_thresh,_ = find_best_threshold(y_val, xgb.predict_proba(X_val)[:,1], eval_cfg)
        print(f"  Best threshold: {xgb_thresh}")

        for sname, X_s, y_s in [("train",X_train,y_train),("val",X_val,y_val),("test",X_test,y_test)]:
            prob = xgb.predict_proba(X_s)[:,1]
            m    = compute_all_metrics(y_s, (prob>=xgb_thresh).astype(int), prob, threshold=xgb_thresh)
            print_metrics_table(m, f"XGB {sname.upper()} | "
                f"n_est={xgb_params['n_estimators']} depth={xgb_params['max_depth']} "
                f"lr={xgb_params['learning_rate']} "
                f"{'GPU' if gpu_ok else 'CPU'}")
            all_metrics[f"XGB_{sname}"] = m

        pd.DataFrame({"feature":feature_names,"importance":xgb.feature_importances_})\
          .sort_values("importance",ascending=False)\
          .to_csv(results_dir/"feature_importance_xgb.csv", index=False)
        pickle.dump({"model":xgb,"threshold":xgb_thresh,"feature_names":feature_names,
                     "params":xgb_params,"train_shape":X_train.shape},
                    open(models_dir/"xgb_model.pkl","wb"))
        print(f"  Saved XGBoost model → {models_dir/'xgb_model.pkl'}")

    except ImportError:
        print("  XGBoost not installed — skipping")

    # ── Save metrics ──────────────────────────────────────────────────────────
    rows = [{"model":k.split("_")[0],"split":k.split("_")[1],**m}
            for k,m in all_metrics.items()]
    mdf = pd.DataFrame(rows)
    mdf.to_csv(results_dir/"metrics_train_val_test.csv", index=False)

    print(f"\n{'='*65}  MODEL COMPARISON — TEST SET")
    cols = ["model","accuracy","f1_score","POD","FAR","CSI","HSS","AUC_ROC"]
    print(mdf[mdf["split"]=="test"][cols].to_string(index=False))
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()
