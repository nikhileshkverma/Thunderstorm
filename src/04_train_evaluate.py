#!/usr/bin/env python3
"""
================================================================================
04_train_evaluate.py — Train Random Forest + XGBoost, evaluate, save models
================================================================================
Trains two baseline models on train.csv, tunes threshold on val.csv,
and evaluates final performance on test.csv.

Run AFTER 03_build_dataset.py:
    python 04_train_evaluate.py

Outputs saved to MODELS_DIR and RESULTS_DIR:
  models/rf_model.pkl
  models/xgb_model.pkl
  models/feature_names.json
  results/metrics_train_val_test.csv
  results/feature_importance_rf.csv
  results/feature_importance_xgb.csv
================================================================================
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATASET_DIR, MODELS_DIR, RESULTS_DIR,
    NON_FEATURE_COLS, RF_PARAMS, XGB_PARAMS,
    DECISION_THRESHOLD, RANDOM_STATE
)
from utils.metrics import compute_all_metrics, print_metrics_table

# =============================================================================
def load_split(name: str):
    """Load train/val/test CSV and return (X, y, df)."""
    path = DATASET_DIR / f"{name}.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run 03_build_dataset.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    return df

def get_features(df: pd.DataFrame, feature_names=None):
    """Drop non-feature columns and return X, y."""
    drop_cols = [c for c in NON_FEATURE_COLS if c in df.columns]
    # Also drop any extra metadata
    drop_cols += [c for c in ["valid_time", "region"] if c in df.columns]

    y = df["label"].values.astype(int) if "label" in df.columns else None
    X = df.drop(columns=drop_cols + (["label"] if "label" in df.columns else []),
                errors="ignore")

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    # If feature_names provided, align columns
    if feature_names is not None:
        for col in feature_names:
            if col not in X.columns:
                X[col] = np.nan
        X = X[feature_names]

    # Fill NaN with column median (robust to missing values)
    X = X.fillna(X.median())

    return X, y

def find_best_threshold(y_val, y_prob):
    """Find threshold maximising CSI on validation set."""
    best_csi = -1; best_thresh = 0.5
    for t in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_prob >= t).astype(int)
        m = compute_all_metrics(y_val, y_pred)
        if m["CSI"] > best_csi:
            best_csi   = m["CSI"]
            best_thresh = t
    return best_thresh, best_csi

# =============================================================================
def main():
    print(f"\n{'='*65}")
    print(f"  STEP 4: TRAIN + EVALUATE MODELS")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")

    X_train, y_train = get_features(train_df)
    feature_names    = list(X_train.columns)
    X_val,   y_val   = get_features(val_df,  feature_names)
    X_test,  y_test  = get_features(test_df, feature_names)

    print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}  |  Test: {X_test.shape}")
    print(f"  Features: {len(feature_names)}")

    # Guard: empty dataset (all labels were -1 / GLM data missing)
    if len(X_train) == 0 or len(feature_names) == 0:
        print()
        print("  *** ERROR: Training set is empty! ***")
        print("  Most likely cause: all labels = -1 (GLM data not found).")
        print("  Fix:")
        print("    1. Run the GLM diagnostic:")
        print("       cd src && python -c \"from utils.glm_reader import run_glm_diagnostic; run_glm_diagnostic('20240102',12)\"")
        print("    2. Confirm GLM path in config.py → GLM_BASE")
        print("    3. Rerun: bash run_pipeline.sh 2 3 4 5")
        sys.exit(1)

    print(f"  Train lightning rate: {y_train.mean():.1%}")
    print()

    # Save feature names
    feat_path = MODELS_DIR / "feature_names.json"
    with open(feat_path, "w") as f:
        json.dump(feature_names, f)
    print(f"  Saved feature list → {feat_path}")

    all_metrics = {}

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 1: RANDOM FOREST
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── Random Forest ────────────────────────────────────────────────")
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    # Find best threshold on validation set
    rf_val_prob = rf.predict_proba(X_val)[:, 1]
    rf_thresh, rf_val_csi = find_best_threshold(y_val, rf_val_prob)
    print(f"  Best threshold (max CSI on val): {rf_thresh:.2f}  CSI={rf_val_csi:.4f}")

    # Evaluate on all splits
    for split_name, X_s, y_s in [("train", X_train, y_train),
                                   ("val",   X_val,   y_val),
                                   ("test",  X_test,  y_test)]:
        prob  = rf.predict_proba(X_s)[:, 1]
        pred  = (prob >= rf_thresh).astype(int)
        m     = compute_all_metrics(y_s, pred, prob, threshold=rf_thresh)
        print_metrics_table(m, f"Random Forest — {split_name.upper()}")
        all_metrics[f"RF_{split_name}"] = m

    # Feature importances
    fi_rf = pd.DataFrame({
        "feature":    feature_names,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_rf.to_csv(RESULTS_DIR / "feature_importance_rf.csv", index=False)
    print(f"\n  Top 10 RF features:")
    print(fi_rf.head(10).to_string(index=False))

    # Save model
    rf_path = MODELS_DIR / "rf_model.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump({"model": rf, "threshold": rf_thresh,
                     "feature_names": feature_names}, f)
    print(f"  Saved → {rf_path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  MODEL 2: XGBOOST
    # ══════════════════════════════════════════════════════════════════════════
    print("\n── XGBoost ──────────────────────────────────────────────────────")
    try:
        from xgboost import XGBClassifier

        # Set scale_pos_weight for class imbalance
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        xgb_params = XGB_PARAMS.copy()
        xgb_params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False)

        xgb_val_prob = xgb.predict_proba(X_val)[:, 1]
        xgb_thresh, xgb_val_csi = find_best_threshold(y_val, xgb_val_prob)
        print(f"  Best threshold (max CSI on val): {xgb_thresh:.2f}  CSI={xgb_val_csi:.4f}")

        for split_name, X_s, y_s in [("train", X_train, y_train),
                                       ("val",   X_val,   y_val),
                                       ("test",  X_test,  y_test)]:
            prob = xgb.predict_proba(X_s)[:, 1]
            pred = (prob >= xgb_thresh).astype(int)
            m    = compute_all_metrics(y_s, pred, prob, threshold=xgb_thresh)
            print_metrics_table(m, f"XGBoost — {split_name.upper()}")
            all_metrics[f"XGB_{split_name}"] = m

        fi_xgb = pd.DataFrame({
            "feature":    feature_names,
            "importance": xgb.feature_importances_
        }).sort_values("importance", ascending=False)
        fi_xgb.to_csv(RESULTS_DIR / "feature_importance_xgb.csv", index=False)
        print(f"\n  Top 10 XGB features:")
        print(fi_xgb.head(10).to_string(index=False))

        xgb_path = MODELS_DIR / "xgb_model.pkl"
        with open(xgb_path, "wb") as f:
            pickle.dump({"model": xgb, "threshold": xgb_thresh,
                         "feature_names": feature_names}, f)
        print(f"  Saved → {xgb_path}")

    except ImportError:
        print("  XGBoost not installed. Install with: pip install xgboost")
        print("  Skipping XGBoost training.")

    # ── Save combined metrics ─────────────────────────────────────────────────
    metrics_rows = []
    for key, m in all_metrics.items():
        model, split = key.split("_", 1)
        row = {"model": model, "split": split}
        row.update(m)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = RESULTS_DIR / "metrics_train_val_test.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  All metrics saved → {metrics_path}")

    # ── Final comparison table ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  MODEL COMPARISON — TEST SET")
    print(f"{'='*65}")
    key_metrics = ["accuracy", "f1_score", "POD", "FAR", "CSI", "HSS", "AUC_ROC"]
    test_rows = metrics_df[metrics_df["split"] == "test"][["model"] + key_metrics]
    print(test_rows.to_string(index=False))
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()