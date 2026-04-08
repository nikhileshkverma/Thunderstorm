"""
XGBoost Lightning Model
January 2024 – South Texas
Multi-threshold experiment (>=1, >=10, >=50 flashes)
Author: Nikhilesh Verma
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading January dataset...")
df = pd.read_csv(
    "/Storage03/nverma1/lightning_project/output/dataset_january_2024_south_texas.csv"
)

print("Total rows:", len(df))

# ============================================================
# 2. FEATURE COLUMN DEFINITIONS
# ============================================================

drop_cols = [
    "year", "doy", "date",
    "box_i", "box_j",
    "lightning_count",
    "target"
]

# ============================================================
# 3. RUN MULTIPLE THRESHOLDS
# ============================================================

for threshold in [1, 10, 50]:

    print("\n==============================================")
    print(f"THRESHOLD >= {threshold} FLASHES")
    print("==============================================")

    # Create binary target
    df["target"] = (df["lightning_count"] >= threshold).astype(int)

    # Time split
    train = df[df["doy"] <= 25]
    val   = df[df["doy"] > 25]

    print("Train size:", len(train))
    print("Validation size:", len(val))

    X_train = train.drop(columns=drop_cols)
    y_train = train["target"]

    X_val = val.drop(columns=drop_cols)
    y_val = val["target"]

    # Handle imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    if pos == 0:
        print("No positive samples for this threshold. Skipping.")
        continue

    scale_pos_weight = neg / pos
    print("Scale pos weight:", round(scale_pos_weight, 2))

    # Train model
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42
    )

    print("Training XGBoost...")
    model.fit(X_train, y_train)

    # Predict probabilities
    y_prob = model.predict_proba(X_val)[:, 1]

    # Baseline threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    POD = tp / (tp + fn) if (tp + fn) > 0 else 0
    FAR = fp / (tp + fp) if (tp + fp) > 0 else 0
    CSI = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    AUC = roc_auc_score(y_val, y_prob)

    print("\nConfusion Matrix:")
    print([[tn, fp], [fn, tp]])

    print("\nSkill Scores:")
    print(f"POD: {POD:.3f}")
    print(f"FAR: {FAR:.3f}")
    print(f"CSI: {CSI:.3f}")
    print(f"AUC: {AUC:.3f}")