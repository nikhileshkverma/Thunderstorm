"""
XGBoost Lightning Model
Train + Validate on January 2024
South Texas
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
# 2. CREATE BINARY TARGET (>=1 flash)
# ============================================================

for threshold in [1, 10, 50]:
    print(f"\n\n===== Threshold >= {threshold} flashes =====")

    df["target"] = (df["lightning_count"] >= threshold).astype(int)

    train = df[df["doy"] <= 25]
    val   = df[df["doy"] > 25]

    X_train = train.drop(columns=drop_cols)
    y_train = train["target"]

    X_val = val.drop(columns=drop_cols)
    y_val = val["target"]

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1

    model = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    POD = tp / (tp + fn) if (tp + fn) > 0 else 0
    FAR = fp / (tp + fp) if (tp + fp) > 0 else 0
    CSI = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

    print("POD:", round(POD,3))
    print("FAR:", round(FAR,3))
    print("CSI:", round(CSI,3))
# ============================================================
# 3. TIME-BASED SPLIT
# Train: DOY 1–25
# Validate: DOY 26–31
# ============================================================

train = df[df["doy"] <= 25]
val   = df[df["doy"] > 25]

print("Train size:", len(train))
print("Validation size:", len(val))

# ============================================================
# 4. SELECT FEATURES
# ============================================================

drop_cols = [
    "year", "doy", "date",
    "box_i", "box_j",
    "lightning_count",
    "target"
]

X_train = train.drop(columns=drop_cols)
y_train = train["target"]

X_val = val.drop(columns=drop_cols)
y_val = val["target"]

# ============================================================
# 5. HANDLE CLASS IMBALANCE
# ============================================================

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

print("Scale pos weight:", scale_pos_weight)

# ============================================================
# 6. TRAIN XGBOOST
# ============================================================

model = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

print("Training XGBoost...")
model.fit(X_train, y_train)

# ============================================================
# 7. PREDICT PROBABILITIES
# ============================================================

y_prob = model.predict_proba(X_val)[:, 1]

# ============================================================
# 8. BASELINE EVALUATION (Threshold = 0.5)
# ============================================================

print("\n--- Baseline (Threshold = 0.5) ---")

y_pred_base = (y_prob >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_val, y_pred_base).ravel()

POD = tp / (tp + fn) if (tp + fn) > 0 else 0
FAR = fp / (tp + fp) if (tp + fp) > 0 else 0
CSI = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
AUC = roc_auc_score(y_val, y_prob)

print("Confusion Matrix:")
print([[tn, fp], [fn, tp]])
print(f"POD: {POD:.3f}")
print(f"FAR: {FAR:.3f}")
print(f"CSI: {CSI:.3f}")
print(f"AUC: {AUC:.3f}")

# ============================================================
# 9. THRESHOLD OPTIMIZATION (Maximize CSI)
# ============================================================

print("\n--- Optimizing Threshold for CSI ---")

best_csi = 0
best_threshold = 0

for t in np.arange(0.05, 0.6, 0.05):
    y_t = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_t).ravel()
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

    if csi > best_csi:
        best_csi = csi
        best_threshold = t

print("Best Threshold:", best_threshold)
print("Best CSI:", round(best_csi, 3))

# ============================================================
# 10. FINAL EVALUATION (Best Threshold)
# ============================================================

print("\n--- Final Evaluation (Optimized Threshold) ---")

y_pred_best = (y_prob >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_val, y_pred_best).ravel()

POD = tp / (tp + fn) if (tp + fn) > 0 else 0
FAR = fp / (tp + fp) if (tp + fp) > 0 else 0
CSI = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

print("Confusion Matrix:")
print([[tn, fp], [fn, tp]])
print(f"POD: {POD:.3f}")
print(f"FAR: {FAR:.3f}")
print(f"CSI: {CSI:.3f}")

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

importances = model.feature_importances_
feature_names = X_train.columns

indices = np.argsort(importances)[::-1]

print("\nTop 10 Important Features:")
for i in range(10):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")