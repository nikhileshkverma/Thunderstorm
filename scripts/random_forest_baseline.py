import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

data = pd.read_csv(
    "/Storage03/nverma1/lightning_project/output/"
    "dataset_january_2024_south_texas.csv"
)

# Binary target
data["target_any"] = (data["lightning_count"] >= 1).astype(int)

# Select meteorological features only
feature_cols = [
    c for c in data.columns
    if c.endswith("_mean") or c.endswith("_max")
]

X = data[feature_cols]
y = data["target_any"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# Skill Scores
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

POD = TP / (TP + FN + 1e-6)
FAR = FP / (TP + FP + 1e-6)
CSI = TP / (TP + FP + FN + 1e-6)

print("Confusion Matrix:")
print(cm)

print("\nSkill Scores:")
print(f"POD (Recall): {POD:.3f}")
print(f"FAR: {FAR:.3f}")
print(f"CSI: {CSI:.3f}")