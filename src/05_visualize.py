#!/usr/bin/env python3
"""
================================================================================
05_visualize.py — Generate all result plots  (v2 — fixed for single-class test)
================================================================================
Plots saved to RESULTS_DIR/plots/:
  1.  label_distribution.png        — class balance overall + by forecast hour
  2.  lightning_timeline.png        — daily lightning rate over time
  3.  confusion_matrix_val_rf.png   — RF on validation set
  4.  confusion_matrix_val_xgb.png  — XGB on validation set
  5.  roc_curve_comparison.png      — ROC both models (val set)
  6.  pr_curve_comparison.png       — Precision-Recall both models (val set)
  7.  feature_importance_rf_top30.png
  8.  feature_importance_xgb_top30.png
  9.  metrics_summary.png           — Bar chart: all key metrics side-by-side
  10. lightning_by_cycle_fh.png     — Heatmap: lightning rate by cycle × FH
================================================================================
"""

import sys, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from config import DATASET_DIR, MODELS_DIR, RESULTS_DIR
from utils.metrics import get_roc_data, get_pr_data, compute_all_metrics

PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"rf": "#2196F3", "xgb": "#FF5722", "neutral": "#4CAF50"}

# =============================================================================
def load_model(name):
    p = MODELS_DIR / f"{name}_model.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None

def load_split(name):
    p = DATASET_DIR / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None

def get_Xy(df, feature_names):
    from config import NON_FEATURE_COLS
    drop = [c for c in NON_FEATURE_COLS + ["label","valid_time","region"] if c in df.columns]
    y = df["label"].values.astype(int)
    X = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
    for col in feature_names:
        if col not in X.columns: X[col] = np.nan
    X = X[feature_names].fillna(X[feature_names].median())
    return X, y

def savefig(name):
    p = PLOTS_DIR / name
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {p.name}")

# =============================================================================
# PLOT 1: LABEL DISTRIBUTION
# =============================================================================
def plot_label_distribution(full_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Label Distribution — January 2024", fontsize=13, fontweight="bold")

    # Pie chart
    n1 = (full_df["label"] == 1).sum()
    n0 = (full_df["label"] == 0).sum()
    axes[0].pie([n0, n1], labels=["No Lightning", "Lightning ⚡"],
                autopct="%1.1f%%", colors=["#90CAF9", "#FF5722"],
                startangle=90, textprops={"fontsize": 11})
    axes[0].set_title(f"Overall  (n={len(full_df)})")

    # By forecast hour
    if "forecast_hour" in full_df.columns:
        fh_rate = full_df.groupby("forecast_hour")["label"].apply(lambda x: (x==1).mean()*100)
        axes[1].bar(fh_rate.index, fh_rate.values, color=COLORS["rf"], alpha=0.85)
        axes[1].set_xlabel("Forecast Hour"); axes[1].set_ylabel("Lightning Rate (%)")
        axes[1].set_title("By Forecast Hour"); axes[1].set_xticks(fh_rate.index)
        for x, y in zip(fh_rate.index, fh_rate.values):
            axes[1].text(x, y+0.1, f"{y:.1f}%", ha="center", fontsize=9)

    # By cycle
    if "cycle_time" in full_df.columns:
        full_df = full_df.copy()
        full_df["cycle"] = full_df["cycle_time"].str[9:11] + "Z"
        cyc_rate = full_df.groupby("cycle")["label"].apply(lambda x: (x==1).mean()*100)
        axes[2].bar(cyc_rate.index, cyc_rate.values, color=COLORS["xgb"], alpha=0.85)
        axes[2].set_xlabel("Cycle"); axes[2].set_ylabel("Lightning Rate (%)")
        axes[2].set_title("By Cycle (UTC)")
        for x, y in zip(cyc_rate.index, cyc_rate.values):
            axes[2].text(x, y+0.1, f"{y:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    savefig("label_distribution.png")

# =============================================================================
# PLOT 2: LIGHTNING TIMELINE
# =============================================================================
def plot_lightning_timeline(full_df):
    if "valid_time" not in full_df.columns:
        return
    df = full_df.copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    daily = df.groupby(df["valid_time"].dt.date)["label"].agg(
        rate=lambda x: (x==1).mean()*100,
        count="count"
    ).reset_index()
    daily["valid_time"] = pd.to_datetime(daily["valid_time"])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(daily["valid_time"], daily["rate"], alpha=0.5,
                    color=COLORS["rf"], label="Lightning %")
    ax.plot(daily["valid_time"], daily["rate"], color="#1565C0", lw=2)
    ax.axhline(daily["rate"].mean(), color="red", ls="--", lw=1,
               label=f"Mean={daily['rate'].mean():.1f}%")
    ax.set_xlabel("Date"); ax.set_ylabel("Daily Lightning Rate (%)")
    ax.set_title("Daily Lightning Occurrence Rate — January 2024", fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45); plt.tight_layout()
    savefig("lightning_timeline.png")

# =============================================================================
# PLOT 3 & 4: CONFUSION MATRICES (on VALIDATION set — where both classes exist)
# =============================================================================
def plot_confusion_matrix_safe(model_name, model, X, y, threshold, split_name):
    """Safe confusion matrix that handles single-class case."""
    n_classes = len(np.unique(y))
    prob  = model.predict_proba(X)[:, 1]
    y_pred = (prob >= threshold).astype(int)

    # Manual confusion matrix
    TP = int(np.sum((y==1) & (y_pred==1)))
    FP = int(np.sum((y==0) & (y_pred==1)))
    FN = int(np.sum((y==1) & (y_pred==0)))
    TN = int(np.sum((y==0) & (y_pred==0)))

    cm = np.array([[TN, FP], [FN, TP]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Lightning", "Lightning ⚡"], fontsize=10)
    ax.set_yticklabels(["No Lightning", "Lightning ⚡"], fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i,j] > thresh else "black")

    m = compute_all_metrics(y, y_pred)
    title = (f"{model_name.upper()} — {split_name}  "
             f"(CSI={m['CSI']:.3f}, POD={m['POD']:.3f}, FAR={m['FAR']:.3f})")
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig(f"confusion_matrix_{split_name.lower()}_{model_name}.png")

# =============================================================================
# PLOT 5: ROC CURVES
# =============================================================================
def plot_roc_curves(model_bundles, X_val, y_val):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"rf": COLORS["rf"], "xgb": COLORS["xgb"]}

    for name, bundle in model_bundles.items():
        if bundle is None: continue
        prob = bundle["model"].predict_proba(X_val)[:, 1]
        try:
            fpr, tpr, _, auc = get_roc_data(y_val, prob)
            ax.plot(fpr, tpr, color=colors[name], lw=2.5,
                    label=f"{name.upper()}  AUC={auc:.3f}")
            ax.fill_between(fpr, tpr, alpha=0.1, color=colors[name])
        except Exception as e:
            print(f"    ROC error for {name}: {e}")

    ax.plot([0,1],[0,1],"k--",lw=1,label="Random (AUC=0.50)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Validation Set", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    plt.tight_layout()
    savefig("roc_curve_comparison.png")

# =============================================================================
# PLOT 6: PRECISION-RECALL CURVES
# =============================================================================
def plot_pr_curves(model_bundles, X_val, y_val):
    fig, ax = plt.subplots(figsize=(7, 6))
    baseline = y_val.mean()

    for name, bundle in model_bundles.items():
        if bundle is None: continue
        prob = bundle["model"].predict_proba(X_val)[:, 1]
        try:
            prec, rec, _, ap = get_pr_data(y_val, prob)
            ax.plot(rec, prec, color=COLORS[name], lw=2.5,
                    label=f"{name.upper()}  AP={ap:.3f}")
            ax.fill_between(rec, prec, alpha=0.08, color=COLORS[name])
        except Exception as e:
            print(f"    PR error for {name}: {e}")

    ax.axhline(baseline, color="k", ls="--", lw=1,
               label=f"No-skill baseline ({baseline:.2f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Validation Set", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    plt.tight_layout()
    savefig("pr_curve_comparison.png")

# =============================================================================
# PLOT 7 & 8: FEATURE IMPORTANCES
# =============================================================================
def plot_feature_importance(model_name, top_n=30):
    fi_path = RESULTS_DIR / f"feature_importance_{model_name}.csv"
    if not fi_path.exists(): return
    fi = pd.read_csv(fi_path).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(7, top_n * 0.28)))
    bars = ax.barh(fi["feature"][::-1], fi["importance"][::-1],
                   color=COLORS["rf"] if model_name=="rf" else COLORS["xgb"], alpha=0.85)
    ax.set_xlabel("Feature Importance", fontsize=11)
    ax.set_title(f"{model_name.upper()} — Top {top_n} Feature Importances",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    # Add value labels
    for bar, val in zip(bars, fi["importance"][::-1]):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=7)
    plt.tight_layout()
    savefig(f"feature_importance_{model_name}_top{top_n}.png")

# =============================================================================
# PLOT 9: METRICS SUMMARY BAR CHART
# =============================================================================
def plot_metrics_summary():
    metrics_path = RESULTS_DIR / "metrics_train_val_test.csv"
    if not metrics_path.exists(): return

    df = pd.read_csv(metrics_path)
    val_df = df[df["split"] == "val"]
    if val_df.empty: return

    metric_cols = ["POD", "FAR", "CSI", "HSS", "f1_score"]
    metric_labels = ["POD\n(Hit Rate)", "FAR\n(False Alarm)", "CSI\n(Threat Score)",
                     "HSS\n(Skill Score)", "F1 Score"]

    n_metrics = len(metric_cols)
    n_models  = len(val_df)
    x = np.arange(n_metrics)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    model_colors = [COLORS["rf"], COLORS["xgb"]]

    for i, (_, row) in enumerate(val_df.iterrows()):
        values = [row.get(m, 0) for m in metric_cols]
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=row["model"].upper(),
                      color=model_colors[i % len(model_colors)], alpha=0.85)
        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance — Validation Set\n(January 2024, South Texas)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)

    # Add a note about FAR (lower is better)
    ax.annotate("↓ FAR: lower is better", xy=(1, 0.02), fontsize=9, color="gray",
                xycoords=("data","axes fraction"))
    plt.tight_layout()
    savefig("metrics_summary.png")

# =============================================================================
# PLOT 10: HEATMAP — LIGHTNING RATE BY CYCLE × FORECAST HOUR
# =============================================================================
def plot_lightning_heatmap(full_df):
    if "cycle_time" not in full_df.columns: return
    df = full_df.copy()
    df["cycle"] = df["cycle_time"].str[9:11] + "Z"
    pivot = df.pivot_table(values="label", index="cycle",
                           columns="forecast_hour", aggfunc=lambda x: (x==1).mean()*100)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=max(pivot.values.max(), 1))
    plt.colorbar(im, ax=ax, label="Lightning Rate (%)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"FH={c}" for c in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("Forecast Hour", fontsize=11)
    ax.set_ylabel("Model Cycle", fontsize=11)
    ax.set_title("Lightning Rate (%) by Cycle × Forecast Hour — January 2024",
                 fontsize=12, fontweight="bold")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    fontsize=9, color="black" if v < 15 else "white",
                    fontweight="bold")
    plt.tight_layout()
    savefig("lightning_by_cycle_fh.png")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f"\n{'='*65}")
    print(f"  STEP 5: VISUALIZATION  (v2 — all plots)")
    print(f"{'='*65}\n")

    # Load data
    full_df  = load_split("dataset_full")
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")

    if full_df is None:
        print("  ERROR: No dataset_full.csv. Run step 3 first.")
        return

    # ── Data exploration plots (no model needed) ──────────────────────────────
    print("  Data exploration plots:")
    plot_label_distribution(full_df)
    plot_lightning_timeline(full_df)
    plot_lightning_heatmap(full_df)
    plot_feature_importance("rf")
    plot_feature_importance("xgb")
    plot_metrics_summary()

    # ── Load models ───────────────────────────────────────────────────────────
    rf_bundle  = load_model("rf")
    xgb_bundle = load_model("xgb")

    if rf_bundle is None and xgb_bundle is None:
        print("  No trained models found. Run step 4 first.")
        return

    feat_path = MODELS_DIR / "feature_names.json"
    if not feat_path.exists():
        print("  No feature_names.json found.")
        return
    feature_names = json.load(open(feat_path))

    # ── Model plots — use VALIDATION set (has both classes) ──────────────────
    print("\n  Model performance plots (validation set — has both classes):")

    if val_df is not None:
        X_val, y_val = get_Xy(val_df, feature_names)

        if rf_bundle:
            plot_confusion_matrix_safe("rf",  rf_bundle["model"],
                                       X_val, y_val, rf_bundle["threshold"], "Val")
        if xgb_bundle:
            plot_confusion_matrix_safe("xgb", xgb_bundle["model"],
                                       X_val, y_val, xgb_bundle["threshold"], "Val")

        model_bundles = {}
        if rf_bundle:  model_bundles["rf"]  = rf_bundle
        if xgb_bundle: model_bundles["xgb"] = xgb_bundle

        if y_val.sum() > 0:
            plot_roc_curves(model_bundles, X_val, y_val)
            plot_pr_curves( model_bundles, X_val, y_val)
        else:
            print("  [SKIP] ROC/PR — validation set has no positive labels")

    # ── Test set note ─────────────────────────────────────────────────────────
    if test_df is not None:
        X_test, y_test = get_Xy(test_df, feature_names)
        n_pos = int(y_test.sum())
        print(f"\n  Test set: {len(y_test)} samples, {n_pos} lightning events")
        if n_pos == 0:
            print("  NOTE: Test set (late Jan 2024) has 0 lightning events.")
            print("  This is meteorologically correct — winter cold front period.")
            print("  To evaluate on test set with lightning, extend to spring/summer months.")

    print(f"\n  All plots saved to: {PLOTS_DIR}")
    print(f"  Total plots generated: {len(list(PLOTS_DIR.glob('*.png')))}")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()