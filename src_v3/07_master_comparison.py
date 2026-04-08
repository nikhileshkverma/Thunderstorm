#!/usr/bin/env python3
"""
07_master_comparison.py — Master summary chart comparing ALL models
=====================================================================
Generates the full-project summary visualization showing:
  - ALL 5 models: RF, XGBoost, Autoencoder, VAE, Transformer
  - Compared against Collins & Tissot (2015) paper baseline
  - Shows which model beats the paper on each metric
  - PSS and HSS as primary metrics (Professor Tissot preference)
  - FAR on separate chart (never mix high/low is better)
  - Detailed annotations explaining every variable shown

Professor Tissot: "Show how much better we are doing than the paper"
"""

import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.config_loader import load_config

# ── Metric descriptions for plot annotations ──────────────────────────────
METRIC_INFO = {
    "PSS":  ("Peirce Skill Score",     "POD − POFD",           "−1 to +1",  "1.0",  True),
    "HSS":  ("Heidke Skill Score",     "(Correct−Expected)/(N−Expected)", "−∞ to 1", "1.0", True),
    "CSI":  ("Critical Success Index / Threat Score", "TP/(TP+FP+FN)", "0 to 1", "1.0", True),
    "POD":  ("Probability of Detection / Hit Rate",   "TP/(TP+FN)",    "0 to 1", "1.0", True),
    "FAR":  ("False Alarm Ratio",      "FP/(TP+FP)",           "0 to 1",    "0.0",  False),
    "POFD": ("Probability of False Detection", "FP/(FP+TN)",   "0 to 1",    "0.0",  False),
    "Bias": ("Frequency Bias",         "(TP+FP)/(TP+FN)",      "0 to ∞",    "1.0",  False),
}

MODEL_COLORS = {
    "RF":           "#2196F3",  # blue
    "XGB":          "#FF5722",  # orange-red
    "AUTOENCODER":  "#9C27B0",  # purple
    "VAE":          "#4CAF50",  # green
    "TRANSFORMER":  "#FF9800",  # amber
    "2015 Paper":   "#607D8B",  # grey
}

MODEL_LABELS = {
    "RF":           "Random Forest",
    "XGB":          "XGBoost (GPU)",
    "AUTOENCODER":  "Autoencoder",
    "VAE":          "VAE (Primary)",
    "TRANSFORMER":  "Transformer",
    "2015 Paper":   "Collins & Tissot\n(2015) ANN",
}


def main():
    cfg         = load_config()
    results_dir = cfg["paths"]["results_dir"]
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    pconf       = cfg["plots"]
    paper       = cfg.get("paper_baseline", {})

    mpath = results_dir / "metrics_train_val_test.csv"
    if not mpath.exists():
        print("  No metrics file found. Run steps 4-6 first."); sys.exit(1)

    mdf = pd.read_csv(mpath)
    val_df = mdf[mdf["split"] == "val"].copy()

    # Add paper baseline row
    paper_row = {
        "model": "2015 Paper", "split": "val",
        "PSS":  paper.get("PSS"),
        "HSS":  paper.get("HSS"),
        "CSI":  paper.get("CSI",  0.54),
        "POD":  paper.get("POD",  0.71),
        "FAR":  paper.get("FAR",  0.32),
        "POFD": paper.get("POFD"),
        "Bias": None,
    }
    val_df = pd.concat([val_df, pd.DataFrame([paper_row])], ignore_index=True)

    # Order: paper first as baseline, then models
    model_order = ["2015 Paper","RF","XGB","AUTOENCODER","VAE","TRANSFORMER"]
    val_df["model_order"] = val_df["model"].map(
        {m: i for i, m in enumerate(model_order)})
    val_df = val_df.sort_values("model_order").reset_index(drop=True)
    present = [m for m in model_order if m in val_df["model"].values]

    print(f"\n  Models present in results: {present}")

    # ── Build master figure ────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 26))
    fig.patch.set_facecolor("#FAFAFA")

    gs = gridspec.GridSpec(4, 1, figure=fig,
                           hspace=0.45,
                           height_ratios=[0.5, 2.5, 2.5, 2.0])

    # ── PANEL 0: Header ────────────────────────────────────────────────────
    ax_header = fig.add_subplot(gs[0])
    ax_header.axis("off")
    ax_header.text(0.5, 0.85,
        "Deep Learning Thunderstorm Model — Full Project Model Comparison",
        ha="center", va="top", fontsize=18, fontweight="bold",
        transform=ax_header.transAxes, color="#1A237E")
    ax_header.text(0.5, 0.55,
        f"Target: {pconf['prediction_target']}",
        ha="center", va="top", fontsize=12, transform=ax_header.transAxes,
        color="#424242")
    ax_header.text(0.5, 0.28,
        f"{pconf['region_label']}  |  {pconf['date_range']}  |  "
        f"Baseline: Collins & Tissot (2015) ANN",
        ha="center", va="top", fontsize=11, transform=ax_header.transAxes,
        color="#555555")

    # ── PANEL 1: Primary metrics (high is better) ──────────────────────────
    ax1 = fig.add_subplot(gs[1])
    hi_metrics = ["PSS","HSS","CSI","POD"]
    x = np.arange(len(hi_metrics))
    n_models = len(present)
    width = 0.75 / n_models
    paper_vals = {m: val_df[val_df["model"]=="2015 Paper"][m].values[0]
                  if "2015 Paper" in val_df["model"].values else None
                  for m in hi_metrics}

    for i, mname in enumerate(present):
        row = val_df[val_df["model"] == mname]
        if row.empty: continue
        vals = [row[m].values[0] if m in row.columns else None for m in hi_metrics]
        offset = (i - n_models/2 + 0.5) * width
        color  = MODEL_COLORS.get(mname, "#999999")
        hatch  = "//" if mname == "2015 Paper" else ""
        bars   = ax1.bar(x + offset, [v if v else 0 for v in vals],
                         width, label=MODEL_LABELS.get(mname, mname),
                         color=color, alpha=0.85, hatch=hatch,
                         edgecolor="white", linewidth=0.5)
        for bar, v, metric in zip(bars, vals, hi_metrics):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                ax1.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+0.01, "N/A",
                         ha="center", va="bottom", fontsize=8, color="#999999")
                continue
            ax1.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.01, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                     color=color)
            # Star if beats paper
            pv = paper_vals.get(metric)
            if pv and v > pv and mname != "2015 Paper":
                ax1.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+0.045, "★",
                         ha="center", va="bottom", fontsize=10, color="#F57F17")

    # Paper baseline reference lines
    for j, metric in enumerate(hi_metrics):
        pv = paper_vals.get(metric)
        if pv and not np.isnan(float(pv if pv else 0)):
            ax1.axhline(pv, xmin=(j/len(hi_metrics))+0.02,
                        xmax=((j+1)/len(hi_metrics))-0.02,
                        color="#607D8B", lw=2, ls="--", alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([
        f"{m}\n({METRIC_INFO[m][0]})\nFormula: {METRIC_INFO[m][1]}\n"
        f"Range: {METRIC_INFO[m][2]}  |  Ideal: {METRIC_INFO[m][3]}"
        for m in hi_metrics], fontsize=8.5)
    ax1.set_ylim(0, 1.25)
    ax1.set_ylabel("Score  (↑ Higher is Better)", fontsize=12)
    ax1.set_title(
        "Primary Skill Metrics — Validation Set  (↑ Higher is Better)\n"
        "★ = beats Collins & Tissot (2015) paper baseline  "
        "|  --- = paper baseline value",
        fontsize=11, pad=10)
    ax1.legend(loc="upper right", fontsize=9, ncol=3,
               framealpha=0.9)
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax1.axhline(1.0, color="#BDBDBD", lw=1, ls=":", alpha=0.7)
    ax1.set_facecolor("#FAFAFA")

    # ── PANEL 2: Alarm metrics (lower is better) ───────────────────────────
    ax2 = fig.add_subplot(gs[2])
    lo_metrics = ["FAR","POFD","Bias"]
    x2 = np.arange(len(lo_metrics))
    paper_vals_lo = {m: val_df[val_df["model"]=="2015 Paper"][m].values[0]
                     if "2015 Paper" in val_df["model"].values else None
                     for m in lo_metrics}

    for i, mname in enumerate(present):
        row = val_df[val_df["model"] == mname]
        if row.empty: continue
        vals = [row[m].values[0] if m in row.columns else None for m in lo_metrics]
        offset = (i - n_models/2 + 0.5) * width
        color  = MODEL_COLORS.get(mname, "#999999")
        hatch  = "//" if mname == "2015 Paper" else ""
        bars   = ax2.bar(x2 + offset, [v if v else 0 for v in vals],
                         width, label=MODEL_LABELS.get(mname, mname),
                         color=color, alpha=0.85, hatch=hatch,
                         edgecolor="white", linewidth=0.5)
        for bar, v, metric in zip(bars, vals, lo_metrics):
            if v is None or (isinstance(v, float) and np.isnan(float(v))):
                continue
            ax2.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.01, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                     color=color)
            pv = paper_vals_lo.get(metric)
            # Star if better than paper (FAR/POFD: lower is better; Bias: closer to 1)
            beats = False
            if metric in ("FAR","POFD") and pv and v < pv and mname != "2015 Paper":
                beats = True
            elif metric == "Bias" and pv and abs(v-1) < abs(pv-1) and mname != "2015 Paper":
                beats = True
            if beats:
                ax2.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+0.045, "★",
                         ha="center", va="bottom", fontsize=10, color="#F57F17")

    for j, metric in enumerate(lo_metrics):
        pv = paper_vals_lo.get(metric)
        if pv and not (isinstance(pv, float) and np.isnan(pv)):
            ax2.axhline(pv, xmin=(j/len(lo_metrics))+0.02,
                        xmax=((j+1)/len(lo_metrics))-0.02,
                        color="#607D8B", lw=2, ls="--", alpha=0.8)

    ax2.axhline(0, color="#4CAF50", lw=1.5, ls=":", alpha=0.6,
                label="FAR/POFD ideal (0)")
    ax2.axhline(1, color="#FF9800", lw=1.5, ls=":", alpha=0.6,
                label="Bias ideal (1)")

    ax2.set_xticks(x2)
    ax2.set_xticklabels([
        f"{m}\n({METRIC_INFO[m][0]})\nFormula: {METRIC_INFO[m][1]}\n"
        f"Ideal: {METRIC_INFO[m][3]}"
        for m in lo_metrics], fontsize=9)
    ax2.set_ylim(0, max(2.2, val_df["Bias"].max()*1.2 if "Bias" in val_df.columns else 2.2))
    ax2.set_ylabel("Score  (↓ Lower is Better / Bias → 1)", fontsize=12)
    ax2.set_title(
        "Alarm & Bias Metrics — Validation Set  (FAR/POFD: ↓ Lower is Better  |  Bias: → 1.0 is Ideal)\n"
        "★ = beats Collins & Tissot (2015) paper baseline  |  --- = paper baseline value",
        fontsize=11, pad=10)
    ax2.legend(loc="upper right", fontsize=9, ncol=3, framealpha=0.9)
    ax2.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax2.set_facecolor("#FAFAFA")

    # ── PANEL 3: Summary table ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    ax3.axis("off")

    cols = ["Model","POD","FAR","CSI","PSS","HSS","Beats Paper?"]
    rows_data = []
    paper_csi = float(paper.get("CSI", 0.54) or 0.54)

    for mname in present:
        row = val_df[val_df["model"]==mname]
        if row.empty: continue
        def gv(m):
            v = row[m].values[0] if m in row.columns else None
            if v is None or (isinstance(v, float) and np.isnan(v)): return "N/A"
            return f"{v:.3f}"
        beats = ""
        if mname != "2015 Paper":
            csi_v = row["CSI"].values[0] if "CSI" in row.columns else 0
            if csi_v and not np.isnan(float(csi_v)):
                beats = "✓ YES" if csi_v > paper_csi else "✗ Not yet"
        rows_data.append([MODEL_LABELS.get(mname,mname),
                          gv("POD"), gv("FAR"), gv("CSI"),
                          gv("PSS"), gv("HSS"), beats])

    if rows_data:
        tbl = ax3.table(cellText=rows_data, colLabels=cols,
                        cellLoc="center", loc="center",
                        bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)

        # Style header
        for j in range(len(cols)):
            tbl[(0,j)].set_facecolor("#1A237E")
            tbl[(0,j)].set_text_props(color="white", fontweight="bold")

        # Style rows
        for i, row_dat in enumerate(rows_data):
            bg = "#E3F2FD" if i % 2 == 0 else "#FAFAFA"
            for j in range(len(cols)):
                tbl[(i+1,j)].set_facecolor(bg)
                if j == 6 and "✓" in str(row_dat[j]):
                    tbl[(i+1,j)].set_facecolor("#E8F5E9")
                    tbl[(i+1,j)].set_text_props(color="#1B5E20", fontweight="bold")

        # Paper row special color
        paper_row_idx = next((i for i,r in enumerate(rows_data)
                              if "2015" in str(r[0])), None)
        if paper_row_idx is not None:
            for j in range(len(cols)):
                tbl[(paper_row_idx+1,j)].set_facecolor("#ECEFF1")
                tbl[(paper_row_idx+1,j)].set_text_props(
                    color="#455A64", fontstyle="italic")

    ax3.set_title(
        "Summary Table — All Models vs Collins & Tissot (2015) Baseline\n"
        "Validation Set  |  ✓ = CSI exceeds paper baseline (CSI=0.54)",
        fontsize=11, pad=10)

    plt.savefig(plots_dir/"00_master_model_comparison.png",
                dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  ✓ 00_master_model_comparison.png saved → {plots_dir}")

    # ── Also save test-set master comparison ──────────────────────────────
    test_df = mdf[mdf["split"]=="test"].copy()
    test_df = pd.concat([test_df, pd.DataFrame([paper_row])], ignore_index=True)

    fig2, ax = plt.subplots(figsize=(14, 6))
    fig2.patch.set_facecolor("#FAFAFA"); ax.set_facecolor("#FAFAFA")
    metrics_show = ["POD","FAR","CSI","PSS","HSS"]
    x3 = np.arange(len(metrics_show))
    test_present = [m for m in model_order if m in test_df["model"].values]

    for i, mname in enumerate(test_present):
        row = test_df[test_df["model"]==mname]
        if row.empty: continue
        vals = [row[m].values[0] if m in row.columns else 0 for m in metrics_show]
        offset = (i - len(test_present)/2 + 0.5) * (0.7/len(test_present))
        bars = ax.bar(x3+offset, [v if v and not np.isnan(float(v or 0)) else 0
                                   for v in vals],
                      0.7/len(test_present),
                      label=MODEL_LABELS.get(mname,mname),
                      color=MODEL_COLORS.get(mname,"#999"),
                      alpha=0.85,
                      hatch="//" if mname=="2015 Paper" else "",
                      edgecolor="white")
        for bar, v in zip(bars, vals):
            if v and not np.isnan(float(v or 0)):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x3)
    ax.set_xticklabels([f"{m}\n({METRIC_INFO[m][0]})" for m in metrics_show],
                       fontsize=9)
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        f"Model Comparison — TEST SET\n"
        f"{pconf['region_label']}  |  {pconf['date_range']}  |  "
        f"Target: {pconf['prediction_target']}\n"
        f"Compared against Collins & Tissot (2015) paper baseline (CSI=0.54, POD=0.71, FAR=0.32)",
        fontsize=9)
    ax.legend(fontsize=9, ncol=3); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir/"00b_test_set_comparison.png", dpi=150,
                bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  ✓ 00b_test_set_comparison.png saved")
    print(f"\n  Master comparison complete → {plots_dir}\n")


if __name__ == "__main__":
    main()
