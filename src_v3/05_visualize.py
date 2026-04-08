#!/usr/bin/env python3
"""
05_visualize.py — All result plots  (v4 — Professor Tissot meeting feedback)
=============================================================================
Changes from v3 (per Professor Tissot March 2026 meeting):

1. Confusion matrix:
   - "Observation" / "Prediction" (not "True label" / "Predicted label")
   - Made smaller — "it's just 4 numbers"

2. Metrics summary — SPLIT into two charts per Professor:
   - Chart A: PSS, HSS, CSI, POD  (high is better)
   - Chart B: FAR, POFD, Bias     (low/1 is target)
   - NEVER mix high-is-better with low-is-better on same chart
   - Add PSS (Peirce Skill Score) — preferred in lightning papers

3. Metrics comparison: add Collins & Tissot 2015 paper baseline
   Professor: "hard-code those measurements so you can always compare"

4. Feature importance:
   - 30 runs (was 10) per Professor: "do 30 times"
   - Add variable GROUP importance chart (e.g., "any RH in top-N")

5. Add AUC-ROC and F1 removed from main comparison (use PSS/HSS instead)

All plots include full context per Professor Tissot:
  model, hyperparams, prediction target, region, dates, n= counts
"""
import sys, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.config_loader import load_config
from utils.metrics import (get_roc_data, get_pr_data, compute_all_metrics,
                            group_feature_importance)

def main():
    cfg         = load_config()
    dataset_dir = cfg["paths"]["dataset_dir"]
    models_dir  = cfg["paths"]["models_dir"]
    results_dir = cfg["paths"]["results_dir"]
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    pconf  = cfg["plots"]
    COLORS = {"RF": "#2196F3", "XGB": "#FF5722", "Paper": "#4CAF50"}

    def savefig(name):
        p = plots_dir / name
        plt.savefig(p, dpi=cfg["plots"]["dpi"], bbox_inches="tight")
        plt.close()
        print(f"  ✓ {name}")

    def load_split(name):
        p = dataset_dir / f"{name}.csv"
        return pd.read_csv(p) if p.exists() else None

    def get_Xy(df, feature_names):
        drop = [c for c in cfg["non_feature_cols"] + ["label","valid_time","region"]
                if c in df.columns]
        y = df["label"].values.astype(int)
        X = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
        for col in feature_names:
            if col not in X.columns: X[col] = np.nan
        return X[feature_names].fillna(X[feature_names].median()), y

    def hp_str(model_name, cfg):
        if "rf" in model_name.lower():
            p = cfg["models"]["random_forest"]
            return (f"n_est={p['n_estimators']}  max_depth={p['max_depth']}  "
                    f"min_leaf={p['min_samples_leaf']}  class_wt={p['class_weight']}")
        else:
            p = cfg["models"]["xgboost"]
            return (f"n_est={p['n_estimators']}  depth={p['max_depth']}  "
                    f"lr={p['learning_rate']}  subsample={p['subsample']}")

    def full_ctx(extra=""):
        return (f"{pconf['region_label']}  |  {pconf['date_range']}  |  "
                f"{pconf['prediction_target']}" + (f"\n{extra}" if extra else ""))

    full_df  = load_split("dataset_full")
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test")
    n_tr = len(train_df) if train_df is not None else 0
    n_va = len(val_df)   if val_df   is not None else 0
    n_te = len(test_df)  if test_df  is not None else 0

    print(f"\n{'='*65}")
    print(f"  STEP 5: VISUALIZATION  (v4 — all Professor feedback applied)")
    print(f"{'='*65}\n")

    # ── 1. LABEL DISTRIBUTION ─────────────────────────────────────────────────
    if full_df is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"Label Distribution — Lightning Occurrence Prediction\n"
            f"{pconf['date_range']}  |  {pconf['region_label']}\n"
            f"Target: {pconf['prediction_target']}  |  n={len(full_df)}", fontsize=10)
        n1=(full_df["label"]==1).sum(); n0=(full_df["label"]==0).sum()
        axes[0].pie([n0,n1],labels=["No Lightning","Lightning ⚡"],
                    autopct="%1.1f%%",colors=["#90CAF9","#FF5722"],startangle=90)
        axes[0].set_title(f"Overall (n={len(full_df)})")
        if "forecast_hour" in full_df.columns:
            fhr = full_df.groupby("forecast_hour")["label"].apply(lambda x:(x==1).mean()*100)
            axes[1].bar(fhr.index,fhr.values,color="#2196F3",alpha=0.85)
            for x,y in zip(fhr.index,fhr.values):
                axes[1].text(x,y+0.1,f"{y:.1f}%",ha="center",fontsize=9)
            axes[1].set_xlabel("Forecast Hour (h)"); axes[1].set_ylabel("Lightning Rate (%)")
            axes[1].set_title("By Forecast Hour"); axes[1].set_xticks(fhr.index)
        if "cycle_time" in full_df.columns:
            df2=full_df.copy(); df2["cycle"]=df2["cycle_time"].str[9:11]+"Z"
            cr=df2.groupby("cycle")["label"].apply(lambda x:(x==1).mean()*100)
            axes[2].bar(cr.index,cr.values,color="#FF5722",alpha=0.85)
            for x,y in zip(cr.index,cr.values):
                axes[2].text(x,y+0.1,f"{y:.1f}%",ha="center",fontsize=9)
            axes[2].set_xlabel("HRRR Cycle (UTC)"); axes[2].set_ylabel("Lightning Rate (%)")
            axes[2].set_title("By Model Cycle")
        plt.tight_layout(); savefig("01_label_distribution.png")

    # ── 2. LIGHTNING TIMELINE ─────────────────────────────────────────────────
    if full_df is not None and "valid_time" in full_df.columns:
        df2=full_df.copy(); df2["valid_time"]=pd.to_datetime(df2["valid_time"])
        daily=df2.groupby(df2["valid_time"].dt.date)["label"].apply(
            lambda x:(x==1).mean()*100).reset_index()
        daily.columns=["date","rate"]; daily["date"]=pd.to_datetime(daily["date"])
        fig,ax=plt.subplots(figsize=(14,4))
        ax.fill_between(daily["date"],daily["rate"],alpha=0.5,color="#2196F3")
        ax.plot(daily["date"],daily["rate"],color="#1565C0",lw=2)
        ax.axhline(daily["rate"].mean(),color="red",ls="--",lw=1,
                   label=f"Mean={daily['rate'].mean():.1f}%")
        ax.set_xlabel("Date"); ax.set_ylabel("Daily Lightning Rate (%)")
        ax.set_title(f"Daily Lightning Occurrence Rate\n{full_ctx(f'n={len(full_df)} rows')}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.legend(); ax.grid(True,alpha=0.3); plt.xticks(rotation=45)
        plt.tight_layout(); savefig("02_lightning_timeline.png")

    # ── 3. CYCLE × FH HEATMAP ─────────────────────────────────────────────────
    if full_df is not None and "cycle_time" in full_df.columns:
        df3=full_df.copy(); df3["cycle"]=df3["cycle_time"].str[9:11]+"Z"
        piv=df3.pivot_table(values="label",index="cycle",columns="forecast_hour",
                            aggfunc=lambda x:(x==1).mean()*100)
        fig,ax=plt.subplots(figsize=(11,4))
        im=ax.imshow(piv.values,cmap="YlOrRd",aspect="auto",
                     vmin=0,vmax=max(piv.values.max(),1))
        plt.colorbar(im,ax=ax,label="Lightning Rate (%)")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"FH={c}" for c in piv.columns],fontsize=10)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index,fontsize=10)
        ax.set_xlabel("Forecast Hour"); ax.set_ylabel("HRRR Model Cycle")
        ax.set_title(f"Lightning Rate (%) by Cycle × Forecast Hour\n{full_ctx(f'n={len(full_df)}')}")
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                v=piv.values[i,j]
                ax.text(j,i,f"{v:.1f}%",ha="center",va="center",fontsize=9,
                        color="white" if v>15 else "black",fontweight="bold")
        plt.tight_layout(); savefig("03_lightning_by_cycle_fh.png")

    # ── Load models ───────────────────────────────────────────────────────────
    def load_model(name):
        p=models_dir/f"{name}_model.pkl"
        return pickle.load(open(p,"rb")) if p.exists() else None
    rf_bundle=load_model("rf"); xgb_bundle=load_model("xgb")
    feat_path=models_dir/"feature_names.json"
    if not feat_path.exists():
        print("  No feature_names.json. Run step 4 first."); return
    feature_names=json.load(open(feat_path))

    # ── 4 & 5. CONFUSION MATRICES — Observation / Prediction labels ──────────
    # Professor: "I would say Prediction and Observation"
    # Professor: "make it much smaller — it's just 4 numbers"
    if val_df is not None:
        X_val,y_val=get_Xy(val_df,feature_names)
        for mname,bundle in [("RF",rf_bundle),("XGB",xgb_bundle)]:
            if bundle is None: continue
            prob=bundle["model"].predict_proba(X_val)[:,1]
            pred=(prob>=bundle["threshold"]).astype(int)
            TP=int(np.sum((y_val==1)&(pred==1))); FP=int(np.sum((y_val==0)&(pred==1)))
            FN=int(np.sum((y_val==1)&(pred==0))); TN=int(np.sum((y_val==0)&(pred==0)))
            cm=np.array([[TN,FP],[FN,TP]])
            # Compute metrics directly — never rely on metrics.py version for these
            _pod  = TP/(TP+FN) if (TP+FN)>0 else 0.0
            _far  = FP/(TP+FP) if (TP+FP)>0 else 0.0
            _csi  = TP/(TP+FP+FN) if (TP+FP+FN)>0 else 0.0
            _pofd = FP/(FP+TN) if (FP+TN)>0 else 0.0
            _pss  = _pod - _pofd
            _n    = TP+FP+FN+TN
            _exp  = ((TP+FN)*(TP+FP)+(TN+FN)*(TN+FP))/_n if _n>0 else 0
            _hss  = (TP+TN-_exp)/(_n-_exp) if (_n-_exp)!=0 else 0.0
            fig,ax=plt.subplots(figsize=(5,4))   # SMALLER per Professor
            im=ax.imshow(cm,interpolation="nearest",cmap="Blues")
            plt.colorbar(im,ax=ax,shrink=0.8)
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["No Lightning","Lightning ⚡"],fontsize=10)
            ax.set_yticklabels(["No Lightning","Lightning ⚡"],fontsize=10)
            # PROFESSOR FIX: Observation / Prediction
            ax.set_xlabel(pconf["confusion_matrix_x_label"],fontsize=11)
            ax.set_ylabel(pconf["confusion_matrix_y_label"],fontsize=11)
            thresh=cm.max()/2
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=14,
                            fontweight="bold",
                            color="white" if cm[i,j]>thresh else "black")
            ax.set_title(
                f"{mname} — Confusion Matrix (Validation Set)\n"
                f"Hyperparams: {hp_str(mname,cfg)}\n"
                f"PSS={_pss:.3f}  CSI={_csi:.3f}  POD={_pod:.3f}  "
                f"FAR={_far:.3f}  HSS={_hss:.3f}  threshold={bundle['threshold']}\n"
                f"{full_ctx(f'val n={n_va}')}",
                fontsize=7.5)
            plt.tight_layout(); savefig(f"04_confusion_matrix_val_{mname.lower()}.png")

    # ── 6. ROC CURVES ─────────────────────────────────────────────────────────
    if val_df is not None:
        X_val,y_val=get_Xy(val_df,feature_names)
        if y_val.sum()>0:
            fig,ax=plt.subplots(figsize=(7,6))
            for mname,bundle in [("RF",rf_bundle),("XGB",xgb_bundle)]:
                if bundle is None: continue
                prob=bundle["model"].predict_proba(X_val)[:,1]
                try:
                    fpr,tpr,_,auc=get_roc_data(y_val,prob)
                    ax.plot(fpr,tpr,color=COLORS[mname],lw=2.5,
                            label=f"{mname}  AUC={auc:.3f}")
                    ax.fill_between(fpr,tpr,alpha=0.08,color=COLORS[mname])
                except Exception: pass
            ax.plot([0,1],[0,1],"k--",lw=1,label="Random (AUC=0.50)")
            ax.set_xlabel("False Positive Rate",fontsize=12)
            ax.set_ylabel("True Positive Rate",fontsize=12)
            ax.set_title(
                f"ROC Curves — Validation Set\n"
                f"RF: {hp_str('RF',cfg)}  |  XGB: {hp_str('XGB',cfg)}\n"
                f"{full_ctx(f'train n={n_tr}  val n={n_va}')}",
                fontsize=7.5)
            ax.legend(loc="lower right",fontsize=11)
            ax.grid(True,alpha=0.3); ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
            plt.tight_layout(); savefig("05_roc_curve_comparison.png")

    # ── 7. PR CURVES ──────────────────────────────────────────────────────────
    if val_df is not None and y_val.sum()>0:
        baseline=y_val.mean()
        fig,ax=plt.subplots(figsize=(7,6))
        for mname,bundle in [("RF",rf_bundle),("XGB",xgb_bundle)]:
            if bundle is None: continue
            prob=bundle["model"].predict_proba(X_val)[:,1]
            try:
                prec,rec,_,ap=get_pr_data(y_val,prob)
                ax.plot(rec,prec,color=COLORS[mname],lw=2.5,label=f"{mname}  AP={ap:.3f}")
                ax.fill_between(rec,prec,alpha=0.08,color=COLORS[mname])
            except Exception: pass
        ax.axhline(baseline,color="k",ls="--",lw=1,label=f"No-skill ({baseline:.2f})")
        ax.set_xlabel("Recall",fontsize=12); ax.set_ylabel("Precision",fontsize=12)
        ax.set_title(f"Precision-Recall Curves — Validation Set\n"
                     f"{full_ctx(f'val n={n_va} ({int(y_val.sum())} lightning)')}",fontsize=8)
        ax.legend(fontsize=11); ax.grid(True,alpha=0.3)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        plt.tight_layout(); savefig("06_pr_curve_comparison.png")

    # ── 8 & 9. FEATURE IMPORTANCES (top 30, with stability counts) ────────────
    for mname in ["rf","xgb"]:
        fi_path=(results_dir/f"feature_importance_{mname}_stable.csv"
                 if (results_dir/f"feature_importance_{mname}_stable.csv").exists()
                 else results_dir/f"feature_importance_{mname}.csv")
        if not fi_path.exists(): continue
        fi=pd.read_csv(fi_path).head(30)
        imp_col="mean_importance" if "mean_importance" in fi.columns else "importance"
        n_runs=cfg["models"]["random_forest"]["importance_runs"] if mname=="rf" else 1
        fig,ax=plt.subplots(figsize=(11,max(7,len(fi)*0.30)))
        color=COLORS["RF"] if mname=="rf" else COLORS["XGB"]
        bars=ax.barh(fi["feature"][::-1],fi[imp_col][::-1],color=color,alpha=0.85)
        for bar,row in zip(bars,fi[::-1].itertuples()):
            v=getattr(row,imp_col)
            if "top5_count" in fi.columns:
                label=f"{v:.4f}  [top5:{row.top5_count}/{n_runs}]"
            else:
                label=f"{v:.4f}"
            ax.text(bar.get_width()+0.0002,bar.get_y()+bar.get_height()/2,
                    label,va="center",fontsize=7)
        ax.set_xlabel("Feature Importance",fontsize=11)
        ax.set_title(
            f"{mname.upper()} — Top 30 Feature Importances\n"
            f"Hyperparams: {hp_str(mname,cfg)}\n"
            f"{'Stability: '+str(n_runs)+' runs | ' if mname=='rf' else ''}"
            f"{full_ctx(f'train n={n_tr}')}",fontsize=7.5)
        ax.grid(True,axis="x",alpha=0.3)
        plt.tight_layout()
        savefig(f"0{8 if mname=='rf' else 9}_feature_importance_{mname}_top30.png")

    # ── 10. VARIABLE GROUP IMPORTANCE (new — Professor suggestion) ────────────
    fi_rf_path=results_dir/"feature_importance_rf_stable.csv"
    if fi_rf_path.exists():
        fi_rf=pd.read_csv(fi_rf_path)
        if "group" not in fi_rf.columns:
            fi_rf["group"]=fi_rf["feature"].str.split("_").str[0]
        grp=group_feature_importance(fi_rf).head(15)
        fig,ax=plt.subplots(figsize=(9,5))
        ax.barh(grp["group"][::-1],grp["total_mean_importance"][::-1],
                color=COLORS["RF"],alpha=0.85)
        for i,(bar,row) in enumerate(zip(ax.patches,grp[::-1].itertuples())):
            ax.text(bar.get_width()+0.0002,bar.get_y()+bar.get_height()/2,
                    f"{row.total_mean_importance:.4f}  ({row.count_features} features)  "
                    f"[top5:{row.total_top5_count}/{cfg['models']['random_forest']['importance_runs']}]",
                    va="center",fontsize=8)
        ax.set_xlabel("Total Group Importance",fontsize=11)
        n_imp = cfg["models"]["random_forest"]["importance_runs"]
        ax.set_title(
            f"RF — Feature Importance by Variable Group (Top 15)\n"
            f"Groups combine correlated levels (e.g., RH+level clusters)\n"
            f"{pconf['region_label']}  |  {pconf['date_range']}  |  train n={n_tr}  |  {n_imp} runs",
            fontsize=8)
        ax.grid(True,axis="x",alpha=0.3)
        plt.tight_layout(); savefig("10_feature_group_importance_rf.png")

    # ── 11A. METRICS COMPARISON — HIGH IS BETTER (PSS, HSS, CSI, POD) ─────────
    # PROFESSOR: "never mix high-is-better and low-is-better"
    mpath=results_dir/"metrics_train_val_test.csv"
    if mpath.exists():
        mdf=pd.read_csv(mpath)
        val_rows=mdf[mdf["split"]=="val"]
        paper=cfg.get("paper_baseline",{})

        # Chart A: high is better
        # Only show metrics that exist in the data (handles old metrics.py without PSS)
        _all_hi = ["PSS","HSS","CSI","POD"]
        _all_hi_labels = ["PSS\n(Peirce Skill Score)","HSS\n(Heidke Skill Score)",
                          "CSI\n(Threat Score)","POD\n(Hit Rate)"]
        hi_metrics = [m for m in _all_hi if m in mdf.columns]
        hi_labels  = [l for m,l in zip(_all_hi,_all_hi_labels) if m in mdf.columns]
        x=np.arange(len(hi_metrics)); width=0.25
        fig,ax=plt.subplots(figsize=(12,6))
        model_colors=[COLORS["RF"],COLORS["XGB"],COLORS["Paper"]]
        rows_to_plot=list(val_rows.iterrows())
        for i,(_,row) in enumerate(rows_to_plot):
            vals=[row.get(c,0) for c in hi_metrics]
            bars=ax.bar(x+(i-len(rows_to_plot)/2+0.5)*width,vals,width,
                        label=row["model"],color=model_colors[i%2],alpha=0.85)
            for bar,v in zip(bars,vals):
                if v and not np.isnan(float(v)):
                    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                            f"{v:.2f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
        # Add paper baseline if available
        if paper.get("PSS") or paper.get("HSS") or paper.get("CSI") or paper.get("POD"):
            paper_vals=[paper.get(m,None) for m in hi_metrics]
            if any(v is not None for v in paper_vals):
                bars=ax.bar(x+(len(rows_to_plot)-len(rows_to_plot)/2+0.5)*width,
                            [v or 0 for v in paper_vals],width,
                            label=paper.get("name","2015 Paper"),
                            color=COLORS["Paper"],alpha=0.7,hatch="//")
                for bar,v in zip(bars,paper_vals):
                    if v:
                        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                                f"{v:.2f}",ha="center",va="bottom",fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(hi_labels,fontsize=11)
        ax.set_ylim(0,1.18); ax.set_ylabel("Score",fontsize=12)
        ax.set_title(
            f"Model Performance — Validation Set  (↑ Higher is Better)\n"
            f"RF: {hp_str('RF',cfg)}  |  XGB: {hp_str('XGB',cfg)}\n"
            f"{full_ctx(f'train n={n_tr}  val n={n_va}  test n={n_te}')}",fontsize=7.5)
        ax.legend(fontsize=11); ax.grid(True,axis="y",alpha=0.3)
        ax.axhline(1.0,color="gray",ls="--",lw=0.8,alpha=0.5)
        plt.tight_layout(); savefig("11a_metrics_high_is_better.png")

        # Chart B: lower is better (FAR, POFD, Bias target=1)
        _all_lo = ["FAR","POFD","Bias"]
        _all_lo_labels = ["FAR\n(False Alarm Ratio)\n↓ lower is better",
                          "POFD\n(False Detection Rate)\n↓ lower is better",
                          "Bias\n(Frequency Bias)\n→ 1.0 is ideal"]
        lo_metrics = [m for m in _all_lo if m in mdf.columns]
        lo_labels  = [l for m,l in zip(_all_lo,_all_lo_labels) if m in mdf.columns]
        x2=np.arange(len(lo_metrics))
        fig,ax=plt.subplots(figsize=(9,6))
        for i,(_,row) in enumerate(rows_to_plot):
            vals=[row.get(c,0) for c in lo_metrics]
            bars=ax.bar(x2+(i-len(rows_to_plot)/2+0.5)*width,vals,width,
                        label=row["model"],color=model_colors[i%2],alpha=0.85)
            for bar,v in zip(bars,vals):
                if v and not np.isnan(float(v)):
                    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                            f"{v:.2f}",ha="center",va="bottom",fontsize=9,fontweight="bold")
        # Add ideal lines
        ax.axhline(0,color="green",ls="--",lw=1.5,alpha=0.7,label="FAR/POFD ideal (0)")
        ax.axhline(1,color="orange",ls="--",lw=1.5,alpha=0.7,label="Bias ideal (1)")
        ax.set_xticks(x2); ax.set_xticklabels(lo_labels,fontsize=10)
        ax.set_ylim(0,max(2,mdf["Bias"].max()*1.2 if "Bias" in mdf.columns else 2))
        ax.set_ylabel("Score",fontsize=12)
        ax.set_title(
            f"Model Performance — Validation Set  (↓ Lower is Better / Bias→1)\n"
            f"RF: {hp_str('RF',cfg)}  |  XGB: {hp_str('XGB',cfg)}\n"
            f"{full_ctx(f'val n={n_va}')}",fontsize=7.5)
        ax.legend(fontsize=10); ax.grid(True,axis="y",alpha=0.3)
        plt.tight_layout(); savefig("11b_metrics_low_is_better.png")

    total=len(list(plots_dir.glob("*.png")))
    print(f"\n  All {total} plots saved → {plots_dir}")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()