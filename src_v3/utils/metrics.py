"""
utils/metrics.py — Meteorological + ML evaluation metrics  (v2)
Includes: POD, FAR, CSI, HSS, Bias, AUC-ROC, F1, accuracy
Also: feature importance stability tracking (Professor Tissot feedback)
"""
import numpy as np
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score, accuracy_score)

def compute_all_metrics(y_true, y_pred, y_prob=None, threshold=0.5):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    TN,FP,FN,TP = cm.ravel() if cm.size==4 else (0,0,0,0)
    n = max(len(y_true), 1)
    n_pos = int(np.sum(y_true==1)); n_neg = int(np.sum(y_true==0))

    pod  = TP/(TP+FN) if (TP+FN)>0 else 0.0
    far  = FP/(TP+FP) if (TP+FP)>0 else 0.0
    csi  = TP/(TP+FP+FN) if (TP+FP+FN)>0 else 0.0
    bias = (TP+FP)/(TP+FN) if (TP+FN)>0 else 0.0
    expected = ((TP+FN)*(TP+FP)+(TN+FN)*(TN+FP))/n
    hss = (TP+TN-expected)/(n-expected) if (n-expected)!=0 else 0.0

    m = {
        "TP":int(TP),"FP":int(FP),"FN":int(FN),"TN":int(TN),
        "N_POS":n_pos,"N_NEG":n_neg,
        "class_balance": round(n_pos/n, 4),
        "accuracy":      round(accuracy_score(y_true,y_pred), 4),
        "f1_score":      round(f1_score(y_true,y_pred,zero_division=0), 4),
        "POD":  round(pod,4), "FAR":  round(far,4),
        "CSI":  round(csi,4), "HSS":  round(hss,4),
        "Bias": round(bias,4), "threshold": threshold,
    }
    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            m["AUC_ROC"] = round(roc_auc_score(y_true,y_prob), 4)
            m["AP"]      = round(average_precision_score(y_true,y_prob), 4)
        except Exception:
            m["AUC_ROC"] = None; m["AP"] = None
    return m

def print_metrics_table(metrics: dict, title: str = ""):
    print(f"\n{'='*55}")
    if title: print(f"  {title}")
    print(f"{'='*55}")
    print(f"  n={metrics['N_POS']+metrics['N_NEG']}  lightning={metrics['N_POS']}  ({metrics['class_balance']:.1%})")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics['FN']}  TN={metrics['TN']}")
    print(f"  {'─'*50}")
    print(f"  Accuracy={metrics['accuracy']:.4f}  F1={metrics['f1_score']:.4f}")
    print(f"  POD={metrics['POD']:.4f}  FAR={metrics['FAR']:.4f}  CSI={metrics['CSI']:.4f}")
    print(f"  HSS={metrics['HSS']:.4f}  Bias={metrics['Bias']:.4f}")
    if metrics.get("AUC_ROC") is not None:
        print(f"  AUC-ROC={metrics['AUC_ROC']:.4f}  AP={metrics['AP']:.4f}")
    print(f"{'='*55}")

def get_roc_data(y_true, y_prob):
    fpr,tpr,thr = roc_curve(y_true,y_prob)
    return fpr, tpr, thr, roc_auc_score(y_true,y_prob)

def get_pr_data(y_true, y_prob):
    p,r,thr = precision_recall_curve(y_true,y_prob)
    return p, r, thr, average_precision_score(y_true,y_prob)

def track_feature_importance(model, X, y, feature_names, n_runs=10, random_state=42):
    """
    Run feature importance n_runs times with different random seeds.
    Track top-5 count and top-10 count per feature.
    Professor Tissot feedback: correlation between features means
    importance is unstable — track counts across runs.
    Returns: DataFrame with columns [feature, mean_importance, top5_count, top10_count]
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    top5_counts  = {f: 0 for f in feature_names}
    top10_counts = {f: 0 for f in feature_names}
    importances  = {f: [] for f in feature_names}

    for i in range(n_runs):
        clone = RandomForestClassifier(**{**model.get_params(), "random_state": random_state+i})
        clone.fit(X, y)
        fi = dict(zip(feature_names, clone.feature_importances_))
        sorted_feats = sorted(fi.items(), key=lambda x: -x[1])
        for rank, (feat, imp) in enumerate(sorted_feats):
            importances[feat].append(imp)
            if rank < 5:  top5_counts[feat]  += 1
            if rank < 10: top10_counts[feat] += 1

    rows = []
    for f in feature_names:
        rows.append({
            "feature":        f,
            "mean_importance": round(float(np.mean(importances[f])), 6),
            "std_importance":  round(float(np.std(importances[f])), 6),
            "top5_count":      top5_counts[f],
            "top10_count":     top10_counts[f],
        })

    df = pd.DataFrame(rows).sort_values("mean_importance", ascending=False)
    return df.reset_index(drop=True)


def group_feature_importance(fi_df) -> "pd.DataFrame":
    """
    Aggregate feature importance by variable group.
    e.g., combines RH_13, RH_14, RH_15 -> group RH
    Professor: "you might see that RH always has one in there"
    """
    import pandas as pd
    if "group" not in fi_df.columns:
        fi_df = fi_df.copy()
        fi_df["group"] = fi_df["feature"].str.split("_").str[0]
    imp_col = "mean_importance" if "mean_importance" in fi_df.columns else "importance"
    grp = fi_df.groupby("group").agg(
        total_mean_importance=(imp_col, "sum"),
        count_features=("feature", "count"),
        total_top5_count=("top5_count", "sum") if "top5_count" in fi_df.columns else ("feature", "count"),
        total_top10_count=("top10_count", "sum") if "top10_count" in fi_df.columns else ("feature", "count"),
        best_feature=("feature", "first"),
    ).sort_values("total_mean_importance", ascending=False)
    return grp.reset_index()
