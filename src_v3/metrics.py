"""
================================================================================
utils/metrics.py — Meteorological + ML evaluation metrics  (v3)
================================================================================
v3 changes (Professor Tissot meeting feedback):
  - Added PSS (Peirce Skill Score) — preferred over F1/AUC in lightning papers
  - Added POFD (Probability of False Detection)
  - track_feature_importance: default runs=30 (Professor: "do 30 times")
  - Added group_feature_importance: counts "any RH in top-N" per variable group

Metrics hierarchy per Professor Tissot:
  PRIMARY:   PSS, HSS (skill scores — compare vs random)
  SECONDARY: CSI, POD, FAR (threat/hit/alarm rates)
  AVOID:     F1, AUC-ROC (less common in meteorological literature)
================================================================================
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, accuracy_score
)


def compute_all_metrics(y_true, y_pred, y_prob=None, threshold=0.5) -> dict:
    """
    Compute full set of binary classification metrics including
    PSS, HSS, CSI, POD, FAR, POFD, Bias, F1, AUC.

    PSS = Peirce Skill Score = POD - POFD  (preferred in lightning papers)
    Range: -1 to +1  (1=perfect, 0=no skill, <0=worse than random)
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    n = max(len(y_true), 1)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    # ── Meteorological scores ────────────────────────────────────────────────
    pod  = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # hit rate
    pofd = FP / (FP + TN) if (FP + TN) > 0 else 0.0   # false detection rate
    far  = FP / (TP + FP) if (TP + FP) > 0 else 0.0   # false alarm ratio
    csi  = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0  # threat score
    bias = (TP + FP) / (TP + FN) if (TP + FN) > 0 else 0.0

    # PSS (Peirce Skill Score) = POD - POFD  [preferred over F1 in met papers]
    pss  = pod - pofd

    # HSS (Heidke Skill Score)
    exp_correct = ((TP + FN) * (TP + FP) + (TN + FN) * (TN + FP)) / n
    hss = (TP + TN - exp_correct) / (n - exp_correct) if (n - exp_correct) != 0 else 0.0

    # ── Standard ML scores ───────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)

    m = {
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
        "N_POS": n_pos, "N_NEG": n_neg,
        "class_balance": round(n_pos / n, 4),
        "accuracy":  round(acc,  4),
        "f1_score":  round(f1,   4),
        "POD":  round(pod,  4),   # Probability of Detection (recall)
        "POFD": round(pofd, 4),   # Probability of False Detection
        "FAR":  round(far,  4),   # False Alarm Ratio
        "CSI":  round(csi,  4),   # Critical Success Index (Threat Score)
        "PSS":  round(pss,  4),   # Peirce Skill Score (POD - POFD) ← PRIMARY
        "HSS":  round(hss,  4),   # Heidke Skill Score ← PRIMARY
        "Bias": round(bias, 4),
        "threshold": threshold,
    }

    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            m["AUC_ROC"] = round(roc_auc_score(y_true, y_prob), 4)
            m["AP"]      = round(average_precision_score(y_true, y_prob), 4)
        except Exception:
            m["AUC_ROC"] = None
            m["AP"]      = None

    return m


def print_metrics_table(metrics: dict, title: str = ""):
    """Pretty-print a metrics dict — PSS and HSS highlighted as primary."""
    print(f"\n{'='*58}")
    if title:
        print(f"  {title}")
    print(f"{'='*58}")
    print(f"  n={metrics['N_POS']+metrics['N_NEG']}  "
          f"lightning={metrics['N_POS']}  ({metrics['class_balance']:.1%})")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}  "
          f"FN={metrics['FN']}  TN={metrics['TN']}")
    print(f"  {'─'*52}")
    print(f"  ★ PSS  = {metrics['PSS']:.4f}   (Peirce Skill Score — primary)")
    print(f"  ★ HSS  = {metrics['HSS']:.4f}   (Heidke Skill Score — primary)")
    print(f"  {'─'*52}")
    print(f"    CSI  = {metrics['CSI']:.4f}   (Threat Score)")
    print(f"    POD  = {metrics['POD']:.4f}   (Hit Rate / Recall)")
    print(f"    POFD = {metrics['POFD']:.4f}   (False Detection Rate)")
    print(f"    FAR  = {metrics['FAR']:.4f}   (False Alarm Ratio — lower is better)")
    print(f"    Bias = {metrics['Bias']:.4f}   (Frequency Bias — 1=unbiased)")
    print(f"  {'─'*52}")
    print(f"    F1   = {metrics['f1_score']:.4f}   Accuracy={metrics['accuracy']:.4f}")
    if metrics.get("AUC_ROC") is not None:
        print(f"    AUC  = {metrics['AUC_ROC']:.4f}")
    print(f"{'='*58}")


def get_roc_data(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return fpr, tpr, thr, roc_auc_score(y_true, y_prob)


def get_pr_data(y_true, y_prob):
    p, r, thr = precision_recall_curve(y_true, y_prob)
    return p, r, thr, average_precision_score(y_true, y_prob)


def track_feature_importance(model, X, y, feature_names, n_runs=30,
                              random_state=42) -> "pd.DataFrame":
    """
    Run RF feature importance n_runs times (Professor: "do 30 times").
    Track top-5 and top-10 counts per feature.
    Also builds variable-group counts (e.g., "any RH level in top-N").

    Returns DataFrame: [feature, mean_importance, std_importance,
                        top5_count, top10_count, group]
    """
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    top5_counts  = {f: 0 for f in feature_names}
    top10_counts = {f: 0 for f in feature_names}
    importances  = {f: [] for f in feature_names}

    for i in range(n_runs):
        params = model.get_params()
        params["random_state"] = random_state + i
        clone = RandomForestClassifier(**params)
        clone.fit(X, y)
        fi = dict(zip(feature_names, clone.feature_importances_))
        ranked = sorted(fi.items(), key=lambda x: -x[1])
        for rank, (feat, imp) in enumerate(ranked):
            importances[feat].append(imp)
            if rank < 5:  top5_counts[feat]  += 1
            if rank < 10: top10_counts[feat] += 1

    rows = []
    for f in feature_names:
        # Determine variable group (e.g., "RH" from "RH_15")
        group = f.split("_")[0] if "_" in f else f
        rows.append({
            "feature":         f,
            "group":           group,
            "mean_importance": round(float(np.mean(importances[f])), 6),
            "std_importance":  round(float(np.std(importances[f])),  6),
            "top5_count":      top5_counts[f],
            "top10_count":     top10_counts[f],
        })

    df = pd.DataFrame(rows).sort_values("mean_importance", ascending=False)
    return df.reset_index(drop=True)


def group_feature_importance(fi_df) -> "pd.DataFrame":
    """
    Aggregate feature importance by variable group.
    e.g., combines RH_13, RH_14, RH_15 → group 'RH'
    Shows which physical variable types matter most.
    Professor: "you might see that RH always has one in there"
    """
    import pandas as pd
    grp = fi_df.groupby("group").agg(
        total_mean_importance=("mean_importance", "sum"),
        count_features=("feature", "count"),
        total_top5_count=("top5_count", "sum"),
        total_top10_count=("top10_count", "sum"),
        best_feature=("feature", "first"),
    ).sort_values("total_mean_importance", ascending=False)
    return grp.reset_index()