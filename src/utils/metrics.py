"""
================================================================================
utils/metrics.py — Meteorological + ML evaluation metrics
================================================================================
Standard metrics for binary storm prediction:
  POD  = Probability of Detection (recall for class=1)
  FAR  = False Alarm Ratio
  CSI  = Critical Success Index (Threat Score)
  HSS  = Heidke Skill Score
  Bias = Frequency Bias
  AUC  = Area Under ROC Curve
================================================================================
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score
)


def compute_all_metrics(y_true, y_pred, y_prob=None, threshold=0.5):
    """
    Compute full set of binary classification metrics including
    meteorological skill scores.

    Parameters
    ----------
    y_true  : array-like of 0/1
    y_pred  : array-like of 0/1  (hard predictions)
    y_prob  : array-like of floats in [0,1]  (probability of class=1), optional
    threshold : decision threshold used for y_pred

    Returns
    -------
    dict of metric_name → value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # ── Meteorological scores ────────────────────────────────────────────────
    # POD (Probability of Detection) = TP / (TP + FN)
    pod  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    # FAR (False Alarm Ratio) = FP / (TP + FP)
    far  = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    # CSI (Critical Success Index / Threat Score) = TP / (TP + FP + FN)
    csi  = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    # Frequency Bias = (TP + FP) / (TP + FN)
    bias = (TP + FP) / (TP + FN) if (TP + FN) > 0 else 0.0

    # HSS (Heidke Skill Score)
    expected_correct = ((TP + FN) * (TP + FP) + (TN + FN) * (TN + FP)) / max(len(y_true), 1)
    total_correct    = TP + TN
    hss = (total_correct - expected_correct) / (len(y_true) - expected_correct) \
          if (len(y_true) - expected_correct) != 0 else 0.0

    # ── Standard ML scores ───────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)

    # Class counts
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    metrics = {
        "TP":        int(TP),
        "FP":        int(FP),
        "FN":        int(FN),
        "TN":        int(TN),
        "N_POS":     n_pos,
        "N_NEG":     n_neg,
        "class_balance": n_pos / len(y_true) if len(y_true) > 0 else 0,
        "accuracy":  round(acc,  4),
        "f1_score":  round(f1,   4),
        "POD":       round(pod,  4),
        "FAR":       round(far,  4),
        "CSI":       round(csi,  4),
        "Bias":      round(bias, 4),
        "HSS":       round(hss,  4),
        "threshold": threshold,
    }

    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            auc = roc_auc_score(y_true, y_prob)
            ap  = average_precision_score(y_true, y_prob)
            metrics["AUC_ROC"] = round(auc, 4)
            metrics["AP"]      = round(ap,  4)
        except Exception:
            metrics["AUC_ROC"] = None
            metrics["AP"]      = None

    return metrics


def print_metrics_table(metrics: dict, title: str = ""):
    """Pretty-print a metrics dict."""
    print(f"\n{'='*55}")
    if title:
        print(f"  {title}")
    print(f"{'='*55}")
    print(f"  Samples:  {metrics['N_POS']} lightning  /  {metrics['N_NEG']} no-lightning")
    print(f"  Class balance: {metrics['class_balance']:.1%} positive")
    print(f"  {'─'*50}")
    print(f"  Confusion Matrix:  TP={metrics['TP']}  FP={metrics['FP']}  "
          f"FN={metrics['FN']}  TN={metrics['TN']}")
    print(f"  {'─'*50}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Score : {metrics['f1_score']:.4f}")
    print(f"  {'─'*50}")
    print(f"  POD (Recall) : {metrics['POD']:.4f}   (hit rate)")
    print(f"  FAR          : {metrics['FAR']:.4f}   (false alarm rate)")
    print(f"  CSI          : {metrics['CSI']:.4f}   (threat score)")
    print(f"  HSS          : {metrics['HSS']:.4f}   (skill score, 1=perfect)")
    print(f"  Bias         : {metrics['Bias']:.4f}   (1=unbiased)")
    if "AUC_ROC" in metrics and metrics["AUC_ROC"] is not None:
        print(f"  AUC-ROC      : {metrics['AUC_ROC']:.4f}")
        print(f"  Avg Prec.    : {metrics['AP']:.4f}")
    print(f"{'='*55}")


def get_roc_data(y_true, y_prob):
    """Return (fpr, tpr, thresholds, auc) for ROC curve plotting."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    return fpr, tpr, thresholds, auc


def get_pr_data(y_true, y_prob):
    """Return (precision, recall, thresholds, ap) for PR curve plotting."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return precision, recall, thresholds, ap
