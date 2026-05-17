"""
Task-specific metrics: classification + regression.

Also provides the per-task score S_k and the overall score S_Overall
as defined in the paper (Eqs. 19 and 20):

    S_SST-2 = 1/2 (Acc + F1)
    S_QQP   = 1/2 (Acc + F1)
    S_STS-B = 1/2 (Pearson + Spearman)
    S_Overall = 1/3 (S_SST-2 + S_QQP + S_STS-B)
"""
from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)
from scipy.stats import pearsonr, spearmanr


# ---------------------------------------------------------------------------
# Per-task raw metrics
# ---------------------------------------------------------------------------
def compute_classification_metrics(preds, labels) -> Dict[str, float]:
    preds = np.asarray(preds); labels = np.asarray(labels)
    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall":    float(recall_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1":  float(f1_score(labels, preds, average="macro", zero_division=0)),
        "pearson":   float("nan"),
        "spearman":  float("nan"),
    }


def compute_regression_metrics(preds, labels) -> Dict[str, float]:
    preds = np.asarray(preds); labels = np.asarray(labels)
    p = pearsonr(preds, labels)[0] if len(preds) > 1 else 0.0
    s = spearmanr(preds, labels)[0] if len(preds) > 1 else 0.0
    return {
        "pearson":  float(p) if p == p else 0.0,
        "spearman": float(s) if s == s else 0.0,
        "accuracy": float("nan"),
        "precision": float("nan"),
        "recall":    float("nan"),
        "macro_f1":  float("nan"),
    }


# ---------------------------------------------------------------------------
# Per-task score S_k (Eq. 19)
# ---------------------------------------------------------------------------
def task_score(task_key: str, metrics: Dict[str, float]) -> float:
    """Return the per-task score S_k as defined in Eq. 19 of the paper."""
    if task_key in ("sst2", "qqp"):
        acc = metrics.get("accuracy", float("nan"))
        f1  = metrics.get("macro_f1", float("nan"))
        return 0.5 * (acc + f1)
    if task_key == "stsb":
        p = metrics.get("pearson",  float("nan"))
        s = metrics.get("spearman", float("nan"))
        return 0.5 * (p + s)
    raise ValueError(f"Unknown task: {task_key}")


def overall_score(per_task_metrics: Dict[str, Dict[str, float]]) -> float:
    """
    Compute S_Overall = (1/K) * sum_k S_k  (Eq. 20).
    `per_task_metrics` maps task_key -> metric dict.
    """
    scores = []
    for tk, m in per_task_metrics.items():
        scores.append(task_score(tk, m))
    if not scores:
        return float("nan")
    return float(np.mean(scores))
