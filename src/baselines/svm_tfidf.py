"""
SVM + TF-IDF baseline (STL): per-task.

Captures the same metric panel as other STL experiments (with VRAM=0 since CPU).
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import scipy.sparse as sp
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, hinge_loss)
from scipy.stats import pearsonr, spearmanr

from ..common.config import TASK_CONFIGS
from ..common.logger import build_logger, HistoryWriter
from ..common.metrics import task_score
from ..common.utils import set_seed, get_hardware_info


def pair_features(Xa, Xb):
    diff = (Xa - Xb); diff.data = np.abs(diff.data)
    prod = Xa.multiply(Xb)
    return sp.hstack([Xa, Xb, diff, prod]).tocsr()


def run_svm(
    task_key: str,
    output_dir: str,
    max_features: int = 200_000,
    ngram_max: int = 2,
    C: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    set_seed(seed)
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    logger = build_logger(f"svm-{task_key}", str(out_dir / "train.log"))
    history = HistoryWriter(str(out_dir))
    logger.info(f"===== SVM+TFIDF | task={task_key} =====")
    logger.info(f"Hardware: {get_hardware_info()}")

    tm = TASK_CONFIGS[task_key]
    is_reg = tm["task_type"] == "regression"

    raw = load_dataset(*tm["hf_name"])
    train_a = list(raw["train"][tm["text_a"]])
    val_a   = list(raw["validation"][tm["text_a"]])
    train_b = list(raw["train"][tm["text_b"]]) if tm["text_b"] else None
    val_b   = list(raw["validation"][tm["text_b"]]) if tm["text_b"] else None
    y_train = np.array(raw["train"]["label"])
    y_val   = np.array(raw["validation"]["label"])

    vec = TfidfVectorizer(ngram_range=(1, ngram_max), max_features=max_features,
                          sublinear_tf=True, lowercase=True)
    if tm["text_b"]:
        vec.fit(train_a + train_b)
        X_train = pair_features(vec.transform(train_a), vec.transform(train_b))
        X_val   = pair_features(vec.transform(val_a),   vec.transform(val_b))
    else:
        vec.fit(train_a)
        X_train = vec.transform(train_a); X_val = vec.transform(val_a)
    logger.info(f"X_train={X_train.shape} | X_val={X_val.shape}")

    t0 = time.perf_counter()
    if is_reg:
        clf = LinearSVR(C=C, max_iter=10000, random_state=seed)
        clf.fit(X_train, y_train.astype(np.float32))
    else:
        clf = LinearSVC(C=C, max_iter=10000, random_state=seed, class_weight="balanced")
        clf.fit(X_train, y_train.astype(int))
    fit_time = time.perf_counter() - t0

    train_params = int(np.prod(clf.coef_.shape) +
                       np.prod(np.atleast_1d(clf.intercept_).shape))
    total_params = train_params
    logger.info(f"fit_time={fit_time:.2f}s params={train_params:,}")

    if is_reg:
        train_pred = clf.predict(X_train)
        train_loss = float(np.mean((train_pred - y_train) ** 2))
        val_pred = clf.predict(X_val)
        eval_loss = float(np.mean((val_pred - y_val) ** 2))
        p = pearsonr(val_pred, y_val)[0] if len(val_pred) > 1 else 0.0
        s = spearmanr(val_pred, y_val)[0] if len(val_pred) > 1 else 0.0
        metrics = {"eval_loss": eval_loss,
                   "pearson":  float(p) if p == p else 0.0,
                   "spearman": float(s) if s == s else 0.0,
                   "accuracy": float("nan"), "precision": float("nan"),
                   "recall":   float("nan"), "macro_f1":  float("nan")}
    else:
        train_pred = clf.predict(X_train)
        try:
            train_loss = float(hinge_loss(y_train, clf.decision_function(X_train),
                                          labels=np.unique(y_train)))
        except Exception:
            train_loss = float("nan")
        val_pred = clf.predict(X_val)
        try:
            eval_loss = float(hinge_loss(y_val, clf.decision_function(X_val),
                                         labels=np.unique(y_train)))
        except Exception:
            eval_loss = float("nan")
        metrics = {"eval_loss": eval_loss,
                   "accuracy":  float(accuracy_score(y_val, val_pred)),
                   "precision": float(precision_score(y_val, val_pred, average="macro", zero_division=0)),
                   "recall":    float(recall_score(y_val, val_pred,    average="macro", zero_division=0)),
                   "macro_f1":  float(f1_score(y_val, val_pred,        average="macro", zero_division=0)),
                   "pearson":   float("nan"), "spearman": float("nan")}
    sk = task_score(task_key, metrics)

    # inference benchmark (CPU)
    t0 = time.perf_counter()
    _ = clf.predict(X_val)
    inf_elapsed = time.perf_counter() - t0
    n_eval = X_val.shape[0]

    record = {"exp_name": f"svm_{task_key}", "mode": "STL", "setting": "svm_tfidf",
              "model": "svm_tfidf", "task": task_key, "epoch": 1,
              "train_loss": train_loss, "time_per_epoch": fit_time,
              "trainable_params": train_params, "total_params": total_params,
              "task_score": sk, **metrics}
    history.append(record)
    joblib.dump({"vectorizer": vec, "model": clf, "task": task_key,
                 "is_pair": tm["text_b"] is not None},
                out_dir / "best_model.joblib")
    joblib.dump({"vectorizer": vec, "model": clf, "task": task_key,
                 "is_pair": tm["text_b"] is not None},
                out_dir / "final_model.joblib")

    primary = tm["primary_metric"]
    summary = {
        "exp_name": f"svm_{task_key}", "mode": "STL", "setting": "svm_tfidf",
        "model": "svm_tfidf", "task": task_key,
        "best_epoch": 1, "best_metric_name": primary,
        f"best_{primary}": metrics[primary],
        "accuracy":  metrics.get("accuracy"),
        "macro_f1":  metrics.get("macro_f1"),
        "precision": metrics.get("precision"),
        "recall":    metrics.get("recall"),
        "pearson":   metrics.get("pearson"),
        "spearman":  metrics.get("spearman"),
        "eval_loss": metrics.get("eval_loss"),
        "task_score": sk,
        "overall_score": sk,
        "total_params":     total_params,
        "trainable_params": train_params,
        "total_train_time_sec": fit_time,
        "avg_time_per_epoch_sec": fit_time,
        "epochs_completed": 1,
        "train_peak_vram_mb": 0.0,    # CPU
        "inference_peak_vram_mb": 0.0,
        "inference_latency_ms_per_sample": (inf_elapsed / max(n_eval, 1)) * 1000.0,
        "inference_throughput_samples_per_sec": n_eval / max(inf_elapsed, 1e-9),
        "inference_benchmark_samples": n_eval,
        **get_hardware_info(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Summary: {summary}")
    return summary
