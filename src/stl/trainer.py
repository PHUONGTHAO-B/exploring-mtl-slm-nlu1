"""
STL trainer: full fine-tuning of one SLM on one GLUE task.

Captures *all* metrics required by the paper:
  - Accuracy / Macro-F1   (SST-2, QQP)
  - Pearson / Spearman    (STS-B)
  - Task score S_k         (Eq. 19)
  - Overall score          (Eq. 20)
  - total_params, trainable_params
  - time_per_epoch, total_train_time
  - train_peak_vram_mb
  - inference_latency_ms_per_sample, throughput_samples_per_sec
  - inference_peak_vram_mb
  - GPU name, CUDA version, hardware metadata
"""
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from ..common.config import ExperimentConfig, TASK_CONFIGS
from ..common.data import build_tokenizer, load_glue_loaders, filter_model_kwargs
from ..common.logger import build_logger, HistoryWriter
from ..common.checkpoint import CheckpointManager
from ..common.benchmark import benchmark_inference, count_parameters
from ..common.metrics import (compute_classification_metrics,
                              compute_regression_metrics,
                              task_score, overall_score)
from ..common.utils import set_seed, get_device, get_hardware_info
from .model import build_stl_model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_stl(model, loader, device, task_meta, use_fp16: bool) -> Dict[str, float]:
    model.eval()
    losses, preds, labels = [], [], []
    is_reg = task_meta["task_type"] == "regression"

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        y = batch["labels"]
        inputs = filter_model_kwargs(model, {k: v for k, v in batch.items() if k != "labels"})
        if use_fp16 and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(**inputs, labels=y)
        else:
            out = model(**inputs, labels=y)
        losses.append(out.loss.item())
        logits = out.logits.float()
        if is_reg:
            preds.append(logits.squeeze(-1).detach().cpu().numpy())
        else:
            preds.append(torch.argmax(logits, dim=-1).detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    preds  = np.concatenate(preds)
    labels = np.concatenate(labels)
    eval_loss = float(np.mean(losses)) if losses else float("nan")

    if is_reg:
        m = compute_regression_metrics(preds, labels)
    else:
        m = compute_classification_metrics(preds, labels)
    m["eval_loss"] = eval_loss
    return m


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------
def _train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                     cfg: ExperimentConfig, logger, epoch: int) -> float:
    model.train()
    total_loss, n_steps = 0.0, 0
    for step, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        y = batch["labels"]
        inputs = filter_model_kwargs(model, {k: v for k, v in batch.items() if k != "labels"})

        optimizer.zero_grad(set_to_none=True)
        if cfg.fp16 and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(**inputs, labels=y); loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            out = model(**inputs, labels=y); loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        scheduler.step()
        total_loss += loss.item(); n_steps += 1
        if cfg.log_steps and step > 0 and step % cfg.log_steps == 0:
            logger.info(f"epoch {epoch} | step {step}/{len(loader)} "
                        f"| train_loss {total_loss/n_steps:.4f} "
                        f"| lr {scheduler.get_last_lr()[0]:.2e}")
    return total_loss / max(n_steps, 1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_stl(cfg: ExperimentConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = get_device()
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(cfg.name, str(out_dir / "train.log"))

    # ---- skip if already done ----
    mgr = CheckpointManager(str(out_dir), keep_n=cfg.keep_n_ckpts)
    if mgr.is_done():
        summary = mgr.load_summary()
        logger.info(f"[SKIP] {cfg.name} already done.")
        return summary

    logger.info(f"===== STL: {cfg.name} =====")
    logger.info(f"Config: {json.dumps(cfg.to_dict(), indent=2, default=str)}")
    logger.info(f"Hardware: {get_hardware_info()}")
    (out_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, default=str), encoding="utf-8")

    # ---- data ----
    tokenizer = build_tokenizer(cfg.model_id)
    train_loader, val_loader = load_glue_loaders(
        task_key=cfg.task_key, tokenizer=tokenizer,
        batch_size=cfg.batch_size, eval_batch_size=cfg.eval_batch_size,
        max_seq_len=cfg.max_seq_len, num_workers=cfg.num_workers,
        seed=cfg.seed,
        max_train_samples=cfg.max_train_samples,
        max_eval_samples=cfg.max_eval_samples,
    )
    logger.info(f"train batches={len(train_loader)} | val batches={len(val_loader)}")

    # ---- model ----
    model = build_stl_model(cfg.model_id, cfg.task_key).to(device)
    pcount = count_parameters(model)
    logger.info(f"trainable={pcount['trainable_params']:,} total={pcount['total_params']:,}")

    # ---- optimizer / scheduler / scaler ----
    no_decay = ("bias", "LayerNorm.weight")
    pg = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(pg, lr=cfg.learning_rate)
    total_steps = len(train_loader) * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.fp16 and device.type == "cuda"))

    history = HistoryWriter(str(out_dir))
    task_meta = TASK_CONFIGS[cfg.task_key]
    primary = task_meta["primary_metric"]

    # ---- resume ----
    start_epoch = 1; best_metric = -float("inf"); patience = 0
    if cfg.resume:
        latest = mgr.latest_checkpoint()
        if latest:
            ck = mgr.load_checkpoint(latest, model, optimizer, scheduler, scaler,
                                     map_location=device.type)
            start_epoch = ck["epoch"] + 1
            best_metric = ck.get("best_metric", -float("inf"))
            patience = ck.get("extra", {}).get("patience", 0)
            logger.info(f"[RESUME] from epoch {start_epoch}, best={best_metric:.4f}")

    # ---- TRAIN: measure train peak VRAM across all epochs ----
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    train_t0 = time.perf_counter()
    best_summary: Dict[str, Any] = {}
    epoch_times = []

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        ep_start = time.perf_counter()
        train_loss = _train_one_epoch(model, train_loader, optimizer, scheduler, scaler,
                                      device, cfg, logger, epoch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ep_time = time.perf_counter() - ep_start
        epoch_times.append(ep_time)

        eval_m = evaluate_stl(model, val_loader, device, task_meta, cfg.fp16)
        record = {
            "exp_name": cfg.name, "mode": "STL", "setting": cfg.setting,
            "model": cfg.model_key, "task": cfg.task_key,
            "epoch": epoch, "train_loss": train_loss,
            "time_per_epoch": ep_time,
            "trainable_params": pcount["trainable_params"],
            "total_params":     pcount["total_params"],
            **eval_m,
        }
        # per-task score (Eq. 19)
        record["task_score"] = task_score(cfg.task_key, eval_m)
        history.append(record)

        logger.info(
            f"epoch {epoch} | train_loss {train_loss:.4f} | eval_loss {eval_m['eval_loss']:.4f} "
            + " | ".join(f"{k}={eval_m[k]:.4f}" for k in task_meta["metric_keys"])
            + f" | S_k={record['task_score']:.4f} | time {ep_time:.1f}s"
        )

        current = eval_m[primary]
        if current > best_metric:
            best_metric = current; patience = 0
            mgr.save_best(model, tokenizer)
            best_summary = dict(record)
            logger.info(f"  ↳ new best {primary}={best_metric:.4f}")
        else:
            patience += 1
            logger.info(f"  ↳ no improvement (pat {patience}/{cfg.early_stop_patience})")

        mgr.save_checkpoint(epoch, model, optimizer, scheduler, scaler,
                            best_metric=best_metric, extra={"patience": patience})

        if patience >= cfg.early_stop_patience:
            logger.info("Early stopping.")
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_train_time = time.perf_counter() - train_t0
    train_peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024**2)
                          if device.type == "cuda" else 0.0)

    mgr.save_final(model, tokenizer)

    # ---- INFERENCE benchmark ----
    if device.type == "cuda":
        torch.cuda.empty_cache()
    bench = benchmark_inference(model, val_loader, device,
                                warmup_batches=5, max_batches=50,
                                use_fp16=cfg.fp16)
    inf_res = {
        "inference_peak_vram_mb": bench["peak_vram_mb"],
        "inference_latency_ms_per_sample": bench["inference_latency_ms"],
        "inference_throughput_samples_per_sec": bench["throughput_samples_per_sec"],
        "inference_benchmark_samples": bench["benchmark_samples"],
    }

    # ---- SUMMARY (everything required by paper) ----
    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else float("nan")
    summary = {
        "exp_name": cfg.name, "mode": "STL", "setting": cfg.setting,
        "model": cfg.model_key, "model_id": cfg.model_id,
        "task": cfg.task_key,
        "best_epoch": best_summary.get("epoch", -1),
        "best_metric_name": primary,
        f"best_{primary}": best_metric,
        # core metrics from best epoch
        "accuracy":  best_summary.get("accuracy"),
        "macro_f1":  best_summary.get("macro_f1"),
        "precision": best_summary.get("precision"),
        "recall":    best_summary.get("recall"),
        "pearson":   best_summary.get("pearson"),
        "spearman":  best_summary.get("spearman"),
        "eval_loss": best_summary.get("eval_loss"),
        # per-task score + overall (for STL the overall is just S_k)
        "task_score": best_summary.get("task_score"),
        "overall_score": best_summary.get("task_score"),
        # model size
        "total_params":     pcount["total_params"],
        "trainable_params": pcount["trainable_params"],
        # train cost
        "total_train_time_sec": total_train_time,
        "avg_time_per_epoch_sec": avg_epoch_time,
        "epochs_completed": len(epoch_times),
        # train memory
        "train_peak_vram_mb": train_peak_vram_mb,
        # inference cost + memory
        **inf_res,
        # hardware
        **get_hardware_info(),
    }
    mgr.save_summary(summary)
    mgr.cleanup_step_ckpts()
    logger.info(f"Summary: {json.dumps(summary, indent=2, default=str)}")
    return summary
