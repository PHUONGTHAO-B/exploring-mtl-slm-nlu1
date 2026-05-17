"""
MTL trainer: shared encoder + per-task heads on SST-2/QQP/STS-B.

Implements Algorithm 1 from the paper:
  - Task-wise mini-batch scheduling (sample one task per step)
  - AdamW + linear warmup
  - Joint objective with equal task weights (lambda_k = 1/K)
  - Same regularization: grad clip, dropout, weight decay, early stopping

Captures *all* metrics required by the paper (per-task + overall).
"""
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup

from ..common.config import ExperimentConfig, TASK_CONFIGS
from ..common.data import build_tokenizer
from ..common.logger import build_logger, HistoryWriter
from ..common.checkpoint import CheckpointManager
from ..common.benchmark import count_parameters
from ..common.metrics import (compute_classification_metrics,
                              compute_regression_metrics,
                              task_score, overall_score)
from ..common.utils import set_seed, get_device, get_hardware_info
from .model import MultiTaskModel


# ---------------------------------------------------------------------------
# Per-task tokenization → tensors
# ---------------------------------------------------------------------------
def _tokenize_task(tokenizer, raw_ds, task_meta, max_len: int):
    a = list(raw_ds[task_meta["text_a"]])
    b = list(raw_ds[task_meta["text_b"]]) if task_meta["text_b"] else None
    kw = dict(truncation=True, padding="max_length",
              max_length=max_len, return_tensors="pt")
    enc = tokenizer(a, b, **kw) if b else tokenizer(a, **kw)
    y = torch.tensor(raw_ds["label"],
                     dtype=torch.float if task_meta["task_type"] == "regression"
                     else torch.long)
    return enc, y


def _make_loader(enc, y, batch_size: int, shuffle: bool, num_workers: int = 2):
    n = enc["input_ids"].size(0)
    class _DS(torch.utils.data.Dataset):
        def __len__(self): return n
        def __getitem__(self, i):
            item = {k: v[i] for k, v in enc.items()}
            item["labels"] = y[i]
            return item
    return DataLoader(_DS(), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


# ---------------------------------------------------------------------------
# Eval per-task using current MTL model + head
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_mtl_task(model, loader, device, task_key: str, task_meta, use_fp16: bool):
    model.eval()
    losses, preds, labels = [], [], []
    is_reg = task_meta["task_type"] == "regression"
    loss_fn = nn.MSELoss() if is_reg else nn.CrossEntropyLoss()
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        y = batch["labels"]
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        if use_fp16 and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(task=task_key, **inputs)
                if is_reg: logits_l = logits.squeeze(-1)
                else:      logits_l = logits
                loss = loss_fn(logits_l, y)
        else:
            logits = model(task=task_key, **inputs)
            logits_l = logits.squeeze(-1) if is_reg else logits
            loss = loss_fn(logits_l, y)
        losses.append(loss.item())
        if is_reg:
            preds.append(logits.float().squeeze(-1).cpu().numpy())
        else:
            preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
        labels.append(y.cpu().numpy())
    preds = np.concatenate(preds); labels = np.concatenate(labels)
    eval_loss = float(np.mean(losses)) if losses else float("nan")
    if is_reg:
        m = compute_regression_metrics(preds, labels)
    else:
        m = compute_classification_metrics(preds, labels)
    m["eval_loss"] = eval_loss
    return m


@torch.no_grad()
def benchmark_mtl_inference(model, val_loaders: Dict[str, DataLoader], device,
                            use_fp16: bool, warmup: int = 5, max_batches: int = 50):
    """Measure inference per task; return per-task dict + aggregates."""
    per_task = {}
    model.eval()
    for tk, loader in val_loaders.items():
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        it = iter(loader)
        for _ in range(warmup):
            try: b = next(it)
            except StopIteration: break
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()
                 if isinstance(v, torch.Tensor) and k != "labels"}
            if use_fp16 and device.type == "cuda":
                with torch.amp.autocast("cuda"): model(task=tk, **b)
            else:
                model(task=tk, **b)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        n = 0; t0 = time.perf_counter()
        for i, b in enumerate(loader):
            if i >= max_batches: break
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()
                 if isinstance(v, torch.Tensor) and k != "labels"}
            n += b["input_ids"].size(0)
            if use_fp16 and device.type == "cuda":
                with torch.amp.autocast("cuda"): model(task=tk, **b)
            else:
                model(task=tk, **b)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        el = time.perf_counter() - t0
        per_task[tk] = {
            "peak_vram_mb": (torch.cuda.max_memory_allocated(device)/(1024**2)
                             if device.type == "cuda" else 0.0),
            "latency_ms_per_sample": (el / max(n, 1)) * 1000.0,
            "throughput_samples_per_sec": n / max(el, 1e-9),
            "benchmark_samples": n,
        }
    return per_task


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_mtl(cfg: ExperimentConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = get_device()
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(cfg.name, str(out_dir / "train.log"))

    mgr = CheckpointManager(str(out_dir), keep_n=cfg.keep_n_ckpts)
    if mgr.is_done():
        summary = mgr.load_summary()
        logger.info(f"[SKIP] {cfg.name} already done.")
        return summary

    logger.info(f"===== MTL: {cfg.name} (LoRA={cfg.use_lora}) =====")
    logger.info(f"Config: {json.dumps(cfg.to_dict(), indent=2, default=str)}")
    logger.info(f"Hardware: {get_hardware_info()}")
    (out_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, default=str), encoding="utf-8")

    tasks = cfg.tasks
    assert tasks, "MTL requires cfg.tasks"
    task_specs = {t: {"num_labels": TASK_CONFIGS[t]["num_labels"],
                      "is_reg": TASK_CONFIGS[t]["task_type"] == "regression"}
                  for t in tasks}

    # ---- model ----
    lora_cfg = None
    if cfg.use_lora:
        lora_cfg = dict(r=cfg.lora_r, alpha=cfg.lora_alpha,
                        dropout=cfg.lora_dropout,
                        target_modules=cfg.lora_target_modules)
    model = MultiTaskModel(cfg.model_id, task_specs, lora_cfg).to(device)
    tokenizer = build_tokenizer(cfg.model_id)
    pcount = count_parameters(model)
    logger.info(f"trainable={pcount['trainable_params']:,} total={pcount['total_params']:,}")

    # ---- per-task data ----
    train_data, val_loaders, task_sizes = {}, {}, {}
    for t in tasks:
        meta = TASK_CONFIGS[t]
        raw = load_dataset(*meta["hf_name"])
        if cfg.max_train_samples:
            raw["train"] = raw["train"].shuffle(seed=cfg.seed).select(
                range(min(cfg.max_train_samples, len(raw["train"]))))
        if cfg.max_eval_samples:
            raw["validation"] = raw["validation"].select(
                range(min(cfg.max_eval_samples, len(raw["validation"]))))

        enc_tr, y_tr = _tokenize_task(tokenizer, raw["train"], meta, cfg.max_seq_len)
        enc_v,  y_v  = _tokenize_task(tokenizer, raw["validation"], meta, cfg.max_seq_len)
        task_sizes[t] = enc_tr["input_ids"].size(0)
        train_data[t] = dict(enc=enc_tr, y=y_tr, cursor=0,
                             is_reg=task_specs[t]["is_reg"])
        val_loaders[t] = _make_loader(enc_v, y_v,
                                      batch_size=cfg.eval_batch_size, shuffle=False,
                                      num_workers=cfg.num_workers)

    total_train_samples = sum(task_sizes.values())
    steps_per_epoch = math.ceil(total_train_samples / cfg.batch_size)
    logger.info(f"tasks={tasks} sizes={task_sizes} steps/epoch={steps_per_epoch}")

    # task sampling
    if cfg.sampling == "proportional":
        probs = np.array([task_sizes[t] / total_train_samples for t in tasks])
    elif cfg.sampling == "uniform":
        probs = np.array([1.0/len(tasks)] * len(tasks))
    else:
        probs = None
    rng = np.random.RandomState(cfg.seed)

    # ---- optimizer / scheduler / scaler ----
    no_decay = ("bias", "LayerNorm.weight")
    pg = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(pg, lr=cfg.learning_rate)
    total_sched_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(cfg.warmup_ratio * total_sched_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_sched_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.fp16 and device.type == "cuda"))
    cls_loss = nn.CrossEntropyLoss(); reg_loss = nn.MSELoss()
    K = len(tasks); lam = 1.0 / K   # equal task weights (paper Eq. 9)

    history = HistoryWriter(str(out_dir))

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

    # ---- TRAIN ----
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    train_t0 = time.perf_counter()
    best_summary: Dict[str, Any] = {}
    epoch_times = []

    def pick_task(step_idx: int) -> str:
        if cfg.sampling == "round_robin":
            return tasks[step_idx % len(tasks)]
        return tasks[rng.choice(len(tasks), p=probs)]

    def one_step(t: str) -> float:
        d = train_data[t]; bs = cfg.batch_size
        n = d["enc"]["input_ids"].size(0)
        if d["cursor"] + bs > n: d["cursor"] = 0
        s, e = d["cursor"], d["cursor"] + bs
        batch = {k: v[s:e].to(device, non_blocking=True) for k, v in d["enc"].items()}
        y = d["y"][s:e].to(device, non_blocking=True)
        d["cursor"] += bs

        optimizer.zero_grad(set_to_none=True)
        if cfg.fp16 and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(task=t, **batch)
                lg = logits.squeeze(-1) if d["is_reg"] else logits
                task_l = reg_loss(lg, y) if d["is_reg"] else cls_loss(lg, y)
                loss = lam * task_l
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(task=t, **batch)
            lg = logits.squeeze(-1) if d["is_reg"] else logits
            task_l = reg_loss(lg, y) if d["is_reg"] else cls_loss(lg, y)
            loss = lam * task_l
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        scheduler.step()
        return float(task_l.detach())

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        model.train()
        ep_start = time.perf_counter()
        running_loss = 0.0; task_step_counts = {t: 0 for t in tasks}
        for step in range(steps_per_epoch):
            t = pick_task(step)
            lv = one_step(t)
            running_loss += lv; task_step_counts[t] += 1
            if cfg.log_steps and step > 0 and step % cfg.log_steps == 0:
                logger.info(f"epoch {epoch} | step {step}/{steps_per_epoch} | "
                            f"avg_loss {running_loss/(step+1):.4f} | "
                            f"lr {scheduler.get_last_lr()[0]:.2e}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ep_time = time.perf_counter() - ep_start
        epoch_times.append(ep_time)

        # eval per-task + overall
        per_task_metrics = {}
        for t in tasks:
            per_task_metrics[t] = evaluate_mtl_task(model, val_loaders[t], device,
                                                   t, TASK_CONFIGS[t], cfg.fp16)
        overall = overall_score(per_task_metrics)

        record = {"exp_name": cfg.name, "mode": "MTL", "setting": cfg.setting,
                  "model": cfg.model_key, "tasks": "|".join(tasks),
                  "epoch": epoch, "time_per_epoch": ep_time,
                  "trainable_params": pcount["trainable_params"],
                  "total_params": pcount["total_params"],
                  "task_step_counts": json.dumps(task_step_counts),
                  "overall_score": overall,
                  "train_loss_avg": running_loss / steps_per_epoch}
        for t, m in per_task_metrics.items():
            for k, v in m.items():
                record[f"{t}_{k}"] = v
            record[f"{t}_score"] = task_score(t, m)
        history.append(record)
        logger.info(
            f"epoch {epoch} | overall={overall:.4f} | "
            + " | ".join(f"{t}_S={record[f'{t}_score']:.4f}" for t in tasks)
            + f" | time {ep_time:.1f}s"
        )

        # checkpoint + early stopping (primary = overall_score)
        if overall > best_metric:
            best_metric = overall; patience = 0
            mgr.save_best(model, tokenizer)
            best_summary = dict(record)
            best_summary["_per_task_metrics"] = per_task_metrics
            logger.info(f"  ↳ new best overall={overall:.4f}")
        else:
            patience += 1
            logger.info(f"  ↳ no improvement (pat {patience}/{cfg.early_stop_patience})")
        mgr.save_checkpoint(epoch, model, optimizer, scheduler, scaler,
                            best_metric=best_metric, extra={"patience": patience})
        if patience >= cfg.early_stop_patience:
            logger.info("Early stopping."); break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_train_time = time.perf_counter() - train_t0
    train_peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                          if device.type == "cuda" else 0.0)
    mgr.save_final(model, tokenizer)

    # ---- INFERENCE benchmark per task ----
    per_task_infer = benchmark_mtl_inference(
        model, val_loaders, device,
        use_fp16=cfg.fp16, warmup=5, max_batches=50)
    inf_peak = max(v["peak_vram_mb"] for v in per_task_infer.values())
    inf_lat_mean = float(np.mean([v["latency_ms_per_sample"] for v in per_task_infer.values()]))
    inf_thr_mean = float(np.mean([v["throughput_samples_per_sec"] for v in per_task_infer.values()]))

    # ---- SUMMARY ----
    avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else float("nan")
    pt = best_summary.get("_per_task_metrics", {})
    summary = {
        "exp_name": cfg.name, "mode": "MTL", "setting": cfg.setting,
        "model": cfg.model_key, "model_id": cfg.model_id,
        "tasks": "|".join(tasks),
        "use_lora": cfg.use_lora,
        "best_epoch": best_summary.get("epoch", -1),
        "overall_score": best_metric,
        # per-task scores at best epoch
        **{f"{t}_score": best_summary.get(f"{t}_score") for t in tasks},
        # per-task raw metrics at best epoch
        "per_task_metrics": pt,
        # also flatten raw metrics for CSV friendliness
        **{f"{t}_accuracy":  pt.get(t, {}).get("accuracy")   for t in tasks},
        **{f"{t}_macro_f1":  pt.get(t, {}).get("macro_f1")   for t in tasks},
        **{f"{t}_pearson":   pt.get(t, {}).get("pearson")    for t in tasks},
        **{f"{t}_spearman":  pt.get(t, {}).get("spearman")   for t in tasks},
        **{f"{t}_eval_loss": pt.get(t, {}).get("eval_loss")  for t in tasks},
        # model size
        "total_params":     pcount["total_params"],
        "trainable_params": pcount["trainable_params"],
        # train cost
        "total_train_time_sec":  total_train_time,
        "avg_time_per_epoch_sec": avg_epoch_time,
        "epochs_completed":      len(epoch_times),
        # train memory
        "train_peak_vram_mb": train_peak_vram_mb,
        # inference (aggregate)
        "inference_peak_vram_mb": inf_peak,
        "inference_latency_ms_per_sample": inf_lat_mean,
        "inference_throughput_samples_per_sec": inf_thr_mean,
        "inference_per_task": per_task_infer,
        # hardware
        **get_hardware_info(),
    }
    mgr.save_summary(summary)
    mgr.cleanup_step_ckpts()
    logger.info(f"Summary: overall={best_metric:.4f}")
    return summary
