"""
BiLSTM baseline (STL): per-task, trained from scratch on GLUE.

Captures: accuracy/F1 (cls), pearson/spearman (reg), task_score (Eq. 19),
trainable_params, train time/epoch + total, train_peak_vram, inference latency
+ vram, hardware info.
"""
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from datasets import load_dataset

from ..common.config import TASK_CONFIGS
from ..common.logger import build_logger, HistoryWriter
from ..common.metrics import (compute_classification_metrics,
                              compute_regression_metrics, task_score)
from ..common.utils import set_seed, get_device, get_hardware_info


TOKEN_RE = re.compile(r"[A-Za-z']+|\d+|[^\sA-Za-z0-9]")
PAD, UNK = "<pad>", "<unk>"


def _tok(text):
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


class Vocab:
    def __init__(self, min_freq=2, max_size=30000):
        self.min_freq, self.max_size = min_freq, max_size
        self.itos = [PAD, UNK]; self.stoi = {PAD: 0, UNK: 1}
    def build(self, texts):
        c = Counter()
        for t in texts: c.update(_tok(t))
        for w, n in c.most_common(self.max_size):
            if n < self.min_freq: continue
            self.stoi[w] = len(self.itos); self.itos.append(w)
    def encode(self, t, max_len):
        ids = [self.stoi.get(x, 1) for x in _tok(t)][:max_len]
        if len(ids) < max_len: ids += [0] * (max_len - len(ids))
        return ids
    def __len__(self): return len(self.itos)


class _GlueDS(Dataset):
    def __init__(self, hf_ds, vocab, task_meta, max_len):
        self.hf = hf_ds; self.vocab = vocab
        self.tm = task_meta; self.max_len = max_len
    def __len__(self): return len(self.hf)
    def __getitem__(self, i):
        ex = self.hf[i]
        item = {"a_ids": torch.tensor(self.vocab.encode(ex[self.tm["text_a"]], self.max_len),
                                      dtype=torch.long)}
        if self.tm["text_b"]:
            item["b_ids"] = torch.tensor(self.vocab.encode(ex[self.tm["text_b"]], self.max_len),
                                         dtype=torch.long)
        if self.tm["task_type"] == "regression":
            item["labels"] = torch.tensor(float(ex["label"]), dtype=torch.float)
        else:
            item["labels"] = torch.tensor(int(ex["label"]), dtype=torch.long)
        return item


class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden * 2
    def forward(self, ids):
        mask = (ids != 0).unsqueeze(-1).float()
        out, _ = self.lstm(self.emb(ids))
        out = out.masked_fill(mask == 0, -1e9)
        pooled, _ = out.max(dim=1)
        return self.drop(pooled)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, task_meta, emb_dim=300, hidden=256):
        super().__init__()
        self.tm = task_meta
        self.is_pair = task_meta["text_b"] is not None
        self.is_reg = task_meta["task_type"] == "regression"
        self.enc = BiLSTMEncoder(vocab_size, emb_dim, hidden)
        feat = self.enc.out_dim * (4 if self.is_pair else 1)
        out_dim = 1 if self.is_reg else task_meta["num_labels"]
        self.head = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(256, out_dim))
    def forward(self, a_ids, b_ids=None):
        u = self.enc(a_ids)
        if self.is_pair:
            v = self.enc(b_ids)
            feats = torch.cat([u, v, torch.abs(u - v), u * v], dim=-1)
        else:
            feats = u
        return self.head(feats)


def _linear_warmup(step, warmup, total):
    if step < warmup:
        return float(step) / float(max(1, warmup))
    return max(0.0, float(total - step) / float(max(1, total - warmup)))


def run_bilstm(
    task_key: str,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_len: int = 128,
    emb_dim: int = 300,
    hidden: int = 256,
    seed: int = 42,
    patience: int = 3,
    weight_decay: float = 0.01,
    num_workers: int = 2,
) -> Dict[str, Any]:
    set_seed(seed)
    device = get_device()
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    logger = build_logger(f"bilstm-{task_key}", str(out_dir / "train.log"))
    history = HistoryWriter(str(out_dir))
    logger.info(f"===== BiLSTM | task={task_key} =====")
    logger.info(f"Hardware: {get_hardware_info()}")

    tm = TASK_CONFIGS[task_key]
    raw = load_dataset(*tm["hf_name"])

    # vocab from train
    texts = list(raw["train"][tm["text_a"]])
    if tm["text_b"]: texts += list(raw["train"][tm["text_b"]])
    vocab = Vocab(min_freq=2, max_size=30000); vocab.build(texts)
    logger.info(f"vocab_size={len(vocab)}")

    train_ds = _GlueDS(raw["train"], vocab, tm, max_len)
    val_ds   = _GlueDS(raw["validation"], vocab, tm, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    model = BiLSTMClassifier(len(vocab), tm, emb_dim, hidden).to(device)
    pcount = {"total_params": sum(p.numel() for p in model.parameters()),
              "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    logger.info(f"trainable={pcount['trainable_params']:,} total={pcount['total_params']:,}")

    loss_fn = nn.MSELoss() if tm["task_type"] == "regression" else nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _linear_warmup(s, warmup_steps, total_steps))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    train_t0 = time.perf_counter()
    best_metric = -float("inf"); pat = 0; best_summary = {}
    primary = tm["primary_metric"]; epoch_times = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        ep_start = time.perf_counter(); tot, n = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["a_ids"], batch.get("b_ids"))
            y = batch["labels"]
            if tm["task_type"] == "regression": logits = logits.squeeze(-1)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            tot += loss.item(); n += 1
        train_loss = tot / max(n, 1)
        ep_time = time.perf_counter() - ep_start
        epoch_times.append(ep_time)

        # eval
        model.eval()
        losses, preds, labels = [], [], []
        with torch.no_grad():
            for b in val_loader:
                b = {k: v.to(device) for k, v in b.items()}
                logits = model(b["a_ids"], b.get("b_ids"))
                y = b["labels"]
                if tm["task_type"] == "regression":
                    logits = logits.squeeze(-1); loss = loss_fn(logits, y)
                    preds.append(logits.cpu().numpy())
                else:
                    loss = loss_fn(logits, y)
                    preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
                losses.append(loss.item()); labels.append(y.cpu().numpy())
        preds = np.concatenate(preds); labels = np.concatenate(labels)
        if tm["task_type"] == "regression":
            m = compute_regression_metrics(preds, labels)
        else:
            m = compute_classification_metrics(preds, labels)
        m["eval_loss"] = float(np.mean(losses)) if losses else float("nan")
        sk = task_score(task_key, m)

        rec = {"exp_name": f"bilstm_{task_key}", "mode": "STL", "setting": "bilstm",
               "model": "bilstm", "task": task_key, "epoch": epoch,
               "train_loss": train_loss, "time_per_epoch": ep_time,
               "trainable_params": pcount["trainable_params"],
               "total_params":     pcount["total_params"],
               "task_score": sk, **m}
        history.append(rec)
        logger.info(f"epoch {epoch} | loss {train_loss:.4f} | "
                    + " | ".join(f"{k}={m[k]:.4f}" for k in tm["metric_keys"])
                    + f" | S={sk:.4f} | time {ep_time:.1f}s")

        if m[primary] > best_metric:
            best_metric = m[primary]; pat = 0; best_summary = dict(rec)
            torch.save({"model_state": model.state_dict(),
                        "vocab_itos": vocab.itos, "task": task_key},
                       out_dir / "best_model.pt")
            logger.info(f"  ↳ new best {primary}={best_metric:.4f}")
        else:
            pat += 1
            logger.info(f"  ↳ no improvement (pat {pat}/{patience})")
        if pat >= patience:
            logger.info("Early stopping."); break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_train_time = time.perf_counter() - train_t0
    train_peak = (torch.cuda.max_memory_allocated(device)/(1024**2)
                  if device.type == "cuda" else 0.0)
    torch.save({"model_state": model.state_dict(),
                "vocab_itos": vocab.itos, "task": task_key},
               out_dir / "final_model.pt")

    # ---- INFERENCE benchmark ----
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    model.eval()
    # warmup
    it = iter(val_loader)
    with torch.no_grad():
        for _ in range(5):
            try: b = next(it)
            except StopIteration: break
            b = {k: v.to(device) for k, v in b.items()}
            model(b["a_ids"], b.get("b_ids"))
    if device.type == "cuda": torch.cuda.synchronize(device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter(); n_proc = 0
    with torch.no_grad():
        for i, b in enumerate(val_loader):
            if i >= 50: break
            b = {k: v.to(device) for k, v in b.items()}
            n_proc += b["a_ids"].size(0)
            model(b["a_ids"], b.get("b_ids"))
    if device.type == "cuda": torch.cuda.synchronize(device)
    inf_elapsed = time.perf_counter() - t0
    inf_peak = (torch.cuda.max_memory_allocated(device)/(1024**2)
                if device.type == "cuda" else 0.0)

    summary = {
        "exp_name": f"bilstm_{task_key}", "mode": "STL", "setting": "bilstm",
        "model": "bilstm", "task": task_key,
        "best_epoch": best_summary.get("epoch", -1),
        "best_metric_name": primary,
        f"best_{primary}": best_metric,
        "accuracy":  best_summary.get("accuracy"),
        "macro_f1":  best_summary.get("macro_f1"),
        "precision": best_summary.get("precision"),
        "recall":    best_summary.get("recall"),
        "pearson":   best_summary.get("pearson"),
        "spearman":  best_summary.get("spearman"),
        "eval_loss": best_summary.get("eval_loss"),
        "task_score": best_summary.get("task_score"),
        "overall_score": best_summary.get("task_score"),
        "total_params":     pcount["total_params"],
        "trainable_params": pcount["trainable_params"],
        "total_train_time_sec": total_train_time,
        "avg_time_per_epoch_sec": float(np.mean(epoch_times)) if epoch_times else float("nan"),
        "epochs_completed": len(epoch_times),
        "train_peak_vram_mb": train_peak,
        "inference_peak_vram_mb": inf_peak,
        "inference_latency_ms_per_sample": (inf_elapsed / max(n_proc, 1)) * 1000.0,
        "inference_throughput_samples_per_sec": n_proc / max(inf_elapsed, 1e-9),
        "inference_benchmark_samples": n_proc,
        **get_hardware_info(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Summary: {summary}")
    return summary
