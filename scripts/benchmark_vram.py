"""
Quick benchmark of train VRAM + inference latency for all 16 SLM-based experiments
WITHOUT running full training.

Runs ~50 train steps + ~1000 inference samples per experiment.
Useful when you want Table 4 (efficiency) without re-doing Table 3 (accuracy).
Total runtime ~10-15 minutes on T4.
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoConfig,
                          AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)
from peft import LoraConfig, TaskType, get_peft_model

from src.common.config import TASK_CONFIGS, MODEL_CONFIGS, LORA_TARGET_MODULES
from src.common.utils import set_seed, get_hardware_info
from src.common.benchmark import count_parameters
from src.common.data import filter_model_kwargs
from src.mtl.model import MultiTaskModel


PAPER = dict(lr=2e-5, weight_decay=0.01, warmup_ratio=0.1,
             batch_size=16, max_length=256, grad_clip=1.0, seed=42,
             lora_r=8, lora_alpha=16, lora_dropout=0.1)
SLM = ["albert", "distilbert", "minilm"]


def tok_data(tok, raw, ds_spec, max_len, limit=None):
    if limit and len(raw) > limit: raw = raw.select(range(limit))
    a = list(raw[ds_spec["text_a"]])
    b = list(raw[ds_spec["text_b"]]) if ds_spec["text_b"] else None
    kw = dict(truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    enc = tok(a, b, **kw) if b else tok(a, **kw)
    y = torch.tensor(raw["label"],
                     dtype=torch.float if ds_spec["task_type"] == "regression"
                     else torch.long)
    return enc, y


def bench_stl(model_key, task_key, train_steps=50, infer_samples=1000):
    set_seed(PAPER["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    ds_spec = TASK_CONFIGS[task_key]
    model_id = MODEL_CONFIGS[model_key]["hf_id"]
    cfg = AutoConfig.from_pretrained(model_id, num_labels=ds_spec["num_labels"],
        problem_type="regression" if ds_spec["task_type"] == "regression"
                     else "single_label_classification")
    model = AutoModelForSequenceClassification.from_pretrained(model_id, config=cfg).to(device).train()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    pc = count_parameters(model); bs = PAPER["batch_size"]
    opt = torch.optim.AdamW(model.parameters(), lr=PAPER["lr"],
                            weight_decay=PAPER["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(PAPER["warmup_ratio"]*train_steps), train_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    raw = load_dataset(*ds_spec["hf_name"])["train"]
    sub = raw.shuffle(seed=PAPER["seed"]).select(range(min((train_steps+15)*bs, len(raw))))
    enc, y = tok_data(tok, sub, ds_spec, PAPER["max_length"])

    def step_(s, e):
        b = {k: v[s:e].to(device, non_blocking=True) for k, v in enc.items()}
        yy = y[s:e].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(**filter_model_kwargs(model, b), labels=yy)
            scaler.scale(out.loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), PAPER["grad_clip"])
            scaler.step(opt); scaler.update()
        else:
            out = model(**filter_model_kwargs(model, b), labels=yy)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PAPER["grad_clip"])
            opt.step()
        sched.step()

    # warmup 5
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    cur = 0
    for _ in range(5):
        step_(cur, cur+bs); cur += bs
    if device.type == "cuda": torch.cuda.synchronize(device)

    # train VRAM
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(train_steps):
        if cur+bs > enc["input_ids"].size(0): cur = 0
        step_(cur, cur+bs); cur += bs
    if device.type == "cuda": torch.cuda.synchronize(device)
    tr_time = time.perf_counter() - t0
    tr_vram = (torch.cuda.max_memory_allocated(device) / (1024**2)
               if device.type == "cuda" else 0.0)

    # inference
    model.eval()
    raw_val = load_dataset(*ds_spec["hf_name"])[ds_spec["val_split"]]
    venc, _ = tok_data(tok, raw_val, ds_spec, PAPER["max_length"], limit=infer_samples)
    nv = venc["input_ids"].size(0); eb = 32
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    with torch.no_grad():
        for i in range(5):
            s=(i*eb)%max(nv,1); e=min(s+eb,nv)
            if e<=s: break
            b = {k: v[s:e].to(device, non_blocking=True) for k, v in venc.items()}
            if use_amp:
                with torch.amp.autocast("cuda"): model(**filter_model_kwargs(model, b))
            else:
                model(**filter_model_kwargs(model, b))
    if device.type == "cuda":
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    t0 = time.perf_counter(); n_proc = 0
    with torch.no_grad():
        for i in range(0, nv, eb):
            b = {k: v[i:i+eb].to(device, non_blocking=True) for k, v in venc.items()}
            n_proc += b["input_ids"].size(0)
            if use_amp:
                with torch.amp.autocast("cuda"): model(**filter_model_kwargs(model, b))
            else:
                model(**filter_model_kwargs(model, b))
    if device.type == "cuda": torch.cuda.synchronize(device)
    inf_t = time.perf_counter() - t0
    inf_vram = (torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda" else 0.0)
    out = {
        "exp_name": f"stl_{model_key}_{task_key}",
        "mode": "STL", "setting": "stl_full_ft",
        "model": model_key, "task": task_key,
        "train_peak_vram_mb": tr_vram,
        "train_time_per_step_ms": (tr_time / train_steps) * 1000,
        "inference_peak_vram_mb": inf_vram,
        "inference_latency_ms_per_sample": (inf_t / max(n_proc, 1)) * 1000,
        "inference_throughput_samples_per_sec": n_proc / max(inf_t, 1e-9),
        **pc, **get_hardware_info(),
    }
    del model, tok, opt, sched, scaler, enc, venc, y
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


def bench_mtl(model_key, lora: bool, train_steps=50, infer_samples=500):
    set_seed(PAPER["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    tasks = ["sst2", "qqp", "stsb"]; bs = PAPER["batch_size"]
    model_id = MODEL_CONFIGS[model_key]["hf_id"]
    specs = {t: {"num_labels": TASK_CONFIGS[t]["num_labels"],
                 "is_reg": TASK_CONFIGS[t]["task_type"] == "regression"}
             for t in tasks}
    lora_cfg = None
    if lora:
        lora_cfg = dict(r=PAPER["lora_r"], alpha=PAPER["lora_alpha"],
                        dropout=PAPER["lora_dropout"],
                        target_modules=LORA_TARGET_MODULES[model_key])
    model = MultiTaskModel(model_id, specs, lora_cfg).to(device).train()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    pc = count_parameters(model)

    opt = torch.optim.AdamW(model.parameters(), lr=PAPER["lr"],
                            weight_decay=PAPER["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(PAPER["warmup_ratio"]*train_steps), train_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cls_l = nn.CrossEntropyLoss(); reg_l = nn.MSELoss()

    tdata = {}; sizes = {}
    for t in tasks:
        sp = TASK_CONFIGS[t]
        raw = load_dataset(*sp["hf_name"])["train"]
        sizes[t] = len(raw)
        sub = raw.shuffle(seed=PAPER["seed"]).select(range(min((train_steps+15)*bs, len(raw))))
        enc, y = tok_data(tok, sub, sp, PAPER["max_length"])
        tdata[t] = {"enc": enc, "y": y, "cur": 0, "reg": specs[t]["is_reg"]}
    total = sum(sizes.values())
    probs = np.array([sizes[t]/total for t in tasks])
    rng = np.random.RandomState(PAPER["seed"])

    def step_(t):
        d = tdata[t]; n = d["enc"]["input_ids"].size(0)
        if d["cur"]+bs > n: d["cur"] = 0
        s, e = d["cur"], d["cur"]+bs
        b = {k: v[s:e].to(device, non_blocking=True) for k, v in d["enc"].items()}
        yy = d["y"][s:e].to(device, non_blocking=True); d["cur"] += bs
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(task=t, **b)
                loss = reg_l(logits.squeeze(-1), yy) if d["reg"] else cls_l(logits, yy)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), PAPER["grad_clip"])
            scaler.step(opt); scaler.update()
        else:
            logits = model(task=t, **b)
            loss = reg_l(logits.squeeze(-1), yy) if d["reg"] else cls_l(logits, yy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PAPER["grad_clip"])
            opt.step()
        sched.step()

    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    for i in range(5):
        step_(tasks[rng.choice(len(tasks), p=probs)])
    if device.type == "cuda": torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for i in range(train_steps):
        step_(tasks[rng.choice(len(tasks), p=probs)])
    if device.type == "cuda": torch.cuda.synchronize(device)
    tr_time = time.perf_counter() - t0
    tr_vram = (torch.cuda.max_memory_allocated(device) / (1024**2)
               if device.type == "cuda" else 0.0)

    # inference per task
    model.eval(); per_task = {}
    for t in tasks:
        sp = TASK_CONFIGS[t]
        raw_val = load_dataset(*sp["hf_name"])[sp["val_split"]]
        venc, _ = tok_data(tok, raw_val, sp, PAPER["max_length"], limit=infer_samples)
        nv = venc["input_ids"].size(0); eb = 32
        if device.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        with torch.no_grad():
            for i in range(5):
                s=(i*eb)%max(nv,1); e=min(s+eb,nv)
                if e<=s: break
                b = {k: v[s:e].to(device, non_blocking=True) for k, v in venc.items()}
                if use_amp:
                    with torch.amp.autocast("cuda"): model(task=t, **b)
                else:
                    model(task=t, **b)
        if device.type == "cuda":
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        t0 = time.perf_counter(); n_proc = 0
        with torch.no_grad():
            for i in range(0, nv, eb):
                b = {k: v[i:i+eb].to(device, non_blocking=True) for k, v in venc.items()}
                n_proc += b["input_ids"].size(0)
                if use_amp:
                    with torch.amp.autocast("cuda"): model(task=t, **b)
                else:
                    model(task=t, **b)
        if device.type == "cuda": torch.cuda.synchronize(device)
        el = time.perf_counter() - t0
        per_task[t] = {
            "peak_vram_mb": (torch.cuda.max_memory_allocated(device)/(1024**2)
                             if device.type == "cuda" else 0.0),
            "latency_ms_per_sample": (el / max(n_proc, 1)) * 1000.0,
            "throughput_samples_per_sec": n_proc / max(el, 1e-9),
        }

    out = {
        "exp_name": f"mtl_{'lora' if lora else 'full'}_{model_key}",
        "mode": "MTL", "setting": "mtl_lora" if lora else "mtl_full_ft",
        "model": model_key, "task": "|".join(tasks),
        "train_peak_vram_mb": tr_vram,
        "train_time_per_step_ms": (tr_time / train_steps) * 1000,
        "inference_peak_vram_mb": max(v["peak_vram_mb"] for v in per_task.values()),
        "inference_latency_ms_per_sample": float(np.mean([v["latency_ms_per_sample"] for v in per_task.values()])),
        "inference_throughput_samples_per_sec": float(np.mean([v["throughput_samples_per_sec"] for v in per_task.values()])),
        "inference_per_task": json.dumps(per_task),
        **pc, **get_hardware_info(),
    }
    del model, tok, opt, sched, scaler, tdata
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs/benchmark_vram")
    p.add_argument("--train_steps", type=int, default=50)
    p.add_argument("--infer_samples", type=int, default=1000)
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # STL × 9
    for m in SLM:
        for t in ["sst2", "qqp", "stsb"]:
            print(f"\n=== STL {m} / {t} ===")
            try:
                r = bench_stl(m, t, a.train_steps, a.infer_samples)
                print(f"  train_vram={r['train_peak_vram_mb']:.0f}MB | "
                      f"inf_lat={r['inference_latency_ms_per_sample']:.2f}ms")
            except Exception as e:
                r = {"exp_name": f"stl_{m}_{t}", "error": str(e)}
            results.append(r)
            pd.DataFrame(results).to_csv(out_dir / "all.csv", index=False)

    # MTL LoRA × 3 + MTL Full × 3
    for m in SLM:
        for lora in (True, False):
            print(f"\n=== MTL {m} {'LoRA' if lora else 'Full'} ===")
            try:
                r = bench_mtl(m, lora, a.train_steps, a.infer_samples)
                print(f"  train_vram={r['train_peak_vram_mb']:.0f}MB | "
                      f"inf_lat={r['inference_latency_ms_per_sample']:.2f}ms")
            except Exception as e:
                r = {"exp_name": f"mtl_{'lora' if lora else 'full'}_{m}", "error": str(e)}
            results.append(r)
            pd.DataFrame(results).to_csv(out_dir / "all.csv", index=False)

    # MTL Full BERT-base
    print(f"\n=== MTL BERT-base Full ===")
    try:
        r = bench_mtl("bert-base", False, a.train_steps, a.infer_samples)
        print(f"  train_vram={r['train_peak_vram_mb']:.0f}MB | "
              f"inf_lat={r['inference_latency_ms_per_sample']:.2f}ms")
    except Exception as e:
        r = {"exp_name": "mtl_full_bertbase", "error": str(e)}
    results.append(r)
    pd.DataFrame(results).to_csv(out_dir / "all.csv", index=False)

    print(f"\nDONE: {out_dir / 'all.csv'}")


if __name__ == "__main__":
    main()
