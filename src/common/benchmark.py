"""Benchmark helpers: VRAM, inference latency, parameter counts."""
import time
from typing import Dict, Any

import torch


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


@torch.no_grad()
def benchmark_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    warmup_batches: int = 5,
    max_batches: int = 50,
    use_fp16: bool = True,
    forward_fn=None,
) -> Dict[str, float]:
    """
    Measure inference latency (ms/sample), throughput, peak VRAM.

    `forward_fn(model, batch)` lets MTL pass `task=...` along with kwargs.
    If None, defaults to `model(**batch_without_labels)`.
    """
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    def _default_forward(m, b):
        b = {k: v for k, v in b.items() if k != "labels"}
        if use_fp16 and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                return m(**b)
        return m(**b)

    fwd = forward_fn or _default_forward

    # warmup
    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()
                 if isinstance(v, torch.Tensor)}
        fwd(model, batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # measure
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    n_samples = 0
    start = time.perf_counter()
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()
                 if isinstance(v, torch.Tensor)}
        bsz = next(iter(batch.values())).size(0)
        fwd(model, batch)
        n_samples += bsz
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    return {
        "inference_latency_ms": (elapsed / max(n_samples, 1)) * 1000.0,
        "throughput_samples_per_sec": n_samples / max(elapsed, 1e-9),
        "peak_vram_mb": (torch.cuda.max_memory_allocated(device) / (1024 ** 2))
                        if device.type == "cuda" else 0.0,
        "benchmark_samples": n_samples,
    }
