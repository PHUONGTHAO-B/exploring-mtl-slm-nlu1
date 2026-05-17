"""Reproducibility + device helpers."""
import os
import random
import platform
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_hardware_info() -> Dict:
    info = {
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "total_vram_mb": 0.0,
        "cuda_version": None,
        "cuda_capability": None,
    }
    if torch.cuda.is_available():
        d = torch.cuda.get_device_properties(0)
        info.update(
            gpu_name=d.name,
            total_vram_mb=d.total_memory / (1024 ** 2),
            cuda_version=torch.version.cuda,
            cuda_capability=f"{d.major}.{d.minor}",
        )
    return info
