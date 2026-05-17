"""
Central registry for tasks + models + hyper-parameter defaults.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Tasks (GLUE)
# ---------------------------------------------------------------------------
TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "sst2": {
        "hf_name": ("glue", "sst2"),
        "task_type": "classification",
        "num_labels": 2,
        "text_a": "sentence",
        "text_b": None,
        "label_col": "label",
        "val_split": "validation",
        "metric_keys": ["accuracy", "macro_f1"],
        "primary_metric": "accuracy",
    },
    "qqp": {
        "hf_name": ("glue", "qqp"),
        "task_type": "classification",
        "num_labels": 2,
        "text_a": "question1",
        "text_b": "question2",
        "label_col": "label",
        "val_split": "validation",
        "metric_keys": ["accuracy", "macro_f1"],
        "primary_metric": "accuracy",
    },
    "stsb": {
        "hf_name": ("glue", "stsb"),
        "task_type": "regression",
        "num_labels": 1,
        "text_a": "sentence1",
        "text_b": "sentence2",
        "label_col": "label",
        "val_split": "validation",
        "metric_keys": ["pearson", "spearman"],
        "primary_metric": "pearson",
    },
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "albert":     {"hf_id": "albert-base-v2"},
    "distilbert": {"hf_id": "distilbert-base-uncased"},
    "minilm":     {"hf_id": "microsoft/MiniLM-L12-H384-uncased"},
    "bert-base":  {"hf_id": "bert-base-uncased"},
}

# LoRA target modules per backbone (from paper Table 2)
LORA_TARGET_MODULES: Dict[str, List[str]] = {
    "albert":     ["query", "key", "value"],
    "distilbert": ["q_lin", "k_lin", "v_lin", "out_lin"],
    "minilm":     ["query", "key", "value"],
    "bert-base":  ["query", "key", "value"],
}


# ---------------------------------------------------------------------------
# Experiment config (single experiment)
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Hyper-parameters for one experiment (paper Table 2)."""
    # identity
    name: str = ""
    model_key: str = "distilbert"
    task_key: Optional[str] = None              # for STL
    tasks: Optional[List[str]] = None           # for MTL
    setting: str = "stl_full_ft"                # stl_full_ft | mtl_lora | mtl_full_ft

    # optimization
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    num_epochs: int = 10
    batch_size: int = 16
    eval_batch_size: int = 32
    max_seq_len: int = 256
    seed: int = 42

    # mixed precision / hardware
    fp16: bool = True
    num_workers: int = 2

    # early stopping / checkpointing
    early_stop_patience: int = 3
    keep_n_ckpts: int = 2

    # LoRA (for MTL+LoRA only)
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1

    # MTL specific
    sampling: str = "proportional"              # proportional | uniform | round_robin

    # I/O
    output_dir: str = ""
    resume: bool = True

    # data subsampling (optional, for dev)
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    log_steps: int = 100

    def __post_init__(self):
        if not self.name:
            if self.task_key:
                self.name = f"{self.setting}_{self.model_key}_{self.task_key}"
            elif self.tasks:
                self.name = f"{self.setting}_{self.model_key}_mtl"
        if not self.output_dir:
            self.output_dir = f"outputs/{self.name}"

    @property
    def model_id(self) -> str:
        return MODEL_CONFIGS[self.model_key]["hf_id"]

    @property
    def lora_target_modules(self) -> List[str]:
        return LORA_TARGET_MODULES[self.model_key]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["model_id"] = self.model_id
        return d
