"""Multi-task model: shared encoder + per-task heads, optional LoRA."""
import inspect
from typing import Dict, Optional, List

import torch.nn as nn
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model


class MultiTaskModel(nn.Module):
    """
    Shared encoder + ModuleDict of per-task heads.
    LoRA is applied to the encoder if `lora_cfg` is provided.

    task_specs: {task_key: {'num_labels': int, 'is_reg': bool}}
    """

    def __init__(
        self,
        base_model_id: str,
        task_specs: Dict[str, Dict],
        lora_cfg: Optional[Dict] = None,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        cfg = AutoConfig.from_pretrained(base_model_id)
        self.encoder = AutoModel.from_pretrained(base_model_id, config=cfg)
        self.lora_applied = False
        if lora_cfg is not None:
            lc = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
                lora_dropout=lora_cfg["dropout"],
                target_modules=lora_cfg["target_modules"],
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lc)
            self.lora_applied = True
        self.hidden_size = cfg.hidden_size
        self.task_specs = task_specs
        self.heads = nn.ModuleDict()
        for tname, spec in task_specs.items():
            self.heads[tname] = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(self.hidden_size, spec["num_labels"]),
            )

    def encoder_signature(self):
        return set(inspect.signature(self.encoder.forward).parameters.keys())

    def forward(self, task: str, **inputs):
        sig = self.encoder_signature()
        kw = {k: v for k, v in inputs.items() if k in sig}
        out = self.encoder(**kw)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = h[:, 0]                       # CLS-token pooling
        return self.heads[task](pooled)
