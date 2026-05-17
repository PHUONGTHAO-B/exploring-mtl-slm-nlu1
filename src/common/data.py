"""Dataset loading + tokenization for GLUE tasks (SST-2, QQP, STS-B)."""
import inspect
from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from .config import TASK_CONFIGS


def build_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def _tokenize(examples, tokenizer, task_meta, max_len: int):
    text_a = examples[task_meta["text_a"]]
    if task_meta["text_b"]:
        text_b = examples[task_meta["text_b"]]
        return tokenizer(text_a, text_b, padding="max_length",
                         truncation=True, max_length=max_len)
    return tokenizer(text_a, padding="max_length", truncation=True, max_length=max_len)


def load_glue_loaders(
    task_key: str,
    tokenizer,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    max_seq_len: int = 256,
    num_workers: int = 2,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Load train + validation DataLoaders for a GLUE task."""
    task_meta = TASK_CONFIGS[task_key]
    hf_root, hf_sub = task_meta["hf_name"]
    raw = load_dataset(hf_root, hf_sub)

    if max_train_samples:
        raw["train"] = raw["train"].shuffle(seed=seed).select(
            range(min(max_train_samples, len(raw["train"]))))
    if max_eval_samples:
        raw["validation"] = raw["validation"].select(
            range(min(max_eval_samples, len(raw["validation"]))))

    def _map(batch):
        enc = _tokenize(batch, tokenizer, task_meta, max_seq_len)
        if task_meta["task_type"] == "regression":
            enc["labels"] = [float(x) for x in batch[task_meta["label_col"]]]
        else:
            enc["labels"] = [int(x) for x in batch[task_meta["label_col"]]]
        return enc

    cols_to_remove = [c for c in raw["train"].column_names if c != "idx"]
    train_ds = raw["train"].map(_map, batched=True, remove_columns=cols_to_remove)
    val_ds   = raw["validation"].map(_map, batched=True, remove_columns=cols_to_remove)

    keep = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in train_ds.column_names:
        keep.append("token_type_ids")
    train_ds.set_format(type="torch", columns=keep)
    val_ds.set_format(type="torch",   columns=keep)

    label_dtype = (torch.float if task_meta["task_type"] == "regression"
                   else torch.long)

    def collate(batch):
        out = {}
        for k in batch[0].keys():
            if k == "labels":
                out[k] = torch.tensor([b[k].item() for b in batch], dtype=label_dtype)
            else:
                out[k] = torch.stack([b[k] for b in batch])
        return out

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=collate, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=collate, drop_last=False)
    return train_loader, val_loader


def filter_model_kwargs(model, batch: dict) -> dict:
    """Drop kwargs the model's forward() doesn't accept (e.g. token_type_ids for DistilBERT)."""
    sig = set(inspect.signature(model.forward).parameters.keys())
    return {k: v for k, v in batch.items() if k in sig}
