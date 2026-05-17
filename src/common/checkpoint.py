"""
Checkpoint manager.
- Saves full state (model + optimizer + scheduler + scaler + step + extras)
- Keeps only N latest checkpoints (paper default = 2)
- Also persists best_model / final_model directories
- Supports resume from latest checkpoint
"""
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    def __init__(self, out_dir: str, keep_n: int = 2):
        self.out_dir = Path(out_dir)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.best_dir = self.out_dir / "best_model"
        self.final_dir = self.out_dir / "final_model"
        self.state_path = self.out_dir / "trainer_state.json"
        self.summary_path = self.out_dir / "summary.json"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.keep_n = keep_n

    # ---------- experiment lifecycle ----------
    def is_done(self) -> bool:
        return self.summary_path.exists()

    def save_summary(self, summary: Dict[str, Any]) -> None:
        self.summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

    def load_summary(self) -> Optional[Dict[str, Any]]:
        if self.summary_path.exists():
            return json.loads(self.summary_path.read_text(encoding="utf-8"))
        return None

    # ---------- step-level checkpoints ----------
    def save_checkpoint(
        self,
        epoch: int,
        model,
        optimizer,
        scheduler,
        scaler,
        best_metric: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        ck = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "best_metric": best_metric,
            "extra": extra or {},
        }
        path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save(ck, path)
        state = {
            "last_epoch": epoch,
            "best_metric": best_metric,
            "last_ckpt": str(path),
            "extra": extra or {},
        }
        self.state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        self._prune_old()
        return str(path)

    def _prune_old(self) -> None:
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        for old in ckpts[:-self.keep_n]:
            try:
                old.unlink()
            except Exception:
                pass

    def latest_checkpoint(self) -> Optional[str]:
        ckpts = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        return str(ckpts[-1]) if ckpts else None

    def load_checkpoint(
        self,
        path: str,
        model,
        optimizer,
        scheduler,
        scaler,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        ck = torch.load(path, map_location=map_location)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optim_state"])
        if scheduler is not None and ck.get("sched_state") is not None:
            scheduler.load_state_dict(ck["sched_state"])
        if scaler is not None and ck.get("scaler_state") is not None:
            scaler.load_state_dict(ck["scaler_state"])
        return ck

    # ---------- model snapshots (HF save_pretrained) ----------
    def save_best(self, model, tokenizer) -> None:
        try:
            model.save_pretrained(self.best_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(self.best_dir)
        except Exception:
            torch.save(model.state_dict(), self.best_dir / "model_state.pt")

    def save_final(self, model, tokenizer) -> None:
        try:
            model.save_pretrained(self.final_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(self.final_dir)
        except Exception:
            torch.save(model.state_dict(), self.final_dir / "model_state.pt")

    def cleanup_step_ckpts(self) -> None:
        """Remove step-level ckpts (call after summary is written)."""
        for p in self.ckpt_dir.glob("epoch_*.pt"):
            try:
                p.unlink()
            except Exception:
                pass
