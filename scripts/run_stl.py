"""
Run ONE Single-Task Learning experiment.

Examples:
    python -m scripts.run_stl --model albert --task sst2
    python -m scripts.run_stl --model distilbert --task qqp --output_dir outputs/foo
"""
import argparse

from src.common.config import ExperimentConfig
from src.stl.trainer import run_stl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["albert", "distilbert", "minilm"])
    p.add_argument("--task", required=True, choices=["sst2", "qqp", "stsb"])
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--keep_n_ckpts", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--name", type=str, default="")
    args, _ = p.parse_known_args()
    return args


def main():
    a = parse_args()
    cfg = ExperimentConfig(
        name=a.name or f"stl_{a.model}_{a.task}",
        model_key=a.model, task_key=a.task, setting="stl_full_ft",
        learning_rate=a.lr, weight_decay=a.weight_decay,
        warmup_ratio=a.warmup_ratio, num_epochs=a.num_epochs,
        batch_size=a.batch_size, max_seq_len=a.max_seq_len,
        seed=a.seed, early_stop_patience=a.patience,
        keep_n_ckpts=a.keep_n_ckpts, num_workers=a.num_workers,
        fp16=not a.no_fp16, resume=not a.no_resume,
        max_train_samples=a.max_train_samples,
        max_eval_samples=a.max_eval_samples,
        output_dir=a.output_dir,
    )
    summary = run_stl(cfg)
    print("\n=== DONE ===")
    print(f"Output dir: {cfg.output_dir}")
    print(f"Overall score: {summary.get('overall_score')}")


if __name__ == "__main__":
    main()
