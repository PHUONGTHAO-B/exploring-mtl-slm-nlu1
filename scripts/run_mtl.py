"""
Run ONE Multi-Task Learning experiment.

Examples:
    python -m scripts.run_mtl --model distilbert --lora
    python -m scripts.run_mtl --model albert --no-lora
    python -m scripts.run_mtl --model bert-base --no-lora --output_dir outputs/mtl_bertbase
"""
import argparse

from src.common.config import ExperimentConfig
from src.mtl.trainer import run_mtl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["albert", "distilbert", "minilm", "bert-base"])
    p.add_argument("--tasks", nargs="+", default=["sst2", "qqp", "stsb"])
    p.add_argument("--sampling", default="proportional",
                   choices=["proportional", "uniform", "round_robin"])
    p.add_argument("--lora", action="store_true",
                   help="Enable LoRA (mtl_lora). Default is no-lora (mtl_full_ft).")
    p.add_argument("--no-lora", dest="lora", action="store_false")
    p.set_defaults(lora=False)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
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
    setting = "mtl_lora" if a.lora else "mtl_full_ft"
    cfg = ExperimentConfig(
        name=a.name or f"{setting}_{a.model}",
        model_key=a.model, tasks=a.tasks, setting=setting,
        use_lora=a.lora, lora_r=a.lora_r, lora_alpha=a.lora_alpha,
        lora_dropout=a.lora_dropout, sampling=a.sampling,
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
    summary = run_mtl(cfg)
    print("\n=== DONE ===")
    print(f"Output dir: {cfg.output_dir}")
    print(f"Overall score: {summary.get('overall_score')}")


if __name__ == "__main__":
    main()
