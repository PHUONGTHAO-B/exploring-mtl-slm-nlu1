"""
Run ONE BiLSTM baseline experiment.

Example:
    python -m scripts.run_bilstm --task sst2
"""
import argparse

from src.baselines.bilstm import run_bilstm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["sst2", "qqp", "stsb"])
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=300)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=2)
    args, _ = p.parse_known_args()
    return args


def main():
    a = parse_args()
    out = a.output_dir or f"outputs/bilstm_{a.task}"
    run_bilstm(
        task_key=a.task, output_dir=out,
        num_epochs=a.num_epochs, batch_size=a.batch_size, lr=a.lr,
        max_len=a.max_len, emb_dim=a.emb_dim, hidden=a.hidden,
        seed=a.seed, patience=a.patience,
        weight_decay=a.weight_decay, num_workers=a.num_workers,
    )
    print(f"\n=== DONE ===\nOutput dir: {out}")


if __name__ == "__main__":
    main()
