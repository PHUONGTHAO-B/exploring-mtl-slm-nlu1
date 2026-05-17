"""
Run ONE SVM+TF-IDF baseline experiment.

Example:
    python -m scripts.run_svm --task qqp
"""
import argparse

from src.baselines.svm_tfidf import run_svm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["sst2", "qqp", "stsb"])
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--max_features", type=int, default=200_000)
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args, _ = p.parse_known_args()
    return args


def main():
    a = parse_args()
    out = a.output_dir or f"outputs/svm_{a.task}"
    run_svm(
        task_key=a.task, output_dir=out,
        max_features=a.max_features, ngram_max=a.ngram_max,
        C=a.C, seed=a.seed,
    )
    print(f"\n=== DONE ===\nOutput dir: {out}")


if __name__ == "__main__":
    main()
