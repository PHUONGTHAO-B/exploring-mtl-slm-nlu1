"""
Run ALL 22 experiments and consolidate into a single CSV.

Usage:
    python -m scripts.run_all                              # all 22
    python -m scripts.run_all --only stl                   # only 9 SLM STL
    python -m scripts.run_all --only mtl                   # only 7 MTL
    python -m scripts.run_all --only baselines             # only 6 baselines
    python -m scripts.run_all --skip baselines             # 16 main only
    python -m scripts.run_all --output_root outputs/v2
"""
import argparse
import gc
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import torch

from src.common.config import ExperimentConfig
from src.stl.trainer import run_stl
from src.mtl.trainer import run_mtl
from src.baselines.bilstm import run_bilstm
from src.baselines.svm_tfidf import run_svm


SLM_MODELS = ["albert", "distilbert", "minilm"]
TASKS = ["sst2", "qqp", "stsb"]


def build_stl_experiments(output_root: str) -> List[Dict[str, Any]]:
    out = []
    for m in SLM_MODELS:
        for t in TASKS:
            out.append({
                "kind": "stl",
                "cfg": ExperimentConfig(
                    name=f"stl_{m}_{t}", model_key=m, task_key=t,
                    setting="stl_full_ft",
                    output_dir=os.path.join(output_root, f"stl_{m}_{t}"),
                ),
            })
    return out


def build_mtl_experiments(output_root: str) -> List[Dict[str, Any]]:
    out = []
    # MTL + LoRA (3 SLMs)
    for m in SLM_MODELS:
        out.append({
            "kind": "mtl",
            "cfg": ExperimentConfig(
                name=f"mtl_lora_{m}", model_key=m, tasks=TASKS,
                use_lora=True, setting="mtl_lora",
                output_dir=os.path.join(output_root, f"mtl_lora_{m}"),
            ),
        })
    # MTL Full FT (3 SLMs)
    for m in SLM_MODELS:
        out.append({
            "kind": "mtl",
            "cfg": ExperimentConfig(
                name=f"mtl_full_{m}", model_key=m, tasks=TASKS,
                use_lora=False, setting="mtl_full_ft",
                output_dir=os.path.join(output_root, f"mtl_full_{m}"),
            ),
        })
    # MTL Full FT BERT-base (large baseline)
    out.append({
        "kind": "mtl",
        "cfg": ExperimentConfig(
            name="mtl_full_bertbase", model_key="bert-base", tasks=TASKS,
            use_lora=False, setting="mtl_full_ft",
            output_dir=os.path.join(output_root, "mtl_full_bertbase"),
        ),
    })
    return out


def build_baseline_experiments(output_root: str) -> List[Dict[str, Any]]:
    out = []
    for t in TASKS:
        out.append({
            "kind": "bilstm",
            "task": t,
            "output_dir": os.path.join(output_root, f"bilstm_{t}"),
        })
    for t in TASKS:
        out.append({
            "kind": "svm",
            "task": t,
            "output_dir": os.path.join(output_root, f"svm_{t}"),
        })
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_root", type=str, default="outputs")
    p.add_argument("--only", choices=["stl", "mtl", "baselines"], default=None)
    p.add_argument("--skip", choices=["stl", "mtl", "baselines"], default=None)
    args, _ = p.parse_known_args()
    return args


def main():
    a = parse_args()
    Path(a.output_root).mkdir(parents=True, exist_ok=True)

    groups = {
        "stl": build_stl_experiments(a.output_root),
        "mtl": build_mtl_experiments(a.output_root),
        "baselines": build_baseline_experiments(a.output_root),
    }
    selected = []
    for g, items in groups.items():
        if a.only is not None and g != a.only: continue
        if a.skip is not None and g == a.skip: continue
        selected.extend(items)

    print(f"Running {len(selected)} experiments → {a.output_root}\n")
    all_summaries = []
    for i, item in enumerate(selected, 1):
        kind = item["kind"]
        if kind == "stl":
            cfg = item["cfg"]
            print(f"\n[{i}/{len(selected)}] STL: {cfg.name}")
            try: s = run_stl(cfg)
            except Exception as e: s = {"exp_name": cfg.name, "error": str(e)}
        elif kind == "mtl":
            cfg = item["cfg"]
            print(f"\n[{i}/{len(selected)}] MTL: {cfg.name}")
            try: s = run_mtl(cfg)
            except Exception as e: s = {"exp_name": cfg.name, "error": str(e)}
        elif kind == "bilstm":
            print(f"\n[{i}/{len(selected)}] BiLSTM: {item['task']}")
            try: s = run_bilstm(task_key=item["task"], output_dir=item["output_dir"])
            except Exception as e: s = {"exp_name": f"bilstm_{item['task']}", "error": str(e)}
        elif kind == "svm":
            print(f"\n[{i}/{len(selected)}] SVM: {item['task']}")
            try: s = run_svm(task_key=item["task"], output_dir=item["output_dir"])
            except Exception as e: s = {"exp_name": f"svm_{item['task']}", "error": str(e)}
        all_summaries.append(s)
        # save consolidated CSV after each experiment
        pd.DataFrame(all_summaries).to_csv(
            os.path.join(a.output_root, "all_summary.csv"), index=False)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(a.output_root, "all_summary.json"), "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, default=str)
    print(f"\n=== DONE ===\nCSV: {a.output_root}/all_summary.csv")
    print(f"JSON: {a.output_root}/all_summary.json")


if __name__ == "__main__":
    main()
