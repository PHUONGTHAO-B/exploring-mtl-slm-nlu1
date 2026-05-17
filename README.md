# Exploring Multi-Task Learning Using Small Language Models with LoRA for NLU

Reproduces the experiments from the paper:
**"Exploring Multi-Task Learning Using Small Language Models with LoRA for Natural Language Understanding Tasks"** (Thao N.P. & Le Q.-H., Quy Nhon University).

## Experiments (22 total)

### STL — Single-Task Learning (15)
- **SLM Full Fine-Tuning (9)**: ALBERT / DistilBERT / MiniLM × {SST-2, QQP, STS-B}
- **BiLSTM baseline (3)**: × {SST-2, QQP, STS-B}
- **SVM + TF-IDF baseline (3)**: × {SST-2, QQP, STS-B}

### MTL — Multi-Task Learning (7)
- **MTL + LoRA (3)**: ALBERT / DistilBERT / MiniLM with shared encoder + 3 heads + LoRA
- **MTL Full Fine-Tuning SLM (3)**: ALBERT / DistilBERT / MiniLM with shared encoder + 3 heads (no LoRA)
- **MTL Full Fine-Tuning BERT-base (1)**: large-capacity baseline

## Configuration (from paper Table 2)

| Component | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-5 (or 1e-5) |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| LR scheduler | Linear with warmup |
| Batch size | 16 |
| Max sequence length | 256 |
| Grad clipping | 1.0 |
| Mixed precision | fp16 |
| Early stopping patience | 3 |
| Checkpoint policy | Keep last 2 |
| Random seed | 42 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |

## Quick start

### 1. Clone & install

```bash
git clone <your_repo_url>
cd exploring-mtl-slm-nlu
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Run a single experiment

```bash
# STL: 1 SLM, 1 task
python -m scripts.run_stl --model albert --task sst2 --output_dir outputs/stl_albert_sst2

# MTL with LoRA
python -m scripts.run_mtl --model distilbert --lora --output_dir outputs/mtl_lora_distilbert

# MTL full fine-tuning
python -m scripts.run_mtl --model bert-base --no-lora --output_dir outputs/mtl_full_bertbase

# BiLSTM baseline
python -m scripts.run_bilstm --task sst2 --output_dir outputs/bilstm_sst2

# SVM + TF-IDF baseline
python -m scripts.run_svm --task qqp --output_dir outputs/svm_qqp
```

### 3. Run all 22 experiments

```bash
bash scripts/run_all.sh                       # Linux/Mac
# powershell -File scripts/run_all.ps1        # Windows
```

Outputs are saved to `outputs/<exp_name>/` by default. Override with `--output_dir`.

### 4. Quick benchmark only (VRAM + inference latency)

If you only need efficiency metrics (Table 4), without retraining:

```bash
python -m scripts.benchmark_vram --output_dir outputs/benchmark
```

This loads each model architecture, runs ~50 train steps + ~1000 inference samples, and dumps a single CSV (~15 minutes for all 22).

## Output structure per experiment

```
outputs/<exp_name>/
├── config.json
├── train.log
├── history.csv
├── history.json
├── benchmark.json
├── summary.json                  # written when experiment completes
├── checkpoints/                  # keep 2 latest, removed after summary
│   ├── checkpoint-step-XXXXXXX/
│   └── checkpoint-step-XXXXXXX/
├── best_model/                   # HF save_pretrained
└── final_model/
```

If `summary.json` exists, the experiment is skipped on rerun.
If only `checkpoint-step-*/` exists, training resumes from the latest checkpoint.

## Repository layout

```
exploring-mtl-slm-nlu/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── stl_experiments.yaml
│   ├── mtl_experiments.yaml
│   └── baseline_experiments.yaml
├── src/
│   ├── common/
│   │   ├── config.py
│   │   ├── data.py
│   │   ├── logger.py
│   │   ├── checkpoint.py
│   │   ├── benchmark.py
│   │   ├── metrics.py
│   │   └── utils.py
│   ├── stl/
│   │   ├── model.py
│   │   └── trainer.py
│   ├── mtl/
│   │   ├── model.py
│   │   └── trainer.py
│   └── baselines/
│       ├── bilstm.py
│       └── svm_tfidf.py
├── scripts/
│   ├── run_stl.py
│   ├── run_mtl.py
│   ├── run_bilstm.py
│   ├── run_svm.py
│   ├── benchmark_vram.py
│   └── run_all.sh
└── outputs/                      # gitignored
```

## Hardware requirements

| Setup | GPU | VRAM | Notes |
|---|---|---|---|
| Minimum | NVIDIA T4 (Colab free) | 15 GB | OK for all experiments |
| Recommended | NVIDIA L4 / A100 | 24+ GB | 2-3× faster |
| CPU only | — | — | Only SVM+TF-IDF baseline runs reasonably |

## License

MIT
