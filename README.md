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
| LoRA target modules (ALBERT) | query, key, value |
| LoRA target modules (DistilBERT) | q_lin, k_lin, v_lin, out_lin |
| LoRA target modules (MiniLM) | query, key, value |

## Metrics captured per experiment

| Type | Metrics | Tasks |
|---|---|---|
| Classification | Accuracy, Macro-F1, Precision, Recall | SST-2, QQP |
| Regression | Pearson, Spearman | STS-B |
| Per-task score (Eq. 19) | task_score (½(Acc+F1) or ½(P+S)) | all |
| Overall score (Eq. 20) | overall_score (1/K × Σ task_score) | all |
| Model size | total_params, trainable_params | all |
| Train cost | total_train_time_sec, avg_time_per_epoch_sec | all |
| Train memory | train_peak_vram_mb | all |
| Inference cost | inference_latency_ms_per_sample, throughput_samples_per_sec | all |
| Inference memory | inference_peak_vram_mb | all |
| Hardware | gpu_name, total_vram_mb, cuda_version, cuda_capability, torch_version, platform | all |

---

# How to Run

## A. Setup server (first-time only)

### A.1 — SSH into server (if remote)

```bash
ssh user@your-server-ip
```

### A.2 — Clone repo + install

```bash
git clone https://github.com/PHUONGTHAO-B/exploring-mtl-slm-nlu1.git
cd exploring-mtl-slm-nlu1

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### A.3 — Verify GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('VRAM:', torch.cuda.get_device_properties(0).total_memory/1024**3 if torch.cuda.is_available() else 0, 'GB')"
```

Expect output like `CUDA: True` and the GPU name (e.g. `Tesla T4`).

---

## B. Run experiments

### B.1 — Run ALL 22 experiments (recommended)

```bash
source .venv/bin/activate
python -m scripts.run_all --output_root outputs/run1
```

Estimated time: **~8-12 hours on T4**.

### B.2 — Run by group

```bash
# Only 9 STL SLM (ALBERT/DistilBERT/MiniLM × 3 tasks)  ~3-4 h
python -m scripts.run_all --output_root outputs/run1 --only stl

# Only 7 MTL (3 LoRA + 3 Full SLM + 1 Full BERT-base)  ~3-5 h
python -m scripts.run_all --output_root outputs/run1 --only mtl

# Only 6 baselines (BiLSTM + SVM-TFIDF)  ~1-2 h
python -m scripts.run_all --output_root outputs/run1 --only baselines

# 16 main (skip baselines)
python -m scripts.run_all --output_root outputs/run1 --skip baselines
```

### B.3 — Run individual experiments

```bash
# STL: one model + one task
python -m scripts.run_stl --model albert     --task sst2  --output_dir outputs/run1/stl_albert_sst2
python -m scripts.run_stl --model albert     --task qqp   --output_dir outputs/run1/stl_albert_qqp
python -m scripts.run_stl --model albert     --task stsb  --output_dir outputs/run1/stl_albert_stsb
python -m scripts.run_stl --model distilbert --task sst2  --output_dir outputs/run1/stl_distilbert_sst2
python -m scripts.run_stl --model distilbert --task qqp   --output_dir outputs/run1/stl_distilbert_qqp
python -m scripts.run_stl --model distilbert --task stsb  --output_dir outputs/run1/stl_distilbert_stsb
python -m scripts.run_stl --model minilm     --task sst2  --output_dir outputs/run1/stl_minilm_sst2
python -m scripts.run_stl --model minilm     --task qqp   --output_dir outputs/run1/stl_minilm_qqp
python -m scripts.run_stl --model minilm     --task stsb  --output_dir outputs/run1/stl_minilm_stsb

# MTL + LoRA
python -m scripts.run_mtl --model albert     --lora --output_dir outputs/run1/mtl_lora_albert
python -m scripts.run_mtl --model distilbert --lora --output_dir outputs/run1/mtl_lora_distilbert
python -m scripts.run_mtl --model minilm     --lora --output_dir outputs/run1/mtl_lora_minilm

# MTL Full Fine-Tuning (no LoRA)
python -m scripts.run_mtl --model albert     --no-lora --output_dir outputs/run1/mtl_full_albert
python -m scripts.run_mtl --model distilbert --no-lora --output_dir outputs/run1/mtl_full_distilbert
python -m scripts.run_mtl --model minilm     --no-lora --output_dir outputs/run1/mtl_full_minilm
python -m scripts.run_mtl --model bert-base  --no-lora --output_dir outputs/run1/mtl_full_bertbase

# Baselines
python -m scripts.run_bilstm --task sst2 --output_dir outputs/run1/bilstm_sst2
python -m scripts.run_bilstm --task qqp  --output_dir outputs/run1/bilstm_qqp
python -m scripts.run_bilstm --task stsb --output_dir outputs/run1/bilstm_stsb
python -m scripts.run_svm    --task sst2 --output_dir outputs/run1/svm_sst2
python -m scripts.run_svm    --task qqp  --output_dir outputs/run1/svm_qqp
python -m scripts.run_svm    --task stsb --output_dir outputs/run1/svm_stsb
```

### B.4 — Quick benchmark only (no full training) — ~15 minutes

If you only need Table 4 (efficiency: VRAM + latency) without Table 3 (accuracy):

```bash
python -m scripts.benchmark_vram --output_dir outputs/benchmark
```

---

## C. Run in background (SSH-safe)

### Option 1 — `tmux` (RECOMMENDED)

```bash
sudo apt install tmux -y                                          # install if missing

tmux new -s exp                                                   # create session
source .venv/bin/activate
python -m scripts.run_all --output_root outputs/run1

# Detach: press Ctrl+B then D  (script keeps running)
# SSH may disconnect — script stays alive

tmux attach -t exp                                                # reconnect later
tmux ls                                                           # list sessions
tmux kill-session -t exp                                          # kill when done
```

### Option 2 — `nohup`

```bash
source .venv/bin/activate
nohup python -m scripts.run_all --output_root outputs/run1 > run.log 2>&1 &
echo "PID: $!"                                                    # remember PID

tail -f run.log                                                   # follow log
ps aux | grep run_all                                             # check still alive
kill <PID>                                                        # stop
```

---

## D. Monitor progress

In a second terminal / tmux pane:

```bash
# Consolidated CSV (updated after each exp completes)
cat outputs/run1/all_summary.csv | column -t -s,

# Live training log
tail -f outputs/run1/stl_albert_qqp/train.log

# GPU usage
watch -n 1 nvidia-smi

# Count completed experiments
ls outputs/run1/*/summary.json 2>/dev/null | wc -l
```

---

## E. Output structure

Each experiment writes to its own folder under `--output_root`:

```
outputs/run1/
├── all_summary.csv                  # 22 rows, all metrics, regenerated after each exp
├── all_summary.json
├── stl_albert_sst2/
│   ├── config.json                  # exact config used
│   ├── train.log                    # full training log
│   ├── history.csv                  # per-epoch metrics
│   ├── history.json
│   ├── summary.json                 # ← presence means "this exp is DONE"
│   ├── checkpoints/                 # keep last 2, auto-deleted after summary
│   ├── best_model/                  # HF save_pretrained
│   └── final_model/
├── ... (21 more directories)
```

### Resume / skip behavior

- If `summary.json` exists → experiment is **skipped** on rerun
- If only `checkpoints/epoch_N.pt` exists → training **resumes** from latest checkpoint
- Otherwise → train from scratch

You can safely re-run the same `run_all` command after a disconnect — it will only do the unfinished work.

---

## F. Fetch results to local machine

From your local machine:

```bash
# Just the consolidated CSV
scp user@your-server-ip:~/exploring-mtl-slm-nlu1/outputs/run1/all_summary.csv ./

# Entire outputs folder
scp -r user@your-server-ip:~/exploring-mtl-slm-nlu1/outputs/run1 ./outputs_local

# rsync (faster, resumable)
rsync -avz user@your-server-ip:~/exploring-mtl-slm-nlu1/outputs/ ./outputs_local/
```

Or open the CSV directly on server with `pandas`:

```python
import pandas as pd
df = pd.read_csv('outputs/run1/all_summary.csv')
print(df[['exp_name','mode','setting','overall_score',
          'total_params','trainable_params',
          'train_peak_vram_mb','inference_latency_ms_per_sample',
          'gpu_name']].to_string())
```

---

## G. Update code workflow

**On local machine** (after editing code):

```bash
cd C:\Users\admin\Downloads\codeneww\exploring-mtl-slm-nlu
git add .
git commit -m "Describe change"
git push
```

**On server:**

```bash
cd exploring-mtl-slm-nlu1
git pull
# Re-run — already-done experiments are auto-skipped
python -m scripts.run_all --output_root outputs/run1
```

---

## H. Hardware requirements

| Setup | GPU | VRAM | Notes |
|---|---|---|---|
| Minimum | NVIDIA T4 (Colab free) | 15 GB | All 22 experiments work, ~8-12h |
| Recommended | NVIDIA L4 / A100 | 24+ GB | 2-3× faster |
| CPU only | — | — | Only SVM+TF-IDF baseline runs reasonably |

---

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
│   ├── common/                     # config, data, logger, checkpoint, benchmark, metrics, utils
│   ├── stl/                        # STL model + trainer
│   ├── mtl/                        # MTL shared-encoder model + trainer
│   └── baselines/                  # BiLSTM, SVM+TF-IDF
└── scripts/
    ├── run_stl.py                  # 1 STL experiment
    ├── run_mtl.py                  # 1 MTL experiment (--lora / --no-lora)
    ├── run_bilstm.py               # 1 BiLSTM baseline
    ├── run_svm.py                  # 1 SVM baseline
    ├── run_all.py                  # ALL 22 experiments + consolidated CSV
    ├── benchmark_vram.py           # quick VRAM+latency only
    └── run_all.sh                  # bash wrapper
```

## License

MIT
