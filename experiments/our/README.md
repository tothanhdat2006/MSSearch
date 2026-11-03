# MSSearch V2 Evaluation Suite (200 Samples)

Evaluation suite for the unified Multi-Source Visual Search pipeline (`METHOD_SPECIFICATION_V2.md`) on `200_samples.json` under candidate pool `N_20` (default) across varying ground-truth evidence cardinalities ($K \in [1, 5]$).

---

## 1. Supported Models (8 Models)

1. `Qwen2.5-VL-7B`: `Qwen/Qwen2.5-VL-7B-Instruct`
2. `LLaVA-OV-7B`: `lmms-lab/llava-onevision-qwen2-7b-ov`
3. `InternVL2.5-8B`: `OpenGVLab/InternVL2.5-8B`
4. `Qwen2.5-VL-3B`: `Qwen/Qwen2.5-VL-3B-Instruct`
5. `Qwen3-VL-2B`: `Qwen/Qwen3-VL-2B-Instruct`
6. `Qwen3-VL-4B`: `Qwen/Qwen3-VL-4B-Instruct`
7. `Qwen3-VL-8B`: `Qwen/Qwen3-VL-8B-Instruct`
8. `Qwen3-VL-32B`: `Qwen/Qwen3-VL-32B-Instruct`

---

## 2. Directory & Modular File Architecture

- `eval_dataset.py`: Slices dataset and resolves raw images `<image_id>.jpg` (<250 lines).
- `eval_metrics.py`: Computes token precision, recall, F1, and exact match (<250 lines).
- `eval_models.py`: Engine orchestrator loading VLM, SAM3 segmentor, SigLIP zero-shot scorer, and spaCy NLP pipeline (<250 lines).
- `eval_runner.py`: Per-sample execution loop with per-sample atomic JSONL disk flushing (`flush()` + `os.fsync()`) and task resumption (`--resume`) (<250 lines).
- `run_our.sh`: Local bash runner for A6000 server with `# EDIT THIS PATH` sections. Default: `POOL_MODE="N_20"`.
- `run_our.slurm`: Slurm batch launcher for A100 HPC clusters (`#SBATCH --gres=gpu:a100:1`). Default: `POOL_MODE="N_20"`.
- `run_our_colab.ipynb`: Google Colab Pro notebook with parameterizable Google Drive mounting (`DRIVE_ROOT`).
- `run_our_modal.py`: Modal serverless runner targeting A10G (for $\le 8\text{B}$) and L40S (for $32\text{B}$) with persistent Modal Volumes (`mssearch-data`, `mssearch-models`, `mssearch-results`).
- `config_default.json`: Production configuration implementing `METHOD_SPECIFICATION_V2.md` (continuous enclosure sub-canvases, prompt directed clean standard `<image>\n<question>`, multi-answer attribute consolidation).

---

## 3. How to Run

### A. Local A6000 Server (`.sh`)
```bash
# Evaluate Qwen2.5-VL-7B-Instruct with default N_20 pool
bash run_our.sh Qwen2.5-VL-7B-Instruct

# Resume interrupted evaluation
bash run_our.sh Qwen2.5-VL-7B-Instruct --resume
```

### B. A100 HPC Cluster (`.slurm`)
```bash
sbatch run_our.slurm Qwen2.5-VL-7B-Instruct
sbatch run_our.slurm Qwen3-VL-8B Qwen3-VL-32B
```

### C. Google Colab Pro (`.ipynb`)
Open `run_our_colab.ipynb`, configure `DRIVE_ROOT = "/content/drive/MyDrive/cvpr_research"` in Cell 1, and run all cells.

### D. Modal Cloud (`run_our_modal.py`)
```bash
# Run on Modal serverless A10G/L40S
modal run run_our_modal.py --model Qwen/Qwen2.5-VL-7B-Instruct --pool-mode N_20

# Run specific slice
modal run run_our_modal.py --model Qwen/Qwen3-VL-8B-Instruct --start-idx 0 --end-idx 20
```
