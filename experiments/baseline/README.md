# Baseline VLM Evaluation Suite (200 Samples)

Decomposed, modular evaluation suite for benchmarking 17 Vision-Language Model baselines on `200_samples.json` under candidate pool size `N_20` (default) across varying ground-truth evidence cardinalities ($K \in [1, 5]$).

---

## 1. Supported Baseline Models (17 Models)

1. **Qwen3-VL Series**:
   - `Qwen3-VL-2B`: `Qwen/Qwen3-VL-2B-Instruct`
   - `Qwen3-VL-4B`: `Qwen/Qwen3-VL-4B-Instruct`
   - `Qwen3-VL-8B`: `Qwen/Qwen3-VL-8B-Instruct`
   - `Qwen3-VL-32B`: `Qwen/Qwen3-VL-32B-Instruct`
2. **Qwen2.5-VL Series**:
   - `Qwen2.5-VL-3B`: `Qwen/Qwen2.5-VL-3B-Instruct`
   - `Qwen2.5-VL-7B`: `Qwen/Qwen2.5-VL-7B-Instruct`
   - `Qwen2.5-VL-32B` / `72B`: `Qwen/Qwen2.5-VL-72B-Instruct`
3. **LLaVA-v1.6 Series (LLaVA-NeXT)**:
   - `LLaVA-v1.6-7B`: `llava-hf/llava-v1.6-vicuna-7b-hf`
   - `LLaVA-v1.6-13B`: `llava-hf/llava-v1.6-vicuna-13b-hf`
   - `LLaVA-v1.6-34B`: `llava-hf/llava-v1.6-34b-hf`
4. **LLaVA-HR-X Series**:
   - `LLaVA-HR-X-7B`: `favor123/llava-hr-7b-sft-1024`
   - `LLaVA-HR-X-13B`: `favor123/llava-hr-13b-x-sft-1024`
5. **LLaVA-OneVision Series**:
   - `LLaVA-OV-7B`: `lmms-lab/llava-onevision-qwen2-7b-ov`
6. **InternVL Series**:
   - `InternVL2.5-8B`: `OpenGVLab/InternVL2.5-8B`
   - `InternVL3-38B`: `OpenGVLab/InternVL3-38B`
7. **Yi-VL Series**:
   - `Yi-VL-34B`: `01-ai/Yi-VL-34B`

---

## 2. Directory & Modular File Architecture

- `eval_dataset.py`: Slices dataset and resolves raw images `<image_id>.jpg` (<250 lines).
- `eval_metrics.py`: Computes token precision, recall, F1, and exact match (<250 lines).
- `eval_models.py`: Model loader, native multi-image routing, vertical multi-panel canvas builder with max-width scaling, 20px white divider band, 2px borders, top-down numbering, and one-by-one mode (<250 lines).
- `eval_runner.py`: Per-sample execution loop with per-sample atomic JSONL disk flushing (`flush()` + `os.fsync()`) and task resumption (`--resume`) (<250 lines).
- `run_baseline.sh`: Local bash runner for A6000 server with prominent `# EDIT THIS PATH` sections. Default: `POOL_MODE="N_20"`.
- `run_baseline.slurm`: Slurm batch launcher for A100 HPC clusters (`#SBATCH --gres=gpu:a100:1`). Default: `POOL_MODE="N_20"`.
- `run_baseline_colab.ipynb`: Google Colab Pro notebook with parameterizable Google Drive mounting (`DRIVE_ROOT`).
- `run_baseline_modal.py`: Modal serverless runner targeting A10G (for $\le 8\text{B}$) and L40S (for $13\text{B}\text{--}38\text{B}$) with persistent Modal Volumes (`mssearch-data`, `mssearch-models`, `mssearch-results`).
- `config_default.json`: Baseline execution configuration.

---

## 3. How to Run

### A. Local A6000 Server (`.sh`)
```bash
# Evaluate Qwen2.5-VL-7B-Instruct with default N_20 pool
bash run_baseline.sh Qwen2.5-VL-7B-Instruct

# Evaluate with one-by-one strategy
bash run_baseline.sh Qwen2.5-VL-7B-Instruct --eval_strategy one_by_one

# Resume interrupted evaluation
bash run_baseline.sh Qwen2.5-VL-7B-Instruct --resume
```

### B. A100 HPC Cluster (`.slurm`)
```bash
sbatch run_baseline.slurm Qwen2.5-VL-7B-Instruct
sbatch run_baseline.slurm Yi-VL-34B LLaVA-v1.6-34B
```

### C. Google Colab Pro (`.ipynb`)
Open `run_baseline_colab.ipynb`, set your `DRIVE_ROOT = "/content/drive/MyDrive/cvpr_research"` in Cell 1, and run all cells.

### D. Modal Cloud (`run_baseline_modal.py`)
```bash
# Run on Modal serverless A10G/L40S
modal run run_baseline_modal.py --model Qwen/Qwen2.5-VL-7B-Instruct --pool-mode N_20

# Run specific slice
modal run run_baseline_modal.py --model 01-ai/Yi-VL-34B --start-idx 0 --end-idx 20
```
