# How to Run MSSearch V2 Evaluation Benchmark

This guide summarizes execution of the complete MSSearch V2 benchmark on **STASHBench** (336 samples) across **3 Vision-Language Models** and **4 Candidate Pool Sizes** ($N \in \{1, 5, 10, 20\}$), totaling **12 evaluation configurations**.

---

## 1. Matrix Overview (3 Models $\times$ 4 Pool Sizes = 12 Configurations)

| Model Key | Full Model Identifier | Architecture | VRAM Requirement |
| :--- | :--- | :--- | :--- |
| `qwen2.5_vl_7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | Native Dynamic ViT | ~22 GB (40GB/80GB GPU) |
| `llava_ov_7b` | `lmms-lab/llava-onevision-qwen2-7b-ov` | AnyRes Multi-Patch | ~24 GB (40GB/80GB GPU) |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B` | Dynamic High-Res ViT | ~24 GB (40GB/80GB GPU) |

### Pool Sizes ($N$):
- **`N_1`**: Ground-truth image only (no distractors, $N=1$).
- **`N_5`**: Ground-truth image + 4 distractors ($N=5$).
- **`N_10`**: Ground-truth image + 9 distractors ($N=10$).
- **`N_20`**: Ground-truth image + 19 distractors ($N=20$).

---

## 2. One-Time Setup: Configure `env_paths.sh`

All local and HPC scripts load paths from a single file:  
`experiments/our/env_paths.sh`

Edit this file once with your local dataset and model paths:
```bash
# 1. Python Environment
export CONDA_ENV_NAME="hrsearch"
export VENV_PATH="/data/shared/cvpr/dat/.venv/bin/activate"

# 2. Checkpoints
export MODEL_DIR="/data/shared/cvpr/dat/models"
export SAM3_PATH="${MODEL_DIR}/sam3/sam3.pt"
export SIGLIP_NAME="${MODEL_DIR}/siglip-base-patch16-384"
export NLP_MODEL="${MODEL_DIR}/spacy/en_core_web_sm-3.8.0"

# 3. STASHBench Dataset
export DATASET_PATH="/data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json"
export IMAGE_DIR="/data/shared/cvpr/dat/simple/data/STASHBench/pool"
```
*(If `/data/shared/...` is not found, scripts automatically fall back to repo-relative paths: `<repo>/models/` and `<repo>/data/STASHBench/`)*.

---

## 3. Option A: Local Workstation / Server (Bash / A6000)

Navigate to the repository root:
```bash
cd /path/to/MSSearch
```

### A. Run an Individual Configuration
```bash
# Qwen2.5-VL-7B
bash experiments/our/scripts/A6000/run_our_qwen2.5_vl_7b_n1.sh
bash experiments/our/scripts/A6000/run_our_qwen2.5_vl_7b_n5.sh
bash experiments/our/scripts/A6000/run_our_qwen2.5_vl_7b_n10.sh
bash experiments/our/scripts/A6000/run_our_qwen2.5_vl_7b_n20.sh

# LLaVA-OneVision-7B
bash experiments/our/scripts/A6000/run_our_llava_ov_7b_n1.sh
bash experiments/our/scripts/A6000/run_our_llava_ov_7b_n5.sh
bash experiments/our/scripts/A6000/run_our_llava_ov_7b_n10.sh
bash experiments/our/scripts/A6000/run_our_llava_ov_7b_n20.sh

# InternVL3-8B
bash experiments/our/scripts/A6000/run_our_internvl3_8b_n1.sh
bash experiments/our/scripts/A6000/run_our_internvl3_8b_n5.sh
bash experiments/our/scripts/A6000/run_our_internvl3_8b_n10.sh
bash experiments/our/scripts/A6000/run_our_internvl3_8b_n20.sh
```

### B. Run All 3 Models on a Specific Pool Size
```bash
# Sequentially runs Qwen, LLaVA, and InternVL on N=1
bash experiments/our/scripts/A6000/run_our_all_n1.sh

# Sequentially runs Qwen, LLaVA, and InternVL on N=5
bash experiments/our/scripts/A6000/run_our_all_n5.sh

# Sequentially runs Qwen, LLaVA, and InternVL on N=10
bash experiments/our/scripts/A6000/run_our_all_n10.sh

# Sequentially runs Qwen, LLaVA, and InternVL on N=20
bash experiments/our/scripts/A6000/run_our_all_n20.sh
```

### C. Run Full 12-Configuration Matrix Sweep
```bash
# Sequentially executes all 12 configurations across N=1, 5, 10, 20
bash experiments/our/scripts/A6000/run_our_sweep.sh
```

### Passing Runtime Options (Resume & Subsetting)
Append CLI flags to any `.sh` script:
```bash
# Resume an interrupted run from existing JSONL log:
bash experiments/our/scripts/A6000/run_our_qwen2.5_vl_7b_n5.sh --resume

# Run a quick test slice (e.g. first 20 samples):
bash experiments/our/scripts/A6000/run_our_qwen2.5_vl_7b_n5.sh --start_idx 0 --end_idx 20
```

---

## 4. Option B: HPC Cluster (Slurm / A100)

Ensure you have access to a node with at least 1x A100 (40GB/80GB):

### A. Submit an Individual Slurm Job
```bash
# Qwen2.5-VL-7B
sbatch experiments/our/scripts/A100/run_our_qwen2.5_vl_7b_n1.slurm
sbatch experiments/our/scripts/A100/run_our_qwen2.5_vl_7b_n5.slurm
sbatch experiments/our/scripts/A100/run_our_qwen2.5_vl_7b_n10.slurm
sbatch experiments/our/scripts/A100/run_our_qwen2.5_vl_7b_n20.slurm

# LLaVA-OneVision-7B
sbatch experiments/our/scripts/A100/run_our_llava_ov_7b_n1.slurm
sbatch experiments/our/scripts/A100/run_our_llava_ov_7b_n5.slurm
sbatch experiments/our/scripts/A100/run_our_llava_ov_7b_n10.slurm
sbatch experiments/our/scripts/A100/run_our_llava_ov_7b_n20.slurm

# InternVL3-8B
sbatch experiments/our/scripts/A100/run_our_internvl3_8b_n1.slurm
sbatch experiments/our/scripts/A100/run_our_internvl3_8b_n5.slurm
sbatch experiments/our/scripts/A100/run_our_internvl3_8b_n10.slurm
sbatch experiments/our/scripts/A100/run_our_internvl3_8b_n20.slurm
```

### B. Submit All 3 Models for a Pool Size
```bash
bash experiments/our/scripts/A100/run_our_all_n1.slurm
bash experiments/our/scripts/A100/run_our_all_n5.slurm
bash experiments/our/scripts/A100/run_our_all_n10.slurm
bash experiments/our/scripts/A100/run_our_all_n20.slurm
```

### C. Submit Full Matrix Sweep
```bash
# Submits all 12 Slurm batch jobs to the cluster queue at once
bash experiments/our/scripts/A100/run_our_sweep.slurm
```

---

## 5. Option C: Google Colab Pro (`notebook/`)

Each notebook in `experiments/our/notebook/` is fully self-contained for a specific model and pool size:

| Model | $N=1$ | $N=5$ | $N=10$ | $N=20$ |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-VL-7B** | `run_our_colab_qwen2.5_vl_7b_n1.ipynb` | `run_our_colab_qwen2.5_vl_7b_n5.ipynb` | `run_our_colab_qwen2.5_vl_7b_n10.ipynb` | `run_our_colab_qwen2.5_vl_7b_n20.ipynb` |
| **LLaVA-OV-7B** | `run_our_colab_llava_ov_7b_n1.ipynb` | `run_our_colab_llava_ov_7b_n5.ipynb` | `run_our_colab_llava_ov_7b_n10.ipynb` | `run_our_colab_llava_ov_7b_n20.ipynb` |
| **InternVL3-8B** | `run_our_colab_internvl3_8b_n1.ipynb` | `run_our_colab_internvl3_8b_n5.ipynb` | `run_our_colab_internvl3_8b_n10.ipynb` | `run_our_colab_internvl3_8b_n20.ipynb` |

**Steps in Colab:**
1. Open the notebook in Google Colab (Select **Runtime > Change runtime type > A100 GPU**).
2. Store your GitHub token under Secrets (the key 🔑 icon) as `GITHUB_TOKEN`.
3. **Run Cell 1**: Mounts Google Drive and auto-detects `annotations_part1234_merged.json` and `pool/`.
4. **Run Cell 2**: Clones branch `draft/improve_all`.
5. **Run Cell 3**: Installs dependencies and downloads SAM3 weights if missing.
6. **Run Cell 4**: Starts evaluation. Results are written directly to Google Drive.

---

## 6. Option D: Modal Serverless Cloud (`scripts/modal/`)

Ensure `modal` is configured (`modal setup` or token set). Target GPU is automatically assigned as **L40S (48GB)**.

### A. Run in Foreground (Live Streaming Logs)
```bash
# Using the run_modal.sh wrapper
bash experiments/our/scripts/modal/run_modal.sh qwen2.5_vl_7b N_5
bash experiments/our/scripts/modal/run_modal.sh llava_ov_7b N_5
bash experiments/our/scripts/modal/run_modal.sh internvl3_8b N_5
```

### B. Run Asynchronously in Background (`--detach`)
Add `--detach` to run in the cloud without keeping your local terminal open:
```bash
# Launch background job
bash experiments/our/scripts/modal/run_modal.sh qwen2.5_vl_7b N_5 --detach

# Or direct Modal command (flag must precede script path):
modal run --detach experiments/our/scripts/modal/run_our_modal.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --pool-mode N_5
```

Check status and retrieve logs anytime:
```bash
modal app list
modal app logs <app-id>
```

### C. Launch All 12 Modal Jobs in Background
```bash
for model in qwen2.5_vl_7b llava_ov_7b internvl3_8b; do
    for pool in N_1 N_5 N_10 N_20; do
        echo "Spawning Modal run: ${model} (${pool})"
        bash experiments/our/scripts/modal/run_modal.sh "${model}" "${pool}" --detach
    done
done
```

---

## 7. Evaluating Predictions & Summarizing Metrics

After a run completes, evaluate the resulting `.jsonl` file against STASHBench ground truth:

```bash
python mssearch/eval/eval_stashbench.py \
    --results_file experiments/our/results_our_a6000_Qwen2.5-VL-7B-Instruct_N_5_*.jsonl \
    --dataset_file data/STASHBench/annotations_part1234_merged.json
```

### Output Report
The evaluator prints two sets of accuracy metrics:
1. **Set-Based Evidence Accuracy**:
   - `Image Accuracy (Set)`: The ground-truth image is included anywhere in `telemetry.evidence_images`.
   - `Joint Accuracy (Set)`: Prediction text matches ground truth AND the GT image is in the evidence set.
2. **Strict Primary Attribution Accuracy**:
   - `Strict Primary Image Accuracy`: The primary answer was derived specifically from the ground-truth image.
   - `Strict Primary Joint Accuracy`: Prediction text is correct AND the primary evidence image matches the GT image.
