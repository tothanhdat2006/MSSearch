# How to Run MSSearch V2 Evaluation Benchmark

This guide outlines the standard 3-step workflow to run MSSearch V2 on **STASHBench** (336 samples) across **3 Vision-Language Models** and **4 Candidate Pool Sizes** ($N \in \{1, 5, 10, 20\}$), totaling **12 evaluation configurations**:

```
[Step 1] Configure Global Paths (env_paths.sh)
    │
    ▼
[Step 2] Download Data, Auxiliary Weights & VLMs
    │
    ▼
[Step 3] Run Main Evaluation Scripts (A6000 / A100 / Colab / Modal)
```

---

## Evaluation Matrix Overview (12 Configurations)

| Model Key | Full Model Identifier | Architecture | VRAM Requirement |
| :--- | :--- | :--- | :--- |
| `qwen3_vl_8b` | `Qwen/Qwen3-VL-8B-Instruct` | Native Dynamic ViT | ~24 GB (40GB/80GB GPU) |
| `llava_ov_7b` | `lmms-lab/llava-onevision-qwen2-7b-ov` | AnyRes Multi-Patch | ~24 GB (40GB/80GB GPU) |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B` | Dynamic High-Res ViT | ~24 GB (40GB/80GB GPU) |

### Candidate Pool Sizes ($N$):
- **`N_1`**: Ground-truth target image only (no distractors, $N=1$).
- **`N_5`**: Ground-truth target image + 4 distractor images ($N=5$).
- **`N_10`**: Ground-truth target image + 9 distractor images ($N=10$).
- **`N_20`**: Ground-truth target image + 19 distractor images ($N=20$).

---

## Step 1: Setup Global Environment & Paths (`env_paths.sh`)

All runner scripts, Slurm batch files, and Python entrypoints load paths from a single centralized file:  
`experiments/our/env_paths.sh`

Edit this file once with your cluster/workstation storage paths:

```bash
# ==============================================================================
# experiments/our/env_paths.sh
# ==============================================================================

# 1. Python Environment Activation
export CONDA_ENV_NAME="hrsearch"
export VENV_PATH=".../.venv/bin/activate"

# 2. Checkpoints Root & Auxiliary Models
export MODEL_DIR=".../models"
export SAM3_PATH=".../models/sam3/sam3.pt"
export SIGLIP_NAME="google/siglip2-so400m-patch16-512"
export NLP_MODEL=".../models/spacy/en_core_web_sm-3.8.0"

# 3. STASHBench Dataset & Pool Images
export DATASET_PATH=".../STASHBench/annotations_part1234_merged.json"
export IMAGE_DIR=".../STASHBench/pool"

# 4. Base Output Directory
export OUTPUT_BASE_DIR=""
```

*(If `/data/shared/...` paths do not exist, scripts automatically fall back to repo-relative paths: `<repo>/models/` and `<repo>/data/STASHBench/`)*.

---

## Step 2: Run Download Scripts for Data & Models

> **Important**: The evaluation runner scripts do **not** automatically download SAM3 checkpoint weights or datasets. Missing weights trigger `FileNotFoundError`. Pre-downloading ensures offline reproducibility and prevents runtime download latency.

Source your environment variables from Step 1, then run the download utilities:

```bash
# Source paths configured in Step 1
source experiments/our/env_paths.sh
```

### Option A: Download Everything in One Command
Downloads dataset annotations, pool images, auxiliary weights (SAM3, SigLIP, spaCy), and target models into the paths configured in `env_paths.sh`:

```bash
python experiments/download/download_all.py \
    --base_model_dir "${MODEL_DIR}" \
    --data_dir "$(dirname "${DATASET_PATH}")" \
    --models Qwen3-VL-8B LLaVA-OV-7B InternVL3-8B
```

### Option B: Download Step-by-Step

#### 1. Download STASHBench Dataset & Candidate Pool
```bash
python experiments/download/download_data.py \
    --data_dir "$(dirname "${DATASET_PATH}")"
```

#### 2. Download Auxiliary Models (SAM3, SigLIP, spaCy)
```bash
# Downloads sam3.pt, siglip2-so400m-patch16-512, and spacy model into MODEL_DIR
python experiments/download/download_auxiliary.py \
    --base_model_dir "${MODEL_DIR}"

# Optional: also download Grounding DINO and Florence-2 for detector ablations
python experiments/download/download_auxiliary.py \
    --base_model_dir "${MODEL_DIR}" \
    --download_ablations

# Ensure spaCy English pipeline is linked in Python environment
python -m spacy download en_core_web_sm
```

#### 3. Download Target Vision-Language Models (VLMs)
```bash
# Qwen3-VL-8B-Instruct
python experiments/download/download_model.py \
    --model_name Qwen3-VL-8B \
    --base_model_dir "${MODEL_DIR}"

# LLaVA-OneVision-7B
python experiments/download/download_model.py \
    --model_name LLaVA-OV-7B \
    --base_model_dir "${MODEL_DIR}"

# InternVL3-8B
python experiments/download/download_model.py \
    --model_name InternVL3-8B \
    --base_model_dir "${MODEL_DIR}"
```

---

## Step 3: Run Main Evaluation Scripts

Navigate to the repository root:
```bash
cd /path/to/MSSearch
```

Choose your execution target:

### 3.1 Local Workstation / Server (A6000 / Single GPU)

#### A. Run an Individual Model & Pool Size
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

#### B. Run All 3 Models on a Specific Pool Size
```bash
bash experiments/our/scripts/A6000/run_our_all_n1.sh
bash experiments/our/scripts/A6000/run_our_all_n5.sh
bash experiments/our/scripts/A6000/run_our_all_n10.sh
bash experiments/our/scripts/A6000/run_our_all_n20.sh
```

#### C. Run Full 12-Configuration Sweep
```bash
# Sequentially executes all 12 combinations
bash experiments/our/scripts/A6000/run_our_sweep.sh
```

#### Passing Runtime Options
Append CLI flags to any `.sh` script:
```bash
# Resume an interrupted run from existing JSONL output:
bash experiments/our/scripts/A6000/run_our_qwen2.5_vl_7b_n5.sh --resume

# Run a quick test slice (e.g. first 10 samples):
bash experiments/our/scripts/A6000/run_our_qwen2.5_vl_7b_n5.sh --start_idx 0 --end_idx 10
```

---

### 3.2 HPC Cluster (Slurm / A100)

#### A. Submit an Individual Slurm Job
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

#### B. Submit All 3 Models for a Pool Size
```bash
bash experiments/our/scripts/A100/run_our_all_n1.slurm
bash experiments/our/scripts/A100/run_our_all_n5.slurm
bash experiments/our/scripts/A100/run_our_all_n10.slurm
bash experiments/our/scripts/A100/run_our_all_n20.slurm
```

#### C. Submit Full Matrix Sweep
```bash
# Submits all 12 Slurm batch jobs to the cluster queue at once
bash experiments/our/scripts/A100/run_our_sweep.slurm
```

---

### 3.3 Google Colab Pro (`notebook/`)

Each notebook in `experiments/our/notebook/` is self-contained with automatic Google Drive mounting:

| Model | $N=1$ | $N=5$ | $N=10$ | $N=20$ |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-VL-7B** | `run_our_colab_qwen2.5_vl_7b_n1.ipynb` | `run_our_colab_qwen2.5_vl_7b_n5.ipynb` | `run_our_colab_qwen2.5_vl_7b_n10.ipynb` | `run_our_colab_qwen2.5_vl_7b_n20.ipynb` |
| **LLaVA-OV-7B** | `run_our_colab_llava_ov_7b_n1.ipynb` | `run_our_colab_llava_ov_7b_n5.ipynb` | `run_our_colab_llava_ov_7b_n10.ipynb` | `run_our_colab_llava_ov_7b_n20.ipynb` |
| **InternVL3-8B** | `run_our_colab_internvl3_8b_n1.ipynb` | `run_our_colab_internvl3_8b_n5.ipynb` | `run_our_colab_internvl3_8b_n10.ipynb` | `run_our_colab_internvl3_8b_n20.ipynb` |

**Execution Steps in Colab:**
1. Open the notebook in Google Colab (Select **Runtime > Change runtime type > A100 GPU**).
2. Set your GitHub personal access token under Colab Secrets (the 🔑 icon) as `GITHUB_TOKEN`.
3. Run **Cell 1** to mount Drive and auto-detect dataset folders.
4. Run **Cells 2–4** to clone code, install dependencies, and launch evaluation. Results stream directly to your Drive.

---

### 3.4 Modal Serverless Cloud (`scripts/modal/`)

Ensure Modal is authenticated (`modal setup`). Target GPU is automatically assigned as **L40S (48GB)**.

#### A. Foreground Execution (Live Terminal Logs)
```bash
bash experiments/our/scripts/modal/run_modal.sh qwen3_vl_8b N_5
bash experiments/our/scripts/modal/run_modal.sh llava_ov_7b N_5
bash experiments/our/scripts/modal/run_modal.sh internvl3_8b N_5
```

#### B. Background Execution (`--detach`)
Add `--detach` to run jobs remotely in the cloud without keeping your local terminal open:
```bash
# Launch detached job via wrapper
bash experiments/our/scripts/modal/run_modal.sh qwen3_vl_8b N_5 --detach

# Or direct Modal command (flag precedes script path):
modal run --detach experiments/our/scripts/modal/run_our_modal.py \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --pool-mode N_5
```

Check status and retrieve live logs:
```bash
modal app list
modal app logs <app-id>
```

#### C. Launch All 12 Modal Jobs in Background
```bash
for model in qwen3_vl_8b llava_ov_7b internvl3_8b; do
    for pool in N_1 N_5 N_10 N_20; do
        echo "Launching Modal job: ${model} (${pool})"
        bash experiments/our/scripts/modal/run_modal.sh "${model}" "${pool}" --detach
    done
done
```

---

## Step 4: Run Ablation Studies (`experiments/our/ablations/`)

The ablation suite dissects components along three independent axes on candidate pool $N=5$:

### 4.1 Detector Backbone Ablation (`run_ablation_detector.sh`)
Compares **SAM3** vs. **Grounding DINO** vs. **Florence-2** vs. **DINOv3** (with fixed SigLIP scorer):
```bash
bash experiments/our/ablations/scripts/run_ablation_detector.sh Qwen/Qwen3-VL-8B-Instruct N_5
```

### 4.2 Visual Screening Scorer Ablation (`run_ablation_scorer.sh`)
Compares **SigLIP** vs. **SigLIP2** (384px, 512px) vs. **CLIP-ViT-B/32** vs. **CLIP-ViT-L/14** vs. **EVA-02-L/14** (with fixed SAM3 detector):
```bash
bash experiments/our/ablations/scripts/run_ablation_scorer.sh Qwen/Qwen3-VL-8B-Instruct N_5
```

### 4.3 VLM Gating Threshold Ablation (`run_ablation_gate_threshold.sh`)
Sweeps probability thresholds **0.3** (high recall), **0.7** (high precision), and **0.5** (balanced baseline):
```bash
bash experiments/our/ablations/scripts/run_ablation_gate_threshold.sh Qwen/Qwen3-VL-8B-Instruct N_5
```

### 4.4 Full Matrix Sweep (`run_ablation_matrix.sh` or Slurm)
```bash
# Local execution
bash experiments/our/ablations/scripts/run_ablation_matrix.sh Qwen/Qwen3-VL-8B-Instruct

# HPC Slurm batch submission
sbatch experiments/our/ablations/scripts/run_ablation_slurm.slurm
```

### 4.5 Cloud Ablation on Modal (`run_modal_ablation.sh`)
Execute any ablation sweep in the cloud using persistent volumes (`mssearch-data`, `mssearch-models`, `mssearch-results` at `/results/ablation`):
```bash
# Detector sweep (SAM3, Grounding DINO, Florence-2, DINOv3)
bash experiments/our/ablations/scripts/run_modal_ablation.sh detector Qwen/Qwen3-VL-8B-Instruct N_5 --detach

# Visual screening scorer sweep (SigLIP, SigLIP2-384, SigLIP2-512, CLIP-B/32, CLIP-L/14, EVA-02-L/14)
bash experiments/our/ablations/scripts/run_modal_ablation.sh scorer Qwen/Qwen3-VL-8B-Instruct N_5 --detach

# Gating threshold sweep (0.3, 0.7, 0.5)
bash experiments/our/ablations/scripts/run_modal_ablation.sh gate Qwen/Qwen3-VL-8B-Instruct N_5 --detach

# Full cross-ablation sweep
bash experiments/our/ablations/scripts/run_modal_ablation.sh all Qwen/Qwen3-VL-8B-Instruct N_5 --detach
```


---

## Step 5: Evaluate Results (`eval_stashbench.py`)

Run the standardized evaluator on any output `.jsonl` file to compute Set-based vs. Strict Primary Image Accuracy, Answer Hit, Joint Accuracy, token usage, and stage-wise latencies:

```bash
# Evaluate a single results file
python mssearch/eval/eval_stashbench.py --input results_our/results_our_Qwen2.5-VL-7B-Instruct_N_5_*.jsonl

# Save structured summary JSON
python mssearch/eval/eval_stashbench.py \
    --input results_our/results_our_Qwen2.5-VL-7B-Instruct_N_5_*.jsonl \
    --save_summary results_our/summary_qwen_n5.json
```

