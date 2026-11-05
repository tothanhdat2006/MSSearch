# MSSearch V2 Evaluation Suite (STASHBench)

Evaluation suite for the unified Multi-Source Visual Search pipeline (`METHOD_SPECIFICATION_V2.md`) on STASHBench (`annotations_part1234_merged.json`, 336 samples) across candidate pools $N \in \{1, 5, 10, 20\}$ and target models.

---

## 1. Directory Structure

```text
experiments/our/
├── config_default.json       # Production hyperparameter configuration
├── eval_dataset.py           # Dataset slicing & deterministic image pool loader
├── eval_metrics.py           # Dual metrics (Set-based vs Strict Primary attribution)
├── eval_models.py            # Engine loader (VLM, SAM3, SigLIP, spaCy) & pipeline caller
├── eval_runner.py            # CLI evaluation runner with per-sample atomic flush
├── requirements.txt          # Python dependencies
├── README.md                 # Evaluation suite documentation (this file)
├── notebook/                 # Google Colab notebooks (12 model x pool configurations)
│   ├── run_our_colab_qwen2.5_vl_7b_n1.ipynb
│   ├── run_our_colab_qwen2.5_vl_7b_n5.ipynb
│   ├── run_our_colab_qwen2.5_vl_7b_n10.ipynb
│   ├── run_our_colab_qwen2.5_vl_7b_n20.ipynb
│   ├── run_our_colab_llava_ov_7b_n1.ipynb
│   ├── run_our_colab_llava_ov_7b_n5.ipynb
│   ├── run_our_colab_llava_ov_7b_n10.ipynb
│   ├── run_our_colab_llava_ov_7b_n20.ipynb
│   ├── run_our_colab_internvl3_8b_n1.ipynb
│   ├── run_our_colab_internvl3_8b_n5.ipynb
│   ├── run_our_colab_internvl3_8b_n10.ipynb
│   └── run_our_colab_internvl3_8b_n20.ipynb
└── scripts/
    ├── A6000/                # Local workstation / server Bash runner scripts (.sh)
    │   ├── run_our_qwen2.5_vl_7b_n1.sh ... run_our_qwen2.5_vl_7b_n20.sh
    │   ├── run_our_llava_ov_7b_n1.sh ... run_our_llava_ov_7b_n20.sh
    │   ├── run_our_internvl3_8b_n1.sh ... run_our_internvl3_8b_n20.sh
    │   ├── run_our_all_n1.sh ... run_our_all_n20.sh
    │   └── run_our_sweep.sh
    ├── A100/                 # HPC Slurm batch submission scripts (.slurm)
    │   ├── run_our_qwen2.5_vl_7b_n1.slurm ... run_our_qwen2.5_vl_7b_n20.slurm
    │   ├── run_our_llava_ov_7b_n1.slurm ... run_our_llava_ov_7b_n20.slurm
    │   ├── run_our_internvl3_8b_n1.slurm ... run_our_internvl3_8b_n20.slurm
    │   ├── run_our_all_n1.slurm ... run_our_all_n20.slurm
    │   └── run_our_sweep.slurm
    └── modal/                # Serverless cloud GPU runners (Modal)
        ├── run_our_modal.py
        └── run_modal.sh
```

---

## 2. Core Python Components

- **`eval_dataset.py`** (<250 lines): Loads STASHBench annotations, slices dataset ranges (`--start_idx`, `--end_idx`, `--max_samples`), and resolves pool images deterministically using `gallery_order_seed42.json`. Enforces `ImageOps.exif_transpose` and `MAX_IMAGE_PIXELS = None`.
- **`eval_metrics.py`** (<250 lines): Computes text answer metrics (accuracy, token precision, recall, F1) and tracks Stage 1-6 telemetry: target retention, background rejection, visual token budget, and crop sizes.
- **`eval_models.py`** (<250 lines): Loads perception engines (`ModelQwenVL`, `ModelGlobalLocal`, `ModelInternvl`, `sam3_inference`, `SigLIPScorer`, spaCy). Fails loudly without fallback masking if dependencies or weights are missing.
- **`eval_runner.py`** (<250 lines): Orchestrates benchmark evaluation over candidate images, auto-saves timestamped config snapshots, and atomically flushes results line-by-line (`flush()` + `os.fsync()`) to prevent data loss.
- **`config_default.json`**: Master configuration setting `entropy_threshold: 0.35`, `max_patches_per_image: 64`, `max_partition_depth: 4`, and clean foveated prompt reading (`<image>\n<question>`).

---

## 3. Supported Foundation Models

| Short Key | Model Identifier | Architecture | VRAM Required |
| :--- | :--- | :--- | :--- |
| `qwen2.5_vl_7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | Native Dynamic Resolution Vision Transformer | ~22 GB (40GB/80GB GPU) |
| `llava_ov_7b` | `lmms-lab/llava-onevision-qwen2-7b-ov` | AnyRes Multi-Patch Vision Transformer | ~24 GB (40GB/80GB GPU) |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B` | Dynamic High-Resolution ViT | ~24 GB (40GB/80GB GPU) |

---

## 4. Execution Guides

All scripts automatically resolve paths relative to the repository, detect local model directories, and accept environment variable overrides (`MODEL_DIR`, `DATASET_PATH`, `IMAGE_DIR`, `SAM3_PATH`, `SIGLIP_NAME`).

### A. Local Workstation / A6000 (`scripts/A6000/`)

Execute individual model and pool scripts:
```bash
# Run Qwen2.5-VL-7B on N=5
bash scripts/A6000/run_our_qwen2.5_vl_7b_n5.sh

# Run LLaVA-OneVision on N=20
bash scripts/A6000/run_our_llava_ov_7b_n20.sh

# Run InternVL3-8B on N=1
bash scripts/A6000/run_our_internvl3_8b_n1.sh

# Pass additional CLI arguments (e.g. resume or index slicing)
bash scripts/A6000/run_our_qwen2.5_vl_7b_n5.sh --resume
bash scripts/A6000/run_our_qwen2.5_vl_7b_n5.sh --start_idx 0 --end_idx 50

# Run all 3 models sequentially on a given pool
bash scripts/A6000/run_our_all_n5.sh

# Run full matrix sweep across N=1, 5, 10, 20
bash scripts/A6000/run_our_sweep.sh
```

### B. HPC Slurm Cluster (`scripts/A100/`)

Submit jobs to the Slurm batch queue (`#SBATCH --gres=gpu:a100:1`):
```bash
# Submit single model job
sbatch scripts/A100/run_our_qwen2.5_vl_7b_n5.slurm
sbatch scripts/A100/run_our_llava_ov_7b_n5.slurm
sbatch scripts/A100/run_our_internvl3_8b_n5.slurm

# Submit all 3 models for N=5
sbatch scripts/A100/run_our_all_n5.slurm

# Submit full matrix sweep jobs
bash scripts/A100/run_our_sweep.slurm
```

Logs are automatically written to `scripts/A100/logs/<job_name>-<job_id>.out` and `.err`.

### C. Google Colab (`notebook/`)

Each notebook in `notebook/` is self-contained with pinned model and pool parameters:
- `run_our_colab_qwen2.5_vl_7b_n*.ipynb`: Qwen2.5-VL-7B-Instruct
- `run_our_colab_llava_ov_7b_n*.ipynb`: llava-onevision-qwen2-7b-ov
- `run_our_colab_internvl3_8b_n*.ipynb`: InternVL3-8B

**Workflow in Colab:**
1. Open the desired notebook in Google Colab (ensure GPU runtime is set to **A100** or **L4**; T4 has insufficient VRAM).
2. Set `GITHUB_TOKEN` (and optional `HF_TOKEN`) in Colab Secrets (the 🔑 icon in the sidebar).
3. Run **Cell 1**: Mounts Google Drive and auto-detects `annotations_part1234_merged.json` and `pool/`.
4. Run **Cell 2**: Clones/pulls branch `draft/improve_all`.
5. Run **Cell 3**: Installs dependencies and downloads SAM3 checkpoint if not already present.
6. Run **Cell 4**: Executes evaluation. Progress is written incrementally to Google Drive.

### D. Modal Cloud (`scripts/modal/`)

Run serverless jobs on Modal cloud infrastructure with persistent network volumes (`mssearch-data`, `mssearch-models`, `mssearch-results`):
```bash
# Direct modal run command
modal run scripts/modal/run_our_modal.py --model Qwen/Qwen2.5-VL-7B-Instruct --pool-mode N_5
modal run scripts/modal/run_our_modal.py --model lmms-lab/llava-onevision-qwen2-7b-ov --pool-mode N_5
modal run scripts/modal/run_our_modal.py --model OpenGVLab/InternVL3-8B --pool-mode N_5

# Convenient launcher wrapper
bash scripts/modal/run_modal.sh qwen2.5_vl_7b N_5
bash scripts/modal/run_modal.sh llava_ov_7b N_10
bash scripts/modal/run_modal.sh internvl3_8b N_20

# Run asynchronously in background (safe to close terminal)
modal run scripts/modal/run_our_modal.py --model Qwen/Qwen2.5-VL-7B-Instruct --pool-mode N_5 --detach-run
```

---

## 5. Evaluating and Summarizing Results

Once an evaluation completes, compute both **Set-based** and **Strict Primary Evidence** accuracy metrics:
```bash
python mssearch/eval/eval_stashbench.py \
    --results_file experiments/our/results_our_a6000_Qwen2.5-VL-7B-Instruct_N_5_*.jsonl \
    --dataset_file data/STASHBench/annotations_part1234_merged.json
```
