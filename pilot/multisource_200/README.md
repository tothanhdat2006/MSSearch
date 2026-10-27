# MSSearch 200-Sample Multi-Source Evaluation Suite

Benchmark evaluation suite for measuring visual search and attribute extraction on `200_samples.json` across varying pool sizes ($N \in \{5, 10, 20\}$) and evidence cardinality ($K \in \{1, 2, 3, 4\}$).

---

## Directory Contents

- `eval_200.py`: Main evaluation runner supporting MSSearch e2e pipeline and Direct VLM baselines (Qwen2.5-VL and LLaVA-OV).
- `run_all.slurm`: Slurm batch launcher on GPU server `gpu-a240` running Qwen2.5-VL-7B and LLaVA-OneVision-7B sequentially.
- `run_baseline.slurm`: Slurm batch launcher for Direct VLM baseline evaluations on `gpu-a240`.
- `run_all.sh`: Local/standalone shell launcher for MSSearch visual search pipeline.
- `run_baseline.sh`: Local/standalone shell launcher for Vanilla VLM direct inference.
- `config_default.json`: Baseline budget configuration.
- `test_metrics.py`: Self-contained runnable validation script for metric computation and dataset schema.

---

## Slurm Execution on A240 Server (`gpu-a240`)

### Strict Exact-Match and Zero-Fallback Policy
- **Exact Model Matching**: Target models must match directory names exactly (e.g. `Qwen3-VL-8B`, `LLaVA-v1.6-7B`, `LLaVA-v1.6-13B`, `LLaVA-v1.6-34B`, `LLaVA-HR-X-7B`, `LLaVA-HR-X-13B`, `Qwen2.5-VL-32B`, `InternVL3-38B`, `Yi-VL-34B`, `Qwen2.5-VL-7B-Instruct`, `llava-onevision-qwen2-7b-ov`).
- **Immediate Failure**: If any path (dataset, image directory, model checkpoint, SAM3, SigLIP, spaCy, or virtual environment) does not exist, execution terminates immediately (`exit 1` / `FileNotFoundError`). No silent guessing or fallback paths are permitted.

### 1. Launch Full MSSearch Pipeline via Slurm
Default execution (runs `Qwen2.5-VL-7B-Instruct` and `llava-onevision-qwen2-7b-ov` sequentially):
```bash
sbatch run_all.slurm
```

Run specific model(s) by exact folder name:
```bash
# Single model
sbatch run_all.slurm Qwen3-VL-8B
sbatch run_all.slurm LLaVA-v1.6-7B
sbatch run_all.slurm Qwen2.5-VL-32B

# Multiple models sequentially
sbatch run_all.slurm Qwen2.5-VL-32B InternVL3-38B Yi-VL-34B

# Via environment variable
sbatch --export=ALL,MODELS="LLaVA-v1.6-13B LLaVA-v1.6-34B" run_all.slurm
```

Forward arguments with exact model name:
```bash
# Smoke test on index 0
sbatch run_all.slurm Qwen3-VL-8B --sample_idx 0

# All pool sizes
sbatch run_all.slurm Qwen2.5-VL-7B-Instruct --pool_mode ALL
```

### 2. Launch Direct VLM Baseline via Slurm
```bash
# Default models
sbatch run_baseline.slurm

# Specific exact models
sbatch run_baseline.slurm Qwen3-VL-8B
sbatch run_baseline.slurm LLaVA-v1.6-7B --pool_mode ALL
```

### 3. Monitor Slurm Jobs
```bash
# View active queue
squeue -u $USER

# Follow live output logs
tail -f logs/mssearch_200-<JobID>.out

# Check accounting
sacct -j <JobID> --format=JobID,JobName,Partition,AllocCPUS,Elapsed,State,ExitCode
```

---

## Quick Start on GPU Server

### 1. MSSearch Full Pipeline (E2E)

Run all 200 samples across default $N=5$ candidate pool:
```bash
./run_all.sh
```

Run across all pool sizes ($N=5, 10, 20$):
```bash
./run_all.sh --pool_mode ALL
```

Single sample smoke test (e.g. sample index 0):
```bash
./run_all.sh --sample_idx 0
```

Slice evaluation (e.g. first 10 samples):
```bash
./run_all.sh --start_idx 0 --end_idx 10
```

Resume an interrupted run:
```bash
./run_all.sh --resume
```

---

### 2. Direct VLM Baseline

Run baseline over $N=5$ candidate pool:
```bash
./run_baseline.sh
```

Run baseline with target-only gold images ($N=K$, distractors bypassed):
```bash
./run_baseline.sh --direct_target_only
```

Single sample smoke test:
```bash
./run_baseline.sh --sample_idx 0
```

---

## Configuration & Environment Overrides

Default server paths configured in shell scripts:
- Model: `/home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct`
- SAM3: `/home/ubuntu/workspace/hr/models/sam3/sam3/sam3.pt`
- SigLIP: `/home/ubuntu/workspace/hr/models/siglip-base-patch16-384`
- spaCy: `/home/ubuntu/workspace/hr/models/spacy/en_core_web_sm-3.8.0`
- Dataset: `/home/ubuntu/workspace/phong/multi_image_visual_search/data/output/200_samples.json`
- Image Directory: `/home/ubuntu/workspace/phong/multi_image_visual_search/data/pilot_5000_images`

Override examples:
```bash
./run_all.sh --model_path /path/to/other/vlm --pool_mode N_10
```

---

## Evaluation Metrics

For each query and ground-truth target attribute set $GT$ vs predicted token set $Pred$:
- **Precision**: $\frac{|Pred \cap GT|}{|Pred|}$
- **Recall**: $\frac{|Pred \cap GT|}{|GT|}$
- **F1 Score**: $\frac{2 \cdot P \cdot R}{P + R}$
- **Exact Match**: $P = 1.0 \land R = 1.0$

Results are automatically saved to `results_*/` as:
1. `.jsonl`: Streaming line-by-line task execution records with per-sample latencies and route telemetry.
2. `_summary.json`: Aggregate metrics stratified across:
   - Overall set performance
   - Pool size $N \in \{5, 10, 20\}$
   - Target evidence cardinality $K \in \{1, 2, 3, 4\}$
   - $(N, K)$ cross-tabulation grid
   - Execution routes
