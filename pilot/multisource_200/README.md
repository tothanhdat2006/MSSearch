# MSSearch 200-Sample Multi-Source Evaluation Suite

Benchmark evaluation suite for measuring visual search and attribute extraction on `200_samples.json` across varying pool sizes ($N \in \{5, 10, 20\}$) and evidence cardinality ($K \in \{1, 2, 3, 4\}$).

---

## Directory Contents

- `eval_200.py`: Main evaluation runner supporting MSSearch e2e pipeline and Direct VLM baselines.
- `run_all.sh`: Shell launcher for MSSearch visual search pipeline.
- `run_baseline.sh`: Shell launcher for Vanilla VLM direct inference.
- `config_default.json`: Baseline budget configuration.
- `test_metrics.py`: Self-contained runnable validation script for metric computation and dataset schema.

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
