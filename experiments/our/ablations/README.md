# MSSearch Ablation Study Suite

Ablation evaluation framework for dissecting the perceptual backbones and decision thresholds of the MSSearch pipeline across three independent axes:
1. **Open-Vocabulary Detection Backbone** (replacing **SAM3** with **Grounding DINO**, **Florence-2**, or **DINOv3**).
2. **Low-Resolution Zero-Shot Screening Scorer** (replacing **SigLIP** with **SigLIP2**, **CLIP-ViT-B/32**, **CLIP-ViT-L/14**, or **EVA-02-L/14**).
3. **VLM Gating Confidence Threshold** (evaluating **0.3**, **0.5** [default], and **0.7**).

---

## 1. Ablation Dimensions

### Axis 1: Open-Vocabulary Detector (`--detector`)
Controls tool fast path (Stage 2) and candidate crop open-vocabulary object gating (Stage 4, Stage 5 recursive zoom):

> [!NOTE]
> The ablation runner reads the detection backbone path from `DETECTOR_PATH` (e.g. `export DETECTOR_PATH="${SAM3_PATH}"` in `experiments/our/env_paths.sh`), not `SAM3_PATH`. Ensure `DETECTOR_PATH` is exported before launching.

| Key | Model Identifier / Class | Default Source / Weight Path | Role |
| :--- | :--- | :--- | :--- |
| `sam3` | `sam3_inference` | `${MODEL_DIR}/sam3/sam3.pt` | Unified promptable segmentation/detection baseline |
| `grounding_dino` | `GroundingDinoDetector` | `IDEA-Research/grounding-dino-base` | Text-grounded open-set object detection |
| `florence2` | `Florence2Detector` | `microsoft/Florence-2-large` | Open-vocabulary phrase grounding (<CAPTION_TO_PHRASE_GROUNDING>) |
| `dinov3` | `DINOv3Detector` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | Dense ViT-L/16 patch features & energy proposals (gated) |

### Axis 2: Low-Resolution Visual Scorer (`--scorer`)
Controls saliency-guided quadtree budget allocation (Stage 3), preliminary patch gating (Stage 4), and candidate crop spatial filtering (Stage 5):

| Key | Model Identifier | HuggingFace / OpenCLIP ID | Resolution / Patch |
| :--- | :--- | :--- | :--- |
| `siglip` | `SigLIPScorer` | `google/siglip-base-patch16-384` | 384x384, Patch 16 (Sigmoid loss) |
| `siglip2_384` | `SigLIPScorer` | `google/siglip2-so400m-patch16-384` | 384x384, Patch 16 (SigLIP 2 SO400M) |
| `siglip2_512` | `SigLIPScorer` | `google/siglip2-so400m-patch16-512` | 512x512, Patch 16 (SigLIP 2 SO400M) |
| `clip_b32` | `CLIPVisualScorer` | `openai/clip-vit-base-patch32` | 224x224, Patch 32 (Calibrated Cosine) |
| `clip_l14` | `CLIPVisualScorer` | `openai/clip-vit-large-patch14` | 224x224, Patch 14 (Calibrated Cosine) |
| `eva02_l14` | `EVA02VisualScorer` | `timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k` | 336x336, Patch 14 (MIM + CLIP) |

### Axis 3: VLM Gating Confidence Threshold (`--gate_threshold`)
Controls candidate patch gating in Stage 4 (existence probe and speculative latent zoom probing):

| Key | Probability $P(\text{Yes})$ | Internal Score $[-1, 1]$ | Behavior |
| :--- | :--- | :--- | :--- |
| `0.3` | $0.30$ | $-0.40$ | High Recall: retains speculative and ambiguous patches for Stage 6 reading |
| `0.5` | $0.50$ (default) | $0.00$ | Balanced Baseline: standard majority softmax decision boundary |
| `0.7` | $0.70$ | $+0.40$ | High Precision: prunes aggressively, only passing high-certainty evidence |

### Axis 4: Search Component Increment (`--component`)
Controls the cumulative progression of search components (Table 1 & Table 2):

| Key | Configuration | Behavior / Route |
| :--- | :--- | :--- |
| `qwen` | Direct Baseline | Multi-image VLM forward pass on full gallery with dynamic pixel budget |
| `qwen_fast_path` | Qwen w/ Fast Path | Tool-based fast path proposal; falls back to Direct Baseline on unresolved queries |
| `qwen_fp_rigid_cg_single` | Qwen w/ FP + Rigid Quadtree + CG + Single-crop | Fast Path + `quadrant_only` (regular quadrants) + Confidence Gating + isolated 1.25x crops |
| `qwen_fp_sgap_cg_single` | Qwen w/ FP + SGAP + CG + Single-crop | Fast Path + `sgap_only` (SLIC superpixels) + Confidence Gating + isolated 1.25x crops |
| `qwen_fp_asp_cg_single` | Qwen w/ FP + ASP + CG + Single-crop | Fast Path + `hybrid` (entropy-steered ASP) + Confidence Gating + isolated 1.25x crops |
| `qwen_fp_asp_cg_agg` | Qwen w/ FP + ASP + CG + Aggregation (Ours) | Full STASH-Search: Continuous Enclosure Sub-Canvas Engine + atomic sibling re-fusion |

---

## 2. Directory Structure

```text
MSSearch/experiments/our/ablations/
├── README.md                      # Ablation framework documentation (this file)
├── env_paths_ablation.sh          # Central environment paths & model checkpoints
├── eval_runner_utils.py           # Shared dataset resumption, telemetry & OOM-safe direct baseline
├── eval_models_ablation.py        # Adapter factory & pipeline dependency injector
├── eval_runner_ablation.py        # CLI evaluation runner with atomic per-sample flushing
├── models/
│   ├── __init__.py                # Factory functions: load_detector, load_visual_scorer
│   ├── base.py                    # BaseDetector and BaseVisualScorer contracts
│   ├── sam3_adapter.py            # SAM3 wrapper conforming to BaseDetector
│   ├── grounding_dino_adapter.py  # Grounding DINO adapter via HF Transformers
│   ├── florence2_adapter.py       # Florence-2-large adapter via HF Transformers
│   ├── dinov3_adapter.py          # DINOv3 adapter with dense patch proposals
│   ├── clip_scorer_adapter.py     # CLIP-ViT-B/32 and CLIP-ViT-L/14 scorer
│   └── eva02_scorer_adapter.py    # EVA-02-L/14 scorer (open_clip / timm / HF)
├── scripts/
│   ├── run_ablation_component.sh  # Sweeps 6-step component increments (Qwen -> Fast Path -> Rigid -> SGAP -> ASP -> Aggregation)
│   ├── run_ablation_detector.sh   # Sweeps SAM3 vs Grounding DINO vs Florence-2 vs DINOv3 (fixed SigLIP)
│   ├── run_ablation_scorer.sh     # Sweeps SigLIP vs SigLIP2 vs CLIP vs EVA-02 (fixed SAM3)
│   ├── run_ablation_gate_threshold.sh # Sweeps Gating Threshold: 0.3 vs 0.7 vs 0.5
│   ├── run_ablation_matrix.sh     # Full cross-ablation sweep
│   └── run_ablation_slurm.slurm   # HPC Slurm batch runner
└── tests/
    ├── __init__.py
    └── test_ablation_adapters.py  # Self-check unit tests for all adapters and factories
```

---

## 3. Environment & Execution Prerequisites

- **Python Dependencies**:
  - `mssearch_requirements_tf4.57.6.txt`: For `LLaVA-OV-7B` and `InternVL3-8B` (`transformers==4.57.6`).
  - `mssearch_requirements_tf5.6.2.txt`: For `Qwen3-VL-8B` (`transformers==5.6.2`).
- **Gallery Order File**: `gallery_order_seed42.json` must be in the same folder as the annotations JSON; without it, `eval_dataset.py` raises `FileNotFoundError` to prevent silent fallback to image id sorting.
- **spaCy NLP Pipeline**: `NLP_MODEL` must point to the spaCy folder. If invalid or missing, runners raise an error and terminate immediately.
- **Detector Path**: Ensure `DETECTOR_PATH` is exported in `env_paths.sh` (e.g. `export DETECTOR_PATH="${SAM3_PATH}"`).
- **Python Path**: Set `PYTHONPATH` to the repository root before running ablation scripts.

---

## 4. How to Run Ablations

### A. Run Detector Ablation (Fixed SigLIP)
Compares **SAM3** vs. **Grounding DINO** vs. **Florence-2** vs. **DINOv3** on pool $N=5$:
```bash
cd MSSearch
bash experiments/our/ablations/scripts/run_ablation_detector.sh Qwen/Qwen3-VL-8B-Instruct N_5
```

### B. Run Visual Scorer Ablation (Fixed SAM3)
Compares **SigLIP** vs. **SigLIP2** vs. **CLIP-ViT-B/32** vs. **CLIP-ViT-L/14** vs. **EVA-02-L/14** on pool $N=5$:
```bash
cd MSSearch
bash experiments/our/ablations/scripts/run_ablation_scorer.sh Qwen/Qwen3-VL-8B-Instruct N_5
```

### C. Run Gate Threshold Ablation (0.3 vs 0.7 vs 0.5)
Evaluates gating probability thresholds **0.3**, **0.7**, and **0.5** (default) on pool $N=5$:
```bash
cd MSSearch
bash experiments/our/ablations/scripts/run_ablation_gate_threshold.sh Qwen/Qwen3-VL-8B-Instruct N_5
```

### D. Run Search Component Increment Sweep
Sweeps the cumulative component hierarchy (`qwen` -> `qwen_fast_path` -> `qwen_fp_rigid_cg_single` -> `qwen_fp_sgap_cg_single` -> `qwen_fp_asp_cg_single` -> `qwen_fp_asp_cg_agg`) on pool $N=5$:
```bash
cd MSSearch
bash experiments/our/ablations/scripts/run_ablation_component.sh all Qwen/Qwen3-VL-8B-Instruct N_5

# Or run an individual increment:
bash experiments/our/ablations/scripts/run_ablation_component.sh qwen_fp_rigid_cg_single Qwen/Qwen3-VL-8B-Instruct N_5
```

### E. Run Specific Pair or Component via CLI Runner
```bash
python experiments/our/ablations/eval_runner_ablation.py \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --detector grounding_dino \
    --detector_path IDEA-Research/grounding-dino-base \
    --scorer clip_l14 \
    --scorer_path openai/clip-vit-large-patch14 \
    --pool_mode N_5 \
    --output_dir experiments/our/ablations/results_ablation
```

### F. Subsetting and Resuming Runs
Pass standard runner flags through any bash script:
```bash
# Evaluate first 20 samples:
bash experiments/our/ablations/scripts/run_ablation_detector.sh Qwen/Qwen3-VL-8B-Instruct N_5 --max_samples 20

# Resume an interrupted ablation run:
bash experiments/our/ablations/scripts/run_ablation_detector.sh Qwen/Qwen3-VL-8B-Instruct N_5 --resume
```

---

## 4. Serverless Cloud GPU Execution (Modal)

Ablation experiments can be run directly on Modal cloud infrastructure (`L40S` 48GB / `A100` 80GB) with persistent storage volumes (`mssearch-data`, `mssearch-models`, `mssearch-results` under `/results/ablation`):

### A. Run via Modal Launcher Script (`run_modal_ablation.sh`)
```bash
# Detector sweep (SAM3, Grounding DINO, Florence-2, DINOv3)
bash experiments/our/ablations/scripts/run_modal_ablation.sh detector Qwen/Qwen3-VL-8B-Instruct N_5 --detach

# Visual scorer sweep (SigLIP, SigLIP2-384, SigLIP2-512, CLIP-B/32, CLIP-L/14, EVA-02-L/14)
bash experiments/our/ablations/scripts/run_modal_ablation.sh scorer Qwen/Qwen3-VL-8B-Instruct N_5 --detach

# Gating threshold sweep (0.3, 0.7, 0.5)
bash experiments/our/ablations/scripts/run_modal_ablation.sh gate Qwen/Qwen3-VL-8B-Instruct N_5 --detach

# Component increment sweep (all 6 components)
bash experiments/our/ablations/scripts/run_modal_ablation.sh component Qwen/Qwen3-VL-8B-Instruct N_5 --detach

# Full matrix sweep across all 3 axes
bash experiments/our/ablations/scripts/run_modal_ablation.sh all Qwen/Qwen3-VL-8B-Instruct N_5 --detach
```

### B. Direct Modal CLI Run (`run_ablation_modal.py`)
```bash
# Run specific configuration
modal run experiments/our/ablations/scripts/run_ablation_modal.py \
    --detector grounding_dino \
    --scorer clip_l14 \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --pool-mode N_5

# Run asynchronously in background
modal run --detach experiments/our/ablations/scripts/run_ablation_modal.py \
    --detector florence2 \
    --scorer siglip2_384 \
    --gate-threshold 0.3 \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --pool-mode N_5
```

---

## 5. HPC Cluster Submission (Slurm)

```bash
sbatch experiments/our/ablations/scripts/run_ablation_slurm.slurm Qwen/Qwen3-VL-8B-Instruct
```

---

## 5. Output Artifacts

All runs produce atomically flushed logs and auto-saved configuration snapshots in `experiments/our/ablations/results_ablation/`:
- `results_ablation_{detector}_{scorer}_{model}_{pool}_{timestamp}.jsonl`: Line-by-line task telemetry and model predictions.
- `config_ablation_{detector}_{scorer}_{model}_{pool}_{timestamp}.json`: Complete frozen parameter snapshot.
