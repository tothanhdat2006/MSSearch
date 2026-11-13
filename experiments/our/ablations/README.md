# MSSearch Ablation Study Suite

Ablation evaluation framework for dissecting the perceptual backbones and decision thresholds of the MSSearch pipeline across three independent axes:
1. **Visual Expert / Detection Backbone** (evaluating **SAM3**, **Grounding DINO**, **Florence-2**, or **OWLv2** without feature grouping under SGAP / Rigid partitioning vs. the Rigid Top-k baseline).
2. **Low-Resolution Zero-Shot Screening Scorer** (evaluating **SigLIP**, **SigLIP2-384**, **SigLIP2-512**, **CLIP-ViT-B/32**, **CLIP-ViT-L/14**, or **EVA-02-L/14** under the default Rigid Top-k architecture).
3. **Candidate Retention Budget Top-K** (sweeping base patches per image $k \in \{1, 5, 10, 20, 40\}$, replacing the obsolete gating threshold).

---

## 1. Ablation Dimensions

### Axis 1: Visual Expert / Detector Backbone (`--detector`)
In the default architecture (Row 2: Rigid quadtree + Top-k retention), the detector is bypassed to save ~3.5 GB VRAM and avoid unnecessary detection inference. This axis evaluates whether re-enabling Stage 4 Feature Grouping with different visual experts improves over the Rigid Top-k baseline:

> [!NOTE]
> For the default SAM3 detector, the ablation runner automatically falls back to `SAM3_PATH` from `env_paths.sh` when `DETECTOR_PATH` is unset. Alternative backbones (Grounding DINO, Florence-2) can be specified via `DETECTOR_PATH` or `--detector_path`. To evaluate with SGAP partitioning instead of Rigid, pass `--sgap`.
> Bounding-box-only detectors (`grounding_dino`, `florence2`) provide detection proposals but do not emit dense feature maps. Under SGAP partitioning, SAM3 feature maps are preserved for Stage 3 superpixel partitioning while the detector expert supplies bounding box proposals.

| Key | Model Identifier / Class | Default Source / Weight Path | Role |
| :--- | :--- | :--- | :--- |
| `sam3` | `sam3_inference` | `${MODEL_DIR}/sam3/sam3.pt` | Unified promptable segmentation/detection baseline |
| `grounding_dino` | `GroundingDinoDetector` | `IDEA-Research/grounding-dino-base` | Text-grounded open-set object detection |
| `florence2` | `Florence2Detector` | `florence-community/Florence-2-large` | Open-vocabulary phrase grounding (<CAPTION_TO_PHRASE_GROUNDING>) |
| `owlv2` | `OWLv2Detector` | `google/owlv2-base-patch16-ensemble` | Open-vocabulary text-guided object detection & ViT patch features |
| `none` | `None` | N/A | Bypasses detector entirely (Rigid Top-k default) |

### Axis 2: Low-Resolution Visual Scorer (`--scorer`)
Controls saliency-guided quadtree budget allocation (Stage 3) and candidate crop ranking for pure Top-k retention (Stage 4). Evaluated with detector bypassed (`--detector none`):

| Key | Model Identifier | HuggingFace / OpenCLIP ID | Resolution / Patch |
| :--- | :--- | :--- | :--- |
| `siglip` | `SigLIPScorer` | `google/siglip-base-patch16-384` | 384x384, Patch 16 (Sigmoid loss) |
| `siglip2_384` | `SigLIPScorer` | `google/siglip2-so400m-patch16-384` | 384x384, Patch 16 (SigLIP 2 SO400M) |
| `siglip2_512` | `SigLIPScorer` | `google/siglip2-so400m-patch16-512` | 512x512, Patch 16 (SigLIP 2 SO400M) |
| `clip_b32` | `CLIPVisualScorer` | `openai/clip-vit-base-patch32` | 224x224, Patch 32 (Calibrated Cosine) |
| `clip_l14` | `CLIPVisualScorer` | `openai/clip-vit-large-patch14` | 224x224, Patch 14 (Calibrated Cosine) |
| `eva02_l14` | `EVA02VisualScorer` | `timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k` | 336x336, Patch 14 (MIM + CLIP) |

### Axis 3: Candidate Retention Budget Top-K (`--top_k`)
Controls the base number of candidate patches retained per image (ranked by visual scorer similarity), superseding the legacy VLM gating confidence threshold. Budgets scale dynamically with gallery size $N$ when `scale_budgets_by_num_images = True`:

| Key ($k$) | Runtime Cap ($N=1$) | Runtime Cap ($N=5$) | Behavior |
| :--- | :---: | :---: | :--- |
| `1` | 1 patch | 5 patches | Extreme Minimalist: single top-scoring patch per image |
| `5` | 5 patches | 25 patches | Ultra-Lean: minimal reading budget, fastest inference |
| `10` | 10 patches | 50 patches | Compact: balanced speed and context coverage |
| `20` | 20 patches | 100 patches | **Default Baseline**: optimal empirical accuracy/cost trade-off |
| `40` | 40 patches | 200 patches | High Recall: generous budget preserving subtle background targets |

### Axis 4: Search Component Increment (`--component`)
Controls the cumulative progression of the 4 core search modules (Table 1 & Table 2) evaluated as paired Rigid vs. SGAP increments:

| Key | Partition | Grouping | Existence Scanning (Detected) | Latent Re-partitioning (Absent) | Retention Budget |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `qwen` | Baseline | Off | Off | Off | Direct zero-shot VLM on full gallery images |
| `qwen_rigid_topk` | Rigid | Off | Off | Off | Base 20 patches/img ranked by SigLIP (scales with N) |
| `qwen_sgap_topk` | SGAP | Off | Off | Off | Base 20 patches/img ranked by SigLIP (scales with N) |
| `qwen_rigid_fg_det_abs` | Rigid | **On** | **Off** *(Trust detector)* | **Off** *(Variant B: Naive centroid)* | 8 det / 1 abs centroid |
| `qwen_sgap_fg_det_abs` | SGAP | **On** | **Off** *(Trust detector)* | **Off** *(Variant B: Naive centroid)* | 8 det / 1 abs centroid |
| `qwen_rigid_fg_exist_abs` | Rigid | **On** | **ON** *(VLM verified)* | **Off** *(Variant B: Naive centroid)* | **Isolates Existence Scanning** |
| `qwen_sgap_fg_exist_abs` | SGAP | **On** | **ON** *(VLM verified)* | **Off** *(Variant B: Naive centroid)* | **Isolates Existence Scanning** |
| `qwen_rigid_fg_det_latent` | Rigid | **On** | **Off** *(Trust detector)* | **ON** *(VLM latent zoom)* | **Isolates Latent Re-partitioning** |
| `qwen_sgap_fg_det_latent` | SGAP | **On** | **Off** *(Trust detector)* | **ON** *(VLM latent zoom)* | **Isolates Latent Re-partitioning** |
| `qwen_rigid_full` | Rigid | **On** | **ON** *(VLM verified)* | **ON** *(VLM latent zoom)* | Full Rigid Single-Crop Method |
| `qwen_sgap_full` *(Ours)* | SGAP | **On** | **ON** *(VLM verified)* | **ON** *(VLM latent zoom)* | Full SGAP Single-Crop Method |

*Note: Dynamic retention elasticity (`enable_dynamic_retention_elasticity = True`) is preserved for grouped methods to retain score ties at the 20th patch cutoff.*
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
│   ├── owlv2_adapter.py           # OWLv2 adapter via HF Transformers
│   ├── unified_vision_extractor.py # Unified spatial feature map & detection extractor
│   ├── clip_scorer_adapter.py     # CLIP-ViT-B/32 and CLIP-ViT-L/14 scorer
│   └── eva02_scorer_adapter.py    # EVA-02-L/14 scorer (open_clip / timm / HF)
├── scripts/
│   ├── run_ablation_component.sh  # Sweeps 11-step component increments (Qwen -> Rigid Top-k -> SGAP -> Grouping -> Full)
│   ├── run_ablation_detector.sh   # Sweeps Visual Experts (SAM3 vs Grounding DINO vs Florence-2) without Feature Grouping
│   ├── run_ablation_scorer.sh     # Sweeps SigLIP vs SigLIP2 vs CLIP vs EVA-02 under default Rigid Top-k architecture
│   ├── run_ablation_topk.sh       # Sweeps Candidate Retention Budget Top-K: k in {5, 10, 20, 40}
│   ├── run_ablation_gate_threshold.sh # Backward-compatible wrapper forwarding to run_ablation_topk.sh
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
- **Detector Path**: For SAM3, the runner automatically uses `SAM3_PATH` from `env_paths.sh` when `DETECTOR_PATH` is unset. For other detector backbones, specify via `DETECTOR_PATH` or `--detector_path`.
- **Python Path**: Set `PYTHONPATH` to the repository root before running ablation scripts.

---

## 4. How to Run Ablations

### A. Run Visual Expert / Detector Ablation
Compares **SAM3** vs. **Grounding DINO** vs. **Florence-2** without Feature Grouping (pure Top-k retention, matching Row 3) on pool $N=5$:
```bash
cd MSSearch
bash experiments/our/ablations/scripts/run_ablation_detector.sh Qwen/Qwen3-VL-8B-Instruct N_5

# Optional: test on SGAP partitioning
bash experiments/our/ablations/scripts/run_ablation_detector.sh Qwen/Qwen3-VL-8B-Instruct N_5 --sgap
```

### B. Run Visual Scorer Ablation (Default Rigid Top-k)
Compares **SigLIP** vs. **SigLIP2** vs. **CLIP-ViT-B/32** vs. **CLIP-ViT-L/14** vs. **EVA-02-L/14** under the default Rigid quadtree + Top-k architecture (detector bypassed) on pool $N=5$:
```bash
cd MSSearch
bash experiments/our/ablations/scripts/run_ablation_scorer.sh Qwen/Qwen3-VL-8B-Instruct N_5
```

### C. Run Candidate Retention Top-K Budget Ablation
Evaluates base retention budgets $k \in \{1, 5, 10, 20, 40\}$ patches per image ($k=20$ default, scaled dynamically by pool size $N$) on pool $N=5$:
```bash
cd MSSearch
# Run directly:
bash experiments/our/ablations/scripts/run_ablation_topk.sh Qwen/Qwen3-VL-8B-Instruct N_5

# Or via backward-compatible wrapper:
bash experiments/our/ablations/scripts/run_ablation_gate_threshold.sh Qwen/Qwen3-VL-8B-Instruct N_5
```

### D. Run Search Component Increment Sweep
Sweeps all 11 component increments on pool $N=5$ (or any specified pool):
```bash
cd MSSearch
bash experiments/our/ablations/scripts/run_ablation_component.sh all Qwen/Qwen3-VL-8B-Instruct N_5

# Or run an individual component (e.g. Full SGAP):
bash experiments/our/ablations/scripts/run_ablation_component.sh qwen_sgap_full Qwen/Qwen3-VL-8B-Instruct N_5

# Or isolate Existence Scanning:
bash experiments/our/ablations/scripts/run_ablation_component.sh qwen_sgap_fg_exist_abs Qwen/Qwen3-VL-8B-Instruct N_5
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
# Detector sweep (SAM3, Grounding DINO, Florence-2)
bash experiments/our/ablations/scripts/run_modal_ablation.sh detector Qwen/Qwen3-VL-8B-Instruct N_5 --detach

# Visual scorer sweep (SigLIP, SigLIP2-384, SigLIP2-512, CLIP-B/32, CLIP-L/14, EVA-02-L/14)
bash experiments/our/ablations/scripts/run_modal_ablation.sh scorer Qwen/Qwen3-VL-8B-Instruct N_5 --detach

# Gating threshold sweep (0.3, 0.7, 0.5)
bash experiments/our/ablations/scripts/run_modal_ablation.sh gate Qwen/Qwen3-VL-8B-Instruct N_5 --detach

# Component increment sweep (all 11 paired components)
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
