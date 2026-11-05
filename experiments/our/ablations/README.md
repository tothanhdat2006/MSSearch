# MSSearch Ablation Study Suite

Ablation evaluation framework for dissecting the perceptual backbones of the MSSearch pipeline across two independent axes:
1. **Open-Vocabulary Detection Backbone** (replacing **SAM3** with **Grounding DINO** or **DINO-X**).
2. **Low-Resolution Zero-Shot Screening Scorer** (replacing **SigLIP** with **CLIP-ViT-B/32**, **CLIP-ViT-L/14**, or **EVA-02-L/14**).

---

## 1. Ablation Dimensions

### Axis 1: Open-Vocabulary Detector (`--detector`)
Controls tool fast path (Stage 2) and candidate crop open-vocabulary object gating (Stage 4, Stage 5 recursive zoom):

| Key | Model Identifier / Class | Default Source / Weight Path | Role |
| :--- | :--- | :--- | :--- |
| `sam3` | `sam3_inference` | `${MODEL_DIR}/sam3/sam3.pt` | Unified promptable segmentation/detection baseline |
| `grounding_dino` | `GroundingDinoDetector` | `IDEA-Research/grounding-dino-base` | Text-grounded open-set object detection |
| `dinox` | `DinoXDetector` | Local checkpoint or DINO-X Cloud API | Next-generation unified open-world detection |

### Axis 2: Low-Resolution Visual Scorer (`--scorer`)
Controls saliency-guided quadtree budget allocation (Stage 3), preliminary patch gating (Stage 4), and candidate crop spatial filtering (Stage 5):

| Key | Model Identifier | HuggingFace / OpenCLIP ID | Resolution / Patch |
| :--- | :--- | :--- | :--- |
| `siglip` | `SigLIPScorer` | `google/siglip-base-patch16-384` | 384x384, Patch 16 (Sigmoid loss) |
| `clip_b32` | `CLIPVisualScorer` | `openai/clip-vit-base-patch32` | 224x224, Patch 32 (Calibrated Cosine) |
| `clip_l14` | `CLIPVisualScorer` | `openai/clip-vit-large-patch14` | 224x224, Patch 14 (Calibrated Cosine) |
| `eva02_l14` | `EVA02VisualScorer` | `EVA02-L-14-336` / `QuanSun/EVA-CLIP` | 336x336, Patch 14 (MIM + CLIP) |

---

## 2. Directory Structure

```text
MSSearch/experiments/our/ablations/
├── README.md                      # Ablation framework documentation (this file)
├── env_paths_ablation.sh          # Central environment paths & model checkpoints
├── eval_models_ablation.py        # Adapter factory & pipeline dependency injector
├── eval_runner_ablation.py        # CLI evaluation runner with atomic per-sample flushing
├── models/
│   ├── __init__.py                # Factory functions: load_detector, load_visual_scorer
│   ├── base.py                    # BaseDetector and BaseVisualScorer contracts
│   ├── sam3_adapter.py            # SAM3 wrapper conforming to BaseDetector
│   ├── grounding_dino_adapter.py  # Grounding DINO adapter via HF Transformers
│   ├── dinox_adapter.py           # DINO-X adapter (local weights or API client)
│   ├── clip_scorer_adapter.py     # CLIP-ViT-B/32 and CLIP-ViT-L/14 scorer
│   └── eva02_scorer_adapter.py    # EVA-02-L/14 scorer (open_clip / timm / HF)
├── scripts/
│   ├── run_ablation_detector.sh   # Sweeps SAM3 vs Grounding DINO vs DINO-X (fixed SigLIP)
│   ├── run_ablation_scorer.sh     # Sweeps SigLIP vs CLIP-B/32 vs CLIP-L/14 vs EVA-02 (fixed SAM3)
│   ├── run_ablation_matrix.sh     # Full cross-ablation sweep
│   └── run_ablation_slurm.slurm   # HPC Slurm batch runner
└── tests/
    ├── __init__.py
    └── test_ablation_adapters.py  # Self-check unit tests for all adapters and factories
```

---

## 3. How to Run Ablations

### A. Run Detector Ablation (Fixed SigLIP)
Compares **SAM3** vs. **Grounding DINO** vs. **DINO-X** on pool $N=5$:
```bash
cd MSSearch
bash experiments/our/ablations/scripts/run_ablation_detector.sh Qwen/Qwen2.5-VL-7B-Instruct N_5
```

### B. Run Visual Scorer Ablation (Fixed SAM3)
Compares **SigLIP** vs. **CLIP-ViT-B/32** vs. **CLIP-ViT-L/14** vs. **EVA-02-L/14** on pool $N=5$:
```bash
cd MSSearch
bash experiments/our/ablations/scripts/run_ablation_scorer.sh Qwen/Qwen2.5-VL-7B-Instruct N_5
```

### C. Run Specific Pair via CLI Runner
```bash
python experiments/our/ablations/eval_runner_ablation.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --detector grounding_dino \
    --detector_path IDEA-Research/grounding-dino-base \
    --scorer clip_l14 \
    --scorer_path openai/clip-vit-large-patch14 \
    --pool_mode N_5 \
    --output_dir experiments/our/ablations/results_ablation
```

### D. Subsetting and Resuming Runs
Pass standard runner flags through any bash script:
```bash
# Evaluate first 20 samples:
bash experiments/our/ablations/scripts/run_ablation_detector.sh Qwen/Qwen2.5-VL-7B-Instruct N_5 --max_samples 20

# Resume an interrupted ablation run:
bash experiments/our/ablations/scripts/run_ablation_detector.sh Qwen/Qwen2.5-VL-7B-Instruct N_5 --resume
```

---

## 4. HPC Cluster Submission (Slurm)

```bash
sbatch experiments/our/ablations/scripts/run_ablation_slurm.slurm Qwen/Qwen2.5-VL-7B-Instruct
```

---

## 5. Output Artifacts

All runs produce atomically flushed logs and auto-saved configuration snapshots in `experiments/our/ablations/results_ablation/`:
- `results_ablation_{detector}_{scorer}_{model}_{pool}_{timestamp}.jsonl`: Line-by-line task telemetry and model predictions.
- `config_ablation_{detector}_{scorer}_{model}_{pool}_{timestamp}.json`: Complete frozen parameter snapshot.
