# HRBench 8K Multi-Source Visual Search & Baseline Evaluation Suite

Headless CLI execution and diagnostic suite for benchmarking **MSSearch** and **Vanilla Direct VLM Baselines** on the HRBench 8K dataset across single-image ($N=1$) and multi-candidate ($N \ge 2$) multi-source pools.

---

## 1. Quickstart

All runners auto-activate Conda environment `hrsearch`, set `PYTHONPATH`, and handle directory setup.

### 1.1 Full MSSearch Benchmark Evaluation (`run_all.sh`)
Executes end-to-end multi-stage visual search across all 999 samples with auto-resume:
```bash
cd MSSearch/pilot/multisource_hrbench

# Default model (Qwen2.5-VL-7B-Instruct)
bash run_all.sh

# Custom local model via positional argument
bash run_all.sh /path/to/local/model

# Custom local model via flag or environment variable
bash run_all.sh --model_path /path/to/local/model
MODEL_PATH=/path/to/local/model bash run_all.sh
```

### 1.2 Direct Vanilla VLM Baseline Evaluation (`run_baseline.sh`)
Queries the raw VLM directly across candidate images without visual search or SAM3:
```bash
cd MSSearch/pilot/multisource_hrbench

# Default model (Qwen2.5-VL-7B-Instruct)
bash run_baseline.sh

# Custom local model via positional argument or flag
bash run_baseline.sh /path/to/local/model
bash run_baseline.sh --model_path /path/to/local/model

# Evaluate strictly on target image (N=1 vanilla single-image baseline)
bash run_baseline.sh /path/to/local/model --direct_target_only

# LLaVA-OneVision-7B execution
bash run_baseline.sh /home/ubuntu/workspace/hr/models/LLaVA-OV-7B
```

> **LLaVA-OneVision (LLaVA-OV-7B) Guide**: For checkpoint structure, offline loading fixes, RoPE configuration, and troubleshooting, refer to [`LLAVA_OV_GUIDE.md`](LLAVA_OV_GUIDE.md).

> **Offline Guarantee (Local Only)**: Both runners strictly enforce `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, ensuring that all weights and processors are resolved strictly from local disk. If a non-existent or remote model path is provided, execution halts immediately without attempting any network connection. When evaluating custom models, results are automatically isolated into model-named output directories (e.g. `results_<model_name>/` or `results_baseline_<model_name>/`) to prevent cache or resume collisions.


---

## 2. CLI Usage & Arguments Reference

Direct invocation via `python debug_stages.py`:

```bash
python debug_stages.py \
    --mode e2e \
    --all_samples \
    --resume \
    --budget medium \
    --num_candidate_images 3 \
    --hrbench_dir /home/ubuntu/workspace/hr/data/hr-bench_8k \
    --output_dir results/
```

### Key Arguments

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--mode` | `str` | `e2e` | `e2e` (full visual search pipeline) or `direct`/`direct_answer`/`baseline` (vanilla VLM baseline). |
| `--direct_target_only` | `flag` | `False` | In direct baseline mode, bypass distractors and evaluate only target canvas ($N=1$). |
| `--all_samples` | `flag` | `False` | Evaluates entire dataset queue (`0` to `len(data)-1`). |
| `--resume` | `flag` | `False` | Auto-detects completed samples in existing JSONL logs and skips them without recomputation. |
| `--sample_idx` | `int` | `0` | Single sample index for quick inspection. |
| `--start_idx` / `--end_idx` | `int` | `None` | Slices dataset queue `[start_idx, end_idx)`. |
| `--budget` | `str` | `None` | Preset budget config: `small`, `medium`, or `large`. |
| `--num_candidate_images` | `int` | `3` | Candidate pool size $N$ (1 ground-truth target + $N-1$ sampled distractors). |
| `--min_pixels` | `int` | `3136` | Minimum patch resolution in pixels ($56 \times 56$). |
| `--max_pixels` | `int` | `12845056` | Maximum visual resolution ceiling in pixels (~12.8M px, roughly $3584 \times 3584$). Auto-adjusted to 4,194,304 for 32B checkpoints. |
| `--log_level` | `str` | `summary` | Output verbosity: `summary` (1-line per stage decision), `verbose`, or `quiet`. |

---

## 3. Visual Pixel Budget & Resolution Controls

Qwen2.5-VL native resolution is governed by `min_pixels` and `max_pixels`:

```python
# Visual Pixel Budget
min_pixels = 3136      # 56 x 56 minimum patch resolution floor
max_pixels = 12845056  # 12.8M pixels (~3584 x 3584) default ceiling
```

### Safety Rules & 32B OOM Protection
- **`min_pixels = 3136`**: Ensures patches smaller than $56 \times 56$ are upscaled, preventing token collapse and preserving optical character readability on fine text and license plates.
- **`max_pixels = 12,845,056` (Default)**: Used for 7B models (e.g. `Qwen2.5-VL-7B-Instruct`), allowing ultra-high-resolution ingestion up to ~12.8 Megapixels (~16,000 vision tokens).
- **`max_pixels = 4,194,304` (32B Override)**: When loading 32B checkpoints (`Qwen2.5-VL-32B-Instruct`), unbounded 12.8M inputs cause SDPA self-attention memory buffers to exceed GPU capacity (triggering CUDA OOM). The runner auto-caps `max_pixels` to 4,194,304 (~4.2M pixels, $2048 \times 2048$, ~5,200 tokens) whenever `32b` is detected in the model path.

You can manually enforce custom bounds at runtime:
```bash
python debug_stages.py --max_pixels 4194304 --min_pixels 3136
```

---

## 4. Pipeline Budget Preset
 
The default configuration is set to **`budget_large.json`** (`--budget large`):
 
| Hyperparameter | Value | Description |
| :--- | :---: | :--- |
| `max_patches_per_image` | 150 | Stage 3 patch proposal budget per candidate image. |
| `max_partition_depth` | 4 | Maximum recursive quadtree/SGAP depth. |
| `max_sam3_eval_patches` | 150 | Stage 4 upper bound on expensive SAM3 patch scans. |
| `max_latent_evals` | 20 | Stage 4 speculative VLM latent checks on centroids. |
| `max_patches_per_detected_group` | 8 | Retained patches per confirmed SAM3 group. |
| `max_patches_per_latent_group` | 2 | Retained patches per speculative latent group. |
| `max_retained_patches` | 20 | Global candidate pool capacity before Stage 5/6. |
| `enable_early_consensus_exit` | `false` | Deactivated for multi-source pool completeness. |
| `max_latent_repartition_patches_per_image` | 4 | Max high-latent patches per image to re-partition in Stage 3. |
| `max_latent_repartition_depth` | 2 | Max recursion levels for latent re-partitioning. |
| `latent_repartition_threshold` | 0.0 | Minimum latent confidence to trigger Stage 3 re-partitioning. |
| `answer_type` | `"option_list"` | Multiple-choice option format for HRBench evaluation. |

---

## 5. Hardware Topology & Model Routing

MSSearch splits models across available CUDA devices:

| Device | Resident Engines | Roles |
| :--- | :--- | :--- |
| **GPU 0 (`cuda:0`)** | Qwen2.5-VL (Layers 0--14) + SigLIP (`siglip-base-patch16-384`) | SigLIP visual embedding, patch scoring, early VLM layers. |
| **GPU 1 (`cuda:1`)** | Qwen2.5-VL (Layers 15--27) + SAM3 (`sam3.pt`) | SAM3 dense mask segmentation, foveated VLM text generation. |
| **CPU / RAM** | spaCy (`en_core_web_sm-3.8.0`) | Grammatical noun chunk parsing & anchor discovery. |

*Note: In direct baseline mode (`--mode direct_answer`), SAM3, SigLIP, and spaCy are completely bypassed, allocating only Qwen2.5-VL on `cuda:0`.*

---

## 6. Output Artifacts & Telemetry

Results are written continuously with real-time atomic flushes:

- **Full Pipeline**:
  - `results/results_<timestamp>.jsonl` and mirrored canonical `results/results.jsonl`.
  - `results/summary.json` containing benchmark accuracy (`Acc@1`), timing breakdown, and stage routing counts.
  - `results/config_used.json` full hyperparameter snapshot.
- **Direct Baseline**:
  - `results_baseline/results_baseline_<timestamp>.jsonl` and `results_baseline/results_baseline.jsonl`.
  - `results_baseline/summary_baseline.json`.

### Visual Diagnostic Inspection
To inspect bounding box proposals, spatial compressed canvases, and Stage 6 foveated crops:
```bash
python visualize_results.py \
    --results_file results/results.jsonl \
    --hrbench_dir /home/ubuntu/workspace/hr/data/hr-bench_8k \
    --output_dir visual_audit/
```
Generates 2-row audit cards:
- **Row 1**: Full original scene canvases with gold proposal overlays.
- **Row 2**: Native-resolution crops (single-object) or continuous-coordinate spatial compressed canvas with 28px neutral visual gutters (cross-object relational).
