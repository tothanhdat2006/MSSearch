# STASH-Search (Minimal Pipeline Implementation)

**STASH-Search** (*Source-Attributed Spatial-Heuristic Visual Search*) is a modular, high-resolution visual search engine designed to solve the Multi-Image High-Resolution Visual Search (HRVS) challenge across large candidate image galleries.

This directory (`minimal/`) provides a clean, self-contained implementation containing only the production pipeline components described in the paper, completely decoupled from legacy experimental code.

---

## Directory Layout

```
minimal/
├── __init__.py                      # Package exports (STASHSearchEngine, STASHConfig)
├── config.py                        # Operational hyperparameters (Table S1)
├── models/
│   ├── __init__.py                  # Model exports with lazy __getattr__ loading
│   ├── base.py                      # Abstract interfaces (BaseMLLM, BaseDualEncoder)
│   ├── factory.py                   # load_mllm factory dispatcher
│   ├── dual_encoder.py              # SigLIP-2 / CLIP scorer with text embedding caching
│   ├── modeling_qwenvl.py           # ModelQwenVL: Qwen3-VL-8B and Qwen3-VL-32B adapter
│   ├── modeling_llava.py            # ModelGlobalLocal: LLaVA-OneVision-7B adapter
│   └── modeling_internvl.py         # ModelInternvl: InternVL3-8B adapter
├── datasets/
│   ├── __init__.py                  # load_benchmark_dataset dispatcher
│   ├── base.py                      # BenchmarkSample dataclass (with inspection_prompt)
│   ├── stash_bench.py               # STASH-Bench nested galleries (N in {1, 5, 10, 20}, crop)
│   ├── vstar.py                     # V*Bench (direct_attributes/, relative_position/)
│   ├── hr_bench.py                  # HR-Bench 4K / 8K (image/, 4-way choice rotations)
│   ├── treebench.py                 # TreeBench loader & format_question_new prompt
│   ├── mme_realworld.py             # MME-RealWorld-Lite loader & format_question_multichoice
│   └── download_data.py             # ModelScope benchmark dataset downloader & tar extractor
├── stages/
│   ├── __init__.py                  # Stage exports
│   ├── target_extraction.py         # Stage 1: Target concept extraction
│   ├── quadtree_partitioning.py     # Stage 2: Quadtree partitioning with plateau stopping
│   ├── visual_screening.py          # Stage 3: Top-kN screening, seam fusion, native crop reading
│   └── consensus.py                 # Stage 4: Source-Attributed Consensus (SAC)
├── engine.py                        # STASHSearchEngine coordinator (supports inspection_prompt)
├── run.py                           # CLI runner (ad-hoc single queries and quick testing)
├── eval_runner.py                   # Batch benchmark evaluation runner with online metrics
├── scripts/                         # Local workstation execution bash scripts
│   ├── env_paths.sh                 # Centralized environment and dataset paths configuration
│   ├── download_datasets.sh         # Dataset download runner (ModelScope llp1995/hr_data)
│   ├── run_stash_qwen3_vl_8b_*.sh   # STASH-Bench Qwen3-VL-8B across N=1, 5, 10, 20
│   ├── run_stash_llava_ov_7b_*.sh   # STASH-Bench LLaVA-OV-7B across N=1, 5, 10, 20
│   ├── run_stash_internvl3_8b_*.sh  # STASH-Bench InternVL3-8B across N=1, 5, 10, 20
│   ├── run_stash_qwen3_vl_32b_*.sh  # STASH-Bench Qwen3-VL-32B across N=1, 5, 10, 20
│   ├── run_vstar.sh                 # V*Bench evaluation script
│   ├── run_hrbench_4k.sh            # HR-Bench 4K evaluation script
│   ├── run_hrbench_8k.sh            # HR-Bench 8K evaluation script
│   ├── run_treebench.sh             # TreeBench evaluation script
│   ├── run_mme_realworld.sh         # MME-RealWorld-Lite evaluation script
│   └── run_all_n5.sh                # Batch runner for all 4 models on N=5
├── tests/
│   ├── __init__.py
│   ├── test_stages.py               # Unit tests for individual stages & batch fallback
│   ├── test_engine.py               # End-to-end integration tests (options & free-form)
│   ├── test_datasets.py             # Schema parsing, prompt formatting & download registry tests
│   └── test_eval_runner.py          # Runner argument resolution, option extraction & metrics tests
└── README.md                        # Documentation
```

---

## Dataset Downloading (`download_data.py`)

STASH-Search provides a unified downloader for high-resolution benchmark datasets hosted on ModelScope (`llp1995/hr_data`).

### Supported Benchmarks
| Benchmark | Archive Filename | Size | Direct Download Source |
|:---|:---|:---|:---|
| **MME-RealWorld-Lite** | `mme-realworld-lite.tar.gz` | 8.78 GB | [ModelScope `hr_data`](https://modelscope.cn/datasets/llp1995/hr_data/resolve/master/mme-realworld-lite.tar.gz) |
| **TreeBench** | `treebench.tar.gz` | 215.5 MB | [ModelScope `hr_data`](https://modelscope.cn/datasets/llp1995/hr_data/resolve/master/treebench.tar.gz) |
| **V\*Bench** | `vstar.tar.gz` | 280.9 MB | [ModelScope `hr_data`](https://modelscope.cn/datasets/llp1995/hr_data/resolve/master/vstar.tar.gz) |
| **HR-Bench 8K** | `hr-bench_8k.tar.gz` | 843.3 MB | [ModelScope `hr_data`](https://modelscope.cn/datasets/llp1995/hr_data/resolve/master/hr-bench_8k.tar.gz) |
| **HR-Bench 4K** | `hr-bench_4k.tar.gz` | 353.2 MB | [ModelScope `hr_data`](https://modelscope.cn/datasets/llp1995/hr_data/resolve/master/hr-bench_4k.tar.gz) |

### Usage
```bash
# Download and extract all datasets to default directory:
bash minimal/scripts/download_datasets.sh all

# Download specific datasets:
bash minimal/scripts/download_datasets.sh vstar treebench

# Python CLI with custom destination:
python -m minimal.datasets.download_data \
    --datasets vstar treebench hrbench8k \
    --output_dir /data/datasets

# Quick connectivity test (streams for 5 seconds without storing multi-GB archives):
python -m minimal.datasets.download_data --datasets vstar --test_seconds 5
```

---

## Python API Usage

```python
from PIL import Image
from minimal.config import STASHConfig
from minimal.engine import STASHSearchEngine

# 1. Initialize engine with any supported backbone (Qwen3-VL, LLaVA-OV, InternVL3)
engine = STASHSearchEngine.from_pretrained(
    mllm_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
    dual_encoder_name="google/siglip2-so400m-patch16-512",
    config=STASHConfig(k=20),
    device="cuda:0"
)

# 2. Run multi-image visual search
gallery = [Image.open(f"img_{i}.jpg") for i in range(5)]
result = engine.search(
    candidate_images=gallery,
    question="What is the registration number of the boat?"
)

print("Answer:", result["primary_answer"])
print("Supporting Image Index:", result["primary_evidence_image"])
```

### Multiple-Choice Benchmarks
For transfer benchmarks with predefined answer options, pass `inspection_prompt` or `options_str`:
```python
# Pass benchmark-native multiple-choice prompt
result = engine.search(
    candidate_images=gallery,
    question="Which leaf has an aphid?",
    options_str="A. Top left leaf\nB. Top right leaf\nC. Center stem\nD. None",
    inspection_prompt=custom_prompt
)
```

---

## Batch Benchmark Evaluation (`eval_runner.py`)

[`minimal/eval_runner.py`](eval_runner.py) manages end-to-end benchmark evaluation, EXIF rotation handling, telemetry tracking, streaming JSONL records, and online accuracy reporting.

### 1. STASH-Bench Evaluation
```bash
python -m minimal.eval_runner \
    --benchmark stash \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset_path /data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/STASHBench/pool \
    --pool_mode N_5 \
    --gallery_order_path /data/STASHBench/gallery_order_seed42.json \
    --output_dir ./results/qwen_n5
```

### 2. V\*Bench (Vstar) Evaluation
```bash
python -m minimal.eval_runner \
    --benchmark vstar \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset_dir /data/Vstar \
    --output_dir ./results/qwen_vstar
```

### 3. HR-Bench (4K / 8K) Evaluation
```bash
python -m minimal.eval_runner \
    --benchmark hrbench8k \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset_dir /data/HRBench8K \
    --output_dir ./results/qwen_hrbench8k
```

### 4. TreeBench Evaluation
```bash
python -m minimal.eval_runner \
    --benchmark treebench \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset_dir /data/TreeBench \
    --output_dir ./results/qwen_treebench
```

### 5. MME-RealWorld-Lite Evaluation
```bash
python -m minimal.eval_runner \
    --benchmark mme-realworld-lite \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset_dir /data/mme-realworld-lite \
    --output_dir ./results/qwen_mme_realworld
```

---

## Workstation Shell Scripts (`minimal/scripts/`)

1. Edit [`minimal/scripts/env_paths.sh`](scripts/env_paths.sh) once to point to local model weights and dataset directories:
   ```bash
   export MODEL_DIR="/data/models"
   export STASH_DATASET_PATH="/data/STASHBench/annotations_part1234_merged.json"
   export STASH_IMAGE_DIR="/data/STASHBench/pool"
   export VSTAR_DIR="/data/Vstar"
   export HRBENCH_4K_DIR="/data/HRBench4K"
   export HRBENCH_8K_DIR="/data/HRBench8K"
   export TREEBENCH_DIR="/data/TreeBench"
   export MME_REALWORLD_DIR="/data/mme-realworld-lite"
   ```

2. Execute evaluation scripts directly:
   ```bash
   # STASH-Bench sweeps:
   bash minimal/scripts/run_stash_qwen3_vl_8b_n5.sh
   bash minimal/scripts/run_stash_llava_ov_7b_n5.sh
   bash minimal/scripts/run_stash_internvl3_8b_n5.sh
   bash minimal/scripts/run_stash_qwen3_vl_32b_n5.sh
   bash minimal/scripts/run_all_n5.sh

   # Transfer benchmarks:
   bash minimal/scripts/run_vstar.sh Qwen/Qwen3-VL-8B-Instruct
   bash minimal/scripts/run_hrbench_4k.sh Qwen/Qwen3-VL-8B-Instruct
   bash minimal/scripts/run_hrbench_8k.sh Qwen/Qwen3-VL-8B-Instruct
   bash minimal/scripts/run_treebench.sh Qwen/Qwen3-VL-8B-Instruct
   bash minimal/scripts/run_mme_realworld.sh Qwen/Qwen3-VL-8B-Instruct
   ```

---

## Key Pipeline Invariants & Production Fixes

- **Stage 5 Deduplication**: Implements $\mathrm{IoS} \ge 0.95$ containment merging and $\mathrm{IoU} \ge 0.60$ spatial per-image NMS prior to atomic sibling re-fusion.
- **Tri-Factor Spatial Entropy Pruning**: Computes Shannon entropy, Sobel gradient density, and Laplacian variance, pruning uninformative background leaves ($H < 0.35$ unless visual alignment score $\ge 0.35$).
- **Top-$kN$ Retention Tie-Breaker**: Sorts globally retained candidate patches by `(alignment_score, entropy)`, resolving exact fp16 SigLIP ties at the cutoff.
- **Native Batched Crop Reading**: Executes batched inference in chunks of 4 (`padding_side="left"`), with warning emission on fallback.
- **EXIF Transposition**: Automatically normalizes rotated camera orientations (`ImageOps.exif_transpose`) across gallery images.
- **Gallery Key Normalization**: Maps STASH-Bench pool keys (`"N_5"` $\to$ `"N5"`, etc.) and enforces `gallery_order_seed42.json` consistency.

---

## Running Automated Tests

Run the full self-contained unit test suite (33 tests, no external network or heavy model weights required):
```bash
python -m unittest discover -s minimal/tests
```
