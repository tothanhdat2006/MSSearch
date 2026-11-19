# STASH-Search (Minimal Pipeline Implementation)

**STASH-Search** (*Source-Attributed Spatial-Heuristic Visual Search*) is a modular, high-resolution visual search engine designed to solve the Multi-Image High-Resolution Visual Search (HRVS) challenge across large candidate image galleries.

This directory (`minimal/`) provides a clean, self-contained implementation containing only the production pipeline components described in the paper, completely decoupling from legacy experimental code.

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
│   ├── base.py                      # BenchmarkSample dataclass
│   ├── stash_bench.py               # STASH-Bench nested galleries (N in {1, 5, 10, 20}, crop)
│   ├── vstar.py                     # V*Bench (direct_attributes/, relative_position/)
│   └── hr_bench.py                  # HR-Bench 4K / 8K (image/, 4-way choice rotations)
├── stages/
│   ├── __init__.py                  # Stage exports
│   ├── target_extraction.py         # Stage 1: Target concept extraction
│   ├── quadtree_partitioning.py     # Stage 2: Quadtree partitioning with plateau stopping
│   ├── visual_screening.py          # Stage 3: Top-kN screening, seam fusion, native crop reading
│   └── consensus.py                 # Stage 4: Source-Attributed Consensus (SAC)
├── engine.py                        # STASHSearchEngine coordinator
├── run.py                           # CLI runner (ad-hoc single queries and quick testing)
├── eval_runner.py                   # Batch benchmark evaluation runner with online metrics
├── scripts/                         # Local workstation execution bash scripts
│   ├── env_paths.sh                 # Centralized environment and dataset paths configuration
│   ├── run_stash_qwen3_vl_8b_*.sh   # STASH-Bench Qwen3-VL-8B across N=1, 5, 10, 20
│   ├── run_stash_llava_ov_7b_*.sh   # STASH-Bench LLaVA-OV-7B across N=1, 5, 10, 20
│   ├── run_stash_internvl3_8b_*.sh  # STASH-Bench InternVL3-8B across N=1, 5, 10, 20
│   ├── run_stash_qwen3_vl_32b_*.sh  # STASH-Bench Qwen3-VL-32B across N=1, 5, 10, 20
│   ├── run_vstar.sh                 # V*Bench evaluation script
│   ├── run_hrbench_4k.sh            # HR-Bench 4K evaluation script
│   ├── run_hrbench_8k.sh            # HR-Bench 8K evaluation script
│   └── run_all_n5.sh                # Batch runner for all 4 models on N=5
├── tests/
│   ├── __init__.py
│   ├── test_stages.py               # Unit tests for individual stages
│   ├── test_engine.py               # End-to-end integration test
│   └── test_datasets.py             # Schema and options parsing tests
└── README.md                        # Documentation
```

---

## Python API Usage

```python
from PIL import Image
from minimal.config import STASHConfig
from minimal.engine import STASHSearchEngine

# 1. Initialize engine with any of the 4 backbones (Qwen3-VL, LLaVA-OV, InternVL3)
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

---

## Batch Benchmark Evaluation (`eval_runner.py`)

`minimal/eval_runner.py` is the dedicated batch benchmark evaluator. It runs end-to-end search, streams JSONL logs with crash recovery, and calculates benchmark accuracy summaries upon completion.

### 1. STASH-Bench Evaluation
```bash
python -m minimal.eval_runner \
    --benchmark stash \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset_path /path/to/STASHBench/annotations_part1234_merged.json \
    --image_dir /path/to/STASHBench/pool \
    --pool_mode N_5 \
    --output_dir ./results/qwen_n5
```

### 2. V*Bench (Vstar) Evaluation
```bash
python -m minimal.eval_runner \
    --benchmark vstar \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset_dir /path/to/Vstar \
    --output_dir ./results/qwen_vstar
```

### 3. HR-Bench (4K / 8K) Evaluation
```bash
python -m minimal.eval_runner \
    --benchmark hrbench8k \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --dataset_dir /path/to/HRBench8K \
    --output_dir ./results/qwen_hrbench8k
```

---

## Running with Shell Scripts (`minimal/scripts/`)

1. Edit [`minimal/scripts/env_paths.sh`](file:///D:/Project/VisualSearch/experiments/multisource/MSSearch/minimal/scripts/env_paths.sh) once to point to your local models and dataset paths:
   ```bash
   export MODEL_DIR="/data/models"
   export STASH_DATASET_PATH="/data/STASHBench/annotations_part1234_merged.json"
   export STASH_IMAGE_DIR="/data/STASHBench/pool"
   export VSTAR_DIR="/data/Vstar"
   export HRBENCH_4K_DIR="/data/HRBench4K"
   export HRBENCH_8K_DIR="/data/HRBench8K"
   ```

2. Execute any benchmark script directly:
   ```bash
   # STASH-Bench evaluations across models and gallery sizes:
   bash minimal/scripts/run_stash_qwen3_vl_8b_n5.sh
   bash minimal/scripts/run_stash_llava_ov_7b_n5.sh
   bash minimal/scripts/run_stash_internvl3_8b_n5.sh
   bash minimal/scripts/run_stash_qwen3_vl_32b_n5.sh

   # External transfer benchmarks:
   bash minimal/scripts/run_vstar.sh Qwen/Qwen3-VL-8B-Instruct
   bash minimal/scripts/run_hrbench_4k.sh Qwen/Qwen3-VL-8B-Instruct
   bash minimal/scripts/run_hrbench_8k.sh Qwen/Qwen3-VL-8B-Instruct
   ```

---

## Running Automated Tests

Run the unit test suite (no external network or model weights required):
```bash
python -m unittest discover -s minimal/tests
```
