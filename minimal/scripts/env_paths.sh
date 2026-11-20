#!/usr/bin/env bash
# ==============================================================================
# STASH-SEARCH MINIMAL: CENTRALIZED ENVIRONMENT & PATH CONFIGURATION
# Edit this single file to point to your local models and dataset paths.
# All run scripts in minimal/scripts/ source this file automatically.
# ==============================================================================

# 1. Python Environment Activation
export CONDA_ENV_NAME="hrsearch"
export VENV_PATH="${VENV_PATH:-/path/to/.venv/bin/activate}"

# 2. Perception & Dual-Encoder Checkpoints
export MODEL_DIR="${MODEL_DIR:-/data/models}"
export SIGLIP_NAME="${SIGLIP_NAME:-google/siglip2-so400m-patch16-512}"
export NLP_MODEL="${NLP_MODEL:-en_core_web_sm}"

# 3. STASH-Bench Dataset & Candidate Pool Images
export STASH_DATASET_PATH="${STASH_DATASET_PATH:-/data/STASHBench/annotations_part1234_merged.json}"
export STASH_IMAGE_DIR="${STASH_IMAGE_DIR:-/data/STASHBench/pool}"
export GALLERY_ORDER_PATH="${GALLERY_ORDER_PATH:-}"

# 4. External High-Resolution Transfer Benchmarks
# V*Bench (Vstar): contains sub-folders 'direct_attributes/', 'relative_position/', and 'annotation_vstar.json'
export VSTAR_DIR="${VSTAR_DIR:-/data/Vstar}"
export VSTAR_ANNOTATION="${VSTAR_ANNOTATION:-${VSTAR_DIR}/annotation_vstar.json}"

# HR-Bench 4K & 8K: contains sub-folder 'image/' and 'annotation_hr-bench_{4k|8k}.json'
export HRBENCH_4K_DIR="${HRBENCH_4K_DIR:-/data/HRBench4K}"
export HRBENCH_4K_ANNOTATION="${HRBENCH_4K_ANNOTATION:-${HRBENCH_4K_DIR}/annotation_hr-bench_4k.json}"

export HRBENCH_8K_DIR="${HRBENCH_8K_DIR:-/data/HRBench8K}"
export HRBENCH_8K_ANNOTATION="${HRBENCH_8K_ANNOTATION:-${HRBENCH_8K_DIR}/annotation_hr-bench_8k.json}"

# TreeBench: contains 'images/' and 'annotation_treebench.json'
export TREEBENCH_DIR="${TREEBENCH_DIR:-D:/MachineLearning/Datasets/treebench}"
[ ! -d "${TREEBENCH_DIR}" ] && export TREEBENCH_DIR="/data/TreeBench"
export TREEBENCH_ANNOTATION="${TREEBENCH_ANNOTATION:-${TREEBENCH_DIR}/annotation_treebench.json}"

# MME-RealWorld-Lite: contains 'imgs/' and 'annotation_mme-realworld-lite.json'
export MME_REALWORLD_DIR="${MME_REALWORLD_DIR:-D:/MachineLearning/Datasets/mme-realworld-lite}"
[ ! -d "${MME_REALWORLD_DIR}" ] && export MME_REALWORLD_DIR="/data/mme-realworld-lite"
export MME_REALWORLD_ANNOTATION="${MME_REALWORLD_ANNOTATION:-${MME_REALWORLD_DIR}/annotation_mme-realworld-lite.json}"

# 5. Base Output Directory
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-./results}"
