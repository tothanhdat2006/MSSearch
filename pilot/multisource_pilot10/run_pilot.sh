#!/usr/bin/env bash
set -euo pipefail

# Project root path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSSEARCH_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
WORKSPACE_ROOT="${PROJECT_ROOT}"

# Activate Conda Environment: hrsearch
CONDA_ENV_NAME="hrsearch"
if [ -f "/home/ubuntu/miniconda3/etc/profile.d/conda.sh" ]; then
    set +u
    source "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
    set -u
elif command -v conda &> /dev/null; then
    set +u
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true
    set -u
fi

# Ensure conda env bin is at the front of PATH
if [ -d "/home/ubuntu/miniconda3/envs/${CONDA_ENV_NAME}/bin" ]; then
    export PATH="/home/ubuntu/miniconda3/envs/${CONDA_ENV_NAME}/bin:${PATH}"
fi

# Python interpreter: prefer hrsearch environment
PYTHON_BIN="/home/ubuntu/miniconda3/envs/${CONDA_ENV_NAME}/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python || echo "python")"
fi

# Local dataset paths
DATASET_PATH="/home/ubuntu/workspace/phong/multi_image_visual_search/data/output/pilot_10_samples.json"
IMAGE_DIR="/home/ubuntu/workspace/phong/multi_image_visual_search/data/pilot_1000_images"
OUTPUT_DIR="${SCRIPT_DIR}/results"

# Model weights
SAM3_PATH="${SAM3_PATH:-/home/ubuntu/workspace/hr/models/sam3/sam3/sam3.pt}"
QWEN_PATH="${QWEN_PATH:-/home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct}"
SIGLIP_NAME="${SIGLIP_NAME:-google/siglip-base-patch16-384}"
NLP_MODEL="${NLP_MODEL:-/home/ubuntu/workspace/hr/models/spacy/en_core_web_sm-3.8.0}"

# Execution mode: "N_5" or "ALL"
POOL_MODE="${1:-N_5}"
MAX_SAMPLES="${2:-}"

echo "=========================================================================="
echo "RUNNING LOCAL PILOT MULTI-SOURCE EVALUATION"
echo "=========================================================================="
echo "Project Root : ${PROJECT_ROOT}"
echo "Conda Env    : ${CONDA_ENV_NAME}"
echo "Python       : ${PYTHON_BIN}"
echo "Dataset      : ${DATASET_PATH}"
echo "Images Dir   : ${IMAGE_DIR}"
echo "Output Dir   : ${OUTPUT_DIR}"
echo "Pool Mode    : ${POOL_MODE}"
echo "SAM3 Path    : ${SAM3_PATH}"
echo "=========================================================================="

ARGS=(
    "--dataset_path" "${DATASET_PATH}"
    "--image_dir" "${IMAGE_DIR}"
    "--output_dir" "${OUTPUT_DIR}"
    "--sam3_path" "${SAM3_PATH}"
    "--qwen_path" "${QWEN_PATH}"
    "--siglip_name" "${SIGLIP_NAME}"
    "--nlp_model" "${NLP_MODEL}"
    "--pool_mode" "${POOL_MODE}"
)

if [ -n "${MAX_SAMPLES}" ]; then
    ARGS+=("--max_samples" "${MAX_SAMPLES}")
fi

if [ -n "${RESUME_FROM:-}" ]; then
    ARGS+=("--resume_from" "${RESUME_FROM}")
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_pilot.py" "${ARGS[@]}"
