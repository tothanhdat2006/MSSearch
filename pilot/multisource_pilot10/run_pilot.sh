#!/usr/bin/env bash
set -euo pipefail

# Project root path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSSEARCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Python interpreter: search .venv in workspace or fallback
PYTHON_BIN="${WORKSPACE_ROOT}/.venv/Scripts/python.exe"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python"
fi

# Local dataset paths
DATASET_PATH="${WORKSPACE_ROOT}/docs/experiments/multisource/datasets/pilot_10_samples.json"
IMAGE_DIR="${WORKSPACE_ROOT}/docs/experiments/multisource/datasets/pilot_1000_images"
OUTPUT_DIR="${SCRIPT_DIR}/results"

if [ ! -f "${DATASET_PATH}" ]; then
    # Fallback to local datasets under MSSearch if cloned independently
    DATASET_PATH="${MSSEARCH_DIR}/datasets/pilot_10_samples.json"
    IMAGE_DIR="${MSSEARCH_DIR}/datasets/pilot_1000_images"
fi

# Model weights
SAM3_PATH="${SAM3_PATH:-/kaggle/input/datasets/thnhtt/vs-sam3/sam3/sam3/sam3.pt}"
QWEN_PATH="${QWEN_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
SIGLIP_NAME="${SIGLIP_NAME:-google/siglip-base-patch16-384}"
NLP_MODEL="${NLP_MODEL:-en_core_web_sm}"

# Execution mode: "N_5" or "ALL"
POOL_MODE="${1:-N_5}"
MAX_SAMPLES="${2:-}"

echo "=========================================================================="
echo "RUNNING LOCAL PILOT MULTI-SOURCE EVALUATION"
echo "=========================================================================="
echo "Project Root : ${PROJECT_ROOT}"
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

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_pilot.py" "${ARGS[@]}"
