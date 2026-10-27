#!/usr/bin/env bash
set -euo pipefail

# Directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSSEARCH_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate Conda Environment: hrsearch (if available)
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

# Python interpreter selection
PYTHON_BIN="/home/ubuntu/miniconda3/envs/${CONDA_ENV_NAME}/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python || echo "/c/Python312/python.exe")"
fi

# PyTorch & CUDA allocations
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# CVSearch Repository Discovery:
# Default path priority: 1) Same level as this folder: ../ICML26-CVSearch
#                        2) Reference path: ../../../../freeform/ICML26-CVSearch
SAME_LEVEL_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)/ICML26-CVSearch"
REF_REPO="$(cd "${SCRIPT_DIR}/../../../../freeform/ICML26-CVSearch" 2>/dev/null && pwd || true)"

if [ -d "${SAME_LEVEL_REPO}" ]; then
    CVSEARCH_REPO="${SAME_LEVEL_REPO}"
elif [ -n "${REF_REPO}" ] && [ -d "${REF_REPO}" ]; then
    CVSEARCH_REPO="${REF_REPO}"
else
    CVSEARCH_REPO="${SAME_LEVEL_REPO}"
fi

# Default model checkpoints on server / Kaggle
QWEN_PATH="${QWEN_PATH:-/home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct}"
SAM_PATH="${SAM_PATH:-/home/ubuntu/workspace/hr/models/sam3/sam3/sam3.pt}"
NLP_MODEL="${NLP_MODEL:-/home/ubuntu/workspace/hr/models/spacy/en_core_web_sm-3.8.0}"
DATASET_PATH="${DATASET_PATH:-}"
IMAGE_DIR="${IMAGE_DIR:-}"
OUTPUT_DIR="${SCRIPT_DIR}/results"
N_SAMPLES="${1:-30}"

echo "=========================================================================="
echo "PILOT STUDY: DOES CVSEARCH REALIZE NO EVIDENCE EXISTS? (N=${N_SAMPLES})"
echo "=========================================================================="
echo "Python Binary : ${PYTHON_BIN}"
echo "CVSearch Repo : ${CVSEARCH_REPO}"
echo "Qwen2.5-VL    : ${QWEN_PATH}"
echo "SAM3 Weights  : ${SAM_PATH}"
echo "NLP Model     : ${NLP_MODEL}"
echo "Output Dir    : ${OUTPUT_DIR}"
echo "=========================================================================="

ARGS=(
    "--cvsearch_repo" "${CVSEARCH_REPO}"
    "--output_dir" "${OUTPUT_DIR}"
    "--n_samples" "${N_SAMPLES}"
    "--seed" "42"
    "--qwen_path" "${QWEN_PATH}"
    "--sam_path" "${SAM_PATH}"
    "--nlp_model" "${NLP_MODEL}"
)

if [ -n "${DATASET_PATH}" ]; then
    ARGS+=("--dataset_path" "${DATASET_PATH}")
fi

if [ -n "${IMAGE_DIR}" ]; then
    ARGS+=("--image_dir" "${IMAGE_DIR}")
fi

# Pass remaining flags (e.g. --direct_answer, --mock)
shift || true
ARGS+=("$@")

"${PYTHON_BIN}" "${SCRIPT_DIR}/pilot_study.py" "${ARGS[@]}"
