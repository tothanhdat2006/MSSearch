#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MSSEARCH V2 EVALUATION RUNNER ON LOCAL A6000 SERVER
# ==============================================================================
# EDIT THIS PATH BLOCK TO MATCH YOUR LOCAL SERVER CONFIGURATION
# ------------------------------------------------------------------------------
CONDA_ENV_NAME="hrsearch"                                                      # EDIT: Conda environment name
DEFAULT_MODEL_DIR="/data/shared/cvpr/dat/models"                               # EDIT: Root folder for local model checkpoints
DEFAULT_SAM3_PATH="/data/shared/cvpr/dat/models/sam3/sam3.pt"                  # EDIT: Path to sam3.pt
DEFAULT_SIGLIP_NAME="/data/shared/cvpr/dat/models/siglip-base-patch16-384"     # EDIT: Path to SigLIP weights
DEFAULT_NLP_MODEL="/data/shared/cvpr/dat/models/spacy/en_core_web_sm-3.8.0"    # EDIT: Path to spaCy pipeline
DEFAULT_DATASET_PATH="/data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json"     # EDIT: Path to STASHBench annotations
DEFAULT_IMAGE_DIR="/data/shared/cvpr/dat/simple/data/STASHBench/pool"                                   # EDIT: Path to pool images directory
DEFAULT_OUTPUT_DIR="./results_our_a6000"                                       # EDIT: Output directory
# ==============================================================================

# Activate Conda Environment
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

PYTHON_BIN="$(command -v python || echo "python")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSSEARCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${MSSEARCH_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"

# Offline HuggingFace execution
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

# Target Defaults
MODEL_ARG="${1:-Qwen2.5-VL-7B-Instruct}"
shift || true

if [ -d "${DEFAULT_MODEL_DIR}/${MODEL_ARG}" ]; then
    MODEL_PATH="${DEFAULT_MODEL_DIR}/${MODEL_ARG}"
elif [ -d "${MODEL_ARG}" ]; then
    MODEL_PATH="${MODEL_ARG}"
else
    MODEL_PATH="${MODEL_ARG}"
fi

POOL_MODE="N_20"  # Default candidate pool size
CONFIG_PATH="${SCRIPT_DIR}/config_default.json"

echo "=========================================================================="
echo "A6000 MSSEARCH V2 EVALUATION SPECIFICATION"
echo "=========================================================================="
echo "Python Exec  : ${PYTHON_BIN}"
echo "Model Path   : ${MODEL_PATH}"
echo "SAM3 Path    : ${DEFAULT_SAM3_PATH}"
echo "SigLIP Path  : ${DEFAULT_SIGLIP_NAME}"
echo "Dataset Path : ${DEFAULT_DATASET_PATH}"
echo "Image Dir    : ${DEFAULT_IMAGE_DIR}"
echo "Pool Mode    : ${POOL_MODE} (Default)"
echo "Config Path  : ${CONFIG_PATH}"
echo "Output Dir   : ${DEFAULT_OUTPUT_DIR}"
echo "=========================================================================="

mkdir -p "${DEFAULT_OUTPUT_DIR}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/eval_runner.py" \
    --model_path "${MODEL_PATH}" \
    --sam3_path "${DEFAULT_SAM3_PATH}" \
    --siglip_name "${DEFAULT_SIGLIP_NAME}" \
    --nlp_model "${DEFAULT_NLP_MODEL}" \
    --dataset_path "${DEFAULT_DATASET_PATH}" \
    --image_dir "${DEFAULT_IMAGE_DIR}" \
    --output_dir "${DEFAULT_OUTPUT_DIR}" \
    --config "${CONFIG_PATH}" \
    --pool_mode "${POOL_MODE}" \
    "$@"
