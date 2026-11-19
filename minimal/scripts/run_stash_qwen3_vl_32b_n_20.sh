#!/usr/bin/env bash
# ==============================================================================
# STASH-SEARCH MINIMAL: Qwen3-VL-32B (N_20)
# Evaluates STASH-Bench multi-image visual search gallery
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIMAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${SCRIPT_DIR}/env_paths.sh"

if [ -f "/home/ubuntu/miniconda3/etc/profile.d/conda.sh" ]; then
    set +u; source "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true; set -u
elif command -v conda &> /dev/null; then
    set +u; eval "$(conda shell.bash hook)" 2>/dev/null || true; conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true; set -u
elif [ -f "${VENV_PATH}" ]; then
    source "${VENV_PATH}"
fi

PYTHON_BIN="$(command -v python || echo 'python')"
export PYTHONPATH="${MINIMAL_DIR}/..:${PYTHONPATH:-}"

TARGET_MODEL="Qwen/Qwen3-VL-32B-Instruct"
M_BASE="$(basename "${TARGET_MODEL}")"
MODEL_PATH="${MODEL_DIR}/${M_BASE}"
[ ! -d "${MODEL_PATH}" ] && MODEL_PATH="${TARGET_MODEL}"

POOL_MODE="N_20"
OUTPUT_DIR="${OUTPUT_BASE_DIR:-./results}/${M_BASE}_${POOL_MODE}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================================================="
echo "STASH-Search Minimal: Qwen3-VL-32B (${POOL_MODE})"
echo "Model Path : ${MODEL_PATH}"
echo "Pool Mode  : ${POOL_MODE}"
echo "Output Dir : ${OUTPUT_DIR}"
echo "=========================================================================="

exec "${PYTHON_BIN}" "${MINIMAL_DIR}/eval_runner.py" \
    --benchmark "stash" \
    --model_path "${MODEL_PATH}" \
    --dual_encoder_name "${SIGLIP_NAME}" \
    --nlp_model "${NLP_MODEL}" \
    --dataset_path "${STASH_DATASET_PATH}" \
    --image_dir "${STASH_IMAGE_DIR}" \
    --gallery_order_path "${GALLERY_ORDER_PATH}" \
    --pool_mode "${POOL_MODE}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
