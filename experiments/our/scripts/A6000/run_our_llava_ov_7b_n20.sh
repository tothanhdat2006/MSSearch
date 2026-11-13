#!/usr/bin/env bash
# ==============================================================================
# MSSEARCH V2 EVALUATION (A6000 SERVER): llava-onevision-qwen2-7b-ov (N_20)
# All paths and environment configurations are loaded from:
#   experiments/our/env_paths.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MSSEARCH_ROOT="$(cd "${OUR_DIR}/../.." && pwd)"

# Source centralized environment and paths
source "${OUR_DIR}/env_paths.sh"

# Activate environment
if [ -f "/home/ubuntu/miniconda3/etc/profile.d/conda.sh" ]; then
    set +u; source "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true; set -u
elif command -v conda &> /dev/null; then
    set +u; eval "$(conda shell.bash hook)" 2>/dev/null || true; conda activate "${CONDA_ENV_NAME}" 2>/dev/null || true; set -u
elif [ -f "${VENV_PATH}" ]; then
    source "${VENV_PATH}"
fi

PYTHON_BIN="$(command -v python || echo "python")"
export PYTHONPATH="${MSSEARCH_ROOT}:${OUR_DIR}:${PYTHONPATH:-}"

TARGET_MODEL="lmms-lab/llava-onevision-qwen2-7b-ov"
M_BASE="$(basename "${TARGET_MODEL}")"
MODEL_PATH="${MODEL_DIR}/${M_BASE}"
[ ! -d "${MODEL_PATH}" ] && MODEL_PATH="${TARGET_MODEL}"

POOL_MODE="N_20"
OUTPUT_DIR="${OUTPUT_BASE_DIR:-${OUR_DIR}/results_our_a6000_${M_BASE}_${POOL_MODE}}"
CONFIG_PATH="${OUR_DIR}/config_default.json"

echo "=========================================================================="
echo "A6000 MSSEARCH V2: ${TARGET_MODEL} (${POOL_MODE})"
echo "Model Path : ${MODEL_PATH}"
echo "Pool Mode  : ${POOL_MODE}"
echo "Output Dir : ${OUTPUT_DIR}"
echo "=========================================================================="

mkdir -p "${OUTPUT_DIR}"
exec "${PYTHON_BIN}" "${OUR_DIR}/eval_runner.py" \
    --model_path "${MODEL_PATH}" \
    --siglip_name "${SIGLIP_NAME}" \
    --nlp_model "${NLP_MODEL}" \
    --dataset_path "${DATASET_PATH}" \
    --image_dir "${IMAGE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --config "${CONFIG_PATH}" \
    --pool_mode "${POOL_MODE}" \
    "$@"
