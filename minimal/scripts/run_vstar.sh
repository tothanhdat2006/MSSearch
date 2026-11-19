#!/usr/bin/env bash
# ==============================================================================
# STASH-SEARCH MINIMAL: V*Bench (Vstar) Evaluation
# Evaluates single-image visual search (direct_attributes & relative_position)
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

TARGET_MODEL="${1:-Qwen/Qwen3-VL-8B-Instruct}"
[ "$#" -ge 1 ] && shift

M_BASE="$(basename "${TARGET_MODEL}")"
MODEL_PATH="${MODEL_DIR}/${M_BASE}"
[ ! -d "${MODEL_PATH}" ] && MODEL_PATH="${TARGET_MODEL}"

OUTPUT_DIR="${OUTPUT_BASE_DIR:-./results}/${M_BASE}_vstar"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================================================="
echo "STASH-Search Minimal: V*Bench Evaluation"
echo "Model Path : ${MODEL_PATH}"
echo "Dataset Dir: ${VSTAR_DIR}"
echo "Annotation : ${VSTAR_ANNOTATION}"
echo "Output Dir : ${OUTPUT_DIR}"
echo "=========================================================================="

exec "${PYTHON_BIN}" "${MINIMAL_DIR}/eval_runner.py" \
    --benchmark "vstar" \
    --model_path "${MODEL_PATH}" \
    --dual_encoder_name "${SIGLIP_NAME}" \
    --nlp_model "${NLP_MODEL}" \
    --dataset_dir "${VSTAR_DIR}" \
    --dataset_path "${VSTAR_ANNOTATION}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
