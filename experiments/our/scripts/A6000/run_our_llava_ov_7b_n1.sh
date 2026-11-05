#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MSSEARCH V2 EVALUATION RUNNER ON LOCAL A6000 SERVER
# Model: llava-onevision-qwen2-7b-ov (lmms-lab/llava-onevision-qwen2-7b-ov)
# Candidate Pool: N_1 (N=1)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MSSEARCH_ROOT="$(cd "${OUR_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${MSSEARCH_ROOT}/../.." && pwd)"

# Conda environment activation
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hrsearch}"
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
export PYTHONPATH="${MSSEARCH_ROOT}:${OUR_DIR}:${PYTHONPATH:-}"

# HuggingFace & Transformers cache settings
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

# Auto-detect Model Directory & Model Path
DEFAULT_MODEL_DIR="${MODEL_DIR:-/data/shared/cvpr/dat/models}"
TARGET_MODEL="lmms-lab/llava-onevision-qwen2-7b-ov"
M_BASE="$(basename "${TARGET_MODEL}")"

if [ -d "${DEFAULT_MODEL_DIR}/llava-onevision-qwen2-7b-ov" ]; then
    MODEL_PATH="${DEFAULT_MODEL_DIR}/llava-onevision-qwen2-7b-ov"
elif [ -d "${DEFAULT_MODEL_DIR}/${M_BASE}" ]; then
    MODEL_PATH="${DEFAULT_MODEL_DIR}/${M_BASE}"
elif [ -d "${DEFAULT_MODEL_DIR}/${TARGET_MODEL}" ]; then
    MODEL_PATH="${DEFAULT_MODEL_DIR}/${TARGET_MODEL}"
elif [ -d "${TARGET_MODEL}" ]; then
    MODEL_PATH="${TARGET_MODEL}"
else
    MODEL_PATH="${TARGET_MODEL}"
fi

# Auto-detect Auxiliary Perception Weights
SAM3_PATH="${SAM3_PATH:-}"
for cand in \
    "${SAM3_PATH}" \
    "${DEFAULT_MODEL_DIR}/sam3/sam3.pt" \
    "${MSSEARCH_ROOT}/models/sam3/sam3.pt" \
    "${REPO_ROOT}/models/sam3/sam3.pt"; do
    if [ -n "${cand}" ] && [ -f "${cand}" ]; then
        SAM3_PATH="${cand}"
        break
    fi
done
SAM3_PATH="${SAM3_PATH:-${DEFAULT_MODEL_DIR}/sam3/sam3.pt}"

SIGLIP_NAME="${SIGLIP_NAME:-}"
for cand in \
    "${SIGLIP_NAME}" \
    "${DEFAULT_MODEL_DIR}/siglip-base-patch16-384" \
    "${MSSEARCH_ROOT}/models/siglip-base-patch16-384" \
    "google/siglip-base-patch16-384"; do
    if [ -n "${cand}" ] && ([ -d "${cand}" ] || [ "${cand}" = "google/siglip-base-patch16-384" ]); then
        SIGLIP_NAME="${cand}"
        break
    fi
done
SIGLIP_NAME="${SIGLIP_NAME:-${DEFAULT_MODEL_DIR}/siglip-base-patch16-384}"

DEFAULT_NLP_MODEL="${NLP_MODEL:-${DEFAULT_MODEL_DIR}/spacy/en_core_web_sm-3.8.0}"

# Auto-detect Dataset Path & Pool Images
DATASET_PATH="${DATASET_PATH:-}"
for cand in \
    "${DATASET_PATH}" \
    "/data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json" \
    "${MSSEARCH_ROOT}/data/STASHBench/annotations_part1234_merged.json" \
    "${REPO_ROOT}/data/STASHBench/annotations_part1234_merged.json" \
    "${OUR_DIR}/../../../../data/STASHBench/annotations_part1234_merged.json"; do
    if [ -n "${cand}" ] && [ -f "${cand}" ]; then
        DATASET_PATH="${cand}"
        break
    fi
done
DATASET_PATH="${DATASET_PATH:-/data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json}"

IMAGE_DIR="${IMAGE_DIR:-}"
for cand in \
    "${IMAGE_DIR}" \
    "/data/shared/cvpr/dat/simple/data/STASHBench/pool" \
    "$(dirname "${DATASET_PATH}")/pool" \
    "${MSSEARCH_ROOT}/data/STASHBench/pool" \
    "${REPO_ROOT}/data/STASHBench/pool"; do
    if [ -n "${cand}" ] && [ -d "${cand}" ]; then
        IMAGE_DIR="${cand}"
        break
    fi
done
IMAGE_DIR="${IMAGE_DIR:-/data/shared/cvpr/dat/simple/data/STASHBench/pool}"

POOL_MODE="N_1"
CONFIG_PATH="${OUR_DIR}/config_default.json"
OUTPUT_DIR="${OUTPUT_DIR:-${OUR_DIR}/results_our_a6000_${M_BASE}_${POOL_MODE}}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================================================="
echo "A6000 MSSEARCH V2 RUNNER: llava-onevision-qwen2-7b-ov (${POOL_MODE})"
echo "Model        : ${MODEL_PATH}"
echo "Pool Mode    : ${POOL_MODE}"
echo "Dataset Path : ${DATASET_PATH}"
echo "Image Dir    : ${IMAGE_DIR}"
echo "SAM3 Path    : ${SAM3_PATH}"
echo "SigLIP Path  : ${SIGLIP_NAME}"
echo "Config Path  : ${CONFIG_PATH}"
echo "Output Dir   : ${OUTPUT_DIR}"
echo "=========================================================================="

exec "${PYTHON_BIN}" "${OUR_DIR}/eval_runner.py" \
    --model_path "${MODEL_PATH}" \
    --sam3_path "${SAM3_PATH}" \
    --siglip_name "${SIGLIP_NAME}" \
    --nlp_model "${DEFAULT_NLP_MODEL}" \
    --dataset_path "${DATASET_PATH}" \
    --image_dir "${IMAGE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --config "${CONFIG_PATH}" \
    --pool_mode "${POOL_MODE}" \
    "$@"
