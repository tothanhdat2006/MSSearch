#!/usr/bin/env bash
#SBATCH --job-name=our_llava_ov_7b_n10
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MSSEARCH_ROOT="$(cd "${OUR_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${MSSEARCH_ROOT}/../.." && pwd)"

cd "${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
mkdir -p logs

# Environment activation
VENV_PATH="${VENV_PATH:-/data/shared/cvpr/dat/.venv/bin/activate}"
if [ -f "${VENV_PATH}" ]; then
    source "${VENV_PATH}"
fi

PYTHON_BIN="$(command -v python || echo "python")"
export PYTHONPATH="${MSSEARCH_ROOT}:${OUR_DIR}:${PYTHONPATH:-}"

# Offline execution
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

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

DATASET_PATH="${DATASET_PATH:-}"
for cand in \
    "${DATASET_PATH}" \
    "/data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json" \
    "${MSSEARCH_ROOT}/data/STASHBench/annotations_part1234_merged.json" \
    "${REPO_ROOT}/data/STASHBench/annotations_part1234_merged.json"; do
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

POOL_MODE="N_10"
CONFIG_PATH="${OUR_DIR}/config_default.json"
OUTPUT_DIR="${OUTPUT_DIR:-${OUR_DIR}/results_our_a100_${M_BASE}_${POOL_MODE}}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================================================="
echo "STARTING A100 MSSEARCH V2 RUN: llava-onevision-qwen2-7b-ov (${POOL_MODE})"
echo "Model Path : ${MODEL_PATH}"
echo "Pool Mode  : ${POOL_MODE}"
echo "Config     : ${CONFIG_PATH}"
echo "Output Dir : ${OUTPUT_DIR}"
echo "=========================================================================="

"${PYTHON_BIN}" -u "${OUR_DIR}/eval_runner.py" \
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
