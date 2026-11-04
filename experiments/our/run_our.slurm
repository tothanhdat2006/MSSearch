#!/usr/bin/env bash
#SBATCH --job-name=our_a100
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

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# ==============================================================================
# EDIT THIS PATH BLOCK FOR YOUR HPC ENVIRONMENT
# ==============================================================================
VENV_PATH="/data/shared/cvpr/dat/.venv/bin/activate"                           # EDIT: Virtual environment
DEFAULT_MODEL_DIR="/data/shared/cvpr/dat/models"                               # EDIT: Base model weights directory
DEFAULT_SAM3_PATH="/data/shared/cvpr/dat/models/sam3/sam3.pt"                  # EDIT: SAM3 checkpoint
DEFAULT_SIGLIP_NAME="/data/shared/cvpr/dat/models/siglip-base-patch16-384"     # EDIT: SigLIP checkpoint
DEFAULT_NLP_MODEL="/data/shared/cvpr/dat/models/spacy/en_core_web_sm-3.8.0"    # EDIT: spaCy pipeline
DEFAULT_DATASET_PATH="/data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json"     # EDIT: Dataset JSON path
DEFAULT_IMAGE_DIR="/data/shared/cvpr/dat/simple/data/STASHBench/pool"                                   # EDIT: Image directory
# ==============================================================================

if [ -f "${VENV_PATH}" ]; then
    source "${VENV_PATH}"
fi

PYTHON_BIN="$(command -v python || echo "python")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSSEARCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${MSSEARCH_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"

POOL_MODE="N_20"  # Default candidate pool size
CONFIG_PATH="${SCRIPT_DIR}/config_default.json"
MODELS=("${@:-Qwen2.5-VL-7B-Instruct}")

for M_ITEM in "${MODELS[@]}"; do
    if [ -d "${DEFAULT_MODEL_DIR}/${M_ITEM}" ]; then
        M_PATH="${DEFAULT_MODEL_DIR}/${M_ITEM}"
    else
        M_PATH="${M_ITEM}"
    fi

    M_NAME="$(basename "${M_PATH}")"
    OUTPUT_DIR="${SCRIPT_DIR}/results_our_${M_NAME}_${POOL_MODE}"
    mkdir -p "${OUTPUT_DIR}"

    echo "=========================================================================="
    echo "STARTING A100 MSSEARCH V2 RUN: ${M_NAME}"
    echo "Model Path : ${M_PATH}"
    echo "Pool Mode  : ${POOL_MODE}"
    echo "Config     : ${CONFIG_PATH}"
    echo "Output Dir : ${OUTPUT_DIR}"
    echo "=========================================================================="

    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/eval_runner.py" \
        --model_path "${M_PATH}" \
        --sam3_path "${DEFAULT_SAM3_PATH}" \
        --siglip_name "${DEFAULT_SIGLIP_NAME}" \
        --nlp_model "${DEFAULT_NLP_MODEL}" \
        --dataset_path "${DEFAULT_DATASET_PATH}" \
        --image_dir "${DEFAULT_IMAGE_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --config "${CONFIG_PATH}" \
        --pool_mode "${POOL_MODE}"
done
