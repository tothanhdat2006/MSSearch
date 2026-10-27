#!/usr/bin/env bash
#SBATCH --job-name=mssearch_200
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=36G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Ensure execution from the submitted directory
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Slurm Environment Telemetry
echo "=========================================================================="
echo "SLURM JOB EXECUTION TELEMETRY"
echo "=========================================================================="
echo "Job ID           : ${SLURM_JOB_ID:-local}"
echo "Job Name         : ${SLURM_JOB_NAME:-unknown}"
echo "Host             : $(hostname)"
echo "Submit Dir       : ${SLURM_SUBMIT_DIR}"
echo "CPUs Per Task    : ${SLURM_CPUS_PER_TASK:-unknown}"
echo "CUDA Devices     : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Timestamp        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================================="
nvidia-smi -L || true
echo "=========================================================================="

# Activate required virtual environment (Strict: Fail immediately if missing)
VENV_PATH="/data/shared/cvpr/dat/.venv/bin/activate"
if [ ! -f "${VENV_PATH}" ]; then
    echo "[FATAL ERROR] Virtual environment not found at: '${VENV_PATH}'" >&2
    echo "Immediate failure enforced. No fallback allowed." >&2
    exit 1
fi
echo "[Env Setup] Activating virtual environment: ${VENV_PATH}"
source "${VENV_PATH}"

# Python and repository paths
PYTHON_BIN="$(command -v python || echo "python")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSSEARCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${MSSEARCH_ROOT}:${PYTHONPATH:-}"

# Offline enforcement
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

# Base server directories
BASE_MODEL_DIR="/data/shared/cvpr/dat/models"
DATA_ROOT="${DATA_ROOT:-/home/cvpr2027/dat/simple/data}"

# Strict Dataset and Image Paths (Fail immediately if missing)
DATASET_PATH="${DATASET_PATH:-${DATA_ROOT}/200_samples.json}"
if [ ! -f "${DATASET_PATH}" ]; then
    echo "[FATAL ERROR] Dataset path does not exist: '${DATASET_PATH}'" >&2
    echo "Immediate failure enforced. No fallback allowed." >&2
    exit 1
fi

IMAGE_DIR="${IMAGE_DIR:-${DATA_ROOT}/images}"
if [ ! -d "${IMAGE_DIR}" ]; then
    echo "[FATAL ERROR] Image directory does not exist: '${IMAGE_DIR}'" >&2
    echo "Immediate failure enforced. No fallback allowed." >&2
    exit 1
fi

# Strict Shared Component Paths (Fail immediately if missing)
SAM3_PATH="${SAM3_PATH:-${BASE_MODEL_DIR}/sam3/sam3.pt}"
if [ ! -f "${SAM3_PATH}" ]; then
    echo "[FATAL ERROR] SAM3 weights do not exist: '${SAM3_PATH}'" >&2
    echo "Immediate failure enforced. No fallback allowed." >&2
    exit 1
fi

SIGLIP_NAME="${SIGLIP_NAME:-${BASE_MODEL_DIR}/siglip-base-patch16-384}"
if [ ! -d "${SIGLIP_NAME}" ]; then
    echo "[FATAL ERROR] SigLIP model directory does not exist: '${SIGLIP_NAME}'" >&2
    echo "Immediate failure enforced. No fallback allowed." >&2
    exit 1
fi

NLP_MODEL="${NLP_MODEL:-${BASE_MODEL_DIR}/en_core_web_sm-3.8.0}"
if [ ! -d "${NLP_MODEL}" ]; then
    echo "[FATAL ERROR] NLP model directory does not exist: '${NLP_MODEL}'" >&2
    echo "Immediate failure enforced. No fallback allowed." >&2
    exit 1
fi

POOL_MODE="${POOL_MODE:-N_5}"

# Parse models and extra forward arguments
# Non-option arguments are treated as exact model folder names
# Supported exact names examples:
#   Qwen2.5-VL-7B-Instruct, llava-onevision-qwen2-7b-ov, Qwen3-VL-8B,
#   LLaVA-v1.6-7B, LLaVA-v1.6-13B, LLaVA-v1.6-34B, LLaVA-HR-X-7B, LLaVA-HR-X-13B,
#   Qwen2.5-VL-32B, InternVL3-38B, Yi-VL-34B
CLI_MODELS=()
FORWARD_ARGS=()

for arg in "$@"; do
    if [[ "$arg" == -* ]]; then
        FORWARD_ARGS+=("$arg")
    else
        CLI_MODELS+=("$arg")
    fi
done

TARGET_MODELS=()
if [ ${#CLI_MODELS[@]} -gt 0 ]; then
    TARGET_MODELS=("${CLI_MODELS[@]}")
elif [ -n "${MODELS:-}" ]; then
    read -r -a TARGET_MODELS <<< "${MODELS}"
elif [ -n "${MODEL_NAME:-}" ]; then
    TARGET_MODELS=("${MODEL_NAME}")
else
    # Default benchmark pair
    TARGET_MODELS=("Qwen2.5-VL-7B-Instruct" "llava-onevision-qwen2-7b-ov")
fi

# Strict Model Path Validation: Check all models exist upfront
RESOLVED_MODEL_PATHS=()
for M_ITEM in "${TARGET_MODELS[@]}"; do
    if [[ "${M_ITEM}" = /* ]]; then
        M_FULL_PATH="${M_ITEM}"
    else
        M_FULL_PATH="${BASE_MODEL_DIR}/${M_ITEM}"
    fi

    if [ ! -d "${M_FULL_PATH}" ]; then
        echo "[FATAL ERROR] Exact model directory does not exist: '${M_FULL_PATH}'" >&2
        echo "Exact match required for model '${M_ITEM}'. Immediate failure enforced." >&2
        exit 1
    fi
    RESOLVED_MODEL_PATHS+=("${M_FULL_PATH}")
done

echo "=========================================================================="
echo "EVALUATION PIPELINE SPECIFICATION"
echo "=========================================================================="
echo "Dataset Path     : ${DATASET_PATH}"
echo "Image Dir        : ${IMAGE_DIR}"
echo "Pool Mode        : ${POOL_MODE}"
echo "SAM3 Weights     : ${SAM3_PATH}"
echo "SigLIP Model     : ${SIGLIP_NAME}"
echo "NLP Model        : ${NLP_MODEL}"
echo "Target Models (${#RESOLVED_MODEL_PATHS[@]}):"
for P in "${RESOLVED_MODEL_PATHS[@]}"; do
    echo "  - ${P}"
done
echo "Extra Args       : ${FORWARD_ARGS[*]:-none}"
echo "=========================================================================="

# Execute evaluation sequentially for each validated model
for M_PATH in "${RESOLVED_MODEL_PATHS[@]}"; do
    M_NAME="$(basename "${M_PATH}")"
    OUTPUT_DIR="${SCRIPT_DIR}/results_mssearch_${M_NAME}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "=========================================================================="
    echo "STARTING EXPERIMENT: ${M_NAME}"
    echo "=========================================================================="
    echo "Model Path : ${M_PATH}"
    echo "Output Dir : ${OUTPUT_DIR}"
    echo "=========================================================================="

    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/eval_200.py" \
        --mode e2e \
        --pool_mode "${POOL_MODE}" \
        --dataset_path "${DATASET_PATH}" \
        --image_dir "${IMAGE_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --model_path "${M_PATH}" \
        --sam3_path "${SAM3_PATH}" \
        --siglip_name "${SIGLIP_NAME}" \
        --nlp_model "${NLP_MODEL}" \
        "${FORWARD_ARGS[@]}"

    echo "Completed experiment for: ${M_NAME}"
done

echo ""
echo "=========================================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY AT $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================================="
