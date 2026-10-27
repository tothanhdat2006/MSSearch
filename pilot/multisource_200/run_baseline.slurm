#!/usr/bin/env bash
#SBATCH --job-name=baseline_200
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=36G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Ensure execution from the submitted directory
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Slurm Environment Telemetry
echo "=========================================================================="
echo "SLURM JOB EXECUTION TELEMETRY (DIRECT VLM BASELINE)"
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

# Activate required virtual environment
VENV_PATH="/data/shared/cvpr/dat/.venv/bin/activate"
if [ -f "${VENV_PATH}" ]; then
    echo "[Env Setup] Activating virtual environment: ${VENV_PATH}"
    source "${VENV_PATH}"
else
    echo "[ERROR] Virtual environment not found at: '${VENV_PATH}'" >&2
    exit 1
fi

# Locate Python binary and repo directories
PYTHON_BIN="$(command -v python || echo "python")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSSEARCH_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${MSSEARCH_ROOT}:${PYTHONPATH:-}"

# Enforce strict offline execution (local checkpoint weights only)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

# Shared Server Paths
BASE_MODEL_DIR="/data/shared/cvpr/dat/models"
DATA_ROOT="${DATA_ROOT:-/home/cvpr2027/dat/simple/data}"

# Resolve Dataset Path (200_samples.json)
if [ -f "${DATA_ROOT}/200_samples.json" ]; then
    DATASET_PATH="${DATA_ROOT}/200_samples.json"
elif [ -f "${DATA_ROOT}/output/200_samples.json" ]; then
    DATASET_PATH="${DATA_ROOT}/output/200_samples.json"
else
    DATASET_PATH="${DATA_ROOT}/200_samples.json"
fi

# Resolve Raw Image Directory
if [ -d "${DATA_ROOT}/images" ]; then
    IMAGE_DIR="${DATA_ROOT}/images"
elif [ -d "${DATA_ROOT}/pilot_5000_images" ]; then
    IMAGE_DIR="${DATA_ROOT}/pilot_5000_images"
else
    IMAGE_DIR="${DATA_ROOT}"
fi

# Benchmark Configuration
POOL_MODE="${POOL_MODE:-N_5}"
FORWARD_ARGS=("$@")

# Define Target Foundation Models
MODELS_TO_EVAL=()
TARGET_MODEL="${TARGET_MODEL:-all}"

case "${TARGET_MODEL,,}" in
    *llava*|*onevision*|*ov*)
        LLAVA_PATH="${BASE_MODEL_DIR}/llava-onevision-qwen2-7b-ov"
        if [ ! -d "${LLAVA_PATH}" ] && [ -d "${BASE_MODEL_DIR}/LLaVA-OV-7B" ]; then
            LLAVA_PATH="${BASE_MODEL_DIR}/LLaVA-OV-7B"
        fi
        MODELS_TO_EVAL+=("llava:${LLAVA_PATH}")
        ;;
    *qwen*)
        MODELS_TO_EVAL+=("qwen:${BASE_MODEL_DIR}/Qwen2.5-VL-7B-Instruct")
        ;;
    all|*)
        LLAVA_PATH="${BASE_MODEL_DIR}/llava-onevision-qwen2-7b-ov"
        if [ ! -d "${LLAVA_PATH}" ] && [ -d "${BASE_MODEL_DIR}/LLaVA-OV-7B" ]; then
            LLAVA_PATH="${BASE_MODEL_DIR}/LLaVA-OV-7B"
        fi
        MODELS_TO_EVAL+=("qwen:${BASE_MODEL_DIR}/Qwen2.5-VL-7B-Instruct")
        MODELS_TO_EVAL+=("llava:${LLAVA_PATH}")
        ;;
esac

echo "=========================================================================="
echo "BASELINE EVALUATION SPECIFICATION"
echo "=========================================================================="
echo "Dataset Path     : ${DATASET_PATH}"
echo "Image Dir        : ${IMAGE_DIR}"
echo "Pool Mode        : ${POOL_MODE}"
echo "Target Models    : ${#MODELS_TO_EVAL[@]} model(s)"
echo "Extra Args       : ${FORWARD_ARGS[*]:-none}"
echo "=========================================================================="

for MODEL_ENTRY in "${MODELS_TO_EVAL[@]}"; do
    M_TAG="${MODEL_ENTRY%%:*}"
    M_PATH="${MODEL_ENTRY#*:}"
    M_NAME="$(basename "${M_PATH}")"
    OUTPUT_DIR="${SCRIPT_DIR}/results_baseline_${M_NAME}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "=========================================================================="
    echo "STARTING BASELINE EXPERIMENT: ${M_NAME} (${M_TAG^^})"
    echo "=========================================================================="
    echo "Model Path : ${M_PATH}"
    echo "Output Dir : ${OUTPUT_DIR}"
    echo "=========================================================================="

    if [ ! -d "${M_PATH}" ]; then
        echo "[WARNING] Model directory does not exist: '${M_PATH}'. Skipping." >&2
        continue
    fi

    "${PYTHON_BIN}" -u "${SCRIPT_DIR}/eval_200.py" \
        --mode direct \
        --pool_mode "${POOL_MODE}" \
        --dataset_path "${DATASET_PATH}" \
        --image_dir "${IMAGE_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --model_path "${M_PATH}" \
        "${FORWARD_ARGS[@]}"

    echo "Completed baseline experiment for: ${M_NAME}"
done

echo ""
echo "=========================================================================="
echo "ALL BASELINE EXPERIMENTS COMPLETED AT $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================================================="
