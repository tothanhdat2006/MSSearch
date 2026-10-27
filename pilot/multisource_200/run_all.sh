#!/usr/bin/env bash
set -euo pipefail

# Directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSSEARCH_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate Conda Environment: hrsearch
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

# Ensure conda env bin is at the front of PATH
if [ -d "/home/ubuntu/miniconda3/envs/${CONDA_ENV_NAME}/bin" ]; then
    export PATH="/home/ubuntu/miniconda3/envs/${CONDA_ENV_NAME}/bin:${PATH}"
fi

# Python interpreter: prefer hrsearch environment
PYTHON_BIN="/home/ubuntu/miniconda3/envs/${CONDA_ENV_NAME}/bin/python"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python || echo "python")"
fi

# Ensure MSSearch root is in PYTHONPATH
export PYTHONPATH="${MSSEARCH_DIR}:${PYTHONPATH:-}"

# Enforce strict offline execution (local weights only, no online downloads)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Attention Backend: sdpa for NVIDIA Turing (T4) / Ampere
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

# Default Model and Dataset Paths on Local GPU Server
DEFAULT_MODEL_PATH="/home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct"
DEFAULT_SAM3_PATH="/home/ubuntu/workspace/hr/models/sam3/sam3/sam3.pt"
DEFAULT_SIGLIP_NAME="/home/ubuntu/workspace/hr/models/siglip-base-patch16-384"
DEFAULT_NLP_MODEL="/home/ubuntu/workspace/hr/models/spacy/en_core_web_sm-3.8.0"
DEFAULT_DATASET_PATH="/home/ubuntu/workspace/phong/multi_image_visual_search/data/output/200_samples.json"
DEFAULT_IMAGE_DIR="/home/ubuntu/workspace/phong/multi_image_visual_search/data/pilot_5000_images"

MODEL_PATH="${MODEL_PATH:-${QWEN_PATH:-${DEFAULT_MODEL_PATH}}}"
SAM3_PATH="${SAM3_PATH:-${DEFAULT_SAM3_PATH}}"
SIGLIP_NAME="${SIGLIP_NAME:-${DEFAULT_SIGLIP_NAME}}"
NLP_MODEL="${NLP_MODEL:-${DEFAULT_NLP_MODEL}}"
DATASET_PATH="${DATASET_PATH:-${DEFAULT_DATASET_PATH}}"
IMAGE_DIR="${IMAGE_DIR:-${DEFAULT_IMAGE_DIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
POOL_MODE="${POOL_MODE:-N_5}"

# Parse CLI arguments
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path|--model-path|--model|--qwen_path|--qwen-path|-m)
            if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                MODEL_PATH="$2"
                shift 2
            else
                echo "[ERROR] $1 requires a model path argument." >&2
                exit 1
            fi
            ;;
        --output_dir|--output-dir|-o)
            if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                OUTPUT_DIR="$2"
                shift 2
            else
                echo "[ERROR] $1 requires an output directory argument." >&2
                exit 1
            fi
            ;;
        --dataset_path|--dataset-path|--dataset)
            if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                DATASET_PATH="$2"
                shift 2
            else
                echo "[ERROR] $1 requires a dataset path argument." >&2
                exit 1
            fi
            ;;
        --image_dir|--image-dir)
            if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                IMAGE_DIR="$2"
                shift 2
            else
                echo "[ERROR] $1 requires an image directory argument." >&2
                exit 1
            fi
            ;;
        --pool_mode|--pool-mode|--pool)
            if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                POOL_MODE="$2"
                shift 2
            else
                echo "[ERROR] $1 requires a pool mode argument (N_5, N_10, N_20, ALL)." >&2
                exit 1
            fi
            ;;
        --sam3_path|--sam3-path)
            if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                SAM3_PATH="$2"
                shift 2
            else
                echo "[ERROR] $1 requires a SAM3 checkpoint path argument." >&2
                exit 1
            fi
            ;;
        --siglip_name|--siglip-name|--siglip_path|--siglip-path)
            if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                SIGLIP_NAME="$2"
                shift 2
            else
                echo "[ERROR] $1 requires a SigLIP model name/path argument." >&2
                exit 1
            fi
            ;;
        --nlp_model|--nlp-model)
            if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                NLP_MODEL="$2"
                shift 2
            else
                echo "[ERROR] $1 requires an NLP model path argument." >&2
                exit 1
            fi
            ;;
        --resume)
            FORWARD_ARGS+=("--resume")
            shift
            ;;
        --resume_from|--resume-from)
            if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                FORWARD_ARGS+=("--resume_from" "$2")
                shift 2
            else
                echo "[ERROR] $1 requires a path or sample index argument." >&2
                exit 1
            fi
            ;;
        -h|--help)
            echo "=========================================================================="
            echo "MSSEARCH 200-SAMPLE MULTI-SOURCE PIPELINE RUNNER (OFFLINE LOCAL INFERENCE)"
            echo "=========================================================================="
            echo "Usage: $0 [LOCAL_MODEL_PATH] [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -m, --model_path PATH      Local directory containing VLM checkpoint weights"
            echo "  -o, --output_dir DIR       Directory to save results"
            echo "      --dataset_path PATH    Path to 200_samples.json"
            echo "      --image_dir DIR        Server directory containing images"
            echo "      --pool_mode MODE       Candidate pool: N_5 (default), N_10, N_20, ALL"
            echo "      --sam3_path PATH       Local path to SAM3 checkpoint weights (.pt)"
            echo "      --siglip_name NAME     SigLIP model identifier or local path"
            echo "      --nlp_model PATH       Local path or name for spaCy model"
            echo "      --sample_idx IDX       Evaluate single sample index (0-199)"
            echo "      --start_idx IDX        Start index for dataset slice"
            echo "      --end_idx IDX          End index for dataset slice"
            echo "      --max_samples N        Limit execution to N samples"
            echo "      --resume               Resume evaluation from existing results file"
            echo "      --resume_from PATH     Resume evaluation from specific JSONL file"
            echo "  -h, --help                 Show this help message"
            echo "=========================================================================="
            exit 0
            ;;
        *)
            if [[ ${#FORWARD_ARGS[@]} -eq 0 && ! "$1" =~ ^- ]]; then
                MODEL_PATH="$1"
                shift
            else
                FORWARD_ARGS+=("$1")
                shift
            fi
            ;;
    esac
done

# Ensure RESUME_FROM environment variable is passed if set
if [[ -n "${RESUME_FROM:-}" && ! " ${FORWARD_ARGS[*]:-} " =~ " --resume_from " ]]; then
    FORWARD_ARGS+=("--resume_from" "${RESUME_FROM}")
fi

# Validate local model path existence (strict offline check)
if [ ! -e "${MODEL_PATH}" ]; then
    echo "=========================================================================="
    echo "[ERROR] Local model path does not exist: '${MODEL_PATH}'"
    echo "Online downloads are strictly disabled (offline mode active)."
    echo "Please specify an existing local checkpoint directory. Example:"
    echo "  $0 /home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct"
    echo "=========================================================================="
    exit 1
fi

# Determine default output directory based on model name
if [ -z "${OUTPUT_DIR}" ]; then
    MODEL_NAME="$(basename "${MODEL_PATH}")"
    OUTPUT_DIR="${SCRIPT_DIR}/results_mssearch_${MODEL_NAME}"
fi
mkdir -p "${OUTPUT_DIR}"

export MODEL_PATH="${MODEL_PATH}"
export QWEN_PATH="${MODEL_PATH}"

echo "=========================================================================="
echo "LAUNCHING MSSEARCH 200-SAMPLE EVALUATION RUN (MODE: E2E PIPELINE)"
echo "=========================================================================="
echo "Script Dir   : ${SCRIPT_DIR}"
echo "Conda Env    : ${CONDA_ENV_NAME}"
echo "Python       : ${PYTHON_BIN}"
echo "Dataset Path : ${DATASET_PATH}"
echo "Image Dir    : ${IMAGE_DIR}"
echo "Pool Mode    : ${POOL_MODE}"
echo "Output Dir   : ${OUTPUT_DIR}"
echo "Model Path   : ${MODEL_PATH} (Local Only)"
echo "SAM3 Path    : ${SAM3_PATH}"
echo "SigLIP       : ${SIGLIP_NAME}"
echo "NLP Model    : ${NLP_MODEL}"
echo "Offline Mode : Active (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)"
echo "Forward Args : ${FORWARD_ARGS[*]:-none}"
echo "=========================================================================="

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/eval_200.py" \
    --mode e2e \
    --pool_mode "${POOL_MODE}" \
    --dataset_path "${DATASET_PATH}" \
    --image_dir "${IMAGE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --sam3_path "${SAM3_PATH}" \
    --siglip_name "${SIGLIP_NAME}" \
    --nlp_model "${NLP_MODEL}" \
    "${FORWARD_ARGS[@]}"
