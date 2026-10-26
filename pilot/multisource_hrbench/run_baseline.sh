#!/usr/bin/env bash
set -euo pipefail

# Directory paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSSEARCH_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
WORKSPACE_ROOT="${PROJECT_ROOT}"

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

# Enforce strict offline execution (local files only, no online downloads)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Default Model and Dataset Paths on Local Server
DEFAULT_MODEL_PATH="/home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct"
DEFAULT_SIGLIP_PATH="/home/ubuntu/workspace/hr/models/siglip-base-patch16-384"
export SIGLIP_MODEL_DIR="${SIGLIP_MODEL_DIR:-${DEFAULT_SIGLIP_PATH}}"
MODEL_PATH="${MODEL_PATH:-${QWEN_PATH:-${DEFAULT_MODEL_PATH}}}"
HRBENCH_DIR="${HRBENCH_DIR:-/home/ubuntu/workspace/hr/data/hr-bench_8k}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

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
        --dataset_dir|--dataset-dir|--data_dir|--data-dir|--hrbench_dir|--hrbench-dir)
            if [[ -n "${2:-}" && ! "$2" =~ ^-- ]]; then
                HRBENCH_DIR="$2"
                shift 2
            else
                echo "[ERROR] $1 requires a dataset directory argument." >&2
                exit 1
            fi
            ;;
        --resume)
            FORWARD_ARGS+=("--resume")
            shift
            ;;
        -h|--help)
            echo "=========================================================================="
            echo "HRBENCH DIRECT VLM BASELINE RUNNER (OFFLINE LOCAL INFERENCE)"
            echo "=========================================================================="
            echo "Usage: $0 [LOCAL_MODEL_PATH] [OPTIONS]"
            echo ""
            echo "Model Selection (Local Only, No Online Downloads):"
            echo "  $0 /path/to/local/model"
            echo "  $0 --model_path /path/to/local/model"
            echo "  MODEL_PATH=/path/to/local/model $0"
            echo ""
            echo "Options:"
            echo "  -m, --model_path PATH      Local directory containing model checkpoint weights"
            echo "  -o, --output_dir DIR       Directory to save results (defaults to results_baseline_<model_name>)"
            echo "      --hrbench_dir DIR      Root path to HRBench dataset"
            echo "      --direct_target_only   Evaluate strictly on target image (N=1, ignoring distractors)"
            echo "      --all_samples          Evaluate all samples in HRBench"
            echo "      --sample_idx IDX       Evaluate single sample index"
            echo "      --resume               Resume evaluation from existing results file"
            echo "  -h, --help                 Show this help message"
            echo "=========================================================================="
            exit 0
            ;;
        *)
            # If first argument does not start with -, treat as positional model path
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

# Validate local model path existence (strict offline check)
if [ ! -e "${MODEL_PATH}" ]; then
    echo "=========================================================================="
    echo "[ERROR] Local model path does not exist: '${MODEL_PATH}'"
    echo "Online downloads are strictly disabled (offline mode active)."
    echo ""
    echo "Please specify an existing local checkpoint directory. Example:"
    echo "  $0 /home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct"
    echo "  $0 --model_path /home/ubuntu/workspace/hr/models/Qwen2.5-VL-3B-Instruct"
    echo "  MODEL_PATH=/path/to/local/model $0"
    echo "=========================================================================="
    exit 1
fi

# Determine default output directory based on model name to prevent result collisions
if [ -z "${OUTPUT_DIR}" ]; then
    if [ "${MODEL_PATH}" = "${DEFAULT_MODEL_PATH}" ]; then
        OUTPUT_DIR="${SCRIPT_DIR}/results_baseline"
    else
        MODEL_NAME="$(basename "${MODEL_PATH}")"
        OUTPUT_DIR="${SCRIPT_DIR}/results_baseline_${MODEL_NAME}"
    fi
fi
mkdir -p "${OUTPUT_DIR}"

export MODEL_PATH="${MODEL_PATH}"
export QWEN_PATH="${MODEL_PATH}"

echo "=========================================================================="
echo "LAUNCHING HRBENCH DIRECT VLM BASELINE EVALUATION (VANILLA VLM)"
echo "=========================================================================="
echo "Script Dir   : ${SCRIPT_DIR}"
echo "Conda Env    : ${CONDA_ENV_NAME}"
echo "Python       : ${PYTHON_BIN}"
echo "HRBench Dir  : ${HRBENCH_DIR}"
echo "Output Dir   : ${OUTPUT_DIR}"
echo "Model Path   : ${MODEL_PATH} (Local Only)"
echo "Offline Mode : Active (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)"
echo "Forward Args : ${FORWARD_ARGS[*]:-none}"
echo "=========================================================================="

# Run in direct baseline mode
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/debug_stages.py" \
    --mode direct_answer \
    --all_samples \
    --dataset_dir "${HRBENCH_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    ${FORWARD_ARGS[@]+"${FORWARD_ARGS[@]}"}
