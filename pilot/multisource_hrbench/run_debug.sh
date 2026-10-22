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


# Default Model and Dataset Paths on Local Server
HRBENCH_DIR="${HRBENCH_DIR:-/home/ubuntu/workspace/hr/data/hr-bench_8k}"
SAM3_PATH="${SAM3_PATH:-/home/ubuntu/workspace/hr/models/sam3/sam3/sam3.pt}"
QWEN_PATH="${QWEN_PATH:-/home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct}"
SIGLIP_NAME="${SIGLIP_NAME:-google/siglip-base-patch16-384}"
NLP_MODEL="${NLP_MODEL:-/home/ubuntu/workspace/hr/models/spacy/en_core_web_sm-3.8.0}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"

echo "=========================================================================="
echo "RUNNING LOCAL HRBENCH MULTI-SOURCE STAGE DIAGNOSTICS"
echo "=========================================================================="
echo "Script Dir   : ${SCRIPT_DIR}"
echo "Conda Env    : ${CONDA_ENV_NAME}"
echo "Python       : ${PYTHON_BIN}"
echo "HRBench Dir  : ${HRBENCH_DIR}"
echo "Output Dir   : ${OUTPUT_DIR}"
echo "SAM3 Path    : ${SAM3_PATH}"
echo "Qwen Path    : ${QWEN_PATH}"
echo "SigLIP       : ${SIGLIP_NAME}"
echo "Forward Args : $*"
echo "=========================================================================="

# Execute Python diagnostic runner, passing default model/data paths and forwarding all CLI flags
"${PYTHON_BIN}" "${SCRIPT_DIR}/debug_stages.py" \
    --hrbench_dir "${HRBENCH_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --sam3_path "${SAM3_PATH}" \
    --qwen_path "${QWEN_PATH}" \
    --siglip_name "${SIGLIP_NAME}" \
    --nlp_model "${NLP_MODEL}" \
    "$@"
