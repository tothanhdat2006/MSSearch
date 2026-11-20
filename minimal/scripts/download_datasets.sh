#!/usr/bin/env bash
# ==============================================================================
# STASH-SEARCH MINIMAL: High-Resolution Benchmark Dataset Downloader
# Downloads and extracts V*Bench, HR-Bench (4K/8K), TreeBench, and MME-RealWorld-Lite
# from ModelScope repository (llp1995/hr_data).
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

DEST_DIR="${DATASETS_DIR:-/data}"
[ ! -d "${DEST_DIR}" ] && [ -d "D:/MachineLearning/Datasets" ] && DEST_DIR="D:/MachineLearning/Datasets"

echo "=========================================================================="
echo "STASH-Search Minimal: Dataset Downloader"
echo "Target Base Dir : ${DEST_DIR}"
echo "Script Location : ${MINIMAL_DIR}/datasets/download_data.py"
echo "=========================================================================="

# If user provided arguments, forward them directly; otherwise download all
if [ "$#" -eq 0 ]; then
    exec "${PYTHON_BIN}" "${MINIMAL_DIR}/datasets/download_data.py" \
        --datasets "all" \
        --output_dir "${DEST_DIR}"
else
    exec "${PYTHON_BIN}" "${MINIMAL_DIR}/datasets/download_data.py" \
        --output_dir "${DEST_DIR}" \
        "$@"
fi
