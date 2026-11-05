#!/usr/bin/env bash
# ==============================================================================
# ABLATION EXPERIMENT: VISUAL SCREENING SCORER (SigLIP vs CLIP-B/32 vs CLIP-L/14 vs EVA-02-L/14)
# Fixed Detector: SAM3
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUR_DIR="$(cd "${ABLATION_DIR}/.." && pwd)"
MSSEARCH_ROOT="$(cd "${OUR_DIR}/../.." && pwd)"

source "${ABLATION_DIR}/env_paths_ablation.sh"

PYTHON_BIN="$(command -v python || echo "python")"
export PYTHONPATH="${MSSEARCH_ROOT}:${OUR_DIR}:${ABLATION_DIR}:${PYTHONPATH:-}"

TARGET_MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
POOL_MODE="${2:-N_5}"
shift 2 || true

FIXED_DETECTOR="sam3"
SCORERS=("siglip" "clip_b32" "clip_l14" "eva02_l14")

for SC in "${SCORERS[@]}"; do
    echo "=========================================================================="
    echo "RUNNING SCORER ABLATION: Scorer=${SC} | Detector=${FIXED_DETECTOR} | Pool=${POOL_MODE}"
    echo "=========================================================================="

    SC_PATH=""
    if [ "${SC}" == "siglip" ]; then
        SC_PATH="${SIGLIP_NAME}"
    elif [ "${SC}" == "clip_b32" ]; then
        SC_PATH="${CLIP_B32_NAME}"
    elif [ "${SC}" == "clip_l14" ]; then
        SC_PATH="${CLIP_L14_NAME}"
    elif [ "${SC}" == "eva02_l14" ]; then
        SC_PATH="${EVA02_L14_NAME}"
    fi

    "${PYTHON_BIN}" "${ABLATION_DIR}/eval_runner_ablation.py" \
        --model_path "${TARGET_MODEL}" \
        --detector "${FIXED_DETECTOR}" \
        --detector_path "${SAM3_PATH}" \
        --scorer "${SC}" \
        --scorer_path "${SC_PATH}" \
        --nlp_model "${NLP_MODEL}" \
        --dataset_path "${DATASET_PATH}" \
        --image_dir "${IMAGE_DIR}" \
        --output_dir "${ABLATION_OUTPUT_DIR}/scorer_ablation" \
        --pool_mode "${POOL_MODE}" \
        "$@"
done
