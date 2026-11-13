#!/usr/bin/env bash
# ==============================================================================
# ABLATION EXPERIMENT: VISUAL SCREENING SCORER
# (SigLIP vs SigLIP2-384 vs SigLIP2-512 vs CLIP-B/32 vs CLIP-L/14 vs EVA-02-L/14)
# Evaluates visual scorers under the default Rigid quadtree + Top-k architecture.
# Detector is bypassed (--detector none) as pure top-k screening relies on the visual scorer.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUR_DIR="$(cd "${ABLATION_DIR}/.." && pwd)"
MSSEARCH_ROOT="$(cd "${OUR_DIR}/../.." && pwd)"

source "${ABLATION_DIR}/env_paths_ablation.sh"

PYTHON_BIN="$(command -v python || echo "python")"
export PYTHONPATH="${MSSEARCH_ROOT}:${OUR_DIR}:${ABLATION_DIR}:${PYTHONPATH:-}"

TARGET_MODEL="${1:-Qwen/Qwen3-VL-8B-Instruct}"
POOL_MODE="${2:-N_5}"
if [ "$#" -ge 2 ]; then
    shift 2
elif [ "$#" -eq 1 ]; then
    shift 1
fi

SCORERS=("siglip" "siglip2_384" "siglip2_512" "clip_b32" "clip_l14" "eva02_l14")

for SC in "${SCORERS[@]}"; do
    echo "=========================================================================="
    echo "RUNNING SCORER ABLATION (Rigid Top-k): Scorer=${SC} | Model=${TARGET_MODEL} | Pool=${POOL_MODE}"
    echo "=========================================================================="

    SC_PATH=""
    if [ "${SC}" == "siglip" ]; then
        SC_PATH="${SIGLIP_NAME}"
    elif [ "${SC}" == "siglip2_384" ]; then
        SC_PATH="${SIGLIP2_384_NAME}"
    elif [ "${SC}" == "siglip2_512" ]; then
        SC_PATH="${SIGLIP2_512_NAME}"
    elif [ "${SC}" == "clip_b32" ]; then
        SC_PATH="${CLIP_B32_NAME}"
    elif [ "${SC}" == "clip_l14" ]; then
        SC_PATH="${CLIP_L14_NAME}"
    elif [ "${SC}" == "eva02_l14" ]; then
        SC_PATH="${EVA02_L14_NAME}"
    fi

    "${PYTHON_BIN}" "${ABLATION_DIR}/eval_runner_ablation.py" \
        --model_path "${TARGET_MODEL}" \
        --detector "none" \
        --scorer "${SC}" \
        --scorer_path "${SC_PATH}" \
        --nlp_model "${NLP_MODEL}" \
        --dataset_path "${DATASET_PATH}" \
        --image_dir "${IMAGE_DIR}" \
        --output_dir "${ABLATION_OUTPUT_DIR}/scorer_ablation" \
        --pool_mode "${POOL_MODE}" \
        "$@"
done
