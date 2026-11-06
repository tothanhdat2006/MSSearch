#!/usr/bin/env bash
# ==============================================================================
# ABLATION EXPERIMENT: DETECTION BACKBONE
# (SAM3 vs Grounding DINO vs Florence-2 vs DINOv3)
# Fixed Visual Scorer: SigLIP (google/siglip-base-patch16-384)
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

DETECTORS=("sam3" "grounding_dino" "florence2" "dinov3")
FIXED_SCORER="siglip"

for DET in "${DETECTORS[@]}"; do
    echo "=========================================================================="
    echo "RUNNING DETECTION ABLATION: Detector=${DET} | Scorer=${FIXED_SCORER} | Pool=${POOL_MODE}"
    echo "=========================================================================="

    DET_PATH=""
    if [ "${DET}" == "sam3" ]; then
        DET_PATH="${SAM3_PATH}"
    elif [ "${DET}" == "grounding_dino" ]; then
        DET_PATH="${GROUNDING_DINO_PATH}"
    elif [ "${DET}" == "florence2" ]; then
        DET_PATH="${FLORENCE2_PATH}"
    elif [ "${DET}" == "dinov3" ]; then
        DET_PATH="${DINOV3_PATH}"
    fi

    "${PYTHON_BIN}" "${ABLATION_DIR}/eval_runner_ablation.py" \
        --model_path "${TARGET_MODEL}" \
        --detector "${DET}" \
        --detector_path "${DET_PATH}" \
        --scorer "${FIXED_SCORER}" \
        --scorer_path "${SIGLIP_NAME}" \
        --nlp_model "${NLP_MODEL}" \
        --dataset_path "${DATASET_PATH}" \
        --image_dir "${IMAGE_DIR}" \
        --output_dir "${ABLATION_OUTPUT_DIR}/detector_ablation" \
        --pool_mode "${POOL_MODE}" \
        "$@"
done
