#!/usr/bin/env bash
# ==============================================================================
# ABLATION EXPERIMENT: DETECTION BACKBONE (SAM3 vs Grounding DINO vs DINO-X)
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

TARGET_MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
POOL_MODE="${2:-N_5}"
shift 2 || true

DETECTORS=("sam3" "grounding_dino" "dinox")
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
    elif [ "${DET}" == "dinox" ]; then
        DET_PATH="${DINOX_PATH}"
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
