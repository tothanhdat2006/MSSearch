#!/usr/bin/env bash
# ==============================================================================
# ABLATION EXPERIMENT: VISUAL EXPERT / DETECTION BACKBONE
# (SAM3 vs Grounding DINO vs Florence-2 vs OWLv2)
# Evaluates visual experts without Feature Grouping (pure Top-k retention, matching Row 3).
# Supports optional --sgap flag to evaluate on SGAP partitioning.
# Fixed Visual Scorer: SigLIP (google/siglip-base-patch16-384 or env default)
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

PARTITION_MODE="quadrant_only"
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "${arg}" == "--sgap" ]; then
        PARTITION_MODE="sgap_only"
    else
        EXTRA_ARGS+=("${arg}")
    fi
done

DETECTORS=("sam3" "grounding_dino" "florence2" "owlv2")
FIXED_SCORER="siglip"

for DET in "${DETECTORS[@]}"; do
    echo "=========================================================================="
    echo "RUNNING VISUAL EXPERT ABLATION: Detector=${DET} | Partition=${PARTITION_MODE} | Scorer=${FIXED_SCORER} | Pool=${POOL_MODE}"
    echo "=========================================================================="

    DET_PATH=""
    if [ "${DET}" == "sam3" ]; then
        DET_PATH="${SAM3_PATH:-}"
    elif [ "${DET}" == "grounding_dino" ]; then
        DET_PATH="${GROUNDING_DINO_PATH:-IDEA-Research/grounding-dino-base}"
    elif [ "${DET}" == "florence2" ]; then
        DET_PATH="${FLORENCE2_PATH:-florence-community/Florence-2-large}"
    elif [ "${DET}" == "owlv2" ]; then
        DET_PATH="${OWLV2_PATH:-google/owlv2-base-patch16-ensemble}"
    fi

    "${PYTHON_BIN}" "${ABLATION_DIR}/eval_runner_ablation.py" \
        --model_path "${TARGET_MODEL}" \
        --detector "${DET}" \
        --detector_path "${DET_PATH}" \
        --scorer "${FIXED_SCORER}" \
        --scorer_path "${SIGLIP_NAME:-google/siglip-base-patch16-384}" \
        --partition_mode "${PARTITION_MODE}" \
        --nlp_model "${NLP_MODEL:-en_core_web_sm}" \
        --dataset_path "${DATASET_PATH:-/data/STASHBench/annotations_part1234_merged.json}" \
        --image_dir "${IMAGE_DIR:-/data/STASHBench/pool}" \
        --output_dir "${ABLATION_OUTPUT_DIR:-./results_ablation}/detector_ablation" \
        --pool_mode "${POOL_MODE}" \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
done
