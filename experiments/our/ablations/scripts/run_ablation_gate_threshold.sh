#!/usr/bin/env bash
# ==============================================================================
# ABLATION EXPERIMENT: VLM GATING THRESHOLD (0.3 vs 0.5 [default] vs 0.7)
# Fixed Detector: SAM3
# Fixed Scorer: SigLIP (google/siglip-base-patch16-384)
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

FIXED_DETECTOR="sam3"
FIXED_SCORER="siglip"
GATE_THRESHOLDS=("0.3" "0.7" "0.5")

for GATE in "${GATE_THRESHOLDS[@]}"; do
    echo "=========================================================================="
    echo "RUNNING GATE THRESHOLD ABLATION: Gate=${GATE} | Model=${TARGET_MODEL} | Pool=${POOL_MODE}"
    echo "=========================================================================="

    "${PYTHON_BIN}" "${ABLATION_DIR}/eval_runner_ablation.py" \
        --model_path "${TARGET_MODEL}" \
        --detector "${FIXED_DETECTOR}" \
        --detector_path "${SAM3_PATH}" \
        --scorer "${FIXED_SCORER}" \
        --scorer_path "${SIGLIP_NAME}" \
        --gate_threshold "${GATE}" \
        --nlp_model "${NLP_MODEL}" \
        --dataset_path "${DATASET_PATH}" \
        --image_dir "${IMAGE_DIR}" \
        --output_dir "${ABLATION_OUTPUT_DIR}/gate_threshold_ablation" \
        --pool_mode "${POOL_MODE}" \
        "$@"
done
