#!/usr/bin/env bash
# ==============================================================================
# ABLATION EXPERIMENT: CANDIDATE RETENTION BUDGET TOP-K
# Sweeps base patches per image k in {1, 5, 10, 20, 40} (default: k=20).
# Patches are ranked globally by SigLIP similarity under Rigid quadtree partitioning.
# Evaluates candidate retention pool plasticity across gallery sizes (N).
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

TOP_KS=(1 5 10 20 40)
FIXED_SCORER="siglip"

for K in "${TOP_KS[@]}"; do
    echo "=========================================================================="
    echo "RUNNING TOP-K RETENTION ABLATION: Top-K=${K} | Scorer=${FIXED_SCORER} | Model=${TARGET_MODEL} | Pool=${POOL_MODE}"
    echo "=========================================================================="

    "${PYTHON_BIN}" "${ABLATION_DIR}/eval_runner_ablation.py" \
        --model_path "${TARGET_MODEL}" \
        --detector "none" \
        --scorer "${FIXED_SCORER}" \
        --scorer_path "${SIGLIP_NAME}" \
        --top_k "${K}" \
        --nlp_model "${NLP_MODEL}" \
        --dataset_path "${DATASET_PATH}" \
        --image_dir "${IMAGE_DIR}" \
        --output_dir "${ABLATION_OUTPUT_DIR}/topk_ablation" \
        --pool_mode "${POOL_MODE}" \
        "$@"
done
