#!/usr/bin/env bash
# ==============================================================================
# ABLATION EXPERIMENT: W/O SOURCE-ATTRIBUTED CONSENSUS (paper Sec. 4.4)
# Same pipeline as the top-k ablation at k=20, but Stage 6 per-crop reading + answer voting is replaced
# by ONE multi-image MLLM call over all retained Stage 5 crops (--disable_consensus).
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

TOP_KS=(20)
FIXED_SCORER="siglip"

for K in "${TOP_KS[@]}"; do
    echo "=========================================================================="
    echo "RUNNING NO-CONSENSUS ABLATION (Sec. 4.4 off): Top-K=${K} | Scorer=${FIXED_SCORER} | Model=${TARGET_MODEL} | Pool=${POOL_MODE}"
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
        --output_dir "${ABLATION_OUTPUT_DIR}/noconsensus_ablation" \
        --disable_consensus \
        --pool_mode "${POOL_MODE}" \
        "$@"
done
