#!/usr/bin/env bash
# ==============================================================================
# MODAL CONVENIENCE RUNNER FOR MSSEARCH V2
# Usage:
#   bash run_modal.sh qwen2.5_vl_7b N_5
#   bash run_modal.sh llava_ov_7b N_10
#   bash run_modal.sh internvl3_8b N_20
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUR_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MSSEARCH_ROOT="$(cd "${OUR_DIR}/../.." && pwd)"

# Source centralized environment and paths
if [ -f "${OUR_DIR}/env_paths.sh" ]; then
    source "${OUR_DIR}/env_paths.sh"
fi

export PYTHONPATH="${MSSEARCH_ROOT}:${OUR_DIR}:${PYTHONPATH:-}"

MODEL_KEY="${1:-qwen2.5_vl_7b}"
POOL_MODE="${2:-N_20}"
shift 2 || true

case "${MODEL_KEY}" in
    qwen2.5_vl_7b|qwen|Qwen2.5-VL-7B-Instruct)
        MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
        ;;
    llava_ov_7b|llava|llava-onevision-qwen2-7b-ov)
        MODEL_NAME="lmms-lab/llava-onevision-qwen2-7b-ov"
        ;;
    internvl3_8b|internvl|InternVL3-8B)
        MODEL_NAME="OpenGVLab/InternVL3-8B"
        ;;
    *)
        MODEL_NAME="${MODEL_KEY}"
        ;;
esac

echo "=========================================================================="
echo "LAUNCHING MODAL EVALUATION: ${MODEL_NAME} (${POOL_MODE})"
echo "=========================================================================="

MODAL_FLAGS=()
REMOTE_ARGS=()
for arg in "$@"; do
    if [ "${arg}" = "--detach" ] || [ "${arg}" = "-d" ]; then
        MODAL_FLAGS+=("--detach")
    else
        REMOTE_ARGS+=("${arg}")
    fi
done

exec modal run ${MODAL_FLAGS[@]+"${MODAL_FLAGS[@]}"} "${SCRIPT_DIR}/run_our_modal.py" \
    --model "${MODEL_NAME}" \
    --pool-mode "${POOL_MODE}" \
    ${REMOTE_ARGS[@]+"${REMOTE_ARGS[@]}"}
