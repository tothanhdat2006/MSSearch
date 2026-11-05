#!/usr/bin/env bash
# ==============================================================================
# MSSEARCH MODAL ABLATION RUNNER
# Dispatches ablation experiments on Modal serverless GPU (L40S / A100).
#
# Usage:
#   bash run_modal_ablation.sh detector [model] [pool] [--detach]
#   bash run_modal_ablation.sh scorer   [model] [pool] [--detach]
#   bash run_modal_ablation.sh gate     [model] [pool] [--detach]
#   bash run_modal_ablation.sh all      [model] [pool] [--detach]
#   bash run_modal_ablation.sh single   --detector <det> --scorer <sc> [--gate-threshold <p>] [--detach]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUR_DIR="$(cd "${ABLATION_DIR}/.." && pwd)"
MSSEARCH_ROOT="$(cd "${OUR_DIR}/../.." && pwd)"

MODE="${1:-single}"
if [ "$#" -ge 1 ]; then shift 1; fi

DETACH_FLAG=""
ARGS=()
for arg in "$@"; do
    if [ "${arg}" == "--detach" ]; then
        DETACH_FLAG="--detach"
    else
        ARGS+=("${arg}")
    fi
done

cd "${MSSEARCH_ROOT}"
MODAL_SCRIPT="${SCRIPT_DIR}/run_ablation_modal.py"

run_job() {
    echo "=========================================================================="
    echo "[Modal Ablation] Launching: modal run ${DETACH_FLAG} ${MODAL_SCRIPT} $*"
    echo "=========================================================================="
    modal run ${DETACH_FLAG} "${MODAL_SCRIPT}" "$@"
}

case "${MODE}" in
    detector|detectors)
        MODEL="${ARGS[0]:-Qwen/Qwen2.5-VL-7B-Instruct}"
        POOL="${ARGS[1]:-N_5}"
        DETECTORS=("sam3" "grounding_dino" "florence2" "dinov3")
        for det in "${DETECTORS[@]}"; do
            run_job --detector "${det}" --scorer "siglip" --model "${MODEL}" --pool-mode "${POOL}"
        done
        ;;

    scorer|scorers)
        MODEL="${ARGS[0]:-Qwen/Qwen2.5-VL-7B-Instruct}"
        POOL="${ARGS[1]:-N_5}"
        SCORERS=("siglip" "siglip2_384" "siglip2_512" "clip_b32" "clip_l14" "eva02_l14")
        for sc in "${SCORERS[@]}"; do
            run_job --detector "sam3" --scorer "${sc}" --model "${MODEL}" --pool-mode "${POOL}"
        done
        ;;

    gate|gate_threshold|threshold)
        MODEL="${ARGS[0]:-Qwen/Qwen2.5-VL-7B-Instruct}"
        POOL="${ARGS[1]:-N_5}"
        THRESHOLDS=("0.3" "0.7" "0.5")
        for th in "${THRESHOLDS[@]}"; do
            run_job --detector "sam3" --scorer "siglip" --gate-threshold "${th}" --model "${MODEL}" --pool-mode "${POOL}"
        done
        ;;

    all|matrix)
        MODEL="${ARGS[0]:-Qwen/Qwen2.5-VL-7B-Instruct}"
        POOL="${ARGS[1]:-N_5}"
        echo "[Modal Ablation] Launching full ablation matrix on Modal..."
        bash "${BASH_SOURCE[0]}" detector "${MODEL}" "${POOL}" ${DETACH_FLAG}
        bash "${BASH_SOURCE[0]}" scorer "${MODEL}" "${POOL}" ${DETACH_FLAG}
        bash "${BASH_SOURCE[0]}" gate "${MODEL}" "${POOL}" ${DETACH_FLAG}
        ;;

    single|custom|*)
        if [ "${MODE}" != "single" ] && [ "${MODE}" != "custom" ]; then
            run_job "${MODE}" "${ARGS[@]}"
        else
            run_job "${ARGS[@]}"
        fi
        ;;
esac
