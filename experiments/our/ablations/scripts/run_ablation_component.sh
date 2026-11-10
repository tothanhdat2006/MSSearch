#!/usr/bin/env bash
# ==============================================================================
# COMPONENT INCREMENT ABLATION SWEEP
# Evaluates cumulative search components:
#   1. Qwen (Direct Baseline)
#   2. Qwen w/ Fast path
#   3. Qwen w/ Fast path + Rigid quadtree + CG + Single-crop
#   4. Qwen w/ Fast path + Rigid quadtree + CG + Aggregation
#   5. Qwen w/ Fast path + SGAP + CG + Single-crop
#   6. Qwen w/ Fast path + SGAP + CG + Aggregation
#   7. Qwen w/ Fast path + ASP + CG + Single-crop
#   8. Qwen w/ Fast path + ASP + CG + Aggregation (Ours)
#
# Usage:
#   bash run_ablation_component.sh [component|all] [model] [pool] [extra_args...]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUR_DIR="$(cd "${ABLATION_DIR}/.." && pwd)"
MSSEARCH_ROOT="$(cd "${OUR_DIR}/../.." && pwd)"

TARGET_COMPONENT="${1:-all}"
TARGET_MODEL="${2:-Qwen/Qwen3-VL-8B-Instruct}"
POOL_MODE="${3:-N_5}"

if [ "$#" -ge 1 ]; then shift 1; fi
if [ "$#" -ge 1 ]; then shift 1; fi
if [ "$#" -ge 1 ]; then shift 1; fi

ALL_COMPONENTS=(
    "qwen"
    "qwen_fast_path"
    "qwen_fp_rigid_cg_single"
    "qwen_fp_rigid_cg_agg"
    "qwen_fp_sgap_cg_single"
    "qwen_fp_sgap_cg_agg"
    "qwen_fp_asp_cg_single"
    "qwen_fp_asp_cg_agg"
)

if [ "${TARGET_COMPONENT}" == "all" ]; then
    COMPONENTS=("${ALL_COMPONENTS[@]}")
else
    COMPONENTS=("${TARGET_COMPONENT}")
fi

cd "${MSSEARCH_ROOT}"

for comp in "${COMPONENTS[@]}"; do
    echo "=========================================================================="
    echo "[Component Sweep] Running: ${comp} | Model: ${TARGET_MODEL} | Pool: ${POOL_MODE}"
    echo "=========================================================================="
    python "${ABLATION_DIR}/eval_runner_ablation.py" \
        --component "${comp}" \
        --model_path "${TARGET_MODEL}" \
        --pool_mode "${POOL_MODE}" \
        "$@"
done

