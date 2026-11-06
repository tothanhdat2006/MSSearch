#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MSSEARCH V2 EVALUATION RUNNER ON LOCAL A6000: ALL 3 MODELS (N_10)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POOL_SLUG="n10"

echo "Starting evaluation run for pool N_10 across all 3 models..."

echo "[1/3] Running Qwen3-VL-8B-Instruct..."
bash "${SCRIPT_DIR}/run_our_qwen2.5_vl_7b_${POOL_SLUG}.sh" "$@"

echo "[2/3] Running llava-onevision-qwen2-7b-ov..."
bash "${SCRIPT_DIR}/run_our_llava_ov_7b_${POOL_SLUG}.sh" "$@"

echo "[3/3] Running InternVL3-8B..."
bash "${SCRIPT_DIR}/run_our_internvl3_8b_${POOL_SLUG}.sh" "$@"

echo "All 3 models finished for N_10."
