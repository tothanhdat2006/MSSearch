#!/usr/bin/env bash
# ==============================================================================
# STASH-SEARCH MINIMAL: Batch Runner for all 4 models on STASH-Bench N=5
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/4] Running Qwen3-VL-8B (N=5) ==="
bash "${SCRIPT_DIR}/run_stash_qwen3_vl_8b_n5.sh" "$@"

echo "=== [2/4] Running LLaVA-OneVision-7B (N=5) ==="
bash "${SCRIPT_DIR}/run_stash_llava_ov_7b_n5.sh" "$@"

echo "=== [3/4] Running InternVL3-8B (N=5) ==="
bash "${SCRIPT_DIR}/run_stash_internvl3_8b_n5.sh" "$@"

echo "=== [4/4] Running Qwen3-VL-32B (N=5) ==="
bash "${SCRIPT_DIR}/run_stash_qwen3_vl_32b_n5.sh" "$@"

echo "=== [ALL MODELS COMPLETE] ==="
