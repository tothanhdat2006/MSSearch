#!/usr/bin/env bash
# ==============================================================================
# FULL ABLATION MATRIX SWEEP: Detectors x Scorers x Pools
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_MODEL="${1:-Qwen/Qwen3-VL-8B-Instruct}"
shift || true

# 1. Run Detector Ablations (SAM3 vs Grounding DINO vs Florence-2)
bash "${SCRIPT_DIR}/run_ablation_detector.sh" "${TARGET_MODEL}" "N_5" "$@"

# 2. Run Scorer Ablations (SigLIP vs SigLIP2 vs CLIP-B/32 vs CLIP-L/14 vs EVA-02-L/14)
bash "${SCRIPT_DIR}/run_ablation_scorer.sh" "${TARGET_MODEL}" "N_5" "$@"

# 3. Run Top-K Retention Budget Ablations (k in {1, 5, 10, 20, 40})
bash "${SCRIPT_DIR}/run_ablation_topk.sh" "${TARGET_MODEL}" "N_5" "$@"
