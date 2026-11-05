#!/usr/bin/env bash
# ==============================================================================
# FULL ABLATION MATRIX SWEEP: Detectors x Scorers x Pools
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_MODEL="${1:-Qwen/Qwen2.5-VL-7B-Instruct}"
shift || true

# 1. Run Detector Ablations (SAM3 vs Grounding DINO vs DINO-X)
bash "${SCRIPT_DIR}/run_ablation_detector.sh" "${TARGET_MODEL}" "N_5" "$@"

# 2. Run Scorer Ablations (SigLIP vs CLIP-B/32 vs CLIP-L/14 vs EVA-02-L/14)
bash "${SCRIPT_DIR}/run_ablation_scorer.sh" "${TARGET_MODEL}" "N_5" "$@"
