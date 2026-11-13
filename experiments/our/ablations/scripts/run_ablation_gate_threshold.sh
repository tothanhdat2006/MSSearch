#!/usr/bin/env bash
# ==============================================================================
# ABLATION EXPERIMENT: CANDIDATE RETENTION TOP-K (Formerly Gate Threshold)
# Note: In the new default architecture (Rigid quadtree + Top-k retention),
# Stage 4 semantic gating and thresholding are replaced by pure Top-k SigLIP ranking.
# This script executes the Top-K retention budget sweep: k in {1, 5, 10, 20, 40}.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[Ablation Note] Gating threshold is superseded by Top-K candidate patch retention in the default architecture."
echo "[Ablation Note] Forwarding to run_ablation_topk.sh..."
exec bash "${SCRIPT_DIR}/run_ablation_topk.sh" "$@"
