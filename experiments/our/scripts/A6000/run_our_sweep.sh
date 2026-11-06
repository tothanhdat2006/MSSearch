#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MSSEARCH V2 EVALUATION RUNNER ON LOCAL A6000: FULL MATRIX SWEEP
# Executes N=1, N=5, N=10, N=20 across Qwen3, LLaVA-OV, InternVL3
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for pool in n1 n5 n10 n20; do
    echo "=========================================================================="
    echo "EXECUTING A6000 POOL SWEEP: ${pool}"
    echo "=========================================================================="
    bash "${SCRIPT_DIR}/run_our_all_${pool}.sh" "$@"
done

echo "Master A6000 sweep completed successfully."
