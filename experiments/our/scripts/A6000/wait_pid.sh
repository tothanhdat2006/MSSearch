#!/usr/bin/env bash
# ==============================================================================
# wait_pid.sh: Wait for a given PID to terminate, then execute a specified command.
#
# Example usage:
#   # Wait for PID 12345 to finish, then run Python evaluation:
#   bash wait_pid.sh 12345 python experiments/our/eval_runner.py ...
#
#   # Chain in background with nohup:
#   nohup bash wait_pid.sh 12345 bash experiments/our/scripts/A6000/run_our_qwen3_vl_8b_n5.sh > run_n5.log 2>&1 &
# ==============================================================================
set -euo pipefail

PID="${1:-}"
shift || true

if [[ -z "${PID}" || $# -eq 0 ]]; then
    echo "Usage: bash wait_pid.sh <PID> <command> [args...]"
    exit 1
fi

echo "[$(date)] Monitoring PID ${PID}..."

while kill -0 "${PID}" 2>/dev/null; do
    sleep 2
done

echo "[$(date)] PID ${PID} is no longer running."
echo "[$(date)] Launching command: $*"

exec "$@"
