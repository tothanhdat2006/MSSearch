#!/usr/bin/env bash
# Master submission helper for HPC
# Submits individual jobs or sweeps to Slurm queue

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for pool in n1 n5 n10 n20; do
    echo "Submitting Slurm job for pool ${pool}..."
    sbatch "${SCRIPT_DIR}/run_our_all_${pool}.slurm" "$@"
done

echo "All pool jobs submitted."
