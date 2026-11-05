#!/usr/bin/env bash
#SBATCH --job-name=our_all_n20
#SBATCH --partition=gpu
#SBATCH --account=lab
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POOL_SLUG="n20"
cd "${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
mkdir -p logs

echo "Executing full model suite for N_20..."
bash "${SCRIPT_DIR}/run_our_qwen2.5_vl_7b_${POOL_SLUG}.slurm" "$@"
bash "${SCRIPT_DIR}/run_our_llava_ov_7b_${POOL_SLUG}.slurm" "$@"
bash "${SCRIPT_DIR}/run_our_internvl3_8b_${POOL_SLUG}.slurm" "$@"

echo "Completed all models for N_20."
