#!/usr/bin/env bash
#SBATCH --job-name=mssearch_ablation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_ablation_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Execute matrix sweep on HPC node
bash "${SCRIPT_DIR}/run_ablation_matrix.sh" "$@"
