#!/usr/bin/env bash
# ==============================================================================
# MSSEARCH V2 CENTRALIZED ENVIRONMENT & PATH CONFIGURATION
# Edit THIS SINGLE FILE to configure all execution environments:
# - Local A6000 workstations (scripts/A6000/*.sh)
# - A100 HPC clusters (scripts/A100/*.slurm)
# - Colab and Modal evaluations
# ==============================================================================

# Derive repo root relative to this configuration file
_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_MSSEARCH_ROOT="$(cd "${_CONFIG_DIR}/../.." && pwd)"
_REPO_ROOT="$(cd "${_MSSEARCH_ROOT}/../.." && pwd)"

# ------------------------------------------------------------------------------
# 1. Python Environment Activation
# ------------------------------------------------------------------------------
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-hrsearch}"
export VENV_PATH="${VENV_PATH:-/data/shared/cvpr/dat/.venv/bin/activate}"

# ------------------------------------------------------------------------------
# 2. Base Model & Perception Checkpoints
# ------------------------------------------------------------------------------
export MODEL_DIR="${MODEL_DIR:-/data/shared/cvpr/dat/models}"
if [ ! -d "${MODEL_DIR}" ] && [ -d "${_REPO_ROOT}/models" ]; then
    export MODEL_DIR="${_REPO_ROOT}/models"
fi

export SAM3_PATH="${SAM3_PATH:-${MODEL_DIR}/sam3/sam3.pt}"
if [ ! -f "${SAM3_PATH}" ] && [ -f "${_REPO_ROOT}/models/sam3/sam3.pt" ]; then
    export SAM3_PATH="${_REPO_ROOT}/models/sam3/sam3.pt"
fi

export SIGLIP_NAME="${SIGLIP_NAME:-${MODEL_DIR}/siglip-base-patch16-384}"
if [ ! -d "${SIGLIP_NAME}" ] && [ -d "${_REPO_ROOT}/models/siglip-base-patch16-384" ]; then
    export SIGLIP_NAME="${_REPO_ROOT}/models/siglip-base-patch16-384"
elif [ ! -d "${SIGLIP_NAME}" ]; then
    export SIGLIP_NAME="google/siglip-base-patch16-384"
fi

export NLP_MODEL="${NLP_MODEL:-${MODEL_DIR}/spacy/en_core_web_sm-3.8.0}"
if [ ! -d "${NLP_MODEL}" ]; then
    export NLP_MODEL="en_core_web_sm"
fi

# ------------------------------------------------------------------------------
# 3. STASHBench Dataset & Candidate Pool Images
# ------------------------------------------------------------------------------
export DATASET_PATH="${DATASET_PATH:-/data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json}"
if [ ! -f "${DATASET_PATH}" ] && [ -f "${_REPO_ROOT}/data/STASHBench/annotations_part1234_merged.json" ]; then
    export DATASET_PATH="${_REPO_ROOT}/data/STASHBench/annotations_part1234_merged.json"
fi

export IMAGE_DIR="${IMAGE_DIR:-/data/shared/cvpr/dat/simple/data/STASHBench/pool}"
if [ ! -d "${IMAGE_DIR}" ] && [ -d "${_REPO_ROOT}/data/STASHBench/pool" ]; then
    export IMAGE_DIR="${_REPO_ROOT}/data/STASHBench/pool"
fi

# ------------------------------------------------------------------------------
# 4. Optional Base Output Directory
# ------------------------------------------------------------------------------
export OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-}"
