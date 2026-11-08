#!/usr/bin/env bash
# ==============================================================================
# MSSEARCH V2 CENTRALIZED ENVIRONMENT & PATH CONFIGURATION
# Edit THIS SINGLE FILE to configure all execution environments:
# - Local A6000 workstations (scripts/A6000/*.sh)
# - A100 HPC clusters (scripts/A100/*.slurm)
# - Colab and Modal evaluations
# ==============================================================================

# 1. Python Environment Activation
export CONDA_ENV_NAME="hrsearch"
export VENV_PATH=".../.venv/bin/activate"

# 2. Base Model & Perception Checkpoints
export MODEL_DIR=".../models"
export SAM3_PATH=".../models/sam3/sam3.pt"
export DETECTOR_PATH="${SAM3_PATH}"
export SIGLIP_NAME="google/siglip2-so400m-patch16-512"
export NLP_MODEL=".../models/spacy/en_core_web_sm-3.8.0"

# 3. STASHBench Dataset & Candidate Pool Images
export DATASET_PATH=".../STASHBench/annotations_part1234_merged.json"
export IMAGE_DIR=".../STASHBench/pool"

# 4. Base Output Directory
export OUTPUT_BASE_DIR=""
