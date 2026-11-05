#!/usr/bin/env bash
# ==============================================================================
# MSSEARCH ABLATION STUDY CENTRALIZED PATHS & CONFIGURATION
# Configures Detection (SAM3, Grounding DINO, DINO-X) and
# Screening (SigLIP, CLIP-ViT-B/32, CLIP-ViT-L/14, EVA-02-L/14) models.
# ==============================================================================

_ABLATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_OUR_DIR="$(cd "${_ABLATION_DIR}/.." && pwd)"

# Source base environment paths
if [ -f "${_OUR_DIR}/env_paths.sh" ]; then
    source "${_OUR_DIR}/env_paths.sh"
fi

# ------------------------------------------------------------------------------
# Ablation Detection Models
# ------------------------------------------------------------------------------
export GROUNDING_DINO_PATH="${GROUNDING_DINO_PATH:-IDEA-Research/grounding-dino-base}"
export DINOX_PATH="${DINOX_PATH:-}"
export DINOX_API_TOKEN="${DINOX_API_TOKEN:-}"

# ------------------------------------------------------------------------------
# Ablation Visual Screening Models
# ------------------------------------------------------------------------------
export CLIP_B32_NAME="${CLIP_B32_NAME:-openai/clip-vit-base-patch32}"
export CLIP_L14_NAME="${CLIP_L14_NAME:-openai/clip-vit-large-patch14}"
export EVA02_L14_NAME="${EVA02_L14_NAME:-EVA02-L-14-336}"

# ------------------------------------------------------------------------------
# Ablation Output Base Directory
# ------------------------------------------------------------------------------
export ABLATION_OUTPUT_DIR="${ABLATION_OUTPUT_DIR:-${_ABLATION_DIR}/results_ablation}"
