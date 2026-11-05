#!/usr/bin/env bash
# ==============================================================================
# MSSEARCH ABLATION STUDY CENTRALIZED PATHS & CONFIGURATION
# Configures Detection (SAM3, Grounding DINO, Florence-2) and
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
export GROUNDING_DINO_PATH="${GROUNDING_DINO_PATH:-${MODEL_DIR}/grounding-dino-base}"
if [ ! -d "${GROUNDING_DINO_PATH}" ]; then export GROUNDING_DINO_PATH="IDEA-Research/grounding-dino-base"; fi

export FLORENCE2_PATH="${FLORENCE2_PATH:-${MODEL_DIR}/Florence-2-large}"
if [ ! -d "${FLORENCE2_PATH}" ]; then export FLORENCE2_PATH="microsoft/Florence-2-large"; fi

export DINOV3_PATH="${DINOV3_PATH:-${MODEL_DIR}/dinov3-vitl16-pretrain-lvd1689m}"
if [ ! -d "${DINOV3_PATH}" ]; then export DINOV3_PATH="facebook/dinov3-vitl16-pretrain-lvd1689m"; fi

# ------------------------------------------------------------------------------
# Ablation Visual Screening Models
# ------------------------------------------------------------------------------
export CLIP_B32_NAME="${CLIP_B32_NAME:-${MODEL_DIR}/clip-vit-base-patch32}"
if [ ! -d "${CLIP_B32_NAME}" ]; then export CLIP_B32_NAME="openai/clip-vit-base-patch32"; fi

export CLIP_L14_NAME="${CLIP_L14_NAME:-${MODEL_DIR}/clip-vit-large-patch14}"
if [ ! -d "${CLIP_L14_NAME}" ]; then export CLIP_L14_NAME="openai/clip-vit-large-patch14"; fi

export EVA02_L14_NAME="${EVA02_L14_NAME:-${MODEL_DIR}/eva02_large_patch14_clip_336}"
if [ ! -d "${EVA02_L14_NAME}" ]; then export EVA02_L14_NAME="timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"; fi

export SIGLIP2_384_NAME="${SIGLIP2_384_NAME:-${MODEL_DIR}/siglip2-so400m-patch16-384}"
if [ ! -d "${SIGLIP2_384_NAME}" ]; then export SIGLIP2_384_NAME="google/siglip2-so400m-patch16-384"; fi

export SIGLIP2_512_NAME="${SIGLIP2_512_NAME:-${MODEL_DIR}/siglip2-so400m-patch16-512}"
if [ ! -d "${SIGLIP2_512_NAME}" ]; then export SIGLIP2_512_NAME="google/siglip2-so400m-patch16-512"; fi

# ------------------------------------------------------------------------------
# Ablation Output Base Directory
# ------------------------------------------------------------------------------
export ABLATION_OUTPUT_DIR="${ABLATION_OUTPUT_DIR:-${_ABLATION_DIR}/results_ablation}"
