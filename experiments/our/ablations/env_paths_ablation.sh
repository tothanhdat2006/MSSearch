#!/usr/bin/env bash
# ==============================================================================
# MSSEARCH ABLATION STUDY CENTRALIZED PATHS & CONFIGURATION
# Configures Detection (SAM3, Grounding DINO, Florence-2, DINOv3) and
# Screening (SigLIP, CLIP-ViT-B/32, CLIP-ViT-L/14, EVA-02-L/14) models.
# ==============================================================================

# 1. Ablation Detection Models
export GROUNDING_DINO_PATH=".../models/grounding-dino-base"
export FLORENCE2_PATH=".../models/Florence-2-large"
export DINOV3_PATH=".../models/dinov3-vitl16-pretrain-lvd1689m"

# 2. Ablation Visual Screening Models
export CLIP_B32_NAME=".../models/clip-vit-base-patch32"
export CLIP_L14_NAME=".../models/clip-vit-large-patch14"
export EVA02_L14_NAME=".../models/eva02_large_patch14_clip_336"
export SIGLIP2_384_NAME=".../models/siglip2-so400m-patch16-384"
export SIGLIP2_512_NAME=".../models/siglip2-so400m-patch16-512"

# 3. Ablation Output Base Directory
export ABLATION_OUTPUT_DIR=""
