#!/bin/bash

# ==============================================================================
# Dual-Path Multi-Source Visual Search (MSSearch) Evaluation Script
# ==============================================================================

ANNOTATION_PATH="/kaggle/input/datasets/thnhtt/vs-dataset/anno.json"
BASE_DATA_DIR="/kaggle/input/datasets/thnhtt/vs-dataset"
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
SAM_MODEL_PATH="/kaggle/input/datasets/thnhtt/vs-sam3/sam3/sam3/sam3.pt"
SIGLIP_MODEL="google/siglip2-so400m-patch16-512"
BENCHMARK="hr-bench_multisource"

NUM_CHUNKS=1
CHUNK_IDX=0

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --annotation-path) ANNOTATION_PATH="$2"; shift ;;
        --base-data-dir) BASE_DATA_DIR="$2"; shift ;;
        --model-path) MODEL_PATH="$2"; shift ;;
        --sam-model-path) SAM_MODEL_PATH="$2"; shift ;;
        --siglip-model-name) SIGLIP_MODEL="$2"; shift ;;
        --num-chunks) NUM_CHUNKS="$2"; shift ;;
        --chunk-idx) CHUNK_IDX="$2"; shift ;;
        --benchmark) BENCHMARK="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "========================================================================"
echo "🚀 Starting Dual-Path Multi-Source Visual Search (MSSearch)"
echo "========================================================================"
echo "Model         : ${MODEL_PATH}"
echo "SAM3 Weights  : ${SAM_MODEL_PATH}"
echo "SigLIP Model  : ${SIGLIP_MODEL}"
echo "Annotations   : ${ANNOTATION_PATH}"
echo "Chunk         : ${CHUNK_IDX} / ${NUM_CHUNKS}"
echo "========================================================================"

python perform_MSSearch.py \
    --annotation-path "${ANNOTATION_PATH}" \
    --base-data-dir "${BASE_DATA_DIR}" \
    --model-path "${MODEL_PATH}" \
    --sam-model-path "${SAM_MODEL_PATH}" \
    --siglip-model-name "${SIGLIP_MODEL}" \
    --benchmark "${BENCHMARK}" \
    --num-chunks ${NUM_CHUNKS} \
    --chunk-idx ${CHUNK_IDX} \
    --fast-path-threshold 0.70 \
    --entropy-threshold 0.35 \
    --partition-mode hybrid \
    --max-patch-size 1024 \
    --bbox-expansion-factor 1.25 \
    --verbose

