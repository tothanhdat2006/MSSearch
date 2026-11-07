# A6000 Evaluation Scripts & Execution Guide

This directory contains execution scripts and workflow orchestration utilities for evaluating MSSearch V2 on the local A6000 workstation (`gpu-a240`).

---

## Directory Overview

| Script Pattern | Model | Pool Configurations |
|---|---|---|
| `run_our_qwen3_vl_8b_n*.sh` | `Qwen3-VL-8B` (`Qwen/Qwen3-VL-8B-Instruct`) | `n1`, `n5`, `n10`, `n20` |
| `run_our_llava_ov_7b_n*.sh` | `llava-onevision-qwen2-7b-ov` | `n1`, `n5`, `n10`, `n20` |
| `run_our_internvl3_8b_n*.sh` | `InternVL3-8B` (`OpenGVLab/InternVL3-8B`) | `n1`, `n5`, `n10`, `n20` |
| `run_our_all_n*.sh` | Master pool runner (runs all 3 models sequentially) | `n1`, `n5`, `n10`, `n20` |
| `run_our_sweep.sh` | Full matrix sweep (runs N1, N5, N10, N20 across all 3 models) | All |
| `wait_pid.sh` | Process watcher to chain background executions | Any PID |

---

## Prerequisites & Environment

Before running direct Python commands, ensure environment variables and offline flags are set:

```bash
# Sourcing centralized paths (from repository root)
source experiments/our/env_paths.sh

# Python path and offline model loading flags
export PYTHONPATH=".:experiments/our:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Common paths on the A6000 server:
- **SAM3**: `${SAM3_PATH:-/data/shared/cvpr/dat/models/sam3/sam3.pt}`
- **SigLIP-2**: `${SIGLIP_NAME:-/data/shared/cvpr/dat/models/siglip2-so400m-patch16-512}`
- **spaCy**: `${NLP_MODEL:-/data/shared/cvpr/dat/models/en_core_web_sm-3.8.0}`
- **Dataset**: `${DATASET_PATH:-/data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json}`
- **Image Dir**: `${IMAGE_DIR:-/data/shared/cvpr/dat/simple/data/STASHBench/pool}`
- **Config**: `experiments/our/config_default.json`

---

## Direct Python Command Lines

### 1. Qwen3-VL-8B (`Qwen3-VL-8B`)

#### N_1
```bash
python experiments/our/eval_runner.py \
    --model_path /data/shared/cvpr/dat/models/Qwen3-VL-8B \
    --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
    --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
    --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
    --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
    --output_dir experiments/our/results_our_a6000_Qwen3-VL-8B_N_1 \
    --config experiments/our/config_default.json \
    --pool_mode N_1
```

#### N_5
```bash
python experiments/our/eval_runner.py \
    --model_path /data/shared/cvpr/dat/models/Qwen3-VL-8B \
    --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
    --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
    --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
    --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
    --output_dir experiments/our/results_our_a6000_Qwen3-VL-8B_N_5 \
    --config experiments/our/config_default.json \
    --pool_mode N_5
```

#### N_10
```bash
python experiments/our/eval_runner.py \
    --model_path /data/shared/cvpr/dat/models/Qwen3-VL-8B \
    --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
    --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
    --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
    --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
    --output_dir experiments/our/results_our_a6000_Qwen3-VL-8B_N_10 \
    --config experiments/our/config_default.json \
    --pool_mode N_10
```

#### N_20
```bash
python experiments/our/eval_runner.py \
    --model_path /data/shared/cvpr/dat/models/Qwen3-VL-8B \
    --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
    --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
    --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
    --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
    --output_dir experiments/our/results_our_a6000_Qwen3-VL-8B_N_20 \
    --config experiments/our/config_default.json \
    --pool_mode N_20
```

---

### 2. LLaVA-OneVision 7B (`llava-onevision-qwen2-7b-ov`)

#### N_1
```bash
python experiments/our/eval_runner.py \
    --model_path /data/shared/cvpr/dat/models/llava-onevision-qwen2-7b-ov \
    --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
    --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
    --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
    --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
    --output_dir experiments/our/results_our_a6000_llava-onevision-qwen2-7b-ov_N_1 \
    --config experiments/our/config_default.json \
    --pool_mode N_1
```

#### N_5
```bash
python experiments/our/eval_runner.py \
    --model_path /data/shared/cvpr/dat/models/llava-onevision-qwen2-7b-ov \
    --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
    --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
    --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
    --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
    --output_dir experiments/our/results_our_a6000_llava-onevision-qwen2-7b-ov_N_5 \
    --config experiments/our/config_default.json \
    --pool_mode N_5
```

#### N_10
```bash
python experiments/our/eval_runner.py \
    --model_path /data/shared/cvpr/dat/models/llava-onevision-qwen2-7b-ov \
    --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
    --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
    --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
    --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
    --output_dir experiments/our/results_our_a6000_llava-onevision-qwen2-7b-ov_N_10 \
    --config experiments/our/config_default.json \
    --pool_mode N_10
```

#### N_20
```bash
python experiments/our/eval_runner.py \
    --model_path /data/shared/cvpr/dat/models/llava-onevision-qwen2-7b-ov \
    --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
    --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
    --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
    --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
    --output_dir experiments/our/results_our_a6000_llava-onevision-qwen2-7b-ov_N_20 \
    --config experiments/our/config_default.json \
    --pool_mode N_20
```

---

### 3. InternVL3-8B (`InternVL3-8B`)

#### N_1
```bash
python experiments/our/eval_runner.py \
    --model_path /data/shared/cvpr/dat/models/InternVL3-8B \
    --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
    --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
    --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
    --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
    --output_dir experiments/our/results_our_a6000_InternVL3-8B_N_1 \
    --config experiments/our/config_default.json \
    --pool_mode N_1
```

#### N_5
```bash
python experiments/our/eval_runner.py \
    --model_path /data/shared/cvpr/dat/models/InternVL3-8B \
    --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
    --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
    --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
    --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
    --output_dir experiments/our/results_our_a6000_InternVL3-8B_N_5 \
    --config experiments/our/config_default.json \
    --pool_mode N_5
```

#### N_10
```bash
python experiments/our/eval_runner.py \
    --model_path /data/shared/cvpr/dat/models/InternVL3-8B \
    --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
    --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
    --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
    --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
    --output_dir experiments/our/results_our_a6000_InternVL3-8B_N_10 \
    --config experiments/our/config_default.json \
    --pool_mode N_10
```

#### N_20
```bash
python experiments/our/eval_runner.py \
    --model_path /data/shared/cvpr/dat/models/InternVL3-8B \
    --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
    --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
    --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
    --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
    --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
    --output_dir experiments/our/results_our_a6000_InternVL3-8B_N_20 \
    --config experiments/our/config_default.json \
    --pool_mode N_20
```

---

## Chaining Runs with `wait_pid.sh`

`wait_pid.sh` monitors an active process by its Process ID (`PID`). Once the monitored process terminates, it immediately executes your next command line.

### Recipe 1: Launch Bash Script, then Chain Python Command
```bash
# 1. Start Qwen3-VL-8B N_1 in background and grab PID
nohup bash experiments/our/scripts/A6000/run_our_qwen3_vl_8b_n1.sh > log_qwen3_n1.txt 2>&1 &
FIRST_PID=$!
echo "Running N_1 with PID: ${FIRST_PID}"

# 2. Chain N_5 Python evaluation to run right after N_1 completes
nohup bash experiments/our/scripts/A6000/wait_pid.sh "${FIRST_PID}" \
    python experiments/our/eval_runner.py \
        --model_path /data/shared/cvpr/dat/models/Qwen3-VL-8B \
        --sam3_path /data/shared/cvpr/dat/models/sam3/sam3.pt \
        --siglip_name /data/shared/cvpr/dat/models/siglip2-so400m-patch16-512 \
        --nlp_model /data/shared/cvpr/dat/models/en_core_web_sm-3.8.0 \
        --dataset_path /data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json \
        --image_dir /data/shared/cvpr/dat/simple/data/STASHBench/pool \
        --output_dir experiments/our/results_our_a6000_Qwen3-VL-8B_N_5 \
        --config experiments/our/config_default.json \
        --pool_mode N_5 \
    > log_qwen3_n5.txt 2>&1 &
```

### Recipe 2: Chain to an Already-Running Process
If you already started a script (e.g. PID `38491` from `ps aux | grep eval_runner`):
```bash
# Chain the next script to start automatically when PID 38491 finishes:
nohup bash experiments/our/scripts/A6000/wait_pid.sh 38491 \
    bash experiments/our/scripts/A6000/run_our_qwen3_vl_8b_n10.sh \
    > log_qwen3_n10.txt 2>&1 &
```

### Recipe 3: Multi-Stage Chain (N1 -> N5 -> N10 -> N20)
```bash
# Run N1
nohup bash experiments/our/scripts/A6000/run_our_qwen3_vl_8b_n1.sh > run_n1.log 2>&1 &
PID1=$!

# Chain N5 to PID1
nohup bash experiments/our/scripts/A6000/wait_pid.sh "${PID1}" \
    bash experiments/our/scripts/A6000/run_our_qwen3_vl_8b_n5.sh > run_n5.log 2>&1 &
PID2=$!

# Chain N10 to PID2
nohup bash experiments/our/scripts/A6000/wait_pid.sh "${PID2}" \
    bash experiments/our/scripts/A6000/run_our_qwen3_vl_8b_n10.sh > run_n10.log 2>&1 &
PID3=$!

# Chain N20 to PID3
nohup bash experiments/our/scripts/A6000/wait_pid.sh "${PID3}" \
    bash experiments/our/scripts/A6000/run_our_qwen3_vl_8b_n20.sh > run_n20.log 2>&1 &
```
