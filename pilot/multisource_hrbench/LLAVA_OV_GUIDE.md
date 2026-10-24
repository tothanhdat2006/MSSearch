# LLaVA-OneVision (LLaVA-OV-7B) Running & Debugging Guide

This document records the exact failure modes, architectural fixes, and execution commands required to run **LLaVA-OneVision-7B (`LLaVA-OV-7B`)** under offline conditions on HRBench.

---

## 1. Quick Execution Commands

Activate the Conda environment and run via standard scripts:

### 1.1 Direct Baseline Mode (Vanilla VLM)
```bash
cd MSSearch/pilot/multisource_hrbench

# Positional model path (auto-creates results_baseline_LLaVA-OV-7B/)
bash run_baseline.sh /home/ubuntu/workspace/hr/models/LLaVA-OV-7B

# Target-only evaluation (N=1, ignoring candidate distractors)
bash run_baseline.sh /home/ubuntu/workspace/hr/models/LLaVA-OV-7B --direct_target_only

# Single sample test (e.g. sample 0)
bash run_baseline.sh /home/ubuntu/workspace/hr/models/LLaVA-OV-7B --sample_idx 0
```

### 1.2 Full MSSearch Pipeline Mode
```bash
cd MSSearch/pilot/multisource_hrbench

# Run full visual search with LLaVA-OV-7B
bash run_all.sh /home/ubuntu/workspace/hr/models/LLaVA-OV-7B
```

---

## 2. Checkpoint Architecture Specification

The checkpoint at `/home/ubuntu/workspace/hr/models/LLaVA-OV-7B` (`lmms-lab/llava-onevision-qwen2-7b-ov`) consists of:
- **Language Backbone**: Qwen2-7B (`LlavaQwenForCausalLM`, `vocab_size=152064`, `hidden_size=3584`).
- **Vision Tower**: `google/siglip-so400m-patch14-384` (`image_size=384`, `patch_size=14`, `num_channels=3`).
- **Multimodal Projector**: 2-layer GELU MLP (`mlp2x_gelu`, `mm_hidden_size=1152` -> `hidden_size=3584`).
- **Vision Resampler**: Spatial 2D Pooling (`spatial_pool`, stride=2, bilinear interpolation).
- **Conversation Template**: `qwen_1_5` (`<|im_start|>system...<|im_start|>user...<|im_start|>assistant...`).
- **Global-Local Hyperparameters**: `bias_value=0.6`, `patch_scale=1.2`, `max_anyres_num=12`.

---

## 3. Four Core Failure Modes & Fixes

### 3.1 Failure 1: Architecture Dispatch Defaulting to LLaMA (`lm_head.weight` Shape Mismatch)
- **Error**: `size mismatch for lm_head.weight: copying a param with shape torch.Size([152064, 3584]) from checkpoint, the shape in current model is torch.Size([32000, 4096])`.
- **Cause**: Legacy LLaVA loader checked `if "qwen" in model_path:`. Because the path was `/home/ubuntu/workspace/hr/models/LLaVA-OV-7B` (contains `"ov"` but not `"qwen"`), it instantiated `LlavaLlamaForCausalLM` (vocab 32000, hidden 4096).
- **Fix**: Introspect `config.json` (`architectures`, `model_type`, `_name_or_path`) in addition to checking `ov`, `onevision`, `qwen`, or `conv_type == "qwen_1_5"` in `MSSearch/llava/model/builder.py` and `MSSearch/mssearch/models/modeling_llava.py`.

### 3.2 Failure 2: RoPE Parameters NoneType Subscript in Modern Transformers
- **Error**: `TypeError: 'NoneType' object is not subscriptable` at `self.rope_type = self.config.rope_parameters["rope_type"]` inside `Qwen2RotaryEmbedding`.
- **Cause**: In `transformers >= 4.43`, `PretrainedConfig.rope_scaling` has a setter that mutates `self.rope_parameters`. Legacy code in `llava_qwen.py`, `llava_qwen_moe.py`, `llava_mistral.py`, and `llava_mixtral.py` executed `config.rope_scaling = None`, wiping `config.rope_parameters` to `None`.
- **Fix**:
  1. Commented out `config.rope_scaling = None` in `MSSearch/llava/model/language_model/llava_qwen.py` (matching `llava_llama.py`).
  2. Added fallback in `llava_qwen.py` and `modeling_llava.py`:
     ```python
     if hasattr(config, "rope_parameters") and config.rope_parameters is None:
         config.rope_parameters = {"rope_type": "default", "rope_theta": getattr(config, "rope_theta", 10000.0)}
     ```

### 3.3 Failure 3: Meta-Device Instantiation under `low_cpu_mem_usage=True`
- **Error**: `NotImplementedError: Cannot copy out of meta tensor; no data!` or remote download attempt in offline mode when loading `SigLipVisionTower`.
- **Cause**: Under PyTorch meta-device context during `from_pretrained(low_cpu_mem_usage=True)`, `from_pretrained` on external vision towers attempted remote hub checks or failed to allocate meta storage.
- **Fix**: In `MSSearch/llava/model/multimodal_encoder/siglip_encoder.py` and `clip_encoder.py`, bypass `from_pretrained` when device is meta or when offline:
  ```python
  if hasattr(torch, "get_default_device") and torch.get_default_device().type == "meta":
      self.vision_tower = SigLipVisionModel(self.config)
  ```

### 3.4 Failure 4: Missing `global` and `zoom` Keys in `ModelGlobalLocal.init_prompts`
- **Error**: `KeyError: 'global'` or `KeyError: 'zoom'` during Stage 5 spatial compression / zoomed view prompt generation.
- **Cause**: `ModelGlobalLocal.init_prompts()` only defined `"search"` sub-dict, omitting `"global"` and `"zoom"` prompt templates required by `free_form_using_nodes` and `run_direct_vlm_inference`.
- **Fix**: Added `"global"` (`<image>\n`) and `"zoom"` (`<image>\nThis is the main image...<image>\nThis is the zoomed-in view...`) prompt entries to `ModelGlobalLocal.init_prompts()` in `MSSearch/mssearch/models/modeling_llava.py`.

---

## 4. Verification Checklist

When deploying to a new worker or environment:

1. [ ] Check `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are set.
2. [ ] Check model checkpoint contains:
   - `config.json` (with `"model_type": "llava_qwen"` or `architectures: ["LlavaQwenForCausalLM"]`).
   - `model.safetensors.index.json` or `pytorch_model.bin.index.json`.
   - `tokenizer.json`, `tokenizer_config.json`.
3. [ ] Check `MSSearch/llava/model/language_model/llava_qwen.py` does not contain uncommented `config.rope_scaling = None`.
4. [ ] Run single-sample verification:
   ```bash
   bash run_baseline.sh /home/ubuntu/workspace/hr/models/LLaVA-OV-7B --sample_idx 0
   ```
   Expected console output:
   ```
   [Loading] LLaVA GlobalLocal Engine from '/home/ubuntu/workspace/hr/models/LLaVA-OV-7B' on cuda:0...
   padding side: left
   model: <class 'llava.model.language_model.llava_qwen.LlavaQwenForCausalLM'>
   bias_value: 0.6
   input size: (384, 384)
   background color: (127, 127, 127)
   patch scale: 1.2
   Yes: [9454]
   No: [2753]
   index_yes: 9454
   index_no: 2753
   ```
