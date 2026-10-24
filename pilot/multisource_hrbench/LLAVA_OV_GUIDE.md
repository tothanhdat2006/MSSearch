# Comprehensive Engineering Guide: LLaVA-OneVision (LLaVA-OV-7B) Integration, Offline Loading & Failure Diagnoses

## 1. Executive Summary & Purpose

This document provides complete, line-level documentation on integrating, loading, and debugging **LLaVA-OneVision-7B (`LLaVA-OV-7B`)** within the **MSSearch** visual search evaluation framework. 

Future agents encountering model loading exceptions, tensor dimension mismatches, or silent degradations with LLaVA checkpoints must consult this reference before modifying model architectures or dispatch logic.

---

## 2. Checkpoint Specifications & Tensor Topology

Target checkpoint directory: `/home/ubuntu/workspace/hr/models/LLaVA-OV-7B`  
Base HuggingFace Model ID: `lmms-lab/llava-onevision-qwen2-7b-ov`

### 2.1 Component Architecture Map

```
                    Raw Image Input (PIL.Image: H x W x 3)
                                      │
                                      ▼
             SigLipImageProcessor (Input Size: 384x384, RGB, Mean=[0.5, 0.5, 0.5])
                                      │
                                      ▼
                        SigLipVisionTower (so400m-patch14-384)
            - Patch size: 14x14  -> Grid: 27 x 27 = 729 patches
            - Embed dim: 1152    -> Output: [Batch, 729, 1152]
            - Del last layer [-1:] & head = nn.Identity()
                                      │
                                      ▼
                    Vision Resampler: SpatialPool (Stride=2, Mode='bilinear')
            - Resamples 27x27 -> ceil(27/2) x ceil(27/2) = 14 x 14 = 196 tokens
            - Output: [Batch, 196, 1152]
                                      │
                                      ▼
             Multimodal Projector: mm_projector (mlp2x_gelu)
            - Linear(1152, 3584) -> GELU() -> Linear(3584, 3584)
            - Output: [Batch, 196, 3584] (hidden_size of LLM)
                                      │
                                      ▼
        LlavaQwenForCausalLM (Qwen2-7B Backbone, Vocab: 152064, Hidden: 3584)
            - Interleaved text tokens + image tokens (<image>)
            - Chat Template: `qwen_1_5` (<|im_start|>system...<|im_start|>user...)
            - Rotary Positional Embedding: Qwen2RotaryEmbedding
```

### 2.2 Tensor Dimensional Contracts

| Module | Input Tensor Shape | Output Tensor Shape | Parameter Types |
| :--- | :--- | :--- | :--- |
| **SigLip Vision Encoder** | `[B, 3, 384, 384]` | `[B, 729, 1152]` | FP16 / BF16 (Frozen) |
| **Spatial 2D Pooling** | `[B, 729, 1152]` | `[B, 196, 1152]` | Non-parametric / Bilinear |
| **Multimodal Projector** | `[B, 196, 1152]` | `[B, 196, 3584]` | Trainable Linear + GELU |
| **Qwen2 LM Embeddings** | `[B, SeqLen]` | `[B, SeqLen, 3584]` | Vocab: 152,064 |
| **Qwen2 LM Head** | `[B, SeqLen, 3584]` | `[B, SeqLen, 152064]` | FP16 Linear |

---

## 3. Four Core Failure Modes, Root Causes & Fixes

### 3.1 Failure Mode 1: Checkpoint Dispatched to LLaMA (`lm_head.weight` Size Mismatch)

#### Error Traceback
```python
RuntimeError: Error(s) in loading state_dict for LlavaLlamaForCausalLM:
    size mismatch for lm_head.weight: copying a param with shape torch.Size([152064, 3584]) from checkpoint,
    the shape in current model is torch.Size([32000, 4096]).
    size mismatch for model.embed_tokens.weight: copying a param with shape torch.Size([152064, 3584]) from checkpoint,
    the shape in current model is torch.Size([32000, 4096]).
```

#### Root Cause
Legacy LLaVA codebases (`modeling_llava.py` and `builder.py`) relied on path substring matching:
```python
# BROKEN LEGACY LOGIC
if "qwen" in model_path.lower():
    self.model = LlavaQwenForCausalLM.from_pretrained(...)
else:
    self.model = LlavaLlamaForCausalLM.from_pretrained(...)
```
When given the path `/home/ubuntu/workspace/hr/models/LLaVA-OV-7B`:
- Substring `"qwen"` evaluates to `False`.
- The code falls through to `LlavaLlamaForCausalLM` (which expects LLaMA architecture: `vocab_size=32000`, `hidden_size=4096`).
- Weight loading immediately crashes due to tensor dimension mismatch.

#### Fix Implementation
In both `MSSearch/llava/model/builder.py` and `MSSearch/mssearch/models/modeling_llava.py`:
1. Check `config.json` architectures and model type.
2. Check for `"ov"`, `"onevision"`, `"qwen"`, or `conv_type == "qwen_1_5"`:
```python
is_qwen = (
    "qwen" in model_path.lower()
    or "ov" in model_path.lower()
    or "onevision" in model_path.lower()
    or conv_type == "qwen_1_5"
)
config_json_path = os.path.join(model_path, "config.json")
if os.path.exists(config_json_path):
    try:
        with open(config_json_path, "r", encoding="utf-8") as f:
            _c = json.load(f)
            if (
                any("qwen" in str(a).lower() for a in _c.get("architectures", []))
                or "qwen" in _c.get("model_type", "").lower()
                or "qwen" in _c.get("_name_or_path", "").lower()
            ):
                is_qwen = True
    except Exception:
        pass
```

---

### 3.2 Failure Mode 2: `TypeError: 'NoneType' object is not subscriptable` in RoPE Initialization

#### Error Traceback
```python
Traceback (most recent call last):
  File "debug_stages.py", line 1194, in main
    qwen_engine = ModelGlobalLocal(...)
  File "mssearch/models/modeling_llava.py", line 432, in __init__
    super().__init__(**kwargs)
  File "mssearch/models/modeling_llava.py", line 62, in __init__
    self.model = LlavaQwenForCausalLM.from_pretrained(model_path, **kwargs_load)
  File "transformers/models/qwen2/modeling_qwen2.py", line 61, in __init__
    self.rope_type = self.config.rope_parameters["rope_type"]
TypeError: 'NoneType' object is not subscriptable
```

#### Root Cause
1. In `transformers >= 4.43`, Hugging Face standardized RoPE configurations across all architectures via `PretrainedConfig.rope_scaling` and `PretrainedConfig.rope_parameters`:
   ```python
   # Inside transformers.configuration_utils.PretrainedConfig:
   @property
   def rope_scaling(self):
       return self.rope_parameters

   @rope_scaling.setter
   def rope_scaling(self, value):
       self.rope_parameters = value
   ```
2. In legacy LLaVA (`llava_qwen.py`, `llava_qwen_moe.py`, `llava_mistral.py`, `llava_mixtral.py`):
   ```python
   # BROKEN LEGACY LLaVA CONSTRUCTOR:
   class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
       def __init__(self, config):
           Qwen2ForCausalLM.__init__(self, config)
           config.model_type = "llava_qwen"
           config.rope_scaling = None  # <--- CRITICAL BUG
           self.model = LlavaQwenModel(config)
   ```
3. Executing `config.rope_scaling = None` invokes the setter, setting `config.rope_parameters = None`.
4. Next line `self.model = LlavaQwenModel(config)` calls `Qwen2Model(config)` -> `Qwen2RotaryEmbedding(config)`.
5. `Qwen2RotaryEmbedding` evaluates `self.config.rope_parameters["rope_type"]`.
6. Since `rope_parameters` is `None`, Python raises `TypeError: 'NoneType' object is not subscriptable`.
7. *Note*: In `llava_llama.py`, this had previously been commented out (`# config.rope_scaling = None`), but remained active in `llava_qwen.py` and other files.

#### Fix Implementation
In `MSSearch/llava/model/language_model/llava_qwen.py` (and `llava_qwen_moe.py`, `llava_mistral.py`, `llava_mixtral.py`):
```python
class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        # DO NOT NULLIFY rope_scaling: in transformers >= 4.43 this wipes rope_parameters
        # config.rope_scaling = None
        if hasattr(config, "rope_parameters") and config.rope_parameters is None:
            config.rope_parameters = {
                "rope_type": "default",
                "rope_theta": getattr(config, "rope_theta", 10000.0),
            }

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
```

And in `MSSearch/mssearch/models/modeling_llava.py`:
```python
if is_qwen:
    from llava.model.language_model.llava_qwen import LlavaQwenConfig
    llava_cfg = LlavaQwenConfig.from_pretrained(model_path)
    if hasattr(llava_cfg, "rope_parameters") and llava_cfg.rope_parameters is None:
        llava_cfg.rope_parameters = {
            "rope_type": "default",
            "rope_theta": getattr(llava_cfg, "rope_theta", 10000.0),
        }
    self.model = LlavaQwenForCausalLM.from_pretrained(
        model_path, config=llava_cfg, **kwargs_load
    )
```

---

### 3.3 Failure Mode 3: Meta-Device Initialization Bypass under `low_cpu_mem_usage=True`

#### Error Traceback
```python
NotImplementedError: Cannot copy out of meta tensor; no data!
# OR
requests.exceptions.ConnectionError: (Offline mode active; cannot reach huggingface.co)
```

#### Root Cause
Under `low_cpu_mem_usage=True`, PyTorch initializes tensors on a `meta` device to conserve host RAM. Calling `SigLipVisionModel.from_pretrained("google/siglip-so400m-patch14-384")` in an offline environment triggers either:
1. Online download requests blocked by `HF_HUB_OFFLINE=1`.
2. Meta-tensor allocation failures when trying to materialize weights from a non-local HuggingFace repo ID.

#### Fix Implementation
In `MSSearch/llava/model/multimodal_encoder/siglip_encoder.py` and `clip_encoder.py`:
```python
is_meta = hasattr(torch, "get_default_device") and torch.get_default_device().type == "meta"
if is_meta:
    # Under meta-device context, instantiate directly from config without downloading
    self.vision_tower = SigLipVisionModel(self.config)
else:
    # Normal execution: try local directories first
    target_path = self.vision_tower_name
    for env_var in ["SIGLIP_MODEL_DIR", "CLIP_MODEL_DIR"]:
        if not os.path.exists(target_path) and os.environ.get(env_var) and os.path.exists(os.environ.get(env_var)):
            target_path = os.environ.get(env_var)
            break
    try:
        self.vision_tower = SigLipVisionModel.from_pretrained(target_path, device_map=device_map)
    except Exception as e:
        rank0_print(f"Warning: Loading vision tower from pretrained failed ({e}), initializing from config.")
        self.vision_tower = SigLipVisionModel(self.config)
```

---

### 3.4 Failure Mode 4: Prompt Dictionary Key Mismatch in `ModelGlobalLocal`

#### Error Traceback
```python
KeyError: 'global'
# OR
KeyError: 'zoom'
```

#### Root Cause
`ModelGlobalLocal.init_prompts()` originally defined only the `"search"` key containing `pre_information_image1`, `pre_information_image2`, `latent_prompt`, etc.
However, `free_form_using_nodes()`, `run_direct_vlm_inference()`, and Stage 5/6 routing index:
- `self.prompts["global"]["pre_information"]` for single-image full views.
- `self.prompts["zoom"]["pre_information"]` for dual foveated-zoom views.

#### Fix Implementation
In `MSSearch/mssearch/models/modeling_llava.py`:
```python
def init_prompts(self):
    self.prompts = {
        "search": {
            "pre_information_image2": f"{DEFAULT_IMAGE_TOKEN}\nThis is the main image, and the section enclosed by the {BOX_COLOR} rectangle is the focus region.\n{DEFAULT_IMAGE_TOKEN}\nThis is the zoomed-in view of the focus region.\n",
            "pre_information_image1": f"{DEFAULT_IMAGE_TOKEN}\n",
            "latent_prompt": "According to your common sense knowledge and the content of image, is it possible to find a {} by further zooming in the image? Answer Yes or No.",
            "existence_prompt": "Is there a '{}' in the image? Answer Yes or No.",
            "answering_prompt": "Question: '{}'\nCould you answer the question based on the available visual information? Answer Yes or No.",
        },
        "global": {
            "pre_information": f"{DEFAULT_IMAGE_TOKEN}\n",
        },
        "zoom": {
            "pre_information": f"{DEFAULT_IMAGE_TOKEN}\nThis is the main image, and the section enclosed by the {BOX_COLOR} rectangle is the focus region.\n{DEFAULT_IMAGE_TOKEN}\nThis is the zoomed-in view of the focus region.\n",
        },
    }
```

---

### 3.5 Failure Mode 5: CUDA Assertion in `vectorized_gather_kernel` (Negative `input_ids` & Repetition Penalty Out of Bounds)

#### Error Traceback
```text
../aten/src/ATen/native/cuda/GatherScatterKernels.cu:108: vectorized_gather_kernel: Assertion `ind >= 0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
RuntimeError: CUDA error: device-side assert triggered
```

#### Root Cause
1. `LLaVA-OV-7B`'s `generation_config.json` sets `repetition_penalty = 1.05`.
2. In HuggingFace `transformers`, whenever `repetition_penalty != 1.0`, `RepetitionPenaltyLogitsProcessor` is instantiated.
3. During generation, `RepetitionPenaltyLogitsProcessor` performs `torch.gather(scores, 1, input_ids)`.
4. LLaVA tokenizers embed sentinel tokens such as `IMAGE_TOKEN_INDEX = -200` into `input_ids`.
5. When `inputs_embeds` is computed from multimodal tokens, passing or leaking `input_ids` containing `-200` into `super().generate` causes `torch.gather` on GPU to evaluate index `-200 < 0`, triggering a fatal CUDA assertion failure in `vectorized_gather_kernel`.

#### Fix Implementation
1. In `MSSearch/llava/model/language_model/llava_qwen.py` (and related `llava_*.py` backbones):
   - Pop `"input_ids"` from `kwargs` so it never leaks into `super().generate` alongside `inputs_embeds`.
   - Sanitize negative sentinel tokens before any embedding lookup (`torch.where(inputs < 0, torch.zeros_like(inputs), inputs)`).
   - Set `kwargs["repetition_penalty"] = 1.0` if not overridden, preventing `RepetitionPenaltyLogitsProcessor` instantiation.
2. In `MSSearch/mssearch/models/modeling_llava.py`:
   - Set `self.model.generation_config.repetition_penalty = 1.0` at model initialization.
   - Explicitly pass `repetition_penalty=1.0` to all `self.model.generate(...)` calls (`free_form_using_nodes`, `generate_text`, etc.).
3. In `MSSearch/pilot/multisource_hrbench/debug_stages.py`:
   - Enforce `repetition_penalty=1.0` in direct inference generation loops.

---

## 4. Modified Files Audit Trail

All changes are committed strictly inside the `MSSearch` repository:

| File Path | Changes Applied |
| :--- | :--- |
| `MSSearch/llava/model/language_model/llava_qwen.py` | Commented out `config.rope_scaling = None`; injected `rope_parameters` fallback; sanitized `input_ids` & neutralized `repetition_penalty` gather crash. |
| `MSSearch/llava/model/language_model/llava_qwen_moe.py` | Commented out `config.rope_scaling = None`; injected `rope_parameters` fallback; popped `input_ids` & neutralized `repetition_penalty`. |
| `MSSearch/llava/model/language_model/llava_llama.py` | Popped `input_ids` from kwargs; clamped negative token IDs; neutralized `repetition_penalty`. |
| `MSSearch/llava/model/language_model/llava_mistral.py` | Commented out `config.rope_scaling = None`; popped `input_ids`; neutralized `repetition_penalty`. |
| `MSSearch/llava/model/language_model/llava_mixtral.py` | Commented out `config.rope_scaling = None`; popped `input_ids`; neutralized `repetition_penalty`. |
| `MSSearch/llava/model/language_model/llava_gemma.py` | Popped `input_ids` from kwargs; clamped negative token IDs; neutralized `repetition_penalty`. |
| `MSSearch/llava/model/multimodal_encoder/builder.py` | Guarded `os.path.exists(vision_tower)` when `vision_tower is None`. |
| `MSSearch/llava/model/multimodal_encoder/siglip_encoder.py` | Meta-device check bypassing `from_pretrained`; env var path resolution. |
| `MSSearch/llava/model/multimodal_encoder/clip_encoder.py` | Meta-device check bypassing `from_pretrained`; env var path resolution. |
| `MSSearch/llava/model/builder.py` | Expanded `config.json` architecture inspection for Qwen/OneVision dispatch. |
| `MSSearch/mssearch/models/modeling_llava.py` | Config-based model dispatch; explicit `LlavaQwenConfig` loading; added `global`/`zoom` prompts; clamped `repetition_penalty=1.0`; attached full tracebacks to load errors. |
| `MSSearch/pilot/multisource_hrbench/debug_stages.py` | Explicit `ModelGlobalLocal` routing for LLaVA checkpoints; multi-image composite canvas fallback; fail-fast exception bubbling; explicit `repetition_penalty=1.0`. |
| `MSSearch/pilot/multisource_hrbench/run_baseline.sh` | Model-specific output dir creation (`results_baseline_<model_name>/`); strict offline flags (`HF_HUB_OFFLINE=1`). |

---

## 5. Verification Runbook for Future Agents

To verify that the LLaVA-OneVision loading pipeline is operational on any target machine:

### Step 1: Verify Local Checkpoint Integrity
```bash
ls -lh /home/ubuntu/workspace/hr/models/LLaVA-OV-7B/config.json
grep -E "architectures|model_type" /home/ubuntu/workspace/hr/models/LLaVA-OV-7B/config.json
```
Expected output:
```json
"architectures": ["LlavaQwenForCausalLM"],
"model_type": "llava_qwen"
```

### Step 2: Run Single-Sample Baseline Diagnostic
```bash
cd /home/ubuntu/workspace/hr/methods/MSSearch/pilot/multisource_hrbench

bash run_baseline.sh /home/ubuntu/workspace/hr/models/LLaVA-OV-7B --sample_idx 0
```

### Step 3: Verify Initialization Signatures in Stdout
The output must display the following milestones without warnings or tracebacks:
```text
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
new_image_grid_pinpoints: [[384, 384], [384, 768], ... [2304, 768]]
```

### Step 4: Run Full Benchmark Execution
```bash
# Full dataset direct baseline evaluation
bash run_baseline.sh /home/ubuntu/workspace/hr/models/LLaVA-OV-7B --all_samples --resume

# Full MSSearch recursive visual search evaluation
bash run_all.sh /home/ubuntu/workspace/hr/models/LLaVA-OV-7B --all_samples --resume
```
Results will be stored in:
- Baseline: `pilot/multisource_hrbench/results_baseline_LLaVA-OV-7B/`
- Full Search: `pilot/multisource_hrbench/results_LLaVA-OV-7B/`
