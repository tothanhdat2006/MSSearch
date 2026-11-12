# 📑 Comprehensive Summary of Architectural Changes, Fixes & Ablation Suite

---

## 1. Executive Summary & Architectural Motivation

Based on empirical evidence across the $N \in \{1, 5, 10, 20\}$ benchmark evaluations:
1. **Canvas Aggregation Degrades Accuracy**: Multi-patch canvas synthesis (Continuous Enclosure and Spatial Compressed Canvases) consistently degraded accuracy across all variants (ASP: $51.49\% \to 55.06\%$, Rigid: $52.38\% \to 54.46\%$, SGAP: $51.49\% \to 55.06\%$).
2. **Fast Path Harms Performance & Distorts Galleries**: Fast Path dropped text accuracy to $29.46\%$ and prematurely truncated candidate images from multi-image galleries before visual search could inspect them.
3. **ASP vs. SGAP Equivalence**: Full-image Shannon entropy across all 843 benchmark pool images is always $H \ge 0.64$ (well above the $0.35$ gating threshold). As a result, ASP degenerated into pure SGAP at the root.
4. **Relational Intent Removal**: Since canvas stitching was eliminated, relational questions are solved directly as clean, native single-crop visual searches.

---

## 2. Main Method Architecture (4 Core Modules)

The visual search engine in `mssearch/` is reorganized into four distinct, testable modules:

```
[Question + Candidate Images (N)]
       │
       ▼
Stage 1: Ultra-Lean Question Parsing (enable_relational_parsing=False)
       ├── Strips relational few-shots and schema instructions
       └── Extracts active_objects; forces is_relational = False, anchor_entity = None
       │
       ├─► [Stage 2 Fast Path: BYPASSED by default (enable_fast_path=False)]
       │
       ▼
Module 1: Image Partitioning (Stage 3)
       ├── Default: partition_mode = "sgap_only"
       └── Autonomous SAM3 backbone feature extraction (replaces all-zero fallbacks)
       │
       ▼
Module 2: Feature Grouping (Stage 4)
       ├── Master Toggle: enable_feature_grouping = True
       └── Cosine clustering (cos >= 0.80) using visual embeddings into:
           ├── Detected Groups (SAM3 found candidate objects)
           └── Absent Groups (SAM3 found no target)
       │
       ├────────────────────────────────────────┬────────────────────────────────────────┐
       ▼                                        ▼                                        ▼
Module 3: Existence Scanning               Module 4: Latent Re-partitioning           Variant B Naive Fallback
(enable_existence_scanning: bool)         (enable_latent_repartition: bool)          (Ablations)
- True: VLM existence probe on best det   - True: VLM latent probe on centroid       - Det: Retain top det patches
  micro-crop (conf >= 0.0)                  (c_latent >= 0.70) + recursive SGAP zoom   by detector score
- False: Direct raw det patches           - False: Permanent pruning or Variant B    - Abs: Retain centroid
  retained by detector score                naive absent centroid                      ranked by SigLIP
       │                                        │                                        │
       └────────────────────────────────────────┴────────────────────────────────────────┘
                                                │
                                                ▼
Stage 5: Single-Crop Resolution Screening
       ├── Synthetic canvas synthesis & sub-canvas stitching PERMANENTLY PURGED (~180 lines removed)
       ├── IoS containment merge (>= 0.95) & Spatial IoU NMS (0.60) & Atomic sibling refusion
       └── Native 1.25x context-padded crops (is_already_expanded = True)
                                                │
                                                ▼
Stage 6: Batched Foveated Reader
       ├── Uses 1.25x crop directly (NO double-expansion to 1.5625x)
       └── Vectorized VLM reading on clean single crops -> consensus answer voting
```

---

## 3. Critical Bug Fixes Implemented

### 🐛 Problem 1: Autonomous SAM3 Feature Extraction (Zero-Feature Bug)
- **Issue**: In `be05f4a`, SAM3 features were only cached as a side-effect of Fast Path. When Fast Path was disabled, `feature_cache = {}`, causing Stage 3 to fall back to `np.zeros((256, 72, 72))`. On zero features, SGAP produced identical, content-blind quadrant splits, and Feature Grouping cosine similarity broke. Furthermore, Latent Re-partitioning (`MSSearch.py:512`) passed `{}` on `parent_crop`, forcing child sub-patches to also use zero features.
- **Fix**:
  1. `MSSearch.py:426`: Extracted root SAM3 vision features into `feature_cache` before Stage 3 when Fast Path is off.
  2. `MSSearch.py:551`: Extracted fine vision features on `parent_crop` before child partitioning.
  3. `stage3_partitioning.py`: Added dynamic on-the-fly extraction fallback if cache is missing.

### 🐛 Problem 5: Double 1.25× Context Expansion (Over-Cropping to 1.5625×)
- **Issue**: Stage 5 (`crops_only`) padded `patch.box` by $1.25\times$ to produce `padded_box`. Stage 6 called `expand_bounding_box(item.box, ..., 1.25)` again, resulting in $1.25 \times 1.25 = 1.5625\times$ context (+56.25% peripheral noise).
- **Fix**: Added `is_already_expanded: bool = True` to `CandidateCrop`. Stage 6 checks this flag and uses `item.crop_image` and `item.box` directly. Stage 6 only applies expansion if unexpanded raw inputs are passed (defensive fallback for ablations).

### 🐛 Problem 8: Confidence Score Miscalibration for Latent Child Sub-Patches
- **Issue**: In `MSSearch.py`, confirmed detections inside high-latent child sub-patches were assigned unboosted score $\in [0.30, 1.00]$, ranking below empty background tiles ($0.50$–$1.50$) in Stage 5 global sorting.
- **Fix**: In `MSSearch.py:634`, calibrated child patch scores to:
  $$\text{confidence\_score} = 2.0 + \text{parent\_latent\_score} + \text{best\_det\_score} \in [3.05, 4.00]$$
  Guarantees verified zoomed targets rank at the very top.

### 🐛 Silent VLM Existence Check Bypass (Safeguard B Bug)
- **Issue**: `stage4_grouping_gating.py:291` checked `if bypass_vlm_for_confirmed_groups and max_det_score >= 0.35: has_active_match = True`. Since any detected group by definition had `score >= 0.35`, Safeguard B auto-confirmed every detected group before the existence branch was reached. The VLM existence check **never ran on a single sample**, rendering Existence Scanning ON and OFF identical.
- **Fix**: Set `bypass_vlm_for_confirmed_groups = False` by default in `config.py` and `stage4_grouping_gating.py:291`. When Existence Scanning is ON, detected groups are genuinely verified by Qwen.

### 🐛 Typing & Module Cleanup
- Added missing `Tuple` import to `mssearch/MSSearch.py:16`.
- Cleaned `__all__` in `mssearch/stages/stage5_screening.py` to match exported symbols.

---

## 4. Retention Budget & Plasticity ($N$-Scaling Semantics)

- **Base Budget**: `max_retained_patches = 20` uniformly across all components.
- **$N$-Linear Scaling** (`scale_budgets_by_num_images = True`):
  - $N=1$: runtime cap = **20 crops** ($20 \times 1$)
  - $N=5$: runtime cap = **100 crops** ($20 \times 5$)
  - $N=10$: runtime cap = **200 crops** ($20 \times 10$)
  - $N=20$: runtime cap = **400 crops** ($20 \times 20$)
- **Preserved Dynamic Elasticity**:
  - `enable_dynamic_retention_elasticity = True` in `config.py`.
  - Tied candidate patches at the cutoff score (quantized $\varepsilon = 10^{-4}$) are dynamically retained, protecting multi-image recall.
  - Empirical finding: in `qwen_fp_sgap_cg_single`, elasticity rescued **71 correct images** and **24 exact text answers** that would have been severed by a strict hard cutoff.
- **Explicit SigLIP Ranking**:
  - For unclustered rows (`qwen_rigid_topk`, `qwen_sgap_topk`), patches are ranked globally by `p.siglip_score`, and each retained patch is assigned `p.confidence_score = float(p.siglip_score)`.
  - In latent re-partitioning, `latent_queue` sorts absent groups by `max_siglip_score` before the budget cap.

---

## 5. Refined 11-Component Paired Ablation Suite

All legacy names (`qwen_fp_*`, `qwen_fast_path`, `_topk` aliases) have been purged. The suite strictly consists of:

| # | `--component` Key | Partition | Grouping | Existence Scanning (Detected) | Latent Re-partitioning (Absent) | Scientific Purpose |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | `qwen` | Baseline | Off | Off | Off | Direct zero-shot VLM on gallery |
| **2** | `qwen_rigid_topk` | Rigid | Off | Off | Off | Quadtree baseline (SigLIP ranked) |
| **3** | `qwen_sgap_topk` | SGAP | Off | Off | Off | Superpixel baseline (SigLIP ranked) |
| **4** | `qwen_rigid_fg_det_abs` | Rigid | **On** | **Off** *(Trust detector)* | **Off** *(Variant B: Naive centroid)* | Raw detector + naive absent baseline |
| **5** | `qwen_sgap_fg_det_abs` | SGAP | **On** | **Off** *(Trust detector)* | **Off** *(Variant B: Naive centroid)* | Raw detector + naive absent baseline |
| **6** | `qwen_rigid_fg_exist_abs` | Rigid | **On** | **ON** *(VLM verified)* | **Off** *(Variant B: Naive centroid)* | **Isolates Existence Scanning** |
| **7** | `qwen_sgap_fg_exist_abs` | SGAP | **On** | **ON** *(VLM verified)* | **Off** *(Variant B: Naive centroid)* | **Isolates Existence Scanning** |
| **8** | `qwen_rigid_fg_det_latent`| Rigid | **On** | **Off** *(Trust detector)* | **ON** *(VLM latent zoom)* | **Isolates Latent Re-partitioning** |
| **9** | `qwen_sgap_fg_det_latent`| SGAP | **On** | **Off** *(Trust detector)* | **ON** *(VLM latent zoom)* | **Isolates Latent Re-partitioning** |
| **10** | `qwen_rigid_full` | Rigid | **On** | **ON** *(VLM verified)* | **ON** *(VLM latent zoom)* | Full Rigid Single-Crop Method |
| **11** | `qwen_sgap_full` *(Ours)* | SGAP | **On** | **ON** *(VLM verified)* | **ON** *(VLM latent zoom)* | Full SGAP Single-Crop Method |

---

## 6. Streamlined Stage 1 Production Prompt

Purged `target_entity`, `anchor_entity`, and `is_relational` from the Stage 1 MLLM prompt schema:

```text
System:
You are an expert visual search parser. Given a question, extract search targets and return a JSON object with:
- "active_objects": list of key visual search entities or text needed to answer the question
- "question_type": one of ['single_object_attribute', 'counting', 'text_reading', 'general']
Return ONLY valid JSON.

User: Question: "What is the registration number of the boat?"
Assistant: {"active_objects": ["boat"], "question_type": "single_object_attribute"}

User: Question: "What is the color of the mailbox?"
Assistant: {"active_objects": ["mailbox"], "question_type": "single_object_attribute"}

User: Question: "How many windows are on the brick building?"
Assistant: {"active_objects": ["brick building", "window"], "question_type": "counting"}

User: Question: "{question}"
```

---

## 7. Standalone Tools Added

1. **`experiments/our/scripts/rerun_fastpath_samples.py`**:
   - Local CLI tool to re-evaluate samples affected by Fast Path (`waterfall_cascade`) and splice them into repaired `.jsonl` files.
2. **`experiments/our/scripts/compare_stage1_prompts.py`** (branch `compare/stage1`):
   - Compares old relational vs. new stripped Stage 1 prompt outputs across the benchmark (`--all` or interactive single query).
   - Computes Set Exact Match %, Strict Match %, Target Match %, and latency speedup.

---

## 8. Verification & File Constraints Status

- **Runner File Length Discipline**: All evaluation runners strictly remain **$< 250$ lines**:
  - `experiments/our/eval_runner.py`: 230 lines
  - `experiments/our/ablations/eval_runner_ablation.py`: 214 lines
  - `experiments/our/scripts/modal/run_our_modal.py`: 238 lines
  - `experiments/our/ablations/scripts/run_ablation_modal.py`: 243 lines
  - `experiments/our/eval_metrics.py`: 205 lines
  - `experiments/our/ablations/scripts/rerun_fastpath_samples.py`: 247 lines
- **Unit Tests**:
  - `test_component_ablation_config.py` $\to$ **10/10 tests passed (OK)**.
  - `test_compare_stage1_prompts.py` (on `compare/stage1`) $\to$ **1/1 test passed (OK)**.
- **Git Commit State**:
  - Main changes pushed to `feat/rename-stages` and `feat/rename_branches` (`ab11a40` / `969d211` / `46a2710`).
  - Stage 1 comparison tool committed to `compare/stage1` (`752a1ed`).
