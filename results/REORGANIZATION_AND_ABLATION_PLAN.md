# STASH-Search (MSSearch V2) Reorganization & Ablation Plan

## 1. Executive Summary & Empirical Findings
From benchmark evaluation across $N=5$ image candidate galleries:
1. **Canvas Aggregation Degrades Accuracy**: Multi-patch canvas synthesis (Continuous Enclosure and Spatial Compressed Canvases) consistently hurt accuracy across all configurations (ASP: $51.49\% \to 55.06\%$, Rigid: $52.38\% \to 54.46\%$, SGAP: $51.49\% \to 55.06\%$).
2. **Fast Path Degrades Accuracy & Truncates Galleries**: Fast path drops text accuracy to $29.46\%$ and prematurely strips candidate images from multi-image pools before visual search.
3. **ASP vs SGAP Equivalence**: With Single Crop, ASP and SGAP achieve identical accuracy ($55.06\%$ Acc_text, $69.05\%$ Acc_image).
4. **Relational Intent Removal**: Since canvas aggregation is eliminated, relational questions are solved as direct visual search queries on single crops. Relational parsing in Stage 1 is disabled.

---

## 2. Main Method Architecture (4 Core Modules)

```
Pipeline Flow:
[Question + Candidate Images (N)]
       │
       ▼
Stage 1: Question Parsing (enable_relational_parsing=False)
       ├── Strips relational few-shots and schema instructions
       └── Extracts active_objects / target_entity; forces is_relational = False
       │
       ├─► [Stage 2 Fast Path: BYPASSED (enable_fast_path=False)]
       │
       ▼
Module 1: Image Partitioning (Stage 3)
       ├── Default: partition_mode = "sgap_only"
       └── Produces: BoundedPatches with cached visual embeddings
       │
       ▼
Module 2: Feature Grouping (Stage 4)
       ├── Master Toggle: enable_feature_grouping = True
       └── Clusters patches into:
           ├── Detected Groups (SAM3 found candidate objects)
           └── Absent Groups (SAM3 found no target)
       │
       ├────────────────────────────────────────┬────────────────────────────────────────┐
       ▼                                        ▼                                        ▼
Module 3: Existence Scanning               Module 4: Latent Re-partitioning           Top-k Fallbacks
(enable_existence_scanning: bool)         (enable_latent_repartition: bool)          (Ablations)
- True: VLM existence probe on best det   - True: VLM latent probe on centroid       - Det: Retain top-100
  crop (conf >= 0.70)                       (conf >= 0.70) + recursive SGAP split.     patches by det score
- False: Direct top-100 det patches       - False: Direct top-100 absent centroids   - Abs: Retain top-100
  retained by detector score                retained by SigLIP score                   centroids by SigLIP
       │                                        │                                        │
       └────────────────────────────────────────┴────────────────────────────────────────┘
                                                │
                                                ▼
Stage 5: Single-Crop Resolution Screening
       ├── Sub-canvas continuous enclosure & canvas stitching permanently bypassed
       ├── IoS containment merge (>= 0.95) & Spatial IoU NMS (0.60)
       └── Native 1.25x context-padded crops (crops_only mode)
                                                │
                                                ▼
Stage 6: Batched Foveated Reader
       └── Vectorized VLM reading on clean single crops -> consensus voting
```

---

## 3. Full 10-Step Paired Ablation Component Matrix

Budget default: $k=100$.

| # | `--component` Key | Partition | Grouping | Existence Scanning | Latent Re-part | Candidate Pool Retention ($k=100$) |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- |
| 0 | `qwen` | Baseline | Off | Off | Off | Direct zero-shot baseline *(already run)* |
| 1 | `qwen_rigid_topk` | Rigid | Off | Off | Off | Top-100 patches ranked globally by SigLIP |
| 2 | `qwen_sgap_topk` | SGAP | Off | Off | Off | Top-100 patches ranked globally by SigLIP |
| 3 | `qwen_rigid_fg_det_abs_topk` | Rigid | **On** | Off | Off | Top-100 det (det score) + Top-100 abs (SigLIP) |
| 4 | `qwen_sgap_fg_det_abs_topk` | SGAP | **On** | Off | Off | Top-100 det (det score) + Top-100 abs (SigLIP) |
| 5 | `qwen_rigid_fg_exist_abs_topk` | Rigid | **On** | **On** | Off | Existence scanned det + Top-100 abs (SigLIP) |
| 6 | `qwen_sgap_fg_exist_abs_topk` | SGAP | **On** | **On** | Off | Existence scanned det + Top-100 abs (SigLIP) |
| 7 | `qwen_rigid_fg_det_latent_repart` | Rigid | **On** | Off | **On** | Top-100 det + Latent re-partitioned abs |
| 8 | `qwen_sgap_fg_det_latent_repart` | SGAP | **On** | Off | **On** | Top-100 det + Latent re-partitioned abs |
| 9 | `qwen_rigid_full` | Rigid | **On** | **On** | **On** | Full Rigid Pipeline (Single crop, no fast path) |
| 10 | `qwen_sgap_full` *(Ours)* | SGAP | **On** | **On** | **On** | Full SGAP Pipeline (Single crop, no fast path) |

---

## 4. Standalone Fast-Path Re-run Tool (`rerun_fastpath_samples.py`)
- **Target**: Exactly the 77 samples where `record["route"] == "waterfall_cascade"`.
- **Method**: Local PyTorch execution with `enable_fast_path = False`.
- **Output**: `<input_stem>_repaired.jsonl`, keeping telemetry schema intact and reporting before/after metrics.
