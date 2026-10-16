"""
Configuration and Ablation Parameters for Dual-Path Multi-Source Visual Search (MSSearch).
Strictly decoupled into a typed dataclass for comprehensive ablation testing.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


@dataclass
class MSSearchConfig:
    """Master configuration for the Dual-Path Visual Search Engine (MSSearch)."""

    # -------------------------------------------------------------------------
    # Stage 1: Question Parsing & Active Object Set
    # -------------------------------------------------------------------------
    use_mllm_object_parser: bool = True          # Use MLLM language model directly for zero-shot object extraction
    use_nlp_fallback: bool = False               # Deprecated in favor of MLLM + Grammar Regex
    nlp_model_name: str = "en_core_web_sm"

    # -------------------------------------------------------------------------
    # Stage 2: Tool-Based Fast Path Canvas Check
    # -------------------------------------------------------------------------
    enable_fast_path: bool = True
    fast_path_conf_threshold: float = 0.70
    enable_visual_prompt_bbox: bool = True       # Draw bounding box on composite crop
    enable_textual_prompt_coords: bool = False   # Feed textual [x1, y1, x2, y2] coordinates into prompt
    visual_prompt_bbox_color: str = "red"
    visual_prompt_bbox_width: int = 3

    # -------------------------------------------------------------------------
    # Stage 3: Entropy-Gated Recursive Patch Partitioning & SAM3-SGAP
    # -------------------------------------------------------------------------
    entropy_threshold: float = 0.35              # H < 0.35 -> Quadrants, H >= 0.35 -> SAM3-SGAP
    min_patch_entropy: float = 0.35              # Minimum entropy required to retain patch (H < min_patch_entropy -> pruned; 0.35 preserves smooth painted walls/signs)
    entropy_weights: Tuple[float, float, float] = (0.35, 0.35, 0.30)  # Shannon, Sobel, Laplacian
    partition_mode: str = "hybrid"               # "hybrid", "quadrant_only", "sgap_only"
    sgap_min_splits: int = 4                     # Default 4 splits (aligned with CVSearch)
    sgap_max_splits: int = 8                     # Default 8 splits (aligned with CVSearch)
    sgap_n_atoms: int = 100
    quadrant_overlap_ratio: float = 0.10         # 10% safety border expansion
    sgap_overlap_ratio: float = 0.10             # 10% safety border expansion
    max_patch_size: int = 512                    # Recursive invariant: longer side <= 512px
    max_partition_depth: int = 3                 # Maximum recursion depth before direct tiling
    max_patches_per_image: int = 100             # Maximum bounded patches budget per image
    stage3_budget_mode: str = "siglip_similarity" # "siglip_similarity", "entropy_descending", "uniform_stride"
    siglip_budget_batch_size: int = 32           # Batch size for SigLIP crop scoring
    reuse_stage2_features: bool = True           # Reuse SAM3 vision features extracted in Stage 2

    # -------------------------------------------------------------------------
    # Stage 4: SAM3 Feature Grouping & Open-Vocabulary Gating
    # -------------------------------------------------------------------------
    enable_feature_grouping: bool = True
    grouping_sim_threshold: float = 0.80         # Cosine similarity threshold for grouping patches
    reuse_stage3_features: bool = True           # Reuse patch feature vectors from Stage 3
    sam3_group_det_threshold: float = 0.35       # SAM3 confidence threshold to keep a group
    enable_latent_gating: bool = True            # Query VLM latent confidence on centroid patch if no SAM3 det
    latent_confidence_threshold: float = 0.0     # Minimum VLM latent confidence [-1, 1] to keep group (default 0.0 -> P(Yes)>=0.5)
    enable_existence_gating: bool = True         # Query VLM existence confidence to verify SAM3 detections
    existence_confidence_threshold: float = 0.0  # Minimum VLM existence confidence [-1, 1] to confirm detection
    latent_prompt_template: str = "According to your common sense knowledge and the content of the image, is it possible to find a {} by further zooming the sub-patch? Answer Yes or No."
    existence_prompt_template: str = "Is there a '{}' in the image? Answer Yes or No."
    bypass_vlm_for_confirmed_groups: bool = True # Skip VLM existence check if SAM3 det >= threshold or SigLIP >= 0.45
    max_latent_eval_groups: int = 8              # Cap speculative VLM latent checks to top 8 groups
    enable_prioritized_sam3_scanning: bool = True # Scan only high-priority candidate patches / centroids with SAM3
    max_sam3_eval_patches: int = 16              # Upper bound on expensive SAM3 patch scans in Stage 4 (drops 270s -> 35s)
    # Group Retention Granularity & Global Hard Cap
    enable_group_retention_caps: bool = True     # Cap number of member patches retained per group
    max_patches_per_latent_group: int = 1        # Max patches kept for latent-retained groups (1 = centroid only)
    max_patches_per_detected_group: int = 4      # Max patches kept for confirmed SAM3 detection groups
    enable_retention_hard_cap: bool = True       # Cap total retained patches across all groups before Stage 5/6
    max_retained_patches: int = 16               # Global upper bound on candidate patches pool
    retention_ranking_metric: str = "confidence_hybrid"  # "confidence_hybrid", "entropy", "detection_first"

    # -------------------------------------------------------------------------
    # Stage 5: Resolution-Aware Dual-Branch Screening
    # -------------------------------------------------------------------------
    branch_b_size_cutoff: int = 336              # > 336px -> Branch A (Direct), <= 336px -> Branch B
    branch_b_score_threshold: float = 0.50       # SigLIP similarity threshold to keep low-res crop
    siglip_model_name: str = "google/siglip-base-patch16-384"
    siglip_device: str = "cuda:0"                # Shared on GPU 0 with Qwen2.5-VL layers 0-14

    # -------------------------------------------------------------------------
    # Stage 6: Foveated VLM Reading & Semantic Deduplication
    # -------------------------------------------------------------------------
    bbox_expansion_factor: float = 1.25          # Independent 25% padding per patch
    max_vlm_tokens: int = 48                     # Concise response token budget (prevents verbose 256-token slowdown)
    enable_concise_reading_prompt: bool = True   # Append concise answering instruction
    enable_early_consensus_exit: bool = True     # Stop evaluating crops once 2 identical/near-identical answers are extracted
    early_consensus_count: int = 2               # Minimum consensus extractions to trigger early exit
    answer_dedup_similarity: float = 0.85        # Semantic deduplication threshold

    # -------------------------------------------------------------------------
    # Diagnostic & Debug Logging
    # -------------------------------------------------------------------------
    verbose: bool = True                         # Print clear debug logs on fallback / non-fallback decisions
