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
    nlp_model_name: str = "en_core_web_sm"

    # -------------------------------------------------------------------------
    # Stage 2: 2-Gate Early Assessment (Direct Global Answer & Fast Path)
    # -------------------------------------------------------------------------
    enable_direct_answer: bool = True            # Phase 1: Direct Global Answer via Yes/No logit sufficiency probing
    direct_answer_conf_threshold: float = 0.60   # Minimum sufficiency confidence c_q [-1, 1] to answer directly on root
    enable_fast_path: bool = True                # Phase 2: Tool-Based Fast Path via SAM3 Proposals
    fast_path_conf_threshold: float = 0.70       # Existence confidence threshold for fast path instant exit
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
    partition_stopping_mode: str = "saliency_plateau"  # "saliency_plateau" (Paradigm 2 dynamic semantic zoom) or "fixed_size" (legacy max_patch_size)
    max_patch_size: int = 512                    # Recursive invariant when partition_stopping_mode == "fixed_size": longer side <= 512px
    min_plateau_size: int = 512                  # Scale floor in saliency_plateau mode: patches <= this dimension stop splitting
    max_plateau_size: int = 1536                 # Physical ceiling in saliency_plateau mode: patches > this dimension must split (VRAM safety)
    saliency_plateau_gain_threshold: float = 0.04 # Saliency gain (max_child - parent) required to justify deeper split
    saliency_target_threshold: float = 0.35      # High confidence target threshold where marginal gain halts zoom
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
    latent_prompt_mode: str = "question_aware"   # "question_aware" (contextual QA prompt) or "target_object" (legacy target prompt)
    latent_prompt_template: str = "According to your common sense knowledge and the content of the image, is it possible to find a {} by further zooming the sub-patch? Answer Yes or No."
    latent_prompt_template_qa: str = "According to visual common sense and the content of the image, is it possible to find the information needed to answer '{}' by further zooming the sub-patch? Answer Yes or No."
    enable_existence_gating: bool = True         # Query VLM existence confidence to verify SAM3 detections
    existence_confidence_threshold: float = 0.0  # Minimum VLM existence confidence [-1, 1] to confirm detection
    existence_prompt_template: str = "Is there a '{}' in the image? Answer Yes or No."
    bypass_vlm_for_confirmed_groups: bool = True # Skip VLM existence check if SAM3 det >= threshold or SigLIP >= 0.45
    max_latent_evals: int = 8                    # Cap speculative VLM latent checks to top K groups (surfaced parameter)
    enable_prioritized_sam3_scanning: bool = True # Scan only high-priority candidate patches / centroids with SAM3
    max_sam3_eval_patches: int = 16              # Upper bound on expensive SAM3 patch scans in Stage 4 (drops 270s -> 35s)
    # Group Retention Granularity & Global Hard Cap
    enable_group_retention_caps: bool = True     # Cap number of member patches retained per group
    max_patches_per_latent_group: int = 1        # Max patches kept for latent-retained groups (1 = centroid only)
    max_patches_per_detected_group: int = 4      # Max patches kept for confirmed SAM3 detection groups
    enable_retention_hard_cap: bool = True       # Cap total retained patches across all groups before Stage 5/6
    max_retained_patches: int = 32               # Global upper bound on candidate patches pool
    retention_ranking_metric: str = "confidence_hybrid"  # "confidence_hybrid", "entropy", "detection_first"

    # -------------------------------------------------------------------------
    # Stage 5: Direct Unified Candidate Pool Aggregation & Spatial NMS
    # -------------------------------------------------------------------------
    siglip_model_name: str = "google/siglip-base-patch16-384"
    siglip_device: str = "cuda:0"                # Shared on GPU 0 with Qwen2.5-VL layers 0-14
    enable_stage5_spatial_nms: bool = True       # Suppress duplicate/overlapping candidate crops on same image
    stage5_nms_iou_threshold: float = 0.60       # Spatial IoU threshold for candidate patch deduplication
    enable_spatial_compressed_canvas: bool = True # Excise empty rows/cols to form spatial compressed canvas for all samples
    spatial_compression_gutter_px: int = 28      # Visual separator gap (px) between non-contiguous compressed intervals

    # -------------------------------------------------------------------------
    # Stage 6: Foveated VLM Reading, Dual-Scale & Collision-Free Consensus
    # -------------------------------------------------------------------------
    vlm_batch_size: int = 4                      # Batched VLM forward passes (4 or 8) across patches and logit probes
    bbox_expansion_factor: float = 1.25          # Independent 25% padding per patch
    max_vlm_tokens: int = 128                    # Response token budget (allows complete answers without truncation)
    enable_early_consensus_exit: bool = True     # Stop evaluating crops once 2 identical/near-identical answers are extracted
    early_consensus_count: int = 2               # Minimum consensus extractions to trigger early exit
    early_consensus_threshold: float = 0.75      # String similarity threshold on extracted core answers
    prompt_directed_final_answer: bool = True    # Instruct VLM to conclude with 'Final Answer: <concise answer>'
    answer_dedup_similarity: float = 0.85        # Semantic deduplication threshold

    # -------------------------------------------------------------------------
    # Multiple-Choice & Options Prompting Flag (e.g. HRBench)
    # -------------------------------------------------------------------------
    include_options_in_prompt: bool = False      # If True, appends candidate options block to reading prompt (like CVSearch option_single); if False, operates in open-ended free_form mode
    answer_type: str = "free_form"               # "free_form" (default) or "option_single" / "multiple_choice"

    # -------------------------------------------------------------------------
    # Multi-Source / Multi-Answer Configuration (Annotation Charter Standard)
    # -------------------------------------------------------------------------
    multi_answer_mode: bool = False              # If True, enables multi-source open-set answering (K>=1, N>=1): disables Stage 6 early consensus exit and aggregates all unique answers into semicolon-separated text
    scale_budgets_by_num_images: bool = True     # If True, scales Stage 4 SAM3 patch scans, latent evaluations, and retention caps linearly by N candidate images
    enable_query_adaptive_halting: bool = True   # Query-adaptive dynamic halting in multi-answer mode (e.g. N=1 disables multi-answer mode)
    query_meta: Optional[dict] = None            # Optional structured query metadata: {'target_attribute_type': ..., 'condition_attribute': ..., 'target_object': ...}

    def __post_init__(self):
        # In multi-answer mode, disable early consensus exit to read all evidence crops across N images
        if self.multi_answer_mode:
            self.enable_early_consensus_exit = False

    # -------------------------------------------------------------------------
    # Diagnostic & Debug Logging
    # -------------------------------------------------------------------------
    verbose: bool = True                         # Print clear debug logs on fallback / non-fallback decisions

