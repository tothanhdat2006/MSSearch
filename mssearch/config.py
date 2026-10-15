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
    use_nlp_fallback: bool = True
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
    min_patch_entropy: float = 0.50              # Minimum entropy required to retain patch (H < min_patch_entropy -> pruned)
    entropy_weights: Tuple[float, float, float] = (0.35, 0.35, 0.30)  # Shannon, Sobel, Laplacian
    partition_mode: str = "hybrid"               # "hybrid", "quadrant_only", "sgap_only"
    sgap_min_splits: int = 4                     # Default 4 splits (aligned with CVSearch)
    sgap_max_splits: int = 8                     # Default 8 splits (aligned with CVSearch)
    sgap_n_atoms: int = 100
    quadrant_overlap_ratio: float = 0.10         # 10% safety border expansion
    sgap_overlap_ratio: float = 0.10             # 10% safety border expansion
    max_patch_size: int = 1024                   # Recursive invariant: longer side <= 1024px
    max_partition_depth: int = 2                 # Maximum recursion depth before direct tiling
    max_patches_per_image: int = 64              # Maximum bounded patches budget per image
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
    max_vlm_tokens: int = 256
    answer_dedup_similarity: float = 0.85        # Semantic deduplication threshold

    # -------------------------------------------------------------------------
    # Diagnostic & Debug Logging
    # -------------------------------------------------------------------------
    verbose: bool = True                         # Print clear debug logs on fallback / non-fallback decisions

    # -------------------------------------------------------------------------
    # Backward Compatibility Aliases & Ingestion
    # -------------------------------------------------------------------------
    fast_path_enabled: Optional[bool] = None
    feature_sim_threshold: Optional[float] = None
    siglip_sim_threshold: Optional[float] = None
    expansion_factor: Optional[float] = None

    def __post_init__(self):
        # Synchronize fast path toggle
        if self.fast_path_enabled is not None:
            self.enable_fast_path = self.fast_path_enabled
        else:
            self.fast_path_enabled = self.enable_fast_path

        # Synchronize feature grouping similarity threshold
        if self.feature_sim_threshold is not None:
            self.grouping_sim_threshold = self.feature_sim_threshold
        else:
            self.feature_sim_threshold = self.grouping_sim_threshold

        # Synchronize SigLIP screening threshold
        if self.siglip_sim_threshold is not None:
            self.branch_b_score_threshold = self.siglip_sim_threshold
        else:
            self.siglip_sim_threshold = self.branch_b_score_threshold

        # Synchronize bounding box expansion factor
        if self.expansion_factor is not None:
            self.bbox_expansion_factor = self.expansion_factor
        else:
            self.expansion_factor = self.bbox_expansion_factor
