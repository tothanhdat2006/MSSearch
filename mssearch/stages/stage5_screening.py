"""
Stage 5: Multi-Modal Dynamic Adaptive Fusion & Resolution Screening.
Selects and formats final candidate crops/canvases for Stage 6 foveated reading.
"""

from typing import List, Tuple, Optional, Dict, Any
from PIL import Image

from ..config import MSSearchConfig
from .stage3_partitioning import BoundedPatch
from .utils_stage5 import (
    CandidateCrop,
    CandidatePool,
    merge_nested_containment_boxes,
    apply_spatial_nms,
    perform_atomic_sibling_refusion,
)

__all__ = [
    "CandidateCrop",
    "CandidatePool",
    "run_resolution_screening",
    "create_spatial_compressed_canvas",
    "merge_intervals",
    "solve_water_filling_allocation",
    "extract_joint_interaction_crops",
    "compute_box_ios",
    "merge_nested_containment_boxes",
    "compute_box_iou",
    "compute_box_distance",
    "merge_relational_proximity_boxes",
    "apply_spatial_nms",
    "check_is_cross_object_llm",
    "is_relational_question",
    "are_seam_adjacent_siblings",
    "perform_atomic_sibling_refusion",
    "partition_continuous_enclosure_subcanvases",
]


def run_resolution_screening(
    remaining_patches: List[BoundedPatch],
    candidate_images: List[Image.Image],
    active_objects: Optional[List[str]] = None,
    siglip_model = None,
    config: Optional[MSSearchConfig] = None,
    question: str = "",
    is_relational: Optional[bool] = None,
    zoom_model = None,
    is_cross_object: Optional[bool] = None,
    **kwargs
) -> List[CandidateCrop]:
    """
    Aggregates surviving candidate patches into the Unified Candidate Pool P:
    1. Applies Stage 5 Intersection-over-Self (IoS >= 0.95) containment merging to drop coarse parents.
    2. Applies Stage 5 Spatial IoU NMS to suppress overlapping patches per image.
    3. Dual-mode output aggregation:
       - Single-Object Question (relational=False): Uses native-resolution crops only.
       - Cross-Object Question (relational=True): Synthesizes Spatial-Awareness Compressed Canvas or
         Continuous Enclosure Sub-Canvases.
    """
    if not remaining_patches:
        return CandidatePool([], pre_aggregation_boxes=[])

    effective_patches = list(remaining_patches)
    verbose = getattr(config, "verbose", True) if config else True

    # 0. Stage 5 Intersection-over-Self (IoS) Containment Merging (>= 95% Coverage)
    enable_ios = getattr(config, "enable_ios_containment_merging", True) if config else True
    if enable_ios:
        ios_thresh = getattr(config, "ios_containment_threshold", 0.95) if config else 0.95
        effective_patches = merge_nested_containment_boxes(
            effective_patches,
            containment_threshold=ios_thresh,
            verbose=verbose
        )

    # 1. Stage 5 Spatial IoU NMS Deduplication
    enable_nms = getattr(config, "enable_stage5_spatial_nms", True) if config else True
    if enable_nms:
        iou_thresh = getattr(config, "stage5_nms_iou_threshold", 0.60) if config else 0.60
        effective_patches = apply_spatial_nms(
            effective_patches,
            iou_threshold=iou_thresh,
            verbose=verbose
        )

    # 2. Extract final crop bounding boxes before aggregation for telemetry
    pre_aggregation_boxes: List[Dict[str, Any]] = [
        {
            "img_idx": int(p.img_idx),
            "box": [int(round(float(c))) for c in p.box],
            "score": float(p.confidence_score) if p.confidence_score is not None else 1.0,
            "source_branch": str(getattr(p, "source_branch", "Direct")),
            "entropy": float(getattr(p, "entropy", 0.0)),
        }
        for p in effective_patches
    ]

    unified_pool: List[CandidateCrop] = []
    stage5_mode_label = "crops_only"

    # 3. Atomic Sibling Refusion along seams
    seam_thresh = getattr(config, "seam_overlap_threshold", 0.10) if config else 0.10
    healed_patches = perform_atomic_sibling_refusion(
        effective_patches, seam_overlap_threshold=seam_thresh
    )

    # 4. Sort patches by confidence score
    sorted_patches = sorted(
        healed_patches,
        key=lambda p: (
            p.confidence_score if p.confidence_score is not None else 0.0,
            getattr(p, "siglip_score", 0.0) if getattr(p, "siglip_score", None) is not None else 0.0,
            (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
        ),
        reverse=True
    )
    for patch in sorted_patches:
        box_ints = tuple(int(round(float(c))) for c in patch.box)
        bx1, by1, bx2, by2 = box_ints
        bw, bh = bx2 - bx1, by2 - by1
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        pw, ph = bw * 1.25, bh * 1.25
        img_w, img_h = candidate_images[patch.img_idx].size
        px1 = max(0, int(round(cx - pw / 2.0)))
        py1 = max(0, int(round(cy - ph / 2.0)))
        px2 = min(img_w, int(round(cx + pw / 2.0)))
        py2 = min(img_h, int(round(cy + ph / 2.0)))
        padded_box = (px1, py1, px2, py2)

        crop = candidate_images[patch.img_idx].crop(padded_box)
        cw, ch = crop.size
        if cw < 4 or ch < 4:
            continue

        orig_boxes = getattr(patch, "constituent_boxes", [box_ints]) or [box_ints]
        unified_pool.append(
            CandidateCrop(
                img_idx=patch.img_idx,
                box=padded_box,
                crop_image=crop,
                score=patch.confidence_score if patch.confidence_score is not None else 1.0,
                source_branch="Direct",
                is_compressed_canvas=False,
                remapped_boxes=[(0, 0, crop.width, crop.height)],
                original_boxes=orig_boxes,
                is_already_expanded=True,
            )
        )
    if verbose:
        print(f"[Stage 5][Single Crop] Prepared {len(unified_pool)} native-resolution crops with 1.25x padding.")

    pool = CandidatePool(
        unified_pool,
        pre_aggregation_boxes=pre_aggregation_boxes,
        is_cross_object=False,
        is_relational=False,
        stage5_mode=stage5_mode_label
    )

    if verbose:
        print(f"[Stage 5] Unified Candidate Pool P aggregated: {len(pool)} total items ready for foveated VLM reading (mode={pool.stage5_mode}, is_cross_object={pool.is_cross_object}).")

    return pool
