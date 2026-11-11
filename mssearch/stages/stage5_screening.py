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
    merge_intervals,
    solve_water_filling_allocation,
    compute_box_distance,
    extract_joint_interaction_crops,
    create_spatial_compressed_canvas,
    compute_box_ios,
    merge_nested_containment_boxes,
    compute_box_iou,
    merge_relational_proximity_boxes,
    apply_spatial_nms,
    check_is_cross_object_llm,
    is_relational_question,
    are_seam_adjacent_siblings,
    perform_atomic_sibling_refusion,
    partition_continuous_enclosure_subcanvases,
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

    # 2. Candidate Pool Aggregation: Crops Only (Single-Object) vs. Spatial Layout (Cross-Object)
    enable_relational = getattr(config, "enable_relational_parsing", False) if config else False
    if not enable_relational:
        is_relational = False
        is_cross_object = False
    elif is_relational is None and is_cross_object is not None:
        is_relational = is_cross_object
    elif is_relational is None:
        is_relational = is_relational_question(active_objects, question=question, zoom_model=zoom_model)

    enable_compressed_canvas = getattr(config, "enable_spatial_compressed_canvas", False) if config else False
    enable_cont_enc = getattr(config, "enable_continuous_enclosure_subcanvas", False) if config else False
    relational_only = getattr(config, "spatial_canvas_relational_only", True) if config else True

    if not (enable_compressed_canvas or enable_cont_enc):
        use_layout = False
    elif relational_only:
        use_layout = (enable_compressed_canvas or enable_cont_enc) and is_relational
    else:
        use_layout = (enable_compressed_canvas or enable_cont_enc)
    gutter_px = getattr(config, "spatial_compression_gutter_px", 28) if config else 28
    max_canvas_dim = getattr(config, "max_canvas_dimension_px", 1024) if config else 1024
    enable_jic = getattr(config, "enable_joint_interaction_crop", True) if config else True
    jic_prox = getattr(config, "jic_proximity_threshold_px", 400.0) if config else 400.0
    jic_max_dim = getattr(config, "jic_max_dimension_px", 1024) if config else 1024
    enable_dynamic_gutters = getattr(config, "enable_dynamic_gutters", True) if config else True
    continuity_thresh = getattr(config, "spatial_continuity_threshold_px", 2) if config else 2
    enable_water_filling = getattr(config, "enable_water_filling_allocation", True) if config else True
    target_max_native_dim = getattr(config, "target_max_native_dim_px", 512) if config else 512

    # Extract final crop bounding boxes before aggregation for re-visualization
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
    stage5_mode_label = "layout"

    if use_layout:
        enable_cont_enc = getattr(config, "enable_continuous_enclosure_subcanvas", True) if config else True
        if enable_cont_enc:
            stage5_mode_label = "continuous_enclosure"
            if verbose:
                print(f"[Stage 5][Continuous Enclosure Sub-Canvas] Cross-object question (relational=True): Synthesizing Continuous Enclosure Sub-Canvases (ceiling <= {max_canvas_dim}px).")
            by_img: Dict[int, List[BoundedPatch]] = {}
            for p in effective_patches:
                by_img.setdefault(p.img_idx, []).append(p)

            for img_idx, img_patches in by_img.items():
                img_pil = candidate_images[img_idx]
                img_w, img_h = img_pil.size

                # Step 1: Atomic Sibling Re-Fusion
                seam_thresh = getattr(config, "seam_overlap_threshold", 0.10) if config else 0.10
                atomic_candidates = perform_atomic_sibling_refusion(
                    img_patches, seam_overlap_threshold=seam_thresh
                )

                # Step 2 & 3: Continuous Enclosure Windowing & Sub-Canvas Dispatch
                clusters = partition_continuous_enclosure_subcanvases(
                    atomic_candidates, (img_w, img_h), max_dim=max_canvas_dim
                )

                # Step 4: Extract Continuous Photographic Crops directly from source image
                for cl in clusters:
                    ux1 = min(p.box[0] for p in cl)
                    uy1 = min(p.box[1] for p in cl)
                    ux2 = max(p.box[2] for p in cl)
                    uy2 = max(p.box[3] for p in cl)

                    ux1 = max(0, min(img_w, ux1))
                    uy1 = max(0, min(img_h, uy1))
                    ux2 = max(0, min(img_w, ux2))
                    uy2 = max(0, min(img_h, uy2))

                    union_box = (ux1, uy1, ux2, uy2)
                    crop_img = img_pil.crop(union_box)

                    all_orig_boxes: List[Tuple[int, int, int, int]] = []
                    for p in cl:
                        if hasattr(p, "constituent_boxes") and p.constituent_boxes:
                            all_orig_boxes.extend(p.constituent_boxes)
                        else:
                            all_orig_boxes.append(p.box)

                    remapped = [
                        (
                            max(0, b[0] - ux1),
                            max(0, b[1] - uy1),
                            min(crop_img.width, b[2] - ux1),
                            min(crop_img.height, b[3] - uy1)
                        )
                        for b in all_orig_boxes
                    ]
                    max_sc = max((p.confidence_score if p.confidence_score is not None else 1.0) for p in cl)

                    unified_pool.append(
                        CandidateCrop(
                            img_idx=img_idx,
                            box=union_box,
                            crop_image=crop_img,
                            score=max_sc,
                            source_branch="ContinuousEnclosure",
                            is_compressed_canvas=True,
                            remapped_boxes=remapped,
                            original_boxes=all_orig_boxes
                        )
                    )

            unified_pool.sort(key=lambda c: c.score, reverse=True)
            max_subcanvases = getattr(config, "max_sub_canvases", 4) if config else 4
            unified_pool = unified_pool[:max_subcanvases]
        else:
            if verbose:
                print(f"[Stage 5][Layout Mode] Cross-object question (relational=True): Synthesizing Tri-Pillar spatial canvas layout.")
            by_img: Dict[int, List[BoundedPatch]] = {}
            for p in effective_patches:
                by_img.setdefault(p.img_idx, []).append(p)

            for img_idx, img_patches in by_img.items():
                img_pil = candidate_images[img_idx]
                # Pillar 1: Proximity-Graph Joint Interaction Crop (PG-JIC)
                is_all_one_jic = False
                if enable_jic and len(img_patches) > 1:
                    img_patches, is_all_one_jic = extract_joint_interaction_crops(
                        img_patches,
                        img_pil.size,
                        prox_threshold=jic_prox,
                        max_dim=jic_max_dim
                    )

                if is_all_one_jic and len(img_patches) == 1:
                    p = img_patches[0]
                    box_ints = tuple(int(round(float(c))) for c in p.box)
                    crop = img_pil.crop(box_ints)
                    unified_pool.append(
                        CandidateCrop(
                            img_idx=img_idx,
                            box=box_ints,
                            crop_image=crop,
                            score=p.confidence_score if p.confidence_score is not None else 1.0,
                            source_branch="JointInteractionCrop",
                            is_compressed_canvas=False,
                            remapped_boxes=[(0, 0, crop.width, crop.height)],
                            original_boxes=[box_ints]
                        )
                    )
                    if verbose:
                        print(
                            f"[Stage 5][PG-JIC Passthrough] Image {img_idx}: Proximate components merged into single "
                            f"{crop.width}x{crop.height} un-split crop (bypassed canvas synthesis, 0px seam cuts)."
                        )
                elif len(img_patches) == 1:
                    p = img_patches[0]
                    box_ints = tuple(int(round(float(c))) for c in p.box)
                    crop = img_pil.crop(box_ints)
                    if crop.width >= 4 and crop.height >= 4:
                        unified_pool.append(
                            CandidateCrop(
                                img_idx=img_idx,
                                box=box_ints,
                                crop_image=crop,
                                score=p.confidence_score if p.confidence_score is not None else 1.0,
                                source_branch="Direct",
                                is_compressed_canvas=False,
                                remapped_boxes=[(0, 0, crop.width, crop.height)],
                                original_boxes=[box_ints]
                            )
                        )
                else:
                    # Pillars 2 & 3: Water-Filling Resolution Allocation & Dynamic Gutters
                    boxes = [tuple(int(round(float(c))) for c in p.box) for p in img_patches]
                    box_roles = [getattr(p, "semantic_role", None) for p in img_patches]
                    canvas_img, remapped_boxes = create_spatial_compressed_canvas(
                        img_pil,
                        boxes,
                        gutter_px=gutter_px,
                        box_roles=box_roles,
                        max_canvas_dim=max_canvas_dim,
                        enable_dynamic_gutters=enable_dynamic_gutters,
                        continuity_thresh=continuity_thresh,
                        enable_water_filling=enable_water_filling,
                        target_max_native_dim=target_max_native_dim
                    )
                    union_box = (
                        min(b[0] for b in boxes),
                        min(b[1] for b in boxes),
                        max(b[2] for b in boxes),
                        max(b[3] for b in boxes)
                    )
                    max_score = max(
                        (p.confidence_score if p.confidence_score is not None else 1.0)
                        for p in img_patches
                    )
                    unified_pool.append(
                        CandidateCrop(
                            img_idx=img_idx,
                            box=union_box,
                            crop_image=canvas_img,
                            score=max_score,
                            source_branch="SpatialCompressedCanvas",
                            is_compressed_canvas=True,
                            remapped_boxes=remapped_boxes,
                            original_boxes=boxes
                        )
                    )
                    if verbose:
                        print(
                            f"[Stage 5][Tri-Pillar Canvas] Image {img_idx}: Compressed {len(img_patches)} patches into "
                            f"{canvas_img.size[0]}x{canvas_img.size[1]} canvas (WF-ARA & Dynamic Gutters, max_dim<={max_canvas_dim})."
                        )
    else:
        stage5_mode_label = "crops_only"
        seam_thresh = getattr(config, "seam_overlap_threshold", 0.10) if config else 0.10
        healed_patches = perform_atomic_sibling_refusion(
            effective_patches, seam_overlap_threshold=seam_thresh
        )

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
                    original_boxes=orig_boxes
                )
            )
        if verbose:
            print(f"[Stage 5][Crops Only] Single-object question (relational=False): Bypassed canvas layout, prepared {len(unified_pool)} native-resolution crops with 1.25x padding.")

    pool = CandidatePool(
        unified_pool,
        pre_aggregation_boxes=pre_aggregation_boxes,
        is_cross_object=bool(is_relational),
        is_relational=bool(is_relational),
        stage5_mode=stage5_mode_label
    )

    if verbose:
        print(f"[Stage 5] Unified Candidate Pool P aggregated: {len(pool)} total items ready for foveated VLM reading (mode={pool.stage5_mode}, is_cross_object={pool.is_cross_object}).")

    return pool
