"""
Stage 3: Entropy-Gated Recursive Patch Partitioning & SAM3-SGAP.
Computes Tri-Factor Spatial Entropy H(I), routes between fast Quadrants and SAM3-SGAP,
enforces recursive bounding to max(W, H) <= 1024px with 10% overlap margins,
and reuses Stage 2 feature maps while caching pooled patch embeddings for Stage 4.
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from PIL import Image

from ..config import MSSearchConfig
from .utils_stage3 import (
    BoundedPatch,
    compute_spatial_entropy,
    split_4_subpatches,
    split_box_into_subpatches,
    partition_oversized_box,
    partition_into_quadrants,
    compute_bbox_overlap_cost,
    partition_into_sam3_sgap,
    slice_feature_map,
)

__all__ = [
    "BoundedPatch",
    "compute_spatial_entropy",
    "split_4_subpatches",
    "split_box_into_subpatches",
    "partition_oversized_box",
    "partition_into_quadrants",
    "compute_bbox_overlap_cost",
    "partition_into_sam3_sgap",
    "slice_feature_map",
    "partition_candidate_images",
]


def partition_candidate_images(
    candidate_images: List[Image.Image],
    stage2_feature_cache: Dict[int, np.ndarray],
    config: MSSearchConfig,
    active_objects: Optional[List[str]] = None,
    siglip_model: Optional[Any] = None,
    target_img_indices: Optional[List[int]] = None,
    sam_model: Optional[Any] = None,
) -> List[BoundedPatch]:
    """
    Executes Entropy-Gated Recursive Patch Partitioning across candidate images.
    If target_img_indices is provided, only partitions those specified image indices.
    Guarantees that all output patches satisfy longer side max(W, H) <= config.max_patch_size.
    Reuses cached Stage 2 feature maps and returns pooled embeddings for Stage 4 grouping.
    """
    bounded_patches: List[BoundedPatch] = []

    for img_idx, img_pil in enumerate(candidate_images):
        if target_img_indices is not None and img_idx not in target_img_indices:
            continue
        img_w, img_h = img_pil.size
        # Queue of patches to evaluate: [((x1, y1, x2, y2), depth, parent_box)]
        work_queue: List[Tuple[Tuple[int, int, int, int], int, Optional[Tuple[int, int, int, int]]]] = [
            ((0, 0, img_w, img_h), 0, None)
        ]

        # Get cached feature map or extract dynamically
        feat_map = stage2_feature_cache.get(img_idx, None)
        if feat_map is not None:
            if config.verbose:
                print(f"[Stage 3][Non-Fallback] Image {img_idx}: Reusing cached vision feature map (Shape: {feat_map.shape}).")
        elif sam_model is not None and active_objects:
            try:
                import torch
                with torch.inference_mode():
                    bb_out, _, _ = sam_model.batch_inference(img_pil, active_objects)
                    if bb_out is not None and "vision_features" in bb_out:
                        vf = bb_out["vision_features"]
                        feat_map = vf.detach().cpu().float().numpy().squeeze(0) if isinstance(vf, torch.Tensor) else np.squeeze(vf, axis=0)
                        if config.verbose:
                            print(f"[Stage 3][Dynamic Feature Extraction] Image {img_idx}: Extracted vision feature map via SAM3 (Shape: {feat_map.shape}).")
            except Exception as e:
                if config.verbose:
                    print(f"[Stage 3][Feature Warning] Dynamic feature extraction failed on image {img_idx}: {e}")
        if feat_map is None:
            det_name = type(sam_model).__name__ if sam_model is not None else "None"
            if config.partition_mode in ("sgap_only", "hybrid"):
                raise RuntimeError(
                    f"[Stage 3 Error] Image {img_idx}: SGAP partitioning ('{config.partition_mode}') requires a dense vision feature map [C, H, W], "
                    f"but detector '{det_name}' provided no 'vision_features'. "
                    f"Silent fallback to zero-initialized feature maps is strictly disabled."
                )
            if getattr(config, "enable_feature_grouping", True):
                raise RuntimeError(
                    f"[Stage 3 Error] Image {img_idx}: Feature Grouping requires pooled feature vectors from a dense vision feature map, "
                    f"but detector '{det_name}' provided no 'vision_features'. "
                    f"Silent fallback to zero-initialized feature maps is strictly disabled."
                )
        img_patches: List[BoundedPatch] = []

        while work_queue:
            box, depth, parent_box = work_queue.pop(0)
            box_ints = tuple(int(round(float(c))) for c in box)
            x1, y1, x2, y2 = box_ints
            pw, ph = x2 - x1, y2 - y1

            cached_siglip_score: Optional[float] = None
            sub_boxes: Optional[List[Tuple[int, int, int, int]]] = None
            saliency_parent_score: Optional[float] = None

            stopping_mode = getattr(config, "partition_stopping_mode", "saliency_plateau")

            if stopping_mode == "fixed_size":
                # Legacy invariant: longer side <= max_patch_size (with 15% tolerance for boundary overlap)
                should_split = (max(pw, ph) > int(config.max_patch_size * 1.15))
            else:
                # Paradigm 2: Semantic Saliency-Plateau Stopping (Self-Terminating Zoom)
                if depth == 0:
                    should_split = True
                elif max(pw, ph) > getattr(config, "max_plateau_size", 1536):
                    # Mandatory split above the physical ceiling, even if depth cap is reached.
                    should_split = True
                elif depth >= config.max_partition_depth:
                    should_split = False
                elif max(pw, ph) <= getattr(config, "min_plateau_size", 256):
                    should_split = False
                else:
                    # Dynamic scale range [min_plateau_size, max_plateau_size]: evaluate Saliency Plateau
                    candidate_sub_boxes = split_box_into_subpatches(
                        box_ints, img_w, img_h, overlap_ratio=config.quadrant_overlap_ratio
                    )
                    crop_parent = img_pil.crop(box_ints)

                    evaluated_with_siglip = False
                    if siglip_model is not None and active_objects:
                        try:
                            child_crops = [
                                img_pil.crop(tuple(int(round(float(c))) for c in sb))
                                for sb in candidate_sub_boxes
                            ]
                            all_crops = [crop_parent] + child_crops
                            scores = siglip_model.score_crops(all_crops, active_objects)
                            s_parent = float(scores[0])
                            s_children = [float(s) for s in scores[1:]]
                            s_max_child = max(s_children) if s_children else 0.0
                            delta_s = s_max_child - s_parent
                            gain_thresh = getattr(config, "saliency_plateau_gain_threshold", 0.04)
                            target_thresh = getattr(config, "saliency_target_threshold", 0.35)

                            is_plateau = (delta_s <= gain_thresh) or (s_parent >= target_thresh and delta_s <= 0.02)

                            if is_plateau:
                                if config.verbose:
                                    print(f"[Stage 3][Plateau Stop] Patch {box_ints} ({pw}x{ph}, depth={depth}) reached saliency plateau (S_parent={s_parent:.3f}, S_max_child={s_max_child:.3f}, d_S={delta_s:+.3f} <= {gain_thresh:.2f}). Terminating zoom.")
                                should_split = False
                                cached_siglip_score = s_parent
                            else:
                                if config.verbose:
                                    print(f"[Stage 3][Saliency Gain] Patch {box_ints} ({pw}x{ph}, depth={depth}) zooming deeper (S_parent={s_parent:.3f} -> S_max_child={s_max_child:.3f}, d_S={delta_s:+.3f} > {gain_thresh:.2f}).")
                                should_split = True
                                promising_sub_boxes = [
                                    sb for sb, sc in zip(candidate_sub_boxes, s_children)
                                    if sc >= (s_parent - 0.05) or sc == s_max_child
                                ]
                                sub_boxes = promising_sub_boxes if promising_sub_boxes else candidate_sub_boxes
                            evaluated_with_siglip = True
                            saliency_parent_score = s_parent
                        except Exception as e:
                            print(f"[Stage 3][Plateau Notice] SigLIP scoring failed: {e}")

                    if not evaluated_with_siglip:
                        raise RuntimeError(f"SigLIP saliency-plateau scoring unavailable; entropy fallback disabled (active_objects={active_objects}, siglip_model={siglip_model})")

            if should_split:
                crop = img_pil.crop(box_ints)
                entropy = compute_spatial_entropy(crop, weights=config.entropy_weights)

                # Apply entropy-based routing to every patch requiring subdivision.
                should_run_sgap = (
                    config.partition_mode in ("hybrid", "sgap_only") and
                    (config.partition_mode == "sgap_only" or entropy >= config.entropy_threshold)
                )

                if should_run_sgap:
                    sub_feat, _ = slice_feature_map(feat_map, box_ints, img_w, img_h)
                    if config.verbose:
                        print(f"[Stage 3][Routing] Root Canvas {box_ints} (size {pw}x{ph}) -> H={entropy:.3f} >= {config.entropy_threshold:.2f} -> SAM3-SGAP Partitioning.")
                    sub_boxes = partition_into_sam3_sgap(
                        sub_feat,
                        box_ints,
                        img_w,
                        img_h,
                        min_splits=config.sgap_min_splits,
                        max_splits=config.sgap_max_splits,
                        overlap_ratio=config.sgap_overlap_ratio,
                        verbose=config.verbose,
                    )
                    if saliency_parent_score is not None and sub_boxes:
                        # The SGAP operator replaces regular candidate children.
                        # Reapply the original SigLIP relevance filter to SGAP crops.
                        sgap_crops = [img_pil.crop(sb) for sb in sub_boxes]
                        sgap_scores = list(siglip_model.score_crops(sgap_crops, active_objects))
                        if len(sgap_scores) != len(sub_boxes):
                            raise RuntimeError("SigLIP returned incomplete SGAP child scores")
                        highest = max(sgap_scores)
                        sub_boxes = [sb for sb, score in zip(sub_boxes, sgap_scores)
                                     if score >= saliency_parent_score - 0.05 or score == highest]
                elif sub_boxes is None:
                    sub_boxes = split_box_into_subpatches(
                        box_ints,
                        img_w,
                        img_h,
                        overlap_ratio=config.quadrant_overlap_ratio,
                    )
                    if config.verbose and depth <= 1:
                        print(f"[Stage 3][Routing] Patch {box_ints} (size {pw}x{ph}, depth={depth}) -> Direction-aware Split ({len(sub_boxes)} patches).")

                # A04: a semantic partition can return a parent-sized region. Such
                # a candidate would prevent the mandatory >1536px rule from terminating.
                # Revert to regular four-way partitioning for this node (not an
                # entropy-scoring fallback) and enforce strict geometric progress.
                parent_area = max(1, pw * ph)
                if not sub_boxes or any(
                    (int(round(float(sb[2]))) - int(round(float(sb[0])))) *
                    (int(round(float(sb[3]))) - int(round(float(sb[1])))) >= parent_area
                    for sb in sub_boxes
                ):
                    sub_boxes = split_box_into_subpatches(
                        box_ints, img_w, img_h, overlap_ratio=config.quadrant_overlap_ratio
                    )
                for sb in sub_boxes:
                    sb_ints = tuple(int(round(float(c))) for c in sb)
                    child_area = (sb_ints[2] - sb_ints[0]) * (sb_ints[3] - sb_ints[1])
                    if child_area <= 0 or child_area >= parent_area:
                        raise RuntimeError(f"Non-shrinking partition at {box_ints} -> {sb_ints}")
                    work_queue.append((sb_ints, depth + 1, box_ints))
            else:
                crop = img_pil.crop(box_ints)
                entropy = compute_spatial_entropy(crop, weights=config.entropy_weights)

                if config.min_patch_entropy > 0.0 and entropy < config.min_patch_entropy:
                    if cached_siglip_score is None or cached_siglip_score < getattr(config, "saliency_target_threshold", 0.35):
                        continue

                _, pooled_vec = slice_feature_map(feat_map, box_ints, img_w, img_h) if feat_map is not None else (None, np.zeros((0,), dtype=np.float32))

                img_patches.append(
                    BoundedPatch(
                        img_idx=img_idx,
                        box=box_ints,
                        feature_vector=pooled_vec,
                        entropy=entropy,
                        siglip_score=cached_siglip_score,
                        parent_id=parent_box,
                        level=depth,
                    )
                )


        # Safety Budget Cap
        if len(img_patches) > config.max_patches_per_image:
            budget_mode = getattr(config, "stage3_budget_mode", "siglip_similarity")
            if budget_mode == "siglip_similarity" and siglip_model is not None and active_objects:
                try:
                    crops = [img_pil.crop(tuple(int(round(float(c))) for c in p.box)) for p in img_patches]
                    bs = getattr(config, "siglip_budget_batch_size", 32)
                    scores = []
                    for b_i in range(0, len(crops), bs):
                        batch_crops = crops[b_i:b_i + bs]
                        scores.extend(siglip_model.score_crops(batch_crops, active_objects))

                    for p, s in zip(img_patches, scores):
                        p.siglip_score = float(s)

                    max_s = max(scores) if scores else 0.0
                    if config.verbose:
                        print(f"[Stage 3][Budget Guard] SigLIP-ranking {len(img_patches)} patches against active objects {active_objects} -> retaining top {config.max_patches_per_image} semantic patches (max score: {max_s:.3f}).")

                    img_patches = sorted(
                        img_patches,
                        key=lambda p: (p.siglip_score if p.siglip_score is not None else 0.0, 1.0 / max(1, (p.box[2]-p.box[0])*(p.box[3]-p.box[1]))),
                        reverse=True,
                    )[:config.max_patches_per_image]
                except Exception as e:
                    raise RuntimeError("SigLIP budget scoring failed; entropy fallback disabled") from e
            elif budget_mode == "uniform_stride":
                if config.verbose:
                    print(f"[Stage 3][Budget Guard] Uniformly downsampling {len(img_patches)} bounded patches to {config.max_patches_per_image} across canvas.")
                step = len(img_patches) / config.max_patches_per_image
                indices = [int(i * step) for i in range(config.max_patches_per_image)]
                img_patches = [img_patches[i] for i in indices]
            else:
                raise RuntimeError("SigLIP budget ranking requires a loaded scorer; no entropy fallback")

        # Ensure all surviving bounded patches have siglip_score populated when siglip_model is available
        if siglip_model is not None and active_objects:
            unscored = [p for p in img_patches if p.siglip_score is None]
            if unscored:
                try:
                    crops = [img_pil.crop(tuple(int(round(float(c))) for c in p.box)) for p in unscored]
                    bs = getattr(config, "siglip_budget_batch_size", 32)
                    scores = []
                    for b_i in range(0, len(crops), bs):
                        batch_crops = crops[b_i:b_i + bs]
                        scores.extend(siglip_model.score_crops(batch_crops, active_objects))
                    for p, s in zip(unscored, scores):
                        p.siglip_score = float(s)
                except Exception as e:
                    raise RuntimeError("SigLIP leaf scoring failed; entropy fallback disabled") from e

        bounded_patches.extend(img_patches)

    if config.verbose:
        stopping_mode = getattr(config, "partition_stopping_mode", "saliency_plateau")
        if stopping_mode == "fixed_size":
            mode_desc = f"fixed_size, max_patch_size <= {config.max_patch_size}px"
        else:
            mode_desc = f"saliency_plateau (gain_thresh={getattr(config, 'saliency_plateau_gain_threshold', 0.04):.2f}, ceiling={getattr(config, 'max_plateau_size', 1536)}px)"
        print(f"[Stage 3] Partitioning finished: Aggregated {len(bounded_patches)} bounded patches ({mode_desc}) across {len(candidate_images)} candidate images.")

    return bounded_patches
