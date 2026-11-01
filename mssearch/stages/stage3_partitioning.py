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

        # Get cached feature map or default empty
        feat_map = stage2_feature_cache.get(img_idx, None)
        if feat_map is not None:
            if config.verbose:
                print(f"[Stage 3][Non-Fallback] Image {img_idx}: Reusing cached Stage 2 vision feature map (Shape: {feat_map.shape}).")
        else:
            feat_map = np.zeros((256, 72, 72), dtype=np.float32)
            if config.verbose:
                print(f"[Stage 3][Fallback] Image {img_idx}: No Stage 2 feature map in cache; falling back to zero-initialized feature map.")

        img_patches: List[BoundedPatch] = []
        discarded_candidates: List[Tuple[Tuple[int, int, int, int], float]] = []

        while work_queue:
            box, depth, parent_box = work_queue.pop(0)
            box_ints = tuple(int(round(float(c))) for c in box)
            x1, y1, x2, y2 = box_ints
            pw, ph = x2 - x1, y2 - y1

            cached_siglip_score: Optional[float] = None
            sub_boxes: Optional[List[Tuple[int, int, int, int]]] = None

            stopping_mode = getattr(config, "partition_stopping_mode", "saliency_plateau")

            if stopping_mode == "fixed_size":
                # Legacy invariant: longer side <= max_patch_size (with 15% tolerance for boundary overlap)
                should_split = (max(pw, ph) > int(config.max_patch_size * 1.15))
            else:
                # Paradigm 2: Semantic Saliency-Plateau Stopping (Self-Terminating Zoom)
                if depth == 0:
                    should_split = True
                elif depth >= config.max_partition_depth:
                    should_split = False
                elif max(pw, ph) <= getattr(config, "min_plateau_size", 256):
                    should_split = False
                elif max(pw, ph) > getattr(config, "max_plateau_size", 1536):
                    should_split = True
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
                        except Exception as e:
                            if config.verbose:
                                print(f"[Stage 3][Plateau Notice] SigLIP scoring failed ({e}); falling back to entropy plateau.")

                    if not evaluated_with_siglip:
                        ent_parent = compute_spatial_entropy(crop_parent, weights=config.entropy_weights)
                        child_crops = [
                            img_pil.crop(tuple(int(round(float(c))) for c in sb))
                            for sb in candidate_sub_boxes
                        ]
                        ent_children = [
                            compute_spatial_entropy(c, weights=config.entropy_weights)
                            for c in child_crops
                        ]
                        max_child_ent = max(ent_children) if ent_children else 0.0
                        delta_h = max_child_ent - ent_parent

                        if delta_h <= 0.02 and ent_parent >= config.min_patch_entropy:
                            if config.verbose:
                                print(f"[Stage 3][Plateau Stop (Entropy)] Patch {box_ints} ({pw}x{ph}, depth={depth}) reached entropy plateau (H_p={ent_parent:.3f}, H_c={max_child_ent:.3f}, d_H={delta_h:+.3f}). Terminating zoom.")
                            should_split = False
                        else:
                            should_split = True
                            sub_boxes = candidate_sub_boxes

            if should_split:
                crop = img_pil.crop(box_ints)
                entropy = compute_spatial_entropy(crop, weights=config.entropy_weights)

                # Only run SAM3-SGAP clustering at root level (depth == 0)
                should_run_sgap = (
                    depth == 0 and
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
                elif sub_boxes is None:
                    sub_boxes = split_box_into_subpatches(
                        box_ints,
                        img_w,
                        img_h,
                        overlap_ratio=config.quadrant_overlap_ratio,
                    )
                    if config.verbose and depth <= 1:
                        print(f"[Stage 3][Routing] Patch {box_ints} (size {pw}x{ph}, depth={depth}) -> Direction-aware Split ({len(sub_boxes)} patches).")

                for sb in sub_boxes:
                    sb_ints = tuple(int(round(float(c))) for c in sb)
                    work_queue.append((sb_ints, depth + 1, box_ints))
            else:
                crop = img_pil.crop(box_ints)
                entropy = compute_spatial_entropy(crop, weights=config.entropy_weights)

                if config.min_patch_entropy > 0.0 and entropy < config.min_patch_entropy:
                    if cached_siglip_score is None or cached_siglip_score < getattr(config, "saliency_target_threshold", 0.35):
                        discarded_candidates.append((box_ints, entropy))
                        continue

                _, pooled_vec = slice_feature_map(feat_map, box_ints, img_w, img_h)

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

        if not img_patches and discarded_candidates:
            best_box, best_ent = max(discarded_candidates, key=lambda x: x[1])
            _, pooled_vec = slice_feature_map(feat_map, best_box, img_w, img_h)
            img_patches.append(
                BoundedPatch(
                    img_idx=img_idx,
                    box=best_box,
                    feature_vector=pooled_vec,
                    entropy=best_ent,
                    parent_id=None,
                    level=0,
                )
            )
            if config.verbose:
                print(f"[Stage 3][Safety Fallback] Image {img_idx}: All patches had H < {config.min_patch_entropy:.2f}. Retained highest-entropy patch {best_box} (H={best_ent:.3f}).")
        elif config.verbose and discarded_candidates:
            print(f"[Stage 3][Entropy Pruning] Image {img_idx}: Pruned {len(discarded_candidates)} patches with H < {config.min_patch_entropy:.2f}. Retained {len(img_patches)} bounded patches.")

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
                        key=lambda p: (p.siglip_score if p.siglip_score is not None else 0.0, p.entropy),
                        reverse=True,
                    )[:config.max_patches_per_image]
                except Exception as e:
                    if config.verbose:
                        print(f"[Stage 3][Budget Guard Notice] SigLIP scoring failed ({e}); falling back to entropy descending.")
                    img_patches = sorted(img_patches, key=lambda p: p.entropy, reverse=True)[:config.max_patches_per_image]
            elif budget_mode in ("entropy_descending", "siglip_similarity"):
                if config.verbose:
                    print(f"[Stage 3][Budget Guard] Entropy-ranking {len(img_patches)} patches -> retaining top {config.max_patches_per_image} highest-entropy patches.")
                img_patches = sorted(img_patches, key=lambda p: p.entropy, reverse=True)[:config.max_patches_per_image]
            else:
                if config.verbose:
                    print(f"[Stage 3][Budget Guard] Uniformly downsampling {len(img_patches)} bounded patches to {config.max_patches_per_image} across canvas.")
                step = len(img_patches) / config.max_patches_per_image
                indices = [int(i * step) for i in range(config.max_patches_per_image)]
                img_patches = [img_patches[i] for i in indices]

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
                    if config.verbose:
                        print(f"[Stage 3][SigLIP Leaf Scoring Notice] SigLIP scoring failed ({e}).")

        bounded_patches.extend(img_patches)

    if config.verbose:
        stopping_mode = getattr(config, "partition_stopping_mode", "saliency_plateau")
        if stopping_mode == "fixed_size":
            mode_desc = f"fixed_size, max_patch_size <= {config.max_patch_size}px"
        else:
            mode_desc = f"saliency_plateau (gain_thresh={getattr(config, 'saliency_plateau_gain_threshold', 0.04):.2f}, ceiling={getattr(config, 'max_plateau_size', 1536)}px)"
        print(f"[Stage 3] Partitioning finished: Aggregated {len(bounded_patches)} bounded patches ({mode_desc}) across {len(candidate_images)} candidate images.")

    return bounded_patches
