"""
Stage 5: Unified Candidate Pool Aggregation & Spatial Optimization.
1. Multi-Entity Relational Proximity Box Merging: When is_relational=True, merges proximate
   boxes on the same candidate image to mitigate relational fragmentation.
2. Spatial IoU NMS Deduplication: Suppresses duplicate/overlapping candidate crops (IoU >= threshold)
   on the same image to prevent redundant Stage 6 VLM evaluations.
3. Converts surviving candidate patches directly into CandidateCrop objects for Stage 6.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
from PIL import Image

from ..config import MSSearchConfig
from .stage3_partitioning import BoundedPatch


@dataclass
class CandidateCrop:
    """Represents a validated candidate crop in the Unified Candidate Pool P."""
    img_idx: int
    box: Tuple[int, int, int, int]
    crop_image: Image.Image
    score: float
    source_branch: str = "Direct"

    @property
    def confidence_score(self) -> float:
        """Alias for score to ensure backward compatibility."""
        return self.score


def compute_box_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return float(inter_area) / float(union_area)


def compute_box_distance(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Computes edge-to-edge Euclidean distance between two bounding boxes (0 if intersecting)."""
    dx = max(0, max(box1[0], box2[0]) - min(box1[2], box2[2]))
    dy = max(0, max(box1[1], box2[1]) - min(box1[3], box2[3]))
    return float(math.hypot(dx, dy))


def merge_relational_proximity_boxes(
    patches: List[BoundedPatch],
    img_shapes: Dict[int, Tuple[int, int]],
    max_merge_dist: float = 350.0,
    max_box_dim: int = 1024,
    verbose: bool = False
) -> List[BoundedPatch]:
    """
    Mitigates Multi-Entity Relational Fragmentation:
    When relational intent is detected, merges spatially proximate candidate boxes
    on the same image into unified composite bounding boxes so that both the reference
    landmark and the target attribute remain intact in the same foveated reading window.
    """
    if not patches or len(patches) <= 1:
        return patches

    by_img: Dict[int, List[BoundedPatch]] = {}
    for p in patches:
        by_img.setdefault(p.img_idx, []).append(p)

    out_patches: List[BoundedPatch] = []
    merged_count = 0

    for img_idx, img_patches in by_img.items():
        if len(img_patches) <= 1:
            out_patches.extend(img_patches)
            continue

        img_w, img_h = img_shapes.get(img_idx, (10000, 10000))
        used = [False] * len(img_patches)
        for i in range(len(img_patches)):
            if used[i]:
                continue
            curr_box = list(img_patches[i].box)
            curr_score = img_patches[i].confidence_score if img_patches[i].confidence_score is not None else 0.0
            curr_siglip = img_patches[i].siglip_score if img_patches[i].siglip_score is not None else 0.0
            curr_entropy = img_patches[i].entropy

            for j in range(i + 1, len(img_patches)):
                if used[j]:
                    continue
                dist = compute_box_distance(tuple(curr_box), img_patches[j].box)
                if dist <= max_merge_dist:
                    cand_box = (
                        max(0, min(curr_box[0], img_patches[j].box[0])),
                        max(0, min(curr_box[1], img_patches[j].box[1])),
                        min(img_w, max(curr_box[2], img_patches[j].box[2])),
                        min(img_h, max(curr_box[3], img_patches[j].box[3]))
                    )
                    cand_w = cand_box[2] - cand_box[0]
                    cand_h = cand_box[3] - cand_box[1]
                    if cand_w <= max_box_dim and cand_h <= max_box_dim:
                        curr_box = list(cand_box)
                        j_score = img_patches[j].confidence_score if img_patches[j].confidence_score is not None else 0.0
                        j_siglip = img_patches[j].siglip_score if img_patches[j].siglip_score is not None else 0.0
                        curr_score = max(curr_score, j_score)
                        curr_siglip = max(curr_siglip, j_siglip)
                        curr_entropy = max(curr_entropy, img_patches[j].entropy)
                        used[j] = True
                        merged_count += 1

            used[i] = True
            out_patches.append(
                BoundedPatch(
                    img_idx=img_idx,
                    box=tuple(curr_box),
                    feature_vector=img_patches[i].feature_vector,
                    entropy=curr_entropy,
                    confidence_score=curr_score,
                    siglip_score=curr_siglip
                )
            )

    if verbose and merged_count > 0:
        print(f"[Stage 5][Relational Box Merge] Merged {merged_count} proximate relational box pairs (dist <= {max_merge_dist}px) to prevent multi-entity fragmentation.")

    return out_patches


def apply_spatial_nms(
    patches: List[BoundedPatch],
    iou_threshold: float = 0.60,
    verbose: bool = False
) -> List[BoundedPatch]:
    """
    Suppresses highly overlapping candidate patches on the same image (IoU >= iou_threshold).
    Ranks patches by confidence_score / siglip_score / entropy to preserve the most informative view.
    """
    if not patches or len(patches) <= 1:
        return patches

    by_img: Dict[int, List[BoundedPatch]] = {}
    for p in patches:
        by_img.setdefault(p.img_idx, []).append(p)

    surviving: List[BoundedPatch] = []
    total_suppressed = 0

    for img_idx, img_patches in by_img.items():
        if len(img_patches) <= 1:
            surviving.extend(img_patches)
            continue

        sorted_p = sorted(
            img_patches,
            key=lambda p: (
                p.confidence_score if p.confidence_score is not None else 0.0,
                p.siglip_score if p.siglip_score is not None else 0.0,
                p.entropy
            ),
            reverse=True
        )

        kept: List[BoundedPatch] = []
        for p in sorted_p:
            discard = False
            for k in kept:
                iou = compute_box_iou(p.box, k.box)
                if iou >= iou_threshold:
                    discard = True
                    break
            if not discard:
                kept.append(p)
            else:
                total_suppressed += 1

        surviving.extend(kept)

    if verbose and total_suppressed > 0:
        print(f"[Stage 5][Spatial NMS] Suppressed {total_suppressed} overlapping patches (IoU >= {iou_threshold:.2f}); {len(surviving)} crops retained.")

    return surviving


def run_resolution_screening(
    remaining_patches: List[BoundedPatch],
    candidate_images: List[Image.Image],
    active_objects: Optional[List[str]] = None,
    siglip_model = None,
    config: Optional[MSSearchConfig] = None,
    is_relational: Optional[bool] = None
) -> List[CandidateCrop]:
    """
    Directly aggregates surviving candidate patches into the Unified Candidate Pool P
    with spatial NMS deduplication and relational box proximity merging.
    """
    if not remaining_patches:
        return []

    effective_patches = list(remaining_patches)
    verbose = getattr(config, "verbose", True) if config else True

    # Detect relational intent if not explicitly passed
    if is_relational is None:
        if active_objects is not None and hasattr(active_objects, "is_relational"):
            is_relational = bool(active_objects.is_relational)
        else:
            is_relational = False

    # 1. Multi-Entity Relational Proximity Box Merging
    enable_rel_merge = getattr(config, "enable_relational_box_merging", True) if config else True
    if is_relational and enable_rel_merge:
        img_shapes = {idx: (img.width, img.height) for idx, img in enumerate(candidate_images)}
        merge_dist = getattr(config, "relational_merge_distance_px", 350.0) if config else 350.0
        effective_patches = merge_relational_proximity_boxes(
            effective_patches,
            img_shapes=img_shapes,
            max_merge_dist=merge_dist,
            max_box_dim=1024,
            verbose=verbose
        )

    # 2. Stage 5 Spatial IoU NMS Deduplication
    enable_nms = getattr(config, "enable_stage5_spatial_nms", True) if config else True
    if enable_nms:
        iou_thresh = getattr(config, "stage5_nms_iou_threshold", 0.60) if config else 0.60
        effective_patches = apply_spatial_nms(
            effective_patches,
            iou_threshold=iou_thresh,
            verbose=verbose
        )

    # 3. Aggregate into Unified Candidate Pool P
    unified_pool: List[CandidateCrop] = []
    for patch in effective_patches:
        box_ints = tuple(int(round(float(c))) for c in patch.box)
        crop = candidate_images[patch.img_idx].crop(box_ints)
        cw, ch = crop.size
        if cw < 4 or ch < 4:
            continue

        unified_pool.append(
            CandidateCrop(
                img_idx=patch.img_idx,
                box=box_ints,
                crop_image=crop,
                score=patch.confidence_score if patch.confidence_score is not None else 1.0,
                source_branch="Direct"
            )
        )

    if verbose:
        print(f"[Stage 5] Unified Candidate Pool P aggregated: {len(unified_pool)} total crops ready for foveated VLM reading (NMS & Relational merging applied).")

    return unified_pool
