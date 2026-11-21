"""
Stage 3: Gallery-Level Visual Screening and Native-Resolution Crop Inspection.
Pools candidates across gallery images, retains top-kN candidates, applies atomic sibling re-fusion
and contextual margin expansion, and evaluates native-resolution crops via MLLM batch inference.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image

from ..config import STASHConfig
from ..models.base import BaseMLLM
from .quadtree_partitioning import BoundedPatch


@dataclass
class CandidateExtraction:
    """Represents a candidate extraction record produced by native-resolution crop inspection."""
    img_idx: int
    box: Tuple[int, int, int, int]
    expanded_box: Tuple[int, int, int, int]
    alignment_score: float
    crop_area: int
    raw_answer: str
    final_answer: str


def compute_box_ios(
    child_box: Tuple[int, int, int, int],
    parent_box: Tuple[int, int, int, int]
) -> float:
    """
    Computes Intersection-over-Self (IoS) / Containment ratio of child_box inside parent_box:
    IoS = Area(child_box cap parent_box) / Area(child_box)
    """
    ix1 = max(child_box[0], parent_box[0])
    iy1 = max(child_box[1], parent_box[1])
    ix2 = min(child_box[2], parent_box[2])
    iy2 = min(child_box[3], parent_box[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter_area = float((ix2 - ix1) * (iy2 - iy1))
    child_area = float(max(1, (child_box[2] - child_box[0]) * (child_box[3] - child_box[1])))
    return inter_area / child_area


def compute_box_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int]
) -> float:
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


def merge_nested_containment_boxes(
    patches: List[BoundedPatch],
    containment_threshold: float = 0.95,
    verbose: bool = False,
    return_count: bool = False
) -> Union[List[BoundedPatch], Tuple[List[BoundedPatch], int]]:
    """
    Prunes nested parent/child redundancies where Box A is >= 95% contained in Box B.
    Favors fine-grained resolution:
    - Drops coarse Parent B in favor of fine Child A while transferring parent context score to Child A.
    """
    if not patches or len(patches) <= 1:
        if return_count:
            return list(patches), 0
        return list(patches)

    by_img: Dict[int, List[BoundedPatch]] = {}
    for p in patches:
        by_img.setdefault(p.img_idx, []).append(p)

    surviving: List[BoundedPatch] = []
    total_suppressed = 0

    for img_idx, img_patches in by_img.items():
        if len(img_patches) <= 1:
            surviving.extend(img_patches)
            continue

        def _area(b):
            return max(1, (b[2] - b[0]) * (b[3] - b[1]))

        sorted_by_area = sorted(img_patches, key=lambda p: _area(p.box))
        suppressed = set()

        for i in range(len(sorted_by_area)):
            if i in suppressed:
                continue
            child_p = sorted_by_area[i]
            child_box = child_p.box
            child_is_fine = getattr(child_p, "is_fine_grain", False) or getattr(child_p, "is_recursive_child", False) or (child_p.alignment_score or 0.0) >= 2.0

            for j in range(i + 1, len(sorted_by_area)):
                if j in suppressed:
                    continue
                parent_p = sorted_by_area[j]
                parent_box = parent_p.box

                ios = compute_box_ios(child_box, parent_box)
                if ios >= containment_threshold:
                    parent_is_fine = getattr(parent_p, "is_fine_grain", False) or getattr(parent_p, "is_recursive_child", False)

                    if child_is_fine or not parent_is_fine:
                        suppressed.add(j)
                        if parent_p.alignment_score is not None:
                            child_p.alignment_score = max(child_p.alignment_score or 0.0, parent_p.alignment_score)
                        total_suppressed += 1
                        if verbose:
                            print(f"[Stage 3][IoS Merging] Dropped coarse parent patch {parent_box} in favor of fine child {child_box} (IoS={ios:.2f}).")
                    else:
                        suppressed.add(i)
                        total_suppressed += 1
                        break

        surviving.extend([p for idx, p in enumerate(sorted_by_area) if idx not in suppressed])

    if verbose and total_suppressed > 0:
        print(f"[Stage 3][IoS Merging] Suppressed {total_suppressed} nested containment patches (IoS >= {containment_threshold:.2f}); {len(surviving)} patches retained.")

    if return_count:
        return surviving, total_suppressed
    return surviving


def apply_spatial_nms(
    patches: List[BoundedPatch],
    iou_threshold: float = 0.60,
    verbose: bool = False,
    return_count: bool = False
) -> Union[List[BoundedPatch], Tuple[List[BoundedPatch], int]]:
    """
    Suppresses highly overlapping candidate patches on the same image (IoU >= iou_threshold).
    Ranks patches by alignment_score / inverse area / entropy to preserve the most informative view.
    """
    if not patches or len(patches) <= 1:
        if return_count:
            return list(patches), 0
        return list(patches)

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
                p.alignment_score if p.alignment_score is not None else 0.0,
                1.0 / float(max(1, (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))),
                getattr(p, "entropy", 0.0)
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
        print(f"[Stage 3][Spatial NMS] Suppressed {total_suppressed} overlapping patches (IoU >= {iou_threshold:.2f}); {len(surviving)} crops retained.")

    if return_count:
        return surviving, total_suppressed
    return surviving


def are_seam_adjacent_siblings(
    p1: BoundedPatch,
    p2: BoundedPatch,
    seam_threshold: float = 0.10,
    tolerance_px: int = 4
) -> bool:
    """
    Checks if two patches share a quadtree parent_box and touch or overlap along a seam boundary.
    Handles both padded overlap and unpadded boundary touching.
    """
    if p1.parent_box is None or p2.parent_box is None:
        return False
    if p1.parent_box != p2.parent_box:
        return False
    if p1.img_idx != p2.img_idx:
        return False

    b1 = p1.box
    b2 = p2.box

    x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    min_w = max(1, min(b1[2] - b1[0], b2[2] - b2[0]))
    min_h = max(1, min(b1[3] - b1[1], b2[3] - b2[1]))

    # Overlapping boundary due to padding along seam
    if x_overlap > 0 and y_overlap > 0:
        if (x_overlap / min_w >= seam_threshold) or (y_overlap / min_h >= seam_threshold):
            return True

    # Touching along vertical seam
    touch_vert = (abs(b1[2] - b2[0]) <= tolerance_px or abs(b2[2] - b1[0]) <= tolerance_px)
    if touch_vert and (y_overlap / min_h >= seam_threshold):
        return True

    # Touching along horizontal seam
    touch_horiz = (abs(b1[3] - b2[1]) <= tolerance_px or abs(b2[3] - b1[1]) <= tolerance_px)
    if touch_horiz and (x_overlap / min_w >= seam_threshold):
        return True

    return False


def perform_atomic_sibling_refusion(
    patches: List[BoundedPatch],
    seam_threshold: float = 0.10,
    tolerance_px: int = 2
) -> List[BoundedPatch]:
    """
    Heals macro-scale visual context by atomically fusing sibling quadrants
    sharing the same parent node that meet along a common seam.
    """
    if len(patches) <= 1:
        return list(patches)

    n = len(patches)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if are_seam_adjacent_siblings(patches[i], patches[j], seam_threshold, tolerance_px):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    fused_patches: List[BoundedPatch] = []

    for i in range(n):
        if visited[i]:
            continue
        comp: List[int] = []
        queue = [i]
        visited[i] = True
        while queue:
            curr = queue.pop(0)
            comp.append(curr)
            for neighbor in adj[curr]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        if len(comp) == 1:
            fused_patches.append(patches[comp[0]])
        else:
            # Fuse all patches in component into a solid encompassing box
            comp_patches = [patches[idx] for idx in comp]
            fused_box = (
                min(p.box[0] for p in comp_patches),
                min(p.box[1] for p in comp_patches),
                max(p.box[2] for p in comp_patches),
                max(p.box[3] for p in comp_patches),
            )
            best_score = max(p.alignment_score for p in comp_patches)
            min_depth = min(p.depth for p in comp_patches)
            parent = comp_patches[0].parent_box

            max_entropy = max(getattr(p, "entropy", 0.0) for p in comp_patches)

            fused_patches.append(
                BoundedPatch(
                    img_idx=comp_patches[0].img_idx,
                    box=fused_box,
                    depth=min_depth,
                    alignment_score=best_score,
                    entropy=max_entropy,
                    parent_box=parent
                )
            )

    return fused_patches


def expand_box(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    expansion_factor: float = 1.25
) -> Tuple[int, int, int, int]:
    """
    Expands bounding box by contextual expansion factor clamped to image borders.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    pw = w * expansion_factor
    ph = h * expansion_factor

    ex1 = max(0, int(round(cx - pw / 2.0)))
    ey1 = max(0, int(round(cy - ph / 2.0)))
    ex2 = min(img_w, int(round(cx + pw / 2.0)))
    ey2 = min(img_h, int(round(cy + ph / 2.0)))

    return (ex1, ey1, ex2, ey2)


def clean_mllm_answer(text: str) -> str:
    """Strips formatting, tags, and extra punctuation from raw MLLM answer string."""
    cleaned = text.strip()
    # Strip <answer> tags if present
    import re
    m = re.search(r'<answer>([\s\S]*?)</answer>', cleaned, flags=re.IGNORECASE)
    if m:
        cleaned = m.group(1).strip()
    # Strip markdown bolding / quotes
    cleaned = re.sub(r'^[\*\(\[\"\'\s]+|[\*\)\]\"\'\.\s]+$', '', cleaned).strip()
    return cleaned


def screen_and_inspect_crops(
    candidate_patches: List[BoundedPatch],
    gallery_images: List[Image.Image],
    question: str,
    mllm: BaseMLLM,
    config: STASHConfig,
    return_stats: bool = False,
    return_crops: bool = False
) -> Union[
    List[CandidateExtraction],
    Tuple[List[CandidateExtraction], Dict[str, Any]],
    Tuple[List[CandidateExtraction], List[Image.Image]],
    Tuple[List[CandidateExtraction], Dict[str, Any], List[Image.Image]]
]:
    """
    Executes Gallery-Level Visual Screening and Native-Resolution Inspection:
    1. Retains top B_retain = k * N candidates globally ranked by alignment score.
    2. Performs atomic sibling re-fusion on seam-adjacent sibling quadrants.
    3. Expands bounding boxes by contextual margin alpha_expand = 1.25.
    4. Evaluates native-resolution crops via MLLM batch inference.

    Returns:
        List of CandidateExtraction records E (and optionally deduplication statistics).
    """
    num_images = len(gallery_images)
    b_retain = config.k * num_images

    # 1. Global ranking and retention (with tie-breaker: alignment_score, entropy)
    sorted_patches = sorted(
        candidate_patches,
        key=lambda p: (
            p.alignment_score if p.alignment_score is not None else 0.0,
            getattr(p, "entropy", 0.0)
        ),
        reverse=True
    )
    retained_patches = sorted_patches[:b_retain]

    # 2. Stage 5 deduplication: IoS containment merging (>= 0.95) and spatial IoU NMS (>= 0.60)
    num_ios_merged = 0
    num_nms_suppressed = 0

    enable_ios = getattr(config, "enable_ios_containment_merging", True)
    if enable_ios:
        ios_thresh = getattr(config, "ios_containment_threshold", 0.95)
        retained_patches, num_ios_merged = merge_nested_containment_boxes(
            retained_patches, containment_threshold=ios_thresh, verbose=config.verbose, return_count=True
        )

    enable_nms = getattr(config, "enable_stage5_spatial_nms", True)
    if enable_nms:
        iou_thresh = getattr(config, "stage5_nms_iou_threshold", 0.60)
        retained_patches, num_nms_suppressed = apply_spatial_nms(
            retained_patches, iou_threshold=iou_thresh, verbose=config.verbose, return_count=True
        )

    total_dedup_affected = num_ios_merged + num_nms_suppressed

    # 3. Per-image atomic sibling re-fusion and crop preparation
    by_img: Dict[int, List[BoundedPatch]] = {}
    for p in retained_patches:
        by_img.setdefault(p.img_idx, []).append(p)

    crop_metadata: List[Dict[str, Any]] = []
    crops_to_read: List[Image.Image] = []
    seam_tol = getattr(config, "spatial_continuity_threshold_px", 2)

    for img_idx, img_patches in by_img.items():
        img = gallery_images[img_idx]
        img_w, img_h = img.size

        # Atomic sibling re-fusion
        healed = perform_atomic_sibling_refusion(
            img_patches, seam_threshold=config.seam_overlap_threshold, tolerance_px=seam_tol
        )

        for p in healed:
            exp_box = expand_box(
                p.box, img_w, img_h, expansion_factor=config.bbox_expansion_factor
            )
            crop = img.crop(exp_box)
            if crop.width < 8 or crop.height < 8:
                continue

            area = (exp_box[2] - exp_box[0]) * (exp_box[3] - exp_box[1])
            crops_to_read.append(crop)
            crop_metadata.append({
                "img_idx": img_idx,
                "box": p.box,
                "expanded_box": exp_box,
                "alignment_score": p.alignment_score if p.alignment_score is not None else 0.0,
                "crop_area": area,
            })

    # Sort crops descending by visual alignment score before batched reading
    if crop_metadata:
        paired = list(zip(crop_metadata, crops_to_read))
        paired.sort(
            key=lambda pair: (
                pair[0]["alignment_score"],
                pair[0]["crop_area"]
            ),
            reverse=True
        )
        crop_metadata = [p[0] for p in paired]
        crops_to_read = [p[1] for p in paired]

    if config.verbose:
        print(f"[Stage 3: Screening & Inspection] Screened {len(sorted_patches[:b_retain])} patches "
              f"(Deduplication affected {total_dedup_affected} crops: {num_ios_merged} merged via IoS, "
              f"{num_nms_suppressed} suppressed via spatial NMS) -> Prepared {len(crops_to_read)} native crops for reading.")

    # 3. Parallel native-resolution crop inspection
    extractions: List[CandidateExtraction] = []
    batch_size = config.vlm_batch_size

    for i in range(0, len(crops_to_read), batch_size):
        batch_crops = crops_to_read[i : i + batch_size]
        batch_meta = crop_metadata[i : i + batch_size]

        responses = mllm.generate_response_batch(
            images=batch_crops,
            question=question,
            max_new_tokens=config.max_vlm_tokens
        )

        for meta, raw_resp in zip(batch_meta, responses):
            final_ans = clean_mllm_answer(raw_resp)
            extractions.append(
                CandidateExtraction(
                    img_idx=meta["img_idx"],
                    box=meta["box"],
                    expanded_box=meta["expanded_box"],
                    alignment_score=meta["alignment_score"],
                    crop_area=meta["crop_area"],
                    raw_answer=raw_resp,
                    final_answer=final_ans,
                )
            )

    dedup_stats = {
        "ios_merged": num_ios_merged,
        "nms_suppressed": num_nms_suppressed,
        "total_affected": total_dedup_affected,
        "retained_count": len(retained_patches),
        "total_screened_patches": len(sorted_patches),
    }
    if return_stats and return_crops:
        return extractions, dedup_stats, crops_to_read
    if return_stats:
        return extractions, dedup_stats
    if return_crops:
        return extractions, crops_to_read
    return extractions
