"""
Stage 3: Gallery-Level Visual Screening and Native-Resolution Crop Inspection.
Pools candidates across gallery images, retains top-kN candidates, applies atomic sibling re-fusion
and contextual margin expansion, and evaluates native-resolution crops via MLLM batch inference.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
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
    tolerance_px: int = 4
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

            fused_patches.append(
                BoundedPatch(
                    img_idx=comp_patches[0].img_idx,
                    box=fused_box,
                    depth=min_depth,
                    alignment_score=best_score,
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
    config: STASHConfig
) -> List[CandidateExtraction]:
    """
    Executes Gallery-Level Visual Screening and Native-Resolution Inspection:
    1. Retains top B_retain = k * N candidates globally ranked by alignment score.
    2. Performs atomic sibling re-fusion on seam-adjacent sibling quadrants.
    3. Expands bounding boxes by contextual margin alpha_expand = 1.25.
    4. Evaluates native-resolution crops via MLLM batch inference.

    Returns:
        List of CandidateExtraction records E.
    """
    num_images = len(gallery_images)
    b_retain = config.k * num_images

    # 1. Global ranking and retention
    sorted_patches = sorted(candidate_patches, key=lambda p: p.alignment_score, reverse=True)
    retained_patches = sorted_patches[:b_retain]

    # 2. Per-image atomic sibling re-fusion and crop preparation
    by_img: Dict[int, List[BoundedPatch]] = {}
    for p in retained_patches:
        by_img.setdefault(p.img_idx, []).append(p)

    crop_metadata: List[Dict[str, Any]] = []
    crops_to_read: List[Image.Image] = []

    for img_idx, img_patches in by_img.items():
        img = gallery_images[img_idx]
        img_w, img_h = img.size

        # Atomic sibling re-fusion
        healed = perform_atomic_sibling_refusion(
            img_patches, seam_threshold=config.seam_overlap_threshold
        )

        for p in healed:
            exp_box = expand_box(
                p.box, img_w, img_h, expansion_factor=config.bbox_expansion_factor
            )
            crop = img.crop(exp_box)
            if crop.width < 4 or crop.height < 4:
                continue

            area = (exp_box[2] - exp_box[0]) * (exp_box[3] - exp_box[1])
            crops_to_read.append(crop)
            crop_metadata.append({
                "img_idx": img_idx,
                "box": p.box,
                "expanded_box": exp_box,
                "alignment_score": p.alignment_score,
                "crop_area": area,
            })

    if config.verbose:
        print(f"[Stage 3: Screening & Inspection] Screened {len(retained_patches)} patches -> Prepared {len(crops_to_read)} native crops for reading.")

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

    return extractions
