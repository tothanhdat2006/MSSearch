"""
Stage 5: Unified Candidate Pool Aggregation & Spatial Optimization.
1. Universal Spatial Compressed Canvas: Synthesizes continuous-coordinate compressed canvases
   excising empty rows and columns with 28px neutral visual gutters.
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
    is_compressed_canvas: bool = False
    remapped_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    original_boxes: Optional[List[Tuple[int, int, int, int]]] = None

    @property
    def confidence_score(self) -> float:
        """Alias for score to ensure backward compatibility."""
        return self.score


class CandidatePool(list):
    """
    List container for unified CandidateCrop instances.
    Subclasses list so all indexing, iteration, and len() calls operate natively,
    while holding pre_aggregation_boxes metadata for downstream visualization.
    """
    def __init__(self, iterable=None, pre_aggregation_boxes: Optional[List[Dict[str, Any]]] = None):
        super().__init__(iterable or [])
        self.pre_aggregation_boxes: List[Dict[str, Any]] = pre_aggregation_boxes or []



def merge_intervals(intervals: List[Tuple[int, int]], min_gap: int = 0) -> List[Tuple[int, int]]:
    """Merges overlapping or near-adjacent 1D intervals [v1, v2]."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_ivs[0]]
    for cur in sorted_ivs[1:]:
        prev = merged[-1]
        if cur[0] <= prev[1] + min_gap:
            merged[-1] = (prev[0], max(prev[1], cur[1]))
        else:
            merged.append(cur)
    return merged


def create_spatial_compressed_canvas(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    gutter_px: int = 8,
    bg_color: Tuple[int, int, int] = (18, 18, 18)
) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
    """
    Synthesizes a spatial-awareness compressed canvas from retained patch bounding boxes:
    1. Pastes only retained patches onto a blank canvas at their original pixel coordinates.
    2. Removes all completely empty rows and columns devoid of image content.
    3. Inserts thin visual gutters between non-contiguous spatial spans to prevent false adjacency.
    4. Computes exact remapped bounding box coordinates on the compressed canvas for prompt rendering.
    """
    if not boxes:
        return image, []
    if len(boxes) == 1:
        box = boxes[0]
        crop = image.crop(box)
        return crop, [(0, 0, crop.width, crop.height)]

    w, h = image.size
    # Clamped 1D intervals
    x_ivs = [(max(0, b[0]), min(w, b[2])) for b in boxes if b[2] > b[0]]
    y_ivs = [(max(0, b[1]), min(h, b[3])) for b in boxes if b[3] > b[1]]

    merged_x = merge_intervals(x_ivs, min_gap=gutter_px)
    merged_y = merge_intervals(y_ivs, min_gap=gutter_px)

    # Compute coordinate mappings
    x_map_start: Dict[Tuple[int, int], int] = {}
    curr_x = 0
    for idx, (x1, x2) in enumerate(merged_x):
        x_map_start[(x1, x2)] = curr_x
        curr_x += (x2 - x1)
        if idx < len(merged_x) - 1:
            curr_x += gutter_px
    canvas_w = max(1, curr_x)

    y_map_start: Dict[Tuple[int, int], int] = {}
    curr_y = 0
    for idx, (y1, y2) in enumerate(merged_y):
        y_map_start[(y1, y2)] = curr_y
        curr_y += (y2 - y1)
        if idx < len(merged_y) - 1:
            curr_y += gutter_px
    canvas_h = max(1, curr_y)

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    def remap_coord(val: int, intervals: List[Tuple[int, int]], map_start: Dict[Tuple[int, int], int]) -> int:
        for (v1, v2) in intervals:
            if v1 <= val <= v2:
                return map_start[(v1, v2)] + (val - v1)
            elif val < v1:
                return map_start[(v1, v2)]
        if intervals:
            last = intervals[-1]
            return map_start[last] + (last[1] - last[0])
        return 0

    # Paste patch crops onto canvas
    for box in boxes:
        crop = image.crop(box)
        nx1 = remap_coord(box[0], merged_x, x_map_start)
        ny1 = remap_coord(box[1], merged_y, y_map_start)
        canvas.paste(crop, (nx1, ny1))

    # Remap bounding boxes
    remapped_boxes: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        nx1 = remap_coord(box[0], merged_x, x_map_start)
        ny1 = remap_coord(box[1], merged_y, y_map_start)
        nx2 = remap_coord(box[2], merged_x, x_map_start)
        ny2 = remap_coord(box[3], merged_y, y_map_start)
        remapped_boxes.append((nx1, ny1, nx2, ny2))

    return canvas, remapped_boxes


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
    config: Optional[MSSearchConfig] = None
) -> List[CandidateCrop]:
    """
    Aggregates surviving candidate patches into the Unified Candidate Pool P:
    1. Applies Stage 5 Spatial IoU NMS to suppress overlapping patches per image.
    2. Synthesizes a Spatial-Awareness Compressed Canvas per image (excising empty rows/cols with 28px gutters).
       Guarantees at most 1 evaluation per candidate image containing surviving evidence.
    """
    if not remaining_patches:
        return CandidatePool([], pre_aggregation_boxes=[])

    effective_patches = list(remaining_patches)
    verbose = getattr(config, "verbose", True) if config else True

    # 1. Stage 5 Spatial IoU NMS Deduplication
    enable_nms = getattr(config, "enable_stage5_spatial_nms", True) if config else True
    if enable_nms:
        iou_thresh = getattr(config, "stage5_nms_iou_threshold", 0.60) if config else 0.60
        effective_patches = apply_spatial_nms(
            effective_patches,
            iou_threshold=iou_thresh,
            verbose=verbose
        )

    # 2. Universal Spatial Compressed Canvas Aggregation
    enable_compressed_canvas = getattr(config, "enable_spatial_compressed_canvas", True) if config else True
    gutter_px = getattr(config, "spatial_compression_gutter_px", 28) if config else 28

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

    if enable_compressed_canvas:
        # Group patches by source candidate image
        by_img: Dict[int, List[BoundedPatch]] = {}
        for p in effective_patches:
            by_img.setdefault(p.img_idx, []).append(p)

        for img_idx, img_patches in by_img.items():
            if len(img_patches) == 1:
                p = img_patches[0]
                box_ints = tuple(int(round(float(c))) for c in p.box)
                crop = candidate_images[img_idx].crop(box_ints)
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
                # Synthesize Spatial Compressed Canvas excising empty rows and columns
                boxes = [tuple(int(round(float(c))) for c in p.box) for p in img_patches]
                canvas_img, remapped_boxes = create_spatial_compressed_canvas(
                    candidate_images[img_idx],
                    boxes,
                    gutter_px=gutter_px
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
                        f"[Stage 5][Compressed Canvas] Image {img_idx}: Compressed {len(img_patches)} patches into "
                        f"{canvas_img.size[0]}x{canvas_img.size[1]} canvas (excised empty rows/cols, gutter={gutter_px}px)."
                    )
    else:
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
                    source_branch="Direct",
                    is_compressed_canvas=False,
                    remapped_boxes=[(0, 0, crop.width, crop.height)],
                    original_boxes=[box_ints]
                )
            )

    pool = CandidatePool(unified_pool, pre_aggregation_boxes=pre_aggregation_boxes)

    if verbose:
        print(f"[Stage 5] Unified Candidate Pool P aggregated: {len(pool)} total items ready for foveated VLM reading.")

    return pool

