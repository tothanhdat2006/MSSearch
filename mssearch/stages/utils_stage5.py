"""
Stage 5 Utilities: Data Structures, Canvas Synthesis, Interval Merging, Water-Filling,
Spatial NMS, Relational Classification, Sibling Refusion, and Sub-Canvas Windowing.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math
import re
import json
from PIL import Image

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
    is_already_expanded: bool = False

    @property
    def confidence_score(self) -> float:
        """Alias for score to ensure backward compatibility."""
        return self.score


class CandidatePool(list):
    """
    List container for unified CandidateCrop instances.
    Subclasses list so all indexing, iteration, and len() calls operate natively,
    while holding pre_aggregation_boxes metadata and relational intent flags for downstream visualization.
    """
    def __init__(
        self,
        iterable=None,
        pre_aggregation_boxes: Optional[List[Dict[str, Any]]] = None,
        is_cross_object: bool = False,
        is_relational: bool = False,
        stage5_mode: str = "crops_only"
    ):
        super().__init__(iterable or [])
        self.pre_aggregation_boxes: List[Dict[str, Any]] = pre_aggregation_boxes or []
        self.is_cross_object: bool = bool(is_cross_object)
        self.is_relational: bool = bool(is_relational)
        self.stage5_mode: str = str(stage5_mode)


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


def solve_water_filling_allocation(
    lengths: List[int],
    roles: List[str],
    total_budget: int,
    target_max_dim: int = 512
) -> List[float]:
    """
    Computes scale factors s_k for each interval under hard upper bound:
    sum(round(lengths[k] * s_k)) <= total_budget.

    Roles: 'target', 'anchor', 'neutral'.
    Target intervals <= target_max_dim are locked at s_k = 1.0.
    Anchor/neutral intervals absorb compression via water-filling cap C*.
    If target intervals alone exceed total_budget, scales all intervals proportionally.
    """
    n = len(lengths)
    if n == 0 or total_budget <= 0:
        return [1.0] * n

    sum_len = sum(lengths)
    if sum_len <= total_budget:
        return [1.0] * n

    # Step 1: Identify targets vs anchors
    is_target = [
        r == "target" or (r != "anchor" and l <= target_max_dim)
        for r, l in zip(roles, lengths)
    ]
    target_sum = sum(l for t, l in zip(is_target, lengths) if t)
    anchor_indices = [i for i, t in enumerate(is_target) if not t]

    # Step 2: Handle target overflow
    if target_sum >= total_budget or not anchor_indices:
        global_scale = min(1.0, float(total_budget) / float(max(1, sum_len)))
        return [global_scale] * n

    # Step 3: Targets locked at 1.0, water-fill remaining budget to anchors
    rem_budget = total_budget - target_sum
    anchor_lengths = [lengths[i] for i in anchor_indices]

    low, high = 1.0, float(max(anchor_lengths))
    best_cap = low
    for _ in range(32):
        mid = (low + high) / 2.0
        allocated = sum(min(float(l), mid) for l in anchor_lengths)
        if allocated <= rem_budget:
            best_cap = mid
            low = mid
        else:
            high = mid

    scales = [1.0] * n
    for i in anchor_indices:
        l = lengths[i]
        allocated = min(float(l), best_cap)
        scales[i] = min(1.0, allocated / float(max(1, l)))

    return scales


def compute_box_distance(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Computes edge-to-edge Euclidean distance between two bounding boxes (0 if intersecting)."""
    dx = max(0, max(box1[0], box2[0]) - min(box1[2], box2[2]))
    dy = max(0, max(box1[1], box2[1]) - min(box1[3], box2[3]))
    return float(math.hypot(dx, dy))


def extract_joint_interaction_crops(
    img_patches: List[BoundedPatch],
    image_shape: Tuple[int, int],
    prox_threshold: float = 400.0,
    max_dim: int = 1024,
    padding_factor: float = 1.15
) -> Tuple[List[BoundedPatch], bool]:
    """
    Pillar 1: Proximity-Graph Joint Interaction Crop (PG-JIC).
    Groups spatially proximate patches (d_edge <= prox_threshold) into connected components.
    If all patches form a single connected component with union <= max_dim,
    returns a single unified patch with is_all_one_jic = True (bypassing canvas synthesis).
    Otherwise merges individual clusters whose union <= max_dim to reduce canvas fragmentation.
    """
    if not img_patches or len(img_patches) <= 1:
        return img_patches, False

    img_w, img_h = image_shape
    n = len(img_patches)
    adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            dist = compute_box_distance(img_patches[i].box, img_patches[j].box)
            if dist <= prox_threshold:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    components: List[List[int]] = []
    for i in range(n):
        if not visited[i]:
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
            components.append(comp)

    def _pad_union_box(indices: List[int]) -> Tuple[int, int, int, int]:
        bx1 = min(img_patches[idx].box[0] for idx in indices)
        by1 = min(img_patches[idx].box[1] for idx in indices)
        bx2 = max(img_patches[idx].box[2] for idx in indices)
        by2 = max(img_patches[idx].box[3] for idx in indices)
        bw = bx2 - bx1
        bh = by2 - by1
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        pw = bw * padding_factor
        ph = bh * padding_factor
        px1 = max(0, int(round(cx - pw / 2.0)))
        py1 = max(0, int(round(cy - ph / 2.0)))
        px2 = min(img_w, int(round(cx + pw / 2.0)))
        py2 = min(img_h, int(round(cy + ph / 2.0)))
        return (px1, py1, px2, py2)

    # If all patches form exactly 1 connected component fitting in max_dim:
    if len(components) == 1:
        u_box = _pad_union_box(components[0])
        uw = u_box[2] - u_box[0]
        uh = u_box[3] - u_box[1]
        if max(uw, uh) <= max_dim:
            max_score = max((p.confidence_score or 1.0) for p in img_patches)
            max_siglip = max((p.siglip_score or 0.0) for p in img_patches)
            max_ent = max(p.entropy for p in img_patches)
            jic_patch = BoundedPatch(
                img_idx=img_patches[0].img_idx,
                box=u_box,
                feature_vector=img_patches[0].feature_vector,
                entropy=max_ent,
                confidence_score=max_score,
                siglip_score=max_siglip,
                semantic_role="jic_cluster"
            )
            return [jic_patch], True

    # Otherwise, merge clusters that individually fit in max_dim
    merged_patches: List[BoundedPatch] = []
    for comp in components:
        if len(comp) > 1:
            u_box = _pad_union_box(comp)
            uw = u_box[2] - u_box[0]
            uh = u_box[3] - u_box[1]
            if max(uw, uh) <= max_dim:
                max_score = max((img_patches[idx].confidence_score or 1.0) for idx in comp)
                max_siglip = max((img_patches[idx].siglip_score or 0.0) for idx in comp)
                max_ent = max(img_patches[idx].entropy for idx in comp)
                roles = [getattr(img_patches[idx], "semantic_role", None) for idx in comp]
                comp_role = "target" if "target" in roles else ("anchor" if "anchor" in roles else None)
                merged_patches.append(
                    BoundedPatch(
                        img_idx=img_patches[comp[0]].img_idx,
                        box=u_box,
                        feature_vector=img_patches[comp[0]].feature_vector,
                        entropy=max_ent,
                        confidence_score=max_score,
                        siglip_score=max_siglip,
                        semantic_role=comp_role
                    )
                )
                continue
        for idx in comp:
            merged_patches.append(img_patches[idx])

    return merged_patches, False


def create_spatial_compressed_canvas(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    gutter_px: int = 28,
    bg_color: Tuple[int, int, int] = (18, 18, 18),
    box_roles: Optional[List[str]] = None,
    max_canvas_dim: int = 1024,
    enable_dynamic_gutters: bool = True,
    continuity_thresh: int = 2,
    enable_water_filling: bool = True,
    target_max_native_dim: int = 512
) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
    """
    Synthesizes a spatial-awareness compressed canvas from retained patch bounding boxes:
    1. Pastes only retained patches onto a blank canvas with water-filling resolution allocation.
    2. Enforces hard canvas dimension bound: max(W, H) <= max_canvas_dim (1024px) to prevent VLM blur.
    3. Dynamic Gutters: 0px seamless stitch for physically connected tiles (gap <= 2px),
       28px neutral visual gutter for excised voids.
    4. Adaptive Water-Filling: Locks micro-targets <= 512px at 1:1 native resolution;
       anchor landmarks absorb compression down to budget.
    """
    if not boxes:
        return image, []
    if len(boxes) == 1:
        box = boxes[0]
        crop = image.crop(box)
        if max(crop.width, crop.height) > max_canvas_dim:
            scale = float(max_canvas_dim) / float(max(crop.width, crop.height))
            nw = max(1, int(round(crop.width * scale)))
            nh = max(1, int(round(crop.height * scale)))
            crop = crop.resize((nw, nh), Image.Resampling.BILINEAR)
            return crop, [(0, 0, nw, nh)]
        return crop, [(0, 0, crop.width, crop.height)]

    w, h = image.size
    x_ivs = [(max(0, b[0]), min(w, b[2])) for b in boxes if b[2] > b[0]]
    y_ivs = [(max(0, b[1]), min(h, b[3])) for b in boxes if b[3] > b[1]]

    merged_x = merge_intervals(x_ivs, min_gap=continuity_thresh)
    merged_y = merge_intervals(y_ivs, min_gap=continuity_thresh)

    g_x: List[int] = []
    for k in range(len(merged_x) - 1):
        gap = merged_x[k + 1][0] - merged_x[k][1]
        g_x.append(0 if (enable_dynamic_gutters and gap <= continuity_thresh) else gutter_px)
    tot_gx = sum(g_x)

    g_y: List[int] = []
    for m in range(len(merged_y) - 1):
        gap = merged_y[m + 1][0] - merged_y[m][1]
        g_y.append(0 if (enable_dynamic_gutters and gap <= continuity_thresh) else gutter_px)
    tot_gy = sum(g_y)

    bx = max(1, max_canvas_dim - tot_gx)
    by = max(1, max_canvas_dim - tot_gy)

    roles_x: List[str] = []
    for (x1, x2) in merged_x:
        int_role = "neutral"
        for b_idx, b in enumerate(boxes):
            if max(b[0], x1) < min(b[2], x2):
                r = box_roles[b_idx] if box_roles and b_idx < len(box_roles) else None
                if r == "target":
                    int_role = "target"
                    break
                elif r == "anchor" and int_role != "target":
                    int_role = "anchor"
        roles_x.append(int_role)

    roles_y: List[str] = []
    for (y1, y2) in merged_y:
        int_role = "neutral"
        for b_idx, b in enumerate(boxes):
            if max(b[1], y1) < min(b[3], y2):
                r = box_roles[b_idx] if box_roles and b_idx < len(box_roles) else None
                if r == "target":
                    int_role = "target"
                    break
                elif r == "anchor" and int_role != "target":
                    int_role = "anchor"
        roles_y.append(int_role)

    lengths_x = [x2 - x1 for (x1, x2) in merged_x]
    lengths_y = [y2 - y1 for (y1, y2) in merged_y]

    if enable_water_filling:
        scales_x = solve_water_filling_allocation(lengths_x, roles_x, bx, target_max_dim=target_max_native_dim)
        scales_y = solve_water_filling_allocation(lengths_y, roles_y, by, target_max_dim=target_max_native_dim)
    else:
        sx = min(1.0, float(bx) / float(max(1, sum(lengths_x))))
        sy = min(1.0, float(by) / float(max(1, sum(lengths_y))))
        scales_x = [sx] * len(lengths_x)
        scales_y = [sy] * len(lengths_y)

    w_primes = [max(1, int(round(l * s))) for l, s in zip(lengths_x, scales_x)]
    h_primes = [max(1, int(round(l * s))) for l, s in zip(lengths_y, scales_y)]

    x_offsets: List[int] = []
    curr_x = 0
    for k, wp in enumerate(w_primes):
        x_offsets.append(curr_x)
        curr_x += wp
        if k < len(g_x):
            curr_x += g_x[k]
    canvas_w = max(1, curr_x)

    y_offsets: List[int] = []
    curr_y = 0
    for m, hp in enumerate(h_primes):
        y_offsets.append(curr_y)
        curr_y += hp
        if m < len(g_y):
            curr_y += g_y[m]
    canvas_h = max(1, curr_y)

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    def remap_coord_x(val: int) -> int:
        for k, (v1, v2) in enumerate(merged_x):
            if v1 <= val <= v2:
                return x_offsets[k] + int(round((val - v1) * scales_x[k]))
            elif val < v1:
                return x_offsets[k]
        if merged_x:
            last_k = len(merged_x) - 1
            last = merged_x[last_k]
            return x_offsets[last_k] + int(round((last[1] - last[0]) * scales_x[last_k]))
        return 0

    def remap_coord_y(val: int) -> int:
        for m, (v1, v2) in enumerate(merged_y):
            if v1 <= val <= v2:
                return y_offsets[m] + int(round((val - v1) * scales_y[m]))
            elif val < v1:
                return y_offsets[m]
        if merged_y:
            last_m = len(merged_y) - 1
            last = merged_y[last_m]
            return y_offsets[last_m] + int(round((last[1] - last[0]) * scales_y[last_m]))
        return 0

    for k in range(len(merged_x)):
        for m in range(len(merged_y)):
            cell_x1, cell_x2 = merged_x[k]
            cell_y1, cell_y2 = merged_y[m]
            overlaps = any(
                max(b[0], cell_x1) < min(b[2], cell_x2) and max(b[1], cell_y1) < min(b[3], cell_y2)
                for b in boxes
            )
            if overlaps:
                crop = image.crop((cell_x1, cell_y1, cell_x2, cell_y2))
                dw = w_primes[k]
                dh = h_primes[m]
                if crop.size != (dw, dh):
                    crop = crop.resize((dw, dh), Image.Resampling.BILINEAR)
                canvas.paste(crop, (x_offsets[k], y_offsets[m]))

    remapped_boxes: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        rx1 = remap_coord_x(box[0])
        ry1 = remap_coord_y(box[1])
        rx2 = remap_coord_x(box[2])
        ry2 = remap_coord_y(box[3])
        remapped_boxes.append((rx1, ry1, max(rx1 + 1, rx2), max(ry1 + 1, ry2)))

    return canvas, remapped_boxes


def compute_box_ios(child_box: Tuple[int, int, int, int], parent_box: Tuple[int, int, int, int]) -> float:
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


def merge_nested_containment_boxes(
    patches: List[BoundedPatch],
    containment_threshold: float = 0.95,
    verbose: bool = False
) -> List[BoundedPatch]:
    """
    Prunes nested parent/child redundancies where Box A is >= 95% contained in Box B.
    Favors fine-grained resolution:
    - If Child A was generated from a detection anchor or recursive zoom, drops coarse Parent B
      while transferring any parent context scores to Child A.
    - If neither is specialized, keeps the higher confidence box.
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

        def _area(b):
            return max(1, (b[2] - b[0]) * (b[3] - b[1]))

        sorted_by_area = sorted(img_patches, key=lambda p: _area(p.box))
        suppressed = set()

        for i in range(len(sorted_by_area)):
            if i in suppressed:
                continue
            child_p = sorted_by_area[i]
            child_box = child_p.box
            child_is_fine = getattr(child_p, "is_fine_grain", False) or getattr(child_p, "is_recursive_child", False) or (child_p.confidence_score or 0.0) >= 2.0

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
                        if parent_p.confidence_score is not None:
                            child_p.confidence_score = max(child_p.confidence_score or 0.0, parent_p.confidence_score)
                        if parent_p.siglip_score is not None:
                            child_p.siglip_score = max(child_p.siglip_score or 0.0, parent_p.siglip_score)
                        total_suppressed += 1
                        if verbose:
                            print(f"[Stage 5][IoS Merging] Dropped coarse parent patch {parent_box} in favor of fine child {child_box} (IoS={ios:.2f}).")
                    else:
                        suppressed.add(i)
                        total_suppressed += 1
                        break

        surviving.extend([p for idx, p in enumerate(sorted_by_area) if idx not in suppressed])

    if verbose and total_suppressed > 0:
        print(f"[Stage 5][IoS Merging] Suppressed {total_suppressed} nested containment patches (IoS >= {containment_threshold:.2f}); {len(surviving)} patches retained.")

    return surviving


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
    on the same image into unified composite bounding boxes.
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
                1.0 / float(max(1, (p.box[2]-p.box[0])*(p.box[3]-p.box[1])))
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


def check_is_cross_object_llm(question: str, zoom_model: Any, verbose: bool = False) -> bool:
    """
    Uses the LLM to classify whether a question is a cross-object relational question
    (requiring spatial layout) or a single-object question (requiring native crops only).
    """
    if zoom_model is None or not question:
        return False

    prompt = (
        "You are an expert visual reasoning classifier. Analyze the following question:\n\n"
        f"Question: \"{question}\"\n\n"
        "Determine if this is a CROSS-OBJECT relational question or a SINGLE-OBJECT question:\n"
        "- A CROSS-OBJECT question asks about the spatial relationship, relative position, direction, or comparison between TWO OR MORE objects (e.g. \"what is above the logo?\", \"what is to the left of the car?\", \"which object is closer?\").\n"
        "- A SINGLE-OBJECT question asks about the attributes, color, text, number, appearance, or identity of a single object (e.g. \"what is the color of the mailbox?\", \"what is the registration number of the boat?\", \"what brand is the shirt?\").\n\n"
        "Output ONLY a JSON object with a single boolean field:\n"
        "{\"is_cross_object\": true} or {\"is_cross_object\": false}"
    )

    raw = ""
    if hasattr(zoom_model, "generate_text"):
        try:
            raw = zoom_model.generate_text(prompt, max_new_tokens=32)
        except Exception:
            raw = ""
    elif hasattr(zoom_model, "generate_response"):
        try:
            raw = zoom_model.generate_response(image_pil=None, question=prompt, max_new_tokens=32)
        except Exception:
            raw = ""

    if verbose and raw:
        print(f"[Stage 5][LLM Relational Classifier] Output: {repr(raw)}")

    m = re.search(r'\{[\s\S]*?\}', raw)
    if m:
        try:
            data = json.loads(m.group(0))
            if "is_cross_object" in data:
                return bool(data["is_cross_object"])
            if "is_relational" in data:
                return bool(data["is_relational"])
        except Exception:
            pass

    raw_lower = raw.lower()
    if '"is_cross_object": true' in raw_lower or '"is_relational": true' in raw_lower:
        return True
    return False


def is_relational_question(active_objects: Any = None, question: str = "", zoom_model: Any = None) -> bool:
    """
    Determines whether a question is a cross-object relational question (requiring spatial layout)
    or a single-object question (requiring native-resolution crops only) using LLM semantic understanding.
    """
    if active_objects is not None and isinstance(active_objects, (list, tuple)):
        if len(active_objects) <= 1:
            return False

    if hasattr(active_objects, "is_relational") and active_objects.is_relational is not None:
        return bool(active_objects.is_relational)

    q_type = getattr(active_objects, "question_type", "")
    if q_type in ["relational_reading", "cross_object_relation", "spatial_orientation", "relational", "cross_object"]:
        return True
    elif q_type in ["single_object_attribute", "fine_grained_attribute", "text_reading", "counting"]:
        return False

    if zoom_model is not None and question:
        return check_is_cross_object_llm(question, zoom_model)

    return False


def are_seam_adjacent_siblings(
    p1: BoundedPatch,
    p2: BoundedPatch,
    seam_overlap_threshold: float = 0.10,
    tolerance_px: int = 2
) -> bool:
    """Checks if two patches share a quadtree parent_id and touch or overlap along a seam boundary."""
    if p1.parent_id is None or p2.parent_id is None:
        return False
    if p1.parent_id != p2.parent_id:
        return False
    if p1.img_idx != p2.img_idx:
        return False

    b1 = p1.box
    b2 = p2.box

    x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    y_overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    min_w = max(1, min(b1[2] - b1[0], b2[2] - b2[0]))
    min_h = max(1, min(b1[3] - b1[1], b2[3] - b2[1]))

    if x_overlap > 0 and y_overlap > 0:
        if (x_overlap / min_w >= seam_overlap_threshold) or (y_overlap / min_h >= seam_overlap_threshold):
            return True

    touch_vert = (abs(b1[2] - b2[0]) <= tolerance_px or abs(b2[2] - b1[0]) <= tolerance_px)
    if touch_vert and (y_overlap / min_h >= seam_overlap_threshold):
        return True

    touch_horiz = (abs(b1[3] - b2[1]) <= tolerance_px or abs(b2[3] - b1[1]) <= tolerance_px)
    if touch_horiz and (x_overlap / min_w >= seam_overlap_threshold):
        return True

    return False


def perform_atomic_sibling_refusion(
    patches: List[BoundedPatch],
    seam_overlap_threshold: float = 0.10,
    tolerance_px: int = 2
) -> List[BoundedPatch]:
    """
    Pillar 1: Atomic Sibling Re-Fusion (Healing Severed Macro-Objects).
    Scans candidate patches sharing parent_id and seam boundary (d_edge = 0, >= 10% overlap)
    and fuses them into solid, un-split atomic bounding boxes.
    """
    if len(patches) <= 1:
        return patches

    n = len(patches)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if are_seam_adjacent_siblings(patches[i], patches[j], seam_overlap_threshold, tolerance_px):
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    fused_patches: List[BoundedPatch] = []

    for i in range(n):
        if visited[i]:
            continue
        comp = []
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
            comp_patches = [patches[idx] for idx in comp]
            fbx1 = min(p.box[0] for p in comp_patches)
            fby1 = min(p.box[1] for p in comp_patches)
            fbx2 = max(p.box[2] for p in comp_patches)
            fby2 = max(p.box[3] for p in comp_patches)
            fused_box = (fbx1, fby1, fbx2, fby2)
            max_score = max((p.confidence_score if p.confidence_score is not None else 1.0) for p in comp_patches)
            max_siglip = max((p.siglip_score if p.siglip_score is not None else 0.0) for p in comp_patches)
            max_entropy = max(p.entropy for p in comp_patches)

            constituent_boxes = []
            for p in comp_patches:
                if hasattr(p, "constituent_boxes") and p.constituent_boxes:
                    constituent_boxes.extend(p.constituent_boxes)
                else:
                    constituent_boxes.append(p.box)

            fused_p = BoundedPatch(
                img_idx=comp_patches[0].img_idx,
                box=fused_box,
                feature_vector=comp_patches[0].feature_vector,
                entropy=max_entropy,
                confidence_score=max_score,
                siglip_score=max_siglip,
                parent_id=comp_patches[0].parent_id,
                level=min(p.level for p in comp_patches)
            )
            setattr(fused_p, "constituent_boxes", constituent_boxes)
            setattr(fused_p, "is_fused", True)
            fused_patches.append(fused_p)

    return fused_patches


def partition_continuous_enclosure_subcanvases(
    candidates: List[BoundedPatch],
    image_shape: Tuple[int, int],
    max_dim: int = 1024
) -> List[List[BoundedPatch]]:
    """
    Pillar 2 & 3: Continuous Enclosure Windowing & Sub-Canvas Dispatch.
    Greedily aggregates candidates into active sub-canvas clusters if their
    bounding union span max(Delta X, Delta Y) <= max_dim (1024px).
    Dispatches to next sub-canvas whenever the span exceeds max_dim.
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return [candidates]

    def _sort_key(p: BoundedPatch):
        sc = p.confidence_score if p.confidence_score is not None else 0.0
        sg = p.siglip_score if p.siglip_score is not None else 0.0
        area = (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
        return (sc, sg, area)

    sorted_cands = sorted(candidates, key=_sort_key, reverse=True)

    clusters: List[List[BoundedPatch]] = []
    current_cluster: List[BoundedPatch] = []

    for cand in sorted_cands:
        if not current_cluster:
            current_cluster.append(cand)
            continue

        test_members = current_cluster + [cand]
        ux1 = min(p.box[0] for p in test_members)
        uy1 = min(p.box[1] for p in test_members)
        ux2 = max(p.box[2] for p in test_members)
        uy2 = max(p.box[3] for p in test_members)
        span_x = ux2 - ux1
        span_y = uy2 - uy1

        if span_x <= max_dim and span_y <= max_dim:
            current_cluster.append(cand)
        else:
            clusters.append(current_cluster)
            current_cluster = [cand]

    if current_cluster:
        clusters.append(current_cluster)

    return clusters
