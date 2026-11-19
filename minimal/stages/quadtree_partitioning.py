"""
Stage 2: Query-Guided Adaptive Partitioning with Saliency Plateau Stopping.
Decomposes gallery images into aspect-ratio aware quadrants, steered by visual alignment gain.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Union
import numpy as np
from PIL import Image

from ..config import STASHConfig
from ..models.base import BaseDualEncoder


@dataclass
class BoundedPatch:
    """Represents a candidate visual patch retained during quadtree decomposition."""
    img_idx: int
    box: Tuple[int, int, int, int]
    depth: int
    alignment_score: Optional[float] = None
    entropy: float = 0.0
    parent_box: Optional[Tuple[int, int, int, int]] = None

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> int:
        return self.width * self.height


def get_quadtree_splits(w: int, h: int) -> Tuple[int, int]:
    """
    Determines grid layout based on aspect ratio:
    - (1, 4) if H/W >= 2.0 (tall vertical slice)
    - (4, 1) if H/W <= 0.5 (wide horizontal slice)
    - (2, 2) otherwise (standard quadrants)
    """
    if h / max(1, w) >= 2.0:
        return (1, 4)
    if h / max(1, w) <= 0.5:
        return (4, 1)
    return (2, 2)


def compute_spatial_entropy(
    crop_img: Image.Image,
    weights: Tuple[float, float, float] = (0.35, 0.35, 0.30)
) -> float:
    """
    Computes Tri-Factor Spatial Entropy H(I) = w1 * H_Shannon + w2 * G_Sobel + w3 * V_Laplacian.
    Accelerated with bilinear downsampling (max dimension 512px) to prevent CPU bottlenecks.
    """
    max_dim = max(crop_img.width, crop_img.height)
    if max_dim > 512:
        scale = 512.0 / max_dim
        safe_img = crop_img.resize(
            (int(round(crop_img.width * scale)), int(round(crop_img.height * scale))),
            Image.Resampling.BILINEAR
        )
    else:
        safe_img = crop_img

    gray = np.array(safe_img.convert("L"), dtype=np.uint8)

    # 1. Shannon Entropy
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist_prob = hist / (hist.sum() + 1e-12)
    hist_nonzero = hist_prob[hist_prob > 0]
    shannon_entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    norm_shannon = shannon_entropy / 8.0  # Max entropy for 256 bins is 8.0 bits

    # 2. Sobel Gradient Density & 3. Laplacian Variance
    try:
        import cv2
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        mean_grad = float(np.mean(sobel_mag))
        norm_sobel = min(1.0, mean_grad / 64.0)

        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        var_lap = float(np.var(laplacian))
        norm_lap = min(1.0, math.log1p(var_lap) / 10.0)
    except Exception:
        # NumPy fallback if cv2 is not installed
        gx, gy = np.gradient(gray.astype(np.float32))
        sobel_mag = np.sqrt(gx ** 2 + gy ** 2)
        mean_grad = float(np.mean(sobel_mag))
        norm_sobel = min(1.0, mean_grad / 64.0)
        var_lap = float(np.var(gx + gy))
        norm_lap = min(1.0, math.log1p(var_lap) / 10.0)

    w1, w2, w3 = weights
    h_total = w1 * norm_shannon + w2 * norm_sobel + w3 * norm_lap
    return float(np.clip(h_total, 0.0, 1.0))


def split_box_into_quadrants(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    overlap_ratio: float = 0.10
) -> List[Tuple[int, int, int, int]]:
    """
    Subdivides bounding box into 4 subpatches with overlap margins along cut seams.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    nx, ny = get_quadtree_splits(w, h)
    w_stride = w // nx
    h_stride = h // ny

    subpatches: List[Tuple[int, int, int, int]] = []
    for j in range(ny):
        for i in range(nx):
            sx1 = x1 + i * w_stride
            sy1 = y1 + j * h_stride
            sx2 = x2 if i == nx - 1 else sx1 + w_stride
            sy2 = y2 if j == ny - 1 else sy1 + h_stride

            sub_w = sx2 - sx1
            sub_h = sy2 - sy1

            pad_x = int(sub_w * overlap_ratio) if nx > 1 else 0
            pad_y = int(sub_h * overlap_ratio) if ny > 1 else 0

            subpatches.append((
                max(0, sx1 - pad_x),
                max(0, sy1 - pad_y),
                min(img_w, sx2 + pad_x),
                min(img_h, sy2 + pad_y)
            ))

    return subpatches


def partition_single_image(
    image: Image.Image,
    img_idx: int,
    active_objects: List[str],
    dual_encoder: BaseDualEncoder,
    config: STASHConfig,
    return_stats: bool = False
) -> Union[List[BoundedPatch], Tuple[List[BoundedPatch], int]]:
    """
    Recursively partitions a single gallery image using saliency plateau stopping.
    """
    img_w, img_h = image.size
    work_queue: List[Tuple[Tuple[int, int, int, int], int, Optional[Tuple[int, int, int, int]]]] = [
        ((0, 0, img_w, img_h), 0, None)
    ]

    leaf_patches: List[BoundedPatch] = []
    pruned_entropy_count = 0

    while work_queue:
        box, depth, parent_box = work_queue.pop(0)
        x1, y1, x2, y2 = box
        pw, ph = x2 - x1, y2 - y1
        max_dim = max(pw, ph)

        sub_boxes: Optional[List[Tuple[int, int, int, int]]] = None
        s_parent: Optional[float] = None

        if depth == 0:
            should_split = True
        elif max_dim > config.max_plateau_size:
            # Mandatory split above physical ceiling to prevent downsampling blur
            should_split = True
        elif depth >= config.max_partition_depth:
            should_split = False
        elif max_dim <= config.min_plateau_size:
            should_split = False
        else:
            # Scale range [min_plateau_size, max_plateau_size]: evaluate saliency plateau
            candidate_children = split_box_into_quadrants(
                box, img_w, img_h, overlap_ratio=config.quadrant_overlap_ratio
            )
            crop_parent = image.crop(box)
            child_crops = [image.crop(cb) for cb in candidate_children]

            all_crops = [crop_parent] + child_crops
            scores = dual_encoder.score_crops(all_crops, active_objects)
            s_parent = float(scores[0])
            s_children = [float(s) for s in scores[1:]]
            s_max_child = max(s_children) if s_children else 0.0

            delta_s = s_max_child - s_parent

            # Plateau condition: marginal alignment gain <= threshold (Delta S <= 0.04)
            is_plateau = (delta_s <= config.saliency_plateau_gain_threshold)

            if is_plateau:
                should_split = False
            else:
                should_split = True
                # Descend into promising child quadrants
                promising = [
                    cb for cb, sc in zip(candidate_children, s_children)
                    if sc >= (s_parent - 0.05) or sc == s_max_child
                ]
                sub_boxes = promising if promising else candidate_children

        if should_split:
            if sub_boxes is None:
                sub_boxes = split_box_into_quadrants(
                    box, img_w, img_h, overlap_ratio=config.quadrant_overlap_ratio
                )
            for sb in sub_boxes:
                work_queue.append((sb, depth + 1, box))
        else:
            # Leaf termination: compute tri-factor spatial entropy
            crop = image.crop(box)
            entropy = compute_spatial_entropy(crop, weights=getattr(config, "entropy_weights", (0.35, 0.35, 0.30)))

            # Entropy pruning: drop low-entropy leaves unless visual alignment score >= saliency_target_threshold (0.35)
            min_entropy = getattr(config, "min_patch_entropy", 0.35)
            target_thresh = getattr(config, "saliency_target_threshold", 0.35)
            if min_entropy > 0.0 and entropy < min_entropy:
                if s_parent is None or s_parent < target_thresh:
                    pruned_entropy_count += 1
                    continue

            leaf_patches.append(
                BoundedPatch(
                    img_idx=img_idx,
                    box=box,
                    depth=depth,
                    alignment_score=s_parent,
                    entropy=entropy,
                    parent_box=parent_box
                )
            )

    # Score any leaf patches whose alignment scores were not evaluated during plateau checks
    unscored = [p for p in leaf_patches if p.alignment_score is None]
    if unscored:
        unscored_crops = [image.crop(p.box) for p in unscored]
        scores = dual_encoder.score_crops(unscored_crops, active_objects)
        for p, s in zip(unscored, scores):
            p.alignment_score = float(s)

    # Rank by visual alignment score descending (with 1.0 / area tie-breaker preferring deeper/smaller patches)
    leaf_patches.sort(
        key=lambda p: (
            p.alignment_score if p.alignment_score is not None else 0.0,
            1.0 / max(1, p.area)
        ),
        reverse=True
    )
    retained = leaf_patches[:config.max_patches_per_image]
    if return_stats:
        return retained, pruned_entropy_count
    return retained


def partition_gallery(
    gallery_images: List[Image.Image],
    active_objects: List[str],
    dual_encoder: BaseDualEncoder,
    config: STASHConfig,
    return_stats: bool = False
) -> Union[List[BoundedPatch], Tuple[List[BoundedPatch], int]]:
    """
    Executes Query-Guided Adaptive Partitioning across all gallery images.

    Returns:
        Unified pool of candidate leaf patches P across all images (and optionally total leaves pruned by entropy).
    """
    pool: List[BoundedPatch] = []
    total_pruned_entropy = 0

    for idx, img in enumerate(gallery_images):
        patches, pruned = partition_single_image(
            img, idx, active_objects, dual_encoder, config, return_stats=True
        )
        pool.extend(patches)
        total_pruned_entropy += pruned

    if getattr(config, "verbose", True):
        print(f"[Stage 2: Partitioning] Partitioned {len(gallery_images)} images into {len(pool)} candidate patches "
              f"({total_pruned_entropy} leaves pruned by entropy H < {getattr(config, 'min_patch_entropy', 0.35):.2f}).")

    if return_stats:
        return pool, total_pruned_entropy
    return pool
