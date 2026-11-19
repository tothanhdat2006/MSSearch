"""
Stage 2: Query-Guided Adaptive Partitioning with Saliency Plateau Stopping.
Decomposes gallery images into aspect-ratio aware quadrants, steered by visual alignment gain.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any
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
    config: STASHConfig
) -> List[BoundedPatch]:
    """
    Recursively partitions a single gallery image using saliency plateau stopping.
    """
    img_w, img_h = image.size
    work_queue: List[Tuple[Tuple[int, int, int, int], int, Optional[Tuple[int, int, int, int]]]] = [
        ((0, 0, img_w, img_h), 0, None)
    ]

    leaf_patches: List[BoundedPatch] = []

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
            leaf_patches.append(
                BoundedPatch(
                    img_idx=img_idx,
                    box=box,
                    depth=depth,
                    alignment_score=s_parent,
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
    return leaf_patches[:config.max_patches_per_image]


def partition_gallery(
    gallery_images: List[Image.Image],
    active_objects: List[str],
    dual_encoder: BaseDualEncoder,
    config: STASHConfig
) -> List[BoundedPatch]:
    """
    Executes Query-Guided Adaptive Partitioning across all gallery images.

    Returns:
        Unified pool of candidate leaf patches P across all images.
    """
    pool: List[BoundedPatch] = []
    for idx, img in enumerate(gallery_images):
        patches = partition_single_image(img, idx, active_objects, dual_encoder, config)
        pool.extend(patches)

    if config.verbose:
        print(f"[Stage 2: Partitioning] Partitioned {len(gallery_images)} images into {len(pool)} candidate patches.")

    return pool
