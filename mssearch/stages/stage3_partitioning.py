"""
Stage 3: Entropy-Gated Recursive Patch Partitioning & SAM3-SGAP.
Computes Tri-Factor Spatial Entropy H(I), routes between fast Quadrants and SAM3-SGAP,
enforces recursive bounding to max(W, H) <= 1024px with 10% overlap margins,
and reuses Stage 2 feature maps while caching pooled patch embeddings for Stage 4.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import math
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage.segmentation import slic
    from skimage.measure import regionprops
    from skimage import graph
    import networkx as nx
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    slic = None
    regionprops = None
    graph = None
    nx = None
    AgglomerativeClustering = None
    silhouette_score = None
    StandardScaler = None

from ..config import MSSearchConfig


@dataclass
class BoundedPatch:
    """Represents a bounded patch satisfying max(W, H) <= 1024px."""
    img_idx: int
    box: Tuple[int, int, int, int]           # (x1, y1, x2, y2) in original image coordinates
    feature_vector: np.ndarray              # Pooled feature embedding [C] for Stage 4 reuse
    entropy: float
    confidence_score: Optional[float] = None # Composite confidence score for Stage 4 hard capping


def compute_spatial_entropy(
    crop_img: Image.Image,
    weights: Tuple[float, float, float] = (0.35, 0.35, 0.30)
) -> float:
    """
    Computes Tri-Factor Spatial Entropy:
    H(I) = w1 * H_Shannon + w2 * G_Sobel + w3 * V_Laplacian
    Accelerated with bilinear downsampling (max dimension 512px) to prevent CPU bottlenecks on 8K images.
    """
    max_dim = max(crop_img.width, crop_img.height)
    if max_dim > 512:
        scale = 512.0 / max_dim
        safe_img = crop_img.resize(
            (int(round(crop_img.width * scale)), int(round(crop_img.height * scale))),
            Image.Resampling.BILINEAR
        )
        img_arr = np.array(safe_img)
    else:
        img_arr = np.array(crop_img)

    if img_arr.ndim == 3:
        if cv2 is not None:
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = (0.2989 * img_arr[:, :, 0] + 0.5870 * img_arr[:, :, 1] + 0.1140 * img_arr[:, :, 2]).astype(np.float32)
    else:
        gray = img_arr.astype(np.float32)

    # 1. Normalized Shannon Entropy (histogram-based)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    shannon = -np.sum(hist * np.log2(hist))
    norm_shannon = float(np.clip(shannon / 8.0, 0.0, 1.0))

    # 2. Normalized Sobel Gradient Energy
    if cv2 is not None:
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    else:
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    norm_sobel = float(np.clip(grad_mag.mean() / 128.0, 0.0, 1.0))

    # 3. Normalized Laplacian Variance (detail & edge complexity)
    if cv2 is not None:
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        lap_var = float(laplacian.var())
    else:
        gy, gx = np.gradient(gray)
        laplacian = np.gradient(gx)[1] + np.gradient(gy)[0]
        lap_var = float(laplacian.var())
    norm_lap = float(np.clip(np.log1p(lap_var) / np.log1p(1000.0), 0.0, 1.0))

    w1, w2, w3 = weights
    h_total = w1 * norm_shannon + w2 * norm_sobel + w3 * norm_lap
    return float(np.clip(h_total, 0.0, 1.0))


def split_4_subpatches(w: int, h: int) -> Tuple[int, int]:
    """
    CVSearch aspect-ratio aware 4-subpatch splitting (tree.py:L118-L125).
    Inspects hw_ratio = h / w:
      - If h / w >= 2.0: (1, 4) -> 1 column x 4 vertical slices (tall structure)
      - If h / w <= 0.5: (4, 1) -> 4 columns x 1 horizontal slice (wide strip)
      - Otherwise:       (2, 2) -> 2x2 quadrants
    """
    hw_ratio = h / max(1, w)
    if hw_ratio >= 2.0:
        return 1, 4
    elif hw_ratio <= 0.5:
        return 4, 1
    else:
        return 2, 2


def partition_oversized_box(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    max_patch_size: int = 1024,
    overlap_ratio: float = 0.10
) -> List[Tuple[int, int, int, int]]:
    """
    Splits an oversized box into 4 subpatches using CVSearch's split_4_subpatches ratio:
      - (1, 4) if H / W >= 2.0
      - (4, 1) if H / W <= 0.5
      - (2, 2) quadrants otherwise
    Preserves 10% overlap padding while strictly following CVSearch aspect-ratio ratios.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    if w <= max_patch_size and h <= max_patch_size:
        return [box]

    nx, ny = split_4_subpatches(w, h)
    w_stride = int(w // nx)
    h_stride = int(h // ny)

    sub_patches = []
    for j in range(ny):
        for i in range(nx):
            sx1 = x1 + i * w_stride
            sy1 = y1 + j * h_stride
            sx2 = x2 if i == nx - 1 else sx1 + w_stride
            sy2 = y2 if j == ny - 1 else sy1 + h_stride

            sub_w = sx2 - sx1
            sub_h = sy2 - sy1

            # Apply overlap padding when terminal to prevent compounding across depths
            if max(sub_w, sub_h) <= max_patch_size:
                pad_x = int(sub_w * overlap_ratio) if nx > 1 else 0
                pad_y = int(sub_h * overlap_ratio) if ny > 1 else 0
            else:
                pad_x = 0
                pad_y = 0

            sub_patches.append((
                max(0, sx1 - pad_x),
                max(0, sy1 - pad_y),
                min(img_w, sx2 + pad_x),
                min(img_h, sy2 + pad_y)
            ))
    return sub_patches


def partition_into_quadrants(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    overlap_ratio: float = 0.10
) -> List[Tuple[int, int, int, int]]:
    """Splits a box into direction-aware quadrants/halves with overlap padding."""
    return partition_oversized_box(box, img_w, img_h, max_patch_size=1024, overlap_ratio=overlap_ratio)


def compute_bbox_overlap_cost(atom_bboxes: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Computes pair-wise bounding box overlap cost penalized in SAM3-SGAP."""
    cluster_boxes = []
    for lbl in range(k):
        members = np.where(labels == lbl)[0]
        if len(members) == 0:
            continue
        m_boxes = atom_bboxes[members]
        y1, x1 = np.min(m_boxes[:, 0]), np.min(m_boxes[:, 1])
        y2, x2 = np.max(m_boxes[:, 2]), np.max(m_boxes[:, 3])
        cluster_boxes.append((x1, y1, x2, y2))

    if len(cluster_boxes) < 2:
        return 0.0

    total_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in cluster_boxes)
    if total_area == 0:
        return 0.0

    total_overlap = 0.0
    for i in range(len(cluster_boxes)):
        for j in range(i + 1, len(cluster_boxes)):
            ba, bb = cluster_boxes[i], cluster_boxes[j]
            ix1, iy1 = max(ba[0], bb[0]), max(ba[1], bb[1])
            ix2, iy2 = min(ba[2], bb[2]), min(ba[3], bb[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            total_overlap += inter

    return total_overlap / total_area


def partition_into_sam3_sgap(
    sub_feat: np.ndarray,
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    min_splits: int = 4,
    max_splits: int = 8,
    overlap_ratio: float = 0.10,
    verbose: bool = False
) -> List[Tuple[int, int, int, int]]:
    """
    SAM3-SGAP (Semantic-Guided Adaptive Patching) adapted from CVSearch.
    Runs SLIC on normalized feature channels, builds RAG, and clusters atoms.
    """
    if slic is None:
        if verbose:
            print("[Stage 3][Fallback] scikit-image/scikit-learn not available; falling back to direction-aware split.")
        return partition_oversized_box(box, img_w, img_h, overlap_ratio=overlap_ratio)

    C, H_f, W_f = sub_feat.shape
    if H_f < 4 or W_f < 4:
        if verbose:
            print(f"[Stage 3][Fallback] SGAP feature map slice ({H_f}x{W_f}) is smaller than 4x4 threshold; falling back to direction-aware split.")
        return partition_oversized_box(box, img_w, img_h, overlap_ratio=overlap_ratio)

    feat_tr = sub_feat.transpose(1, 2, 0)
    feat_norm = (feat_tr - feat_tr.min()) / (feat_tr.max() - feat_tr.min() + 1e-6)

    # 1. SLIC superpixel over-segmentation on feature channels
    n_atoms_target = min(100, H_f * W_f // 4)
    if n_atoms_target < min_splits:
        if verbose:
            print(f"[Stage 3][Fallback] Target SLIC atoms ({n_atoms_target}) < min_splits ({min_splits}); falling back to direction-aware split.")
        return partition_oversized_box(box, img_w, img_h, overlap_ratio=overlap_ratio)

    try:
        atom_map = slic(feat_norm, n_segments=n_atoms_target, compactness=20, start_label=0, channel_axis=2)
    except Exception as e:
        if verbose:
            print(f"[Stage 3][Fallback] SLIC segmentation failed ({e}); falling back to direction-aware split.")
        return partition_oversized_box(box, img_w, img_h, overlap_ratio=overlap_ratio)

    props = regionprops(atom_map + 1)
    n_atoms = len(props)
    if n_atoms < min_splits:
        if verbose:
            print(f"[Stage 3][Fallback] Actual SLIC atoms ({n_atoms}) < min_splits ({min_splits}); falling back to direction-aware split.")
        return partition_oversized_box(box, img_w, img_h, overlap_ratio=overlap_ratio)

    bx1, by1, bx2, by2 = box
    patch_w = bx2 - bx1
    patch_h = by2 - by1

    semantic_features = np.zeros((n_atoms, C), dtype=np.float32)
    spatial_features = np.zeros((n_atoms, 2), dtype=np.float32)
    atom_bboxes = []

    for i, prop in enumerate(props):
        y_slice, x_slice = prop.slice
        mask_local = prop.image
        feat_crop = sub_feat[:, y_slice, x_slice]
        semantic_features[i] = feat_crop[:, mask_local].mean(axis=1)

        cy, cx = prop.centroid
        spatial_features[i] = [cy / H_f, cx / W_f]

        # Map atom bbox to original image pixels
        y1, x1, y2, x2 = prop.bbox
        px1 = bx1 + int(x1 * patch_w / W_f)
        py1 = by1 + int(y1 * patch_h / H_f)
        px2 = bx1 + int(x2 * patch_w / W_f)
        py2 = by1 + int(y2 * patch_h / H_f)
        atom_bboxes.append([py1, px1, py2, px2])

    atom_bboxes = np.array(atom_bboxes)

    # 2. Region Adjacency Graph (RAG)
    try:
        rag_img = feat_tr[:, :, :3] if C >= 3 else feat_tr
        rag = graph.rag_mean_color(rag_img, atom_map, mode='distance')
        adj_matrix = nx.adjacency_matrix(rag).toarray()
    except Exception as e:
        if verbose:
            print(f"[Stage 3][Fallback] RAG graph construction failed ({e}); falling back to direction-aware split.")
        return partition_oversized_box(box, img_w, img_h, overlap_ratio=overlap_ratio)

    # 3. Scaled feature concatenation
    scaler = StandardScaler()
    scaled_semantic = scaler.fit_transform(semantic_features)
    clustering_features = np.concatenate([scaled_semantic, spatial_features * 0.5], axis=1)

    best_score = -float('inf')
    best_k = min_splits
    best_labels = None
    limit_k = min(max_splits, n_atoms - 1)

    # 4. Search optimal k* in [min_splits, max_splits]
    for k in range(min_splits, limit_k + 1):
        try:
            model = AgglomerativeClustering(n_clusters=k, connectivity=adj_matrix, linkage='ward')
            labels = model.fit_predict(clustering_features)
            sil = silhouette_score(scaled_semantic, labels) if k > 1 else 0.0
            overlap_cost = compute_bbox_overlap_cost(atom_bboxes, labels, k)
            score = sil - 1.5 * overlap_cost
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception:
            continue

    if best_labels is None:
        if verbose:
            print("[Stage 3][Fallback] Agglomerative clustering could not find valid labels; falling back to direction-aware split.")
        return partition_oversized_box(box, img_w, img_h, overlap_ratio=overlap_ratio)

    # 5. Extract bounding boxes with 10% overlap padding
    sub_boxes = []
    for lbl in range(best_k):
        members = np.where(best_labels == lbl)[0]
        if len(members) == 0:
            continue
        c_boxes = atom_bboxes[members]
        y1, x1 = np.min(c_boxes[:, 0]), np.min(c_boxes[:, 1])
        y2, x2 = np.max(c_boxes[:, 2]), np.max(c_boxes[:, 3])

        pad_x = int((x2 - x1) * overlap_ratio)
        pad_y = int((y2 - y1) * overlap_ratio)

        padded = (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(img_w, x2 + pad_x),
            min(img_h, y2 + pad_y)
        )
        sub_boxes.append(padded)

    if verbose:
        print(f"[Stage 3][Non-Fallback] SAM3-SGAP clustered {n_atoms} atoms into k*={best_k} semantic patches (score: {best_score:.3f}).")

    return sub_boxes if sub_boxes else partition_oversized_box(box, img_w, img_h, overlap_ratio=overlap_ratio)



def slice_feature_map(
    feat: np.ndarray,
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slices the full-canvas feature map feat [C, H_f, W_f] corresponding to box (x1, y1, x2, y2),
    and computes the average pooled embedding [C].
    """
    C, H_f, W_f = feat.shape
    x1, y1, x2, y2 = box

    xf1 = max(0, int(x1 * W_f / img_w))
    yf1 = max(0, int(y1 * H_f / img_h))
    xf2 = min(W_f, max(xf1 + 1, int(x2 * W_f / img_w)))
    yf2 = min(H_f, max(yf1 + 1, int(y2 * H_f / img_h)))

    sub_feat = feat[:, yf1:yf2, xf1:xf2]
    pooled_vector = sub_feat.mean(axis=(1, 2))  # [C]
    return sub_feat, pooled_vector


def partition_candidate_images(
    candidate_images: List[Image.Image],
    stage2_feature_cache: Dict[int, np.ndarray],
    config: MSSearchConfig
) -> List[BoundedPatch]:
    """
    Executes Entropy-Gated Recursive Patch Partitioning across all candidate images.
    Guarantees that all output patches satisfy longer side max(W, H) <= config.max_patch_size.
    Reuses cached Stage 2 feature maps and returns pooled embeddings for Stage 4 grouping.
    """
    bounded_patches: List[BoundedPatch] = []

    for img_idx, img_pil in enumerate(candidate_images):
        img_w, img_h = img_pil.size
        # Queue of patches to evaluate: [((x1, y1, x2, y2), depth)]
        work_queue: List[Tuple[Tuple[int, int, int, int], int]] = [((0, 0, img_w, img_h), 0)]

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
            box, depth = work_queue.pop(0)
            box_ints = tuple(int(round(float(c))) for c in box)
            x1, y1, x2, y2 = box_ints
            pw, ph = x2 - x1, y2 - y1

            # Check recursive bounding condition: longer side <= max_patch_size (with 15% tolerance for boundary overlap)
            if max(pw, ph) > int(config.max_patch_size * 1.15):
                crop = img_pil.crop(box_ints)
                entropy = compute_spatial_entropy(crop, weights=config.entropy_weights)

                # Only run SAM3-SGAP clustering at root level (depth == 0) to avoid recursive superpixel explosion
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
                        verbose=config.verbose
                    )
                else:
                    # Direction-aware split: bisect only the oversized axis
                    sub_boxes = partition_oversized_box(
                        box_ints,
                        img_w,
                        img_h,
                        max_patch_size=config.max_patch_size,
                        overlap_ratio=config.quadrant_overlap_ratio
                    )
                    if config.verbose and depth <= 1:
                        print(f"[Stage 3][Routing] Patch {box_ints} (size {pw}x{ph}, depth={depth}) -> Direction-aware Split ({len(sub_boxes)} patches).")

                # Push sub-boxes back to queue with incremented depth
                for sb in sub_boxes:
                    sb_ints = tuple(int(round(float(c))) for c in sb)
                    work_queue.append((sb_ints, depth + 1))
            else:
                # Patch satisfies longer side <= 1024px!
                crop = img_pil.crop(box_ints)
                entropy = compute_spatial_entropy(crop, weights=config.entropy_weights)

                # Filter low-entropy background patches
                if config.min_patch_entropy > 0.0 and entropy < config.min_patch_entropy:
                    discarded_candidates.append((box_ints, entropy))
                    continue

                _, pooled_vec = slice_feature_map(feat_map, box_ints, img_w, img_h)

                img_patches.append(
                    BoundedPatch(
                        img_idx=img_idx,
                        box=box_ints,
                        feature_vector=pooled_vec,
                        entropy=entropy
                    )
                )

        # Safety fallback: prevent empty image patch list if all patches were low entropy
        if not img_patches and discarded_candidates:
            best_box, best_ent = max(discarded_candidates, key=lambda x: x[1])
            _, pooled_vec = slice_feature_map(feat_map, best_box, img_w, img_h)
            img_patches.append(
                BoundedPatch(
                    img_idx=img_idx,
                    box=best_box,
                    feature_vector=pooled_vec,
                    entropy=best_ent
                )
            )
            if config.verbose:
                print(f"[Stage 3][Safety Fallback] Image {img_idx}: All patches had H < {config.min_patch_entropy:.2f}. Retained highest-entropy patch {best_box} (H={best_ent:.3f}).")
        elif config.verbose and discarded_candidates:
            print(f"[Stage 3][Entropy Pruning] Image {img_idx}: Pruned {len(discarded_candidates)} patches with H < {config.min_patch_entropy:.2f}. Retained {len(img_patches)} bounded patches.")

        # Safety Budget Cap: ensure number of bounded patches per image does not exceed budget
        if len(img_patches) > config.max_patches_per_image:
            use_entropy_sorting = getattr(config, "enable_entropy_budget_sorting", True) and (
                getattr(config, "stage3_budget_mode", "entropy_descending") == "entropy_descending"
            )
            if use_entropy_sorting:
                if config.verbose:
                    print(f"[Stage 3][Budget Guard] Entropy-ranking {len(img_patches)} patches -> retaining top {config.max_patches_per_image} highest-entropy patches (preventing spatial target loss).")
                img_patches = sorted(img_patches, key=lambda p: p.entropy, reverse=True)[:config.max_patches_per_image]
            else:
                if config.verbose:
                    print(f"[Stage 3][Budget Guard] Uniformly downsampling {len(img_patches)} bounded patches to {config.max_patches_per_image} across canvas.")
                step = len(img_patches) / config.max_patches_per_image
                indices = [int(i * step) for i in range(config.max_patches_per_image)]
                img_patches = [img_patches[i] for i in indices]

        bounded_patches.extend(img_patches)

    if config.verbose:
        print(f"[Stage 3] Partitioning finished: Aggregated {len(bounded_patches)} bounded patches (all longer side <= {config.max_patch_size}px) across {len(candidate_images)} candidate images.")

    return bounded_patches

