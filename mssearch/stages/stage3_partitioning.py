"""
Stage 3: Entropy-Gated Recursive Patch Partitioning & SAM3-SGAP.
Computes Tri-Factor Spatial Entropy H(I), routes between fast Quadrants and SAM3-SGAP,
enforces recursive bounding to max(W, H) <= 1024px with 10% overlap margins,
and reuses Stage 2 feature maps while caching pooled patch embeddings for Stage 4.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from PIL import Image
import cv2
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage import graph
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..config import MSSearchConfig


@dataclass
class BoundedPatch:
    """Represents a bounded patch satisfying max(W, H) <= 1024px."""
    img_idx: int
    box: Tuple[int, int, int, int]           # (x1, y1, x2, y2) in original image coordinates
    feature_vector: np.ndarray              # Pooled feature embedding [C] for Stage 4 reuse
    entropy: float


def compute_spatial_entropy(
    crop_img: Image.Image,
    weights: Tuple[float, float, float] = (0.35, 0.35, 0.30)
) -> float:
    """
    Computes Tri-Factor Spatial Entropy:
    H(I) = w1 * H_Shannon + w2 * G_Sobel + w3 * V_Laplacian
    """
    img_arr = np.array(crop_img)
    if img_arr.ndim == 3:
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_arr

    # 1. Normalized Shannon Entropy (histogram-based)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    shannon = -np.sum(hist * np.log2(hist))
    norm_shannon = float(np.clip(shannon / 8.0, 0.0, 1.0))

    # 2. Normalized Sobel Gradient Energy
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    norm_sobel = float(np.clip(grad_mag.mean() / 128.0, 0.0, 1.0))

    # 3. Normalized Laplacian Variance (detail & edge complexity)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(laplacian.var())
    norm_lap = float(np.clip(np.log1p(lap_var) / np.log1p(1000.0), 0.0, 1.0))

    w1, w2, w3 = weights
    h_total = w1 * norm_shannon + w2 * norm_sobel + w3 * norm_lap
    return float(np.clip(h_total, 0.0, 1.0))


def partition_into_quadrants(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    overlap_ratio: float = 0.10
) -> List[Tuple[int, int, int, int]]:
    """Splits a box into 2x2 quadrants with 10% overlap padding."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    mid_x = x1 + w // 2
    mid_y = y1 + h // 2

    pad_x = int(w * overlap_ratio)
    pad_y = int(h * overlap_ratio)

    quads = [
        # Top-Left
        (max(0, x1 - pad_x), max(0, y1 - pad_y), min(img_w, mid_x + pad_x), min(img_h, mid_y + pad_y)),
        # Top-Right
        (max(0, mid_x - pad_x), max(0, y1 - pad_y), min(img_w, x2 + pad_x), min(img_h, mid_y + pad_y)),
        # Bottom-Left
        (max(0, x1 - pad_x), max(0, mid_y - pad_y), min(img_w, mid_x + pad_x), min(img_h, y2 + pad_y)),
        # Bottom-Right
        (max(0, mid_x - pad_x), max(0, mid_y - pad_y), min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)),
    ]
    return quads


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
    min_splits: int = 8,
    max_splits: int = 16,
    overlap_ratio: float = 0.10,
    verbose: bool = False
) -> List[Tuple[int, int, int, int]]:
    """
    SAM3-SGAP (Semantic-Guided Adaptive Patching) adapted from CVSearch.
    Runs SLIC on normalized feature channels, builds RAG, and clusters atoms.
    """
    C, H_f, W_f = sub_feat.shape
    if H_f < 4 or W_f < 4:
        if verbose:
            print(f"[Stage 3][Fallback] SGAP feature map slice ({H_f}x{W_f}) is smaller than 4x4 threshold; falling back to 2x2 Quadrant split.")
        return partition_into_quadrants(box, img_w, img_h, overlap_ratio)

    feat_tr = sub_feat.transpose(1, 2, 0)
    feat_norm = (feat_tr - feat_tr.min()) / (feat_tr.max() - feat_tr.min() + 1e-6)

    # 1. SLIC superpixel over-segmentation on feature channels
    n_atoms_target = min(100, H_f * W_f // 4)
    if n_atoms_target < min_splits:
        if verbose:
            print(f"[Stage 3][Fallback] Target SLIC atoms ({n_atoms_target}) < min_splits ({min_splits}); falling back to 2x2 Quadrant split.")
        return partition_into_quadrants(box, img_w, img_h, overlap_ratio)

    try:
        atom_map = slic(feat_norm, n_segments=n_atoms_target, compactness=20, start_label=0, channel_axis=2)
    except Exception as e:
        if verbose:
            print(f"[Stage 3][Fallback] SLIC segmentation failed ({e}); falling back to 2x2 Quadrant split.")
        return partition_into_quadrants(box, img_w, img_h, overlap_ratio)

    props = regionprops(atom_map + 1)
    n_atoms = len(props)
    if n_atoms < min_splits:
        if verbose:
            print(f"[Stage 3][Fallback] Actual SLIC atoms ({n_atoms}) < min_splits ({min_splits}); falling back to 2x2 Quadrant split.")
        return partition_into_quadrants(box, img_w, img_h, overlap_ratio)

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
            print(f"[Stage 3][Fallback] RAG graph construction failed ({e}); falling back to 2x2 Quadrant split.")
        return partition_into_quadrants(box, img_w, img_h, overlap_ratio)

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
            print("[Stage 3][Fallback] Agglomerative clustering could not find valid labels; falling back to 2x2 Quadrant split.")
        return partition_into_quadrants(box, img_w, img_h, overlap_ratio)

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

    return sub_boxes if sub_boxes else partition_into_quadrants(box, img_w, img_h, overlap_ratio)



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
        # Queue of patches to evaluate: [(x1, y1, x2, y2)]
        work_queue: List[Tuple[int, int, int, int]] = [(0, 0, img_w, img_h)]

        # Get cached feature map or default empty
        feat_map = stage2_feature_cache.get(img_idx, None)
        if feat_map is not None:
            if config.verbose:
                print(f"[Stage 3][Non-Fallback] Image {img_idx}: Reusing cached Stage 2 vision feature map (Shape: {feat_map.shape}).")
        else:
            feat_map = np.zeros((256, 72, 72), dtype=np.float32)
            if config.verbose:
                print(f"[Stage 3][Fallback] Image {img_idx}: No Stage 2 feature map in cache; falling back to zero-initialized feature map.")

        while work_queue:
            box = work_queue.pop(0)
            box_ints = tuple(int(round(float(c))) for c in box)
            x1, y1, x2, y2 = box_ints
            pw, ph = x2 - x1, y2 - y1

            # Check recursive bounding condition: longer side <= max_patch_size
            if max(pw, ph) > config.max_patch_size:
                crop = img_pil.crop(box_ints)
                entropy = compute_spatial_entropy(crop, weights=config.entropy_weights)

                # Partitioning route
                if config.partition_mode == "quadrant_only" or (config.partition_mode == "hybrid" and entropy < config.entropy_threshold):
                    sub_boxes = partition_into_quadrants(
                        box_ints, img_w, img_h, overlap_ratio=config.quadrant_overlap_ratio
                    )
                    if config.verbose:
                        print(f"[Stage 3][Routing] Patch {box_ints} (size {pw}x{ph}) -> H={entropy:.3f} < {config.entropy_threshold:.2f} -> Quadrant 2x2 Split ({len(sub_boxes)} patches).")
                else:
                    # SAM3-SGAP (Semantic-Guided Adaptive Patching reusing Stage 2 features)
                    sub_feat, _ = slice_feature_map(feat_map, box_ints, img_w, img_h)
                    if config.verbose:
                        print(f"[Stage 3][Routing] Patch {box_ints} (size {pw}x{ph}) -> H={entropy:.3f} >= {config.entropy_threshold:.2f} -> SAM3-SGAP Partitioning.")
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

                # Push sub-boxes back to queue for recursive evaluation
                for sb in sub_boxes:
                    sb_ints = tuple(int(round(float(c))) for c in sb)
                    work_queue.append(sb_ints)
            else:
                # Patch satisfies longer side <= 1024px!
                crop = img_pil.crop(box_ints)
                entropy = compute_spatial_entropy(crop, weights=config.entropy_weights)
                _, pooled_vec = slice_feature_map(feat_map, box_ints, img_w, img_h)

                bounded_patches.append(
                    BoundedPatch(
                        img_idx=img_idx,
                        box=box_ints,
                        feature_vector=pooled_vec,
                        entropy=entropy
                    )
                )

    if config.verbose:
        print(f"[Stage 3] Partitioning finished: Aggregated {len(bounded_patches)} bounded patches (all longer side <= {config.max_patch_size}px) across {len(candidate_images)} candidate images.")

    return bounded_patches

