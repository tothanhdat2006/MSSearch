"""
Stage 4 Utilities: Feature Clustering, Geometry Coverage, Fovea Extraction, and VLM Confidence Probing.
"""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from PIL import Image

from sklearn.cluster import AgglomerativeClustering

from ..config import MSSearchConfig
from .stage3_partitioning import BoundedPatch


def find_group_representative(
    member_indices: List[int],
    bounded_patches: List[BoundedPatch]
) -> int:
    """
    Selects the representative patch for a group:
    Primary: highest siglip_score.
    Tiebreaker: minimum Euclidean distance to group feature centroid in embedding space.
    """
    if len(member_indices) == 1:
        return member_indices[0]

    g_feats = np.stack([bounded_patches[i].feature_vector for i in member_indices], axis=0)
    centroid = g_feats.mean(axis=0, keepdims=True)
    dists = np.linalg.norm(g_feats - centroid, axis=1)

    sorted_members = sorted(
        zip(member_indices, dists),
        key=lambda pair: (
            bounded_patches[pair[0]].siglip_score if bounded_patches[pair[0]].siglip_score is not None else 0.0,
            -pair[1]
        ),
        reverse=True
    )
    return sorted_members[0][0]


def group_patches_by_cosine_similarity(
    patches: List[BoundedPatch],
    sim_threshold: float = 0.80,
    return_sim_mat: bool = False
) -> Union[List[List[int]], Tuple[List[List[int]], np.ndarray]]:
    """
    Groups patches into semantic clusters based on Stage 3 pooled feature cosine similarity.
    Reuses the features extracted during Stage 3 with zero redundant network forward passes.
    """
    n_patches = len(patches)
    if n_patches == 0:
        empty_mat = np.zeros((0, 0), dtype=np.float32)
        return ([], empty_mat) if return_sim_mat else []
    if n_patches == 1:
        single_mat = np.ones((1, 1), dtype=np.float32)
        return ([[0]], single_mat) if return_sim_mat else [[0]]

    # 1. Normalize feature vectors
    feats = np.stack([p.feature_vector for p in patches], axis=0)  # [N, C]
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6
    norm_feats = feats / norms

    # 2. Pairwise Cosine Distance: D = 1.0 - CosineSimilarity
    sim_mat = np.matmul(norm_feats, norm_feats.T)
    dist_mat = np.clip(1.0 - sim_mat, 0.0, 2.0)

    # 3. Agglomerative Clustering with distance threshold
    dist_threshold = max(0.01, 1.0 - sim_threshold)
    if AgglomerativeClustering is not None:
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=dist_threshold,
                metric="precomputed",
                linkage="average"
            )
            labels = clustering.fit_predict(dist_mat)
            groups: Dict[int, List[int]] = {}
            for idx, lbl in enumerate(labels):
                groups.setdefault(lbl, []).append(idx)
            groups_list = list(groups.values())
            return (groups_list, sim_mat) if return_sim_mat else groups_list
        except Exception:
            pass

    # Native graph-connected components clustering (stdlib + numpy)
    visited = [False] * n_patches
    groups_list: List[List[int]] = []
    for i in range(n_patches):
        if visited[i]:
            continue
        cluster = []
        queue = [i]
        visited[i] = True
        while queue:
            curr = queue.pop(0)
            cluster.append(curr)
            for j in range(n_patches):
                if not visited[j] and sim_mat[curr, j] >= sim_threshold:
                    visited[j] = True
                    queue.append(j)
        groups_list.append(cluster)
    return (groups_list, sim_mat) if return_sim_mat else groups_list


def compute_target_coverage_ratio(patch_box: Tuple[int, int, int, int], target_box: Tuple[int, int, int, int]) -> float:
    """Computes the fraction of target_box covered by patch_box."""
    ix1, iy1 = max(patch_box[0], target_box[0]), max(patch_box[1], target_box[1])
    ix2, iy2 = min(patch_box[2], target_box[2]), min(patch_box[3], target_box[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter_area = float((ix2 - ix1) * (iy2 - iy1))
    target_area = float(max(1, (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])))
    return inter_area / target_area


def target_center_in_patch(patch_box: Tuple[int, int, int, int], target_box: Tuple[int, int, int, int]) -> bool:
    """Checks whether the center point of target_box lies inside patch_box."""
    cx = (target_box[0] + target_box[2]) / 2.0
    cy = (target_box[1] + target_box[3]) / 2.0
    return (patch_box[0] <= cx <= patch_box[2]) and (patch_box[1] <= cy <= patch_box[3])


def bbox_intersects(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> bool:
    """Checks whether two bounding boxes (x1, y1, x2, y2) intersect with non-zero area."""
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])


def extract_micro_fovea_crop(
    patch_crop: Image.Image,
    local_box: List[int],
    expansion_ratio: float = 1.75,
    min_dim: int = 256
) -> Tuple[Image.Image, List[int]]:
    """
    Challenge 1 (Resolution Dilution / Micro vs Macro View):
    Extracts a focused foveated crop centered on the localized object with an expanded margin,
    ensuring minimum dimension min_dim (default 256px) to maximize VLM visual token density.
    """
    pw, ph = patch_crop.size
    lx1, ly1, lx2, ly2 = local_box
    bw = max(1, lx2 - lx1)
    bh = max(1, ly2 - ly1)
    cx = (lx1 + lx2) / 2.0
    cy = (ly1 + ly2) / 2.0

    target_w = max(bw * expansion_ratio, min_dim)
    target_h = max(bh * expansion_ratio, min_dim)
    side = max(target_w, target_h)

    fx1 = max(0, int(round(cx - side / 2.0)))
    fy1 = max(0, int(round(cy - side / 2.0)))
    fx2 = min(pw, int(round(cx + side / 2.0)))
    fy2 = min(ph, int(round(cy + side / 2.0)))

    # Adjust boundaries if clamped at edges
    if fx2 - fx1 < side and fx1 == 0:
        fx2 = min(pw, int(round(side)))
    if fy2 - fy1 < side and fy1 == 0:
        fy2 = min(ph, int(round(side)))
    if fx2 - fx1 < side and fx2 == pw:
        fx1 = max(0, int(round(pw - side)))
    if fy2 - fy1 < side and fy2 == ph:
        fy1 = max(0, int(round(ph - side)))

    fovea_box = [fx1, fy1, fx2, fy2]
    fovea_crop = patch_crop.crop((fx1, fy1, fx2, fy2))
    return fovea_crop, fovea_box


def _query_vlm_existence(
    zoom_model,
    image_pil: Image.Image,
    query: str,
    config: MSSearchConfig,
    custom_prompt_template: Optional[str] = None
) -> float:
    """Safely invokes VLM existence confidence query on crop with resolution safety shield."""
    if zoom_model is None or not hasattr(zoom_model, "get_confidence_value"):
        return 0.0

    safe_img = image_pil
    if max(image_pil.width, image_pil.height) > 1024:
        scale = 1024.0 / max(image_pil.width, image_pil.height)
        safe_img = image_pil.resize(
            (int(round(image_pil.width * scale)), int(round(image_pil.height * scale))),
            Image.Resampling.LANCZOS
        )

    original_prompt = None
    template_to_use = custom_prompt_template or getattr(config, "existence_prompt_template", None)
    if hasattr(zoom_model, "prompts") and isinstance(zoom_model.prompts, dict):
        search_prompts = zoom_model.prompts.get("search", {})
        if "existence_prompt" in search_prompts and template_to_use:
            original_prompt = search_prompts["existence_prompt"]
            search_prompts["existence_prompt"] = template_to_use

    try:
        conf = zoom_model.get_confidence_value(
            node=None,
            image_pil=safe_img,
            confidence_type="existence",
            input_ele=query
        )
        if hasattr(conf, "item"):
            conf = conf.item()
        return float(conf)
    except Exception as e:
        if config.verbose:
            print(f"[Stage 4][VLM Existence Notice] {e}")
        return 0.0
    finally:
        if original_prompt is not None:
            zoom_model.prompts["search"]["existence_prompt"] = original_prompt

    return 0.0


def _query_vlm_latent(
    zoom_model,
    image_pil: Image.Image,
    query: str,
    config: MSSearchConfig,
    custom_prompt_template: Optional[str] = None
) -> float:
    """Safely invokes VLM latent confidence query on centroid crop with resolution safety shield."""
    if zoom_model is None or not hasattr(zoom_model, "get_confidence_value"):
        return 0.0

    safe_img = image_pil
    if max(image_pil.width, image_pil.height) > 1024:
        scale = 1024.0 / max(image_pil.width, image_pil.height)
        safe_img = image_pil.resize(
            (int(round(image_pil.width * scale)), int(round(image_pil.height * scale))),
            Image.Resampling.LANCZOS
        )

    original_prompt = None
    template_to_use = custom_prompt_template or getattr(config, "latent_prompt_template", None)
    if hasattr(zoom_model, "prompts") and isinstance(zoom_model.prompts, dict):
        search_prompts = zoom_model.prompts.get("search", {})
        if "latent_prompt" in search_prompts and template_to_use:
            original_prompt = search_prompts["latent_prompt"]
            search_prompts["latent_prompt"] = template_to_use

    try:
        conf = zoom_model.get_confidence_value(
            node=None,
            image_pil=safe_img,
            confidence_type="latent",
            input_ele=query
        )
        if hasattr(conf, "item"):
            conf = conf.item()
        return float(conf)
    except Exception as e:
        if config.verbose:
            print(f"[Stage 4][VLM Latent Notice] {e}")
        return 0.0
    finally:
        if original_prompt is not None:
            zoom_model.prompts["search"]["latent_prompt"] = original_prompt

    return 0.0
