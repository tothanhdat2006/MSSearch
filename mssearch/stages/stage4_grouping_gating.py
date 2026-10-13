"""
Stage 4: SAM3 Feature Grouping & Open-Vocabulary Active Object Gating.
Clusters bounded patches via feature cosine similarity (cos >= 0.80),
reuses Stage 3 pooled embeddings (zero re-extraction), runs open-vocabulary SAM3 detection,
and prunes background groups in a single step into a flat Remaining Patches pool.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from PIL import Image
import torch
from sklearn.cluster import AgglomerativeClustering

from ..config import MSSearchConfig
from .stage3_partitioning import BoundedPatch


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
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dist_threshold,
            metric="precomputed",
            linkage="average"
        )
        labels = clustering.fit_predict(dist_mat)
    except Exception:
        # Fallback each patch as its own group
        groups_list = [[i] for i in range(n_patches)]
        return (groups_list, sim_mat) if return_sim_mat else groups_list

    groups: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(labels):
        groups.setdefault(lbl, []).append(idx)

    groups_list = list(groups.values())
    return (groups_list, sim_mat) if return_sim_mat else groups_list


def bbox_intersects(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> bool:
    """Checks whether two bounding boxes (x1, y1, x2, y2) intersect with non-zero area."""
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])


def run_feature_grouping_and_gating(
    bounded_patches: List[BoundedPatch],
    candidate_images: List[Image.Image],
    active_objects: List[str],
    sam_model,
    config: MSSearchConfig,
    detected_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    return_debug: bool = False
) -> Union[List[BoundedPatch], Tuple[List[BoundedPatch], List[Dict[str, Any]]]]:
    """
    Executes Feature Grouping and Active Object Gating.
    Prunes background clusters in 1 step; keeps surviving patches in a flat list.
    Safeguards against false pruning via Stage 2 detection overlap and dual-patch testing.
    """
    if not bounded_patches:
        return ([], []) if return_debug else []

    # If grouping is ablated, keep all bounded patches directly
    if not config.enable_feature_grouping or sam_model is None:
        if config.verbose:
            print(f"[Stage 4][Fallback] Feature grouping disabled by config or SAM3 absent; retaining all {len(bounded_patches)} bounded patches directly without semantic pruning.")
        return (bounded_patches, []) if return_debug else bounded_patches

    # 1. Group patches using cached Stage 3 feature vectors
    groups, sim_mat = group_patches_by_cosine_similarity(
        bounded_patches,
        sim_threshold=config.grouping_sim_threshold,
        return_sim_mat=True
    )
    if config.verbose:
        print(f"[Stage 4][Non-Fallback] Reused Stage 3 feature embeddings (zero forward passes) to cluster {len(bounded_patches)} patches into {len(groups)} semantic groups (cos >= {config.grouping_sim_threshold:.2f}).")

    remaining_patches: List[BoundedPatch] = []
    group_records: List[Dict[str, Any]] = []

    # 2. Active Object Gating per group
    for g_idx, member_indices in enumerate(groups):
        # Select representative patch (patch closest to group centroid)
        if len(member_indices) == 1:
            rep_idx = member_indices[0]
        else:
            group_feats = np.stack([bounded_patches[i].feature_vector for i in member_indices], axis=0)
            centroid = group_feats.mean(axis=0, keepdims=True)
            dists = np.linalg.norm(group_feats - centroid, axis=1)
            rep_idx = member_indices[int(np.argmin(dists))]

        rep_patch = bounded_patches[rep_idx]
        rep_box_ints = tuple(int(round(float(c))) for c in rep_patch.box)
        rep_img = candidate_images[rep_patch.img_idx].crop(rep_box_ints)
        rw, rh = rep_img.size
        if rw < 8 or rh < 8:
            continue

        has_active_match = False
        match_reason = "none_pruned"
        best_entropy_idx = rep_idx

        # Safeguard A: Zero-cost Stage 2 detection overlap check
        if detected_boxes:
            for d_box in detected_boxes:
                d_box_clean = tuple(int(round(float(c))) for c in d_box)
                for mem_i in member_indices:
                    if bounded_patches[mem_i].img_idx == rep_patch.img_idx:
                        if bbox_intersects(bounded_patches[mem_i].box, d_box_clean):
                            has_active_match = True
                            match_reason = "stage2_detection_anchor"
                            if config.verbose:
                                print(f"[Stage 4][Detection Anchor] Group {g_idx} intersects Stage 2 detected target {d_box_clean} -> automatically [KEPT].")
                            break
                if has_active_match:
                    break

        # Safeguard B: SAM3 detection on representative patch
        if not has_active_match:
            try:
                with torch.inference_mode():
                    _, processed_results, text_id = sam_model.batch_inference(
                        rep_img, active_objects
                    )
                for tid in text_id:
                    res = processed_results.get(tid, {}) if isinstance(processed_results, dict) else {}
                    scores = res.get('scores', [])
                    if any(s >= config.sam3_group_det_threshold for s in scores):
                        has_active_match = True
                        match_reason = "representative_sam3_det"
                        break
            except Exception as e:
                if config.verbose:
                    print(f"[Stage 4][Exception Fallback] SAM3 group detection failed on Group {g_idx} ({e}); conservatively keeping group.")
                has_active_match = True
                match_reason = "exception_fallback"

        # Safeguard C: Dual-patch check on highest-entropy patch for multi-patch groups
        if not has_active_match and len(member_indices) > 1:
            best_entropy_idx = max(member_indices, key=lambda idx: bounded_patches[idx].entropy)
            if best_entropy_idx != rep_idx:
                he_patch = bounded_patches[best_entropy_idx]
                he_box_ints = tuple(int(round(float(c))) for c in he_patch.box)
                he_img = candidate_images[he_patch.img_idx].crop(he_box_ints)
                hew, heh = he_img.size
                if hew >= 8 and heh >= 8:
                    try:
                        with torch.inference_mode():
                            _, processed_results, text_id = sam_model.batch_inference(
                                he_img, active_objects
                            )
                        for tid in text_id:
                            res = processed_results.get(tid, {}) if isinstance(processed_results, dict) else {}
                            scores = res.get('scores', [])
                            if any(s >= config.sam3_group_det_threshold for s in scores):
                                has_active_match = True
                                match_reason = "secondary_entropy_anchor"
                                if config.verbose:
                                    print(f"[Stage 4][Entropy Anchor] Group {g_idx} matched active object on secondary high-entropy patch -> [KEPT].")
                                break
                    except Exception:
                        pass

        # Pairwise cosine similarity submatrix for all member patches in this group
        if sim_mat.size > 0 and len(member_indices) > 0:
            sub_sim = sim_mat[member_indices][:, member_indices].tolist()
        else:
            sub_sim = [[1.0]]

        rep_local = member_indices.index(rep_idx) if rep_idx in member_indices else 0
        he_local = member_indices.index(best_entropy_idx) if best_entropy_idx in member_indices else 0

        # Record rich group debug telemetry
        group_records.append({
            "group_idx": g_idx,
            "member_indices": member_indices,
            "rep_idx": rep_idx,
            "rep_idx_local": rep_local,
            "best_entropy_idx": best_entropy_idx,
            "best_entropy_idx_local": he_local,
            "rep_box": list(rep_box_ints),
            "img_idx": rep_patch.img_idx,
            "status": "KEPT" if has_active_match else "PRUNED",
            "match_reason": match_reason,
            "num_patches": len(member_indices),
            "pairwise_sim": sub_sim,
            "member_boxes": [list(tuple(int(round(float(c))) for c in bounded_patches[i].box)) for i in member_indices],
            "member_entropies": [float(bounded_patches[i].entropy) for i in member_indices],
        })

        if has_active_match:
            # Active object present -> KEEP all member patches into flat candidate pool
            for i in member_indices:
                remaining_patches.append(bounded_patches[i])
            if config.verbose:
                print(f"[Stage 4] Group {g_idx} ({len(member_indices)} patches): Active object detected -> [KEPT].")
        else:
            # Disjoint from A_q -> PRUNE entire group immediately (zero VLM cost!)
            if config.verbose:
                print(f"[Stage 4] Group {g_idx} ({len(member_indices)} patches): No active objects -> [PRUNED] (Saved {len(member_indices)} downstream VLM calls).")

    # Safety fallback: if all groups got pruned, retain top 20% highest entropy patches
    if not remaining_patches and bounded_patches:
        sorted_by_entropy = sorted(bounded_patches, key=lambda p: p.entropy, reverse=True)
        fallback_k = max(1, len(bounded_patches) // 5)
        remaining_patches = sorted_by_entropy[:fallback_k]
        if config.verbose:
            print(f"[Stage 4][Safety Fallback Trigger] All {len(groups)} groups were pruned! Fallback activated: keeping top {fallback_k} highest-entropy patches.")
    elif config.verbose:
        pruned_count = len(bounded_patches) - len(remaining_patches)
        prune_pct = (pruned_count / max(1, len(bounded_patches))) * 100
        print(f"[Stage 4][Non-Fallback Summary] Gating complete: {len(remaining_patches)}/{len(bounded_patches)} patches kept ({pruned_count} pruned, {prune_pct:.1f}% reduction).")

    if return_debug:
        return remaining_patches, group_records
    return remaining_patches

