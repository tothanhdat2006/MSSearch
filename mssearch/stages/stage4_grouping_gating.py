"""
Stage 4: SAM3 Feature Grouping & Open-Vocabulary Active Object Gating.
Clusters bounded patches via feature cosine similarity (cos >= 0.80),
reuses Stage 3 pooled embeddings (zero re-extraction), runs open-vocabulary SAM3 detection,
and prunes background groups in a single step into a flat Remaining Patches pool.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from PIL import Image
try:
    import torch
except ImportError:
    torch = None

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError:
    AgglomerativeClustering = None

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


def _query_vlm_existence(zoom_model, image_pil: Image.Image, query: str, config: MSSearchConfig) -> float:
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

    # Set custom existence prompt template if provided
    original_prompt = None
    if hasattr(zoom_model, "prompts") and isinstance(zoom_model.prompts, dict):
        search_prompts = zoom_model.prompts.get("search", {})
        if "existence_prompt" in search_prompts and hasattr(config, "existence_prompt_template"):
            original_prompt = search_prompts["existence_prompt"]
            search_prompts["existence_prompt"] = config.existence_prompt_template

    try:
        for kwarg in [{"node": None}, {"node_list": None}, {}]:
            try:
                conf = zoom_model.get_confidence_value(
                    image_pil=safe_img,
                    confidence_type="existence",
                    input_ele=query,
                    **kwarg
                )
                if hasattr(conf, "item"):
                    conf = conf.item()
                return float(conf)
            except TypeError:
                continue
    except Exception as e:
        if config.verbose:
            print(f"[Stage 4][VLM Existence Notice] {e}")
        return 0.0
    finally:
        if original_prompt is not None:
            zoom_model.prompts["search"]["existence_prompt"] = original_prompt

    return 0.0


def _query_vlm_latent(zoom_model, image_pil: Image.Image, query: str, config: MSSearchConfig) -> float:
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

    # Set custom latent prompt template if provided
    original_prompt = None
    if hasattr(zoom_model, "prompts") and isinstance(zoom_model.prompts, dict):
        search_prompts = zoom_model.prompts.get("search", {})
        if "latent_prompt" in search_prompts and hasattr(config, "latent_prompt_template"):
            original_prompt = search_prompts["latent_prompt"]
            search_prompts["latent_prompt"] = config.latent_prompt_template

    try:
        for kwarg in [{"node": None}, {"node_list": None}, {}]:
            try:
                conf = zoom_model.get_confidence_value(
                    image_pil=safe_img,
                    confidence_type="latent",
                    input_ele=query,
                    **kwarg
                )
                if hasattr(conf, "item"):
                    conf = conf.item()
                return float(conf)
            except TypeError:
                continue
    except Exception as e:
        if config.verbose:
            print(f"[Stage 4][VLM Latent Notice] {e}")
        return 0.0
    finally:
        if original_prompt is not None:
            zoom_model.prompts["search"]["latent_prompt"] = original_prompt

    return 0.0


def run_feature_grouping_and_gating(
    bounded_patches: List[BoundedPatch],
    candidate_images: List[Image.Image],
    active_objects: List[str],
    sam_model,
    config: MSSearchConfig,
    detected_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    zoom_model=None,
    return_debug: bool = False
) -> Union[List[BoundedPatch], Tuple[List[BoundedPatch], List[Dict[str, Any]]]]:
    """
    Executes Feature Grouping and Active Object Gating.
    1. Runs SAM3 open-vocabulary detection across ALL candidate bounded patches.
    2. Clusters patches by cosine similarity (cos >= 0.80).
    3. For each group:
       - If SAM3 detected an object: verify with VLM existence confidence on best detection.
       - If SAM3 detected no object: query VLM latent confidence on the representative centroid patch.
    Prunes non-matching clusters into a flat Remaining Patches pool.
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

    # 2. Run SAM3 open-vocabulary detection across ALL bounded patches
    patch_detections: Dict[int, List[Dict[str, Any]]] = {}
    if sam_model is not None and active_objects:
        if config.verbose:
            print(f"[Stage 4][SAM3 All-Patch Detection] Scanning all {len(bounded_patches)} bounded patches for active objects {active_objects}...")
        for p_idx, patch in enumerate(bounded_patches):
            p_box_ints = tuple(int(round(float(c))) for c in patch.box)
            p_img = candidate_images[patch.img_idx].crop(p_box_ints)
            pw, ph = p_img.size
            if pw < 8 or ph < 8:
                patch_detections[p_idx] = []
                continue

            dets = []
            try:
                if torch is not None and hasattr(torch, "inference_mode"):
                    with torch.inference_mode():
                        _, processed_results, text_id = sam_model.batch_inference(
                            p_img, active_objects
                        )
                else:
                    _, processed_results, text_id = sam_model.batch_inference(
                        p_img, active_objects
                    )
                for tid in text_id:
                    res = processed_results.get(tid, {}) if isinstance(processed_results, dict) else {}
                    boxes = res.get('boxes', [])
                    scores = res.get('scores', [])
                    for b, s in zip(boxes, scores):
                        s_float = float(s) if not isinstance(s, (list, tuple)) else float(s[0])
                        b_clean = [int(round(float(c))) for c in (b.tolist() if hasattr(b, 'tolist') else b)]
                        dets.append({
                            "label": str(tid),
                            "score": round(s_float, 3),
                            "local_box": b_clean,
                            "global_box": [b_clean[0] + p_box_ints[0], b_clean[1] + p_box_ints[1],
                                           b_clean[2] + p_box_ints[0], b_clean[3] + p_box_ints[1]],
                            "patch_idx": p_idx
                        })
            except Exception as e:
                if config.verbose:
                    print(f"[Stage 4][SAM3 Detection Exception] Patch {p_idx} ({e})")
            patch_detections[p_idx] = dets

    remaining_patches: List[BoundedPatch] = []
    group_records: List[Dict[str, Any]] = []

    # 3. Dual Gating per group
    for g_idx, member_indices in enumerate(groups):
        # Select representative patch (patch closest to group centroid in feature space)
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

        has_active_match = False
        match_reason = "none_pruned"
        vlm_exist_score = None
        vlm_latent_score = None

        # Aggregate SAM3 detections across all member patches of this group
        group_detected_objects: List[Dict[str, Any]] = []
        for mem_i in member_indices:
            for det in patch_detections.get(mem_i, []):
                if det["score"] >= config.sam3_group_det_threshold:
                    group_detected_objects.append(det)

        max_det_score = max([d["score"] for d in group_detected_objects], default=0.0)

        # Safeguard A: Zero-cost Stage 2 detection anchor check
        if detected_boxes:
            for d_idx, d_box in enumerate(detected_boxes):
                d_box_clean = tuple(int(round(float(c))) for c in d_box)
                for mem_i in member_indices:
                    if bounded_patches[mem_i].img_idx == rep_patch.img_idx:
                        mem_box = bounded_patches[mem_i].box
                        center_in = target_center_in_patch(mem_box, d_box_clean)
                        cov_ratio = compute_target_coverage_ratio(mem_box, d_box_clean)

                        if center_in or cov_ratio >= 0.35:
                            has_active_match = True
                            match_reason = "stage2_detection_anchor"
                            target_label = active_objects[d_idx] if d_idx < len(active_objects) else f"Target_{d_idx}"
                            rel_bx1 = max(0, d_box_clean[0] - mem_box[0])
                            rel_by1 = max(0, d_box_clean[1] - mem_box[1])
                            rel_bx2 = min(mem_box[2] - mem_box[0], d_box_clean[2] - mem_box[0])
                            rel_by2 = min(mem_box[3] - mem_box[1], d_box_clean[3] - mem_box[1])

                            group_detected_objects.append({
                                "label": target_label,
                                "score": 1.0,
                                "local_box": [rel_bx1, rel_by1, rel_bx2, rel_by2],
                                "global_box": list(d_box_clean),
                                "coverage_ratio": round(cov_ratio, 2)
                            })
                            if config.verbose:
                                print(f"[Stage 4][Detection Anchor] Group {g_idx} encompasses Stage 2 target '{target_label}' -> automatically [KEPT].")
                            break
                if has_active_match:
                    break

        # Dual VLM Gating
        vlm_prompt = None
        vlm_tested_patch_idx = None
        vlm_tested_role = None
        vlm_per_object_scores = {}

        if not has_active_match:
            if group_detected_objects:
                # -------------------------------------------------------------
                # Branch (2): Object Found by SAM3 in group
                # Verify detection via VLM existence confidence
                # -------------------------------------------------------------
                best_det = max(group_detected_objects, key=lambda d: d["score"])
                best_patch = bounded_patches[best_det["patch_idx"]]
                best_box_ints = tuple(int(round(float(c))) for c in best_patch.box)
                best_patch_crop = candidate_images[best_patch.img_idx].crop(best_box_ints)

                vlm_tested_patch_idx = best_det["patch_idx"]
                vlm_tested_role = "Detection Crop"
                vlm_prompt = config.existence_prompt_template.format(best_det["label"])

                if config.enable_existence_gating and zoom_model is not None:
                    vlm_exist_score = _query_vlm_existence(
                        zoom_model, best_patch_crop, best_det["label"], config
                    )
                    vlm_per_object_scores[best_det["label"]] = vlm_exist_score
                    if vlm_exist_score >= config.existence_confidence_threshold:
                        has_active_match = True
                        match_reason = f"sam3_det_vlm_confirmed (conf={vlm_exist_score:.2f} >= {config.existence_confidence_threshold:.2f})"
                        if config.verbose:
                            print(f"[Stage 4][Existence Confirmed] Group {g_idx} SAM3 detection '{best_det['label']}' verified by VLM existence (conf={vlm_exist_score:.2f}) -> [KEPT].")
                    else:
                        if config.verbose:
                            print(f"[Stage 4][Existence Rejected] Group {g_idx} SAM3 detection '{best_det['label']}' rejected by VLM existence (conf={vlm_exist_score:.2f} < {config.existence_confidence_threshold:.2f}).")
                else:
                    # Gating disabled or zoom_model absent -> retain on SAM3 detection score
                    has_active_match = True
                    match_reason = f"sam3_detection (score={best_det['score']:.2f})"

            if not has_active_match:
                # -------------------------------------------------------------
                # Branch (1): No Object Found (or existence check rejected)
                # Check VLM latent confidence on the representative centroid patch
                # -------------------------------------------------------------
                if config.enable_latent_gating and zoom_model is not None and active_objects:
                    vlm_tested_patch_idx = rep_idx
                    vlm_tested_role = "Centroid (p_rep)"
                    latent_scores_dict = {}
                    for obj in active_objects:
                        s = _query_vlm_latent(zoom_model, rep_img, obj, config)
                        latent_scores_dict[obj] = s

                    vlm_per_object_scores = latent_scores_dict
                    best_obj = max(latent_scores_dict, key=latent_scores_dict.get) if latent_scores_dict else (active_objects[0] if active_objects else "")
                    vlm_latent_score = latent_scores_dict[best_obj] if latent_scores_dict else -1.0
                    vlm_prompt = config.latent_prompt_template.format(best_obj)

                    if vlm_latent_score >= config.latent_confidence_threshold:
                        has_active_match = True
                        match_reason = f"centroid_latent_confidence (conf={vlm_latent_score:.2f} >= {config.latent_confidence_threshold:.2f})"
                        if config.verbose:
                            print(f"[Stage 4][Latent Kept] Group {g_idx} centroid has latent zoom confidence ({vlm_latent_score:.2f} >= {config.latent_confidence_threshold:.2f}) -> [KEPT].")
                    else:
                        if config.verbose:
                            print(f"[Stage 4][Latent Pruned] Group {g_idx} centroid latent confidence below threshold ({vlm_latent_score:.2f} < {config.latent_confidence_threshold:.2f}) -> [PRUNED].")
                elif active_objects:
                    # Simulation or no zoom model: record what prompt would be asked
                    vlm_tested_patch_idx = rep_idx
                    vlm_tested_role = "Centroid (p_rep)"
                    vlm_prompt = config.latent_prompt_template.format(active_objects[0])
        else:
            vlm_tested_role = "Stage 2 Anchor"
            vlm_prompt = "N/A (Bypassed VLM Gating via Stage 2 Global Anchor)"

        # Pairwise cosine similarity submatrix for member patches
        if sim_mat.size > 0 and len(member_indices) > 0:
            sub_sim = sim_mat[member_indices][:, member_indices].tolist()
        else:
            sub_sim = [[1.0]]

        rep_local = member_indices.index(rep_idx) if rep_idx in member_indices else 0

        # Record rich group debug telemetry
        group_records.append({
            "group_idx": g_idx,
            "member_indices": member_indices,
            "rep_idx": rep_idx,
            "rep_idx_local": rep_local,
            "rep_box": list(rep_box_ints),
            "img_idx": rep_patch.img_idx,
            "status": "KEPT" if has_active_match else "PRUNED",
            "match_reason": match_reason,
            "num_patches": len(member_indices),
            "pairwise_sim": sub_sim,
            "member_boxes": [list(tuple(int(round(float(c))) for c in bounded_patches[i].box)) for i in member_indices],
            "member_entropies": [bounded_patches[i].entropy for i in member_indices],
            "detected_objects": group_detected_objects,
            "max_det_score": round(max_det_score, 3),
            "vlm_exist_score": round(vlm_exist_score, 3) if vlm_exist_score is not None else None,
            "vlm_latent_score": round(vlm_latent_score, 3) if vlm_latent_score is not None else None,
            "vlm_prompt": vlm_prompt,
            "vlm_tested_patch_idx": vlm_tested_patch_idx,
            "vlm_tested_role": vlm_tested_role,
            "vlm_per_object_scores": {k: round(v, 3) for k, v in vlm_per_object_scores.items()}
        })

        if has_active_match:
            # Active object present or latent context promising -> KEEP all member patches
            for i in member_indices:
                remaining_patches.append(bounded_patches[i])
            if config.verbose:
                print(f"[Stage 4] Group {g_idx} ({len(member_indices)} patches): Match confirmed ({match_reason}) -> [KEPT].")
        else:
            # PRUNE entire group immediately (zero downstream VLM cost!)
            if config.verbose:
                print(f"[Stage 4] Group {g_idx} ({len(member_indices)} patches): No target & latent < threshold -> [PRUNED] (Saved {len(member_indices)} downstream VLM calls).")

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

