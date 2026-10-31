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

    # Set custom existence prompt template if provided
    original_prompt = None
    template_to_use = custom_prompt_template or getattr(config, "existence_prompt_template", None)
    if hasattr(zoom_model, "prompts") and isinstance(zoom_model.prompts, dict):
        search_prompts = zoom_model.prompts.get("search", {})
        if "existence_prompt" in search_prompts and template_to_use:
            original_prompt = search_prompts["existence_prompt"]
            search_prompts["existence_prompt"] = template_to_use

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

    # Set custom latent prompt template if provided
    original_prompt = None
    template_to_use = custom_prompt_template or getattr(config, "latent_prompt_template", None)
    if hasattr(zoom_model, "prompts") and isinstance(zoom_model.prompts, dict):
        search_prompts = zoom_model.prompts.get("search", {})
        if "latent_prompt" in search_prompts and template_to_use:
            original_prompt = search_prompts["latent_prompt"]
            search_prompts["latent_prompt"] = template_to_use

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
    return_debug: bool = False,
    question: Optional[str] = None
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

    if sam_model is None:
        raise RuntimeError(
            "[Stage 4 Error] sam_model (SAM3) is None. Stage 4 Feature Grouping & Gating requires a loaded SAM3 model; fallback execution is strictly disabled."
        )
    if zoom_model is None:
        raise RuntimeError(
            "[Stage 4 Error] zoom_model (Qwen2.5-VL) is None. Stage 4 Feature Grouping & Gating requires a loaded VLM model; fallback execution is strictly disabled."
        )

    # If grouping is ablated by config, keep all bounded patches directly (ranked by confidence/entropy)
    if not config.enable_feature_grouping:
        if config.verbose:
            print(f"[Stage 4] Feature grouping disabled by config; retaining all {len(bounded_patches)} bounded patches directly without semantic pruning.")
        sorted_bp = sorted(
            bounded_patches,
            key=lambda p: ((p.confidence_score or 0.0), (p.siglip_score if p.siglip_score is not None else 0.0), p.entropy),
            reverse=True
        )
        return (sorted_bp, []) if return_debug else sorted_bp

    # 1. Group patches using cached Stage 3 feature vectors
    groups, sim_mat = group_patches_by_cosine_similarity(
        bounded_patches,
        sim_threshold=config.grouping_sim_threshold,
        return_sim_mat=True
    )
    if config.verbose:
        print(f"[Stage 4][Non-Fallback] Reused Stage 3 feature embeddings (zero forward passes) to cluster {len(bounded_patches)} patches into {len(groups)} semantic groups (cos >= {config.grouping_sim_threshold:.2f}).")

    # 2. Run SAM3 open-vocabulary detection across Prioritized / Centroid patches
    patch_detections: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(bounded_patches))}

    # Step 2a: Zero-cost inheritance of Stage 2 canvas detections (0 ms)
    if detected_boxes:
        for p_idx, patch in enumerate(bounded_patches):
            p_box = patch.box
            for d_idx, d_box in enumerate(detected_boxes):
                d_box_clean = tuple(int(round(float(c))) for c in d_box)
                center_in = target_center_in_patch(p_box, d_box_clean)
                cov_ratio = compute_target_coverage_ratio(p_box, d_box_clean)
                if center_in or cov_ratio >= 0.25:
                    target_label = active_objects[d_idx] if d_idx < len(active_objects) else f"Target_{d_idx}"
                    rel_bx1 = max(0, d_box_clean[0] - p_box[0])
                    rel_by1 = max(0, d_box_clean[1] - p_box[1])
                    rel_bx2 = min(p_box[2] - p_box[0], d_box_clean[2] - p_box[0])
                    rel_by2 = min(p_box[3] - p_box[1], d_box_clean[3] - p_box[1])
                    patch_detections[p_idx].append({
                        "label": target_label,
                        "score": 1.0,
                        "local_box": [rel_bx1, rel_by1, rel_bx2, rel_by2],
                        "global_box": list(d_box_clean),
                        "patch_idx": p_idx,
                        "source": "stage2_anchor"
                    })

    # Step 2b: Prioritized Candidate Patch Selection for SAM3 crop inference
    if sam_model is not None and active_objects:
        if getattr(config, "enable_prioritized_sam3_scanning", True):
            # Prioritize: 
            # 1. Group centroids
            # 2. Patches with high SigLIP scores (>= 0.15)
            # 3. Highest entropy patches
            group_rep_set = set()
            for member_indices in groups:
                if len(member_indices) == 1:
                    group_rep_set.add(member_indices[0])
                else:
                    g_feats = np.stack([bounded_patches[i].feature_vector for i in member_indices], axis=0)
                    centroid = g_feats.mean(axis=0, keepdims=True)
                    dists = np.linalg.norm(g_feats - centroid, axis=1)
                    group_rep_set.add(member_indices[int(np.argmin(dists))])

            candidate_pool = sorted(
                list(range(len(bounded_patches))),
                key=lambda i: (
                    1 if i in group_rep_set else 0,
                    bounded_patches[i].siglip_score if bounded_patches[i].siglip_score is not None else 0.0,
                    (bounded_patches[i].box[2] - bounded_patches[i].box[0]) * (bounded_patches[i].box[3] - bounded_patches[i].box[1])
                ),
                reverse=True
            )
            max_evals = getattr(config, "max_sam3_eval_patches", 16)
            patches_to_scan = candidate_pool[:max_evals]
            if config.verbose:
                print(f"[Stage 4][Prioritized SAM3 Scan] Scanning top {len(patches_to_scan)} / {len(bounded_patches)} prioritized candidate patches/centroids (max_evals={max_evals})...")
        # Step 2c: Batched SAM3 crop scanning
        valid_patches = []
        valid_crops = []
        for p_idx in patches_to_scan:
            patch = bounded_patches[p_idx]
            p_box_ints = tuple(int(round(float(c))) for c in patch.box)
            p_img = candidate_images[patch.img_idx].crop(p_box_ints)
            pw, ph = p_img.size
            if pw >= 8 and ph >= 8:
                valid_patches.append((p_idx, patch, p_box_ints))
                valid_crops.append(p_img)

        sam3_bs = getattr(config, "sam3_batch_size", 4)
        if hasattr(sam_model, "batch_inference_multi_images") and sam3_bs > 1:
            batched_outputs = sam_model.batch_inference_multi_images(
                valid_crops, active_objects, batch_size=sam3_bs
            )
        else:
            batched_outputs = []
            for p_img in valid_crops:
                try:
                    if torch is not None and hasattr(torch, "inference_mode"):
                        with torch.inference_mode():
                            batched_outputs.append(sam_model.batch_inference(p_img, active_objects))
                    else:
                        batched_outputs.append(sam_model.batch_inference(p_img, active_objects))
                except Exception as e:
                    batched_outputs.append((None, {}, []))

        for (p_idx, patch, p_box_ints), (_, processed_results, text_id) in zip(valid_patches, batched_outputs):
            dets = []
            for tid in text_id:
                obj_idx = (int(tid) - 1) if str(tid).isdigit() else 0
                label_str = active_objects[obj_idx] if 0 <= obj_idx < len(active_objects) else str(tid)
                res = processed_results.get(tid, {}) if isinstance(processed_results, dict) else {}
                boxes = res.get('boxes', [])
                scores = res.get('scores', [])
                for b, s in zip(boxes, scores):
                    s_float = float(s) if not isinstance(s, (list, tuple)) else float(s[0])
                    b_clean = [int(round(float(c))) for c in (b.tolist() if hasattr(b, 'tolist') else b)]
                    dets.append({
                        "label": label_str,
                        "score": round(s_float, 3),
                        "local_box": b_clean,
                        "global_box": [b_clean[0] + p_box_ints[0], b_clean[1] + p_box_ints[1],
                                       b_clean[2] + p_box_ints[0], b_clean[3] + p_box_ints[1]],
                        "patch_idx": p_idx
                    })
            patch_detections[p_idx].extend(dets)

    # 3. Two-Phase Batched VLM Gating per group
    # Initial per-group state extraction
    group_states: List[Dict[str, Any]] = []
    existence_queue: List[Dict[str, Any]] = []
    latent_queue: List[Dict[str, Any]] = []

    for g_idx, member_indices in enumerate(groups):
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
        vlm_prompt = None
        vlm_tested_patch_idx = None
        vlm_tested_role = None
        fovea_box = None
        best_det_box = None
        is_repart_candidate = False

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
                            vlm_tested_role = "Stage 2 Anchor"
                            vlm_prompt = "N/A (Bypassed VLM Gating via Stage 2 Global Anchor)"
                            break
                if has_active_match:
                    break

        # Safeguard B: High-confidence SAM3 or SigLIP bypass
        max_siglip_score = max([bounded_patches[mem_i].siglip_score or 0.0 for mem_i in member_indices], default=0.0)
        if getattr(config, "bypass_vlm_for_confirmed_groups", True) and not has_active_match:
            if max_det_score >= config.sam3_group_det_threshold:
                has_active_match = True
                best_det = max(group_detected_objects, key=lambda d: d["score"])
                match_reason = f"sam3_high_confidence (score={best_det['score']:.2f} >= {config.sam3_group_det_threshold:.2f})"
                vlm_tested_patch_idx = best_det["patch_idx"]
                vlm_tested_role = "Detection Crop (SAM3 Confirmed)"
            elif max_siglip_score >= 0.45:
                has_active_match = True
                match_reason = f"siglip_high_confidence (score={max_siglip_score:.2f} >= 0.45)"
                vlm_tested_patch_idx = rep_idx
                vlm_tested_role = "Centroid (SigLIP Confirmed)"

        state = {
            "g_idx": g_idx,
            "member_indices": member_indices,
            "rep_idx": rep_idx,
            "rep_patch": rep_patch,
            "rep_box_ints": rep_box_ints,
            "rep_img": rep_img,
            "has_active_match": has_active_match,
            "match_reason": match_reason,
            "vlm_exist_score": vlm_exist_score,
            "vlm_latent_score": vlm_latent_score,
            "vlm_prompt": vlm_prompt,
            "vlm_tested_patch_idx": vlm_tested_patch_idx,
            "vlm_tested_role": vlm_tested_role,
            "fovea_box": fovea_box,
            "best_det_box": best_det_box,
            "group_detected_objects": group_detected_objects,
            "max_det_score": max_det_score,
            "is_repart_candidate": is_repart_candidate
        }
        group_states.append(state)

        # Queue unresolved groups for VLM probing
        if not has_active_match:
            if group_detected_objects and config.enable_existence_gating and zoom_model is not None:
                best_det = max(group_detected_objects, key=lambda d: d["score"])
                best_patch = bounded_patches[best_det["patch_idx"]]
                best_box_ints = tuple(int(round(float(c))) for c in best_patch.box)
                best_patch_crop = candidate_images[best_patch.img_idx].crop(best_box_ints)
                target_label = best_det["label"]
                local_box = best_det.get("local_box", [0, 0, best_patch_crop.width, best_patch_crop.height])
                state["best_det_box"] = local_box
                fovea_crop, fb = extract_micro_fovea_crop(best_patch_crop, local_box, expansion_ratio=1.75, min_dim=256)
                state["fovea_box"] = fb
                state["vlm_tested_patch_idx"] = best_det["patch_idx"]
                state["vlm_tested_role"] = "Detection Crop (Micro-Fovea)"
                state["vlm_prompt"] = config.existence_prompt_template.format(target_label)

                existence_queue.append({
                    "g_idx": g_idx,
                    "crop": fovea_crop,
                    "label": target_label,
                    "state": state
                })
            elif not group_detected_objects and config.enable_latent_gating and zoom_model is not None and active_objects:
                latent_queue.append({
                    "g_idx": g_idx,
                    "state": state
                })

    # Phase 1: Vectorized VLM Existence Probing
    vlm_batch_size = getattr(config, "vlm_gating_batch_size", 4)
    if existence_queue and zoom_model is not None:
        crops = [item["crop"] for item in existence_queue]
        labels = [item["label"] for item in existence_queue]
        if hasattr(zoom_model, "batch_get_confidence_values"):
            exist_scores = zoom_model.batch_get_confidence_values(
                image_list=crops,
                confidence_type="existence",
                input_ele=labels,
                batch_size=vlm_batch_size
            )
        else:
            exist_scores = [_query_vlm_existence(zoom_model, c, l, config) for c, l in zip(crops, labels)]

        for item, score in zip(existence_queue, exist_scores):
            st = item["state"]
            st["vlm_exist_score"] = float(score)
            if score >= config.existence_confidence_threshold:
                st["has_active_match"] = True
                st["match_reason"] = f"sam3_det_vlm_confirmed (conf={score:.2f} >= {config.existence_confidence_threshold:.2f})"
            else:
                # Existence rejected: route to Latent Queue if budget allows
                if config.enable_latent_gating and active_objects:
                    latent_queue.append({"g_idx": st["g_idx"], "state": st})

    # Phase 2: Vectorized VLM Latent Probing (Budget Capped)
    max_latent_evals = getattr(config, "max_latent_evals", 8)
    prompt_mode = getattr(config, "latent_prompt_mode", "question_aware")
    primary_obj = active_objects[0] if active_objects else "object"

    if prompt_mode == "question_aware" and question:
        latent_query_str = question
        latent_tmpl = getattr(
            config,
            "latent_prompt_template_qa",
            "According to visual common sense and the content of the image, is it possible to find the information needed to answer '{}' by further zooming the sub-patch? Answer Yes or No."
        )
    else:
        latent_query_str = primary_obj
        latent_tmpl = getattr(
            config,
            "latent_prompt_template",
            "According to your common sense knowledge and the content of the image, is it possible to find a {} by further zooming the sub-patch? Answer Yes or No."
        )

    evaluated_latent_items = latent_queue[:max_latent_evals]
    capped_latent_items = latent_queue[max_latent_evals:]
    for item in capped_latent_items:
        st = item["state"]
        st["match_reason"] = f"budget_capped (max {max_latent_evals} reached)"

    if evaluated_latent_items and zoom_model is not None:
        latent_crops = []
        for item in evaluated_latent_items:
            st = item["state"]
            if st["group_detected_objects"]:
                best_d = max(st["group_detected_objects"], key=lambda d: d["score"])
                eval_p = bounded_patches[best_d["patch_idx"]]
                st["vlm_tested_patch_idx"] = best_d["patch_idx"]
                st["vlm_tested_role"] = "Detection Crop (p_det)"
            else:
                eval_p = st["rep_patch"]
                st["vlm_tested_patch_idx"] = st["rep_idx"]
                st["vlm_tested_role"] = "Centroid (p_rep)"
            eb_ints = tuple(int(round(float(c))) for c in eval_p.box)
            latent_crops.append(candidate_images[eval_p.img_idx].crop(eb_ints))
            st["vlm_prompt"] = latent_tmpl.format(latent_query_str)

        if hasattr(zoom_model, "batch_get_confidence_values"):
            latent_scores = zoom_model.batch_get_confidence_values(
                image_list=latent_crops,
                confidence_type="latent",
                input_ele=latent_query_str,
                batch_size=vlm_batch_size,
                custom_prompt_templates=latent_tmpl
            )
        else:
            latent_scores = [_query_vlm_latent(zoom_model, c, latent_query_str, config, custom_prompt_template=latent_tmpl) for c in latent_crops]

        repart_thresh = getattr(config, "latent_repartition_threshold", 0.70)
        for item, score in zip(evaluated_latent_items, latent_scores):
            st = item["state"]
            st["vlm_latent_score"] = float(score)
            if score >= config.latent_confidence_threshold:
                st["has_active_match"] = True
                st["match_reason"] = f"latent_confidence (conf={score:.2f} >= {config.latent_confidence_threshold:.2f})"
                if score >= repart_thresh:
                    st["is_repart_candidate"] = True
            else:
                st["match_reason"] = f"latent_pruned (conf={score:.2f} < {config.latent_confidence_threshold:.2f})"

    # Finalize remaining patches and telemetry group records
    remaining_patches: List[BoundedPatch] = []
    group_records: List[Dict[str, Any]] = []

    for state in group_states:
        g_idx = state["g_idx"]
        member_indices = state["member_indices"]
        rep_idx = state["rep_idx"]
        rep_patch = state["rep_patch"]
        rep_box_ints = state["rep_box_ints"]
        has_active_match = state["has_active_match"]
        match_reason = state["match_reason"]
        vlm_exist_score = state["vlm_exist_score"]
        vlm_latent_score = state["vlm_latent_score"]
        vlm_prompt = state["vlm_prompt"]
        vlm_tested_patch_idx = state["vlm_tested_patch_idx"]
        vlm_tested_role = state["vlm_tested_role"]
        fovea_box = state["fovea_box"]
        best_det_box = state["best_det_box"]
        group_detected_objects = state["group_detected_objects"]
        max_det_score = state["max_det_score"]
        is_repart_candidate = state["is_repart_candidate"]

        if sim_mat.size > 0 and len(member_indices) > 0:
            sub_sim = sim_mat[member_indices][:, member_indices].tolist()
        else:
            sub_sim = [[1.0]]

        rep_local = member_indices.index(rep_idx) if rep_idx in member_indices else 0

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
            "vlm_per_object_scores": {primary_obj: round(vlm_latent_score, 3)} if vlm_latent_score is not None else {},
            "fovea_box": fovea_box,
            "best_det_box": best_det_box,
            "is_latent_repartition_candidate": is_repart_candidate
        })

        if has_active_match:
            is_detection = ("sam3_det" in match_reason or "detection" in match_reason or "stage2" in match_reason)
            group_score = 0.0
            if is_detection:
                group_score = 2.0 + (vlm_exist_score if vlm_exist_score is not None else 0.5) + max_det_score
            elif "latent" in match_reason:
                group_score = 1.0 + (vlm_latent_score if vlm_latent_score is not None else 0.0)
            else:
                group_score = 0.5

            if getattr(config, "enable_group_retention_caps", True):
                if is_detection:
                    k_det = getattr(config, "max_patches_per_detected_group", 4)
                    detected_members = []
                    non_detected_members = []
                    for i in member_indices:
                        p = bounded_patches[i]
                        cur_dets = [d for d in group_detected_objects if d.get("patch_idx") == i]
                        if not cur_dets:
                            cur_dets = patch_detections.get(i, [])
                        if cur_dets:
                            max_area = 0
                            max_det_score = 0.0
                            for d in cur_dets:
                                dbox = d.get("local_box") or d.get("box") or d.get("global_box")
                                if dbox and len(dbox) == 4:
                                    dw = max(0, dbox[2] - dbox[0])
                                    dh = max(0, dbox[3] - dbox[1])
                                    darea = dw * dh
                                else:
                                    darea = 0
                                if darea > max_area:
                                    max_area = darea
                                sc = float(d.get("score", 0.0))
                                if sc > max_det_score:
                                    max_det_score = sc
                            siglip = float(p.siglip_score) if p.siglip_score is not None else 0.0
                            detected_members.append((i, max_area, max_det_score, siglip))
                        else:
                            p_area = (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
                            siglip = float(p.siglip_score) if p.siglip_score is not None else 0.0
                            non_detected_members.append((i, siglip, p_area))

                    # 1. Primary ranker: max detected bbox area desc, then det score desc, then siglip desc
                    detected_sorted = [
                        item[0] for item in sorted(
                            detected_members,
                            key=lambda x: (x[1], x[2], x[3]),
                            reverse=True
                        )
                    ]
                    # 2. Non-detection context fill: siglip_score desc, then larger patch area desc (no entropy!)
                    non_detected_sorted = [
                        item[0] for item in sorted(
                            non_detected_members,
                            key=lambda x: (x[1], x[2]),
                            reverse=True
                        )
                    ]
                    chosen_indices = (detected_sorted + non_detected_sorted)[:k_det]
                else:
                    k_lat = getattr(config, "max_patches_per_latent_group", 1)
                    if rep_idx in member_indices:
                        other_indices = [i for i in member_indices if i != rep_idx]
                        chosen_indices = ([rep_idx] + other_indices)[:k_lat]
                    else:
                        chosen_indices = member_indices[:k_lat]
            else:
                chosen_indices = member_indices

            target_ent = getattr(active_objects, "target_entity", None)
            anchor_ent = getattr(active_objects, "anchor_entity", None)
            for i in chosen_indices:
                p = bounded_patches[i]
                p.confidence_score = group_score

                # Assign matched_label and semantic_role from detections
                cur_dets = [d for d in group_detected_objects if d.get("patch_idx") == i]
                if not cur_dets:
                    cur_dets = patch_detections.get(i, [])
                if cur_dets:
                    best_d = max(cur_dets, key=lambda d: d.get("score", 0.0))
                    p.matched_label = str(best_d.get("label", ""))

                if p.matched_label:
                    lbl_lower = p.matched_label.lower()
                    if target_ent and (target_ent.lower() in lbl_lower or lbl_lower in target_ent.lower()):
                        p.semantic_role = "target"
                    elif anchor_ent and (anchor_ent.lower() in lbl_lower or lbl_lower in anchor_ent.lower()):
                        p.semantic_role = "anchor"

                remaining_patches.append(p)

            if config.verbose:
                cap_note = f" (retained {len(chosen_indices)}/{len(member_indices)} patches)" if len(chosen_indices) < len(member_indices) else ""
                print(f"[Stage 4] Group {g_idx} ({len(member_indices)} patches): Match confirmed ({match_reason}) -> [KEPT]{cap_note}.")
        else:
            # PRUNE entire group immediately (zero downstream VLM cost!)
            if config.verbose:
                print(f"[Stage 4] Group {g_idx} ({len(member_indices)} patches): No target & latent < threshold -> [PRUNED] (Saved {len(member_indices)} downstream VLM calls).")

    # Rank retained patches by detection confidence, SigLIP semantic similarity, and inverse area (deeper zoom saccades)
    def _patch_inv_area(p: BoundedPatch) -> float:
        pw = max(1, p.box[2] - p.box[0])
        ph = max(1, p.box[3] - p.box[1])
        return 1.0 / float(pw * ph)

    if remaining_patches:
        metric = getattr(config, "retention_ranking_metric", "confidence_hybrid")
        if metric == "detection_first":
            sorted_patches = sorted(remaining_patches, key=lambda p: (p.confidence_score or 0.0), reverse=True)
        else:  # "confidence_hybrid"
            sorted_patches = sorted(
                remaining_patches,
                key=lambda p: (
                    (p.confidence_score or 0.0),
                    (p.siglip_score if p.siglip_score is not None else 0.0),
                    _patch_inv_area(p)
                ),
                reverse=True
            )

        if getattr(config, "enable_retention_hard_cap", True) and len(sorted_patches) > config.max_retained_patches:
            k_cap = config.max_retained_patches
            # Dynamic elasticity for tied cutoff candidates (count(identical) - 1)
            cutoff_p = sorted_patches[k_cap - 1]
            cutoff_profile = (
                round(float(cutoff_p.confidence_score or 0.0), 4),
                round(float(cutoff_p.siglip_score or 0.0), 4)
            )
            while k_cap < len(sorted_patches):
                next_p = sorted_patches[k_cap]
                next_profile = (
                    round(float(next_p.confidence_score or 0.0), 4),
                    round(float(next_p.siglip_score or 0.0), 4)
                )
                if next_profile == cutoff_profile:
                    k_cap += 1
                else:
                    break

            if config.verbose and k_cap > config.max_retained_patches:
                delta_k = k_cap - config.max_retained_patches
                print(f"[Stage 4][Dynamic Elasticity] Cutoff profile {cutoff_profile} tied: dynamically expanded retention budget by +{delta_k} (retaining {k_cap} patches).")

            pruned_by_cap = len(sorted_patches) - k_cap
            if config.verbose and pruned_by_cap > 0:
                print(f"[Stage 4][Budget Guard] Global retention cap applied: keeping {k_cap} patches (pruned {pruned_by_cap} excess patches).")
            remaining_patches = sorted_patches[:k_cap]
        else:
            remaining_patches = sorted_patches

    # Safety fallback: if all groups got pruned, retain top patches by SigLIP score and larger area (entropy purged)
    if not remaining_patches and bounded_patches:
        def _patch_area(p: BoundedPatch) -> int:
            return max(1, (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))
        sorted_by_relevance = sorted(
            bounded_patches,
            key=lambda p: (
                p.siglip_score if p.siglip_score is not None else 0.0,
                _patch_area(p)
            ),
            reverse=True
        )
        fallback_k = max(1, min(len(bounded_patches), getattr(config, "max_retained_patches", 16)))
        remaining_patches = sorted_by_relevance[:fallback_k]
        if config.verbose:
            print(f"[Stage 4][Safety Fallback Trigger] All {len(groups)} groups were pruned! Fallback activated: keeping top {fallback_k} most relevant patches (SigLIP/area, entropy purged).")
    elif config.verbose:
        pruned_count = len(bounded_patches) - len(remaining_patches)
        prune_pct = (pruned_count / max(1, len(bounded_patches))) * 100
        print(f"[Stage 4][Non-Fallback Summary] Gating complete: {len(remaining_patches)}/{len(bounded_patches)} patches kept ({pruned_count} pruned, {prune_pct:.1f}% reduction).")

    if return_debug:
        return remaining_patches, group_records
    return remaining_patches

