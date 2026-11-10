import re
from typing import List, Dict, Any, Optional

NEGATIVE_RESPONSES = {
    "none", "no", "not visible", "not found", "cannot determine", "unanswerable",
    "not clearly visible", "cannot be seen", "no target", "nothing", "n/a", "not present"
}

def is_negative_or_unanswerable(text: str) -> bool:
    """Returns True if the response is a refusal, empty, or negative detection."""
    clean = re.sub(r"[^\w\s]", "", str(text).lower()).strip()
    if not clean or clean in NEGATIVE_RESPONSES:
        return True
    return any(neg in clean for neg in ["not clearly visible", "cannot see", "not visible", "no evidence"])

def compute_stage_ret_rej_tokens(
    telemetry: Dict[str, Any],
    gt_img_idx: Optional[int],
    target_bbox: Optional[Any],
    candidate_images: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Computes target retention, noise rejection, and visual tokens across pipeline stages."""
    import math

    is_direct = telemetry.get("stage5_mode") in ["direct_answer", "fast_path", "direct_baseline"]
    ev_imgs = telemetry.get("evidence_images", [])

    tb = None
    if isinstance(target_bbox, dict):
        if "x" in target_bbox and "width" in target_bbox:
            tb = [target_bbox["x"], target_bbox["y"], target_bbox["x"] + target_bbox["width"], target_bbox["y"] + target_bbox["height"]]
        elif "x1" in target_bbox and "x2" in target_bbox:
            tb = [target_bbox["x1"], target_bbox["y1"], target_bbox["x2"], target_bbox["y2"]]
    elif isinstance(target_bbox, (list, tuple)) and len(target_bbox) >= 4:
        tb = list(target_bbox[:4])

    t_area = max(1.0, float((tb[2] - tb[0]) * (tb[3] - tb[1]))) if tb else 1.0

    def _box_containment(b, img_idx):
        if not tb or gt_img_idx is None or img_idx != gt_img_idx or len(b) < 4:
            return 0.0
        ix1, iy1 = max(tb[0], b[0]), max(tb[1], b[1])
        ix2, iy2 = min(tb[2], b[2]), min(tb[3], b[3])
        inter = max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))
        return inter / t_area

    # Direct answer bypass retention
    direct_retained = (gt_img_idx in ev_imgs) if (tb and gt_img_idx is not None and is_direct) else None

    # Stage 3
    s3_boxes = telemetry.get("stage3_boxes", [])
    s3_ret = direct_retained if is_direct else (any(_box_containment(b.get("box", []), b.get("img_idx")) >= 0.5 for b in s3_boxes) if tb else None)
    s3_noise = sum(1 for b in s3_boxes if gt_img_idx is not None and b.get("img_idx") != gt_img_idx)

    # Stage 4
    s4_boxes = telemetry.get("stage4_boxes", [])
    s4_ret = direct_retained if is_direct else (any(_box_containment(b.get("box", []), b.get("img_idx")) >= 0.5 for b in s4_boxes) if tb else None)
    s4_noise = sum(1 for b in s4_boxes if gt_img_idx is not None and b.get("img_idx") != gt_img_idx)

    # Stage 5
    s5_boxes = telemetry.get("stage5_boxes", [])
    s5_ret = direct_retained if is_direct else (any(_box_containment(b.get("box", []), b.get("img_idx")) >= 0.5 for b in s5_boxes) if tb else None)
    s5_noise = sum(1 for b in s5_boxes if gt_img_idx is not None and b.get("img_idx") != gt_img_idx)

    # Stage 6
    s6_crops = telemetry.get("stage6_crops", [])
    s6_ret = direct_retained if is_direct else (any(_box_containment(c.get("expanded_box") or c.get("box", [0, 0, 0, 0]), c.get("img_idx")) >= 0.5 for c in s6_crops) if tb else None)
    s6_noise = sum(1 for c in s6_crops if gt_img_idx is not None and c.get("img_idx") != gt_img_idx)

    # Visual tokens (28x28 pixel block = 1 visual token in Qwen2.5-VL / Qwen3-VL)
    s6_crop_sizes = []
    s6_tokens = 0
    for c in s6_crops:
        box = c.get("expanded_box") or c.get("box") or [0, 0, 0, 0]
        w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
        s6_crop_sizes.append([w, h])
        s6_tokens += int(math.ceil(w / 28) * math.ceil(h / 28))

    direct_tokens = 0
    stages_exec = telemetry.get("stages_executed", [])
    if any(s in stages_exec for s in ["phase1_direct_answer", "stage2_fast_path", "direct_baseline", "direct_fallback"]) and candidate_images:
        for img in candidate_images:
            iw, ih = img.size if hasattr(img, "size") else (448, 448)
            direct_tokens += int(math.ceil(iw / 28) * math.ceil(ih / 28))

    has_gt_bbox = bool(tb and len(tb) == 4 and (tb[2] > tb[0]) and (tb[3] > tb[1]))

    def _box_iou(b, target):
        if len(b) < 4 or len(target) < 4:
            return 0.0
        ix1, iy1 = max(b[0], target[0]), max(b[1], target[1])
        ix2, iy2 = min(b[2], target[2]), min(b[3], target[3])
        inter = max(0.0, float(ix2 - ix1)) * max(0.0, float(iy2 - iy1))
        a1 = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
        a2 = max(0.0, float(target[2] - target[0])) * max(0.0, float(target[3] - target[1]))
        union = a1 + a2 - inter
        return inter / union if union > 0.0 else 0.0

    if has_gt_bbox and gt_img_idx is not None:
        gt_s4_boxes = [b.get("box", []) for b in s4_boxes if b.get("img_idx") == gt_img_idx]
        max_iou_stage4 = round(max([_box_iou(b, tb) for b in gt_s4_boxes], default=0.0), 4)
    else:
        max_iou_stage4 = None

    if is_direct:
        gt_survived_stage4 = (gt_img_idx in ev_imgs) if gt_img_idx is not None else None
    elif has_gt_bbox and gt_img_idx is not None:
        gt_survived_stage4 = bool((max_iou_stage4 is not None and max_iou_stage4 >= 0.5) or s4_ret is True)
    elif gt_img_idx is not None:
        gt_survived_stage4 = bool(any(b.get("img_idx") == gt_img_idx for b in s4_boxes))
    else:
        gt_survived_stage4 = None

    num_imgs = len(candidate_images) if candidate_images is not None else telemetry.get("num_candidate_images", 0)
    surv_s4_imgs = set(b.get("img_idx") for b in s4_boxes)
    if gt_img_idx is not None and num_imgs > 1:
        distractor_imgs_pruned = sum(1 for i in range(num_imgs) if i != gt_img_idx and i not in surv_s4_imgs)
    else:
        distractor_imgs_pruned = 0

    raw_scores = telemetry.get("stage4_image_scores", {})
    stage4_image_scores = {int(k): float(v) for k, v in raw_scores.items()}
    if stage4_image_scores and gt_img_idx is not None and gt_img_idx in stage4_image_scores:
        sorted_imgs = sorted(stage4_image_scores.keys(), key=lambda idx: stage4_image_scores[idx], reverse=True)
        gt_rank_stage4 = sorted_imgs.index(gt_img_idx) + 1
        stage4_r1 = 1.0 if gt_rank_stage4 == 1 else 0.0
        stage4_mrr = round(1.0 / gt_rank_stage4, 4)
    else:
        gt_rank_stage4 = None
        stage4_r1 = None
        stage4_mrr = None

    return {
        "stage_retention": {
            "stage3": s3_ret,
            "stage4": s4_ret,
            "stage5": s5_ret,
            "stage6": s6_ret,
        },
        "stage_rejection": {
            "stage3_noise_patches": s3_noise,
            "stage4_noise_patches": s4_noise,
            "stage5_noise_crops": s5_noise,
            "stage6_distractor_reads": s6_noise,
        },
        "visual_tokens": {
            "stage2_direct_tokens": direct_tokens,
            "stage6_reader_tokens": s6_tokens,
            "total_vlm_tokens": direct_tokens + s6_tokens,
        },
        "stage6_crop_sizes": s6_crop_sizes,
        "has_gt_bbox": has_gt_bbox,
        "max_iou_stage4": max_iou_stage4,
        "gt_survived_stage4": gt_survived_stage4,
        "distractor_imgs_pruned": distractor_imgs_pruned,
        "stage4_image_scores": stage4_image_scores,
        "gt_rank_stage4": gt_rank_stage4,
        "stage4_r1": stage4_r1,
        "stage4_mrr": stage4_mrr,
        "s6_visual_tokens": s6_tokens,
    }
