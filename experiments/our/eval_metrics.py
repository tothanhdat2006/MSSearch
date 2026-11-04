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

def evaluate_predictions(
    pred_text: str,
    raw_pred_list: Optional[List[str]],
    gt_list: List[str]
) -> Dict[str, Any]:
    """Computes token-level set Precision, Recall, F1, and Exact Match."""
    gt_set = set(g.lower().strip() for g in gt_list if g and str(g).strip())

    extracted_preds: List[str] = []
    if raw_pred_list and isinstance(raw_pred_list, list):
        for item in raw_pred_list:
            if isinstance(item, str):
                parts = re.split(r"[,;\n/]+", item)
                extracted_preds.extend(p.strip().lower() for p in parts if p.strip())
    elif pred_text and isinstance(pred_text, str):
        clean_str = re.sub(r"^(the|an|a)\s+", "", pred_text.strip(), flags=re.IGNORECASE)
        clean_str = re.sub(r"[\.\!\?]+$", "", clean_str)
        parts = re.split(r"[,;\n/]+|\band\b", clean_str)
        extracted_preds.extend(p.strip().lower() for p in parts if p.strip())

    cleaned_preds: List[str] = []
    for p in extracted_preds:
        p_clean = re.sub(r"\b(color|colour|view|side|front|rear|facing)\b", "", p, flags=re.IGNORECASE).strip()
        p_clean = re.sub(r"[^\w\s-]", "", p_clean).strip()
        if p_clean and p_clean not in NEGATIVE_RESPONSES:
            cleaned_preds.append(p_clean)

    pred_set = set(cleaned_preds)
    if not pred_set:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "exact_match": 1.0 if not gt_set else 0.0,
            "matched_gt": [], "matched_pred": [],
            "pred_items": [], "gt_items": list(gt_set),
        }

    matched_gt = set()
    matched_pred = set()
    for p in pred_set:
        for g in gt_set:
            if p == g or p in g or g in p:
                matched_gt.add(g)
                matched_pred.add(p)

    precision = len(matched_pred) / len(pred_set) if pred_set else 0.0
    recall = len(matched_gt) / len(gt_set) if gt_set else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = 1.0 if (gt_set and pred_set == gt_set) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_match": exact_match,
        "matched_gt": sorted(list(matched_gt)),
        "matched_pred": sorted(list(matched_pred)),
        "pred_items": sorted(list(pred_set)),
        "gt_items": sorted(list(gt_set)),
    }


def compute_stage_ret_rej_tokens(
    telemetry: Dict[str, Any],
    gt_img_idx: Optional[int],
    target_bbox: Optional[Any],
    candidate_images: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """Computes target retention, noise rejection, and visual tokens across pipeline stages."""
    import math

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

    # Stage 3
    s3_boxes = telemetry.get("stage3_boxes", [])
    s3_ret = any(_box_containment(b.get("box", []), b.get("img_idx")) >= 0.5 for b in s3_boxes) if tb else None
    s3_noise = sum(1 for b in s3_boxes if gt_img_idx is not None and b.get("img_idx") != gt_img_idx)

    # Stage 4
    s4_boxes = telemetry.get("stage4_boxes", [])
    s4_ret = any(_box_containment(b.get("box", []), b.get("img_idx")) >= 0.5 for b in s4_boxes) if tb else None
    s4_noise = sum(1 for b in s4_boxes if gt_img_idx is not None and b.get("img_idx") != gt_img_idx)

    # Stage 5
    s5_boxes = telemetry.get("stage5_boxes", [])
    s5_ret = any(_box_containment(b.get("box", []), b.get("img_idx")) >= 0.5 for b in s5_boxes) if tb else None
    s5_noise = sum(1 for b in s5_boxes if gt_img_idx is not None and b.get("img_idx") != gt_img_idx)

    # Stage 6
    s6_crops = telemetry.get("stage6_crops", [])
    s6_ret = any(_box_containment(c.get("expanded_box") or c.get("box", [0, 0, 0, 0]), c.get("img_idx")) >= 0.5 for c in s6_crops) if tb else None
    s6_noise = sum(1 for c in s6_crops if gt_img_idx is not None and c.get("img_idx") != gt_img_idx)

    # Visual tokens (28x28 pixel block = 1 visual token in Qwen2.5-VL)
    s6_crop_sizes = []
    s6_tokens = 0
    for c in s6_crops:
        box = c.get("expanded_box") or c.get("box") or [0, 0, 0, 0]
        w, h = max(1, box[2] - box[0]), max(1, box[3] - box[1])
        s6_crop_sizes.append([w, h])
        s6_tokens += int(math.ceil(w / 28) * math.ceil(h / 28))

    s2_tokens = 0
    if "stage2_fast_path" in telemetry.get("stages_executed", []) and candidate_images:
        for img in candidate_images:
            iw, ih = img.size if hasattr(img, "size") else (448, 448)
            s2_tokens += int(math.ceil(iw / 28) * math.ceil(ih / 28))

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
            "stage2_direct_tokens": s2_tokens,
            "stage6_reader_tokens": s6_tokens,
            "total_vlm_tokens": s2_tokens + s6_tokens,
        },
        "stage6_crop_sizes": s6_crop_sizes,
    }

