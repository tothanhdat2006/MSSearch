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
