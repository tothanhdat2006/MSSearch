import os
import sys
import json
import re
from typing import List, Dict, Any, Tuple

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS]
    return " ".join(tokens)

def compute_token_overlap(pred: str, gt: str) -> Tuple[float, float, float]:
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gt).split()
    if not p_tokens or not g_tokens:
        return 0.0, 0.0, 0.0
    common = set(p_tokens) & set(g_tokens)
    prec = len(common) / len(p_tokens)
    rec = len(common) / len(g_tokens)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1

def score_prediction(pred_str: str, gt_list: List[str]) -> Dict[str, Any]:
    norm_p = normalize_text(pred_str)
    best_rec, best_prec, best_f1, hit, em = 0.0, 0.0, 0.0, False, False

    for gt in gt_list:
        norm_gt = normalize_text(gt)
        if not norm_gt:
            continue
        if norm_gt == norm_p:
            em = True
            hit = True
        elif norm_gt in norm_p or (len(norm_gt) > 4 and norm_p in norm_gt):
            hit = True
        p, r, f = compute_token_overlap(pred_str, gt)
        best_prec = max(best_prec, p)
        best_rec = max(best_rec, r)
        best_f1 = max(best_f1, f)

    return {"hit": hit, "exact_match": em, "precision": best_prec, "recall": best_rec, "f1": best_f1}

def parse_baseline_json_response(raw_resp: str, candidate_ids: List[str]) -> Tuple[List[Any], List[int], List[str], str]:
    cleaned = (raw_resp or "").strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    json_str = match.group(1) if match else None
    if not json_str:
        match = re.search(r"(\{.*?\})", cleaned, re.DOTALL)
        json_str = match.group(1) if match else cleaned

    raw_ids = []
    pred_ans = ""
    try:
        data = json.loads(json_str)
        val = data.get("image_ids")
        if val is None:
            val = data.get("image_id")
        if isinstance(val, list):
            raw_ids = val
        elif val is not None:
            raw_ids = [val]
        pred_ans = str(data.get("answer", "")).strip()
    except Exception:
        ids_m = re.search(r"""["']image_ids?["']\s*:\s*\[([^\]]*)\]""", cleaned)
        if ids_m:
            raw_ids = [x.strip().strip("\"'") for x in ids_m.group(1).split(",") if x.strip()]
        else:
            id_m = re.search(r"""["']image_ids?["']\s*:\s*([0-9]+|["'][^"']+["'])""", cleaned)
            if id_m:
                raw_ids = [id_m.group(1).strip("\"'")]
        ans_m = re.search(r"""["']answer["']\s*:\s*["']([^"']+)["']""", cleaned)
        pred_ans = ans_m.group(1).strip() if ans_m else cleaned

    pred_indices = []
    actual_ids = []
    for r_id in raw_ids:
        idx_int = None
        try:
            idx_int = int(r_id)
        except (ValueError, TypeError):
            pass
        if idx_int is not None and 1 <= idx_int <= len(candidate_ids):
            pred_indices.append(idx_int)
            actual_ids.append(str(candidate_ids[idx_int - 1]))
        elif str(r_id) in [str(c) for c in candidate_ids]:
            idx = [str(c) for c in candidate_ids].index(str(r_id)) + 1
            pred_indices.append(idx)
            actual_ids.append(str(r_id))
        else:
            if r_id is not None:
                actual_ids.append(str(r_id))

    return raw_ids, pred_indices, actual_ids, pred_ans

base_dir = r"D:\Project\VisualSearch\experiments\multisource\MSSearch\results\baseline"
folders = ["native", "fixed_budget", "old_prompt_v1"]
all_results = {}

for fld in folders:
    fld_path = os.path.join(base_dir, fld)
    if not os.path.exists(fld_path):
        continue
    all_results[fld] = {}
    models = sorted([m for m in os.listdir(fld_path) if os.path.isdir(os.path.join(fld_path, m))])
    for model in models:
        m_path = os.path.join(fld_path, model)
        all_results[fld][model] = {}
        for setting in ["crop", "N1", "N5", "N10", "N20"]:
            fp = os.path.join(m_path, f"{setting}.jsonl")
            if not os.path.isfile(fp):
                all_results[fld][model][setting] = None
                continue
            records = []
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            n = len(records)
            if n == 0:
                all_results[fld][model][setting] = None
                continue
            img_prim_cnt = 0
            ans_hit_cnt = 0
            joint_prim_cnt = 0
            for r in records:
                cand_ids = r.get("image_ids", [])
                raw_pred = r.get("prediction", "")
                raw_ids, pred_indices, actual_ids, pred_ans = parse_baseline_json_response(raw_pred, cand_ids)
                gt_idx = r.get("gt_index")
                gt_id = str(r.get("gt_image_id", ""))
                gt_ans = r.get("gt_answer", "")
                gt_list = [gt_ans] if isinstance(gt_ans, str) else list(gt_ans or [])

                is_prim_ok = False
                if actual_ids:
                    is_prim_ok = (str(gt_id) == str(actual_ids[0]))
                elif pred_indices and gt_idx is not None:
                    is_prim_ok = (int(gt_idx) == int(pred_indices[0]))

                scores = score_prediction(pred_ans, gt_list)
                hit = scores["hit"]

                img_prim_cnt += int(is_prim_ok)
                ans_hit_cnt += int(hit)
                joint_prim_cnt += int(is_prim_ok and hit)

            all_results[fld][model][setting] = {
                "n": n,
                "primary_image_acc": round((img_prim_cnt / n) * 100.0, 1),
                "answer_hit_acc": round((ans_hit_cnt / n) * 100.0, 1),
                "joint_primary_acc": round((joint_prim_cnt / n) * 100.0, 1),
            }

for fld in folders:
    print(f"\n==================== {fld.upper()} ====================")
    print("--- Acc_text (%) ---")
    print(f"{'Model':<22} |  crop  |   N1   |   N5   |  N10   |  N20   ")
    print("-" * 62)
    for m, s in all_results[fld].items():
        row = [f"{s[k]['answer_hit_acc']:5.1f}%" if s.get(k) else "  --  " for k in ["crop", "N1", "N5", "N10", "N20"]]
        print(f"{m:<22} | " + " | ".join(row))

    print("\n--- Acc_img (%) ---")
    print(f"{'Model':<22} |  crop  |   N1   |   N5   |  N10   |  N20   ")
    print("-" * 62)
    for m, s in all_results[fld].items():
        row = [f"{s[k]['primary_image_acc']:5.1f}%" if s.get(k) else "  --  " for k in ["crop", "N1", "N5", "N10", "N20"]]
        print(f"{m:<22} | " + " | ".join(row))

    print("\n--- Acc_overall (%) ---")
    print(f"{'Model':<22} |  crop  |   N1   |   N5   |  N10   |  N20   ")
    print("-" * 62)
    for m, s in all_results[fld].items():
        row = [f"{s[k]['joint_primary_acc']:5.1f}%" if s.get(k) else "  --  " for k in ["crop", "N1", "N5", "N10", "N20"]]
        print(f"{m:<22} | " + " | ".join(row))

print("\n==================== OUR MSSEARCH ====================")
our_files = [
    ("Qwen2.5-VL-7B-Instruct", r"D:\Project\VisualSearch\experiments\multisource\MSSearch\results\our\results_our_Qwen2.5-VL-7B-Instruct_N_5_20261104_103020.jsonl"),
    ("llava-onevision-qwen2-7b-ov", r"D:\Project\VisualSearch\experiments\multisource\MSSearch\results\our\results_our_llava-onevision-qwen2-7b-ov_N_5_20261104_105502.jsonl"),
]
for name, fp in our_files:
    if not os.path.exists(fp):
        continue
    records = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    n = len(records)
    img_prim_cnt = 0
    ans_hit_cnt = 0
    joint_prim_cnt = 0
    for r in records:
        gt_id = str(r.get("gt_image_id", ""))
        p_ids = [str(x) for x in r.get("pred_image_ids", [])]
        is_prim = (len(p_ids) > 0 and p_ids[0] == gt_id)
        pred_ans = r.get("prediction", "")
        gt_list = r.get("ground_truth_list", [])
        scores = score_prediction(pred_ans, gt_list)
        hit = scores["hit"]
        img_prim_cnt += int(is_prim)
        ans_hit_cnt += int(hit)
        if is_prim and hit:
            joint_prim_cnt += 1
    print(f"Ours {name} N=5: Acc_text={ans_hit_cnt/n*100:.1f}%, Acc_img={img_prim_cnt/n*100:.1f}%, Acc_overall={joint_prim_cnt/n*100:.1f}%")

out_fp = os.path.join(base_dir, "baseline_evaluation_all.json")
with open(out_fp, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
print(f"Updated {out_fp} successfully!")

