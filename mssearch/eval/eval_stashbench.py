import os
import sys
import json
import re
import argparse
from typing import Dict, List, Any, Tuple

DEFAULT_JSONL = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../results/our/results_our_Qwen2.5-VL-7B-Instruct_N_5_20261104_081923.jsonl",
    )
)

def normalize_text(text: str) -> str:
    clean = re.sub(r"[^\w\s]", " ", str(text).lower()).strip()
    return " ".join(clean.split())

def compute_token_overlap(pred_text: str, gt_text: str) -> Tuple[float, float, float]:
    p_tokens = set(normalize_text(pred_text).split())
    g_tokens = set(normalize_text(gt_text).split())
    if not p_tokens or not g_tokens:
        return 0.0, 0.0, 0.0
    common = p_tokens.intersection(g_tokens)
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
        if r >= 0.9:
            hit = True

    return {"hit": hit, "exact_match": em, "precision": best_prec, "recall": best_rec, "f1": best_f1}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate STASHBench MSSearch / Baseline JSONL Results.")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_JSONL, help="Path to evaluation JSONL file.")
    parser.add_argument("--save_summary", "-o", type=str, default=None, help="Path to save summary JSON.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-sample breakdown.")
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"[FATAL] Input file not found: {input_path}")

    records: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    n = len(records)
    if n == 0:
        print(f"[WARN] No valid records loaded from '{input_path}'.")
        return

    img_correct_cnt = 0
    ans_hit_cnt = 0
    joint_cnt = 0
    em_cnt = 0
    total_f1, total_rec, total_prec = 0.0, 0.0, 0.0
    total_lat = 0.0

    retention_counts = {"stage3": [0, 0], "stage4": [0, 0], "stage5": [0, 0], "stage6": [0, 0]}
    rejection_sums = {"stage3_noise_patches": 0, "stage4_noise_patches": 0, "stage5_noise_crops": 0, "stage6_distractor_reads": 0}
    token_sums = {"stage2_direct_tokens": 0, "stage6_reader_tokens": 0, "total_vlm_tokens": 0}
    timing_sums: Dict[str, float] = {}
    routes: Dict[str, int] = {}
    per_sample_details = []

    for r in records:
        s_id = r.get("sample_id", "")
        pred = r.get("prediction", "")
        gt_list = r.get("ground_truth_list", [])
        is_img_ok = bool(r.get("is_image_correct", False))

        scores = score_prediction(pred, gt_list)
        hit = scores["hit"]
        is_joint = is_img_ok and hit

        img_correct_cnt += int(is_img_ok)
        ans_hit_cnt += int(hit)
        joint_cnt += int(is_joint)
        em_cnt += int(scores["exact_match"])
        total_f1 += scores["f1"]
        total_rec += scores["recall"]
        total_prec += scores["precision"]

        lat = float(r.get("latency_sec", 0.0))
        total_lat += lat

        rt = r.get("route", "unknown")
        routes[rt] = routes.get(rt, 0) + 1

        for s_key in retention_counts:
            val = r.get("stage_retention", {}).get(s_key)
            if val is not None:
                retention_counts[s_key][1] += 1
                if val:
                    retention_counts[s_key][0] += 1

        for r_key in rejection_sums:
            rejection_sums[r_key] += r.get("stage_rejection", {}).get(r_key, 0)

        for t_key in token_sums:
            token_sums[t_key] += r.get("visual_tokens", {}).get(t_key, 0)

        for k_t, v_t in r.get("timings_sec", {}).items():
            timing_sums[k_t] = timing_sums.get(k_t, 0.0) + float(v_t)

        per_sample_details.append({
            "sample_id": s_id,
            "img_ok": is_img_ok,
            "hit": hit,
            "joint": is_joint,
            "gt": gt_list,
            "pred": pred,
            "route": rt,
            "latency_sec": lat,
        })

    img_acc = (img_correct_cnt / n) * 100.0
    ans_acc = (ans_hit_cnt / n) * 100.0
    joint_acc = (joint_cnt / n) * 100.0
    em_acc = (em_cnt / n) * 100.0
    avg_f1 = (total_f1 / n) * 100.0
    avg_rec = (total_rec / n) * 100.0
    avg_lat = total_lat / n

    print("=" * 72)
    print("STASHBENCH EVALUATION SUMMARY")
    print("=" * 72)
    print(f"File              : {os.path.basename(input_path)}")
    print(f"Total Samples     : {n}")
    print(f"Image Selection   : {img_acc:5.1f}% ({img_correct_cnt}/{n})")
    print(f"Answer Match (Hit): {ans_acc:5.1f}% ({ans_hit_cnt}/{n})")
    print(f"Joint (Img + Ans) : {joint_acc:5.1f}% ({joint_cnt}/{n})")
    print(f"Exact Match (EM)  : {em_acc:5.1f}% ({em_cnt}/{n})")
    print(f"Token Recall      : {avg_rec:5.1f}%")
    print(f"Token F1 Score    : {avg_f1:5.1f}%")
    print(f"Average Latency   : {avg_lat:5.2f}s")
    print("-" * 72)
    print("Pipeline Retention Rates (Target Bounding Box Preserved):")
    for s_k, (hit_c, tot_c) in retention_counts.items():
        pct = (hit_c / tot_c * 100.0) if tot_c > 0 else 0.0
        print(f"  {s_k:<12}: {pct:5.1f}% ({hit_c}/{tot_c})")
    print("-" * 72)
    print("Average Noise Rejection per Sample:")
    print(f"  Stage 3 Noise Patches    : {rejection_sums['stage3_noise_patches'] / n:6.1f}")
    print(f"  Stage 4 Noise Patches    : {rejection_sums['stage4_noise_patches'] / n:6.1f}")
    print(f"  Stage 5 Noise Crops      : {rejection_sums['stage5_noise_crops'] / n:6.1f}")
    print(f"  Stage 6 Distractor Reads : {rejection_sums['stage6_distractor_reads'] / n:6.1f}")
    print("-" * 72)
    print("Visual Token Consumption:")
    print(f"  Stage 2 Direct Tokens    : {token_sums['stage2_direct_tokens'] / n:7.0f}")
    print(f"  Stage 6 Reader Tokens    : {token_sums['stage6_reader_tokens'] / n:7.0f}")
    print(f"  Total VLM Tokens / Sample: {token_sums['total_vlm_tokens'] / n:7.0f}")
    print("-" * 72)
    print("Stage Latency Breakdown (Avg Seconds):")
    for st, total_s in sorted(timing_sums.items()):
        print(f"  {st:<24}: {total_s / n:6.2f}s")
    print("-" * 72)
    print("Execution Routes:")
    for rt, cnt in sorted(routes.items(), key=lambda x: -x[1]):
        print(f"  {rt:<24}: {cnt:3d} ({cnt / n * 100.0:5.1f}%)")
    print("=" * 72)

    if args.verbose:
        print("\nPER-SAMPLE BREAKDOWN:")
        for sm in per_sample_details:
            mark = "[OK]" if sm["joint"] else ("[IMG_ONLY]" if sm["img_ok"] else ("[ANS_ONLY]" if sm["hit"] else "[FAIL]"))
            print(f"  #{sm['sample_id']} {mark:<10} Route: {sm['route']:<20} Time: {sm['latency_sec']:5.1f}s | GT: {sm['gt']} | Pred: '{sm['pred']}'")

    summary_file = args.save_summary or input_path.replace(".jsonl", "_metrics.json")
    summary_data = {
        "file": input_path,
        "num_samples": n,
        "image_selection_acc": round(img_acc, 2),
        "answer_match_acc": round(ans_acc, 2),
        "joint_acc": round(joint_acc, 2),
        "exact_match_acc": round(em_acc, 2),
        "token_recall": round(avg_rec, 2),
        "token_f1": round(avg_f1, 2),
        "avg_latency_sec": round(avg_lat, 2),
        "retention_rates": {k: round(v[0] / v[1] * 100, 2) if v[1] > 0 else 0.0 for k, v in retention_counts.items()},
        "avg_rejection": {k: round(v / n, 2) for k, v in rejection_sums.items()},
        "avg_visual_tokens": {k: round(v / n, 1) for k, v in token_sums.items()},
        "avg_timings_sec": {k: round(v / n, 3) for k, v in timing_sums.items()},
        "routes": routes,
    }
    with open(summary_file, "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics summary to: {summary_file}")

if __name__ == "__main__":
    main()
