"""
HRBench 8K Accuracy Evaluation Script for MSSearch.
Compares `mssearch_prediction` directly against the Ground Truth answer (always "A" in HRBench canonical order).

Usage:
    python eval_hrbench_predictions.py
    python eval_hrbench_predictions.py --input results/mssearch_hrbench8k_predictions_v2.jsonl
    python eval_hrbench_predictions.py --input results/mssearch_hrbench8k_predictions_v2.jsonl --save-wrong wrong_samples.jsonl
"""

import os
import sys
import json
import re
import argparse
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict


def extract_predicted_option(
    pred_str: Any,
    options_raw: Optional[Any] = None
) -> Optional[str]:
    """
    Extracts the chosen option letter ('A', 'B', 'C', 'D') from model prediction.
    
    1. Direct letter at start: 'A. 27B', '(A) Blue', 'A: Front' -> 'A'
    2. Single letter: 'A' -> 'A'
    3. Concluding answer: 'Final Answer: A', 'Answer: Option A' -> 'A'
    4. Option text match: Matches prediction text against option strings in `options_raw`.
    """
    if pred_str is None or not isinstance(pred_str, str):
        return None

    clean_pred = pred_str.strip()
    if not clean_pred or clean_pred.lower() in ("no answer found", "none", "n/a", "[error: image not found]"):
        return None

    # 1. Regex check for leading letter (e.g., 'A. 27B', '(A)', 'A: ')
    m = re.match(r"^[\(\[\"']*([A-Da-d])[\)\]\.\:\s]", clean_pred)
    if m:
        return m.group(1).upper()

    # 2. Exact single letter
    if clean_pred.upper() in ("A", "B", "C", "D"):
        return clean_pred.upper()

    # 3. Final Answer / Option phrase anywhere in text
    m = re.search(r"\b(?:final answer|answer|choice|option)\b[^\w\n\r]*([A-Da-d])\b", clean_pred, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 4. Fallback: match prediction content against option candidate text
    if options_raw:
        opt_dict = parse_options_text(options_raw)
        p_lower = clean_pred.lower()
        # Direct match against option texts
        for letter, text in opt_dict.items():
            if text and (p_lower == text or p_lower.startswith(text) or text in p_lower):
                return letter

    return None


def parse_options_text(options_raw: Any) -> Dict[str, str]:
    """Parses candidate options block into { 'A': text, 'B': text, ... }."""
    opt_dict: Dict[str, str] = {}
    if not options_raw:
        return opt_dict

    # If list of permutation strings, use cycle 0 (first permutation where A is GT)
    if isinstance(options_raw, list) and len(options_raw) > 0:
        opt_block = options_raw[0]
    else:
        opt_block = str(options_raw)

    if isinstance(opt_block, str):
        for line in opt_block.splitlines():
            m = re.match(r"^[\(\[\"']*([A-Da-d])[\)\]\.\:\s]+(.*)$", line.strip())
            if m:
                opt_dict[m.group(1).upper()] = m.group(2).strip().lower()
    elif isinstance(opt_block, dict):
        for k, v in opt_block.items():
            opt_dict[str(k).upper()] = str(v).strip().lower()

    return opt_dict


def evaluate_predictions(
    jsonl_path: str,
    gt_letter: str = "A",
    save_wrong_path: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluates all prediction records in jsonl_path and prints summary report."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Predictions file not found: {jsonl_path}")

    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    records.append(json.loads(line_str))
                except json.JSONDecodeError:
                    continue

    total_samples = len(records)
    if total_samples == 0:
        print(f"[EVAL] Warning: 0 records found in '{jsonl_path}'.")
        return {}

    correct_count = 0
    predictions_counter = Counter()
    by_category = defaultdict(lambda: {"correct": 0, "total": 0, "latencies": []})
    by_route = defaultdict(lambda: {"correct": 0, "total": 0, "latencies": []})
    latencies: List[float] = []
    wrong_records: List[Dict[str, Any]] = []

    for item in records:
        pred_text = item.get("mssearch_prediction", "")
        options = item.get("options", None)
        category = item.get("category", "single")
        route = item.get("route", "unknown")
        latency = float(item.get("latency_sec", 0.0))
        latencies.append(latency)

        # Ground truth in HRBench is canonically "A"
        item_gt = item.get("ground_truth") or item.get("gt_letter") or gt_letter
        if isinstance(item_gt, list) and len(item_gt) > 0:
            item_gt = item_gt[0]
        item_gt = str(item_gt).strip().upper()

        # Extract prediction letter
        pred_letter = extract_predicted_option(pred_text, options)
        predictions_counter[pred_letter or "None"] += 1

        is_correct = (pred_letter == item_gt)
        if is_correct:
            correct_count += 1
        else:
            wrong_records.append({
                "index": item.get("index", item.get("id")),
                "question": item.get("question"),
                "category": category,
                "route": route,
                "gt": item_gt,
                "predicted_letter": pred_letter,
                "mssearch_prediction": pred_text,
                "latency_sec": latency
            })

        by_category[category]["total"] += 1
        by_category[category]["latencies"].append(latency)
        if is_correct:
            by_category[category]["correct"] += 1

        by_route[route]["total"] += 1
        by_route[route]["latencies"].append(latency)
        if is_correct:
            by_route[route]["correct"] += 1

    overall_acc = (correct_count / total_samples) * 100.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    median_latency = sorted_lat[len(sorted_lat) // 2] if sorted_lat else 0.0

    # Print Clean Formatted Summary Report
    sep = "=" * 78
    sub_sep = "-" * 78
    print(sep)
    print(f" HRBENCH 8K EVALUATION REPORT: {os.path.basename(jsonl_path)}")
    print(sep)
    print(f" Total Evaluated Samples : {total_samples}")
    print(f" Ground Truth Standard   : Option '{gt_letter}'")
    print(f" Correct Predictions     : {correct_count} / {total_samples}")
    print(f" Overall Accuracy        : {overall_acc:.2f}%")
    print(f" Average Latency         : {mean_latency:.2f}s  (Median: {median_latency:.2f}s)")
    print(sub_sep)

    print(" 1. ACCURACY BY TASK CATEGORY:")
    print(f"    {'Category':<16} {'Correct':<10} {'Total':<10} {'Accuracy (%)':<15} {'Mean Latency (s)':<18}")
    print("    " + "-" * 66)
    for cat, stats in sorted(by_category.items()):
        c_acc = (stats["correct"] / stats["total"]) * 100.0
        c_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0.0
        print(f"    {cat:<16} {stats['correct']:<10} {stats['total']:<10} {c_acc:6.2f}%         {c_lat:6.2f}s")
    print(sub_sep)

    print(" 2. EXECUTION ROUTING BREAKDOWN:")
    print(f"    {'Route':<26} {'Count':<8} {'Share (%)':<12} {'Route Acc (%)':<15} {'Mean Latency (s)':<18}")
    print("    " + "-" * 72)
    for r, stats in sorted(by_route.items(), key=lambda x: x[1]["total"], reverse=True):
        r_share = (stats["total"] / total_samples) * 100.0
        r_acc = (stats["correct"] / stats["total"]) * 100.0
        r_lat = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0.0
        print(f"    {r:<26} {stats['total']:<8} {r_share:5.1f}%       {r_acc:6.2f}%         {r_lat:6.2f}s")
    print(sub_sep)

    print(" 3. PREDICTED OPTION DISTRIBUTION:")
    for letter in ("A", "B", "C", "D", "None"):
        cnt = predictions_counter[letter]
        pct = (cnt / total_samples) * 100.0
        tag = " [MATCH]" if letter == gt_letter else ""
        print(f"    Option {letter:<5}: {cnt:4d} ({pct:5.1f}%){tag}")
    print(sep)

    # Save wrong samples if requested
    if save_wrong_path:
        os.makedirs(os.path.dirname(save_wrong_path) or ".", exist_ok=True)
        with open(save_wrong_path, "w", encoding="utf-8") as f:
            for item in wrong_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[EVAL] Exported {len(wrong_records)} incorrect samples to: {save_wrong_path}")

    return {
        "total_samples": total_samples,
        "correct_count": correct_count,
        "overall_accuracy": overall_acc,
        "mean_latency_sec": mean_latency,
        "median_latency_sec": median_latency,
        "by_category": {c: {"correct": s["correct"], "total": s["total"], "acc": 100.0 * s["correct"] / s["total"]} for c, s in by_category.items()},
        "by_route": {r: {"correct": s["correct"], "total": s["total"], "acc": 100.0 * s["correct"] / s["total"]} for r, s in by_route.items()},
        "predictions_distribution": dict(predictions_counter)
    }


def main():
    default_input = os.path.join(
        os.path.dirname(__file__), "results", "mssearch_hrbench8k_predictions_v2.jsonl"
    )
    if not os.path.exists(default_input):
        default_input = "results/mssearch_hrbench8k_predictions_v2.jsonl"

    parser = argparse.ArgumentParser(description="Evaluate HRBench 8K predictions against Ground Truth (GT = 'A').")
    parser.add_argument("--input", "-i", type=str, default=default_input, help="Path to predictions .jsonl file.")
    parser.add_argument("--gt", type=str, default="A", help="Ground truth option letter (default: 'A').")
    parser.add_argument("--save-wrong", "-w", type=str, default=None, help="Optional output path to save wrong predictions JSONL.")
    parser.add_argument("--export-json", type=str, default=None, help="Optional output path to save summary metrics JSON.")
    args = parser.parse_args()

    results = evaluate_predictions(
        jsonl_path=args.input,
        gt_letter=args.gt,
        save_wrong_path=args.save_wrong
    )

    if args.export_json:
        os.makedirs(os.path.dirname(args.export_json) or ".", exist_ok=True)
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[EVAL] Saved metrics JSON to: {args.export_json}")


if __name__ == "__main__":
    main()
