#!/usr/bin/env python3
r"""
summarize_results.py: Comprehensive Component & Telemetry Statistics Summarizer.
Analyzes evaluation results JSONL files (from eval_runner.py) across all MSSearch pipeline stages,
with optional deep correlation against STASHBench dataset annotations.

Usage:
    python experiments/our/summarize_results.py [path/to/results.jsonl]
    python experiments/our/summarize_results.py results/results.jsonl --annotation "D:\MachineLearning\Datasets\STASHBench\annotations_part1234_merged.json"
    python experiments/our/summarize_results.py results/results.jsonl --format markdown --save results/summary_report.md
"""

import os
import sys
import json
import re
import math
import argparse
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict


DEFAULT_ANNOTATION_PATHS = [
    r"D:\MachineLearning\Datasets\STASHBench\annotations_part1234_merged.json",
    "/data/shared/cvpr/dat/simple/data/STASHBench/annotations_part1234_merged.json",
    "/data/STASHBench/annotations_part1234_merged.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize MSSearch components and telemetry statistics.")
    parser.add_argument("input_file", nargs="?", default="results/results.jsonl",
                        help="Path to results.jsonl file. Defaults to 'results/results.jsonl'.")
    parser.add_argument("--annotation", type=str, default=None,
                        help="Path to STASHBench annotations_part1234_merged.json for category and bbox breakdown.")
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="text",
                        help="Output display format: 'text' (default), 'markdown', or 'json'.")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional file path to save the formatted report.")
    parser.add_argument("--pool_mode", type=str, default=None,
                        help="Filter evaluation results by specific pool mode (e.g. N_1, N_5).")
    return parser.parse_args()


def calc_stats(values: List[float]) -> Dict[str, float]:
    """Calculates mean, median, std, min, max, sum for a numeric list."""
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
    vals_sorted = sorted(vals)
    n = len(vals)
    mean_val = sum(vals) / n
    median_val = vals_sorted[n // 2] if n % 2 == 1 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2.0
    variance = sum((x - mean_val) ** 2 for x in vals) / n if n > 1 else 0.0
    return {
        "count": n,
        "mean": mean_val,
        "median": median_val,
        "std": math.sqrt(variance),
        "min": vals_sorted[0],
        "max": vals_sorted[-1],
        "sum": sum(vals)
    }


def normalize_answer(text: str) -> str:
    """Lowercases, removes punctuation and extra whitespace for relaxed lexical match."""
    return re.sub(r"[^\w\s]", "", str(text).lower()).strip()


def resolve_annotation_path(explicit_path: Optional[str]) -> Optional[str]:
    """Finds valid STASHBench annotation path from argument, env, or default candidate paths."""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    env_p = os.environ.get("DATASET_PATH")
    if env_p and os.path.isfile(env_p):
        return env_p
    for cand in DEFAULT_ANNOTATION_PATHS:
        if os.path.isfile(cand):
            return cand
    return None


def analyze_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes comprehensive metrics across stages, accuracy, latency, and tokens."""
    total_samples = len(records)
    if total_samples == 0:
        return {"total_samples": 0}

    # 1. Overview & Routing
    pools = Counter(r.get("pool_name", "Unknown") for r in records)
    routes = Counter(r.get("route", "Unknown") for r in records)
    stage5_modes = Counter(r.get("stage5_mode", "Unknown") for r in records)

    # 2. Accuracy & Localization
    primary_correct = sum(1 for r in records if r.get("is_primary_image_correct") is True)
    image_correct = sum(1 for r in records if r.get("is_image_correct") is True)

    # Relaxed QA matching (prediction contains any ground truth string)
    qa_matches = 0
    for r in records:
        pred_norm = normalize_answer(r.get("prediction", ""))
        gts = [normalize_answer(gt) for gt in r.get("ground_truth_list", []) if gt]
        if any(gt in pred_norm for gt in gts if gt):
            qa_matches += 1

    # Stage 4 Retrieval & Ranking
    r1_list = [r.get("stage4_r1") for r in records if r.get("stage4_r1") is not None]
    mrr_list = [r.get("stage4_mrr") for r in records if r.get("stage4_mrr") is not None]
    ranks_list = [r.get("gt_rank_stage4") for r in records if r.get("gt_rank_stage4") is not None]
    pruned_list = [r.get("distractor_imgs_pruned") for r in records if r.get("distractor_imgs_pruned") is not None]

    # 3. Stage Latencies
    timing_keys = [
        ("phase1_direct_answer", "Phase 1: Direct Global Answer"),
        ("stage1_parser", "Stage 1: Question Parser"),
        ("stage2_fast_path", "Stage 2: Tool Fast Path"),
        ("stage3_partitioning", "Stage 3: Spatial Partitioning"),
        ("stage4_grouping_gating", "Stage 4: Grouping & Gating"),
        ("stage5_screening", "Stage 5: Resolution Screening"),
        ("stage6_reading", "Stage 6: Foveated Reading"),
    ]
    latencies = {}
    for k, label in timing_keys:
        vals = [r.get("timings_sec", {}).get(k) for r in records if r.get("timings_sec", {}).get(k) is not None]
        latencies[k] = {"label": label, "stats": calc_stats(vals)}

    total_latency_stats = calc_stats([r.get("latency_sec") for r in records if r.get("latency_sec") is not None])

    # 4. Stage Retention Rates
    retention = {}
    for sk in ["stage3", "stage4", "stage5", "stage6"]:
        vals = [r.get("stage_retention", {}).get(sk) for r in records if r.get("stage_retention", {}).get(sk) is not None]
        survived = sum(1 for v in vals if v is True)
        rate = (survived / len(vals) * 100) if vals else 0.0
        retention[sk] = {"survived": survived, "total": len(vals), "rate": rate}

    iou_stats = calc_stats([r.get("max_iou_stage4") for r in records if r.get("max_iou_stage4") is not None])

    # 5. Stage Noise Rejection
    rejection_keys = [
        ("stage3_noise_patches", "Stage 3 Noise Patches Filtered"),
        ("stage4_noise_patches", "Stage 4 Noise Patches Filtered"),
        ("stage5_noise_crops", "Stage 5 Noise Crops Filtered"),
        ("stage6_distractor_reads", "Stage 6 Distractor Crops Avoided"),
    ]
    rejection = {}
    for k, label in rejection_keys:
        vals = [r.get("stage_rejection", {}).get(k) for r in records if r.get("stage_rejection", {}).get(k) is not None]
        rejection[k] = {"label": label, "stats": calc_stats(vals)}

    # 6. Unified Crops & Visual Tokens
    crops_stats = calc_stats([r.get("num_unified_crops") for r in records if r.get("num_unified_crops") is not None])

    crop_widths, crop_heights, crop_aspects = [], [], []
    for r in records:
        for size in r.get("stage6_crop_sizes", []):
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                w, h = size[0], size[1]
                if w > 0 and h > 0:
                    crop_widths.append(w)
                    crop_heights.append(h)
                    crop_aspects.append(max(w, h) / min(w, h))

    token_keys = [
        ("stage2_direct_tokens", "Direct Global Answer Tokens"),
        ("stage6_reader_tokens", "Stage 6 Foveated Reader Tokens"),
        ("total_vlm_tokens", "Total Pipeline VLM Tokens"),
    ]
    tokens = {}
    for k, label in token_keys:
        vals = [r.get("visual_tokens", {}).get(k) for r in records if r.get("visual_tokens", {}).get(k) is not None]
        tokens[k] = {"label": label, "stats": calc_stats(vals)}

    return {
        "total_samples": total_samples,
        "pools": dict(pools),
        "routes": dict(routes),
        "stage5_modes": dict(stage5_modes),
        "primary_image_acc": {
            "correct": primary_correct,
            "total": total_samples,
            "rate": (primary_correct / total_samples * 100) if total_samples else 0.0
        },
        "image_pool_acc": {
            "correct": image_correct,
            "total": total_samples,
            "rate": (image_correct / total_samples * 100) if total_samples else 0.0
        },
        "qa_relaxed_acc": {
            "correct": qa_matches,
            "total": total_samples,
            "rate": (qa_matches / total_samples * 100) if total_samples else 0.0
        },
        "stage4_retrieval": {
            "r1": calc_stats(r1_list),
            "mrr": calc_stats(mrr_list),
            "ranks": Counter(ranks_list),
            "distractor_imgs_pruned": calc_stats(pruned_list)
        },
        "latencies": latencies,
        "total_latency": total_latency_stats,
        "retention": retention,
        "stage4_iou": iou_stats,
        "rejection": rejection,
        "unified_crops": crops_stats,
        "crop_dimensions": {
            "widths": calc_stats(crop_widths),
            "heights": calc_stats(crop_heights),
            "aspect_ratios": calc_stats(crop_aspects),
        },
        "tokens": tokens,
    }


def analyze_with_annotations(records: List[Dict[str, Any]], annot_file: str) -> Optional[Dict[str, Any]]:
    """Cross-references results records with dataset ground truth annotations."""
    if not os.path.isfile(annot_file):
        return None

    with open(annot_file, "r", encoding="utf-8") as f:
        annot_data = json.load(f)

    all_samples = annot_data.get("samples", [])
    annot_by_id = {s["sample_id"]: s for s in all_samples}
    total_dataset_samples = len(all_samples)

    # 1. Dataset Coverage & Progress
    all_parts = Counter(s.get("source_part", "Unknown") for s in all_samples)
    eval_parts = Counter()
    unprocessed_by_part = defaultdict(list)

    matched_records = []
    consistency_issues = []

    for s in all_samples:
        sid = s["sample_id"]
        part = s.get("source_part", "Unknown")
        matched_eval = next((r for r in records if r.get("sample_id") == sid), None)
        if matched_eval:
            eval_parts[part] += 1
            matched_records.append((matched_eval, s))

            # Consistency checks
            q_eval = normalize_answer(matched_eval.get("question", ""))
            q_annot = normalize_answer(s.get("question", ""))
            if q_eval != q_annot:
                consistency_issues.append(f"Sample {sid}: Question mismatch.")

            gt_eval_id = str(matched_eval.get("gt_image_id", ""))
            gt_annot_id = str(s.get("gt_image", {}).get("image_id", ""))
            if gt_eval_id and gt_annot_id and gt_eval_id != gt_annot_id:
                consistency_issues.append(f"Sample {sid}: GT image ID mismatch ({gt_eval_id} vs {gt_annot_id}).")
        else:
            unprocessed_by_part[part].append(sid)

    # 2. Performance Breakdown by Question Type
    by_qtype = defaultdict(list)
    by_atype = defaultdict(list)
    by_size = defaultdict(list)

    for r, a in matched_records:
        qtype = a.get("question_type", "Unknown")
        by_qtype[qtype].append((r, a))

        atype = a.get("answer_type", "Unknown")
        by_atype[atype].append((r, a))

        ratio = a.get("bbox_area_ratio")
        if ratio is not None:
            if ratio < 0.0002:
                s_bucket = "Extreme Micro (<0.02%)"
            elif ratio < 0.0005:
                s_bucket = "Micro (0.02% - 0.05%)"
            elif ratio < 0.0008:
                s_bucket = "Small (0.05% - 0.08%)"
            else:
                s_bucket = "Standard (0.08% - 0.10%)"
            by_size[s_bucket].append((r, a))

    def summarize_subset(items):
        n = len(items)
        if n == 0:
            return {}
        p_acc = sum(1 for r, a in items if r.get("is_primary_image_correct") is True) / n * 100
        pool_acc = sum(1 for r, a in items if r.get("is_image_correct") is True) / n * 100
        qa_acc = sum(1 for r, a in items if any(normalize_answer(gt) in normalize_answer(r.get("prediction", "")) for gt in r.get("ground_truth_list", []))) / n * 100
        s3_ret = sum(1 for r, a in items if r.get("stage_retention", {}).get("stage3") is True) / n * 100
        s4_ret = sum(1 for r, a in items if r.get("stage_retention", {}).get("stage4") is True) / n * 100
        s5_ret = sum(1 for r, a in items if r.get("stage_retention", {}).get("stage5") is True) / n * 100
        s6_ret = sum(1 for r, a in items if r.get("stage_retention", {}).get("stage6") is True) / n * 100
        avg_lat = sum(r.get("latency_sec", 0.0) for r, a in items) / n
        ious = [r.get("max_iou_stage4") for r, a in items if r.get("max_iou_stage4") is not None]
        avg_iou = sum(ious) / len(ious) if ious else 0.0
        return {
            "count": n,
            "primary_acc": p_acc,
            "pool_acc": pool_acc,
            "qa_acc": qa_acc,
            "stage3_ret": s3_ret,
            "stage4_ret": s4_ret,
            "stage5_ret": s5_ret,
            "stage6_ret": s6_ret,
            "avg_latency": avg_lat,
            "avg_iou": avg_iou
        }

    qtype_summary = {k: summarize_subset(v) for k, v in by_qtype.items()}
    atype_summary = {k: summarize_subset(v) for k, v in by_atype.items()}
    size_summary = {k: summarize_subset(v) for k, v in by_size.items()}

    return {
        "annot_file": annot_file,
        "total_dataset_samples": total_dataset_samples,
        "matched_count": len(matched_records),
        "parts_total": dict(all_parts),
        "parts_evaluated": dict(eval_parts),
        "unprocessed_count_by_part": {k: len(v) for k, v in unprocessed_by_part.items()},
        "unprocessed_sample_ids": dict(unprocessed_by_part),
        "consistency_issues": consistency_issues,
        "by_question_type": qtype_summary,
        "by_answer_type": atype_summary,
        "by_target_size": size_summary,
    }


def format_text_report(summary: Dict[str, Any], file_path: str, annot_summary: Optional[Dict[str, Any]] = None) -> str:
    """Renders the summary report as clean formatted plain text."""
    N = summary["total_samples"]
    if N == 0:
        return f"No records found in '{file_path}'."

    lines = []
    div = "=" * 80
    subdiv = "-" * 80

    lines.append(div)
    lines.append(f"MSSEARCH V2 COMPONENT & TELEMETRY SUMMARY REPORT")
    lines.append(f"Source File : {file_path}")
    lines.append(f"Evaluated   : {N} samples | Pools: {summary['pools']}")
    if annot_summary:
        lines.append(f"Annotation  : {annot_summary['annot_file']} (Matched: {annot_summary['matched_count']} / {annot_summary['total_dataset_samples']})")
    lines.append(div)

    # 1. Pipeline Execution Routes
    lines.append("\n[1] PIPELINE EXECUTION & GATING ROUTING")
    lines.append(subdiv)
    for route, count in summary["routes"].items():
        pct = (count / N) * 100
        lines.append(f"  • Route: {route:<26s} : {count:4d} / {N} ({pct:5.1f}%)")
    for s5m, count in summary["stage5_modes"].items():
        pct = (count / N) * 100
        lines.append(f"  • Stage 5 Mode: {s5m:<19s} : {count:4d} / {N} ({pct:5.1f}%)")

    # 2. Accuracy & Localization
    lines.append("\n[2] ACCURACY & LOCALIZATION BENCHMARKS")
    lines.append(subdiv)
    p_acc = summary["primary_image_acc"]
    lines.append(f"  • Primary Evidence Image Acc (Top-1)  : {p_acc['rate']:5.2f}% ({p_acc['correct']}/{p_acc['total']})")
    i_acc = summary["image_pool_acc"]
    lines.append(f"  • Image Pool Hit Rate (Recall@K)      : {i_acc['rate']:5.2f}% ({i_acc['correct']}/{i_acc['total']})")
    qa_acc = summary["qa_relaxed_acc"]
    lines.append(f"  • Answer Exact / Containment Match    : {qa_acc['rate']:5.2f}% ({qa_acc['correct']}/{qa_acc['total']})")

    s4_ret = summary["stage4_retrieval"]
    lines.append(f"  • Stage 4 Image Recall@1 (R@1)        : {s4_ret['r1']['mean']*100:5.2f}% (MRR: {s4_ret['mrr']['mean']:.4f})")
    lines.append(f"  • Stage 4 Distractor Images Pruned    : Avg {s4_ret['distractor_imgs_pruned']['mean']:.2f} imgs (Max: {s4_ret['distractor_imgs_pruned']['max']:.0f})")

    # 3. Target Retention & Noise Rejection
    lines.append("\n[3] HIERARCHICAL STAGE RETENTION & NOISE REJECTION")
    lines.append(subdiv)
    ret = summary["retention"]
    lines.append(f"  • Stage 3 Spatial Partitioning Ret.   : {ret['stage3']['rate']:5.1f}% ({ret['stage3']['survived']}/{ret['stage3']['total']})")
    lines.append(f"  • Stage 4 Grouping & Gating Ret.      : {ret['stage4']['rate']:5.1f}% ({ret['stage4']['survived']}/{ret['stage4']['total']})")
    lines.append(f"  • Stage 5 Resolution Screening Ret.   : {ret['stage5']['rate']:5.1f}% ({ret['stage5']['survived']}/{ret['stage5']['total']})")
    lines.append(f"  • Stage 6 Foveated Reader Ret.        : {ret['stage6']['rate']:5.1f}% ({ret['stage6']['survived']}/{ret['stage6']['total']})")
    lines.append(f"  • Stage 4 GT Bounding Box Max IoU     : Mean={summary['stage4_iou']['mean']:.4f} | Median={summary['stage4_iou']['median']:.4f} | Max={summary['stage4_iou']['max']:.4f}")

    lines.append("")
    for k, d in summary["rejection"].items():
        st = d["stats"]
        lines.append(f"  • {d['label']:<36s}: Mean={st['mean']:6.1f} | Med={st['median']:6.1f} | Max={st['max']:6.0f} | Total={st['sum']:8,.0f}")

    # 4. Crop & Token Economy
    lines.append("\n[4] FOVEATED READING & VISUAL TOKEN ECONOMY")
    lines.append(subdiv)
    uc = summary["unified_crops"]
    lines.append(f"  • Unified Crops per Sample (Stage 6)  : Mean={uc['mean']:5.1f} | Med={uc['median']:5.1f} | Min={uc['min']:3.0f} | Max={uc['max']:3.0f}")

    cd = summary["crop_dimensions"]
    if cd["widths"]["count"] > 0:
        lines.append(f"  • Crop Resolutions (W x H)            : Avg {cd['widths']['mean']:.0f}x{cd['heights']['mean']:.0f} px (Aspect: {cd['aspect_ratios']['mean']:.2f}x)")

    for k, d in summary["tokens"].items():
        st = d["stats"]
        lines.append(f"  • {d['label']:<36s}: Mean={st['mean']:9,.0f} | Med={st['median']:9,.0f} | Total={st['sum']:12,.0f}")

    # 5. Latency & Bottleneck Breakdown
    lines.append("\n[5] PER-STAGE LATENCY & BOTTLENECK PROFILING")
    lines.append(subdiv)
    lines.append(f"  {'Stage / Component':<34s} {'Mean (s)':>9s} {'Med (s)':>9s} {'Min (s)':>8s} {'Max (s)':>8s} {'Total (m)':>10s} {'Share':>7s}")
    lines.append("  " + "-" * 76)

    total_time_sum = summary["total_latency"]["sum"]
    for k, d in summary["latencies"].items():
        st = d["stats"]
        share = (st["sum"] / total_time_sum * 100) if total_time_sum > 0 else 0.0
        total_min = st["sum"] / 60.0
        lines.append(
            f"  {d['label']:<34s} {st['mean']:9.2f} {st['median']:9.2f} {st['min']:8.2f} {st['max']:8.2f} {total_min:10.1f} {share:6.1f}%"
        )
    lines.append("  " + "-" * 76)
    tot = summary["total_latency"]
    tot_hr = tot["sum"] / 3600.0
    lines.append(
        f"  {'TOTAL PIPELINE LATENCY':<34s} {tot['mean']:9.2f} {tot['median']:9.2f} {tot['min']:8.2f} {tot['max']:8.2f} {tot['sum']/60.0:10.1f} (Total: {tot_hr:.2f}h)"
    )

    # 6. STASHBench Annotation Breakdown (if available)
    if annot_summary:
        lines.append("\n[6] STASHBENCH ANNOTATION BREAKDOWN & CROSS-VERIFICATION")
        lines.append(subdiv)
        lines.append(f"  • Dataset Progress: {annot_summary['matched_count']}/{annot_summary['total_dataset_samples']} evaluated")
        for part, total_p in sorted(annot_summary["parts_total"].items()):
            done_p = annot_summary["parts_evaluated"].get(part, 0)
            lines.append(f"    - {part:<10s}: {done_p:2d} / {total_p:2d} done ({done_p/total_p*100:5.1f}%) | Remaining: {total_p - done_p:2d}")

        if annot_summary["consistency_issues"]:
            lines.append(f"  • Consistency Warnings: {len(annot_summary['consistency_issues'])} detected!")
            for issue in annot_summary["consistency_issues"][:3]:
                lines.append(f"    ! {issue}")
        else:
            lines.append("  • Integrity Check: 100% agreement on questions, GT image IDs, and answer references.")

        lines.append("\n  [6.1] Performance by Question Type:")
        lines.append(f"  {'Question Type':<16s} {'N':>4s} {'Prim Acc':>10s} {'Pool Hit':>10s} {'QA Match':>10s} {'S6 Ret':>9s} {'Avg Lat (s)':>12s}")
        lines.append("  " + "-" * 65)
        for qtype, s in sorted(annot_summary["by_question_type"].items()):
            lines.append(f"  {qtype:<16s} {s['count']:4d} {s['primary_acc']:9.1f}% {s['pool_acc']:9.1f}% {s['qa_acc']:9.1f}% {s['stage6_ret']:8.1f}% {s['avg_latency']:11.1f}s")

        lines.append("\n  [6.2] Performance by Target Bounding Box Area Ratio (Canvas Scale):")
        lines.append(f"  {'BBox Size Bucket':<25s} {'N':>4s} {'Prim Acc':>10s} {'S3 Ret':>9s} {'S6 Ret':>9s} {'QA Match':>10s} {'S4 IoU':>9s}")
        lines.append("  " + "-" * 73)
        for sbucket in ["Extreme Micro (<0.02%)", "Micro (0.02% - 0.05%)", "Small (0.05% - 0.08%)", "Standard (0.08% - 0.10%)"]:
            s = annot_summary["by_target_size"].get(sbucket)
            if s:
                lines.append(f"  {sbucket:<25s} {s['count']:4d} {s['primary_acc']:9.1f}% {s['stage3_ret']:8.1f}% {s['stage6_ret']:8.1f}% {s['qa_acc']:9.1f}% {s['avg_iou']:9.4f}")

        lines.append("\n  [6.3] Performance by Answer Type:")
        lines.append(f"  {'Answer Type':<22s} {'N':>4s} {'Prim Acc':>10s} {'Pool Hit':>10s} {'QA Match':>10s}")
        lines.append("  " + "-" * 52)
        for atype, s in sorted(annot_summary["by_answer_type"].items()):
            lines.append(f"  {atype:<22s} {s['count']:4d} {s['primary_acc']:9.1f}% {s['pool_acc']:9.1f}% {s['qa_acc']:9.1f}%")

    lines.append(div)
    return "\n".join(lines)


def format_markdown_report(summary: Dict[str, Any], file_path: str, annot_summary: Optional[Dict[str, Any]] = None) -> str:
    """Renders the summary report as standard GitHub-flavored Markdown."""
    N = summary["total_samples"]
    if N == 0:
        return f"No records found in `{file_path}`."

    md = []
    md.append(f"# MSSearch V2 Component & Telemetry Summary\n")
    md.append(f"**Source File:** `{file_path}` | **Total Samples Evaluated:** {N} | **Pools:** `{summary['pools']}`\n")
    if annot_summary:
        md.append(f"**Dataset Annotations:** `{annot_summary['annot_file']}` (Matched: {annot_summary['matched_count']} / {annot_summary['total_dataset_samples']})\n")

    # Accuracy Table
    md.append("## 1. Accuracy & Evidence Localization\n")
    md.append("| Metric | Result | Count / Total |")
    md.append("|---|:---:|:---:|")
    p_acc = summary["primary_image_acc"]
    md.append(f"| **Primary Evidence Image Top-1 Acc** | **{p_acc['rate']:.2f}%** | {p_acc['correct']} / {p_acc['total']} |")
    i_acc = summary["image_pool_acc"]
    md.append(f"| **Image Pool Recall@K** | **{i_acc['rate']:.2f}%** | {i_acc['correct']} / {i_acc['total']} |")
    qa_acc = summary["qa_relaxed_acc"]
    md.append(f"| **Answer Match (Containment)** | **{qa_acc['rate']:.2f}%** | {qa_acc['correct']} / {qa_acc['total']} |")
    s4_ret = summary["stage4_retrieval"]
    md.append(f"| **Stage 4 Image Recall@1 (R@1)** | **{s4_ret['r1']['mean']*100:.2f}%** | MRR: {s4_ret['mrr']['mean']:.4f} |")
    md.append(f"| **Distractor Images Pruned (Stage 4)** | Avg **{s4_ret['distractor_imgs_pruned']['mean']:.2f}** imgs | Max: {s4_ret['distractor_imgs_pruned']['max']:.0f} |")
    md.append("")

    # Routing & Gating
    md.append("## 2. Execution Routing & Gating\n")
    md.append("| Pipeline Route | Count | Share (%) |")
    md.append("|---|:---:|:---:|")
    for route, count in summary["routes"].items():
        md.append(f"| `{route}` | {count} | {count/N*100:.1f}% |")
    md.append("")
    md.append("| Stage 5 Screening Mode | Count | Share (%) |")
    md.append("|---|:---:|:---:|")
    for s5m, count in summary["stage5_modes"].items():
        md.append(f"| `{s5m}` | {count} | {count/N*100:.1f}% |")
    md.append("")

    # Retention & Rejection Table
    md.append("## 3. Hierarchical Retention & Noise Rejection\n")
    md.append("| Stage | Target Retention | Noise Rejection (Mean / Total) |")
    md.append("|---|:---:|:---:|")
    ret = summary["retention"]
    rej = summary["rejection"]
    md.append(f"| **Stage 3 Partitioning** | {ret['stage3']['rate']:.1f}% ({ret['stage3']['survived']}/{ret['stage3']['total']}) | {rej['stage3_noise_patches']['stats']['mean']:.1f} patches ({rej['stage3_noise_patches']['stats']['sum']:,.0f} total) |")
    md.append(f"| **Stage 4 Grouping & Gating** | {ret['stage4']['rate']:.1f}% ({ret['stage4']['survived']}/{ret['stage4']['total']}) | {rej['stage4_noise_patches']['stats']['mean']:.1f} patches ({rej['stage4_noise_patches']['stats']['sum']:,.0f} total) |")
    md.append(f"| **Stage 5 Screening** | {ret['stage5']['rate']:.1f}% ({ret['stage5']['survived']}/{ret['stage5']['total']}) | {rej['stage5_noise_crops']['stats']['mean']:.1f} crops ({rej['stage5_noise_crops']['stats']['sum']:,.0f} total) |")
    md.append(f"| **Stage 6 Foveated Reader** | {ret['stage6']['rate']:.1f}% ({ret['stage6']['survived']}/{ret['stage6']['total']}) | {rej['stage6_distractor_reads']['stats']['mean']:.1f} reads avoided ({rej['stage6_distractor_reads']['stats']['sum']:,.0f} total) |")
    md.append(f"\n*Stage 4 Target Box IoU: Mean = **{summary['stage4_iou']['mean']:.4f}**, Median = **{summary['stage4_iou']['median']:.4f}**, Max = **{summary['stage4_iou']['max']:.4f}**.*\n")

    # Tokens & Crops
    md.append("## 4. Visual Tokens & Crop Statistics\n")
    md.append("| Metric | Mean | Median | Min | Max | Total |")
    md.append("|---|:---:|:---:|:---:|:---:|:---:|")
    uc = summary["unified_crops"]
    md.append(f"| **Stage 6 Unified Crops Read** | {uc['mean']:.1f} | {uc['median']:.1f} | {uc['min']:.0f} | {uc['max']:.0f} | {uc['sum']:,.0f} |")
    for k, d in summary["tokens"].items():
        st = d["stats"]
        md.append(f"| **{d['label']}** | {st['mean']:,.0f} | {st['median']:,.0f} | {st['min']:,.0f} | {st['max']:,.0f} | {st['sum']:,.0f} |")
    md.append("")

    # Latencies Table
    md.append("## 5. Per-Stage Latency & Bottleneck Breakdown\n")
    md.append("| Component | Mean (s) | Median (s) | Min (s) | Max (s) | Total (min) | Share (%) |")
    md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    total_time_sum = summary["total_latency"]["sum"]
    for k, d in summary["latencies"].items():
        st = d["stats"]
        share = (st["sum"] / total_time_sum * 100) if total_time_sum > 0 else 0.0
        md.append(f"| {d['label']} | {st['mean']:.2f} | {st['median']:.2f} | {st['min']:.2f} | {st['max']:.2f} | {st['sum']/60.0:.1f} | {share:.1f}% |")
    tot = summary["total_latency"]
    md.append(f"| **TOTAL PIPELINE** | **{tot['mean']:.2f}** | **{tot['median']:.2f}** | **{tot['min']:.2f}** | **{tot['max']:.2f}** | **{tot['sum']/60.0:.1f}** | **100.0%** |")
    md.append(f"\n*Cumulative execution runtime across {N} samples: **{tot['sum']/3600.0:.2f} hours**.*\n")

    # Annotation breakdown
    if annot_summary:
        md.append("## 6. STASHBench Annotation Breakdown\n")
        md.append("### 6.1 Progress by Dataset Part\n")
        md.append("| Dataset Part | Evaluated | Total Samples | Progress (%) | Remaining |")
        md.append("|---|:---:|:---:|:---:|:---:|")
        for part, total_p in sorted(annot_summary["parts_total"].items()):
            done_p = annot_summary["parts_evaluated"].get(part, 0)
            md.append(f"| `{part}` | {done_p} | {total_p} | {done_p/total_p*100:.1f}% | {total_p - done_p} |")
        md.append("")

        md.append("### 6.2 Breakdown by Question Type\n")
        md.append("| Question Type | Samples | Primary Acc | Pool Hit | QA Match | S6 Ret. | Avg Latency |")
        md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for qtype, s in sorted(annot_summary["by_question_type"].items()):
            md.append(f"| **{qtype}** | {s['count']} | {s['primary_acc']:.1f}% | {s['pool_acc']:.1f}% | {s['qa_acc']:.1f}% | {s['stage6_ret']:.1f}% | {s['avg_latency']:.1f}s |")
        md.append("")

        md.append("### 6.3 Breakdown by Target BBox Area Ratio\n")
        md.append("| Scale Bucket | Samples | Primary Acc | Stage 3 Ret. | Stage 6 Ret. | QA Match | Stage 4 IoU |")
        md.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for sbucket in ["Extreme Micro (<0.02%)", "Micro (0.02% - 0.05%)", "Small (0.05% - 0.08%)", "Standard (0.08% - 0.10%)"]:
            s = annot_summary["by_target_size"].get(sbucket)
            if s:
                md.append(f"| **{sbucket}** | {s['count']} | {s['primary_acc']:.1f}% | {s['stage3_ret']:.1f}% | {s['stage6_ret']:.1f}% | {s['qa_acc']:.1f}% | {s['avg_iou']:.4f} |")
        md.append("")

        md.append("### 6.4 Breakdown by Answer Type\n")
        md.append("| Answer Type | Samples | Primary Acc | Pool Hit | QA Match |")
        md.append("|---|:---:|:---:|:---:|:---:|")
        for atype, s in sorted(annot_summary["by_answer_type"].items()):
            md.append(f"| **{atype}** | {s['count']} | {s['primary_acc']:.1f}% | {s['pool_acc']:.1f}% | {s['qa_acc']:.1f}% |")
        md.append("")

    return "\n".join(md)


def main():
    args = parse_args()
    if not os.path.isfile(args.input_file):
        print(f"Error: Target file '{args.input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    if args.pool_mode:
        records = [r for r in records if r.get("pool_name") == args.pool_mode]

    summary = analyze_records(records)

    # Check and load annotations
    annot_path = resolve_annotation_path(args.annotation)
    annot_summary = None
    if annot_path:
        annot_summary = analyze_with_annotations(records, annot_path)

    if args.format == "json":
        full_res = {"summary": summary, "annotation_breakdown": annot_summary}
        output_str = json.dumps(full_res, indent=2)
    elif args.format == "markdown":
        output_str = format_markdown_report(summary, args.input_file, annot_summary)
    else:
        output_str = format_text_report(summary, args.input_file, annot_summary)

    print(output_str)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(output_str)
        print(f"\n[Saved] Summary report successfully written to '{args.save}'.")


if __name__ == "__main__":
    main()
