"""
Stage-by-Stage Latency Benchmark & Profiling Suite for MSSearch.
Measures wall-clock execution time across all 6 stages of MSSearch:
  - Stage 1: Active Object Parser (NLP / VLM)
  - Stage 2: Tool-Based Fast Path (Full Canvas SAM3 + VLM Anchor)
  - Stage 3: Entropy-Gated Recursive Partitioning (Tri-Factor H(I) >= 0.50)
  - Stage 4: SAM3 Feature Grouping & Gating (cos >= 0.80 + Micro-Fovea VLM)
  - Stage 5: Resolution Screening (Branch A Pass-through vs Branch B SigLIP)
  - Stage 6: Foveated VLM Reading & Consensus Deduplication

Usage:
  python benchmark_stages_latency.py --num_samples 5 --output_json latency_benchmark.json
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

# Ensure MSSearch root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from mssearch.config import MSSearchConfig
from mssearch.stages.stage1_parser import parse_question_for_active_objects
from mssearch.stages.stage2_fast_path import run_tool_based_fast_path, FastPathResult
from mssearch.stages.stage3_partitioning import partition_candidate_images
from mssearch.stages.stage4_grouping_gating import run_feature_grouping_and_gating
from mssearch.stages.stage5_screening import run_resolution_screening
from mssearch.stages.stage6_foveated_reader import run_foveated_vlm_reading


def find_hrbench_dataset() -> Tuple[Optional[str], Optional[str]]:
    """Auto-discovers HRBench annotation and base directory across standard search paths."""
    search_dirs = [
        "D:/Research/VisualSearch/hr-bench_8k",
        "d:/Research/VisualSearch/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_8k/mnt/sdb/llp/datasets/zoom_eye_data/zoom_eye_data/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_4k",
        "/kaggle/input/hr-bench-8k",
        "/kaggle/input/hr-bench-4k",
        "docs/experiments/multisource/data/hr-bench_8k"
    ]
    for d in search_dirs:
        for anno_name in ["annotation_hr-bench_8k.json", "annotation_hr-bench_4k.json", "annotation.json"]:
            p = os.path.join(d, anno_name)
            if os.path.exists(p):
                return p, d
    return None, None


def benchmark_sample(
    sample: Dict[str, Any],
    base_dir: str,
    config: MSSearchConfig,
    sam_engine: Any = None,
    vlm_engine: Any = None,
    siglip_engine: Any = None,
    nlp_engine: Any = None
) -> Dict[str, Any]:
    """Runs a single sample through all 6 stages, profiling exact latency per stage."""
    question = sample.get("question", "")
    rel_path = sample.get("input_image", "")
    img_path = os.path.join(base_dir, rel_path) if base_dir else rel_path

    if os.path.exists(img_path):
        canvas = Image.open(img_path).convert("RGB")
    else:
        # Fallback synthetic canvas matching HRBench dimensions
        canvas = Image.new("RGB", (4992, 7680), color=(235, 230, 225))

    timings = {}
    stage_outputs = {}

    # -------------------------------------------------------------------------
    # Stage 1: Active Object Parser
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    active_objects, parser_strat = parse_question_for_active_objects(
        question=question,
        vlm_model=vlm_engine,
        nlp_model=nlp_engine,
        verbose=False,
        return_strategy=True
    )
    timings["stage1_parser"] = round(time.perf_counter() - t0, 4)
    stage_outputs["active_objects"] = active_objects
    stage_outputs["parser_strategy"] = parser_strat

    # -------------------------------------------------------------------------
    # Stage 2: Tool-Based Fast Path
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    fast_path_res: FastPathResult = run_tool_based_fast_path(
        candidate_images=[canvas],
        active_objects=active_objects,
        question=question,
        sam_model=sam_engine,
        zoom_model=vlm_engine,
        config=config
    )
    timings["stage2_fast_path"] = round(time.perf_counter() - t0, 4)
    stage_outputs["fast_path_triggered"] = fast_path_res.triggered

    detected_boxes = fast_path_res.detected_boxes
    feat_cache = getattr(fast_path_res, "feature_cache", {})

    # -------------------------------------------------------------------------
    # Stage 3: Entropy-Gated Recursive Partitioning (H >= 0.50)
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    bounded_patches = partition_candidate_images(
        candidate_images=[canvas],
        stage2_feature_cache=feat_cache,
        config=config
    )
    timings["stage3_partitioning"] = round(time.perf_counter() - t0, 4)
    stage_outputs["num_bounded_patches"] = len(bounded_patches)

    # -------------------------------------------------------------------------
    # Stage 4: SAM3 Feature Grouping & Gating
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    remaining_patches, stage4_groups = run_feature_grouping_and_gating(
        bounded_patches=bounded_patches,
        candidate_images=[canvas],
        active_objects=active_objects,
        sam_model=sam_engine,
        config=config,
        detected_boxes=detected_boxes,
        zoom_model=vlm_engine,
        return_debug=True,
        question=question
    )
    timings["stage4_grouping_gating"] = round(time.perf_counter() - t0, 4)
    stage_outputs["num_groups"] = len(stage4_groups)
    stage_outputs["num_remaining_patches"] = len(remaining_patches)

    # -------------------------------------------------------------------------
    # Stage 5: Resolution Screening
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    unified_pool = run_resolution_screening(
        remaining_patches=remaining_patches,
        candidate_images=[canvas],
        active_objects=active_objects,
        siglip_model=siglip_engine,
        config=config
    )
    timings["stage5_screening"] = round(time.perf_counter() - t0, 4)
    stage_outputs["unified_pool_size"] = len(unified_pool)

    # -------------------------------------------------------------------------
    # Stage 6: Foveated VLM Reading & Deduplication
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    final_result = run_foveated_vlm_reading(
        unified_pool=unified_pool,
        candidate_images=[canvas],
        question=question,
        zoom_model=vlm_engine,
        config=config
    )
    timings["stage6_foveated_reader"] = round(time.perf_counter() - t0, 4)
    stage_outputs["num_read_crops"] = len(getattr(final_result, "crop_records", []))
    stage_outputs["answers"] = getattr(final_result, "answers", [])

    total_latency = sum(timings.values())
    timings["total_pipeline"] = round(total_latency, 4)

    return {
        "timings": timings,
        "stage_outputs": stage_outputs,
        "canvas_size": list(canvas.size)
    }


def compute_benchmark_statistics(per_sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates per-sample timings into statistical metrics (mean, std, min, max, p50, p90)."""
    stage_keys = [
        "stage1_parser",
        "stage2_fast_path",
        "stage3_partitioning",
        "stage4_grouping_gating",
        "stage5_screening",
        "stage6_foveated_reader",
        "total_pipeline"
    ]
    stats = {}
    n = len(per_sample_results)

    for k in stage_keys:
        vals = [r["timings"][k] for r in per_sample_results if k in r["timings"]]
        if vals:
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            p50_v = float(np.percentile(vals, 50))
            p90_v = float(np.percentile(vals, 90))
            stats[k] = {
                "mean": round(mean_v, 4),
                "std": round(std_v, 4),
                "min": round(min_v, 4),
                "max": round(max_v, 4),
                "p50": round(p50_v, 4),
                "p90": round(p90_v, 4)
            }

    # Proportional breakdown based on mean timings
    tot_mean = stats.get("total_pipeline", {}).get("mean", 1.0)
    for k in stage_keys[:-1]:
        if k in stats:
            stats[k]["pct_total"] = round((stats[k]["mean"] / max(1e-6, tot_mean)) * 100.0, 2)

    return stats


def plot_benchmark_results(stats: Dict[str, Any], output_path: str = "stage_latency_benchmark.png"):
    """Generates a publication-grade 2-panel chart summarizing stage latencies."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] Matplotlib not installed. Skipping chart generation.")
        return

    stages_metadata = [
        ("stage1_parser", "Stage 1: Active Object Parser", "#1976d2", "GPU 0 (VLM / NLP)"),
        ("stage2_fast_path", "Stage 2: Tool-Based Fast Path", "#0097a7", "GPU 1 (SAM3) + GPU 0 (VLM)"),
        ("stage3_partitioning", "Stage 3: Entropy Partitioning (H>=0.50)", "#e65100", "CPU (Tri-Factor H(I))"),
        ("stage4_grouping_gating", "Stage 4: SAM3 Grouping & Gating", "#7b1fa2", "GPU 1 (SAM3) + GPU 0 (VLM)"),
        ("stage5_screening", "Stage 5: Resolution Screening", "#388e3c", "GPU 0 (SigLIP / Pass-through)"),
        ("stage6_foveated_reader", "Stage 6: Foveated VLM Reading & Dedup", "#c2185b", "GPU 0 (Qwen2.5-VL)")
    ]

    latencies = [stats[k]["mean"] for k, _, _, _ in stages_metadata if k in stats]
    std_errs = [stats[k]["std"] for k, _, _, _ in stages_metadata if k in stats]
    labels = [lbl for k, lbl, _, _ in stages_metadata if k in stats]
    colors = [col for k, _, col, _ in stages_metadata if k in stats]
    pcts = [stats[k].get("pct_total", 0.0) for k, _, _, _ in stages_metadata if k in stats]

    total_mean = stats.get("total_pipeline", {}).get("mean", sum(latencies))

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(15.5, 5.2), gridspec_kw={'width_ratios': [1.35, 1.0]})
    fig.patch.set_facecolor('#ffffff')

    # Left: Horizontal Bar Chart with Error Bars
    y_pos = np.arange(len(latencies))
    bars = ax_bar.barh(y_pos, latencies, xerr=std_errs, color=colors, height=0.55,
                       edgecolor='black', linewidth=0.8, alpha=0.9, capsize=4)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, fontsize=9.0, fontweight='bold')
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Mean Execution Latency (Seconds)", fontsize=10, fontweight='bold', labelpad=8)
    ax_bar.set_title("Per-Stage Execution Latency (Mean +/- Std Dev)", fontsize=11.5, fontweight='bold', pad=10)
    ax_bar.grid(axis='x', linestyle=':', alpha=0.6)

    max_w = max(latencies) if latencies else 1.0
    ax_bar.set_xlim(0, max_w * 1.35)

    for bar, val, p in zip(bars, latencies, pcts):
        w = bar.get_width()
        ax_bar.text(w + max_w * 0.03, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}s ({p:.1f}%)", va='center', fontsize=8.5, fontweight='bold', color='#212121')

    # Right: Donut Chart of Latency Distribution
    wedges, texts, autotexts = ax_pie.pie(
        latencies,
        labels=[f"S{i+1}" for i in range(len(latencies))],
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        pctdistance=0.78,
        textprops=dict(fontsize=8.5, fontweight='bold'),
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2.0)
    )
    for at in autotexts:
        at.set_fontsize(7.8)
        at.set_color('white')
        at.set_weight('bold')

    ax_pie.text(0, 0, f"TOTAL\n{total_mean:.2f}s", ha='center', va='center', fontsize=11, fontweight='bold', color='#1a237e')
    ax_pie.set_title("Pipeline Latency Distribution (%)", fontsize=11.5, fontweight='bold', pad=10)

    # Inset Banner
    max_idx = int(np.argmax(latencies))
    b_name = stages_metadata[max_idx][1]
    b_time = latencies[max_idx]
    b_pct = pcts[max_idx]
    banner = f"PRIMARY BOTTLENECK: {b_name} -> {b_time:.2f}s ({b_pct:.1f}% of total)\n" \
             f"Asymmetric GPU Allocation: GPU 0 (Qwen2.5-VL / SigLIP) + GPU 1 (SAM3) + CPU (Spatial Entropy Partitioning)"
    fig.text(0.5, 0.02, banner, ha='center', fontsize=8.5, style='italic', color='#37474f',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#b0bec5', linewidth=1.2))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=180)
    print(f"[BENCHMARK] Saved timing breakdown figure to {output_path}!")


def main():
    parser = argparse.ArgumentParser(description="Benchmark run time for every stage of MSSearch.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of benchmark samples to profile.")
    parser.add_argument("--annotation_path", type=str, default=None, help="Path to annotation JSON file.")
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory of benchmark images.")
    parser.add_argument("--output_json", type=str, default="benchmark_stages_latency.json", help="Output path for JSON stats.")
    parser.add_argument("--output_plot", type=str, default="stage_latency_benchmark.png", help="Output plot path.")
    args = parser.parse_args()

    anno_path = args.annotation_path
    base_dir = args.base_dir

    if not anno_path or not os.path.exists(anno_path):
        auto_anno, auto_base = find_hrbench_dataset()
        if auto_anno:
            anno_path = auto_anno
            base_dir = auto_base
            print(f"[BENCHMARK] Auto-discovered dataset annotation: {anno_path}")
        else:
            print("[BENCHMARK WARNING] Dataset not found in standard paths. Creating synthetic benchmark sample.")
            samples = [{
                "question": "What is the number displayed above the entrance where the woman is standing?",
                "input_image": "test_8192.jpg",
                "answer": "27B"
            }]
            anno_path = "synthetic"
            base_dir = "."

    if anno_path != "synthetic" and os.path.exists(anno_path):
        with open(anno_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"[BENCHMARK] Ingested {len(samples)} samples from {anno_path}")
    else:
        samples = [{
            "question": "What is the number displayed above the entrance where the woman is standing?",
            "input_image": "test_8192.jpg",
            "answer": "27B"
        }]

    num_samples = min(args.num_samples, len(samples))
    print("=" * 80)
    print(f"BENCHMARKING RUN TIME FOR EVERY STAGE (Profiling {num_samples} Samples)")
    print("=" * 80)

    config = MSSearchConfig(
        enable_fast_path=True,
        fast_path_conf_threshold=0.70,
        min_patch_entropy=0.50,
        enable_latent_gating=True,
        latent_confidence_threshold=0.0,
        enable_existence_gating=True,
        existence_confidence_threshold=0.0,
        partition_mode="hybrid",
        max_patch_size=1024,
        max_patches_per_image=64,
        grouping_sim_threshold=0.80,
        branch_b_score_threshold=0.50,
        verbose=False
    )

    per_sample_results = []
    for idx in range(num_samples):
        s = samples[idx]
        print(f"\n[Sample {idx + 1}/{num_samples}] Profiling question: '{s.get('question', '')[:60]}...'")
        res = benchmark_sample(
            sample=s,
            base_dir=base_dir,
            config=config,
            sam_engine=None,
            vlm_engine=None,
            siglip_engine=None,
            nlp_engine=None
        )
        t = res["timings"]
        print(f"  S1: {t['stage1_parser']:.3f}s | S2: {t['stage2_fast_path']:.3f}s | S3: {t['stage3_partitioning']:.3f}s | "
              f"S4: {t['stage4_grouping_gating']:.3f}s | S5: {t['stage5_screening']:.3f}s | S6: {t['stage6_foveated_reader']:.3f}s | "
              f"Total: {t['total_pipeline']:.3f}s")
        per_sample_results.append(res)

    # Compute statistical aggregation
    stats = compute_benchmark_statistics(per_sample_results)

    # Display Summary Table
    print("\n" + "=" * 105)
    print(f"{'Stage':<9} | {'Component Name':<34} | {'Mean (s)':<10} | {'Std (s)':<10} | {'Min (s)':<10} | {'Max (s)':<10} | {'% Total':<8}")
    print("-" * 105)
    names = {
        "stage1_parser": "Stage 1: Active Object Parser",
        "stage2_fast_path": "Stage 2: Tool-Based Fast Path",
        "stage3_partitioning": "Stage 3: Entropy Partitioning",
        "stage4_grouping_gating": "Stage 4: SAM3 Grouping & Gating",
        "stage5_screening": "Stage 5: Resolution Screening",
        "stage6_foveated_reader": "Stage 6: Foveated VLM Reading",
        "total_pipeline": "TOTAL PIPELINE LATENCY"
    }
    for k in ["stage1_parser", "stage2_fast_path", "stage3_partitioning", "stage4_grouping_gating", "stage5_screening", "stage6_foveated_reader"]:
        if k in stats:
            st = stats[k]
            pct_s = f"{st.get('pct_total', 0.0):.1f}%"
            print(f"{k[:6]:<9} | {names[k]:<34} | {st['mean']:>8.3f}s | {st['std']:>8.3f}s | {st['min']:>8.3f}s | {st['max']:>8.3f}s | {pct_s:>7}")

    tot = stats.get("total_pipeline", {})
    print("=" * 105)
    print(f"{'TOTAL':<9} | {'END-TO-END PIPELINE':<34} | {tot.get('mean', 0.0):>8.3f}s | {tot.get('std', 0.0):>8.3f}s | {tot.get('min', 0.0):>8.3f}s | {tot.get('max', 0.0):>8.3f}s | 100.0%")
    print("=" * 105)

    # Export to JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"statistics": stats, "per_sample": per_sample_results}, f, indent=2)
    print(f"\n[BENCHMARK] Exported full benchmark metrics to {args.output_json}")

    # Plot visual chart
    plot_benchmark_results(stats, args.output_plot)


if __name__ == "__main__":
    main()
