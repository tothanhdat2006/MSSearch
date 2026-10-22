#!/usr/bin/env python3
"""
Multi-Source Visual Search Engine (MSSearch) Benchmark & Evaluation Runner
=============================================================================
Lean, production-grade runner calling MSSearch directly from the `mssearch` package.
- Evaluates HRBench samples across candidate image pool C = {I_1, ..., I_N}
- Exposes all hyperparameters via JSON config file and command-line overrides
- Prints clean 1-line final decisions for each stage (via MSSearch summary mode)
- Saves final bounding box coordinates before Stage 5 aggregation for re-visualization
- Writes comprehensive JSONL telemetry logs, summary stats, and config snapshots
"""

import os
import sys
import gc
import re
import json
import time
import random
import argparse
from typing import List, Dict, Tuple, Optional, Any, Set, Union

import torch
from PIL import Image

# Ensure MSSearch repository root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MSSEARCH_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if MSSEARCH_ROOT not in sys.path:
    sys.path.insert(0, MSSEARCH_ROOT)

# MSSearch Imports
from mssearch.config import MSSearchConfig
from mssearch.models.modeling_sam3 import sam3_inference
from mssearch.models.modeling_siglip import SigLIPScorer
from mssearch.models.modeling_qwenvl import ModelQwenVL
from mssearch.MSSearch import get_mssearch_response


# ==============================================================================
# DATASET DISCOVERY & OPTION PARSING
# ==============================================================================

def discover_hrbench_dir(preferred_path: Optional[str]) -> Tuple[str, str]:
    """Discovers HRBench root directory and annotation file with multiple local fallbacks."""
    search_dirs = []
    if preferred_path:
        search_dirs.append(preferred_path)
    env_dir = os.environ.get("HRBENCH_DIR")
    if env_dir:
        search_dirs.append(env_dir)

    search_dirs.extend([
        "/home/ubuntu/workspace/hr/data/hr-bench_8k",
        "/home/ubuntu/workspace/hr/data/hr-bench_4k",
        "D:/Research/VisualSearch/hr-bench_8k",
        "d:/Research/VisualSearch/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_8k/mnt/sdb/llp/datasets/zoom_eye_data/zoom_eye_data/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_4k",
        "docs/experiments/multisource/data/hr-bench_8k",
        "docs/experiments/freeform/ICML26-CVSearch/data/hr-bench_8k",
    ])

    resolved_base = None
    for d in search_dirs:
        if d and os.path.exists(d):
            resolved_base = d
            break

    if resolved_base is None:
        raise FileNotFoundError(
            f"[Dataset Error] Could not find HRBench directory in candidate paths: {search_dirs}"
        )

    annotation_path = os.path.join(resolved_base, "annotation_hr-bench_8k.json")
    if not os.path.exists(annotation_path):
        annotation_path = os.path.join(resolved_base, "annotation_hr-bench_4k.json")

    if not os.path.exists(annotation_path):
        raise FileNotFoundError(
            f"[Dataset Error] Could not find annotation JSON in HRBench directory: {resolved_base}"
        )

    return resolved_base, annotation_path


def resolve_gt_letter(answer_obj: Any, cycle_idx: int = 0) -> str:
    """Extracts ground truth option letter (e.g. 'A', 'B') from HRBench annotations."""
    if isinstance(answer_obj, list):
        if 0 <= cycle_idx < len(answer_obj):
            return str(answer_obj[cycle_idx]).strip().upper()
        return str(answer_obj[0]).strip().upper() if answer_obj else ""
    return str(answer_obj).strip().upper()


def parse_options_text(options_raw: Any) -> Dict[str, str]:
    """Parses candidate options block into { 'A': text, 'B': text, ... }."""
    opt_dict: Dict[str, str] = {}
    if not options_raw:
        return opt_dict

    if isinstance(options_raw, list) and len(options_raw) > 0:
        opt_block = options_raw[0]
    else:
        opt_block = options_raw

    if isinstance(opt_block, dict):
        for k, v in opt_block.items():
            opt_dict[str(k).upper()] = str(v).strip().lower()
    elif isinstance(opt_block, str):
        for line in opt_block.splitlines():
            m = re.match(r"^[\(\[\"']*([A-Da-d])[\)\]\.\:\s]+(.*)$", line.strip())
            if m:
                opt_dict[m.group(1).upper()] = m.group(2).strip().lower()

    return opt_dict


def extract_predicted_option(pred_str: Optional[str], options_raw: Any = None) -> Optional[str]:
    """Extracts the multiple-choice option letter ('A', 'B', 'C', 'D') from model output."""
    if pred_str is None or not isinstance(pred_str, str):
        return None

    clean_pred = pred_str.strip()
    if not clean_pred or clean_pred.lower() in ("no answer found", "none", "n/a", "[error: image not found]"):
        return None

    # 1. Regex check for leading letter (e.g. 'A. 27B', '(A)', 'A: ')
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
        for letter, text in opt_dict.items():
            if text and (p_lower == text or p_lower.startswith(text) or text in p_lower):
                return letter

    return None


# ==============================================================================
# SAMPLE EXECUTION VIA MSSearch
# ==============================================================================

def execute_single_sample(
    sample_index: int,
    hrbench_data: List[Dict[str, Any]],
    hrbench_base_dir: str,
    sam3_engine: Any,
    siglip_engine: Any,
    qwen_engine: Any,
    nlp_parser: Any,
    config: MSSearchConfig,
    num_candidate_images: int = 3,
    window_size: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Constructs candidate pool C = {I_1, ..., I_N} and executes MSSearch directly.
    """
    target_sample = hrbench_data[sample_index]
    real_question = target_sample.get("question", "")
    cycle_idx = target_sample.get("cycle_category", 0)
    gt_answer = target_sample.get("answer", "")
    gt_letter = resolve_gt_letter(gt_answer, cycle_idx)
    raw_options = target_sample.get("options", {})

    print(f"\n" + "-" * 75)
    print(f"[Sample #{sample_index}] Question: '{real_question}'")
    print(f"[Sample #{sample_index}] Ground Truth: '{gt_letter}' | Candidate Images: N={num_candidate_images}")

    # 1. Load Target Image
    target_img_rel = target_sample.get("input_image", "")
    target_img_path = os.path.join(hrbench_base_dir, target_img_rel)
    if not os.path.exists(target_img_path):
        target_img_path = target_img_rel
    if not os.path.exists(target_img_path):
        raise FileNotFoundError(f"Target image not found at: {target_img_path}")
    target_canvas = Image.open(target_img_path).convert("RGB")

    # 2. Sample Distractor Images from [sample_index - window_size, sample_index + window_size]
    total_samples = len(hrbench_data)
    min_win = max(0, sample_index - window_size)
    max_win = min(total_samples - 1, sample_index + window_size)

    candidate_distractor_indices = [idx for idx in range(min_win, max_win + 1) if idx != sample_index]
    if len(candidate_distractor_indices) < (num_candidate_images - 1):
        candidate_distractor_indices = [idx for idx in range(total_samples) if idx != sample_index]

    rng = random.Random(random_seed + sample_index)
    sampled_distractor_indices = rng.sample(candidate_distractor_indices, max(0, num_candidate_images - 1))

    pool_items = [
        {"is_target": True, "sample_idx": sample_index, "image": target_canvas, "path": target_img_rel}
    ]

    for d_idx in sampled_distractor_indices:
        d_sample = hrbench_data[d_idx]
        d_rel = d_sample.get("input_image", "")
        d_path = os.path.join(hrbench_base_dir, d_rel)
        if not os.path.exists(d_path):
            d_path = d_rel
        d_img = Image.open(d_path).convert("RGB")
        pool_items.append({"is_target": False, "sample_idx": d_idx, "image": d_img, "path": d_rel})

    # Shuffle candidate images deterministically
    rng.shuffle(pool_items)
    candidate_images = [item["image"] for item in pool_items]
    target_pool_idx = next(i for i, item in enumerate(pool_items) if item["is_target"])

    # 3. Call MSSearch Pipeline Directly
    t0 = time.perf_counter()
    resp = get_mssearch_response(
        sam_model=sam3_engine,
        zoom_model=qwen_engine,
        siglip_model=siglip_engine,
        nlp_model=nlp_parser,
        annotation=target_sample,
        candidate_images=candidate_images,
        config=config,
    )
    total_latency = round(time.perf_counter() - t0, 3)

    telemetry = resp.get("telemetry", {})
    primary_answer = resp.get("primary_answer", "")
    route_selected = telemetry.get("route", "UNKNOWN")
    evidence_images = resp.get("evidence_images", [])

    # Final bounding box coordinates before Stage 5 aggregation (Requirement 2)
    final_crop_boxes_before_aggregate = resp.get("final_crop_boxes_before_aggregate", [])

    pred_letter = extract_predicted_option(primary_answer, raw_options)
    is_correct = (pred_letter == gt_letter) if pred_letter and gt_letter else False
    res_tag = "[CORRECT]" if is_correct else "[INCORRECT]"

    print(f"[Result] Predicted: '{pred_letter}' vs GT: '{gt_letter}' -> {res_tag} | Primary: '{primary_answer}' | Route: {route_selected} | Latency: {total_latency}s")

    record = {
        "sample_index": sample_index,
        "question": real_question,
        "cycle_idx": cycle_idx,
        "gt_letter": gt_letter,
        "predicted_letter": pred_letter,
        "is_correct": is_correct,
        "primary_answer": primary_answer,
        "raw_vlm_response": resp.get("raw_vlm_response", ""),
        "target_pool_idx": target_pool_idx,
        "evidence_images": evidence_images,
        "route_selected": route_selected,
        "latency_sec": total_latency,
        "timings": telemetry.get("timings_sec", {}),
        "final_crop_boxes_before_aggregate": final_crop_boxes_before_aggregate,
        "stage6_patch_boxes": resp.get("stage6_patch_boxes", []),
    }
    return record


# ==============================================================================
# MAIN ENTRY POINT & CONFIG MANAGEMENT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Source Visual Search Engine (MSSearch) Runner for HRBench"
    )

    # Dataset & Environment Arguments
    parser.add_argument(
        "--hrbench_dir",
        type=str,
        default=None,
        help="Root path to HRBench dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "results"),
        help="Directory to save JSONL telemetry logs and summaries.",
    )
    parser.add_argument(
        "--sam3_path",
        type=str,
        default=os.environ.get("SAM3_PATH", "/home/ubuntu/workspace/hr/models/sam3/sam3/sam3.pt"),
        help="Path to local SAM3 checkpoint.",
    )
    parser.add_argument(
        "--qwen_path",
        type=str,
        default=os.environ.get("QWEN_PATH", "/home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct"),
        help="Path or HuggingFace ID for Qwen2.5-VL-7B-Instruct.",
    )
    parser.add_argument(
        "--siglip_name",
        type=str,
        default=os.environ.get("SIGLIP_NAME", "google/siglip-base-patch16-384"),
        help="HuggingFace model ID or path for SigLIP.",
    )
    parser.add_argument(
        "--nlp_model",
        type=str,
        default=os.environ.get("NLP_MODEL", "/home/ubuntu/workspace/hr/models/spacy/en_core_web_sm-3.8.0"),
        help="Path or model name for spaCy NLP parser.",
    )

    # Queue & Sampling Arguments
    parser.add_argument("--sample_idx", type=int, default=0, help="Single sample index to evaluate.")
    parser.add_argument("--sample_indices", type=int, nargs="*", default=None, help="Explicit list of sample indices.")
    parser.add_argument("--all_samples", "--all", dest="all_samples", action="store_true", help="Evaluate all samples.")
    parser.add_argument("--max_samples", type=int, default=None, help="Ceiling on evaluated samples.")
    parser.add_argument("--start_idx", type=int, default=None, help="Starting sample index when slicing.")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending sample index (exclusive) when slicing.")
    parser.add_argument("--num_candidate_images", type=int, default=3, help="Candidate pool size N (1 target + N-1 distractors).")
    parser.add_argument("--window_size", type=int, default=5, help="Window for distractor sampling.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--resume", action="store_true", help="Skip already evaluated samples in results.jsonl.")
    parser.add_argument("--mode", type=str, default="e2e", help="Execution mode (default: e2e).")
    parser.add_argument("--save_visuals", nargs="*", default=None, help="Visual artifact controls (kept for backward compatibility).")


    # Configuration File & Budget Presets
    default_cfg_path = os.path.join(SCRIPT_DIR, "config_default.json")
    parser.add_argument(
        "--config_file",
        type=str,
        default=default_cfg_path if os.path.exists(default_cfg_path) else None,
        help="Path to external JSON config file specifying all MSSearch parameters.",
    )
    parser.add_argument(
        "--budget",
        type=str,
        choices=["small", "medium", "large"],
        default=None,
        help="Quick budget preset: 'small' (fast), 'medium' (balanced sweet spot), 'large' (max accuracy).",
    )

    # Logging & Display Arguments (Requirement 1: Only print final decision of each stage)
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["summary", "verbose", "quiet"],
        default="summary",
        help="Logging level: 'summary' (only print final decision of each stage), 'verbose' (full internal loops), 'quiet' (minimal stdout).",
    )

    # Key Pipeline Hyperparameter & Budget Overrides
    parser.add_argument("--max_patches_per_image", type=int, default=None, help="Stage 3 budget: max patches per image.")
    parser.add_argument("--max_partition_depth", type=int, default=None, help="Stage 3 budget: max recursion depth.")
    parser.add_argument("--max_sam3_eval_patches", type=int, default=None, help="Stage 4 budget: max SAM3 patch scans.")
    parser.add_argument("--max_latent_evals", type=int, default=None, help="Stage 4 budget: max speculative VLM latent checks.")
    parser.add_argument("--max_patches_per_detected_group", type=int, default=None, help="Stage 4 budget: patches kept per detected group.")
    parser.add_argument("--max_patches_per_latent_group", type=int, default=None, help="Stage 4 budget: patches kept per latent group.")
    parser.add_argument("--max_retained_patches", type=int, default=None, help="Stage 4 budget: global hard cap on candidate pool.")
    parser.add_argument("--early_consensus_count", type=int, default=None, help="Stage 6 budget: consensus count before exit.")
    parser.add_argument("--enable_direct_answer", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=None)
    parser.add_argument("--direct_answer_conf_threshold", type=float, default=None)
    parser.add_argument("--enable_fast_path", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=None)
    parser.add_argument("--fast_path_conf_threshold", type=float, default=None)
    parser.add_argument("--partition_mode", type=str, default=None)
    parser.add_argument("--grouping_sim_threshold", type=float, default=None)
    parser.add_argument("--sam3_group_det_threshold", type=float, default=None)
    parser.add_argument("--enable_latent_gating", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=None)
    parser.add_argument("--latent_confidence_threshold", type=float, default=None)
    parser.add_argument("--enable_stage5_spatial_nms", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=None)
    parser.add_argument("--stage5_nms_iou_threshold", type=float, default=None)
    parser.add_argument("--enable_spatial_compressed_canvas", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=None)
    parser.add_argument("--enable_early_consensus_exit", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=None)

    args, unknown = parser.parse_known_args()
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Base Configuration
    if args.config_file and os.path.exists(args.config_file):
        print(f"[Config] Ingesting configuration from JSON: {args.config_file}")
        config = MSSearchConfig.load_json(args.config_file)
    else:
        print("[Config] Using default MSSearchConfig dataclass parameters.")
        config = MSSearchConfig()

    # 1b. Ingest Budget Preset if Specified (--budget small|medium|large)
    if args.budget:
        budget_path = os.path.join(SCRIPT_DIR, f"budget_{args.budget}.json")
        if os.path.exists(budget_path):
            print(f"[Budget Preset] Ingesting budget preset '{args.budget}' from: {budget_path}")
            with open(budget_path, "r", encoding="utf-8") as bf:
                b_dict = json.load(bf)
            for k, v in b_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)
                    print(f"  - {k} -> {v}")
        else:
            print(f"[Warning] Budget preset file not found: {budget_path}")

    # 2. Apply Logging Mode (Requirement 1)
    config.log_level = args.log_level
    if args.log_level == "summary":
        config.stage_summary_only = True
        config.verbose = True
    elif args.log_level == "verbose":
        config.stage_summary_only = False
        config.verbose = True
    elif args.log_level == "quiet":
        config.stage_summary_only = False
        config.verbose = False

    # 3. Apply CLI Overrides
    cli_overrides = [
        "max_patches_per_image",
        "max_partition_depth",
        "max_sam3_eval_patches",
        "max_latent_evals",
        "max_patches_per_detected_group",
        "max_patches_per_latent_group",
        "max_retained_patches",
        "early_consensus_count",
        "enable_direct_answer",
        "direct_answer_conf_threshold",
        "enable_fast_path",
        "fast_path_conf_threshold",
        "partition_mode",
        "grouping_sim_threshold",
        "sam3_group_det_threshold",
        "enable_latent_gating",
        "latent_confidence_threshold",
        "enable_stage5_spatial_nms",
        "stage5_nms_iou_threshold",
        "enable_spatial_compressed_canvas",
        "enable_early_consensus_exit",
    ]

    for param in cli_overrides:
        val = getattr(args, param, None)
        if val is not None:
            setattr(config, param, val)
            print(f"[Config Override] {param} -> {val}")

    # 4. Save config_used.json (Requirement 3: Write all configs out so user can change it)
    config_out_path = os.path.join(args.output_dir, "config_used.json")
    config.save_json(config_out_path)
    print(f"[Config] Active configuration written to: {config_out_path}")

    # 5. Discover HRBench Dataset
    hrbench_base_dir, hrbench_annotation_path = discover_hrbench_dir(args.hrbench_dir)
    print(f"[Dataset] HRBench root: {hrbench_base_dir}")
    print(f"[Dataset] Loading annotations: {hrbench_annotation_path}")

    with open(hrbench_annotation_path, "r", encoding="utf-8") as f:
        hrbench_data = json.load(f)
    print(f"[Dataset] Ingested {len(hrbench_data)} total samples.")

    # 6. Resolve Target Queue
    if args.all_samples:
        target_indices = list(range(len(hrbench_data)))
    elif args.sample_indices is not None and len(args.sample_indices) > 0:
        target_indices = args.sample_indices
    else:
        target_indices = [args.sample_idx]

    if args.start_idx is not None or args.end_idx is not None:
        start_i = args.start_idx or 0
        end_i = args.end_idx if args.end_idx is not None else len(target_indices)
        target_indices = target_indices[start_i:end_i]

    if args.max_samples is not None and args.max_samples > 0:
        target_indices = target_indices[: args.max_samples]

    results_log_path = os.path.abspath(os.path.join(args.output_dir, "results.jsonl"))
    if args.resume and os.path.exists(results_log_path):
        completed_ids = set()
        with open(results_log_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if line.strip():
                    try:
                        r_data = json.loads(line)
                        if "sample_index" in r_data:
                            completed_ids.add(r_data["sample_index"])
                    except Exception:
                        pass
        orig_count = len(target_indices)
        target_indices = [idx for idx in target_indices if idx not in completed_ids]
        print(f"[Resume] Found {len(completed_ids)} completed samples. Remaining: {len(target_indices)} / {orig_count}")

    print(f"[Queue] Ready to evaluate {len(target_indices)} sample(s).")

    # 7. Model Allocation & Initialization
    num_gpus = torch.cuda.device_count()
    sam3_device = "cuda:1" if num_gpus > 1 else ("cuda:0" if num_gpus > 0 else "cpu")
    siglip_device = "cuda:0" if num_gpus > 0 else "cpu"
    qwen_device = "cuda:0" if num_gpus > 0 else "cpu"

    print("=" * 80)
    print(f"HARDWARE TOPOLOGY: {num_gpus} CUDA GPUs Available")
    print(f"  - SAM3 Engine      : {sam3_device}")
    print(f"  - SigLIP Engine    : {siglip_device}")
    print(f"  - Qwen2.5-VL Engine: {qwen_device}")
    print("=" * 80)

    print(f"[Loading] SAM3 from '{args.sam3_path}' on {sam3_device}...")
    sam3_engine = sam3_inference(model_path=args.sam3_path, device=sam3_device)

    print(f"[Loading] SigLIP '{args.siglip_name}' on {siglip_device}...")
    siglip_engine = SigLIPScorer(
        model_name=args.siglip_name,
        device=siglip_device,
        torch_dtype=torch.float16 if "cuda" in siglip_device else torch.float32,
    )

    print(f"[Loading] Qwen2.5-VL from '{args.qwen_path}' on {qwen_device}...")
    qwen_engine = ModelQwenVL(
        model_path=args.qwen_path,
        device=qwen_device,
        torch_dtype=torch.float16 if "cuda" in qwen_device else torch.float32,
        patch_scale=1.2,
    )

    nlp_parser = None
    if args.nlp_model:
        try:
            import spacy
            nlp_parser = spacy.load(args.nlp_model) if os.path.exists(args.nlp_model) else spacy.load("en_core_web_sm")
            print(f"[Loaded] spaCy NLP Grammar Engine ({args.nlp_model})")
        except Exception as e:
            print(f"[Notice] spaCy load skipped: {e}")

    # 8. Evaluation Loop
    all_records = []
    correct_count = 0
    evaluated_count = 0
    route_counter = {}
    latencies = []

    for queue_pos, s_idx in enumerate(target_indices, start=1):
        if s_idx < 0 or s_idx >= len(hrbench_data):
            continue

        try:
            rec = execute_single_sample(
                sample_index=s_idx,
                hrbench_data=hrbench_data,
                hrbench_base_dir=hrbench_base_dir,
                sam3_engine=sam3_engine,
                siglip_engine=siglip_engine,
                qwen_engine=qwen_engine,
                nlp_parser=nlp_parser,
                config=config,
                num_candidate_images=args.num_candidate_images,
                window_size=args.window_size,
                random_seed=args.random_seed,
            )
            all_records.append(rec)
            evaluated_count += 1
            if rec.get("is_correct"):
                correct_count += 1
            r_sel = rec.get("route_selected", "UNKNOWN")
            route_counter[r_sel] = route_counter.get(r_sel, 0) + 1
            latencies.append(rec.get("latency_sec", 0.0))

            with open(results_log_path, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(rec) + "\n")

            acc_pct = (correct_count / evaluated_count * 100.0) if evaluated_count > 0 else 0.0
            avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
            print(
                f">>> PROGRESS: {queue_pos}/{len(target_indices)} (Sample #{s_idx}) | "
                f"Acc: {correct_count}/{evaluated_count} ({acc_pct:.1f}%) | "
                f"Avg Latency: {avg_lat:.2f}s <<<\n"
            )

        except Exception as e:
            print(f"[ERROR] Exception on sample #{s_idx}: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_eval = len(all_records)
    overall_acc = (correct_count / total_eval * 100.0) if total_eval > 0 else 0.0
    avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0

    summary_data = {
        "total_samples_evaluated": total_eval,
        "correct_samples": correct_count,
        "accuracy_acc1": round(overall_acc, 2),
        "avg_latency_sec": round(avg_lat, 2),
        "routes": route_counter,
        "results_log_path": results_log_path,
        "config_used_path": config_out_path,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary_data, f_sum, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED: Evaluated {total_eval} / {len(target_indices)} samples.")
    print(f"Overall Accuracy (Acc@1): {correct_count}/{total_eval} ({overall_acc:.2f}%)")
    print(f"Average Latency         : {avg_lat:.2f}s per sample")
    print(f"Route Breakdown         : {route_counter}")
    print(f"Telemetry log written to: {results_log_path}")
    print(f"Summary JSON written to : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
