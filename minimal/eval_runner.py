"""
Batch Benchmark Evaluation Runner for STASH-Search Minimal.
Evaluates STASH-Bench (N in {1, 5, 10, 20}), V*Bench (Vstar), HR-Bench 4K/8K,
TreeBench, and MME-RealWorld-Lite.
Outputs streaming JSONL predictions and computes online summary metrics.
"""

import os
import sys
import gc
import json
import time
import re
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image, ImageOps

# Ensure repository root is in python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from minimal.config import STASHConfig
from minimal.engine import STASHSearchEngine
from minimal.datasets import load_benchmark_dataset, BenchmarkSample


def _expand_bash_val(val: str, env_map: Dict[str, str]) -> str:
    """Expands bash-style parameter substitutions like ${VAR:-default} and ${VAR}, supporting nesting."""
    s = str(val).strip()
    pattern = re.compile(r"\$\{([^{}]+)\}")
    # Expand innermost ${...} iteratively up to 10 passes
    for _ in range(10):
        if not pattern.search(s):
            break

        def repl(match):
            content = match.group(1).strip()
            if ":-" in content:
                k, default_val = content.split(":-", 1)
                k = k.strip()
                default_val = default_val.strip()
                return env_map.get(k) or os.environ.get(k) or default_val
            return env_map.get(content) or os.environ.get(content) or ""

        s = pattern.sub(repl, s)
    return s


def load_env_defaults():
    """Loads environment configuration if present in environment or env_paths.sh."""
    script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
    env_file = os.path.join(script_dir, "env_paths.sh")
    if os.path.isfile(env_file):
        parsed_env: Dict[str, str] = {}
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:].strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    expanded_v = _expand_bash_val(v, parsed_env)
                    parsed_env[k] = expanded_v
                    if k not in os.environ and expanded_v:
                        os.environ[k] = expanded_v


def extract_predicted_option(pred_str: Any, options_str: Optional[str] = None) -> Optional[str]:
    """Extracts chosen option letter (A through K) from free-form model prediction."""
    if not pred_str or not isinstance(pred_str, str):
        return None
    clean = pred_str.strip()
    if clean.lower() in ("no answer found", "none", "n/a", "[error: image not found]"):
        if options_str and clean.lower() in ("none", "n/a"):
            for line in options_str.splitlines():
                line = line.strip()
                opt_m = re.match(r"^[\(\[]?([A-Ka-k])[\)\]\.\:\s]+(.*)", line)
                if opt_m:
                    opt_letter = opt_m.group(1).upper()
                    opt_content = opt_m.group(2).strip().lower()
                    if clean.lower() == opt_content or opt_content.startswith(clean.lower()) or "none" in opt_content:
                        return opt_letter
        return None

    # Strip wrapping markdown bolding or backticks e.g. **A** -> A, `B` -> B
    clean = re.sub(r"^\*+([A-Ka-k])\*+$", r"\1", clean).strip()
    clean = re.sub(r"^`+([A-Ka-k])`+$", r"\1", clean).strip()

    # 1. Exact single letter (with optional period/bracket/quotes) e.g. "A", "B.", "(C)", "[D]"
    m = re.match(r"^[\s\*\(\[\"']*([A-Ka-k])[\s\*\)\]\.\:\'\"]*$", clean)
    if m:
        return m.group(1).upper()

    # 2. Explicit answer phrases anywhere in text (e.g. 'The best answer is: B', 'Answer is **C**',
    #    'I believe the answer is D', 'The correct choice is (A)', 'Final answer: B')
    m = re.search(
        r"\b(?:final answer|best answer|correct answer|answer|choice|option)\b"
        r"(?:\s+(?:is|to|be|would be|must be|of|should be|selected is))?"
        r"[\s\:\*\(\[\"\']*([A-Ka-k])\b",
        clean,
        re.IGNORECASE
    )
    if m:
        return m.group(1).upper()

    # 3. Direct letter at start formatted as option label e.g. "A. 27B", "(A) Blue", "A: ", "B - text", "B\nBecause..."
    m = re.match(r"^[\s\*\(\[\"']*([A-Ka-k])(?:[\)\]\.\:\-]|(?:\s*\n))\s*", clean)
    if m:
        return m.group(1).upper()

    # 4. Trailing answer at end of text e.g. "... is therefore: D." or "... is B."
    m = re.search(r"\b(?:is|as)\s+[\*\(\[\"\']*([A-Ka-k])[\*\)\]\.\:\'\"]*$", clean, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 5. Standalone letters B-K (skips 'I' and 'A' to prevent false matches with English pronoun 'I' or article 'A')
    matches = re.findall(r"\b([B-HJ-Kb-hj-k])\b", clean)
    if matches:
        return matches[-1].upper()

    # 6. Content matching against options_str if available (in case model repeated option text)
    if options_str:
        for line in options_str.splitlines():
            line = line.strip()
            opt_m = re.match(r"^[\(\[]?([A-Ka-k])[\)\]\.\:\s]+(.*)", line)
            if opt_m:
                opt_letter = opt_m.group(1).upper()
                opt_content = opt_m.group(2).strip().lower()
                if opt_content and len(opt_content) > 3 and opt_content in clean.lower():
                    return opt_letter

    # 7. Fallback for isolated 'A' if preceded by option/letter indicators
    m = re.search(r"(?:letter|choice|option|selection)\s*[\:\=]?\s*[\*\(\[\"\']*(A)\b", clean, re.IGNORECASE)
    if m:
        return "A"

    return None


def normalize_text(text: str) -> str:
    """Normalizes string for open-ended answer comparison."""
    t = re.sub(r'[^\w\s]', '', str(text).lower()).strip()
    t = re.sub(r'^(?:a|an|the)\s+', '', t)
    return t


def parse_args():
    load_env_defaults()
    parser = argparse.ArgumentParser(description="STASH-Search Minimal Evaluation Runner")

    # Benchmark selection: strict benchmark names only, no aliases
    parser.add_argument("--benchmark", type=str, default="stash",
                        choices=["stash", "vstar", "hrbench4k", "hrbench8k", "treebench", "mme-realworld-lite"],
                        help="Benchmark dataset to evaluate: 'stash', 'vstar', 'hrbench4k', 'hrbench8k', 'treebench', 'mme-realworld-lite'.")

    # Paths
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Root directory for Vstar, HR-Bench, TreeBench, or MME-RealWorld dataset.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Explicit annotation path (or STASH-Bench annotations JSON).")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Base image directory (for STASH-Bench pool).")
    parser.add_argument("--gallery_order_path", type=str, default=None,
                        help="Path to gallery_order_seed42.json (for STASH-Bench).")
    parser.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_BASE_DIR", "./results"),
                        help="Output directory to save JSONL logs and metrics summary.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Explicit path to output JSONL file.")

    # Model & Engine
    parser.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"),
                        help="Path or HuggingFace ID of MLLM backbone.")
    parser.add_argument("--dual_encoder_name", type=str, default=os.environ.get("SIGLIP_NAME", "google/siglip2-so400m-patch16-512"),
                        help="Path or HuggingFace ID of SigLIP dual-encoder.")
    parser.add_argument("--nlp_model", type=str, default="en_core_web_sm",
                        help="spaCy NLP model name or path.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Inference device (e.g. 'cuda:0', 'cpu').")

    # Pool mode (for STASH-Bench)
    parser.add_argument("--pool_mode", type=str, default="N_5",
                        choices=["N_1", "N_5", "N_10", "N_20", "crop"],
                        help="Candidate gallery pool size for STASH-Bench.")
    parser.add_argument("--cycle_idx", type=int, default=None,
                        help="Option rotation index for HR-Bench (0-3). If None, evaluates all 4 rotations.")

    # Hyperparameters
    parser.add_argument("--k", type=int, default=20,
                        help="Base candidate crop retention budget per image (default: 20).")
    parser.add_argument("--saliency_gain_thresh", type=float, default=0.04,
                        help="Saliency plateau gain threshold (default: 0.04).")
    parser.add_argument("--consensus_sim", type=float, default=0.85,
                        help="Consensus string deduplication threshold (default: 0.85).")

    # Sample slicing & resumption
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate.")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Starting sample index for dataset slicing.")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Ending sample index for dataset slicing.")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Single sample index to evaluate.")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from existing output JSONL file.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Explicit JSONL file path to resume from.")
    parser.add_argument("--quiet", action="store_true",
                        help="Silence stage-level debug logs.")

    return parser.parse_args()


def resolve_dataset_args(args) -> Dict[str, Any]:
    """Resolves dataset paths using arguments or environment defaults strictly without aliases."""
    b = args.benchmark.lower().strip()
    if b == "stash":
        d_path = args.dataset_path or args.dataset_dir or os.environ.get("STASH_DATASET_PATH") or os.environ.get("DATASET_PATH", "")
        img_dir = args.image_dir or os.environ.get("STASH_IMAGE_DIR") or os.environ.get("IMAGE_DIR", "")
        return {
            "dataset_dir_or_path": d_path,
            "image_dir": img_dir,
            "pool_mode": args.pool_mode,
            "gallery_order_path": args.gallery_order_path
        }
    elif b == "vstar":
        a_path = args.dataset_path or os.environ.get("VSTAR_ANNOTATION")
        v_dir = args.dataset_dir or os.environ.get("VSTAR_DIR", "")
        if not v_dir and a_path and os.path.isfile(a_path):
            v_dir = os.path.dirname(a_path)
        return {
            "dataset_dir_or_path": v_dir or a_path or "",
            "annotation_path": a_path
        }
    elif b == "hrbench4k":
        h_dir = args.dataset_dir or os.environ.get("HRBENCH_4K_DIR", "")
        a_path = args.dataset_path or os.environ.get("HRBENCH_4K_ANNOTATION")
        if not h_dir and a_path and os.path.isfile(a_path):
            h_dir = os.path.dirname(a_path)
        return {
            "dataset_dir_or_path": h_dir or a_path or "",
            "annotation_path": a_path,
            "cycle_idx": args.cycle_idx
        }
    elif b == "hrbench8k":
        h_dir = args.dataset_dir or os.environ.get("HRBENCH_8K_DIR", "")
        a_path = args.dataset_path or os.environ.get("HRBENCH_8K_ANNOTATION")
        if not h_dir and a_path and os.path.isfile(a_path):
            h_dir = os.path.dirname(a_path)
        return {
            "dataset_dir_or_path": h_dir or a_path or "",
            "annotation_path": a_path,
            "cycle_idx": args.cycle_idx
        }
    elif b == "treebench":
        a_path = args.dataset_path or os.environ.get("TREEBENCH_ANNOTATION")
        t_dir = args.dataset_dir or os.environ.get("TREEBENCH_DIR", "")
        if not t_dir and a_path and os.path.isfile(a_path):
            t_dir = os.path.dirname(a_path)
        return {
            "dataset_dir_or_path": t_dir or a_path or "",
            "annotation_path": a_path
        }
    elif b == "mme-realworld-lite":
        a_path = args.dataset_path or os.environ.get("MME_REALWORLD_ANNOTATION")
        m_dir = args.dataset_dir or os.environ.get("MME_REALWORLD_DIR", "")
        if not m_dir and a_path and os.path.isfile(a_path):
            m_dir = os.path.dirname(a_path)
        return {
            "dataset_dir_or_path": m_dir or a_path or "",
            "annotation_path": a_path
        }
    else:
        raise ValueError(
            f"Unknown benchmark '{args.benchmark}'. Supported: 'stash', 'vstar', 'hrbench4k', 'hrbench8k', 'treebench', 'mme-realworld-lite'."
        )


def main():
    args = parse_args()
    b = args.benchmark.lower().strip()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Resolve dataset inputs
    ds_kwargs = resolve_dataset_args(args)
    primary_path = ds_kwargs.get("dataset_dir_or_path") or ds_kwargs.get("annotation_path") or ""
    if not primary_path:
        print(f"[Error] Dataset path for benchmark '{b}' is not configured.")
        print(f"Please provide --dataset_dir or --dataset_path, or configure env_paths.sh.")
        sys.exit(1)

    print(f"\n==========================================================================")
    print(f"STASH-Search Minimal Runner | Benchmark: {b.upper()}")
    print(f"Model      : {args.model_path}")
    print(f"Dataset    : {primary_path}")
    if b == "stash":
        print(f"Pool Mode  : {args.pool_mode}")
    print(f"Output Dir : {args.output_dir}")
    print(f"==========================================================================")

    # 2. Load benchmark dataset
    print(f"[DataLoader] Loading {b.upper()} benchmark...")
    try:
        samples = load_benchmark_dataset(
            benchmark=b,
            **ds_kwargs
        )
    except Exception as e:
        print(f"[Fatal DataLoader Error]: {e}")
        sys.exit(1)

    print(f"[DataLoader] Successfully loaded {len(samples)} total benchmark instances.")

    # 3. Apply sample slicing
    if args.sample_idx is not None:
        samples = [samples[args.sample_idx]] if args.sample_idx < len(samples) else []
    else:
        start = args.start_idx if args.start_idx is not None else 0
        end = args.end_idx if args.end_idx is not None else len(samples)
        samples = samples[start:end]
        if args.max_samples is not None:
            samples = samples[:args.max_samples]

    print(f"[DataLoader] Active evaluation slice: {len(samples)} samples.")
    if not samples:
        print("[Notice] No samples to evaluate. Exiting.")
        return

    # 4. Resolve output file path & handle resumption
    m_base = os.path.basename(os.path.normpath(args.model_path))
    if args.output_file:
        out_jsonl = args.output_file
    elif args.resume_from:
        out_jsonl = args.resume_from
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if b == "stash":
            out_jsonl = os.path.join(args.output_dir, f"results_{b}_{m_base}_{args.pool_mode}_{stamp}.jsonl")
        else:
            out_jsonl = os.path.join(args.output_dir, f"results_{b}_{m_base}_{stamp}.jsonl")

    metrics_file = out_jsonl.replace(".jsonl", "_metrics.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)

    stats: Dict[str, Any] = {
        "total": 0,
        "correct": 0,
        "correct_img": 0,
        "correct_overall": 0,
        "by_category": {},
        "by_rotation": {}
    }

    completed_ids = set()
    if (args.resume or args.resume_from) and os.path.isfile(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as rf:
            for line in rf:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if "sample_id" in rec:
                            s_id = str(rec["sample_id"])
                            completed_ids.add(s_id)
                            # Re-populate historical cumulative stats
                            cat = rec.get("category") or "default"
                            if cat not in stats["by_category"]:
                                stats["by_category"][cat] = {"total": 0, "correct": 0, "correct_img": 0, "correct_overall": 0}
                            stats["total"] += 1
                            stats["by_category"][cat]["total"] += 1
                            if rec.get("is_correct_answer"):
                                stats["correct"] += 1
                                stats["by_category"][cat]["correct"] += 1
                            if rec.get("is_correct_image"):
                                stats["correct_img"] += 1
                                stats["by_category"][cat]["correct_img"] += 1
                            if rec.get("is_correct_overall"):
                                stats["correct_overall"] += 1
                                stats["by_category"][cat]["correct_overall"] += 1

                            # Re-populate rotation stats if present
                            if "cycle_idx" in rec and rec["cycle_idx"] is not None:
                                rot_k = f"rot{rec['cycle_idx']}"
                                if rot_k not in stats["by_rotation"]:
                                    stats["by_rotation"][rot_k] = {"total": 0, "correct": 0}
                                stats["by_rotation"][rot_k]["total"] += 1
                                if rec.get("is_correct_answer"):
                                    stats["by_rotation"][rot_k]["correct"] += 1
                    except Exception:
                        pass
        print(f"[Resume] Found {len(completed_ids)} completed records in '{out_jsonl}'. Restored baseline statistics.")

    # 5. Initialize search engine
    config = STASHConfig(
        k=args.k,
        saliency_plateau_gain_threshold=args.saliency_gain_thresh,
        answer_dedup_similarity=args.consensus_sim,
        allow_none_answer=(b == "mme-realworld-lite"),
        verbose=not args.quiet
    )

    print(f"\n[Engine Setup] Initializing STASHSearchEngine...")
    engine = STASHSearchEngine.from_pretrained(
        mllm_name_or_path=args.model_path,
        dual_encoder_name=args.dual_encoder_name,
        config=config,
        device=args.device,
        nlp_model_path=args.nlp_model
    )

    # 6. Run evaluation loop
    print(f"\n[Execution] Starting inference across {len(samples)} samples -> '{out_jsonl}'...\n")

    with open(out_jsonl, "a", encoding="utf-8") as out_f:
        for idx, sample in enumerate(samples):
            if str(sample.sample_id) in completed_ids:
                continue

            # Load candidate PIL images with EXIF orientation correction
            pil_images: List[Image.Image] = []
            for img_path in sample.candidate_image_paths:
                if os.path.isfile(img_path):
                    try:
                        with Image.open(img_path) as img:
                            pil_images.append(ImageOps.exif_transpose(img).convert("RGB"))
                    except Exception as exif_err:
                        print(f"[Warning] Failed to open '{img_path}' with EXIF transpose ({exif_err}). "
                              f"Falling back to standard image open without EXIF transpose.")
                        try:
                            with Image.open(img_path) as img:
                                pil_images.append(img.convert("RGB"))
                        except Exception as read_err:
                            print(f"[Error] Failed to load image '{img_path}': {read_err}")
                else:
                    print(f"[Warning] Image file not found: '{img_path}'")

            if not pil_images:
                print(f"[Error] No valid images loaded for sample {sample.sample_id}. Skipping.")
                continue

            t0 = time.perf_counter()
            res = engine.search(
                candidate_images=pil_images,
                question=sample.question,
                options_str=sample.options_str,
                inspection_prompt=sample.inspection_prompt
            )
            lat = round(time.perf_counter() - t0, 3)

            pred_ans = res["primary_answer"]
            pred_img = res["primary_evidence_image"]

            # Evaluation logic
            is_correct_ans = False
            is_correct_img = False

            if b == "stash":
                gt_norm = normalize_text(sample.gt_answer)
                pred_norm = normalize_text(pred_ans)
                is_correct_ans = (gt_norm == pred_norm) or (gt_norm in pred_norm)
                is_correct_img = (pred_img == sample.gt_image_idx) if sample.gt_image_idx is not None else False
                is_correct_overall = is_correct_ans and is_correct_img
                pred_letter = None
            else:
                # Multiple-choice benchmark (vstar, hrbench4k, hrbench8k, treebench, mme-realworld-lite)
                pred_letter = extract_predicted_option(pred_ans, sample.options_str)
                m_gt = re.search(r"([A-Ka-k])", str(sample.gt_answer))
                gt_letter = m_gt.group(1).upper() if m_gt else str(sample.gt_answer).strip().upper()
                is_correct_ans = (pred_letter == gt_letter) if pred_letter else False
                is_correct_img = True
                is_correct_overall = is_correct_ans

            # Accumulate statistics
            cat = sample.category or "default"
            if cat not in stats["by_category"]:
                stats["by_category"][cat] = {"total": 0, "correct": 0, "correct_img": 0, "correct_overall": 0}

            stats["total"] += 1
            stats["by_category"][cat]["total"] += 1

            if is_correct_ans:
                stats["correct"] += 1
                stats["by_category"][cat]["correct"] += 1
            if is_correct_img:
                stats["correct_img"] += 1
                stats["by_category"][cat]["correct_img"] += 1
            if is_correct_overall:
                stats["correct_overall"] += 1
                stats["by_category"][cat]["correct_overall"] += 1

            # Accumulate rotation stats if sample belongs to a rotation benchmark
            rot = sample.metadata.get("cycle_idx") if sample.metadata else None
            if rot is not None:
                rot_k = f"rot{rot}"
                if rot_k not in stats["by_rotation"]:
                    stats["by_rotation"][rot_k] = {"total": 0, "correct": 0}
                stats["by_rotation"][rot_k]["total"] += 1
                if is_correct_ans:
                    stats["by_rotation"][rot_k]["correct"] += 1

            # Log sample to JSONL
            record = {
                "sample_id": sample.sample_id,
                "benchmark": b,
                "cycle_idx": rot,
                "question": sample.question,
                "inspection_prompt": sample.inspection_prompt,
                "options_str": sample.options_str,
                "gt_answer": sample.gt_answer,
                "gt_letter": gt_letter if b != "stash" else None,
                "gt_image_idx": sample.gt_image_idx,
                "predicted_answer": pred_ans,
                "predicted_letter": pred_letter,
                "predicted_image_index": pred_img,
                "is_correct_answer": is_correct_ans,
                "is_correct_image": is_correct_img,
                "is_correct_overall": is_correct_overall,
                "category": cat,
                "latency_sec": lat,
                "entropy_pruned_leaves": res.get("entropy_pruned_leaves", 0),
                "dedup_affected_crops": res.get("dedup_affected_crops", {}),
                "telemetry": res.get("telemetry", {}),
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            # Live CLI progress log
            status_tag = "PASS" if is_correct_overall else "FAIL"
            dedup_info = res.get("dedup_affected_crops", {})
            total_dedup = dedup_info.get("total_affected", 0) if isinstance(dedup_info, dict) else 0
            entropy_pruned = res.get("entropy_pruned_leaves", 0)
            if b == "stash":
                print(f"[{idx + 1}/{len(samples)}] [{status_tag}] Sample {sample.sample_id} | "
                      f"Pred: '{pred_ans}' (GT: '{sample.gt_answer}') | Img: {pred_img} (GT: {sample.gt_image_idx}) | "
                      f"Dedup (a): {total_dedup} | Pruned (b): {entropy_pruned} | {lat}s")
            else:
                rot_tag = f" [rot{rot}]" if rot is not None else ""
                print(f"[{idx + 1}/{len(samples)}] [{status_tag}] Sample {sample.sample_id}{rot_tag} | "
                      f"Pred: '{pred_letter}' ('{pred_ans}') (GT: '{gt_letter}') | "
                      f"Dedup (a): {total_dedup} | Pruned (b): {entropy_pruned} | {lat}s")

            # Periodic memory management
            if (idx + 1) % 10 == 0:
                gc.collect()

    # 7. Compute & Print Final Summary Metrics
    total = max(1, stats["total"])
    acc_text = round(stats["correct"] / total * 100, 2)
    acc_img = round(stats["correct_img"] / total * 100, 2)
    acc_overall = round(stats["correct_overall"] / total * 100, 2)

    avg_rot_acc = None
    if stats["by_rotation"]:
        rot_accs = [
            r_d["correct"] / max(1, r_d["total"]) * 100
            for r_d in stats["by_rotation"].values()
        ]
        avg_rot_acc = round(sum(rot_accs) / len(rot_accs), 2)

    print("\n" + "=" * 70)
    print(f"BENCHMARK EVALUATION SUMMARY: {b.upper()}")
    print("=" * 70)
    print(f"Total Evaluated Samples : {stats['total']}")
    if b == "stash":
        print(f"Overall Accuracy (Joint): {acc_overall:.2f}%")
        print(f"Text Accuracy           : {acc_text:.2f}%")
        print(f"Image Identification    : {acc_img:.2f}%")
    else:
        print(f"Total Option Accuracy   : {acc_overall:.2f}%")
        if avg_rot_acc is not None:
            print(f"Accuracy (Rotation)     : {avg_rot_acc:.2f}%")

    if stats["by_rotation"]:
        print("-" * 70)
        print("Rotation Breakdown:")
        for r_k, r_d in sorted(stats["by_rotation"].items()):
            r_tot = max(1, r_d["total"])
            r_acc = round(r_d["correct"] / r_tot * 100, 2)
            r_d["accuracy"] = r_acc
            print(f"  • Rotation {r_k:<6}: {r_acc:6.2f}% ({r_d['correct']}/{r_d['total']})")

    print("-" * 70)
    print("Category Breakdown:")
    for cat_name, c_data in sorted(stats["by_category"].items()):
        c_tot = max(1, c_data["total"])
        c_acc = round(c_data["correct_overall"] / c_tot * 100, 2)
        c_data["accuracy"] = c_acc
        print(f"  • {cat_name:<36}: {c_acc:6.2f}% ({c_data['correct_overall']}/{c_data['total']})")
    print("=" * 70)

    # Save metrics JSON
    summary_metrics = {
        "benchmark": b,
        "model_path": args.model_path,
        "pool_mode": args.pool_mode if b == "stash" else "N_1",
        "total_samples": stats["total"],
        "acc_overall": acc_overall,
        "acc_text": acc_text,
        "acc_img": acc_img,
        "by_category": stats["by_category"]
    }
    if avg_rot_acc is not None:
        summary_metrics["accuracy (rotation)"] = avg_rot_acc
        summary_metrics["by_rotation"] = stats["by_rotation"]

    with open(metrics_file, "w", encoding="utf-8") as mf:
        json.dump(summary_metrics, mf, indent=2)
    print(f"[Done] Metrics saved to '{metrics_file}'.")


if __name__ == "__main__":
    main()
