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
import warnings
import logging
import contextlib

# Enforce strict offline execution (local files only, no online downloads)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Set, Union

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None

from PIL import Image

# Suppress harmless upstream deprecation warnings and checkpoint config token ID warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*min_pixels.*max_pixels.*")
warnings.filterwarnings("ignore", message=r".*bos_token_id.*")
warnings.filterwarnings("ignore", message=r".*eos_token_id.*")
warnings.filterwarnings("ignore", message=r".*`torch_dtype` is deprecated.*")
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

# Ensure MSSearch repository root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MSSEARCH_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if MSSEARCH_ROOT not in sys.path:
    sys.path.insert(0, MSSEARCH_ROOT)

# MSSearch Imports
from mssearch.config import MSSearchConfig
try:
    from mssearch.models.modeling_sam3 import sam3_inference
except (ImportError, ModuleNotFoundError):
    sam3_inference = None

try:
    from mssearch.models.modeling_siglip import SigLIPScorer
except (ImportError, ModuleNotFoundError):
    SigLIPScorer = None

try:
    from mssearch.models.modeling_qwenvl import ModelQwenVL
except (ImportError, ModuleNotFoundError):
    ModelQwenVL = None

try:
    from mssearch.MSSearch import get_mssearch_response
except (ImportError, ModuleNotFoundError):
    get_mssearch_response = None

try:
    from mssearch.stages.stage1_parser import format_options, format_question_with_options
except (ImportError, ModuleNotFoundError):
    def format_options(options: Any, cycle_idx: int = 0) -> str:
        """Fallback option block formatting when stages module is unavailable."""
        if not options:
            return ""
        if isinstance(options, dict):
            return "\n".join(f"{k}. {v}" if not str(v).startswith(f"{k}.") else str(v) for k, v in options.items())
        if isinstance(options, list):
            if not options:
                return ""
            if all(isinstance(opt, str) for opt in options):
                if any("\n" in opt for opt in options):
                    chosen_block = options[cycle_idx] if 0 <= cycle_idx < len(options) else options[0]
                    return chosen_block.strip()
                return "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
            chosen_block = options[cycle_idx] if 0 <= cycle_idx < len(options) else options[0]
            if isinstance(chosen_block, dict):
                return "\n".join(f"{k}. {v}" for k, v in chosen_block.items())
            return str(chosen_block).strip()
        return str(options).strip()

    def format_question_with_options(question: str, options: Any = None, cycle_idx: int = 0) -> str:
        """Fallback question + option block formatting when stages module is unavailable."""
        if not options:
            return question
        opt_str = format_options(options, cycle_idx=cycle_idx)
        if not opt_str:
            return question
        return (
            f"{question}\n\n"
            f"Options:\n{opt_str}\n\n"
            f"Select the best answer to the above multiple-choice question based on the image. "
            f"Respond with the option letter and concise answer text."
        )


# ==============================================================================
# DATASET DISCOVERY & OPTION PARSING
# ==============================================================================

def resolve_image_path(base_dir: str, rel_path: str) -> str:
    """Resolves relative image paths across varying benchmark folder structures."""
    if not rel_path:
        return ""
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    candidates = [
        os.path.join(base_dir, rel_path),
        os.path.join(base_dir, "images", rel_path),
        os.path.join(base_dir, "vstar", rel_path),
        os.path.join(base_dir, os.path.basename(rel_path)),
        rel_path,
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.join(base_dir, rel_path)


def discover_benchmark_dataset(
    benchmark: str = "auto",
    preferred_dir: Optional[str] = None,
    preferred_ann: Optional[str] = None,
    vstar_dir: Optional[str] = None,
    hrbench_dir: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Discovers dataset base directory, annotation file, and resolved benchmark name.
    Returns (benchmark_name, resolved_base_dir, annotation_path).
    Supports HRBench (8k/4k) and V* benchmark.
    """
    bm_lower = (benchmark or "auto").lower().strip()

    # Determine candidate benchmark(s) to check
    if bm_lower in ("vstar", "v_star", "v*"):
        target_benchmarks = ["vstar"]
    elif bm_lower in ("hrbench", "hr-bench", "hr-bench_8k", "hr-bench_4k"):
        target_benchmarks = ["hrbench"]
    else:  # auto mode
        combined_hints = f"{preferred_dir or ''} {preferred_ann or ''} {vstar_dir or ''}".lower()
        if "vstar" in combined_hints:
            target_benchmarks = ["vstar", "hrbench"]
        else:
            target_benchmarks = ["hrbench", "vstar"]

    for bm in target_benchmarks:
        if bm == "vstar":
            v_dirs = []
            if preferred_dir:
                v_dirs.append(preferred_dir)
            if vstar_dir:
                v_dirs.append(vstar_dir)
            env_vstar = os.environ.get("VSTAR_DIR")
            if env_vstar:
                v_dirs.append(env_vstar)

            v_dirs.extend([
                "/home/ubuntu/workspace/hr/data/vstar",
                "/home/ubuntu/workspace/hr/data/vstar_bench",
                "/data/shared/cvpr/dat/datasets/vstar",
                os.path.join(SCRIPT_DIR, "../../../../docs/experiments/freeform/ICML26-CVSearch/cvsearch/CVsearch"),
                os.path.join(SCRIPT_DIR, "../../../freeform/ICML26-CVSearch/cvsearch/CVsearch"),
                "docs/experiments/freeform/ICML26-CVSearch/cvsearch/CVsearch",
                "../freeform/ICML26-CVSearch/cvsearch/CVsearch",
                "docs/experiments/multisource/data/vstar",
                "d:/Project/mysite/docs/experiments/freeform/ICML26-CVSearch/cvsearch/CVsearch",
            ])

            for d in v_dirs:
                if not d:
                    continue
                if preferred_ann and os.path.exists(preferred_ann):
                    return "vstar", (d if os.path.exists(d) else os.path.dirname(os.path.abspath(preferred_ann))), os.path.abspath(preferred_ann)

                if os.path.exists(d):
                    ann_names = [
                        "annotation_vstar.json",
                        "annotation_vstar_updated.json",
                        "annotation.json",
                        "vstar.json",
                        "vstar/annotation_vstar.json",
                    ]
                    for a_name in ann_names:
                        a_path = os.path.join(d, a_name)
                        if os.path.exists(a_path):
                            return "vstar", d, os.path.abspath(a_path)

        elif bm == "hrbench":
            h_dirs = []
            if preferred_dir:
                h_dirs.append(preferred_dir)
            if hrbench_dir:
                h_dirs.append(hrbench_dir)
            env_hr = os.environ.get("HRBENCH_DIR")
            if env_hr:
                h_dirs.append(env_hr)

            h_dirs.extend([
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

            for d in h_dirs:
                if not d:
                    continue
                if preferred_ann and os.path.exists(preferred_ann):
                    resolved_name = "hr-bench_4k" if "4k" in preferred_ann else "hr-bench_8k"
                    return resolved_name, (d if os.path.exists(d) else os.path.dirname(os.path.abspath(preferred_ann))), os.path.abspath(preferred_ann)

                if os.path.exists(d):
                    for a_name in ["annotation_hr-bench_8k.json", "annotation_hr-bench_4k.json", "annotation.json"]:
                        a_path = os.path.join(d, a_name)
                        if os.path.exists(a_path):
                            resolved_name = "hr-bench_4k" if "4k" in a_name else "hr-bench_8k"
                            return resolved_name, d, os.path.abspath(a_path)

    raise FileNotFoundError(
        f"[Dataset Error] Could not discover dataset for benchmark='{benchmark}'. "
        f"Please provide --dataset_dir, --vstar_dir, or --hrbench_dir pointing to dataset folder containing annotations."
    )


def discover_hrbench_dir(preferred_path: Optional[str]) -> Tuple[str, str]:
    """Discovers HRBench root directory and annotation file with multiple local fallbacks."""
    _, base_dir, ann_path = discover_benchmark_dataset(
        benchmark="hrbench",
        preferred_dir=preferred_path,
    )
    return base_dir, ann_path


def resolve_gt_info(sample: Dict[str, Any], cycle_idx: int = 0) -> Tuple[str, str, int]:
    """
    Resolves ground truth letter, answer string, and option index for both HRBench and V*.
    Returns (gt_letter, gt_text, gt_index).
    """
    # 1. Explicit label (e.g., annotation_vstar_updated.json)
    if "label" in sample and sample["label"]:
        lbl = str(sample["label"]).strip().upper()
        idx = ord(lbl) - ord('A') if len(lbl) == 1 and 'A' <= lbl <= 'Z' else 0
        raw_opts = sample.get("options", [])
        gt_txt = raw_opts[0] if (isinstance(raw_opts, list) and len(raw_opts) > 0) else ""
        return lbl, str(gt_txt), idx

    # 2. HRBench answer format
    if "answer" in sample and sample["answer"] is not None:
        ans = sample["answer"]
        if isinstance(ans, list):
            letter = str(ans[cycle_idx]).strip().upper() if 0 <= cycle_idx < len(ans) else str(ans[0]).strip().upper()
        else:
            letter = str(ans).strip().upper()
        idx = ord(letter) - ord('A') if len(letter) == 1 and 'A' <= letter <= 'Z' else 0
        return letter, letter, idx

    # 3. Standard V* format: options[0] is ground truth, corresponds to letter 'A' (index 0)
    raw_opts = sample.get("options", [])
    if isinstance(raw_opts, list) and len(raw_opts) > 0:
        return "A", str(raw_opts[0]), 0

    return "", "", 0


def resolve_gt_letter(answer_obj: Any, cycle_idx: int = 0) -> str:
    """Extracts ground truth option letter (e.g. 'A', 'B') from HRBench/V* annotations."""
    if isinstance(answer_obj, dict):
        letter, _, _ = resolve_gt_info(answer_obj, cycle_idx)
        return letter
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
        # Check if list of preformatted blocks (HRBench style: ["A. ...\nB. ...", ...])
        if isinstance(options_raw[0], str) and ("\n" in options_raw[0] or re.match(r"^[\(\[\"']*[A-Fa-f][\)\]\.\:\s]", options_raw[0].strip())):
            opt_block = options_raw[0]
        elif all(isinstance(opt, str) for opt in options_raw):
            # List of option strings (V* style: ["The color...", "The color..."])
            letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
            for i, opt in enumerate(options_raw):
                let = letters[i] if i < len(letters) else str(i)
                opt_dict[let] = str(opt).strip().lower()
            return opt_dict
        else:
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


def extract_predicted_option(pred_str: Optional[Any], options_raw: Any = None) -> Optional[str]:
    """Extracts the multiple-choice option letter ('A', 'B', 'C', 'D') from model output or option index."""
    if pred_str is None:
        return None

    # Handle numeric index from multiple_choices_inference (0 -> 'A', 1 -> 'B', etc.)
    int_types = (int, np.integer) if np is not None else (int,)
    if isinstance(pred_str, int_types):
        idx = int(pred_str)
        return chr(65 + idx) if 0 <= idx < 26 else str(idx)

    if not isinstance(pred_str, str):
        pred_str = str(pred_str)

    clean_pred = pred_str.strip()
    if not clean_pred or clean_pred.lower() in ("no answer found", "none", "n/a", "[error: image not found]"):
        return None

    # Numeric string check (e.g. "0", "1", "2", "3")
    if clean_pred.isdigit():
        idx = int(clean_pred)
        if 0 <= idx < 26:
            return chr(65 + idx)

    # 0. XML tag extraction <answer>...</answer> (exact match with Stage 6 XML format directive)
    xml_m = re.search(r"<answer>(.*?)</answer>", clean_pred, re.DOTALL | re.IGNORECASE)
    if xml_m:
        inside = xml_m.group(1).strip()
        if inside.lower() in ("none", "no answer found", "n/a"):
            return None
        if inside.isdigit():
            idx = int(inside)
            if 0 <= idx < 26:
                return chr(65 + idx)
        m = re.search(r"\b([A-Da-d])\b", inside)
        if m:
            return m.group(1).upper()
        clean_pred = inside

    # 1. Exact single letter alone: "A", "b", "(A)", "[C]", "A.", "B:", "(C)."
    if re.fullmatch(r"[\(\[]?([A-Ea-e])[\)\]]?[\.\:\,]?", clean_pred):
        return re.sub(r"[^\w]", "", clean_pred).upper()

    # 2. Explicit prefix like "Option A", "Choice (B)", "Answer: C", "The answer is (D)."
    match_prefix = re.search(
        r"(?:(?:the\s+)?(?:final\s+)?(?:answer|choice|option)\s*(?:is|\:|\=)?\s*)[\(\[]?([A-Ea-e])[\)\]]?(?:[\.\:\,\s]|\Z)",
        clean_pred,
        re.IGNORECASE,
    )
    if match_prefix:
        return match_prefix.group(1).upper()

    # 3. Formatted list item starting the answer, e.g. "(A) white" or "A. white" or "A: white"
    match_item = re.match(r"^\s*[\(\[]([A-Ea-e])[\)\]]\s*", clean_pred)
    if match_item:
        return match_item.group(1).upper()
    match_dot = re.match(r"^\s*([A-Ea-e])[\.\:\)]\s*", clean_pred)
    if match_dot:
        return match_dot.group(1).upper()

    # 4. Fallback: match prediction content against option candidate text
    if options_raw:
        opt_dict = parse_options_text(options_raw)
        p_lower = clean_pred.lower()
        for letter, text in opt_dict.items():
            if text:
                if p_lower == text or p_lower.startswith(text) or text in p_lower:
                    return letter
                if len(p_lower) >= 3 and p_lower in text:
                    return letter

    return None


# ==============================================================================
# SAMPLE EXECUTION VIA MSSearch
# ==============================================================================

def execute_single_sample(
    sample_index: int,
    dataset_data: List[Dict[str, Any]],
    dataset_base_dir: str,
    sam3_engine: Any,
    siglip_engine: Any,
    qwen_engine: Any,
    nlp_parser: Any,
    config: MSSearchConfig,
    num_candidate_images: int = 3,
    window_size: int = 5,
    random_seed: int = 42,
    ic_examples: Optional[Dict[str, Any]] = None,
    use_logits_match: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Constructs candidate pool C = {I_1, ..., I_N} and executes MSSearch directly.
    Supports both HRBench and V* benchmark sample schemas.
    """
    if "hrbench_data" in kwargs:
        dataset_data = kwargs.pop("hrbench_data")
    if "hrbench_base_dir" in kwargs:
        dataset_base_dir = kwargs.pop("hrbench_base_dir")

    target_sample = dataset_data[sample_index]
    real_question = target_sample.get("question", "")
    cycle_idx = target_sample.get("cycle_category", 0)
    test_type = target_sample.get("test_type", "")
    gt_letter, gt_text, gt_idx = resolve_gt_info(target_sample, cycle_idx)
    raw_options = target_sample.get("options", {})
    answer_type = target_sample.get("answer_type", getattr(config, "answer_type", "free_form"))
    is_logits_match = (answer_type == "logits_match" or use_logits_match)

    print(f"\n" + "-" * 75)
    print(f"[Sample #{sample_index}] Question: '{real_question}'")
    test_type_str = f" | Test Type: {test_type}" if test_type else ""
    print(f"[Sample #{sample_index}] Ground Truth: '{gt_letter}' (index: {gt_idx}){test_type_str} | Candidate Images: N={num_candidate_images}")

    # 1. Load Target Image
    target_img_rel = target_sample.get("input_image", "")
    target_img_path = resolve_image_path(dataset_base_dir, target_img_rel)
    if not os.path.exists(target_img_path):
        raise FileNotFoundError(f"Target image not found at: {target_img_path} (rel: '{target_img_rel}', base: '{dataset_base_dir}')")
    target_canvas = Image.open(target_img_path).convert("RGB")

    # 2. Sample Distractor Images from [sample_index - window_size, sample_index + window_size]
    total_samples = len(dataset_data)
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
        d_sample = dataset_data[d_idx]
        d_rel = d_sample.get("input_image", "")
        d_path = resolve_image_path(dataset_base_dir, d_rel)
        if os.path.exists(d_path):
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
        ic_examples=ic_examples,
    )
    total_latency = round(time.perf_counter() - t0, 3)

    telemetry = resp.get("telemetry", {})
    primary_answer = resp.get("primary_answer", "")
    route_selected = telemetry.get("route", "UNKNOWN")
    evidence_images = resp.get("evidence_images", [])

    # Final bounding box coordinates before Stage 5 aggregation (Requirement 2)
    final_crop_boxes_before_aggregate = resp.get("final_crop_boxes_before_aggregate", [])

    pred_letter = extract_predicted_option(primary_answer, raw_options)

    # Logits match fallback if primary_answer did not yield an option and engine supports it
    if pred_letter is None and is_logits_match and hasattr(qwen_engine, "multiple_choices_inference") and raw_options:
        opt_list = raw_options if isinstance(raw_options, list) else list(raw_options.values())
        eval_img = candidate_images[target_pool_idx]
        try:
            chosen_idx = qwen_engine.multiple_choices_inference(
                image_pil=eval_img,
                question=real_question,
                options=opt_list,
                searched_nodes=[],
            )
            chosen_idx = int(chosen_idx)
            pred_letter = chr(65 + chosen_idx) if 0 <= chosen_idx < 26 else str(chosen_idx)
            primary_answer = f"[logits_match fallback] Option {pred_letter} ({chosen_idx}): {opt_list[chosen_idx] if 0 <= chosen_idx < len(opt_list) else ''}"
        except Exception:
            pass

    is_correct = (pred_letter == gt_letter) if pred_letter and gt_letter else False
    res_tag = "[CORRECT]" if is_correct else "[INCORRECT]"

    print(f"[Result] Predicted: '{pred_letter}' vs GT: '{gt_letter}' -> {res_tag} | Primary: '{primary_answer}' | Route: {route_selected} | Latency: {total_latency}s")

    is_cross_object = bool(resp.get("is_cross_object", telemetry.get("is_cross_object", False)))
    is_relational = bool(resp.get("is_relational", telemetry.get("is_relational", False)))
    stage5_mode = str(resp.get("stage5_mode", telemetry.get("stage5_mode", "crops_only")))

    record = {
        "sample_index": sample_index,
        "question": real_question,
        "cycle_idx": cycle_idx,
        "test_type": test_type,
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
        "is_cross_object": is_cross_object,
        "is_relational": is_relational,
        "stage5_mode": stage5_mode,
        "final_crop_boxes_before_aggregate": final_crop_boxes_before_aggregate,
        "stage6_patch_boxes": resp.get("stage6_patch_boxes", []),
    }
    return record


# ==============================================================================
# DIRECT ANSWER BASELINE EXECUTION (VANILLA VLM WITHOUT VISUAL SEARCH)
# ==============================================================================

def _create_composite_canvas(images: List[Image.Image], max_side: int = 1024) -> Image.Image:
    """Stitches candidate images into a single horizontal composite canvas for single-image models."""
    if not images:
        return Image.new("RGB", (224, 224), (128, 128, 128))
    if len(images) == 1:
        return images[0]
    target_h = min(max(img.height for img in images), max_side // 2)
    resized = []
    for img in images:
        scale = target_h / max(1, img.height)
        new_w = max(1, int(round(img.width * scale)))
        resized.append(img.resize((new_w, target_h), Image.Resampling.BILINEAR))
    total_w = sum(img.width for img in resized)
    composite = Image.new("RGB", (total_w, target_h), (255, 255, 255))
    x_offset = 0
    for img in resized:
        composite.paste(img, (x_offset, 0))
        x_offset += img.width
    if composite.width > max_side:
        scale = max_side / composite.width
        composite = composite.resize((max_side, max(1, int(round(composite.height * scale)))), Image.Resampling.BILINEAR)
    return composite


def run_direct_vlm_inference(
    qwen_engine: Any,
    images: List[Image.Image],
    prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """Executes direct VLM inference over candidate images for baseline evaluation."""
    # 1. Clamp candidate images to safe max resolution (1024px)
    safe_images = []
    for img in images:
        if max(img.width, img.height) > 1024:
            scale = 1024.0 / max(img.width, img.height)
            safe_images.append(
                img.resize(
                    (int(round(img.width * scale)), int(round(img.height * scale))),
                    Image.Resampling.BILINEAR,
                )
            )
        else:
            safe_images.append(img)

    # 2. InternVL Route (supports both N=1 and N>1 multi-image pools via .chat)
    if hasattr(qwen_engine, "model") and hasattr(qwen_engine, "tokenizer") and hasattr(getattr(qwen_engine, "model", None), "chat"):
        if len(safe_images) == 1 and hasattr(qwen_engine, "free_form_using_nodes"):
            return qwen_engine.free_form_using_nodes(
                image_pil=safe_images[0],
                question=prompt,
                searched_nodes=[],
                max_new_tokens=max_new_tokens,
            )
        try:
            from mssearch.models.modeling_internvl import load_image, IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN
            full_pixel_values = []
            num_patches_list = []
            for img in safe_images:
                pv = load_image(img, input_size=getattr(qwen_engine, "input_size", (448, 448))[0], max_num=getattr(qwen_engine, "anyres_num", 12))
                full_pixel_values.append(pv.to(qwen_engine.dtype))
                num_patches_list.append(pv.shape[0])
            full_pixel_values = torch.cat(full_pixel_values, dim=0).to(qwen_engine.device)

            qs_parts = []
            for idx in range(len(safe_images)):
                qs_parts.append(f"Candidate Image {idx + 1}:\n<image>\n")
            qs_parts.append(f"\n{prompt}")
            chat_prompt = "".join(qs_parts)

            image_idx = 0
            for num_patches in num_patches_list:
                while "<image>" in chat_prompt:
                    num_patches = num_patches_list[image_idx]
                    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * qwen_engine.model.num_image_token * num_patches + IMG_END_TOKEN
                    chat_prompt = chat_prompt.replace("<image>", image_tokens, 1)
                    image_idx += 1

            gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
            response = qwen_engine.model.chat(qwen_engine.tokenizer, full_pixel_values, chat_prompt, gen_cfg)
            return response.strip()
        except Exception as e_ivl_chat:
            print(f"[Notice] InternVL multi-image direct inference exception: {e_ivl_chat}")

    # 3. ModelGlobalLocal route (LLaVA models using image_processor + tokenizer)
    if hasattr(qwen_engine, "image_processor") and hasattr(qwen_engine, "tokenizer") and not hasattr(qwen_engine, "processor"):
        if len(safe_images) > 1:
            composite = _create_composite_canvas(safe_images)
            return qwen_engine.generate_response(image_pil=composite, question=prompt, max_new_tokens=max_new_tokens)
        else:
            return qwen_engine.generate_response(image_pil=safe_images[0], question=prompt, max_new_tokens=max_new_tokens)

    # 4. Single-image route (N=1) for other models
    if len(safe_images) == 1:
        if hasattr(qwen_engine, "generate_response"):
            return qwen_engine.generate_response(
                image_pil=safe_images[0],
                question=prompt,
                max_new_tokens=max_new_tokens,
            )
        elif hasattr(qwen_engine, "free_form_using_nodes"):
            return qwen_engine.free_form_using_nodes(
                image_pil=safe_images[0],
                question=prompt,
                searched_nodes=[],
                max_new_tokens=max_new_tokens,
            )

    # 4. Multi-image route (N > 1) via HuggingFace processor (Qwen2-VL, LLaVA-OneVision, etc.)
    if (
        hasattr(qwen_engine, "processor")
        and hasattr(qwen_engine, "model")
        and hasattr(qwen_engine, "get_prompt_from_qs")
    ):
        # Inoculate LlavaProcessor against TypeError: unsupported operand type(s) for //: 'int' and 'NoneType'
        p_size = getattr(qwen_engine.processor, "patch_size", None)
        if p_size is None:
            if hasattr(qwen_engine, "model") and hasattr(qwen_engine.model, "config"):
                cfg = qwen_engine.model.config
                v_cfg = getattr(cfg, "vision_config", None)
                if isinstance(v_cfg, dict):
                    p_size = v_cfg.get("patch_size")
                elif v_cfg is not None:
                    p_size = getattr(v_cfg, "patch_size", None)
                if p_size is None:
                    p_size = getattr(cfg, "patch_size", None)
            if p_size is None and hasattr(qwen_engine.processor, "image_processor"):
                p_size = getattr(qwen_engine.processor.image_processor, "patch_size", None)
            if p_size is None:
                p_size = 14
            qwen_engine.processor.patch_size = int(p_size)
            if hasattr(qwen_engine.processor, "image_processor") and getattr(qwen_engine.processor.image_processor, "patch_size", None) is None:
                qwen_engine.processor.image_processor.patch_size = int(p_size)

        if getattr(qwen_engine.processor, "num_additional_image_tokens", None) is None:
            qwen_engine.processor.num_additional_image_tokens = 0

        if getattr(qwen_engine.processor, "vision_feature_select_strategy", None) is None:
            strat = "default"
            if hasattr(qwen_engine, "model") and hasattr(qwen_engine.model, "config"):
                cfg = qwen_engine.model.config
                v_cfg = getattr(cfg, "vision_config", None)
                if isinstance(v_cfg, dict):
                    strat = v_cfg.get("vision_feature_select_strategy", strat)
                elif v_cfg is not None:
                    strat = getattr(v_cfg, "vision_feature_select_strategy", strat)
                if strat is None:
                    strat = getattr(cfg, "vision_feature_select_strategy", "default")
            qwen_engine.processor.vision_feature_select_strategy = strat or "default"

        qs_parts = []
        for idx in range(len(safe_images)):
            qs_parts.append(f"Candidate Image {idx + 1}:\n<image>\n")
        qs_parts.append(f"\n{prompt}")
        qs_full = "".join(qs_parts)

        prompt_text = qwen_engine.get_prompt_from_qs(qs_full)

        # Multi-image direct evaluation (Tier A: flat image list; Tier B: nested image list)
        for images_arg in (safe_images, [safe_images]):
            try:
                model_inputs = qwen_engine.processor(
                    text=[prompt_text],
                    images=images_arg,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                ).to(qwen_engine.device)

                if "pixel_values" in model_inputs and hasattr(model_inputs["pixel_values"], "dtype"):
                    m_dtype = getattr(qwen_engine.model, "dtype", getattr(qwen_engine, "dtype", None))
                    if m_dtype in (torch.float16, torch.bfloat16):
                        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(m_dtype)

                if model_inputs.input_ids.shape[0] == 1 and "attention_mask" in model_inputs:
                    if not (model_inputs.attention_mask == 0).any():
                        model_inputs.pop("attention_mask", None)

                infer_ctx = torch.inference_mode() if torch is not None else contextlib.nullcontext()
                with infer_ctx:
                    generated_ids = qwen_engine.model.generate(
                        **model_inputs,
                        use_cache=True,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=1.0,
                        do_sample=False,
                    )
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
                decoded = qwen_engine.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                return decoded[0].strip()
            except Exception:
                pass

        # Tier C: Composite Canvas fallback (stitching candidate images horizontally into a single canvas)
        try:
            composite_canvas = _create_composite_canvas(safe_images)
            single_prompt = f"<image>\n{prompt}"
            single_prompt_text = qwen_engine.get_prompt_from_qs(single_prompt)
            model_inputs = qwen_engine.processor(
                text=[single_prompt_text],
                images=composite_canvas,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            ).to(qwen_engine.device)

            if "pixel_values" in model_inputs and hasattr(model_inputs["pixel_values"], "dtype"):
                m_dtype = getattr(qwen_engine.model, "dtype", getattr(qwen_engine, "dtype", None))
                if m_dtype in (torch.float16, torch.bfloat16):
                    model_inputs["pixel_values"] = model_inputs["pixel_values"].to(m_dtype)

            if model_inputs.input_ids.shape[0] == 1 and "attention_mask" in model_inputs:
                if not (model_inputs.attention_mask == 0).any():
                    model_inputs.pop("attention_mask", None)

            infer_ctx = torch.inference_mode() if torch is not None else contextlib.nullcontext()
            with infer_ctx:
                generated_ids = qwen_engine.model.generate(
                    **model_inputs,
                    use_cache=True,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.0,
                    do_sample=False,
                )
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
            decoded = qwen_engine.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return decoded[0].strip()
        except Exception:
            pass

        # Tier D: Target-only image fallback (safe_images[0])
        try:
            single_prompt = f"<image>\n{prompt}"
            single_prompt_text = qwen_engine.get_prompt_from_qs(single_prompt)
            model_inputs = qwen_engine.processor(
                text=[single_prompt_text],
                images=safe_images[0],
                return_tensors="pt",
                padding=True,
                padding_side="left",
            ).to(qwen_engine.device)

            if "pixel_values" in model_inputs and hasattr(model_inputs["pixel_values"], "dtype"):
                m_dtype = getattr(qwen_engine.model, "dtype", getattr(qwen_engine, "dtype", None))
                if m_dtype in (torch.float16, torch.bfloat16):
                    model_inputs["pixel_values"] = model_inputs["pixel_values"].to(m_dtype)

            if model_inputs.input_ids.shape[0] == 1 and "attention_mask" in model_inputs:
                if not (model_inputs.attention_mask == 0).any():
                    model_inputs.pop("attention_mask", None)

            infer_ctx = torch.inference_mode() if torch is not None else contextlib.nullcontext()
            with infer_ctx:
                generated_ids = qwen_engine.model.generate(
                    **model_inputs,
                    use_cache=True,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.0,
                    do_sample=False,
                )
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
            decoded = qwen_engine.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return decoded[0].strip()
        except Exception:
            pass

    # Fallback for generic VLM
    if hasattr(qwen_engine, "generate_response"):
        return qwen_engine.generate_response(image_pil=safe_images[0], question=prompt, max_new_tokens=max_new_tokens)
    if hasattr(qwen_engine, "free_form_using_nodes"):
        return qwen_engine.free_form_using_nodes(image_pil=safe_images[0], question=prompt, searched_nodes=[], max_new_tokens=max_new_tokens)
    return str(qwen_engine(image_pil=safe_images, question=prompt))


def execute_single_sample_direct(
    sample_index: int,
    dataset_data: List[Dict[str, Any]],
    dataset_base_dir: str,
    qwen_engine: Any,
    config: MSSearchConfig,
    num_candidate_images: int = 3,
    window_size: int = 5,
    random_seed: int = 42,
    target_only: bool = False,
    use_logits_match: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Direct answer baseline evaluation: queries the VLM directly without visual search.
    Supports single-image (target_only=True or N=1) and multi-image candidate pools (N>1).
    Supports both free-form generation and logits-matching multiple-choice inference.
    """
    if "hrbench_data" in kwargs:
        dataset_data = kwargs.pop("hrbench_data")
    if "hrbench_base_dir" in kwargs:
        dataset_base_dir = kwargs.pop("hrbench_base_dir")

    target_sample = dataset_data[sample_index]
    real_question = target_sample.get("question", "")
    cycle_idx = target_sample.get("cycle_category", 0)
    test_type = target_sample.get("test_type", "")
    gt_letter, gt_text, gt_idx = resolve_gt_info(target_sample, cycle_idx)
    raw_options = target_sample.get("options", {})
    answer_type = target_sample.get("answer_type", getattr(config, "answer_type", "free_form"))
    is_logits_match = (answer_type == "logits_match" or use_logits_match)

    # 1. Load Target Image
    target_img_rel = target_sample.get("input_image", "")
    target_img_path = resolve_image_path(dataset_base_dir, target_img_rel)
    if not os.path.exists(target_img_path):
        raise FileNotFoundError(f"Target image not found at: {target_img_path} (rel: '{target_img_rel}', base: '{dataset_base_dir}')")
    target_canvas = Image.open(target_img_path).convert("RGB")

    # 2. Build candidate pool
    if target_only or num_candidate_images <= 1:
        candidate_images = [target_canvas]
        target_pool_idx = 0
    else:
        total_samples = len(dataset_data)
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
            d_sample = dataset_data[d_idx]
            d_rel = d_sample.get("input_image", "")
            d_path = resolve_image_path(dataset_base_dir, d_rel)
            if os.path.exists(d_path):
                d_img = Image.open(d_path).convert("RGB")
                pool_items.append({"is_target": False, "sample_idx": d_idx, "image": d_img, "path": d_rel})

        rng.shuffle(pool_items)
        candidate_images = [item["image"] for item in pool_items]
        target_pool_idx = next(i for i, item in enumerate(pool_items) if item["is_target"])

    print(f"\n" + "-" * 75)
    print(f"[Direct Baseline][Sample #{sample_index}] Question: '{real_question}'")
    test_type_str = f" | Test Type: {test_type}" if test_type else ""
    print(f"[Direct Baseline][Sample #{sample_index}] Ground Truth: '{gt_letter}' (index: {gt_idx}){test_type_str} | Candidate Images: N={len(candidate_images)} (Target Index: {target_pool_idx})")

    # 3. Direct evaluation: Check logits_match first (V* standard protocol)
    if is_logits_match and hasattr(qwen_engine, "multiple_choices_inference") and raw_options:
        opt_list = raw_options if isinstance(raw_options, list) else list(raw_options.values())
        query_image = candidate_images[target_pool_idx] if (target_only or len(candidate_images) == 1) else _create_composite_canvas(candidate_images)
        t0 = time.perf_counter()
        chosen_idx = qwen_engine.multiple_choices_inference(
            image_pil=query_image,
            question=real_question,
            options=opt_list,
            searched_nodes=[],
        )
        total_latency = round(time.perf_counter() - t0, 3)
        chosen_idx = int(chosen_idx)
        pred_letter = chr(65 + chosen_idx) if 0 <= chosen_idx < 26 else str(chosen_idx)
        chosen_text = opt_list[chosen_idx] if 0 <= chosen_idx < len(opt_list) else ""
        raw_response = f"Option {pred_letter} ({chosen_idx}): {chosen_text}"
    else:
        # 3b. Format prompt with candidate options if present
        if "text" in target_sample and target_sample["text"]:
            base_question = target_sample["text"]
            opt_str = target_sample["text"]
        else:
            opt_str = format_options(raw_options, cycle_idx=cycle_idx)
            if "Options:" not in real_question and opt_str:
                base_question = format_question_with_options(real_question, raw_options, cycle_idx=cycle_idx)
            else:
                base_question = real_question

        if opt_str or "Options:" in real_question or "text" in target_sample:
            prompt_text = (
                f"{base_question}\n\n"
                "Select the best option from the choices above that answers the question.\n"
                "State ONLY the option letter and concise choice (e.g. <answer>(A)</answer> or <answer>A. choice text</answer>).\n"
                "Do NOT write conversational filler, reasoning, or explanations.\n"
                "If the target object or requested information is NOT clearly visible, output strictly:\n"
                "<answer>None</answer>\n\n"
                "Otherwise, output the option letter inside <answer>...</answer> tags:\n"
                "<answer>(A)</answer>"
            )
        else:
            prompt_text = (
                f"{real_question}\n\n"
                "State ONLY the direct factual answer (1 to 5 words). Do NOT write conversational filler or explanations.\n"
                "If the target object or requested text/information is NOT clearly visible, output strictly:\n"
                "<answer>None</answer>\n\n"
                "Otherwise, output the concise answer inside <answer>...</answer> tags:\n"
                "<answer>concise answer</answer>"
            )

        # 4. Direct VLM Query
        t0 = time.perf_counter()
        max_tokens = getattr(config, "max_vlm_tokens", 128)
        raw_response = run_direct_vlm_inference(
            qwen_engine=qwen_engine,
            images=candidate_images,
            prompt=prompt_text,
            max_new_tokens=max_tokens,
        )
        total_latency = round(time.perf_counter() - t0, 3)
        pred_letter = extract_predicted_option(raw_response, raw_options)

    is_correct = (pred_letter == gt_letter) if pred_letter and gt_letter else False
    res_tag = "[CORRECT]" if is_correct else "[INCORRECT]"

    print(f"[Direct Baseline Result] Predicted: '{pred_letter}' vs GT: '{gt_letter}' -> {res_tag} | Primary: '{raw_response}' | Route: direct_baseline | Latency: {total_latency}s")

    record = {
        "sample_index": sample_index,
        "question": real_question,
        "cycle_idx": cycle_idx,
        "test_type": test_type,
        "gt_letter": gt_letter,
        "predicted_letter": pred_letter,
        "is_correct": is_correct,
        "primary_answer": raw_response,
        "raw_vlm_response": raw_response,
        "target_pool_idx": target_pool_idx,
        "evidence_images": [target_pool_idx] if is_correct else [],
        "route_selected": "direct_baseline",
        "latency_sec": total_latency,
        "timings": {"direct_baseline": total_latency},
        "is_cross_object": False,
        "is_relational": False,
        "stage5_mode": "none",
        "final_crop_boxes_before_aggregate": [],
        "stage6_patch_boxes": [],
    }
    return record


# ==============================================================================
# MAIN ENTRY POINT & CONFIG MANAGEMENT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Source Visual Search Engine (MSSearch) Runner for HRBench and V* Benchmark"
    )

    # Dataset & Benchmark Arguments
    parser.add_argument(
        "--benchmark",
        type=str,
        default="auto",
        choices=["auto", "vstar", "hrbench", "hr-bench_8k", "hr-bench_4k"],
        help="Benchmark dataset to evaluate: 'auto' (detect from paths), 'vstar', or 'hrbench' / 'hr-bench_8k' / 'hr-bench_4k'.",
    )
    parser.add_argument(
        "--dataset_dir",
        "--data_dir",
        type=str,
        default=None,
        dest="dataset_dir",
        help="Unified root path to benchmark dataset directory.",
    )
    parser.add_argument(
        "--vstar_dir",
        type=str,
        default=os.environ.get("VSTAR_DIR", "/home/ubuntu/workspace/hr/data/vstar"),
        help="Root path to V* benchmark dataset (default: /home/ubuntu/workspace/hr/data/vstar or $VSTAR_DIR).",
    )
    parser.add_argument(
        "--hrbench_dir",
        type=str,
        default=None,
        help="Root path to HRBench dataset.",
    )
    parser.add_argument(
        "--annotation_file",
        "--annotation_path",
        type=str,
        default=None,
        dest="annotation_file",
        help="Explicit path to annotation JSON file.",
    )
    parser.add_argument(
        "--use_logits_match",
        action="store_true",
        help="Force logits-matching evaluation for multiple choice inference (standard V* protocol).",
    )
    parser.add_argument(
        "--answer_type",
        "--answer-type",
        type=str,
        default="auto",
        choices=["auto", "logits_match", "free_form", "option_single", "multiple_choice"],
        help="Prompt answer format: 'auto' (use annotation type), 'logits_match', or 'free_form'/'option_single'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "results"),
        help="Directory to save JSONL telemetry logs and summaries.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Explicit results JSONL filename or path (default: auto-timestamped results_YYYYMMDD_HHMMSS.jsonl in output_dir).",
    )
    parser.add_argument(
        "--sam3_path",
        type=str,
        default=os.environ.get("SAM3_PATH", "/home/ubuntu/workspace/hr/models/sam3/sam3/sam3.pt"),
        help="Path to local SAM3 checkpoint.",
    )
    parser.add_argument(
        "--qwen_path",
        "--model_path",
        "--model-path",
        "--model",
        type=str,
        dest="qwen_path",
        default=os.environ.get("MODEL_PATH", os.environ.get("QWEN_PATH", "/home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct")),
        help="Path to local VLM checkpoint directory (local only, no online downloads).",
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
    parser.add_argument("--resume", action="store_true", help="Skip already evaluated samples and continue appending to results file.")
    parser.add_argument("--resume_from", type=str, default=None, help="Specific previous results JSONL path to resume from (defaults to latest results_*.jsonl in output_dir).")
    parser.add_argument(
        "--mode",
        type=str,
        default="e2e",
        choices=["e2e", "direct", "direct_answer", "baseline"],
        help="Execution mode: 'e2e' (full MSSearch pipeline) or 'direct'/'direct_answer'/'baseline' (direct VLM baseline evaluation).",
    )
    parser.add_argument(
        "--direct_target_only",
        action="store_true",
        help="In direct answer mode, evaluate only the target image (N=1), ignoring distractor images.",
    )
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
        choices=["large"],
        default="large",
        help="Quick budget preset: 'large' (default high-depth visual search).",
    )

    # Logging & Display Arguments (Requirement 1: Only print final decision of each stage)
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["summary", "verbose", "quiet"],
        default="summary",
        help="Logging level: 'summary' (only print final decision of each stage), 'verbose' (full internal loops), 'quiet' (minimal stdout).",
    )

    # Visual Pixel Budget Arguments
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=3136,
        help="Minimum patch resolution in pixels (default: 3136, corresponds to 56x56 minimum patch resolution).",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=None,
        help="Maximum image resolution in pixels (default: 12845056, ~12.8M pixels, 3584x3584; overridden to 4194304, ~4.2M pixels, 2048x2048 for 32B checkpoints to prevent OOM).",
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
    parser.add_argument("--sam3_batch_size", type=int, default=None, help="Batch size for SAM3 patch inference.")
    parser.add_argument("--vlm_gating_batch_size", type=int, default=None, help="Batch size for vectorized VLM gating.")
    parser.add_argument("--enable_latent_repartition", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=None)
    parser.add_argument("--max_latent_repartition_depth", type=int, default=None)
    parser.add_argument("--max_latent_repartition_patches_per_image", type=int, default=None)
    parser.add_argument("--latent_repartition_threshold", type=float, default=None)
    parser.add_argument("--enable_ios_containment_merging", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=None)
    parser.add_argument("--ios_containment_threshold", type=float, default=None)
    parser.add_argument("--enable_stage5_spatial_nms", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=None)
    parser.add_argument("--stage5_nms_iou_threshold", type=float, default=None)
    parser.add_argument("--enable_spatial_compressed_canvas", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=None)
    parser.add_argument("--spatial_canvas_relational_only", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=None)
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
        "sam3_batch_size",
        "vlm_gating_batch_size",
        "enable_latent_repartition",
        "max_latent_repartition_depth",
        "max_latent_repartition_patches_per_image",
        "latent_repartition_threshold",
        "enable_ios_containment_merging",
        "ios_containment_threshold",
        "enable_stage5_spatial_nms",
        "stage5_nms_iou_threshold",
        "enable_spatial_compressed_canvas",
        "spatial_canvas_relational_only",
        "enable_early_consensus_exit",
    ]

    for param in cli_overrides:
        val = getattr(args, param, None)
        if val is not None:
            setattr(config, param, val)
            print(f"[Config Override] {param} -> {val}")

    # 4. Discover Benchmark Dataset
    benchmark_name, dataset_base_dir, annotation_path = discover_benchmark_dataset(
        benchmark=args.benchmark,
        preferred_dir=args.dataset_dir or args.hrbench_dir,
        preferred_ann=args.annotation_file,
        vstar_dir=args.vstar_dir,
        hrbench_dir=args.hrbench_dir,
    )
    print(f"[Dataset] Active benchmark : {benchmark_name}")
    print(f"[Dataset] Root directory   : {dataset_base_dir}")
    print(f"[Dataset] Annotation file  : {annotation_path}")

    with open(annotation_path, "r", encoding="utf-8") as f:
        dataset_data = json.load(f)
    print(f"[Dataset] Ingested {len(dataset_data)} total samples.")

    # Discover in-context examples if available
    ic_examples = None
    ic_path_candidates = [
        os.path.join(SCRIPT_DIR, f"../../mssearch/ic_examples/{benchmark_name}.json"),
        os.path.join(SCRIPT_DIR, f"../../mssearch/ic_examples/{benchmark_name.replace('-bench', '')}.json"),
        os.path.join(SCRIPT_DIR, "../../mssearch/ic_examples/vstar.json") if "vstar" in benchmark_name else None,
        os.path.join(SCRIPT_DIR, "../../mssearch/ic_examples/hr-bench_8k.json") if "hr" in benchmark_name else None,
    ]
    for ic_p in ic_path_candidates:
        if ic_p and os.path.exists(ic_p):
            try:
                with open(ic_p, "r", encoding="utf-8") as ic_f:
                    ic_examples = json.load(ic_f)
                print(f"[IC Examples] Loaded in-context examples from: {ic_p}")
                break
            except Exception:
                pass

    # 5. Generate timestamp string for de-duplication
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    # If --resume_from is explicitly passed, automatically enable resume mode
    if args.resume_from:
        args.resume = True

    is_direct_mode = args.mode in ("direct", "direct_answer", "baseline")
    prefix = f"results_{benchmark_name}_baseline" if is_direct_mode else f"results_{benchmark_name}"

    # 5a. Resolve results JSONL path and resume tracking
    if args.output_file:
        results_log_path = os.path.abspath(args.output_file)
        if args.resume:
            print(f"[Resume] Resuming from explicit output file: {results_log_path}")
        else:
            print(f"[Results] Telemetry will be logged to: {results_log_path}")
    elif args.resume:
        if args.resume_from and os.path.exists(args.resume_from):
            results_log_path = os.path.abspath(args.resume_from)
            print(f"[Resume] Resuming from explicit file: {results_log_path}")
        else:
            timestamped_candidates = []
            other_candidates = []
            if os.path.exists(args.output_dir):
                for f in os.listdir(args.output_dir):
                    if f.endswith(".jsonl"):
                        fp = os.path.join(args.output_dir, f)
                        if f.startswith(f"{prefix}_"):
                            timestamped_candidates.append((os.path.getmtime(fp), fp))
                        elif f == f"{prefix}.jsonl" or f == "results.jsonl":
                            other_candidates.append((os.path.getmtime(fp), fp))
            candidates = timestamped_candidates or other_candidates
            if candidates:
                candidates.sort(reverse=True)
                results_log_path = candidates[0][1]
                print(f"[Resume] Auto-detected most recent results file to resume: {results_log_path}")
            else:
                results_log_path = os.path.abspath(os.path.join(args.output_dir, f"{prefix}_{timestamp_str}.jsonl"))
                print(f"[Resume] No existing results file found in '{args.output_dir}'. Starting new log: {results_log_path}")
    else:
        results_log_path = os.path.abspath(os.path.join(args.output_dir, f"{prefix}_{timestamp_str}.jsonl"))
        print(f"[Results] Telemetry will be logged to timestamped file: {results_log_path}")

    # Config file snapshot (timestamped + canonical pointer)
    config_out_path = os.path.join(args.output_dir, f"config_used_{timestamp_str}.json")
    config.save_json(config_out_path)
    config.save_json(os.path.join(args.output_dir, "config_used.json"))
    print(f"[Config] Active configuration written to: {config_out_path}")

    # 6. Resolve Target Queue
    if args.all_samples:
        target_indices = list(range(len(dataset_data)))
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
        print(f"[Resume] Skipping {len(completed_ids)} already-evaluated samples. Remaining: {len(target_indices)} / {orig_count}")

    print(f"[Queue] Ready to evaluate {len(target_indices)} sample(s).")

    # 7. Model Allocation & Initialization
    if torch is None:
        raise ImportError("PyTorch is required for benchmark execution.")

    num_gpus = torch.cuda.device_count()
    sam3_device = "cuda:1" if num_gpus > 1 else ("cuda:0" if num_gpus > 0 else "cpu")
    siglip_device = "cuda:0" if num_gpus > 0 else "cpu"
    qwen_device = "cuda:0" if num_gpus > 0 else "cpu"

    print("=" * 80)
    print(f"HARDWARE TOPOLOGY: {num_gpus} CUDA GPUs Available | Mode: {args.mode}")
    if not is_direct_mode:
        print(f"  - SAM3 Engine      : {sam3_device}")
        print(f"  - SigLIP Engine    : {siglip_device}")
    print(f"  - Qwen2.5-VL Engine: {qwen_device}")
    print("=" * 80)

    if not is_direct_mode:
        if sam3_inference is None:
            raise ImportError("SAM3 dependencies not installed; required for mode='e2e'.")
        if SigLIPScorer is None:
            raise ImportError("SigLIP dependencies not installed; required for mode='e2e'.")
        if get_mssearch_response is None:
            raise ImportError("MSSearch pipeline not installed; required for mode='e2e'.")

        print(f"[Loading] SAM3 from '{args.sam3_path}' on {sam3_device}...")
        sam3_engine = sam3_inference(model_path=args.sam3_path, device=sam3_device)

        print(f"[Loading] SigLIP '{args.siglip_name}' on {siglip_device}...")
        siglip_engine = SigLIPScorer(
            model_name=args.siglip_name,
            device=siglip_device,
            torch_dtype=torch.float16 if "cuda" in siglip_device else torch.float32,
        )
    else:
        print(f"[Mode: {args.mode}] Direct answer baseline mode active - bypassing SAM3 and SigLIP initialization.")
        sam3_engine = None
        siglip_engine = None

    if ModelQwenVL is None:
        raise ImportError("ModelQwenVL dependencies not installed; required for VLM evaluation.")

    # Validate local model path exists (no online downloads)
    if not os.path.exists(args.qwen_path):
        raise FileNotFoundError(
            f"Local model path '{args.qwen_path}' not found! "
            f"Only local model directories are supported; online downloading is disabled."
        )

    # Resolve visual pixel budget
    if args.max_pixels is not None:
        resolved_max_pixels = args.max_pixels
    elif any(tag in str(args.qwen_path).lower() for tag in ["32b", "72b"]):
        resolved_max_pixels = 4194304
    else:
        resolved_max_pixels = 12845056

    model_path_str = str(args.qwen_path).lower()
    cfg_data = {}
    cfg_path = os.path.join(args.qwen_path, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
        except Exception:
            pass
    archs = [a.lower() for a in cfg_data.get("architectures", [])]
    model_type = cfg_data.get("model_type", "").lower()

    if "internvl" in model_path_str or any("internvl" in a for a in archs) or "internvl" in model_type:
        try:
            from mssearch.models.modeling_internvl import ModelInternvl
            print(f"[Loading] InternVL Engine from '{args.qwen_path}' on {qwen_device}...")
            qwen_engine = ModelInternvl(
                model_path=args.qwen_path,
                device=qwen_device,
                torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16,
                patch_scale=1.2,
            )
        except Exception as e_ivl:
            print(f"[Notice] ModelInternvl failed to initialize ({e_ivl}), falling back to ModelQwenVL unified loader.")
            qwen_engine = ModelQwenVL(
                model_path=args.qwen_path,
                device=qwen_device,
                torch_dtype=torch.float16 if "cuda" in qwen_device else torch.float32,
                patch_scale=1.2,
                min_pixels=args.min_pixels,
                max_pixels=resolved_max_pixels,
            )
    elif "llava" in model_path_str or any("llava" in a for a in archs) or "llava" in model_type:
        try:
            from mssearch.models.modeling_llava import ModelGlobalLocal
            print(f"[Loading] LLaVA GlobalLocal Engine from '{args.qwen_path}' on {qwen_device}...")
            is_qwen = "qwen" in model_path_str or "ov" in model_path_str or "onevision" in model_path_str or any("qwen" in a for a in archs)
            qwen_engine = ModelGlobalLocal(
                model_path=args.qwen_path,
                device=qwen_device,
                torch_dtype=torch.float16 if "cuda" in qwen_device else torch.float32,
                conv_type="qwen_1_5" if is_qwen else "llava_v1",
                patch_scale=1.2,
                bias_value=0.6,
            )
        except Exception as e_llava:
            print(f"[Error] ModelGlobalLocal failed to initialize: {e_llava}")
            raise
    else:
        print(f"[Loading] Local VLM from '{args.qwen_path}' on {qwen_device}...")
        guard_msg = "Large Model OOM-guard" if resolved_max_pixels == 4194304 else "full-scale"
        print(f"  • Pixel Budget: min_pixels={args.min_pixels} (56x56) | max_pixels={resolved_max_pixels} ({guard_msg})")
        qwen_engine = ModelQwenVL(
            model_path=args.qwen_path,
            device=qwen_device,
            torch_dtype=torch.float16 if "cuda" in qwen_device else torch.float32,
            patch_scale=1.2,
            min_pixels=args.min_pixels,
            max_pixels=resolved_max_pixels,
        )

    nlp_parser = None
    if not is_direct_mode and args.nlp_model:
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
    use_lm = args.use_logits_match or (args.answer_type == "logits_match")

    for queue_pos, s_idx in enumerate(target_indices, start=1):
        if s_idx < 0 or s_idx >= len(dataset_data):
            continue

        try:
            if is_direct_mode:
                rec = execute_single_sample_direct(
                    sample_index=s_idx,
                    dataset_data=dataset_data,
                    dataset_base_dir=dataset_base_dir,
                    qwen_engine=qwen_engine,
                    config=config,
                    num_candidate_images=args.num_candidate_images,
                    window_size=args.window_size,
                    random_seed=args.random_seed,
                    target_only=args.direct_target_only,
                    use_logits_match=use_lm,
                )
            else:
                rec = execute_single_sample(
                    sample_index=s_idx,
                    dataset_data=dataset_data,
                    dataset_base_dir=dataset_base_dir,
                    sam3_engine=sam3_engine,
                    siglip_engine=siglip_engine,
                    qwen_engine=qwen_engine,
                    nlp_parser=nlp_parser,
                    config=config,
                    num_candidate_images=args.num_candidate_images,
                    window_size=args.window_size,
                    random_seed=args.random_seed,
                    ic_examples=ic_examples,
                    use_logits_match=use_lm,
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

            # Mirror to canonical results_{benchmark_name}.jsonl in output_dir
            canonical_results_path = os.path.join(args.output_dir, f"{prefix}.jsonl")
            if os.path.abspath(canonical_results_path) != os.path.abspath(results_log_path):
                with open(canonical_results_path, "a", encoding="utf-8") as f_canon:
                    f_canon.write(json.dumps(rec) + "\n")

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

    # Ingest cumulative records from results file if resuming
    cumulative_records = []
    if os.path.exists(results_log_path):
        with open(results_log_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if line.strip():
                    try:
                        cumulative_records.append(json.loads(line))
                    except Exception:
                        pass

    eval_records_for_stats = cumulative_records if (args.resume and cumulative_records) else all_records

    # Test type category breakdown (e.g., V* direct_attributes vs relative_position)
    test_type_stats: Dict[str, Dict[str, int]] = {}
    for r in eval_records_for_stats:
        tt = r.get("test_type")
        if tt:
            if tt not in test_type_stats:
                test_type_stats[tt] = {"total": 0, "correct": 0}
            test_type_stats[tt]["total"] += 1
            if r.get("is_correct"):
                test_type_stats[tt]["correct"] += 1

    test_type_breakdown: Dict[str, Dict[str, Any]] = {}
    if test_type_stats:
        for tt, s in test_type_stats.items():
            acc = round((s["correct"] / s["total"] * 100.0), 2) if s["total"] > 0 else 0.0
            test_type_breakdown[tt] = {
                "total": s["total"],
                "correct": s["correct"],
                "accuracy": acc,
            }

    if args.resume and cumulative_records:
        cum_total = len(cumulative_records)
        cum_correct = sum(1 for r in cumulative_records if r.get("is_correct"))
        cum_acc = (cum_correct / cum_total * 100.0) if cum_total > 0 else 0.0
        cum_routes: Dict[str, int] = {}
        cum_latencies: List[float] = []
        for r in cumulative_records:
            r_sel = r.get("route_selected", "UNKNOWN")
            cum_routes[r_sel] = cum_routes.get(r_sel, 0) + 1
            if "latency_sec" in r and isinstance(r["latency_sec"], (int, float)):
                cum_latencies.append(float(r["latency_sec"]))
        cum_avg_lat = (sum(cum_latencies) / len(cum_latencies)) if cum_latencies else 0.0

        summary_data = {
            "timestamp": timestamp_str,
            "benchmark": benchmark_name,
            "resumed": True,
            "session_samples_evaluated": total_eval,
            "session_correct_samples": correct_count,
            "total_samples_evaluated": cum_total,
            "correct_samples": cum_correct,
            "accuracy_acc1": round(cum_acc, 2),
            "avg_latency_sec": round(cum_avg_lat, 2),
            "routes": cum_routes,
            "test_type_breakdown": test_type_breakdown,
            "results_log_path": results_log_path,
            "config_used_path": config_out_path,
        }
    else:
        summary_data = {
            "timestamp": timestamp_str,
            "benchmark": benchmark_name,
            "resumed": False,
            "total_samples_evaluated": total_eval,
            "correct_samples": correct_count,
            "accuracy_acc1": round(overall_acc, 2),
            "avg_latency_sec": round(avg_lat, 2),
            "routes": route_counter,
            "test_type_breakdown": test_type_breakdown,
            "results_log_path": results_log_path,
            "config_used_path": config_out_path,
        }

    sum_prefix = f"summary_{benchmark_name}_baseline" if is_direct_mode else f"summary_{benchmark_name}"
    summary_path = os.path.join(args.output_dir, f"{sum_prefix}_{timestamp_str}.json")
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary_data, f_sum, indent=2)

    canonical_summary_path = os.path.join(args.output_dir, f"{sum_prefix}.json")
    with open(canonical_summary_path, "w", encoding="utf-8") as f_canon_sum:
        json.dump(summary_data, f_canon_sum, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED [{benchmark_name.upper()}]: Evaluated {total_eval} new sample(s) this session.")
    if args.resume and cumulative_records:
        print(f"Cumulative Evaluated   : {len(cumulative_records)} total samples")
        print(f"Cumulative Acc (Acc@1) : {summary_data['correct_samples']}/{summary_data['total_samples_evaluated']} ({summary_data['accuracy_acc1']}%)")
        print(f"Cumulative Avg Latency : {summary_data['avg_latency_sec']}s per sample")
        print(f"Cumulative Routes      : {summary_data['routes']}")
    else:
        print(f"Overall Accuracy (Acc@1): {correct_count}/{total_eval} ({overall_acc:.2f}%)")
        print(f"Average Latency         : {avg_lat:.2f}s per sample")
        print(f"Route Breakdown         : {route_counter}")

    if test_type_breakdown:
        print("-" * 80)
        print("Subcategory / Test Type Breakdown:")
        for tt, s in test_type_breakdown.items():
            print(f"  - {tt:<22}: {s['correct']}/{s['total']} ({s['accuracy']}%)")
    print(f"Telemetry log written to: {results_log_path}")
    print(f"Summary JSON written to : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
