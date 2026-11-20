"""
TreeBench Dataset Loader for STASH-Search Minimal.
Loads visual reasoning and perception benchmark samples from TreeBench
containing 'images/' sub-folder and annotation_treebench.json (or dataset_info.json).
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from .base import BenchmarkSample

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]


def clean_option_text(text: str) -> str:
    """Strips leading option label like '(A) ', 'A. ', 'A: ' without stripping plain words starting with A-K."""
    return re.sub(r"^(?:\([A-Ka-k]\)[\.\:\s]*|[A-Ka-k][\.\:]\s*)", "", str(text).strip()).strip()


def format_question_new(question, option_str):
    return question + " Options:\n" + option_str + "\nSelect the best answer to the above multiple-choice question based on the image. Respond with only the letter of the correct option.\nThe best answer is:"


def format_treebench_options(item: Dict[str, Any]) -> str:
    """
    Extracts or formats multiple-choice options string for TreeBench items.
    Checks 'multi-choice options', 'options', or individual keys 'A', 'B', etc.
    """
    raw_opts = item.get("multi-choice options") or item.get("options")
    if isinstance(raw_opts, str) and raw_opts.strip():
        # Standardize options lines
        lines = [line.strip() for line in raw_opts.strip().splitlines() if line.strip()]
        return "\n".join(lines)
    elif isinstance(raw_opts, list):
        formatted = []
        for i, opt in enumerate(raw_opts):
            letter = OPTION_LETTERS[i] if i < len(OPTION_LETTERS) else f"Option_{i}"
            clean_opt = clean_option_text(opt)
            formatted.append(f"{letter}. {clean_opt}")
        return "\n".join(formatted)
    elif isinstance(raw_opts, dict):
        formatted = []
        for letter in OPTION_LETTERS:
            val = raw_opts.get(letter) or raw_opts.get(letter.lower())
            if val is not None and str(val).strip():
                clean_opt = clean_option_text(val)
                formatted.append(f"{letter}. {clean_opt}")
        return "\n".join(formatted)

    # Reconstruct from individual letter keys
    formatted = []
    for letter in OPTION_LETTERS:
        val = item.get(letter)
        if val is not None and str(val).strip():
            clean_opt = clean_option_text(val)
            formatted.append(f"{letter}. {clean_opt}")

    return "\n".join(formatted) if formatted else ""


def load_treebench(
    dataset_dir: str,
    annotation_path: Optional[str] = None
) -> List[BenchmarkSample]:
    """
    Loads TreeBench evaluation samples from annotation_treebench.json or dataset_info.json.

    Args:
        dataset_dir: Root directory containing 'images/' and annotation file.
        annotation_path: Optional explicit path to annotation JSON.

    Returns:
        List of BenchmarkSample instances.
    """
    anno_file = annotation_path
    if not anno_file or not os.path.isfile(anno_file):
        candidates = [
            os.path.join(dataset_dir, "annotation_treebench.json"),
            os.path.join(dataset_dir, "dataset_info.json"),
            os.path.join(dataset_dir, "annotations.json"),
            os.path.join(dataset_dir, "treebench.json"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                anno_file = c
                break

    if not anno_file or not os.path.isfile(anno_file):
        raise FileNotFoundError(f"[TreeBench] Annotation file not found in '{dataset_dir}'.")

    with open(anno_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        raw_items = raw_data.get("samples") or raw_data.get("data") or list(raw_data.values())
    else:
        raw_items = raw_data

    bench_samples: List[BenchmarkSample] = []

    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue

        raw_id = item.get("id")
        if raw_id is not None:
            sample_id = str(raw_id)
        else:
            item_idx = item.get("index", idx)
            sample_id = f"treebench_{item_idx:04d}"
        item_idx = item.get("index", idx)
        question = item.get("question", "")
        category = item.get("category") or item.get("l2-category") or "default"

        input_image_rel = item.get("image") or item.get("input_image", "")
        if not input_image_rel:
            input_image_rel = f"images/img_{item_idx:04d}.jpg"

        full_img_path = os.path.join(dataset_dir, input_image_rel)
        if not os.path.isfile(full_img_path):
            alt_path = os.path.join(dataset_dir, "images", os.path.basename(input_image_rel))
            if os.path.isfile(alt_path):
                full_img_path = alt_path

        options_str = format_treebench_options(item)
        if options_str:
            inspection_prompt = format_question_new(question, options_str)
        else:
            inspection_prompt = question

        gt_raw = item.get("answer") or item.get("ground_truth") or item.get("Ground truth") or item.get("gt_answer") or "A"
        gt_answer = str(gt_raw).strip().upper()
        m = re.search(r"([A-Ka-k])", gt_answer)
        if m:
            gt_answer = m.group(1).upper()

        bench_samples.append(BenchmarkSample(
            sample_id=sample_id,
            question=question,
            candidate_image_paths=[os.path.abspath(full_img_path)],
            gt_answer=gt_answer,
            gt_image_idx=0,
            options_str=options_str if options_str else None,
            category=category,
            target_hint=None,
            metadata={
                "benchmark": "treebench",
                "index": item_idx,
                "l2_category": item.get("l2-category"),
                "target_instances": item.get("target_instances"),
            },
            inspection_prompt=inspection_prompt
        ))

    return bench_samples
