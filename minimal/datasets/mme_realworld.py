"""
MME-RealWorld-Lite Dataset Loader for STASH-Search Minimal.
Loads multimodal perception and reasoning samples from MME-RealWorld-Lite
containing 'imgs/' sub-folder and annotation_mme-realworld-lite.json (or MME-RealWorld-Lite.json).
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from .base import BenchmarkSample

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]


def clean_mme_option_text(text: str) -> str:
    """Strips leading option label like '(A) ', 'A. ', 'A: ' without stripping plain words starting with A-K."""
    return re.sub(r"^(?:\([A-Ka-k]\)[\.\:\s]*|[A-Ka-k][\.\:]\s*)", "", str(text).strip()).strip()


def format_mme_options(options_raw: Any) -> str:
    """
    Standardizes multiple-choice options for MME-RealWorld-Lite into:
    A. text
    B. text
    ...
    """
    if isinstance(options_raw, list):
        formatted = []
        for i, opt in enumerate(options_raw):
            letter = OPTION_LETTERS[i] if i < len(OPTION_LETTERS) else f"Option_{i}"
            clean_opt = clean_mme_option_text(opt)
            formatted.append(f"{letter}. {clean_opt}")
        return "\n".join(formatted)
    elif isinstance(options_raw, dict):
        formatted = []
        for letter in OPTION_LETTERS:
            val = options_raw.get(letter) or options_raw.get(letter.lower())
            if val is not None and str(val).strip():
                clean_opt = clean_mme_option_text(val)
                formatted.append(f"{letter}. {clean_opt}")
        return "\n".join(formatted)
    elif isinstance(options_raw, str) and options_raw.strip():
        lines = [l.strip() for l in options_raw.strip().splitlines() if l.strip()]
        return "\n".join(lines)
    return ""


def load_mme_realworld(
    dataset_dir: str,
    annotation_path: Optional[str] = None
) -> List[BenchmarkSample]:
    """
    Loads MME-RealWorld-Lite evaluation samples from annotation_mme-realworld-lite.json or MME-RealWorld-Lite.json.

    Args:
        dataset_dir: Root directory containing 'imgs/' and annotation file.
        annotation_path: Optional explicit path to annotation JSON.

    Returns:
        List of BenchmarkSample instances.
    """
    anno_file = annotation_path
    if not anno_file or not os.path.isfile(anno_file):
        candidates = [
            os.path.join(dataset_dir, "annotation_mme-realworld-lite.json"),
            os.path.join(dataset_dir, "MME-RealWorld-Lite.json"),
            os.path.join(dataset_dir, "annotations.json"),
            os.path.join(dataset_dir, "mme_realworld_lite.json"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                anno_file = c
                break

    if not anno_file or not os.path.isfile(anno_file):
        raise FileNotFoundError(f"[MME-RealWorld-Lite] Annotation file not found in '{dataset_dir}'.")

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

        q_id = item.get("id") or item.get("Question_id") or item.get("question_id") or f"mme_{idx:04d}"
        sample_id = str(q_id)
        question = item.get("question") or item.get("Text") or ""
        
        task = item.get("Task") or item.get("task")
        subtask = item.get("Subtask") or item.get("subtask")
        cat = item.get("Category") or item.get("category")
        if task and subtask:
            category = f"{task}/{subtask}"
        elif cat:
            category = str(cat)
        else:
            category = "default"

        input_image_rel = item.get("input_image") or item.get("Image") or ""
        full_img_path = os.path.join(dataset_dir, input_image_rel)
        if not os.path.isfile(full_img_path):
            alt_path = os.path.join(dataset_dir, "imgs", os.path.basename(input_image_rel))
            if os.path.isfile(alt_path):
                full_img_path = alt_path

        options_raw = item.get("options") or item.get("Answer choices")
        options_str = format_mme_options(options_raw)

        gt_raw = item.get("Ground truth") or item.get("ground_truth") or item.get("answer") or item.get("gt_answer") or "A"
        gt_answer = str(gt_raw).strip().upper()
        # Clean up any surrounding brackets/quotes e.g. "(A)" -> "A"
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
                "benchmark": "mme-realworld-lite",
                "question_id": q_id,
                "task": item.get("Task"),
                "subtask": item.get("Subtask"),
                "question_type": item.get("Question Type"),
            }
        ))

    return bench_samples
