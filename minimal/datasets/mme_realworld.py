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


def format_question_multichoice(question, options):
    ret = question
    for o in options:
        ret += '\n'
        ret += o
    # This prompt is copied from the original paper of MME-RealWorld
    ret += '\nSelect the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option.\nThe best answer is:'
    return ret


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
        sample_id = f"{q_id}_{idx:04d}"
        question = item.get("question") if item.get("question") is not None else item.get("Text", "")
        
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

        # Do not trim and preprocess the question and options
        options_raw = item.get("options") if item.get("options") is not None else item.get("Answer choices")
        if isinstance(options_raw, list):
            options_list = [str(o) for o in options_raw]
        elif isinstance(options_raw, dict):
            options_list = [str(v) for v in options_raw.values()]
        elif isinstance(options_raw, str):
            options_list = [line for line in options_raw.splitlines() if line]
        else:
            options_list = []

        inspection_prompt = format_question_multichoice(question, options_list)
        options_str = "\n".join(options_list) if options_list else None

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
            options_str=options_str,
            category=category,
            target_hint=None,
            metadata={
                "benchmark": "mme-realworld-lite",
                "question_id": q_id,
                "task": item.get("Task"),
                "subtask": item.get("Subtask"),
                "question_type": item.get("Question Type"),
                "raw_options": options_list,
            },
            inspection_prompt=inspection_prompt
        ))

    return bench_samples
