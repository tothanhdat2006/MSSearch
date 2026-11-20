"""
V*Bench (Vstar) Dataset Loader for STASH-Search Minimal.
Loads single-image high-resolution visual search benchmarks containing
sub-folders 'direct_attributes/' and 'relative_position/' with annotation_vstar.json.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from .base import BenchmarkSample

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def format_options(options_list: List[str]) -> Tuple[str, Dict[str, str]]:
    """Formats raw option strings into letter-labeled candidate options."""
    formatted_lines = []
    letter_to_text: Dict[str, str] = {}
    for i, opt in enumerate(options_list):
        letter = OPTION_LETTERS[i] if i < len(OPTION_LETTERS) else f"Option_{i}"
        clean_opt = str(opt).strip()
        formatted_lines.append(f"{letter}. {clean_opt}")
        letter_to_text[letter] = clean_opt
    return "\n".join(formatted_lines), letter_to_text


def load_vstar(
    dataset_dir: str,
    annotation_path: Optional[str] = None
) -> List[BenchmarkSample]:
    """
    Loads V*Bench evaluation samples from annotation_vstar.json.

    Args:
        dataset_dir: Root directory containing 'direct_attributes/', 'relative_position/',
                     and 'annotation_vstar.json'.
        annotation_path: Optional explicit path to annotation_vstar.json.

    Returns:
        List of BenchmarkSample instances.
    """
    anno_file = annotation_path
    if not anno_file or not os.path.isfile(anno_file):
        candidates = [
            os.path.join(dataset_dir, "annotation_vstar_updated.json"),
            os.path.join(dataset_dir, "annotation_vstar.json"),
            os.path.join(dataset_dir, "annotations.json"),
            os.path.join(dataset_dir, "vstar.json")
        ]
        for c in candidates:
            if os.path.isfile(c):
                anno_file = c
                break

    if not anno_file or not os.path.isfile(anno_file):
        raise FileNotFoundError(f"[V*Bench] Neither annotation_vstar_updated.json nor annotation_vstar.json found in '{dataset_dir}'.")

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

        sample_id = str(item.get("id") or item.get("sample_idx") or f"vstar_{idx:04d}")
        question = item.get("question", "")
        test_type = item.get("test_type") or item.get("category") or "direct_attributes"
        input_image_rel = item.get("input_image") or item.get("image", "")

        full_img_path = os.path.join(dataset_dir, input_image_rel) if not os.path.isabs(input_image_rel) else input_image_rel

        # In annotation_vstar_updated.json, options are shuffled and label is explicitly provided (A, B, C, D)
        # In legacy annotation_vstar.json, option 0 is canonically the ground truth target option ("A")
        options_list = item.get("options", [])
        if options_list:
            options_str, letter_map = format_options(options_list)
        else:
            options_str = None
            letter_map = {}

        raw_ans = item.get("label") if item.get("label") is not None else item.get("answer")
        if raw_ans is not None:
            raw_ans_str = str(raw_ans).strip().upper()
            m = re.search(r"([A-Da-d])", raw_ans_str)
            if m:
                gt_letter = m.group(1).upper()
            else:
                try:
                    gt_letter = OPTION_LETTERS[int(raw_ans)]
                except Exception:
                    gt_letter = "A"
        else:
            gt_letter = "A"

        gt_text = letter_map.get(gt_letter, options_list[0] if options_list else "")

        bench_samples.append(BenchmarkSample(
            sample_id=sample_id,
            question=question,
            candidate_image_paths=[os.path.abspath(full_img_path)],
            gt_answer=gt_letter,
            gt_image_idx=0,
            options_str=options_str,
            category=test_type,
            metadata={
                "benchmark": "vstar",
                "test_type": test_type,
                "input_image": input_image_rel,
                "gt_text": gt_text,
                "options": options_list
            }
        ))

    return bench_samples
