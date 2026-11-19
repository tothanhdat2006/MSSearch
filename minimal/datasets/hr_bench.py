"""
HR-Bench (4K / 8K) Dataset Loader for STASH-Search Minimal.
Loads ultra-high-resolution benchmarks containing sub-folder 'image/'
and annotation_hr-bench_4k.json or annotation_hr-bench_8k.json.
"""

import os
import json
from typing import List, Dict, Any, Optional
from .base import BenchmarkSample


def load_hr_bench(
    dataset_dir: str,
    annotation_path: Optional[str] = None,
    resolution: str = "8k",
    cycle_idx: int = 0
) -> List[BenchmarkSample]:
    """
    Loads HR-Bench evaluation samples from annotation_hr-bench_{4k|8k}.json.

    Args:
        dataset_dir: Root directory containing 'image/' and annotation file.
        annotation_path: Optional explicit path to annotation JSON.
        resolution: '4k' or '8k'.
        cycle_idx: Option rotation index (0 to 3, default: 0).

    Returns:
        List of BenchmarkSample instances.
    """
    anno_file = annotation_path
    if not anno_file or not os.path.isfile(anno_file):
        res_tag = resolution.lower().strip()
        candidates = [
            os.path.join(dataset_dir, f"annotation_hr-bench_{res_tag}.json"),
            os.path.join(dataset_dir, f"hr-bench_{res_tag}.json"),
            os.path.join(dataset_dir, "annotation_hr-bench_8k.json"),
            os.path.join(dataset_dir, "annotation_hr-bench_4k.json"),
            os.path.join(dataset_dir, "annotations.json"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                anno_file = c
                break

    if not anno_file or not os.path.isfile(anno_file):
        raise FileNotFoundError(f"[HR-Bench] Annotation file not found in '{dataset_dir}'.")

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

        item_idx = item.get("index", idx)
        sample_id = f"hrbench_{resolution}_{item_idx:04d}"
        question = item.get("question", "")
        category = item.get("category", "single")
        input_image_rel = item.get("input_image", f"image/{item_idx}.jpg")

        full_img_path = os.path.join(dataset_dir, input_image_rel) if not os.path.isabs(input_image_rel) else input_image_rel

        # Options handling: extract rotation string for cycle_idx
        options_raw = item.get("options", [])
        if isinstance(options_raw, list) and len(options_raw) > cycle_idx:
            options_str = str(options_raw[cycle_idx]).strip()
        elif isinstance(options_raw, str):
            options_str = options_raw.strip()
        else:
            options_str = None

        # Ground truth answer letter: extract letter for cycle_idx
        answers_raw = item.get("answer", [])
        if isinstance(answers_raw, list) and len(answers_raw) > cycle_idx:
            gt_letter = str(answers_raw[cycle_idx]).strip().upper()
        elif isinstance(answers_raw, str):
            gt_letter = answers_raw.strip().upper()
        else:
            gt_letter = "A"

        bench_samples.append(BenchmarkSample(
            sample_id=sample_id,
            question=question,
            candidate_image_paths=[os.path.abspath(full_img_path)],
            gt_answer=gt_letter,
            gt_image_idx=0,
            options_str=options_str,
            category=category,
            target_hint=None,
            metadata={
                "benchmark": f"hrbench_{resolution}",
                "index": item_idx,
                "input_image": input_image_rel,
                "category": category,
                "cycle_idx": cycle_idx,
                "raw_answers": answers_raw
            }
        ))

    return bench_samples
