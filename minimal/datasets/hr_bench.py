"""
HR-Bench (4K / 8K) Dataset Loader for STASH-Search Minimal.
Loads ultra-high-resolution benchmarks containing sub-folder 'image/'
and annotation_hr-bench_4k.json or annotation_hr-bench_8k.json.
"""

import os
import json
from typing import List, Dict, Any, Optional
from .base import BenchmarkSample


def normalize_resolution(resolution: str) -> str:
    """Normalizes resolution string to '4k' or '8k' strictly."""
    res_clean = resolution.lower().replace("-", "_").strip()
    if "4" in res_clean:
        return "4k"
    if "8" in res_clean:
        return "8k"
    raise ValueError(f"[HR-Bench] Invalid resolution '{resolution}'. Expected '4k' or '8k'.")


def load_hr_bench(
    dataset_dir: str,
    annotation_path: Optional[str] = None,
    resolution: str = "8k",
    cycle_idx: Optional[int] = None
) -> List[BenchmarkSample]:
    """
    Loads HR-Bench evaluation samples from annotation_hr-bench_{4k|8k}.json.

    Args:
        dataset_dir: Root directory containing 'image/' and annotation file.
        annotation_path: Optional explicit path to annotation JSON.
        resolution: '4k' or '8k'.
        cycle_idx: Optional option rotation index (0 to 3). If None, all 4 rotations
                   are generated per item to allow computing average rotation accuracy.

    Returns:
        List of BenchmarkSample instances.
    """
    res_tag = normalize_resolution(resolution)
    expected_anno_name = f"annotation_hr-bench_{res_tag}.json"

    anno_file: Optional[str] = None

    # 1. Explicit annotation_path
    if annotation_path and os.path.isfile(annotation_path):
        anno_file = annotation_path
    # 2. dataset_dir points directly to an annotation file
    elif os.path.isfile(dataset_dir):
        anno_file = dataset_dir
    # 3. Look up exact resolution annotation file in dataset_dir (strictly resolution-specific)
    elif os.path.isdir(dataset_dir):
        exact_path = os.path.join(dataset_dir, expected_anno_name)
        alt_path = os.path.join(dataset_dir, f"hr-bench_{res_tag}.json")
        if os.path.isfile(exact_path):
            anno_file = exact_path
        elif os.path.isfile(alt_path):
            anno_file = alt_path

    if not anno_file or not os.path.isfile(anno_file):
        raise FileNotFoundError(
            f"[HR-Bench] Exact annotation file '{expected_anno_name}' not found in '{dataset_dir}'."
        )

    base_img_dir = os.path.dirname(anno_file) if os.path.isfile(dataset_dir) else dataset_dir

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
        question = item.get("question", "")
        category = item.get("category", "single")
        input_image_rel = item.get("input_image", f"image/{item_idx}.jpg")

        full_img_path = os.path.join(base_img_dir, input_image_rel) if not os.path.isabs(input_image_rel) else input_image_rel
        if not os.path.isfile(full_img_path):
            alt_img = os.path.join(base_img_dir, "image", os.path.basename(input_image_rel))
            if os.path.isfile(alt_img):
                full_img_path = alt_img

        options_raw = item.get("options", [])
        answers_raw = item.get("answer", [])

        # Determine which rotations to generate
        if cycle_idx is not None:
            rotations = [cycle_idx]
        elif isinstance(options_raw, list) and len(options_raw) > 0:
            rotations = list(range(len(options_raw)))
        else:
            rotations = [0]

        for rot in rotations:
            # Options handling for current rotation
            if isinstance(options_raw, list) and len(options_raw) > rot:
                options_str = str(options_raw[rot]).strip()
            elif isinstance(options_raw, str):
                options_str = options_raw.strip()
            else:
                options_str = None

            # Ground truth answer letter for current rotation
            if isinstance(answers_raw, list) and len(answers_raw) > rot:
                gt_letter = str(answers_raw[rot]).strip().upper()
            elif isinstance(answers_raw, str):
                gt_letter = answers_raw.strip().upper()
            else:
                gt_letter = "A"

            sample_id = f"hrbench_{res_tag}_{item_idx:04d}_rot{rot}"

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
                    "cycle_idx": rot,
                    "raw_answers": answers_raw
                }
            ))

    return bench_samples
