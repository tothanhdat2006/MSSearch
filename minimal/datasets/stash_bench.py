"""
STASH-Bench Dataset Loader for STASH-Search Minimal.
Loads nested multi-image candidate galleries (N in {1, 5, 10, 20} or crop)
with deterministic seed42 gallery ordering and source attribution metadata.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from .base import BenchmarkSample


def resolve_image_file(image_ref: Any, image_dir: str) -> str:
    """Finds full image path in image_dir or image_dir/pool across common extensions."""
    if isinstance(image_ref, dict):
        img_id = str(image_ref.get("path") or image_ref.get("name") or image_ref.get("image_id", ""))
    else:
        img_id = str(image_ref)

    clean_stem = os.path.splitext(os.path.basename(img_id))[0] if "." in img_id else img_id

    candidates = [
        os.path.join(image_dir, img_id),
        os.path.join(image_dir, "pool", img_id),
        os.path.join(image_dir, os.path.basename(img_id)),
        os.path.join(image_dir, "pool", os.path.basename(img_id)),
    ]
    for ext in ["", ".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"]:
        candidates.append(os.path.join(image_dir, f"{img_id}{ext}"))
        candidates.append(os.path.join(image_dir, f"{clean_stem}{ext}"))
        candidates.append(os.path.join(image_dir, "pool", f"{img_id}{ext}"))
        candidates.append(os.path.join(image_dir, "pool", f"{clean_stem}{ext}"))

    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)

    # Return primary candidate path even if not found locally (allows validation / dry-run)
    return os.path.abspath(candidates[0])


def load_stash_bench(
    dataset_path: str,
    image_dir: str,
    pool_mode: str = "N_5",
    gallery_order_path: Optional[str] = None
) -> List[BenchmarkSample]:
    """
    Loads STASH-Bench evaluation samples.

    Args:
        dataset_path: Path to annotations_part1234_merged.json or annotations directory.
        image_dir: Path to directory containing candidate images or pool subdirectory.
        pool_mode: Gallery size 'N_1', 'N_5', 'N_10', 'N_20', or 'crop'.
        gallery_order_path: Optional path to gallery_order_seed42.json.

    Returns:
        List of BenchmarkSample instances ready for evaluation.
    """
    if os.path.isdir(dataset_path):
        for name in ["annotations_part1234_merged.json", "STASHBench/annotations_part1234_merged.json", "annotations.json"]:
            cand = os.path.join(dataset_path, name)
            if os.path.isfile(cand):
                dataset_path = cand
                break

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"[STASH-Bench] Annotation file not found: '{dataset_path}'")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    all_samples = raw_data.get("samples", []) if isinstance(raw_data, dict) else raw_data

    # Load gallery order
    order_file = gallery_order_path
    if not order_file:
        order_file = os.path.join(os.path.dirname(os.path.abspath(dataset_path)), "gallery_order_seed42.json")
    gallery_data = {}
    if os.path.isfile(order_file):
        with open(order_file, "r", encoding="utf-8") as gf:
            gallery_data = json.load(gf).get("samples", {})

    canonical_pool = "N_5"
    if pool_mode.upper() in ["N_1", "N1", "1"]:
        canonical_pool = "N_1"
    elif pool_mode.upper() in ["N_5", "N5", "5"]:
        canonical_pool = "N_5"
    elif pool_mode.upper() in ["N_10", "N10", "10"]:
        canonical_pool = "N_10"
    elif pool_mode.upper() in ["N_20", "N20", "20"]:
        canonical_pool = "N_20"
    elif pool_mode.lower() in ["crop", "n_crop"]:
        canonical_pool = "crop"

    bench_samples: List[BenchmarkSample] = []

    for s_idx, s in enumerate(all_samples):
        sample_id = str(s.get("sample_id", f"{s_idx:03d}"))
        question = s.get("question") or s.get("query", {}).get("question", "")
        gt_answer = s.get("reference_answer") or s.get("ground_truth") or s.get("answer", "")
        category = s.get("question_type") or s.get("category", "Attribute")
        target_obj = s.get("target_object") or s.get("query", {}).get("target_object", "")

        gt_img = s.get("gt_image")
        sid_keys = [sample_id, f"{s_idx:03d}", str(s_idx)]
        g_entry = next((gallery_data[k] for k in sid_keys if k in gallery_data), None)

        if canonical_pool in ["N_1", "1"]:
            cand_paths = [resolve_image_file(gt_img, image_dir)] if gt_img else []
            gt_idx = 0
        elif canonical_pool == "crop":
            cand_paths = [resolve_image_file(gt_img, image_dir)] if gt_img else []
            gt_idx = 0
        else:
            k = canonical_pool
            if g_entry and k in g_entry and "image_ids" in g_entry[k]:
                ordered_ids = g_entry[k]["image_ids"]
                cand_paths = [resolve_image_file(iid, image_dir) for iid in ordered_ids]
                gt_idx = g_entry[k].get("gt_index", 0)
                # Ensure 0-indexed internal index
                if gt_idx is not None and gt_idx >= 1:
                    gt_idx = gt_idx - 1
            else:
                # Fallback if gallery_order_seed42 is missing
                distractors = s.get("other_images", {}).get(k, [])
                pool_raw = [gt_img] + distractors if gt_img else distractors
                cand_paths = [resolve_image_file(img, image_dir) for img in pool_raw]
                gt_idx = 0

        bench_samples.append(BenchmarkSample(
            sample_id=sample_id,
            question=question,
            candidate_image_paths=cand_paths,
            gt_answer=str(gt_answer),
            gt_image_idx=gt_idx,
            options_str=None,
            category=category,
            target_hint=target_obj if target_obj else None,
            metadata={
                "pool_mode": canonical_pool,
                "benchmark": "stash",
                "sample_index": s_idx
            }
        ))

    return bench_samples
