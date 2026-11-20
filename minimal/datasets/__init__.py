"""
Benchmark dataset loaders for STASH-Search Minimal.
Supports STASH-Bench, V*Bench (Vstar), HR-Bench 4K/8K, TreeBench, and MME-RealWorld-Lite.
"""

from typing import List, Optional
from .base import BenchmarkSample
from .stash_bench import load_stash_bench
from .vstar import load_vstar
from .hr_bench import load_hr_bench
from .treebench import load_treebench, format_question_new
from .mme_realworld import load_mme_realworld, format_question_multichoice

__all__ = [
    "BenchmarkSample",
    "load_benchmark_dataset",
    "load_stash_bench",
    "load_vstar",
    "load_hr_bench",
    "load_treebench",
    "load_mme_realworld",
    "format_question_new",
    "format_question_multichoice",
]


def load_benchmark_dataset(
    benchmark: str,
    dataset_dir_or_path: str,
    image_dir: Optional[str] = None,
    pool_mode: str = "N_5",
    gallery_order_path: Optional[str] = None,
    **kwargs
) -> List[BenchmarkSample]:
    """
    Unified entrypoint to load benchmark datasets.

    Args:
        benchmark: Name of the benchmark ('stash', 'vstar', 'hrbench4k', 'hrbench8k', 'treebench', 'mme-realworld-lite').
        dataset_dir_or_path: Root folder of the dataset or path to annotation JSON.
        image_dir: Directory containing images (for STASH-Bench).
        pool_mode: Multi-image pool size ('N_1', 'N_5', 'N_10', 'N_20') for STASH-Bench.
        gallery_order_path: Optional path to gallery_order_seed42.json for STASH-Bench.

    Returns:
        List of BenchmarkSample instances.
    """
    b = str(benchmark).lower().strip()

    if b == "stash":
        img_dir = image_dir if image_dir else dataset_dir_or_path
        return load_stash_bench(
            dataset_path=dataset_dir_or_path,
            image_dir=img_dir,
            pool_mode=pool_mode,
            gallery_order_path=gallery_order_path
        )
    elif b == "vstar":
        return load_vstar(
            dataset_dir=dataset_dir_or_path,
            annotation_path=kwargs.get("annotation_path")
        )
    elif b == "hrbench4k":
        return load_hr_bench(
            dataset_dir=dataset_dir_or_path,
            annotation_path=kwargs.get("annotation_path"),
            resolution="4k",
            cycle_idx=kwargs.get("cycle_idx", None)
        )
    elif b == "hrbench8k":
        return load_hr_bench(
            dataset_dir=dataset_dir_or_path,
            annotation_path=kwargs.get("annotation_path"),
            resolution="8k",
            cycle_idx=kwargs.get("cycle_idx", None)
        )
    elif b == "treebench":
        return load_treebench(
            dataset_dir=dataset_dir_or_path,
            annotation_path=kwargs.get("annotation_path")
        )
    elif b == "mme-realworld-lite":
        return load_mme_realworld(
            dataset_dir=dataset_dir_or_path,
            annotation_path=kwargs.get("annotation_path")
        )
    else:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. Supported: 'stash', 'vstar', 'hrbench4k', 'hrbench8k', 'treebench', 'mme-realworld-lite'."
        )
