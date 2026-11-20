"""
Baseline inference package for STASH-Search Minimal.
Direct MLLM inference across V*Bench, HR-Bench 4K/8K, TreeBench, and MME-RealWorld-Lite.
"""

from .run_inference import (
    MODELS,
    BENCHMARKS,
    BenchmarkDataset,
    build_prompt_text,
    family,
    render_chat,
    load_image,
    main,
)

__all__ = [
    "MODELS",
    "BENCHMARKS",
    "BenchmarkDataset",
    "build_prompt_text",
    "family",
    "render_chat",
    "load_image",
    "main",
]
