"""
Stages exports for STASH-Search.
"""

from .target_extraction import (
    extract_visual_targets,
    extract_targets,
    build_few_shot_prompt,
    build_target_extraction_messages,
    format_prompt_for_text_completion,
)
from .quadtree_partitioning import BoundedPatch, partition_gallery
from .visual_screening import CandidateExtraction, screen_and_inspect_crops
from .consensus import resolve_source_attributed_consensus

__all__ = [
    "extract_visual_targets",
    "extract_targets",
    "build_few_shot_prompt",
    "build_target_extraction_messages",
    "format_prompt_for_text_completion",
    "BoundedPatch",
    "partition_gallery",
    "CandidateExtraction",
    "screen_and_inspect_crops",
    "resolve_source_attributed_consensus",
]
