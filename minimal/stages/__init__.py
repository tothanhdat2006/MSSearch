"""
Stages exports for STASH-Search.
"""

from .target_extraction import extract_visual_targets
from .quadtree_partitioning import BoundedPatch, partition_gallery
from .visual_screening import CandidateExtraction, screen_and_inspect_crops
from .consensus import resolve_source_attributed_consensus

__all__ = [
    "extract_visual_targets",
    "BoundedPatch",
    "partition_gallery",
    "CandidateExtraction",
    "screen_and_inspect_crops",
    "resolve_source_attributed_consensus",
]
