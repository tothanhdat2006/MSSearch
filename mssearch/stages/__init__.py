"""
Modular stages for the Dual-Path Multi-Source Visual Search Engine (MSSearch).
"""

from .stage1_parser import parse_question_to_active_objects
from .stage2_fast_path import run_tool_based_fast_path, FastPathResult
from .stage3_partitioning import partition_candidate_images, BoundedPatch, compute_spatial_entropy
from .stage4_grouping_gating import run_feature_grouping_and_gating
from .stage5_screening import run_resolution_screening, CandidateCrop
from .stage6_foveated_reader import run_foveated_vlm_reading, FinalSearchResult

__all__ = [
    "parse_question_to_active_objects",
    "run_tool_based_fast_path",
    "FastPathResult",
    "partition_candidate_images",
    "BoundedPatch",
    "compute_spatial_entropy",
    "run_feature_grouping_and_gating",
    "run_resolution_screening",
    "CandidateCrop",
    "run_foveated_vlm_reading",
    "FinalSearchResult",
]
