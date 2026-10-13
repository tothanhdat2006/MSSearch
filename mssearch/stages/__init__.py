"""
Modular stages for the Dual-Path Multi-Source Visual Search Engine (MSSearch).
"""

from .stage1_parser import parse_question_to_active_objects, parse_question_for_active_objects
from .stage2_fast_path import run_tool_based_fast_path, run_fast_path_check, FastPathResult
from .stage3_partitioning import partition_candidate_images, BoundedPatch, compute_spatial_entropy, partition_oversized_box, split_4_subpatches
from .stage4_grouping_gating import run_feature_grouping_and_gating, group_patches_by_cosine_similarity
from .stage5_screening import run_resolution_screening, CandidateCrop
from .stage6_foveated_reader import run_foveated_vlm_reading, FinalSearchResult, is_negative_or_unanswerable, deduplicate_answers

__all__ = [
    "parse_question_to_active_objects",
    "parse_question_for_active_objects",
    "run_tool_based_fast_path",
    "run_fast_path_check",
    "FastPathResult",
    "partition_candidate_images",
    "BoundedPatch",
    "compute_spatial_entropy",
    "partition_oversized_box",
    "split_4_subpatches",
    "run_feature_grouping_and_gating",
    "group_patches_by_cosine_similarity",
    "run_resolution_screening",
    "CandidateCrop",
    "run_foveated_vlm_reading",
    "FinalSearchResult",
    "is_negative_or_unanswerable",
    "deduplicate_answers",
]
