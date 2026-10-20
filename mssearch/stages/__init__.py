"""
Modular stages for the Dual-Path Multi-Source Visual Search Engine (MSSearch).
"""

from .stage1_parser import (
    parse_question_to_active_objects,
    parse_question_for_active_objects,
    ParsedQuestion,
    detect_relational_intent,
    reorder_anchor_first,
    format_options,
    format_question_with_options,
    parse_visual_query_structure
)
from .stage2_fast_path import (
    run_direct_global_assessment,
    DirectAnswerResult,
    run_tool_based_fast_path,
    run_fast_path_check,
    FastPathResult
)
from .stage3_partitioning import (
    partition_candidate_images,
    BoundedPatch,
    compute_spatial_entropy,
    partition_oversized_box,
    split_4_subpatches,
    partition_into_sam3_sgap,
)
from .stage4_grouping_gating import run_feature_grouping_and_gating, group_patches_by_cosine_similarity
from .stage5_screening import run_resolution_screening, CandidateCrop
from .stage6_foveated_reader import run_foveated_vlm_reading, FinalSearchResult, is_negative_or_unanswerable, deduplicate_answers

__all__ = [
    "run_direct_global_assessment",
    "DirectAnswerResult",
    "parse_question_to_active_objects",
    "parse_question_for_active_objects",
    "ParsedQuestion",
    "detect_relational_intent",
    "reorder_anchor_first",
    "format_options",
    "format_question_with_options",
    "parse_visual_query_structure",
    "run_tool_based_fast_path",
    "run_fast_path_check",
    "FastPathResult",
    "partition_candidate_images",
    "BoundedPatch",
    "compute_spatial_entropy",
    "partition_oversized_box",
    "split_4_subpatches",
    "partition_into_sam3_sgap",
    "run_feature_grouping_and_gating",
    "group_patches_by_cosine_similarity",
    "run_resolution_screening",
    "CandidateCrop",
    "run_foveated_vlm_reading",
    "FinalSearchResult",
    "is_negative_or_unanswerable",
    "deduplicate_answers",
]
