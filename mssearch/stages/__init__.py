"""
Modular stages for the Dual-Path Multi-Source Visual Search Engine (MSSearch).
"""

from .stage1_parser import (
    parse_question_to_active_objects,
    ParsedQuestion,
    format_options,
    format_question_with_options,
    parse_visual_query_structure
)
from .stage2_fast_path import (
    run_direct_global_assessment,
    DirectAnswerResult,
    run_tool_based_fast_path,
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
from .stage5_screening import run_resolution_screening, CandidateCrop, CandidatePool
from .stage6_foveated_reader import run_foveated_vlm_reading, FinalSearchResult, is_negative_or_unanswerable, deduplicate_answers
from .utils_profiler import StageVRAMTracker

__all__ = [
    "run_direct_global_assessment",
    "DirectAnswerResult",
    "parse_question_to_active_objects",
    "ParsedQuestion",
    "format_options",
    "format_question_with_options",
    "parse_visual_query_structure",
    "run_tool_based_fast_path",
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
    "CandidatePool",
    "run_foveated_vlm_reading",
    "FinalSearchResult",
    "is_negative_or_unanswerable",
    "deduplicate_answers",
    "StageVRAMTracker",
]
