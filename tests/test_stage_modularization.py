"""
Tests for Modular Stage Architectures and Utils Stage Exports.
Verifies backward compatibility and symbol accessibility across all stages.
"""

import unittest


class TestStageModularization(unittest.TestCase):
    def test_stage1_imports_and_utils(self):
        from mssearch.stages.stage1_parser import parse_question_to_active_objects, ParsedQuestion
        from mssearch.stages.utils_stage1 import ParsedQuestion as UParsedQuestion
        self.assertIs(ParsedQuestion, UParsedQuestion)
        self.assertTrue(callable(parse_question_to_active_objects))

    def test_stage2_imports_and_utils(self):
        from mssearch.stages.stage2_fast_path import (
            run_direct_global_assessment,
            DirectAnswerResult,
            run_tool_based_fast_path,
            FastPathResult,
        )
        from mssearch.stages.utils_stage2 import DirectAnswerResult as UDirectAnswerResult
        self.assertIs(DirectAnswerResult, UDirectAnswerResult)
        self.assertTrue(callable(run_direct_global_assessment))
        self.assertTrue(callable(run_tool_based_fast_path))

    def test_stage3_imports_and_utils(self):
        from mssearch.stages.stage3_partitioning import (
            partition_candidate_images,
            BoundedPatch,
            compute_spatial_entropy,
            partition_oversized_box,
            split_4_subpatches,
            partition_into_sam3_sgap,
        )
        from mssearch.stages.utils_stage3 import BoundedPatch as UBoundedPatch
        self.assertIs(BoundedPatch, UBoundedPatch)
        self.assertTrue(callable(partition_candidate_images))
        self.assertTrue(callable(compute_spatial_entropy))
        self.assertTrue(callable(partition_oversized_box))
        self.assertTrue(callable(split_4_subpatches))
        self.assertTrue(callable(partition_into_sam3_sgap))

    def test_stage4_imports_and_utils(self):
        from mssearch.stages.stage4_grouping_gating import (
            run_feature_grouping_and_gating,
            group_patches_by_cosine_similarity,
        )
        from mssearch.stages.utils_stage4 import group_patches_by_cosine_similarity as UGroup
        self.assertIs(group_patches_by_cosine_similarity, UGroup)
        self.assertTrue(callable(run_feature_grouping_and_gating))

    def test_stage5_imports_and_utils(self):
        from mssearch.stages.stage5_screening import (
            run_resolution_screening,
            CandidateCrop,
            CandidatePool,
            perform_atomic_sibling_refusion,
            partition_continuous_enclosure_subcanvases,
        )
        from mssearch.stages.utils_stage5 import (
            CandidateCrop as UCandidateCrop,
            CandidatePool as UCandidatePool,
        )
        self.assertIs(CandidateCrop, UCandidateCrop)
        self.assertIs(CandidatePool, UCandidatePool)
        self.assertTrue(callable(run_resolution_screening))
        self.assertTrue(callable(perform_atomic_sibling_refusion))
        self.assertTrue(callable(partition_continuous_enclosure_subcanvases))

    def test_stage6_imports_and_utils(self):
        from mssearch.stages.stage6_foveated_reader import (
            run_foveated_vlm_reading,
            FinalSearchResult,
            is_negative_or_unanswerable,
            deduplicate_answers,
            split_attribute_answer,
            extract_final_answer,
        )
        from mssearch.stages.utils_stage6 import (
            FinalSearchResult as UFinalSearchResult,
            extract_final_answer as UExtract,
        )
        self.assertIs(FinalSearchResult, UFinalSearchResult)
        self.assertIs(extract_final_answer, UExtract)
        self.assertTrue(callable(run_foveated_vlm_reading))
        self.assertTrue(callable(is_negative_or_unanswerable))
        self.assertTrue(callable(deduplicate_answers))
        self.assertTrue(callable(split_attribute_answer))


if __name__ == "__main__":
    unittest.main()
