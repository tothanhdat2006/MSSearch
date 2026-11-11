"""
Unit test for Stage 1 Prompt Comparison tool.
Verifies dual-prompt invocation, matching logic, and discrepancy aggregation.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

for mod in [
    "torch", "torch.nn", "torch.nn.functional", "torch.cuda",
    "PIL", "PIL.Image", "spacy", "numpy", "cv2", "transformers"
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from experiments.our.scripts.compare_stage1_prompts import run_comparison


class TestCompareStage1Prompts(unittest.TestCase):

    def test_run_comparison_matching_and_mismatch(self):
        mock_vlm = MagicMock()
        mock_nlp = MagicMock()

        # Mock generate_text behavior based on prompt content
        def mock_generate_text(prompt, max_new_tokens=96):
            p_str = str(prompt)
            # If prompt has is_relational, simulate old prompt returning relational targets
            if "is_relational" in p_str:
                return '{"active_objects": ["red car", "white suv"], "target_entity": "red car", "question_type": "cross_object_relation", "is_relational": true}'
            # Otherwise new stripped prompt returning single target
            return '{"active_objects": ["red car"], "target_entity": "red car", "question_type": "single_object_attribute"}'

        mock_vlm.generate_text.side_effect = mock_generate_text

        questions = [
            {"sample_index": 0, "sample_id": "test_q1", "question": "Where is the red car relative to the white suv?"}
        ]

        report = run_comparison(mock_vlm, mock_nlp, questions, verbose=False)
        summary = report["summary"]

        self.assertEqual(summary["total_questions"], 1)
        self.assertEqual(summary["set_exact_match_count"], 0)
        self.assertEqual(summary["target_entity_match_count"], 1)
        self.assertEqual(len(report["mismatches"]), 1)

        diff = report["mismatches"][0]
        self.assertEqual(diff["old_active_objects"], ["red car", "white suv"])
        self.assertEqual(diff["new_active_objects"], ["red car"])
        self.assertTrue(diff["old_is_relational"])
        self.assertFalse(diff["new_is_relational"])
        self.assertTrue(diff["target_match"])
        self.assertFalse(diff["set_match"])


if __name__ == "__main__":
    unittest.main()
