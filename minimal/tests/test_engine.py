"""
Integration test for STASHSearchEngine.
Runs the entire 4-stage pipeline end-to-end with mock backbones and synthetic images.
"""

import unittest
from PIL import Image

from minimal.config import STASHConfig
from minimal.engine import STASHSearchEngine
from minimal.models.base import BaseMLLM, BaseDualEncoder


class MockMLLM(BaseMLLM):
    def generate_text(self, prompt_or_messages, max_new_tokens=96) -> str:
        return '{"active_objects": ["waterfall"], "question_type": "single_object_attribute"}'

    def generate_response_batch(self, images, question, max_new_tokens=512):
        # Return answer for first image, refusal for second
        return ["beside the trees" for _ in images]


class MockDualEncoder(BaseDualEncoder):
    def score_crops(self, crops, text_queries, batch_size=32):
        # Slightly favor top crops
        return [0.75 for _ in crops]


class TestSTASHSearchEngine(unittest.TestCase):
    def test_end_to_end_search(self):
        # Create 2 synthetic gallery images (N=2)
        img1 = Image.new("RGB", (1000, 1000), color="blue")
        img2 = Image.new("RGB", (1000, 1000), color="green")
        gallery = [img1, img2]

        config = STASHConfig(
            k=5,
            min_plateau_size=512,
            max_plateau_size=1536,
            saliency_plateau_gain_threshold=0.04,
            verbose=False
        )

        engine = STASHSearchEngine(
            mllm=MockMLLM(),
            dual_encoder=MockDualEncoder(),
            config=config
        )

        result = engine.search(
            candidate_images=gallery,
            question="Where is the waterfall located?"
        )

        # Assert output contract
        self.assertIn("primary_answer", result)
        self.assertIn("primary_evidence_image", result)
        self.assertIn("active_objects", result)
        self.assertIn("entropy_pruned_leaves", result)
        self.assertIn("dedup_affected_crops", result)
        self.assertIn("telemetry", result)

        self.assertEqual(result["primary_answer"], "beside the trees")
        self.assertIn(result["primary_evidence_image"], [0, 1])
        self.assertEqual(result["active_objects"], ["waterfall"])
        self.assertTrue(result["num_candidate_patches"] > 0)
        self.assertTrue(result["num_inspected_crops"] > 0)

        # Telemetry & ablation tracking check
        self.assertIn("entropy_pruned_leaves", result["telemetry"])
        self.assertIn("dedup_affected_crops", result["telemetry"])
        self.assertIn("total_affected", result["dedup_affected_crops"])
        self.assertIn("ios_merged", result["dedup_affected_crops"])
        self.assertIn("nms_suppressed", result["dedup_affected_crops"])

        # Timings check
        timings = result["telemetry"]["timings_sec"]
        self.assertIn("stage1_target_extraction", timings)
        self.assertIn("stage2_quadtree_partitioning", timings)
        self.assertIn("stage3_visual_screening", timings)
        self.assertIn("stage4_consensus", timings)
        self.assertIn("total_latency_sec", timings)

    def test_search_with_options(self):
        img = Image.new("RGB", (1000, 1000), color="blue")
        config = STASHConfig(k=5, verbose=False)
        engine = STASHSearchEngine(mllm=MockMLLM(), dual_encoder=MockDualEncoder(), config=config)
        res = engine.search(
            candidate_images=[img],
            question="What is the color of the comb?",
            options_str="A. purple\nB. black\nC. red\nD. brown"
        )
        self.assertIn("primary_answer", res)
        self.assertEqual(res["primary_evidence_image"], 0)

    def test_search_with_inspection_prompt(self):
        img = Image.new("RGB", (1000, 1000), color="blue")
        config = STASHConfig(k=5, verbose=False)
        engine = STASHSearchEngine(mllm=MockMLLM(), dual_encoder=MockDualEncoder(), config=config)
        res = engine.search(
            candidate_images=[img],
            question="What is the color of the woman's jacket?",
            options_str="A. black\nB. red\nC. blue\nD. orange",
            inspection_prompt="What is the color of the woman's jacket? (A) black (B) red (C) blue (D) orange Answer with the option's letter from the given choices directly."
        )
        self.assertIn("primary_answer", res)
        self.assertEqual(res["primary_evidence_image"], 0)


if __name__ == "__main__":
    unittest.main()
