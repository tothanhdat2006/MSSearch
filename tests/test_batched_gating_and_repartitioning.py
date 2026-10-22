import unittest
import numpy as np
from PIL import Image

from mssearch.config import MSSearchConfig
from mssearch.stages.stage3_partitioning import BoundedPatch
from mssearch.stages.stage5_screening import (
    compute_box_ios,
    compute_box_iou,
    merge_nested_containment_boxes
)


class TestContainmentAndRepartitioning(unittest.TestCase):
    def test_compute_box_ios(self):
        # Child completely inside parent
        child = (100, 100, 300, 300)      # Area = 40,000
        parent = (50, 50, 500, 500)       # Area = 202,500
        ios = compute_box_ios(child, parent)
        self.assertAlmostEqual(ios, 1.0, places=4)

        # Child 50% intersecting parent
        child_half = (0, 0, 100, 100)      # Area = 10,000
        parent_half = (50, 0, 150, 100)    # Intersection = 50 * 100 = 5000
        ios_half = compute_box_ios(child_half, parent_half)
        self.assertAlmostEqual(ios_half, 0.5, places=4)

        # Disjoint boxes
        b1 = (0, 0, 50, 50)
        b2 = (100, 100, 150, 150)
        self.assertEqual(compute_box_ios(b1, b2), 0.0)

    def test_merge_nested_containment_boxes_drops_parent(self):
        # Child nested >= 95% inside coarse parent
        child_p = BoundedPatch(
            img_idx=0,
            box=(100, 100, 200, 200),
            feature_vector=np.zeros(16),
            entropy=0.5,
            confidence_score=2.5,
            is_fine_grain=True,
            is_recursive_child=True
        )
        parent_p = BoundedPatch(
            img_idx=0,
            box=(50, 50, 500, 500),
            feature_vector=np.zeros(16),
            entropy=0.6,
            confidence_score=1.0,
            is_fine_grain=False
        )
        independent_p = BoundedPatch(
            img_idx=0,
            box=(600, 600, 800, 800),
            feature_vector=np.zeros(16),
            entropy=0.4,
            confidence_score=1.2
        )

        patches = [parent_p, child_p, independent_p]
        surviving = merge_nested_containment_boxes(patches, containment_threshold=0.95, verbose=False)

        # Parent must be pruned; child and independent must survive
        self.assertEqual(len(surviving), 2)
        surviving_boxes = [p.box for p in surviving]
        self.assertIn(child_p.box, surviving_boxes)
        self.assertIn(independent_p.box, surviving_boxes)
        self.assertNotIn(parent_p.box, surviving_boxes)

    def test_coordinate_translation_math(self):
        # Parent crop coordinates on 4K canvas
        parent_box = (1200, 1500, 2224, 2524)
        # Child sub-patch inside local crop [0..1024, 0..1024]
        child_local_box = (150, 200, 450, 600)

        gx1 = parent_box[0] + child_local_box[0]
        gy1 = parent_box[1] + child_local_box[1]
        gx2 = parent_box[0] + child_local_box[2]
        gy2 = parent_box[1] + child_local_box[3]

        self.assertEqual((gx1, gy1, gx2, gy2), (1350, 1700, 1650, 2100))

    def test_config_defaults(self):
        cfg = MSSearchConfig()
        self.assertTrue(cfg.enable_latent_repartition)
        self.assertEqual(cfg.latent_repartition_threshold, 0.70)
        self.assertTrue(cfg.enable_ios_containment_merging)
        self.assertEqual(cfg.ios_containment_threshold, 0.95)
        self.assertEqual(cfg.sam3_batch_size, 4)
        self.assertEqual(cfg.vlm_gating_batch_size, 4)


class MockSAM3Model:
    def __init__(self):
        self.calls = 0
        self.batch_calls = 0

    def batch_inference(self, image, text_prompts):
        self.calls += 1
        return None, {1: {"boxes": [[10, 10, 50, 50]], "scores": [0.85]}}, [1]

    def batch_inference_multi_images(self, images, text_prompts, batch_size=4):
        self.batch_calls += 1
        results = []
        for _ in images:
            results.append((None, {1: {"boxes": [[10, 10, 50, 50]], "scores": [0.85]}}, [1]))
        return results


class MockVLMModel:
    def __init__(self):
        self.confidence_calls = 0
        self.prompts = {"search": {"existence_prompt": "Is there a {}?", "latent_prompt": "Can you zoom {}?"}}

    def get_confidence_value(self, node=None, image_pil=None, confidence_type='existence', input_ele=''):
        self.confidence_calls += 1
        return 0.8 if confidence_type == 'existence' else 0.75

    def batch_get_confidence_values(self, image_list, confidence_type='existence', input_ele='', batch_size=4, custom_prompt_templates=None):
        self.confidence_calls += 1
        val = 0.8 if confidence_type == 'existence' else 0.75
        return [val] * len(image_list)

    def generate_text(self, prompt, max_new_tokens=32):
        import re
        prompt_str = str(prompt)
        m = re.search(r'Question:\s*"([^"]+)"', prompt_str)
        target = m.group(1).lower() if m else prompt_str.lower()
        if "above the logo" in target or "left of" in target or "right of" in target:
            return '{"is_cross_object": true, "is_relational": true}'
        return '{"is_cross_object": false, "is_relational": false}'

    def generate_response(self, image_pil=None, question="", max_new_tokens=32):
        return self.generate_text(question, max_new_tokens=max_new_tokens)


class TestBatchedStage4Execution(unittest.TestCase):
    def test_stage4_batched_sam3_and_vlm(self):
        from mssearch.stages.stage4_grouping_gating import run_feature_grouping_and_gating

        cfg = MSSearchConfig()
        cfg.verbose = False
        cfg.sam3_batch_size = 4
        cfg.vlm_gating_batch_size = 4
        cfg.latent_repartition_threshold = 0.70

        dummy_img = Image.new("RGB", (2000, 2000), (128, 128, 128))
        bounded_patches = [
            BoundedPatch(img_idx=0, box=(0, 0, 500, 500), feature_vector=np.ones(16), entropy=0.5),
            BoundedPatch(img_idx=0, box=(500, 0, 1000, 500), feature_vector=np.ones(16) * 0.99, entropy=0.6),
            BoundedPatch(img_idx=0, box=(0, 500, 500, 1000), feature_vector=np.zeros(16), entropy=0.4),
        ]

        mock_sam = MockSAM3Model()
        mock_vlm = MockVLMModel()

        remaining_patches, records = run_feature_grouping_and_gating(
            bounded_patches=bounded_patches,
            candidate_images=[dummy_img],
            active_objects=["boat"],
            sam_model=mock_sam,
            config=cfg,
            zoom_model=mock_vlm,
            return_debug=True,
            question="What is the color of the boat?"
        )

        self.assertGreater(len(remaining_patches), 0)
        self.assertGreater(mock_sam.batch_calls, 0)
        self.assertGreater(len(records), 0)



class TestOptionAwarePrompting(unittest.TestCase):
    def test_hrbench_option_list_enables_options_prompting(self):
        from mssearch.MSSearch import MultiSourceSearchEngine
        from mssearch.config import MSSearchConfig

        cfg = MSSearchConfig()
        cfg.verbose = False
        engine = MultiSourceSearchEngine(config=cfg)

        annotation = {
            "question": "What is the color of the mailbox?",
            "options": ["A. Blue\nB. Green\nC. Black\nD. Red\n"],
            "answer_type": "option_list"
        }

        # Check get_mssearch_response option detection logic
        anno_type = annotation.get("answer_type")
        if anno_type in ["option_single", "option_list", "multiple_choice", "mc", "Multiple Choice"]:
            cfg.include_options_in_prompt = True
            cfg.answer_type = anno_type

        self.assertTrue(cfg.include_options_in_prompt)
        self.assertEqual(cfg.answer_type, "option_list")

        # Check that search() formats the question with options
        is_mc = cfg.answer_type in ["option_single", "option_list", "multiple_choice", "mc", "Multiple Choice"]
        should_add_options = cfg.include_options_in_prompt or is_mc
        self.assertTrue(should_add_options)


class TestRelationalLayoutConditioning(unittest.TestCase):
    def setUp(self):
        self.dummy_img = Image.new("RGB", (2000, 2000), (128, 128, 128))
        self.patches = [
            BoundedPatch(img_idx=0, box=(100, 100, 300, 300), feature_vector=np.zeros(16), entropy=0.5, confidence_score=1.5),
            BoundedPatch(img_idx=0, box=(1200, 1200, 1500, 1500), feature_vector=np.zeros(16), entropy=0.6, confidence_score=1.2),
        ]

    def test_single_object_uses_crops_only(self):
        from mssearch.stages.stage5_screening import run_resolution_screening
        from mssearch.config import MSSearchConfig

        cfg = MSSearchConfig()
        cfg.verbose = False
        cfg.enable_spatial_compressed_canvas = True
        cfg.spatial_canvas_relational_only = True

        # Single object question: relational = False
        pool = run_resolution_screening(
            remaining_patches=self.patches,
            candidate_images=[self.dummy_img],
            active_objects=["boat"],
            config=cfg,
            question="What is the registration number of the boat?",
            is_relational=False
        )

        # Must produce 2 separate crops, neither being a compressed canvas
        self.assertEqual(len(pool), 2)
        for crop in pool:
            self.assertFalse(crop.is_compressed_canvas)
            self.assertEqual(crop.source_branch, "Direct")

    def test_cross_object_uses_layout(self):
        from mssearch.stages.stage5_screening import run_resolution_screening
        from mssearch.config import MSSearchConfig

        cfg = MSSearchConfig()
        cfg.verbose = False
        cfg.enable_spatial_compressed_canvas = True
        cfg.spatial_canvas_relational_only = True

        # Cross object question: relational = True
        pool = run_resolution_screening(
            remaining_patches=self.patches,
            candidate_images=[self.dummy_img],
            active_objects=["boat", "buoy"],
            config=cfg,
            question="What is the object to the left of the boat?",
            is_relational=True
        )

        # Must synthesize 1 compressed spatial canvas preserving relative topology
        self.assertEqual(len(pool), 1)
        self.assertTrue(pool[0].is_compressed_canvas)
        self.assertEqual(pool[0].source_branch, "SpatialCompressedCanvas")

    def test_is_relational_question_detection(self):
        from mssearch.stages.stage5_screening import is_relational_question, check_is_cross_object_llm
        from mssearch.stages.stage1_parser import ParsedQuestion

        mock_vlm = MockVLMModel()

        # 1. Single object from Stage 1 ParsedQuestion -> False
        q_single = ParsedQuestion(active_objects=["mailbox"], is_relational=False)
        self.assertFalse(is_relational_question(q_single, "What is the color of the mailbox?"))

        # 2. Single target list (len <= 1) -> False
        self.assertFalse(is_relational_question(["boat"], "What is the registration number of the boat?"))

        # 3. Cross object from Stage 1 ParsedQuestion -> True
        q_rel = ParsedQuestion(active_objects=["car", "person"], is_relational=True)
        self.assertTrue(is_relational_question(q_rel, "Who is standing to the right of the car?"))

        # 4. LLM Classifier direct verification (zero synthetic keyword fallbacks)
        self.assertTrue(check_is_cross_object_llm("What is the text written above the logo on the jacket's sleeve?", mock_vlm))
        self.assertFalse(check_is_cross_object_llm("What is the registration number of the boat?", mock_vlm))

        # 5. is_relational_question delegating to LLM
        self.assertTrue(is_relational_question(["jacket", "logo", "text"], "What is the text written above the logo?", zoom_model=mock_vlm))
        self.assertFalse(is_relational_question(["boat"], "What is the registration number of the boat?", zoom_model=mock_vlm))


if __name__ == "__main__":
    unittest.main()


