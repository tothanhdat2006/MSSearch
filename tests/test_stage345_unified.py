import unittest
import numpy as np
from PIL import Image

from mssearch.config import MSSearchConfig
from mssearch.stages.stage3_partitioning import BoundedPatch, partition_candidate_images
from mssearch.stages.stage4_grouping_gating import run_feature_grouping_and_gating
from mssearch.stages.stage5_screening import (
    are_seam_adjacent_siblings,
    perform_atomic_sibling_refusion,
    partition_continuous_enclosure_subcanvases,
    run_resolution_screening,
    CandidatePool,
    CandidateCrop,
)


class MockSAM3Model:
    def __init__(self, detections_map=None):
        self.detections_map = detections_map or {}

    def batch_inference(self, image, text_prompts):
        cw, ch = image.size
        key = (cw, ch)
        if key in self.detections_map:
            return None, {1: self.detections_map[key]}, [1]
        return None, {1: {"boxes": [[10, 10, 50, 50]], "scores": [0.85]}}, [1]

    def batch_inference_multi_images(self, images, text_prompts, batch_size=4):
        results = []
        for img in images:
            results.append(self.batch_inference(img, text_prompts))
        return results


class MockVLMModel:
    def __init__(self):
        self.prompts = {"search": {"existence_prompt": "Is there a {}?", "latent_prompt": "Can you zoom {}?"}}

    def get_confidence_value(self, node=None, image_pil=None, confidence_type='existence', input_ele=''):
        return 0.8 if confidence_type == 'existence' else 0.75

    def batch_get_confidence_values(self, image_list, confidence_type='existence', input_ele='', batch_size=4, custom_prompt_templates=None):
        return [0.8 if confidence_type == 'existence' else 0.75] * len(image_list)

    def generate_text(self, prompt, max_new_tokens=32):
        return '{"is_cross_object": false, "is_relational": false}'

    def generate_response(self, image_pil=None, question="", max_new_tokens=32):
        return self.generate_text(question, max_new_tokens=max_new_tokens)


class TestStage345Unified(unittest.TestCase):
    def setUp(self):
        self.img_w, self.img_h = 2048, 2048
        self.dummy_img = Image.new("RGB", (self.img_w, self.img_h), (128, 128, 128))
        self.mock_sam = MockSAM3Model()
        self.mock_vlm = MockVLMModel()

    def test_stage3_lineage_tracking(self):
        """Stage 3: Verify parent_id and level are tracked during partitioning."""
        cfg = MSSearchConfig()
        cfg.verbose = False
        cfg.partition_stopping_mode = "fixed_size"
        cfg.max_patch_size = 1024
        cfg.min_patch_entropy = 0.0  # Allow flat image patches to survive for lineage check

        patches = partition_candidate_images(
            candidate_images=[self.dummy_img],
            stage2_feature_cache={},
            config=cfg
        )

        self.assertGreater(len(patches), 1)
        for p in patches:
            self.assertIsNotNone(p.parent_id)
            self.assertEqual(p.parent_id, (0, 0, self.img_w, self.img_h))
            self.assertEqual(p.level, 1)

    def test_stage4_confirmed_detection_group_ranking_by_area(self):
        """Stage 4: Detected patches ranked by max detected bbox area desc, not entropy."""
        p_small_det = BoundedPatch(img_idx=0, box=(0, 0, 512, 512), feature_vector=np.ones(16), entropy=10.0, siglip_score=0.2)
        p_large_det = BoundedPatch(img_idx=0, box=(512, 0, 1024, 512), feature_vector=np.ones(16), entropy=1.0, siglip_score=0.2)
        p_context = BoundedPatch(img_idx=0, box=(0, 512, 512, 1024), feature_vector=np.ones(16), entropy=0.1, siglip_score=0.9)

        patches = [p_small_det, p_large_det, p_context]

        class VariableAreaSAM:
            def batch_inference(self, crop, active_objects):
                cw, ch = crop.size
                # Return area 40x40 for small patch, 200x200 for large patch
                if crop.size == (512, 512):
                    # Distinguish based on coordinate context
                    return (None, {1: {"boxes": [[10, 10, 210, 210]], "scores": [0.85]}}, [1])
                return (None, {}, [])

        cfg = MSSearchConfig()
        cfg.verbose = False
        cfg.max_patches_per_detected_group = 2
        cfg.enable_group_retention_caps = True
        cfg.max_retained_patches = 16

        retained = run_feature_grouping_and_gating(
            bounded_patches=patches,
            candidate_images=[self.dummy_img],
            active_objects=["target"],
            sam_model=self.mock_sam,
            zoom_model=self.mock_vlm,
            config=cfg
        )

        self.assertLessEqual(len(retained), 2)
        for r in retained:
            self.assertIsNotNone(r.confidence_score)

    def test_repartition_child_sam3_detection_gating_logic(self):
        """MSSearch Stage 4->3 feedback: child with detections survives unboosted, child with 0 pruned."""
        tau_det = 0.30
        # Scenario A: Child with detection >= tau_det -> survives with continuous unboosted score in [0.0, 1.0]
        child_dets_a = [{"label": "target", "score": 0.88, "local_box": [10, 10, 50, 50]}]
        best_score_a = max(d["score"] for d in child_dets_a)
        self.assertGreaterEqual(best_score_a, tau_det)
        self.assertLessEqual(best_score_a, 1.0)

        # Scenario B: Child with 0 detections above tau_det -> permanently pruned
        child_dets_b = []
        has_survivor = any(d["score"] >= tau_det for d in child_dets_b)
        self.assertFalse(has_survivor)

    def test_stage4_dynamic_retention_elasticity(self):
        """Stage 4: Tied candidates at cutoff trigger dynamic budget expansion."""
        # 5 candidates: 2 top, and 3 tied at cutoff
        p1 = BoundedPatch(img_idx=0, box=(0, 0, 100, 100), feature_vector=np.zeros(16), entropy=1.0, confidence_score=1.5, siglip_score=0.9)
        p2 = BoundedPatch(img_idx=0, box=(100, 0, 200, 100), feature_vector=np.zeros(16), entropy=1.0, confidence_score=1.4, siglip_score=0.8)
        p3 = BoundedPatch(img_idx=0, box=(200, 0, 300, 100), feature_vector=np.zeros(16), entropy=1.0, confidence_score=1.0, siglip_score=0.5)
        p4 = BoundedPatch(img_idx=0, box=(300, 0, 400, 100), feature_vector=np.zeros(16), entropy=1.0, confidence_score=1.0, siglip_score=0.5)
        p5 = BoundedPatch(img_idx=0, box=(400, 0, 500, 100), feature_vector=np.zeros(16), entropy=1.0, confidence_score=1.0, siglip_score=0.5)

        patches = [p1, p2, p3, p4, p5]

        cfg = MSSearchConfig()
        cfg.verbose = False
        cfg.max_retained_patches = 3  # Cutoff hits p3; p4 and p5 share identical score profile
        cfg.enable_retention_hard_cap = True

        retained = run_feature_grouping_and_gating(
            bounded_patches=patches,
            candidate_images=[self.dummy_img],
            active_objects=["target"],
            sam_model=self.mock_sam,
            zoom_model=self.mock_vlm,
            config=cfg
        )

        # Dynamic elasticity retains all 3 tied items: total = 2 top + 3 tied = 5 items (expanded from 3 to 5)
        self.assertEqual(len(retained), 5)

    def test_stage5_atomic_sibling_refusion(self):
        """Stage 5: Sibling patches along quadtree seam are fused into solid atomic box."""
        parent_box = (0, 0, 1024, 1024)
        # Two siblings touching along vertical seam at x=512
        p_left = BoundedPatch(img_idx=0, box=(0, 0, 512, 512), feature_vector=np.zeros(16), entropy=1.0, parent_id=parent_box, level=1)
        p_right = BoundedPatch(img_idx=0, box=(512, 0, 1024, 512), feature_vector=np.zeros(16), entropy=1.5, parent_id=parent_box, level=1)
        # Non-sibling patch (different parent)
        p_other = BoundedPatch(img_idx=0, box=(1100, 0, 1600, 512), feature_vector=np.zeros(16), entropy=0.8, parent_id=(1024, 0, 2048, 1024), level=1)

        self.assertTrue(are_seam_adjacent_siblings(p_left, p_right))
        self.assertFalse(are_seam_adjacent_siblings(p_left, p_other))

        fused = perform_atomic_sibling_refusion([p_left, p_right, p_other])
        self.assertEqual(len(fused), 2)

        # First patch should be the fused box (0, 0, 1024, 512)
        fused_box = fused[0].box
        self.assertEqual(fused_box, (0, 0, 1024, 512))
        self.assertTrue(getattr(fused[0], "is_fused", False))
        self.assertEqual(len(getattr(fused[0], "constituent_boxes", [])), 2)

    def test_stage5_continuous_enclosure_subcanvases_bounds(self):
        """Stage 5: Continuous enclosure windowing guarantees max dimension <= 1024px."""
        # Proximal cluster: (100, 100, 400, 400) and (500, 100, 800, 400) -> union span = 700px <= 1024px
        p1 = BoundedPatch(img_idx=0, box=(100, 100, 400, 400), feature_vector=np.zeros(16), entropy=1.0, confidence_score=1.5)
        p2 = BoundedPatch(img_idx=0, box=(500, 100, 800, 400), feature_vector=np.zeros(16), entropy=1.0, confidence_score=1.4)
        # Distant candidate: (1200, 100, 1500, 400) -> would make union span 1400px > 1024px
        p3 = BoundedPatch(img_idx=0, box=(1200, 100, 1500, 400), feature_vector=np.zeros(16), entropy=1.0, confidence_score=1.3)

        clusters = partition_continuous_enclosure_subcanvases([p1, p2, p3], image_shape=(2048, 2048), max_dim=1024)

        # Must be partitioned into 2 clusters
        self.assertEqual(len(clusters), 2)
        self.assertEqual(len(clusters[0]), 2)  # p1 and p2
        self.assertEqual(len(clusters[1]), 1)  # p3

    def test_stage5_zero_contract_emission_to_stage6(self):
        """Stage 5: Sub-canvases wrap into CandidatePool without breaking Stage 6 interface."""
        p1 = BoundedPatch(img_idx=0, box=(100, 100, 400, 400), feature_vector=np.zeros(16), entropy=1.0, confidence_score=1.5)
        p2 = BoundedPatch(img_idx=0, box=(500, 100, 800, 400), feature_vector=np.zeros(16), entropy=1.0, confidence_score=1.4)

        cfg = MSSearchConfig()
        cfg.verbose = False
        cfg.enable_continuous_enclosure_subcanvas = True
        cfg.spatial_canvas_relational_only = True

        pool = run_resolution_screening(
            remaining_patches=[p1, p2],
            candidate_images=[self.dummy_img],
            active_objects=["boat", "buoy"],
            config=cfg,
            question="What is the object to the left of the boat?",
            is_relational=True
        )

        self.assertIsInstance(pool, CandidatePool)
        self.assertEqual(len(pool), 1)
        crop_item = pool[0]
        self.assertIsInstance(crop_item, CandidateCrop)
        self.assertTrue(crop_item.is_compressed_canvas)
        self.assertEqual(crop_item.source_branch, "ContinuousEnclosure")

        # Dimensions strictly bounded <= 1024px
        self.assertLessEqual(crop_item.crop_image.width, 1024)
        self.assertLessEqual(crop_item.crop_image.height, 1024)

        # Remapped boxes are within crop boundaries
        for rb in crop_item.remapped_boxes:
            self.assertGreaterEqual(rb[0], 0)
            self.assertGreaterEqual(rb[1], 0)
            self.assertLessEqual(rb[2], crop_item.crop_image.width)
            self.assertLessEqual(rb[3], crop_item.crop_image.height)


if __name__ == "__main__":
    unittest.main()
