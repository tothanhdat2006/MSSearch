"""
Unit tests for STASH-Search stages.
Verifies Stage 1, Stage 2, Stage 3, and Stage 4 algorithmic invariants using standard unittest.
"""

import unittest
from PIL import Image

from minimal.config import STASHConfig
from minimal.models.base import BaseMLLM, BaseDualEncoder
from minimal.stages.target_extraction import (
    parse_mllm_json,
    normalize_target,
    extract_visual_targets,
    build_target_extraction_messages,
)
from minimal.stages.quadtree_partitioning import (
    get_quadtree_splits,
    split_box_into_quadrants,
    compute_spatial_entropy,
    BoundedPatch,
    partition_single_image,
    partition_gallery,
)
from minimal.stages.visual_screening import (
    compute_box_ios,
    compute_box_iou,
    merge_nested_containment_boxes,
    apply_spatial_nms,
    are_seam_adjacent_siblings,
    perform_atomic_sibling_refusion,
    expand_box,
    clean_mllm_answer,
    screen_and_inspect_crops,
    CandidateExtraction,
)
from minimal.stages.consensus import (
    is_uninformative_response,
    normalize_answer_string,
    resolve_source_attributed_consensus,
)


class MockMLLM(BaseMLLM):
    def __init__(self, parse_output: str = '{"active_objects": ["mailbox"], "question_type": "single_object_attribute"}', read_output: str = "blue"):
        self.parse_output = parse_output
        self.read_output = read_output

    def generate_text(self, prompt_or_messages, max_new_tokens=96) -> str:
        return self.parse_output

    def generate_response_batch(self, images, question, max_new_tokens=512):
        return [self.read_output for _ in images]


class MockDualEncoder(BaseDualEncoder):
    def __init__(self, fixed_score: float = 0.50):
        self.fixed_score = fixed_score

    def score_crops(self, crops, text_queries, batch_size=32):
        return [self.fixed_score for _ in crops]


class TestStage1TargetExtraction(unittest.TestCase):
    def test_json_parsing(self):
        json_text = '```json\n{"active_objects": ["brick building", "window"], "question_type": "counting"}\n```'
        targets = parse_mllm_json(json_text)
        self.assertEqual(targets, ["brick building", "window"])

    def test_bullet_list_parsing(self):
        bullet_text = "- mailbox\n- person\n- car"
        targets = parse_mllm_json(bullet_text)
        self.assertEqual(targets, ["mailbox", "person", "car"])

    def test_target_normalization(self):
        # Edge trimming only, internal hyphens and plurals preserved (no article stripping)
        self.assertEqual(normalize_target("The Windows"), "the windows")
        self.assertEqual(normalize_target("a red mailbox!"), "a red mailbox")
        self.assertEqual(normalize_target("!open-front bar?"), "open-front bar")
        self.assertEqual(normalize_target("pants"), "pants")
        self.assertEqual(normalize_target("shoes"), "shoes")
        self.assertEqual(normalize_target("all shoes"), "shoe")

    def test_extraction_pipeline(self):
        mock_mllm = MockMLLM(parse_output='{"active_objects": ["it", "boats", "that"]}')
        targets = extract_visual_targets("What is the boat registration?", mock_mllm, verbose=False)
        # Pronouns 'it' and 'that' should be stripped, 'boats' preserved without singularization
        self.assertEqual(targets, ["boats"])

    def test_fallback_when_empty(self):
        mock_mllm = MockMLLM(parse_output='{}')
        targets = extract_visual_targets("Where is the waterfall located?", mock_mllm, verbose=False)
        self.assertTrue(len(targets) > 0)
        self.assertIn("waterfall", targets)


class TestStage2QuadtreePartitioning(unittest.TestCase):
    def test_aspect_ratio_splits(self):
        # Tall image (H/W = 3.0 >= 2.0) -> vertical 1x4
        self.assertEqual(get_quadtree_splits(500, 1500), (1, 4))
        # Wide image (H/W = 0.33 <= 0.5) -> horizontal 4x1
        self.assertEqual(get_quadtree_splits(1500, 500), (4, 1))
        # Standard square -> 2x2
        self.assertEqual(get_quadtree_splits(1000, 1000), (2, 2))

    def test_quadrant_padding(self):
        box = (0, 0, 1000, 1000)
        children = split_box_into_quadrants(box, 1000, 1000, overlap_ratio=0.10)
        self.assertEqual(len(children), 4)
        # With 10% overlap on 500px quadrant, padding is 50px
        # First child: (0, 0, 550, 550)
        self.assertEqual(children[0], (0, 0, 550, 550))
        # All children clamped within image boundary
        for c in children:
            self.assertTrue(0 <= c[0] < c[2] <= 1000)
            self.assertTrue(0 <= c[1] < c[3] <= 1000)

    def test_plateau_stopping_and_scale_bounds(self):
        # Synthetic image of size 2048 x 2048 (> 1536 mandatory split ceiling)
        img = Image.new("RGB", (2048, 2048), color="white")
        config = STASHConfig(
            min_plateau_size=512,
            max_plateau_size=1536,
            saliency_plateau_gain_threshold=0.04,
            verbose=False
        )
        mock_encoder = MockDualEncoder(fixed_score=0.40)
        patches = partition_single_image(img, 0, ["target"], mock_encoder, config)
        self.assertTrue(len(patches) > 0)
        # Verify that all leaf patches obey scale floor or depth limit
        for p in patches:
            self.assertTrue(max(p.width, p.height) <= 1536 * 1.2)

    def test_spatial_entropy_calculation(self):
        # Uniform image has low entropy near 0.0
        uniform_img = Image.new("RGB", (256, 256), color="black")
        ent_low = compute_spatial_entropy(uniform_img)
        self.assertLess(ent_low, 0.35)

        # High-contrast checkerboard image has high entropy
        import numpy as np
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[::16, :] = 255
        arr[:, ::16] = 255
        textured_img = Image.fromarray(arr)
        ent_high = compute_spatial_entropy(textured_img)
        self.assertGreater(ent_high, 0.35)

    def test_entropy_leaf_pruning(self):
        # Uniform image: entropy < 0.35
        img = Image.new("RGB", (1024, 1024), color="black")
        # With alignment_score < saliency_target_threshold (0.35), leaves are pruned
        config = STASHConfig(
            min_patch_entropy=0.35,
            saliency_target_threshold=0.35,
            min_plateau_size=512,
            verbose=False
        )
        mock_low_encoder = MockDualEncoder(fixed_score=0.20)
        patches_pruned = partition_single_image(img, 0, ["target"], mock_low_encoder, config)
        self.assertEqual(len(patches_pruned), 0)

        # With alignment_score >= saliency_target_threshold (0.35), leaves are kept despite low entropy
        mock_high_encoder = MockDualEncoder(fixed_score=0.40)
        patches_kept = partition_single_image(img, 0, ["target"], mock_high_encoder, config)
        self.assertGreater(len(patches_kept), 0)


class TestStage3VisualScreening(unittest.TestCase):
    def test_seam_overlap(self):
        parent = (0, 0, 1000, 1000)
        # Two boxes touching along vertical seam x=500
        p1 = BoundedPatch(img_idx=0, box=(0, 0, 500, 500), depth=1, parent_box=parent)
        p2 = BoundedPatch(img_idx=0, box=(500, 0, 1000, 500), depth=1, parent_box=parent)
        self.assertTrue(are_seam_adjacent_siblings(p1, p2, seam_threshold=0.10))

        # Two boxes overlapping along vertical seam (e.g. padded by 50px)
        p1_padded = BoundedPatch(img_idx=0, box=(0, 0, 550, 550), depth=1, parent_box=parent)
        p2_padded = BoundedPatch(img_idx=0, box=(450, 0, 1000, 550), depth=1, parent_box=parent)
        self.assertTrue(are_seam_adjacent_siblings(p1_padded, p2_padded, seam_threshold=0.10))

        # Non-adjacent boxes
        p3 = BoundedPatch(img_idx=0, box=(600, 600, 900, 900), depth=1, parent_box=parent)
        self.assertFalse(are_seam_adjacent_siblings(p1, p3, seam_threshold=0.10))

    def test_atomic_sibling_refusion(self):
        parent = (0, 0, 1000, 1000)
        p1 = BoundedPatch(img_idx=0, box=(0, 0, 500, 500), depth=1, alignment_score=0.8, parent_box=parent)
        p2 = BoundedPatch(img_idx=0, box=(500, 0, 1000, 500), depth=1, alignment_score=0.7, parent_box=parent)
        fused = perform_atomic_sibling_refusion([p1, p2], seam_threshold=0.10)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].box, (0, 0, 1000, 500))
        self.assertEqual(fused[0].alignment_score, 0.8)

    def test_expand_box(self):
        box = (100, 100, 300, 300) # 200x200
        # 1.25x expansion expands 200 -> 250 (+25 on each side)
        exp = expand_box(box, 1000, 1000, expansion_factor=1.25)
        self.assertEqual(exp, (75, 75, 325, 325))

    def test_clean_mllm_answer(self):
        self.assertEqual(clean_mllm_answer("<answer> 42 </answer>"), "42")
        self.assertEqual(clean_mllm_answer('**"red"**'), "red")

    def test_ios_and_iou_computation(self):
        # Child box completely inside parent box
        child = (100, 100, 200, 200)  # Area = 10000
        parent = (50, 50, 300, 300)    # Area = 62500
        ios = compute_box_ios(child, parent)
        self.assertAlmostEqual(ios, 1.0)

        iou = compute_box_iou(child, parent)
        # Intersection = 10000, Union = 62500 -> IoU = 10000/62500 = 0.16
        self.assertAlmostEqual(iou, 10000.0 / 62500.0)

    def test_merge_nested_containment_boxes(self):
        # Child box is 100% contained in parent box (IoS = 1.0 >= 0.95)
        parent_p = BoundedPatch(img_idx=0, box=(50, 50, 500, 500), depth=1, alignment_score=0.4)
        child_p = BoundedPatch(img_idx=0, box=(100, 100, 200, 200), depth=2, alignment_score=0.8)
        surviving = merge_nested_containment_boxes([parent_p, child_p], containment_threshold=0.95)
        # Should drop coarse parent in favor of fine child, transferring parent score
        self.assertEqual(len(surviving), 1)
        self.assertEqual(surviving[0].box, child_p.box)
        self.assertEqual(surviving[0].alignment_score, 0.8)

    def test_apply_spatial_nms(self):
        # Two patches with IoU > 0.60 on the same image
        p1 = BoundedPatch(img_idx=0, box=(100, 100, 300, 300), depth=1, alignment_score=0.9)
        p2 = BoundedPatch(img_idx=0, box=(120, 100, 300, 300), depth=1, alignment_score=0.6)
        surviving = apply_spatial_nms([p1, p2], iou_threshold=0.60)
        self.assertEqual(len(surviving), 1)
        self.assertEqual(surviving[0].alignment_score, 0.9)


class TestStage4Consensus(unittest.TestCase):
    def test_is_uninformative(self):
        self.assertTrue(is_uninformative_response("No answer found."))
        self.assertTrue(is_uninformative_response("The image does not show any mailbox"))
        self.assertTrue(is_uninformative_response("not visible"))
        self.assertTrue(is_uninformative_response("none"))
        self.assertFalse(is_uninformative_response("red mailbox"))
        self.assertFalse(is_uninformative_response("42"))

    def test_majority_voting(self):
        config = STASHConfig(answer_dedup_similarity=0.85, verbose=False)
        extractions = [
            CandidateExtraction(img_idx=0, box=(0,0,10,10), expanded_box=(0,0,10,10), alignment_score=0.9, crop_area=100, raw_answer="blue", final_answer="blue"),
            CandidateExtraction(img_idx=0, box=(0,0,10,10), expanded_box=(0,0,10,10), alignment_score=0.8, crop_area=200, raw_answer="the blue", final_answer="the blue"),
            CandidateExtraction(img_idx=1, box=(0,0,10,10), expanded_box=(0,0,10,10), alignment_score=0.5, crop_area=100, raw_answer="red", final_answer="red"),
        ]
        ans, img_idx = resolve_source_attributed_consensus(extractions, config)
        self.assertIn("blue", ans.lower())
        self.assertEqual(img_idx, 0)

    def test_unanswerable_fallback(self):
        config = STASHConfig(verbose=False)
        extractions = [
            CandidateExtraction(img_idx=0, box=(0,0,10,10), expanded_box=(0,0,10,10), alignment_score=0.1, crop_area=100, raw_answer="none", final_answer="none"),
            CandidateExtraction(img_idx=1, box=(0,0,10,10), expanded_box=(0,0,10,10), alignment_score=0.2, crop_area=100, raw_answer="not visible", final_answer="not visible"),
        ]
        ans, img_idx = resolve_source_attributed_consensus(extractions, config)
        self.assertEqual(ans, "No answer found")
        self.assertIsNone(img_idx)

    def test_batched_reading_fallback_warning(self):
        try:
            import torch
            from minimal.models.modeling_qwenvl import ModelQwenVL
        except ImportError:
            self.skipTest("torch or transformers not installed in local test environment")

        import warnings
        from PIL import Image

        model_adapter = object.__new__(ModelQwenVL)
        model_adapter.device = "cpu"

        class DummyInputs:
            def __init__(self):
                self.input_ids = torch.tensor([[1, 2, 3]])
            def to(self, dev):
                return self

        class MockFailingProcessor:
            def __init__(self):
                self.tokenizer = self
            def __call__(self, text, images, **kwargs):
                if len(images) > 1:
                    raise RuntimeError("Simulated OOM on batch")
                return DummyInputs()
            def decode(self, ids, **kwargs):
                return "single answer"

        class MockGenModel:
            def generate(self, **kwargs):
                return torch.tensor([[1, 2, 3, 10, 11]])

        model_adapter.processor = MockFailingProcessor()
        model_adapter.model = MockGenModel()

        imgs = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = model_adapter.generate_response_batch(imgs, "test question?", batch_size=2)
            self.assertEqual(len(results), 2)
            self.assertEqual(results, ["single answer", "single answer"])
            self.assertGreaterEqual(len(w), 1)
            self.assertIn("Falling back to single-crop generation", str(w[-1].message))


if __name__ == "__main__":
    unittest.main()
