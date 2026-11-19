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
    BoundedPatch,
    partition_single_image,
    partition_gallery,
)
from minimal.stages.visual_screening import (
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
        self.assertEqual(normalize_target("The Windows"), "window")
        self.assertEqual(normalize_target("a red mailbox!"), "red mailbox")
        self.assertEqual(normalize_target("berries"), "berry")

    def test_extraction_pipeline(self):
        mock_mllm = MockMLLM(parse_output='{"active_objects": ["it", "boats", "that"]}')
        targets = extract_visual_targets("What is the boat registration?", mock_mllm, verbose=False)
        # Pronouns 'it' and 'that' should be stripped, 'boats' normalized to 'boat'
        self.assertEqual(targets, ["boat"])

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


if __name__ == "__main__":
    unittest.main()
