import sys
import os
import unittest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mssearch.config import MSSearchConfig
from mssearch.stages.stage5_screening import CandidateCrop
from mssearch.stages.stage6_foveated_reader import run_foveated_vlm_reading, FinalSearchResult


class MockZoomModel:
    def __init__(self):
        self.recorded_calls = []

    def generate_response(self, image_pil, question, max_new_tokens=512, **kwargs):
        self.recorded_calls.append({
            "image": image_pil,
            "question": question,
            "max_new_tokens": max_new_tokens,
        })
        return "Nike"


class TestStage6Reading(unittest.TestCase):
    def setUp(self):
        self.img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        self.candidate_images = [self.img]
        self.crop_item = CandidateCrop(
            img_idx=0,
            box=(40, 40, 80, 80),
            crop_image=self.img.crop((40, 40, 80, 80)),
            score=0.9,
            source_branch="SAM3Detection"
        )
        self.unified_pool = [self.crop_item]

    def test_prompt_format_image_question_no_bbox(self):
        config = MSSearchConfig(
            enable_visual_prompt_bbox=True,
            prompt_directed_final_answer=False,
            vlm_batch_size=1
        )
        mock_vlm = MockZoomModel()
        q_text = "What is the brand name of the shoes?"

        res: FinalSearchResult = run_foveated_vlm_reading(
            unified_pool=self.unified_pool,
            candidate_images=self.candidate_images,
            question=q_text,
            zoom_model=mock_vlm,
            config=config
        )

        self.assertEqual(len(mock_vlm.recorded_calls), 1)
        call = mock_vlm.recorded_calls[0]

        # 1. Prompt format strictly conforms to <image>\n<question>
        expected_prompt = f"<image>\n{q_text}"
        self.assertEqual(call["question"], expected_prompt)

        # 2. VLM input image is clean expanded crop with no bounding boxes drawn on it
        input_image = call["image"]
        self.assertIsInstance(input_image, Image.Image)
        # Pixel values should be original gray (128, 128, 128), untouched by red box drawing
        px = input_image.getpixel((0, 0))
        self.assertEqual(px, (128, 128, 128))

        # 3. Final answer extraction
        self.assertEqual(res.answers, ["Nike"])

    def test_qwen_prompt_from_qs_structure(self):
        from mssearch.models.modeling_qwenvl import ModelQwenVL

        class MockProcessor:
            def apply_chat_template(self, message, tokenize=False, add_generation_prompt=True):
                return message

        obj = object.__new__(ModelQwenVL)
        obj.processor = MockProcessor()

        # Check prompt with single image prefix
        qs = "<image>\nWhich store is on the corner?"
        parsed = obj.get_prompt_from_qs(qs)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["role"], "user")
        self.assertEqual(len(parsed[0]["content"]), 2)
        self.assertEqual(parsed[0]["content"][0], {"type": "image"})
        self.assertEqual(parsed[0]["content"][1], {"type": "text", "text": "Which store is on the corner?"})

    def test_direct_answer_prompt_and_extraction(self):
        from mssearch.stages.utils_stage2 import build_direct_answer_prompt
        from mssearch.stages.utils_stage6 import extract_final_answer

        # Test prompt creation
        p1 = build_direct_answer_prompt("What color is the car?")
        self.assertEqual(p1, "<image>\nWhat color is the car?")

        p2 = build_direct_answer_prompt("<image>\nWhat color is the car?")
        self.assertEqual(p2, "<image>\nWhat color is the car?")

        # Test answer extraction (raw text and legacy XML tags)
        self.assertEqual(extract_final_answer("  **Blue**  "), "Blue")
        self.assertEqual(extract_final_answer("<answer>Red</answer>"), "Red")
        self.assertEqual(extract_final_answer("(A)"), "A")

    def test_majority_voting_conflicts(self):
        from mssearch.stages.utils_stage6 import EvidenceExtraction, resolve_by_majority_voting

        # Scenario 1: Clear frequency majority (2 vs 1)
        exts_1 = [
            EvidenceExtraction(img_idx=1, box=(0, 0, 50, 50), expanded_box=(0, 0, 60, 60), raw_answer="Blue", final_answer="Blue", source_branch="test", crop_score=0.7),
            EvidenceExtraction(img_idx=2, box=(0, 0, 50, 50), expanded_box=(0, 0, 60, 60), raw_answer="Red", final_answer="Red", source_branch="test", crop_score=0.9),
            EvidenceExtraction(img_idx=3, box=(0, 0, 50, 50), expanded_box=(0, 0, 60, 60), raw_answer="Dark Blue", final_answer="Dark Blue", source_branch="test", crop_score=0.8),
        ]
        winner, win_img, ranked = resolve_by_majority_voting(exts_1)
        self.assertIn("blue", winner.lower())
        self.assertIn(win_img, [1, 3])
        self.assertEqual(len(ranked), 2)

        # Scenario 2: Tie break by confidence score (1 vs 1)
        exts_2 = [
            EvidenceExtraction(img_idx=0, box=(0, 0, 50, 50), expanded_box=(0, 0, 60, 60), raw_answer="Black", final_answer="Black", source_branch="test", crop_score=0.95),
            EvidenceExtraction(img_idx=1, box=(0, 0, 50, 50), expanded_box=(0, 0, 60, 60), raw_answer="White", final_answer="White", source_branch="test", crop_score=0.40),
        ]
        winner, win_img, ranked = resolve_by_majority_voting(exts_2)
        self.assertEqual(winner, "Black")
        self.assertEqual(win_img, 0)

        # Scenario 3: Negative answer filtering
        exts_3 = [
            EvidenceExtraction(img_idx=0, box=(0, 0, 50, 50), expanded_box=(0, 0, 60, 60), raw_answer="None", final_answer="None", source_branch="test", crop_score=0.9),
            EvidenceExtraction(img_idx=1, box=(0, 0, 50, 50), expanded_box=(0, 0, 60, 60), raw_answer="Green", final_answer="Green", source_branch="test", crop_score=0.5),
        ]
        winner, win_img, ranked = resolve_by_majority_voting(exts_3)
        self.assertEqual(winner, "Green")
        self.assertEqual(win_img, 1)


if __name__ == "__main__":
    unittest.main()
