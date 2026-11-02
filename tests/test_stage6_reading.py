import unittest
from PIL import Image
from mssearch.config import MSSearchConfig
from mssearch.stages.stage5_screening import CandidateCrop
from mssearch.stages.stage6_foveated_reader import run_foveated_vlm_reading, FinalSearchResult


class MockZoomModel:
    def __init__(self):
        self.recorded_calls = []

    def generate_response(self, image_pil, question, max_new_tokens=128, **kwargs):
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


if __name__ == "__main__":
    unittest.main()
