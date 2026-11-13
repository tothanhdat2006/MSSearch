"""
Unit tests for SigLIP / CLIP Scorer (mssearch/models/modeling_siglip.py).
"""

import sys
import os
import unittest
from unittest.mock import MagicMock
from PIL import Image
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mssearch.models.modeling_siglip import SigLIPScorer, _extract_embedding_tensor, format_text_queries


class TestSigLIPScorer(unittest.TestCase):

    def test_format_text_queries(self):
        siglip_prompts = format_text_queries(["car", " white board "], is_siglip=True)
        self.assertEqual(siglip_prompts, ["car", "white board"])

        clip_prompts = format_text_queries(["car", "apple", "the dog"], is_siglip=False)
        self.assertEqual(clip_prompts, ["a photo of a car", "a photo of an apple", "the dog"])

    def test_extract_embedding_tensor(self):
        t = torch.randn(2, 64)
        self.assertTrue(torch.equal(_extract_embedding_tensor(t), t))

        mock_obj = MagicMock()
        mock_obj.pooler_output = t
        self.assertTrue(torch.equal(_extract_embedding_tensor(mock_obj), t))

    def test_score_crops_huggingface_native(self):
        scorer = SigLIPScorer(model_name="google/siglip2-so400m-patch16-512", device="cpu")
        scorer.is_siglip = True

        mock_model = MagicMock()
        mock_processor = MagicMock()

        # Mock outputs.logits_per_image: [3 crops, 2 queries]
        # Crop 0: logits 0.0 -> prob 0.5
        # Crop 1: logits 10.0 -> prob ~1.0
        # Crop 2: logits -10.0 -> prob ~0.0
        logits = torch.tensor([
            [0.0, -1.0],
            [10.0, 5.0],
            [-10.0, -12.0],
        ])
        mock_output = MagicMock()
        mock_output.logits_per_image = logits
        mock_model.return_value = mock_output

        mock_processor.return_value = {
            "input_ids": torch.zeros(2, 10, dtype=torch.long),
            "pixel_values": torch.zeros(3, 3, 224, 224),
        }

        scorer.model = mock_model
        scorer.processor = mock_processor

        crops = [Image.new("RGB", (50, 50)) for _ in range(3)]
        scores = scorer.score_crops(crops, ["car", "apple"])

        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(scores[0], 0.5, places=3)
        self.assertAlmostEqual(scores[1], 1.0, places=3)
        self.assertAlmostEqual(scores[2], 0.0, places=3)

    def test_score_crops_clip_single_query_non_collapse(self):
        """Tests that CLIP does not collapse to 1.0 for single-target queries (|O_Q| = 1)."""
        scorer = SigLIPScorer(model_name="openai/clip-vit-large-patch14", device="cpu", calib_center=0.20, calib_scale=12.0)
        self.assertFalse(scorer.is_siglip)

        mock_model = MagicMock()
        mock_processor = MagicMock()

        # logit_scale = ln(100) -> scale = 100
        mock_model.logit_scale.exp.return_value = 100.0

        # Feature embeddings path:
        # 2 crops, 1 query
        # Crop 0: cos_sim = 0.10 (background) -> logit = 12 * (0.10 - 0.20) = -1.2 -> sigmoid(-1.2) ~ 0.231
        # Crop 1: cos_sim = 0.35 (strong target) -> logit = 12 * (0.35 - 0.20) = +1.8 -> sigmoid(+1.8) ~ 0.858
        crop_embeds = torch.tensor([
            [1.0, 0.0],
            [0.7071, 0.7071],
        ])
        query_embeds = torch.tensor([
            [0.10, 0.9949],
        ])  # cos(crop0, query) = 0.10

        # We can mock get_text_features and get_image_features directly
        mock_model.get_image_features.side_effect = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.7071, 0.7071]]),
        ]
        # Or mock the outputs.logits_per_image in fallback mode:
        mock_output = MagicMock()
        # Crop 0: raw CLIP logit = 10.0 (cos = 0.10)
        # Crop 1: raw CLIP logit = 35.0 (cos = 0.35)
        mock_output.logits_per_image = torch.tensor([[10.0], [35.0]])
        mock_model.return_value = mock_output

        mock_processor.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "pixel_values": torch.zeros(2, 3, 224, 224),
        }

        # Force fallback path to test outputs.logits_per_image branch directly
        del mock_model.get_text_features
        scorer.model = mock_model
        scorer.processor = mock_processor

        crops = [Image.new("RGB", (50, 50)), Image.new("RGB", (50, 50))]
        scores = scorer.score_crops(crops, ["license plate"])

        self.assertEqual(len(scores), 2)
        # Verify pure cosine similarity: background crop = 0.10, target crop = 0.35
        self.assertAlmostEqual(scores[0], 0.10, places=2)
        self.assertAlmostEqual(scores[1], 0.35, places=2)


if __name__ == "__main__":
    unittest.main()
