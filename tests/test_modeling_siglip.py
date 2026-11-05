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


if __name__ == "__main__":
    unittest.main()
