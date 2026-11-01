"""
Unit tests for SigLIP / CLIP Scorer (mssearch/models/modeling_siglip.py).
"""

import unittest
from unittest.mock import MagicMock
from PIL import Image

from mssearch.models.modeling_siglip import SigLIPScorer, _extract_embedding_tensor, format_text_queries

try:
    import torch
except ImportError:
    torch = None


class TestSigLIPScorerPrompting(unittest.TestCase):

    def test_format_text_queries_siglip(self):
        """SigLIP retains raw tags directly without prefixing."""
        queries = ["car", "apple", "the bus", " red flower "]
        formatted = format_text_queries(queries, is_siglip=True)
        self.assertEqual(formatted, ["car", "apple", "the bus", "red flower"])

    def test_format_text_queries_clip(self):
        """CLIP applies grammatically appropriate articles (a/an) unless already prefixed."""
        queries = ["car", "apple", "the bus", "a train", "photo of a cat"]
        formatted = format_text_queries(queries, is_siglip=False)
        self.assertEqual(
            formatted,
            ["a photo of a car", "a photo of an apple", "the bus", "a train", "photo of a cat"]
        )


@unittest.skipIf(torch is None, "PyTorch is not installed in the test environment")
class TestSigLIPScorerTorch(unittest.TestCase):

    def test_extract_embedding_tensor(self):
        t = torch.randn(2, 64)
        # 1. Raw tensor
        self.assertTrue(torch.equal(_extract_embedding_tensor(t), t))

        # 2. Object with pooler_output
        mock_obj = MagicMock()
        mock_obj.pooler_output = t
        self.assertTrue(torch.equal(_extract_embedding_tensor(mock_obj), t))

        # 3. Tuple
        self.assertTrue(torch.equal(_extract_embedding_tensor((t, "extra")), t))

    def test_siglip_dense_matching_bypassed(self):
        """Verify SigLIP does not run dense spatial token matching (bypassed)."""
        scorer = SigLIPScorer(model_name="google/siglip-base-patch16-384", device="cpu")
        scorer.is_siglip = True

        mock_model = MagicMock()
        mock_processor = MagicMock()

        # Text features: [2, 768]
        mock_model.get_text_features.return_value = torch.ones(2, 768)
        # Image features: [3, 768]
        mock_model.get_image_features.return_value = torch.ones(3, 768)

        # Vision model returning spatial tokens
        mock_vision = MagicMock()
        vis_out = MagicMock()
        vis_out.last_hidden_state = torch.ones(3, 49, 768)
        mock_vision.return_value = vis_out
        mock_model.vision_model = mock_vision
        # Even if hasattr visual_projection exists on model, is_siglip must bypass it
        mock_model.visual_projection = MagicMock(return_value=torch.ones(3, 49, 768))

        mock_processor.return_value.to.return_value = {
            "input_ids": torch.zeros(2, 10, dtype=torch.long),
            "pixel_values": torch.zeros(3, 3, 224, 224),
        }

        scorer.model = mock_model
        scorer.processor = mock_processor

        crops = [Image.new("RGB", (100, 100)) for _ in range(3)]
        scores = scorer.score_crops(crops, ["car", "apple"])

        self.assertEqual(len(scores), 3)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)
        # Vision model should NOT have been queried for dense tokens under SigLIP
        mock_vision.assert_not_called()


if __name__ == "__main__":
    unittest.main()
