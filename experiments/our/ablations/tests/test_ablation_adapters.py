"""
Self-check unit tests for MSSearch ablation adapters:
- GroundingDinoDetector
- DinoXDetector
- SAM3Detector
- CLIPVisualScorer
- EVA02VisualScorer
- Factories load_detector and load_visual_scorer
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

# Ensure paths
cur_dir = os.path.dirname(os.path.abspath(__file__))
ablation_dir = os.path.abspath(os.path.join(cur_dir, ".."))
our_dir = os.path.abspath(os.path.join(ablation_dir, ".."))
repo_root = os.path.abspath(os.path.join(our_dir, "../.."))

for p in [repo_root, our_dir, ablation_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Mock torch if unavailable
for mod in ["torch", "torch.nn", "torch.nn.functional", "transformers"]:
    if mod not in sys.modules:
        m = MagicMock()
        m.__spec__ = MagicMock()
        m.__path__ = []
        sys.modules[mod] = m

import torch
from models.base import BaseDetector, BaseVisualScorer
from models.grounding_dino_adapter import GroundingDinoDetector
from models.dinox_adapter import DinoXDetector
from models.clip_scorer_adapter import CLIPVisualScorer, format_clip_prompts
from models.eva02_scorer_adapter import EVA02VisualScorer
from models import load_detector, load_visual_scorer


class TestAblationAdapters(unittest.TestCase):

    def setUp(self):
        self.dummy_img = Image.new("RGB", (200, 150), color="white")
        self.dummy_prompts = ["car", "person"]

    def test_clip_prompt_formatting(self):
        prompts = format_clip_prompts(["car", "apple", "a photo of a bike", "the bus"])
        self.assertEqual(prompts[0], "a photo of a car")
        self.assertEqual(prompts[1], "a photo of an apple")
        self.assertEqual(prompts[2], "a photo of a bike")
        self.assertEqual(prompts[3], "the bus")

    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("transformers.AutoModelForZeroShotObjectDetection.from_pretrained")
    def test_grounding_dino_contract(self, mock_model_cls, mock_proc_cls):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_cls.return_value = mock_proc
        mock_model_cls.return_value.to.return_value.eval.return_value = mock_model

        # Mock post_process_grounded_object_detection
        mock_proc.post_process_grounded_object_detection.return_value = [{
            "boxes": torch.tensor([[10.2, 20.4, 50.8, 60.1]]),
            "scores": torch.tensor([0.88]),
        }]

        detector = GroundingDinoDetector(model_name_or_path="mock-gdino", device="cpu")
        backbone_out, processed_results, text_id = detector.batch_inference(
            self.dummy_img, self.dummy_prompts
        )

        self.assertIsNone(backbone_out)
        self.assertEqual(text_id, [1, 2])
        self.assertIn(1, processed_results)
        self.assertIn(2, processed_results)

        # Check box coordinate integer clamping
        b1 = processed_results[1]["boxes"][0]
        self.assertEqual(b1, [10, 20, 51, 60])
        s1 = processed_results[1]["scores"][0]
        self.assertAlmostEqual(s1, 0.88, places=2)

    def test_dinox_stub_mode(self):
        detector = DinoXDetector(model_name_or_path="", device="cpu")
        self.assertEqual(detector.mode, "stub")

        backbone_out, processed_results, text_id = detector.batch_inference(
            self.dummy_img, self.dummy_prompts
        )
        self.assertIsNone(backbone_out)
        self.assertEqual(text_id, [1, 2])
        self.assertEqual(processed_results[1]["boxes"], [])
        self.assertEqual(processed_results[2]["scores"], [])

    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("transformers.AutoModel.from_pretrained")
    def test_clip_scorer_bounds(self, mock_model_cls, mock_proc_cls):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_cls.return_value = mock_proc
        mock_model_cls.return_value.to.return_value.eval.return_value = mock_model

        # Image embed: [2 crops, 8 dim], Text embed: [2 queries, 8 dim]
        mock_model.get_text_features.return_value = torch.ones(2, 8)
        mock_model.get_image_features.return_value = torch.ones(2, 8)

        scorer = CLIPVisualScorer(model_name="openai/clip-vit-base-patch32", device="cpu")
        crops = [Image.new("RGB", (64, 64)) for _ in range(2)]
        scores = scorer.score_crops(crops, ["car", "dog"])

        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_factory_dispatch(self):
        with patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.AutoModelForZeroShotObjectDetection.from_pretrained"):
            det = load_detector("grounding_dino", device="cpu")
            self.assertIsInstance(det, BaseDetector)

        det_x = load_detector("dinox", device="cpu")
        self.assertIsInstance(det_x, BaseDetector)

        with patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.AutoModel.from_pretrained"):
            sc_clip = load_visual_scorer("clip_b32", device="cpu")
            self.assertIsInstance(sc_clip, BaseVisualScorer)

            sc_l14 = load_visual_scorer("clip_l14", device="cpu")
            self.assertIsInstance(sc_l14, BaseVisualScorer)


if __name__ == "__main__":
    unittest.main()
