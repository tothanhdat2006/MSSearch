"""
Self-check unit tests for MSSearch ablation adapters:
- GroundingDinoDetector
- Florence2Detector
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

# Safe import of torch and transformers
try:
    import torch
    import torch.nn
    import torch.nn.functional
except ImportError:
    class _NullCM:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __call__(self, fn): return fn

    class MockTensor:
        def __init__(self, val):
            self.val = val
        def detach(self): return self
        def cpu(self): return self
        def tolist(self): return self.val if isinstance(self.val, list) else [self.val]
        def to(self, *a, **k): return self
        def t(self): return self
        def norm(self, *a, **k): return self
        def max(self, *a, **k):
            m = MagicMock()
            m.values = self
            return m
        def __matmul__(self, other): return self
        def __sub__(self, other): return self
        def __mul__(self, other): return self
        def __truediv__(self, other): return self

    torch = MagicMock()
    torch.inference_mode = lambda *a, **k: _NullCM()
    torch.no_grad = lambda *a, **k: _NullCM()
    torch.tensor = lambda val, **k: MockTensor(val)
    torch.ones = lambda *a, **k: MockTensor([0.8] * (a[0] if a else 1))
    torch.sigmoid = lambda x: MockTensor([0.8, 0.8])
    torch.matmul = lambda a, b: a
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.nn.functional"] = MagicMock()

try:
    import transformers
except ImportError:
    transformers = MagicMock()
    sys.modules["transformers"] = transformers

from models.base import BaseDetector, BaseVisualScorer
from models.grounding_dino_adapter import GroundingDinoDetector
from models.florence2_adapter import Florence2Detector
from models.owlv2_adapter import OWLv2Detector
from models.unified_vision_extractor import UnifiedVisionExtractor
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
        mock_proc.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": torch.ones(1, 3, 150, 200),
        }

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

    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("transformers.AutoModelForImageTextToText.from_pretrained")
    def test_florence2_contract(self, mock_model_cls, mock_proc_cls):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_cls.return_value = mock_proc
        mock_model_cls.return_value.to.return_value.eval.return_value = mock_model
        mock_proc.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.ones(1, 3, 150, 200),
        }
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_proc.batch_decode.return_value = ["dummy output"]
        mock_proc.post_process_generation.return_value = {
            "<CAPTION_TO_PHRASE_GROUNDING>": {
                "bboxes": [[5.1, 10.2, 50.4, 80.9]],
                "labels": ["car"],
            }
        }

        detector = Florence2Detector(model_name_or_path="florence-community/Florence-2-large", device="cpu")
        backbone_out, processed_results, text_id = detector.batch_inference(
            self.dummy_img, ["car", "person"]
        )

        self.assertEqual(text_id, [1, 2])
        self.assertIn(1, processed_results)
        self.assertEqual(processed_results[1]["boxes"][0], [5, 10, 50, 81])
        self.assertEqual(processed_results[1]["scores"][0], 0.95)

    @patch("transformers.Owlv2Processor.from_pretrained")
    @patch("transformers.Owlv2ForObjectDetection.from_pretrained")
    def test_owlv2_contract(self, mock_model_cls, mock_proc_cls):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_cls.return_value = mock_proc
        mock_model_cls.return_value.to.return_value.eval.return_value = mock_model
        mock_proc.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.ones(1, 3, 150, 200),
        }

        # Mock patch tokens: 1 CLS + 16 patches (4x4 grid), dim 32
        mock_vision = MagicMock()
        mock_vision.last_hidden_state = torch.ones(1, 17, 32)
        mock_model.owlv2.vision_model.return_value = mock_vision

        mock_proc.post_process_grounded_object_detection.return_value = [{
            "boxes": torch.tensor([[10.0, 20.0, 60.0, 80.0]]),
            "scores": torch.tensor([0.92]),
            "labels": torch.tensor([0]),
        }]

        detector = OWLv2Detector(model_name_or_path="mock-owlv2", device="cpu")
        backbone_out, processed_results, text_id = detector.batch_inference(
            self.dummy_img, ["car", "person"]
        )

        self.assertIsNotNone(backbone_out)
        self.assertIn("vision_features", backbone_out)
        self.assertEqual(backbone_out["vision_features"].shape, (1, 32, 4, 4))
        self.assertEqual(text_id, [1, 2])
        self.assertIn(1, processed_results)
        self.assertEqual(processed_results[1]["boxes"][0], [10, 20, 60, 80])
        self.assertAlmostEqual(processed_results[1]["scores"][0], 0.92, places=2)

    def test_owlv2_stub_mode(self):
        detector = OWLv2Detector(model_name_or_path="stub", device="cpu")
        self.assertTrue(detector.stub_mode)
        backbone_out, processed_results, text_id = detector.batch_inference(
            self.dummy_img, ["car", "person"]
        )
        self.assertIsNone(backbone_out)
        self.assertEqual(text_id, [1, 2])
        self.assertEqual(processed_results[1]["boxes"], [])

    def test_slice_feature_map_contracts(self):
        import numpy as np
        from mssearch.stages.utils_stage3 import slice_feature_map

        # 1. Standard 3D feature map [C=32, H=14, W=14]
        feat_3d = np.ones((32, 14, 14), dtype=np.float32)
        sub_feat, pooled = slice_feature_map(feat_3d, (0, 0, 100, 100), 200, 200)
        self.assertEqual(sub_feat.ndim, 3)
        self.assertEqual(len(pooled), 32)

        # 2. Defensive handling of unpermuted 2D patch tokens [196, 32]
        feat_2d = np.ones((196, 32), dtype=np.float32)
        sub_feat_2d, pooled_2d = slice_feature_map(feat_2d, (0, 0, 100, 100), 200, 200)
        self.assertEqual(sub_feat_2d.ndim, 3)
        self.assertEqual(len(pooled_2d), 32)

        # 3. None feature map raises RuntimeError
        with self.assertRaises(RuntimeError):
            slice_feature_map(None, (0, 0, 100, 100), 200, 200)

    def test_strict_no_fallback_on_missing_features(self):
        from mssearch.stages.stage3_partitioning import partition_candidate_images
        from mssearch.config import MSSearchConfig

        # Mock detector with no vision features
        mock_det = MagicMock()
        mock_det.batch_inference.return_value = (None, {}, [1])

        cfg = MSSearchConfig(partition_mode="sgap_only")
        with self.assertRaises(RuntimeError):
            partition_candidate_images(
                candidate_images=[self.dummy_img],
                stage2_feature_cache={},
                config=cfg,
                active_objects=["car"],
                sam_model=mock_det,
            )

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

        with patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.AutoModelForCausalLM.from_pretrained"):
            det_f2 = load_detector("florence2", device="cpu")
            self.assertIsInstance(det_f2, BaseDetector)

        with patch("transformers.Owlv2Processor.from_pretrained"), \
             patch("transformers.Owlv2ForObjectDetection.from_pretrained"):
            det_owl = load_detector("owlv2", device="cpu")
            self.assertIsInstance(det_owl, BaseDetector)

        with patch("transformers.AutoProcessor.from_pretrained"), \
             patch("transformers.AutoModel.from_pretrained"):
            sc_clip = load_visual_scorer("clip_b32", device="cpu")
            self.assertIsInstance(sc_clip, BaseVisualScorer)

            sc_l14 = load_visual_scorer("clip_l14", device="cpu")
            self.assertIsInstance(sc_l14, BaseVisualScorer)

            sc_sig2_384 = load_visual_scorer("siglip2_384", device="cpu")
            self.assertIsInstance(sc_sig2_384, BaseVisualScorer)

            sc_sig2_512 = load_visual_scorer("siglip2_512", device="cpu")
            self.assertIsInstance(sc_sig2_512, BaseVisualScorer)

    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("transformers.AutoModel.from_pretrained")
    def test_eva02_scorer_contract(self, mock_model_cls, mock_proc_cls):
        mock_proc = MagicMock()
        mock_model = MagicMock()
        mock_proc_cls.return_value = mock_proc
        mock_model_cls.return_value.to.return_value.eval.return_value = mock_model
        mock_proc.return_value = {"pixel_values": torch.ones(2, 3, 336, 336)}
        mock_model.get_text_features.return_value = torch.ones(2, 16)
        mock_model.get_image_features.return_value = torch.ones(2, 16)

        scorer = EVA02VisualScorer(model_name="QuanSun/EVA-CLIP", device="cpu")
        crops = [Image.new("RGB", (64, 64)) for _ in range(2)]
        scores = scorer.score_crops(crops, ["car", "dog"])
        self.assertEqual(len(scores), 2)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_sam3_adapter_contract(self):
        mock_engine = MagicMock()
        mock_engine.batch_inference.return_value = (None, {1: {"boxes": [[0, 0, 10, 10]], "scores": [0.9]}}, [1])
        mock_engine.batch_inference_multi_images.return_value = [(None, {}, [1])]

        from models.sam3_adapter import SAM3Detector
        det = SAM3Detector(model_path="dummy.pt", device="cpu", engine=mock_engine)
        _, res, tids = det.batch_inference(self.dummy_img, ["car"])
        self.assertEqual(tids, [1])
        self.assertIn(1, res)
        multi_res = det.batch_inference_multi_images([self.dummy_img], ["car"])
        self.assertEqual(len(multi_res), 1)

    def test_gate_threshold_configuration(self):
        from mssearch.config import MSSearchConfig
        # Default (0.5)
        cfg_def = MSSearchConfig()
        self.assertAlmostEqual(cfg_def.gate_threshold, 0.5)
        self.assertAlmostEqual(cfg_def.latent_confidence_threshold, 0.0)
        self.assertAlmostEqual(cfg_def.existence_confidence_threshold, 0.0)

        # 0.3 threshold (high recall: 2*(0.3 - 0.5) = -0.4)
        cfg_03 = MSSearchConfig(gate_threshold=0.3)
        self.assertAlmostEqual(cfg_03.gate_threshold, 0.3)
        self.assertAlmostEqual(cfg_03.latent_confidence_threshold, -0.4)
        self.assertAlmostEqual(cfg_03.existence_confidence_threshold, -0.4)

        # 0.7 threshold (high precision: 2*(0.7 - 0.5) = +0.4)
        cfg_07 = MSSearchConfig(gate_threshold=0.7)
        self.assertAlmostEqual(cfg_07.gate_threshold, 0.7)
        self.assertAlmostEqual(cfg_07.latent_confidence_threshold, 0.4)
        self.assertAlmostEqual(cfg_07.existence_confidence_threshold, 0.4)


if __name__ == "__main__":
    unittest.main()
