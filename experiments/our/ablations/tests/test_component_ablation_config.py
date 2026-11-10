"""
Unit tests for Component Increment Ablation suite.
Verifies config parameter mapping, helper formatting, and pipeline routing logic.
"""

import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

from mssearch.config import MSSearchConfig
from experiments.our.ablations.eval_runner_utils import (
    resolve_ground_truth,
    format_multi_image_prompt,
    parse_baseline_json_response,
    run_qwen_direct_multi_image,
)
from experiments.our.ablations.eval_models_ablation import (
    load_ablation_engines,
    run_ablation_pipeline,
)


class TestComponentAblationConfig(unittest.TestCase):

    def test_resolve_ground_truth(self):
        self.assertEqual(resolve_ground_truth({"answer": "red"}), ["red"])
        self.assertEqual(resolve_ground_truth({"answer": ["red", "blue"]}), ["red", "blue"])
        self.assertEqual(resolve_ground_truth({"answer": {"list": ["dog", "cat"]}}), ["dog", "cat"])
        self.assertEqual(resolve_ground_truth({}), [])

    def test_format_multi_image_prompt(self):
        prompt = format_multi_image_prompt(3, "What color is the car?")
        self.assertIn("Image 1: <image>", prompt)
        self.assertIn("Image 2: <image>", prompt)
        self.assertIn("Image 3: <image>", prompt)
        self.assertIn("Question: What color is the car?", prompt)
        self.assertIn('"image_ids"', prompt)
        self.assertIn('"answer"', prompt)

    def test_parse_baseline_json_response(self):
        # Clean JSON in markdown block
        raw = '```json\n{"image_ids": [2], "answer": "blue sedan"}\n```'
        meta = [{"image_id": "imgA"}, {"image_id": "imgB"}, {"image_id": "imgC"}]
        raw_ids, actual_ids, indices, ans = parse_baseline_json_response(raw, meta)
        self.assertEqual(raw_ids, [2])
        self.assertEqual(actual_ids, ["imgB"])
        self.assertEqual(indices, [2])
        self.assertEqual(ans, "blue sedan")

        # Parsing without candidate_meta should still parse integer IDs into indices
        raw_no_meta = '{"image_ids": [3], "answer": "yellow taxi"}'
        r_ids, a_ids, idxs, ans_nm = parse_baseline_json_response(raw_no_meta)
        self.assertEqual(r_ids, [3])
        self.assertEqual(idxs, [3])
        self.assertEqual(ans_nm, "yellow taxi")

        # Fallback regex when invalid JSON
        raw_fallback = 'I think the answer is: "answer": "golden retriever", "image_id": 1'
        raw_ids, actual_ids, indices, ans = parse_baseline_json_response(raw_fallback, meta)
        self.assertEqual(raw_ids, ["1"])
        self.assertEqual(actual_ids, ["imgA"])
        self.assertEqual(indices, [1])
        self.assertEqual(ans, "golden retriever")

    def test_component_config_mappings(self):
        # 1. Rigid Quadtree + Single Crop
        with patch("mssearch.models.ModelQwenVL", MagicMock()), \
             patch("experiments.our.ablations.models.load_detector", MagicMock()), \
             patch("experiments.our.ablations.models.load_visual_scorer", MagicMock()), \
             patch.dict("sys.modules", {"spacy": MagicMock()}):
            engines_rigid = load_ablation_engines(component="qwen_fp_rigid_cg_single")
            cfg = engines_rigid["config"]
            self.assertEqual(cfg.partition_mode, "quadrant_only")
            self.assertFalse(cfg.enable_continuous_enclosure_subcanvas)
            self.assertFalse(cfg.enable_spatial_compressed_canvas)

            # 2. SGAP + Single Crop
            engines_sgap = load_ablation_engines(component="qwen_fp_sgap_cg_single")
            cfg = engines_sgap["config"]
            self.assertEqual(cfg.partition_mode, "sgap_only")
            self.assertFalse(cfg.enable_continuous_enclosure_subcanvas)
            self.assertFalse(cfg.enable_spatial_compressed_canvas)

            # 3. ASP + Single Crop
            engines_asp = load_ablation_engines(component="qwen_fp_asp_cg_single")
            cfg = engines_asp["config"]
            self.assertEqual(cfg.partition_mode, "hybrid")
            self.assertFalse(cfg.enable_continuous_enclosure_subcanvas)
            self.assertFalse(cfg.enable_spatial_compressed_canvas)

            # 4. ASP + Aggregation (Ours)
            engines_agg = load_ablation_engines(component="qwen_fp_asp_cg_agg")
            cfg = engines_agg["config"]
            self.assertEqual(cfg.partition_mode, "hybrid")
            self.assertTrue(cfg.enable_continuous_enclosure_subcanvas)
            self.assertTrue(cfg.enable_spatial_compressed_canvas)

    @patch("experiments.our.ablations.eval_models_ablation.run_qwen_direct_multi_image")
    def test_run_ablation_pipeline_qwen_direct(self, mock_qwen):
        mock_qwen.return_value = ("white bird", [1], [2])
        images = [Image.new("RGB", (100, 100)), Image.new("RGB", (100, 100))]
        engines = {"component": "qwen", "zoom_model": MagicMock(), "config": MSSearchConfig()}
        sample = {"sample_id": "test_1", "question": "What bird is shown?"}

        resp = run_ablation_pipeline(engines, images, sample, 0)
        self.assertEqual(resp["primary_answer"], "white bird")
        self.assertEqual(resp["primary_evidence_image"], 1)
        self.assertEqual(resp["telemetry"]["route"], "direct_baseline")
        self.assertEqual(resp["telemetry"]["stage5_mode"], "direct_baseline")

    @patch("experiments.our.ablations.eval_models_ablation.run_qwen_direct_multi_image")
    def test_run_ablation_pipeline_qwen_direct_empty_pred(self, mock_qwen):
        mock_qwen.return_value = ("not found", [], [])
        images = [Image.new("RGB", (100, 100)), Image.new("RGB", (100, 100))]
        engines = {"component": "qwen", "zoom_model": MagicMock(), "config": MSSearchConfig()}
        sample = {"sample_id": "test_empty", "question": "What is here?"}

        resp = run_ablation_pipeline(engines, images, sample, 0)
        self.assertEqual(resp["primary_answer"], "not found")
        self.assertIsNone(resp["primary_evidence_image"])
        self.assertEqual(resp["evidence_images"], [])

    @patch("mssearch.stages.stage1_parser.parse_question_to_active_objects")
    @patch("mssearch.stages.stage2_fast_path.run_tool_based_fast_path")
    def test_run_ablation_pipeline_fast_path_triggered(self, mock_fp, mock_parser):
        mock_result = MagicMock()
        mock_result.triggered = True
        mock_result.answer = "red bicycle"
        mock_result.confidence = 0.85
        mock_result.evidence_images = [0]
        mock_result.evidence_img_idx = 0
        mock_result.detected_boxes = [(10, 10, 50, 50)]
        mock_result.raw_response = "red bicycle"
        mock_fp.return_value = mock_result

        images = [Image.new("RGB", (100, 100))]
        engines = {
            "component": "qwen_fast_path", "zoom_model": MagicMock(),
            "sam_model": MagicMock(), "nlp_model": MagicMock(), "config": MSSearchConfig()
        }
        sample = {"sample_id": "test_fp", "question": "What is parked?"}
        resp = run_ablation_pipeline(engines, images, sample, 0)

        self.assertEqual(resp["primary_answer"], "red bicycle")
        self.assertEqual(resp["telemetry"]["route"], "tool_based_fast_path")
        self.assertEqual(resp["telemetry"]["stage5_mode"], "fast_path")

    @patch("experiments.our.ablations.eval_models_ablation.run_qwen_direct_multi_image")
    @patch("mssearch.stages.stage1_parser.parse_question_to_active_objects")
    @patch("mssearch.stages.stage2_fast_path.run_tool_based_fast_path")
    def test_run_ablation_pipeline_fast_path_fallback(self, mock_fp, mock_parser, mock_qwen):
        mock_result = MagicMock()
        mock_result.triggered = False
        mock_fp.return_value = mock_result
        mock_qwen.return_value = ("fallback answer", [0], [1])

        images = [Image.new("RGB", (100, 100))]
        engines = {
            "component": "qwen_fast_path", "zoom_model": MagicMock(),
            "sam_model": MagicMock(), "nlp_model": MagicMock(), "config": MSSearchConfig()
        }
        sample = {"sample_id": "test_fp_fb", "question": "Where is the cat?"}
        resp = run_ablation_pipeline(engines, images, sample, 0)

        self.assertEqual(resp["primary_answer"], "fallback answer")
        self.assertEqual(resp["telemetry"]["route"], "fast_path_direct_fallback")
        self.assertEqual(resp["telemetry"]["stage5_mode"], "direct_baseline")

    def test_run_qwen_direct_multi_image_uncapped_resolution_and_image_ids(self):
        mock_engine = MagicMock()
        mock_engine.device = "cpu"
        mock_engine.processor.apply_chat_template.return_value = "formatted prompt"

        processor_kwargs = {}
        def mock_processor_call(*args, **kwargs):
            processor_kwargs.update(kwargs)
            res = MagicMock()
            res.input_ids = MagicMock()
            res.input_ids.__getitem__.return_value = [1, 2]
            res.to.return_value = res
            return res

        mock_engine.processor.side_effect = mock_processor_call
        mock_engine.model.generate.return_value = [[1, 2, 3, 4]]
        mock_engine.processor.tokenizer.decode.return_value = '```json\n{"image_ids": [2], "answer": "silver car"}\n```'

        images = [Image.new("RGB", (500, 500)), Image.new("RGB", (500, 500)), Image.new("RGB", (500, 500))]
        ans, pred_0based, pred_indices = run_qwen_direct_multi_image(mock_engine, images, "What car is there?")

        # Resolution must be uncapped (no max_pixels passed in initial generation)
        self.assertNotIn("max_pixels", processor_kwargs)
        # Image IDs and 0-based indices must be correctly resolved for candidate meta
        self.assertEqual(ans, "silver car")
        self.assertEqual(pred_indices, [2])
        self.assertEqual(pred_0based, [1])

if __name__ == "__main__":
    unittest.main()
