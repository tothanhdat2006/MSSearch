"""
Unit tests for STASH-Search visualization and qualitative telemetry engine.
Verifies multi-stage bounding box overlays, candidate crop saving, composite overview generation,
and engine integration for STASH-Bench multi-image galleries.
"""

import os
import json
import shutil
import tempfile
import unittest
from typing import List
from PIL import Image

from minimal.config import STASHConfig
from minimal.stages.quadtree_partitioning import BoundedPatch
from minimal.stages.visual_screening import CandidateExtraction
from minimal.engine import STASHSearchEngine
from minimal.visualization import (
    render_stage2_quadtree_overlay,
    render_stage3_screened_overlay,
    render_stage4_evidence_overlay,
    generate_pipeline_overview_composite,
    save_sample_visualization
)


class MockDualEncoder:
    def score_crops(self, crops: List[Image.Image], active_objects: List[str]) -> List[float]:
        # Return synthetic alignment scores: first crop 0.35, rest 0.15
        return [0.35 if i == 0 else 0.15 for i in range(len(crops))]


class MockMLLM:
    def generate_response(self, image: Image.Image, question: str, max_new_tokens: int = 512) -> str:
        return "DL-4029"

    def generate_response_batch(self, images: List[Image.Image], question: str, max_new_tokens: int = 512) -> List[str]:
        return ["DL-4029" if i == 0 else "unclear" for i in range(len(images))]


class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.img1 = Image.new("RGB", (600, 400), color=(100, 150, 200))
        self.img2 = Image.new("RGB", (600, 400), color=(200, 100, 150))
        self.gallery = [self.img1, self.img2]

        self.patches = [
            BoundedPatch(img_idx=0, box=(10, 10, 200, 200), depth=1, alignment_score=0.35, entropy=0.5),
            BoundedPatch(img_idx=0, box=(210, 10, 400, 200), depth=1, alignment_score=0.20, entropy=0.4),
            BoundedPatch(img_idx=1, box=(50, 50, 300, 300), depth=1, alignment_score=0.15, entropy=0.45),
        ]

        self.extractions = [
            CandidateExtraction(
                img_idx=0,
                box=(10, 10, 200, 200),
                expanded_box=(5, 5, 210, 210),
                alignment_score=0.35,
                crop_area=40000,
                raw_answer="<answer>DL-4029</answer>",
                final_answer="DL-4029"
            ),
            CandidateExtraction(
                img_idx=1,
                box=(50, 50, 300, 300),
                expanded_box=(40, 40, 310, 310),
                alignment_score=0.15,
                crop_area=72900,
                raw_answer="unclear",
                final_answer="unclear"
            )
        ]

        self.crops = [
            self.img1.crop((5, 5, 210, 210)),
            self.img2.crop((40, 40, 310, 310))
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_render_stage2_quadtree_overlay(self):
        overlay = render_stage2_quadtree_overlay(self.img1, self.patches, img_idx=0)
        self.assertIsInstance(overlay, Image.Image)
        self.assertEqual(overlay.size, self.img1.size)

    def test_render_stage3_screened_overlay(self):
        overlay = render_stage3_screened_overlay(self.img1, self.extractions, img_idx=0)
        self.assertIsInstance(overlay, Image.Image)
        self.assertEqual(overlay.size, self.img1.size)

    def test_render_stage4_evidence_overlay(self):
        overlay = render_stage4_evidence_overlay(
            image=self.img1,
            winning_extraction=self.extractions[0],
            other_extractions=self.extractions,
            img_idx=0,
            is_gt_image=True
        )
        self.assertIsInstance(overlay, Image.Image)
        self.assertEqual(overlay.size, self.img1.size)

    def test_generate_pipeline_overview_composite(self):
        composite = generate_pipeline_overview_composite(
            gallery_images=self.gallery,
            sample_id="stash_test_001",
            question="What is the registration number of the boat?",
            active_objects=["boat"],
            extractions=self.extractions,
            crops_to_read=self.crops,
            primary_answer="DL-4029",
            primary_evidence_img=0,
            gt_answer="DL-4029",
            gt_image_idx=0,
            timings={"total_latency_sec": 1.25},
            consensus_details={
                "clusters": [{"representative": "DL-4029", "count": 1, "weighted_score": 1.35}]
            }
        )
        self.assertIsInstance(composite, Image.Image)
        self.assertGreaterEqual(composite.width, 1000)
        self.assertGreaterEqual(composite.height, 800)

    def test_save_sample_visualization(self):
        trace = save_sample_visualization(
            vis_root_dir=self.temp_dir,
            benchmark_name="stash",
            pool_mode="N_5",
            sample_id="sample_42",
            gallery_images=self.gallery,
            question="What is the registration number of the boat?",
            active_objects=["boat"],
            candidate_patches=self.patches,
            extractions=self.extractions,
            crops_to_read=self.crops,
            primary_answer="DL-4029",
            primary_evidence_img=0,
            consensus_details={
                "clusters": [{"representative": "DL-4029", "count": 1, "weighted_score": 1.35}],
                "winning_extraction_idx": 0
            },
            timings={"total_latency_sec": 0.5},
            gt_answer="DL-4029",
            gt_image_idx=0
        )

        expected_sample_dir = os.path.join(self.temp_dir, "stash", "N_5", "sample_42")
        self.assertTrue(os.path.isdir(expected_sample_dir))

        # Check subdirectories
        self.assertTrue(os.path.isfile(os.path.join(expected_sample_dir, "00_inputs", "query.json")))
        self.assertTrue(os.path.isfile(os.path.join(expected_sample_dir, "00_inputs", "gallery_img_00_GT.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(expected_sample_dir, "00_inputs", "gallery_img_01.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(expected_sample_dir, "01_stage2_partitioning", "quadtree_overlay_img_00.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(expected_sample_dir, "02_stage3_screening", "screened_boxes_overlay_img_00.jpg")))
        self.assertTrue(os.path.isdir(os.path.join(expected_sample_dir, "02_stage3_screening", "crops")))
        self.assertTrue(os.path.isfile(os.path.join(expected_sample_dir, "pipeline_overview.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(expected_sample_dir, "pipeline_trace.json")))

        # Check trace content
        self.assertEqual(trace["sample_id"], "sample_42")
        self.assertEqual(trace["predicted_answer"], "DL-4029")
        self.assertEqual(trace["predicted_evidence_image"], 0)
        self.assertTrue(trace["is_correct_answer"])
        self.assertTrue(trace["is_correct_image"])

    def test_engine_search_with_save_vis(self):
        cfg = STASHConfig(
            save_vis=True,
            vis_dir=self.temp_dir,
            verbose=False,
            min_plateau_size=100,
            max_plateau_size=800
        )
        engine = STASHSearchEngine(
            mllm=MockMLLM(),
            dual_encoder=MockDualEncoder(),
            config=cfg
        )

        res = engine.search(
            candidate_images=self.gallery,
            question="What is the registration number of the boat?",
            sample_id="test_engine_001",
            pool_mode="N_5",
            gt_answer="DL-4029",
            gt_image_idx=0
        )

        self.assertIn("vis_trace", res)
        self.assertIsNotNone(res["vis_trace"])
        self.assertEqual(res["primary_answer"], "DL-4029")
        self.assertIn("consensus", res["telemetry"])

        expected_sample_dir = os.path.join(self.temp_dir, "stash", "N_5", "test_engine_001")
        self.assertTrue(os.path.isdir(expected_sample_dir))
        self.assertTrue(os.path.isfile(os.path.join(expected_sample_dir, "pipeline_overview.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(expected_sample_dir, "pipeline_trace.json")))

    def test_zero_overhead_when_vis_disabled(self):
        vis_dir = os.path.join(self.temp_dir, "should_stay_empty")
        cfg = STASHConfig(
            save_vis=False,
            vis_dir=vis_dir,
            verbose=False,
            min_plateau_size=100,
            max_plateau_size=800
        )
        engine = STASHSearchEngine(
            mllm=MockMLLM(),
            dual_encoder=MockDualEncoder(),
            config=cfg
        )

        res = engine.search(
            candidate_images=self.gallery,
            question="What is the registration number of the boat?",
            sample_id="test_disabled_001",
            pool_mode="N_5"
        )

        self.assertIsNone(res["vis_trace"])
        self.assertFalse(os.path.exists(vis_dir))

    def test_eval_runner_with_stash_vis(self):
        from unittest.mock import patch
        from minimal.datasets.base import BenchmarkSample
        from minimal.eval_runner import main

        sample_img_path = os.path.join(self.temp_dir, "test_img.jpg")
        self.img1.save(sample_img_path)

        mock_sample = BenchmarkSample(
            sample_id="stash_runner_001",
            question="What is the registration number of the boat?",
            candidate_image_paths=[sample_img_path],
            gt_answer="DL-4029",
            gt_image_idx=0
        )

        out_jsonl = os.path.join(self.temp_dir, "output.jsonl")
        vis_out_dir = os.path.join(self.temp_dir, "vis_runner")

        with patch("minimal.eval_runner.load_benchmark_dataset", return_value=[mock_sample]), \
             patch("minimal.eval_runner.STASHSearchEngine.from_pretrained", return_value=STASHSearchEngine(MockMLLM(), MockDualEncoder(), STASHConfig(verbose=False, min_plateau_size=100, max_plateau_size=800))), \
             patch("sys.argv", ["eval_runner.py", "--benchmark", "stash", "--pool_mode", "N_1", "--output_file", out_jsonl, "--save_vis", "--vis_dir", vis_out_dir, "--device", "cpu"]):
            main()

        self.assertTrue(os.path.isfile(out_jsonl))
        with open(out_jsonl, "r") as f:
            line = json.loads(f.readline())
            self.assertEqual(line["sample_id"], "stash_runner_001")
            self.assertIn("telemetry", line)
            self.assertIn("vis_trace", line["telemetry"])

        expected_vis_dir = os.path.join(vis_out_dir, "stash", "N_1", "stash_runner_001")
        self.assertTrue(os.path.isdir(expected_vis_dir))
        self.assertTrue(os.path.isfile(os.path.join(expected_vis_dir, "pipeline_overview.jpg")))
        self.assertTrue(os.path.isfile(os.path.join(expected_vis_dir, "pipeline_trace.json")))


if __name__ == "__main__":
    unittest.main()
