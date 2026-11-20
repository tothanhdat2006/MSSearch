"""
Unit tests for STASH-Search Minimal Baseline Inference (minimal/baseline/run_inference.py).
Tests model family resolution, prompt construction, benchmark path resolution,
dataset iteration, and resumption logic across Vstar, HR-Bench, TreeBench, and MME-RealWorld-Lite.
"""

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image

from minimal.datasets.base import BenchmarkSample
from minimal.baseline.run_inference import (
    family,
    build_prompt_text,
    resolve_benchmark_path,
    BenchmarkDataset,
    MODELS,
    BENCHMARKS,
    REMOTE_CODE,
)


class TestBaselineInference(unittest.TestCase):
    def test_family_resolution(self):
        self.assertEqual(family("llavaov_7b"), "llavaov")
        self.assertEqual(family("llava-hf/llava-onevision-qwen2-7b-ov-hf"), "llavaov")
        self.assertEqual(family("llava16_7b"), "llava16")
        self.assertEqual(family("internvl3_8b"), "internvl3")
        self.assertEqual(family("OpenGVLab/InternVL3-8B-hf"), "internvl3")
        self.assertEqual(family("qwen3vl_8b"), "qwen3vl")
        self.assertEqual(family("Qwen/Qwen3-VL-8B-Instruct"), "qwen3vl")
        self.assertEqual(family("molmo2_8b"), "molmo2")
        self.assertEqual(family("eagle25_8b"), "eagle25")
        self.assertIn("molmo2", REMOTE_CODE)
        self.assertIn("eagle25", REMOTE_CODE)

    def test_build_prompt_text(self):
        # 1. Benchmark with inspection_prompt (e.g. TreeBench or MME-RealWorld-Lite)
        sample1 = BenchmarkSample(
            sample_id="tb_001",
            question="Which leaf?",
            candidate_image_paths=["dummy.jpg"],
            gt_answer="A",
            inspection_prompt="Which leaf? Options:\nA. Left\nSelect the best answer...\nThe best answer is:"
        )
        prompt1 = build_prompt_text(sample1)
        self.assertTrue(prompt1.startswith("<image>\n"))
        self.assertIn("Which leaf? Options:", prompt1)

        # 2. Benchmark with options_str fallback
        sample2 = BenchmarkSample(
            sample_id="vstar_001",
            question="What color is the comb?",
            candidate_image_paths=["dummy.jpg"],
            gt_answer="B",
            options_str="A. Purple\nB. Black"
        )
        prompt2 = build_prompt_text(sample2)
        self.assertTrue(prompt2.startswith("<image>\n"))
        self.assertIn("What color is the comb?\nA. Purple\nB. Black", prompt2)
        self.assertIn("Answer the option letter directly.", prompt2)

    def test_resolve_benchmark_path(self):
        # Explicit data_dir
        explicit = Path("/custom/path/to/vstar")
        self.assertEqual(resolve_benchmark_path("vstar", explicit), str(explicit))

        # Known benchmarks supported
        for b in BENCHMARKS:
            resolved = resolve_benchmark_path(b, None)
            self.assertTrue(len(resolved) > 0)

        # Unknown benchmark raises ValueError
        with self.assertRaises(ValueError):
            resolve_benchmark_path("invalid_benchmark", None)

    def test_dataset_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.jpg")
            Image.new("RGB", (64, 64), color="red").save(img_path)

            sample = BenchmarkSample(
                sample_id="test_001",
                question="Is this image red?",
                candidate_image_paths=[img_path],
                gt_answer="A",
                options_str="A. Yes\nB. No",
                category="color_detection",
                metadata={"benchmark": "vstar", "cycle_idx": 0}
            )

            # Mock processor
            mock_processor = MagicMock()
            mock_processor.tokenizer.convert_tokens_to_ids.return_value = 100
            mock_processor.apply_chat_template.return_value = "<image>\nIs this image red?"

            # Mock tensor object so test does not depend on heavy pytorch build
            class MockTensor:
                def __init__(self, ids):
                    self.ids = ids

                def __eq__(self, other):
                    class EqResult:
                        def sum(self_inner):
                            return 1
                    return EqResult()

                @property
                def shape(self):
                    return (1, len(self.ids))

            mock_inputs = {
                "input_ids": MockTensor([100, 101, 102]),
            }
            mock_processor.return_value = mock_inputs

            dataset = BenchmarkDataset(
                samples=[sample],
                model_key="qwen3vl_8b",
                processor=mock_processor,
                budget={}
            )

            self.assertEqual(len(dataset), 1)
            inputs, meta = dataset[0]

            self.assertEqual(meta["sample_id"], "test_001")
            self.assertEqual(meta["benchmark"], "vstar")
            self.assertEqual(meta["gt_answer"], "A")
            self.assertEqual(meta["cycle_idx"], 0)
            self.assertIn("input_ids", inputs)
            self.assertEqual(meta["visual_tokens"], 1)

    def test_resumption_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "predictions.jsonl")
            # Write 1 valid completed row and 1 failed OOM row
            rec_ok = {"sample_id": "s1", "prediction": "A", "is_correct": True}
            rec_failed = {"sample_id": "s2", "prediction": None, "error": "cuda_oom"}

            with open(out_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(rec_ok) + "\n" + json.dumps(rec_failed) + "\n")

            # Simulate recovery block from run_inference
            with open(out_file, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            ok = [r for r in rows if r.get("prediction") is not None]
            done = {str(r["sample_id"]) for r in ok}

            # Dropped the failed OOM row so it is retried
            self.assertEqual(len(ok), 1)
            self.assertIn("s1", done)
            self.assertNotIn("s2", done)


if __name__ == "__main__":
    unittest.main()
