"""
Unit tests for STASH-Search Minimal Dataset Loaders & Parsers.
Validates Vstar, HR-Bench, and STASH-Bench schemas and option extractors.
"""

import unittest
import json
import tempfile
import os

from minimal.datasets.base import BenchmarkSample
from minimal.datasets.vstar import format_options
from minimal.datasets import load_benchmark_dataset
from minimal.eval_runner import extract_predicted_option, normalize_text


class TestDatasetLoaders(unittest.TestCase):
    def test_option_extractor_formats(self):
        # 1. Direct letter at start
        self.assertEqual(extract_predicted_option("A. 27B"), "A")
        self.assertEqual(extract_predicted_option("(B) Blue shirt"), "B")
        self.assertEqual(extract_predicted_option("C: Front entrance"), "C")
        self.assertEqual(extract_predicted_option("D."), "D")
        self.assertEqual(extract_predicted_option("(E) Fifth choice"), "E")
        self.assertEqual(extract_predicted_option("F. Sixth option"), "F")

        # 2. Exact single letter
        self.assertEqual(extract_predicted_option("A"), "A")
        self.assertEqual(extract_predicted_option("b"), "B")

        # 3. Final answer pattern
        self.assertEqual(extract_predicted_option("The correct option is (C)"), "C")
        self.assertEqual(extract_predicted_option("Final Answer: D"), "D")

        # 4. Refusals / empty
        self.assertIsNone(extract_predicted_option("none"))
        self.assertIsNone(extract_predicted_option("no answer found"))

    def test_normalize_text(self):
        self.assertEqual(normalize_text("A red apple."), "red apple")
        self.assertEqual(normalize_text("The blue jacket!"), "blue jacket")
        self.assertEqual(normalize_text("an elephant"), "elephant")

    def test_vstar_schema_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirs and annotation_vstar.json matching user Image 1
            os.makedirs(os.path.join(tmpdir, "direct_attributes"), exist_ok=True)
            os.makedirs(os.path.join(tmpdir, "relative_position"), exist_ok=True)

            vstar_data = [
                {
                    "target_object": ["comb"],
                    "bbox": [[[10, 10, 50, 50]]],
                    "question": "What is the color of the comb?",
                    "options": [
                        "The color of the comb is purple.",
                        "The color of the comb is black.",
                        "The color of the comb is red.",
                        "The color of the comb is brown."
                    ],
                    "input_image": "direct_attributes/sa_7429.jpg",
                    "test_type": "direct_attributes",
                    "answer_type": "logits_match"
                }
            ]
            anno_path = os.path.join(tmpdir, "annotation_vstar.json")
            with open(anno_path, "w", encoding="utf-8") as f:
                json.dump(vstar_data, f)

            samples = load_benchmark_dataset(
                benchmark="vstar",
                dataset_dir_or_path=tmpdir
            )

            self.assertEqual(len(samples), 1)
            s = samples[0]
            self.assertEqual(s.question, "What is the color of the comb?")
            self.assertEqual(s.gt_answer, "A")
            self.assertEqual(s.category, "direct_attributes")
            self.assertEqual(s.target_hint, "comb")
            self.assertIn("A. The color of the comb is purple.", s.options_str)
            self.assertIn("B. The color of the comb is black.", s.options_str)

    def test_hrbench_schema_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create image folder and annotation_hr-bench_8k.json matching user Image 2
            os.makedirs(os.path.join(tmpdir, "image"), exist_ok=True)

            hr_data = [
                {
                    "index": 0,
                    "question": "What is the number displayed above the entrance?",
                    "answer": ["A", "B", "C", "D"],
                    "category": "single",
                    "cycle_category": 0,
                    "options": [
                        "A. 27B B. 27D C. 27E D. 37B ",
                        "A. 37B B. 27B C. 27E D. 27D ",
                        "A. 37B B. 27D C. 27B D. 27E ",
                        "A. 27E B. 27D C. 37B D. 27B "
                    ],
                    "answer_type": "option_list",
                    "input_image": "image/0.jpg"
                }
            ]
            anno_path = os.path.join(tmpdir, "annotation_hr-bench_8k.json")
            with open(anno_path, "w", encoding="utf-8") as f:
                json.dump(hr_data, f)

            samples = load_benchmark_dataset(
                benchmark="hrbench8k",
                dataset_dir_or_path=tmpdir
            )

            self.assertEqual(len(samples), 1)
            s = samples[0]
            self.assertEqual(s.question, "What is the number displayed above the entrance?")
            self.assertEqual(s.gt_answer, "A")
            self.assertEqual(s.category, "single")
            self.assertIn("A. 27B", s.options_str)

    def test_stash_bench_gallery_keys_and_ordering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic annotation and gallery order
            anno_data = {
                "samples": [
                    {
                        "sample_id": "000",
                        "question": "What is the registration number of the boat?",
                        "gt_image": "boat_gt.jpg",
                        "other_images": {
                            "N5": ["distractor1.jpg", "distractor2.jpg", "distractor3.jpg", "distractor4.jpg"]
                        },
                        "reference_answer": "NY-1234",
                        "category": "Attribute"
                    }
                ]
            }
            order_data = {
                "samples": {
                    "000": {
                        "N5": {
                            "image_ids": ["distractor1.jpg", "distractor2.jpg", "boat_gt.jpg", "distractor3.jpg", "distractor4.jpg"],
                            "gt_index": 3
                        }
                    }
                }
            }

            anno_path = os.path.join(tmpdir, "annotations_part1234_merged.json")
            order_path = os.path.join(tmpdir, "gallery_order_seed42.json")
            with open(anno_path, "w", encoding="utf-8") as f:
                json.dump(anno_data, f)
            with open(order_path, "w", encoding="utf-8") as f:
                json.dump(order_data, f)

            # Load with pool_mode="N_5" (should normalize "N_5" to "N5" and look up seed42 order)
            samples = load_benchmark_dataset(
                benchmark="stash",
                dataset_dir_or_path=anno_path,
                image_dir=tmpdir,
                pool_mode="N_5",
                gallery_order_path=order_path
            )

            self.assertEqual(len(samples), 1)
            s = samples[0]
            # Must load all 5 candidate images in the exact seed42 order
            self.assertEqual(len(s.candidate_image_paths), 5)
            self.assertTrue(s.candidate_image_paths[2].endswith("boat_gt.jpg"))
            # gt_image_idx must be 0-indexed: 3 - 1 = 2
            self.assertEqual(s.gt_image_idx, 2)

            # Test that missing order entry raises RuntimeError
            with self.assertRaises(RuntimeError):
                load_benchmark_dataset(
                    benchmark="stash",
                    dataset_dir_or_path=anno_path,
                    image_dir=tmpdir,
                    pool_mode="N_10",  # N10 is not in order_data
                    gallery_order_path=order_path
                )

    def test_treebench_schema_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = os.path.join(tmpdir, "images")
            os.makedirs(images_dir, exist_ok=True)
            # Create dummy image file
            with open(os.path.join(images_dir, "img_0001.jpg"), "w") as f:
                f.write("fake")

            sample_data = [
                {
                    "id": "tb_001",
                    "image": "img_0001.jpg",
                    "question": "Which leaf has an aphid?",
                    "options": {
                        "A": "Top left leaf",
                        "B": "Top right leaf",
                        "C": "Bottom left leaf",
                        "D": "Bottom right leaf",
                        "E": "Center stem",
                        "F": "None"
                    },
                    "ground_truth": "E",
                    "category": "Perception/Insect Detection"
                }
            ]
            anno_path = os.path.join(tmpdir, "annotation_treebench.json")
            with open(anno_path, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)

            samples = load_benchmark_dataset(
                benchmark="treebench",
                dataset_dir_or_path=tmpdir
            )

            self.assertEqual(len(samples), 1)
            s = samples[0]
            self.assertEqual(s.sample_id, "tb_001")
            self.assertEqual(s.gt_answer, "E")
            self.assertEqual(s.category, "Perception/Insect Detection")
            self.assertIn("A. Top left leaf", s.options_str)
            self.assertIn("E. Center stem", s.options_str)
            self.assertIn("F. None", s.options_str)
            self.assertTrue(os.path.isabs(s.candidate_image_paths[0]))

    def test_mme_realworld_schema_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            imgs_dir = os.path.join(tmpdir, "imgs")
            os.makedirs(imgs_dir, exist_ok=True)
            with open(os.path.join(imgs_dir, "real_001.jpg"), "w") as f:
                f.write("fake")

            sample_data = [
                {
                    "id": "mme_001",
                    "image": "real_001.jpg",
                    "question": "What is the speed limit shown?",
                    "options": [
                        "(A) 30 mph",
                        "(B) 40 mph",
                        "(C) 50 mph",
                        "(D) 60 mph",
                        "(E) 70 mph"
                    ],
                    "ground_truth": "C",
                    "category": "Reasoning",
                    "subtask": "Autonomous Driving",
                    "task": "Reasoning"
                }
            ]
            anno_path = os.path.join(tmpdir, "annotation_mme-realworld-lite.json")
            with open(anno_path, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)

            samples = load_benchmark_dataset(
                benchmark="mme_realworld_lite",
                dataset_dir_or_path=tmpdir
            )

            self.assertEqual(len(samples), 1)
            s = samples[0]
            self.assertEqual(s.sample_id, "mme_001")
            self.assertEqual(s.gt_answer, "C")
            self.assertEqual(s.category, "Reasoning/Autonomous Driving")
            self.assertIn("A. 30 mph", s.options_str)
            self.assertIn("E. 70 mph", s.options_str)
            self.assertTrue(os.path.isabs(s.candidate_image_paths[0]))

    def test_real_dataset_paths_if_present(self):
        # TreeBench real dataset test
        tb_dir = r"D:\MachineLearning\Datasets\treebench"
        if os.path.isdir(tb_dir):
            samples = load_benchmark_dataset("treebench", tb_dir)
            self.assertGreater(len(samples), 0)
            s = samples[0]
            self.assertTrue(os.path.exists(s.candidate_image_paths[0]))
            self.assertIn(s.gt_answer, ["A", "B", "C", "D", "E", "F"])

        # MME-RealWorld-Lite real dataset test
        mme_dir = r"D:\MachineLearning\Datasets\mme-realworld-lite"
        if os.path.isdir(mme_dir):
            samples = load_benchmark_dataset("mme_realworld_lite", mme_dir)
            self.assertGreater(len(samples), 0)
            s = samples[0]
            self.assertTrue(os.path.exists(s.candidate_image_paths[0]))
            self.assertIn(s.gt_answer, ["A", "B", "C", "D", "E"])


if __name__ == "__main__":
    unittest.main()
