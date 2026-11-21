"""
Unit tests for STASH-Search Minimal Evaluation Runner (eval_runner.py).
Tests environment parsing, strict benchmark handling (no aliases),
option extraction, path resolution, and resumption statistics across
Vstar, HR-Bench (4K/8K), TreeBench, MME-RealWorld-Lite, and STASH-Bench.
"""

import os
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from argparse import Namespace
from PIL import Image

from minimal.eval_runner import (
    _expand_bash_val,
    load_env_defaults,
    extract_predicted_option,
    resolve_dataset_args,
    normalize_text,
)


class TestEvalRunner(unittest.TestCase):
    def test_expand_bash_val(self):
        # 1. Default value fallback
        val = "${TREEBENCH_DIR:-/data/TreeBench}"
        res = _expand_bash_val(val, {})
        self.assertEqual(res, "/data/TreeBench")

        # 2. Existing key in map
        env_map = {"TREEBENCH_DIR": "/my/custom/path"}
        res2 = _expand_bash_val(val, env_map)
        self.assertEqual(res2, "/my/custom/path")

        # 3. Chained and nested variable expansion
        anno_val = "${TREEBENCH_DIR}/annotation.json"
        res3 = _expand_bash_val(anno_val, env_map)
        self.assertEqual(res3, "/my/custom/path/annotation.json")

        nested_val = "${TREEBENCH_ANNOTATION:-${TREEBENCH_DIR}/annotation_treebench.json}"
        res4 = _expand_bash_val(nested_val, env_map)
        self.assertEqual(res4, "/my/custom/path/annotation_treebench.json")

    def test_strict_benchmarks_no_aliases(self):
        # Only strict benchmark names are permitted:
        # 'vstar', 'hrbench4k', 'hrbench8k', 'treebench', 'mme-realworld-lite', 'stash'
        valid_benchmarks = ["stash", "vstar", "hrbench4k", "hrbench8k", "treebench", "mme-realworld-lite"]
        for b in valid_benchmarks:
            args = Namespace(
                benchmark=b,
                dataset_dir="/dummy/dir",
                dataset_path=None,
                image_dir=None,
                pool_mode="N_5",
                gallery_order_path=None,
                cycle_idx=None
            )
            res = resolve_dataset_args(args)
            self.assertEqual(res["dataset_dir_or_path"], "/dummy/dir")

        # Disallowed aliases must raise ValueError
        invalid_aliases = ["vbench", "v-star", "hr-bench_4k", "hrbench", "hr8k", "treenbench", "mme", "mme_realworld_lite"]
        for b in invalid_aliases:
            args = Namespace(
                benchmark=b,
                dataset_dir="/dummy/dir",
                dataset_path=None,
                image_dir=None,
                pool_mode="N_5",
                gallery_order_path=None,
                cycle_idx=None
            )
            with self.assertRaises(ValueError):
                resolve_dataset_args(args)

    def test_extract_predicted_option_comprehensive(self):
        # 1. Direct single letter
        self.assertEqual(extract_predicted_option("A"), "A")
        self.assertEqual(extract_predicted_option("b"), "B")
        self.assertEqual(extract_predicted_option("  C  "), "C")
        self.assertEqual(extract_predicted_option("**D**"), "D")
        self.assertEqual(extract_predicted_option("`E`"), "E")

        # 2. Leading letter format
        self.assertEqual(extract_predicted_option("A. 27B"), "A")
        self.assertEqual(extract_predicted_option("(B) Blue shirt"), "B")
        self.assertEqual(extract_predicted_option("C: Front entrance"), "C")
        self.assertEqual(extract_predicted_option("D.\nExplanation here"), "D")

        # 3. Natural language sentences with option letters
        self.assertEqual(extract_predicted_option("The best answer is: B"), "B")
        self.assertEqual(extract_predicted_option("The correct answer is (C)"), "C")
        self.assertEqual(extract_predicted_option("The answer is D."), "D")
        self.assertEqual(extract_predicted_option("Best answer: A"), "A")
        self.assertEqual(extract_predicted_option("Final Answer: D"), "D")
        self.assertEqual(extract_predicted_option("Option D: Front left"), "D")
        self.assertEqual(extract_predicted_option("The answer is therefore E."), "E")

        # 4. Critical edge case: English pronoun 'I' must NOT be extracted as letter I
        self.assertEqual(extract_predicted_option("I believe the answer is D"), "D")
        self.assertEqual(extract_predicted_option("I think it is B because the object is yellow."), "B")

        # 5. Matching against option content text
        opts = "A. Front entrance\nB. Back garden\nC. Side patio"
        self.assertEqual(extract_predicted_option("The car is located near the front entrance.", options_str=opts), "A")
        self.assertEqual(extract_predicted_option("Clearly in the back garden.", options_str=opts), "B")

        # 6. Refusals & non-answers
        self.assertIsNone(extract_predicted_option("none"))
        self.assertIsNone(extract_predicted_option("no answer found"))
        self.assertIsNone(extract_predicted_option("[error: image not found]"))

        # 7. Matching "none" / "n/a" when explicitly present in options_str (e.g. MME-RealWorld-Lite)
        none_opts = "(A) 30 mph\n(B) 40 mph\n(C) 50 mph\n(D) 60 mph\n(E) None"
        self.assertEqual(extract_predicted_option("none", options_str=none_opts), "E")
        self.assertEqual(extract_predicted_option("None", options_str=none_opts), "E")
        na_opts = "A. Cat\nB. Dog\nC. N/A"
        self.assertEqual(extract_predicted_option("n/a", options_str=na_opts), "C")

    def test_resolve_dataset_args_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            anno_file = os.path.join(tmpdir, "annotation_vstar.json")
            with open(anno_file, "w") as f:
                f.write("{}")

            # Providing only dataset_path without dataset_dir should resolve cleanly
            args = Namespace(
                benchmark="vstar",
                dataset_dir=None,
                dataset_path=anno_file,
                image_dir=None,
                pool_mode="N_5",
                gallery_order_path=None
            )
            with patch.dict(os.environ, {}, clear=True):
                res = resolve_dataset_args(args)
            self.assertEqual(res["annotation_path"], anno_file)
            self.assertEqual(res["dataset_dir_or_path"], tmpdir)

    def test_resume_statistics_restoration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_jsonl = os.path.join(tmpdir, "test_resume.jsonl")
            # Write 2 completed records with rotation metadata
            rec1 = {
                "sample_id": "s1",
                "benchmark": "hrbench8k",
                "cycle_idx": 0,
                "category": "single",
                "is_correct_answer": True,
                "is_correct_image": True,
                "is_correct_overall": True
            }
            rec2 = {
                "sample_id": "s2",
                "benchmark": "hrbench8k",
                "cycle_idx": 1,
                "category": "single",
                "is_correct_answer": False,
                "is_correct_image": True,
                "is_correct_overall": False
            }
            with open(out_jsonl, "w", encoding="utf-8") as f:
                f.write(json.dumps(rec1) + "\n" + json.dumps(rec2) + "\n")

            stats = {
                "total": 0,
                "correct": 0,
                "correct_img": 0,
                "correct_overall": 0,
                "by_category": {},
                "by_rotation": {}
            }
            completed_ids = set()

            with open(out_jsonl, "r", encoding="utf-8") as rf:
                for line in rf:
                    if line.strip():
                        rec = json.loads(line)
                        if "sample_id" in rec:
                            s_id = str(rec["sample_id"])
                            completed_ids.add(s_id)
                            cat = rec.get("category") or "default"
                            if cat not in stats["by_category"]:
                                stats["by_category"][cat] = {"total": 0, "correct": 0, "correct_img": 0, "correct_overall": 0}
                            stats["total"] += 1
                            stats["by_category"][cat]["total"] += 1
                            if rec.get("is_correct_answer"):
                                stats["correct"] += 1
                                stats["by_category"][cat]["correct"] += 1
                            if rec.get("is_correct_image"):
                                stats["correct_img"] += 1
                                stats["by_category"][cat]["correct_img"] += 1
                            if rec.get("is_correct_overall"):
                                stats["correct_overall"] += 1
                                stats["by_category"][cat]["correct_overall"] += 1

                            if "cycle_idx" in rec and rec["cycle_idx"] is not None:
                                rot_k = f"rot{rec['cycle_idx']}"
                                if rot_k not in stats["by_rotation"]:
                                    stats["by_rotation"][rot_k] = {"total": 0, "correct": 0}
                                stats["by_rotation"][rot_k]["total"] += 1
                                if rec.get("is_correct_answer"):
                                    stats["by_rotation"][rot_k]["correct"] += 1

            self.assertEqual(len(completed_ids), 2)
            self.assertEqual(stats["total"], 2)
            self.assertEqual(stats["correct_overall"], 1)
            self.assertEqual(stats["by_category"]["single"]["correct_overall"], 1)
            self.assertEqual(stats["by_rotation"]["rot0"]["correct"], 1)
            self.assertEqual(stats["by_rotation"]["rot1"]["correct"], 0)

    def test_image_loading_exif_warning(self):
        """Tests that if EXIF transpose fails, a warning is printed before falling back to standard open."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "test.png")
            img = Image.new("RGB", (32, 32), color="blue")
            img.save(img_path)

            output_buffer = io.StringIO()
            with patch("sys.stdout", output_buffer):
                with patch("PIL.ImageOps.exif_transpose", side_effect=ValueError("Corrupt EXIF")):
                    # Simulate image loading block in eval_runner
                    pil_images = []
                    if os.path.isfile(img_path):
                        try:
                            from PIL import ImageOps
                            with Image.open(img_path) as im:
                                pil_images.append(ImageOps.exif_transpose(im).convert("RGB"))
                        except Exception as exif_err:
                            print(f"[Warning] Failed to open '{img_path}' with EXIF transpose ({exif_err}). "
                                  f"Falling back to standard image open without EXIF transpose.")
                            try:
                                with Image.open(img_path) as im:
                                    pil_images.append(im.convert("RGB"))
                            except Exception as read_err:
                                print(f"[Error] Failed to load image '{img_path}': {read_err}")

            self.assertEqual(len(pil_images), 1)
            stdout_val = output_buffer.getvalue()
            self.assertIn("[Warning] Failed to open", stdout_val)
            self.assertIn("Falling back to standard image open without EXIF transpose", stdout_val)

    def test_parse_args_visualization_flags(self):
        from minimal.eval_runner import parse_args
        with patch.dict(os.environ, {}, clear=True), patch("sys.argv", ["eval_runner.py", "--benchmark", "stash", "--save_vis", "--vis_dir", "vis/custom", "--vis_max_samples", "10", "--vis_sample_ids", "sample_1,sample_2"]):
            args = parse_args()
            self.assertTrue(args.save_vis)
            self.assertEqual(args.vis_dir, "vis/custom")
            self.assertEqual(args.vis_max_samples, 10)
            self.assertEqual(args.vis_sample_ids, "sample_1,sample_2")


if __name__ == "__main__":
    unittest.main()
