import unittest
from PIL import Image
import numpy as np

from mssearch.config import MSSearchConfig
from mssearch.stages.stage3_partitioning import BoundedPatch
from mssearch.stages.stage5_screening import (
    solve_water_filling_allocation,
    extract_joint_interaction_crops,
    create_spatial_compressed_canvas,
    run_resolution_screening,
    CandidateCrop,
)


class TestStage5TriPillar(unittest.TestCase):
    def test_water_filling_target_locked_anchor_compressed(self):
        # Target = 200px (micro), Anchor = 1600px (macro)
        lengths = [200, 1600]
        roles = ["target", "anchor"]
        budget = 996  # 1024 - 28px gutter
        scales = solve_water_filling_allocation(lengths, roles, budget, target_max_dim=512)

        # Target must be strictly 1.0 (native optical resolution)
        self.assertAlmostEqual(scales[0], 1.0, places=4)
        # Anchor must absorb compression
        allocated_anchor = lengths[1] * scales[1]
        self.assertLess(scales[1], 1.0)
        self.assertLessEqual(lengths[0] * scales[0] + allocated_anchor, budget + 1e-4)

    def test_water_filling_target_overflow_option_3a(self):
        # 4 targets of 400px each = 1600px > 900px budget
        lengths = [400, 400, 400, 400]
        roles = ["target", "target", "target", "target"]
        budget = 900
        scales = solve_water_filling_allocation(lengths, roles, budget, target_max_dim=512)

        # Under Option 3A: all targets scaled equally to fit within budget
        total_len = sum(int(round(l * s)) for l, s in zip(lengths, scales))
        self.assertLessEqual(total_len, budget)
        for s in scales:
            self.assertAlmostEqual(s, scales[0], places=4)

    def test_dynamic_gutters_touching_vs_excised(self):
        # Synthetic 8K image
        img = Image.new("RGB", (4000, 2000), (255, 255, 255))
        # Touching tiles (gap = 0px) vs distant tile (gap = 1500px)
        b1 = (100, 100, 400, 400)
        b2 = (400, 100, 700, 400)  # touches b1 (gap = 0px)
        b3 = (2200, 100, 2500, 400)  # distant from b2 (gap = 1500px)

        canvas, remapped = create_spatial_compressed_canvas(
            img,
            boxes=[b1, b2, b3],
            gutter_px=28,
            max_canvas_dim=1024,
            enable_dynamic_gutters=True,
            continuity_thresh=2
        )

        # Canvas dimension must strictly not exceed 1024px
        self.assertLessEqual(canvas.width, 1024)
        self.assertLessEqual(canvas.height, 1024)

        # Touching tiles b1 and b2 must have 0px gap between them in remapped coords
        rem_b1 = remapped[0]
        rem_b2 = remapped[1]
        rem_b3 = remapped[2]
        self.assertEqual(rem_b1[2], rem_b2[0])  # Seamless stitch (0px gutter)

        # Distant tile b3 must have gutter_px gap (28px)
        self.assertEqual(rem_b3[0] - rem_b2[2], 28)

    def test_pillar1_joint_interaction_crop_merging(self):
        # Two patches within proximity threshold (150px apart, union = 550px <= 1024px)
        p1 = BoundedPatch(img_idx=0, box=(100, 100, 300, 300), feature_vector=np.zeros(16), entropy=1.0)
        p2 = BoundedPatch(img_idx=0, box=(450, 120, 650, 320), feature_vector=np.zeros(16), entropy=1.2)

        merged, is_all_one_jic = extract_joint_interaction_crops(
            [p1, p2],
            image_shape=(2000, 2000),
            prox_threshold=400.0,
            max_dim=1024,
            padding_factor=1.15
        )

        self.assertTrue(is_all_one_jic)
        self.assertEqual(len(merged), 1)
        # Unified box covers both patches
        ubox = merged[0].box
        self.assertLessEqual(ubox[0], 100)
        self.assertGreaterEqual(ubox[2], 650)
        self.assertLessEqual(ubox[2] - ubox[0], 1024)

    def test_single_object_query_bypasses_canvas(self):
        # Single-object query (is_relational=False)
        img = Image.new("RGB", (3000, 2000), (100, 100, 100))
        p1 = BoundedPatch(img_idx=0, box=(200, 200, 400, 400), feature_vector=np.zeros(16), entropy=1.0)
        p2 = BoundedPatch(img_idx=0, box=(1200, 200, 1400, 400), feature_vector=np.zeros(16), entropy=1.0)

        cfg = MSSearchConfig()
        cfg.enable_joint_interaction_crop = True
        cfg.spatial_canvas_relational_only = True

        pool = run_resolution_screening(
            remaining_patches=[p1, p2],
            candidate_images=[img],
            active_objects=["barge"],
            config=cfg,
            is_relational=False
        )

        # Mode must be crops_only, no compressed canvas
        self.assertEqual(pool.stage5_mode, "crops_only")
        self.assertFalse(pool.is_cross_object)
        for crop in pool:
            self.assertFalse(crop.is_compressed_canvas)
            self.assertEqual(crop.source_branch, "Direct")
            # Width with 1.25x padding = 200 * 1.25 = 250px
            self.assertEqual(crop.crop_image.width, 250)


if __name__ == "__main__":
    unittest.main()
