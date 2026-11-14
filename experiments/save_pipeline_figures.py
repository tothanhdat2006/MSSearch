import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CUR_DIR, ".."))
_OUR_DIR = os.path.join(_CUR_DIR, "our")
for _p in [_REPO_ROOT, _CUR_DIR, _OUR_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from eval_dataset import resolve_image, load_and_slice_dataset
    from eval_models import load_our_engines, run_our_pipeline
except ModuleNotFoundError:
    from experiments.our.eval_dataset import resolve_image, load_and_slice_dataset
    from experiments.our.eval_models import load_our_engines, run_our_pipeline


def draw_labeled_box(img: Image.Image, box: List[int], label: str, color: str = "#10B981", width: int = 3) -> Image.Image:
    """Draws a highlighted bounding box with text badge on an image crop."""
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    draw.rectangle(box, outline=color, width=width)
    if label:
        tx, ty = max(0, box[0]), max(0, box[1] - 16)
        draw.rectangle([tx, ty, tx + len(label) * 8 + 6, ty + 15], fill=color)
        draw.text((tx + 3, ty + 1), label, fill="#FFFFFF")
    return vis


def save_stage_assets(sample_dir: str, sample: Dict[str, Any], candidate_meta: List[Any],
                      images: List[Image.Image], resp: Dict[str, Any], gt_img_idx: int) -> None:
    """Saves gallery, quadtree, patch ranking, containment, and foveated crop assets."""
    telemetry = resp.get("telemetry", {})
    ev_idx = telemetry.get("primary_evidence_image", gt_img_idx if gt_img_idx is not None else 0)

    # 1. Gallery Full Images
    d_gal = os.path.join(sample_dir, "01_gallery_full")
    os.makedirs(d_gal, exist_ok=True)
    for i, img in enumerate(images):
        tag = "evidence_I_star" if i == gt_img_idx else f"distractor_{i:02d}"
        img.save(os.path.join(d_gal, f"img_{i:02d}_{tag}.png"))

    # 2. Stage 2: Quadtree Overlays
    d_tree = os.path.join(sample_dir, "02_stage2_quadtree")
    os.makedirs(d_tree, exist_ok=True)
    ev_img = images[ev_idx].copy()
    tree_vis = ev_img.copy()
    tdraw = ImageDraw.Draw(tree_vis)
    w, h = ev_img.size
    tdraw.line([(w // 2, 0), (w // 2, h)], fill="#0284C7", width=4)
    tdraw.line([(0, h // 2), (w, h // 2)], fill="#0284C7", width=4)
    tdraw.text((15, 15), "Q1", fill="#0284C7")
    tdraw.text((w // 2 + 15, 15), "Q2", fill="#0284C7")
    tdraw.text((15, h // 2 + 15), "Q3", fill="#0284C7")
    tdraw.text((w // 2 + 15, h // 2 + 15), "Q4", fill="#0284C7")

    # Draw sub-quadrant target box if available
    s3_boxes = telemetry.get("stage3_boxes", [])
    ev_boxes = [b for b in s3_boxes if b.get("img_idx") == ev_idx]
    if ev_boxes:
        best_b = max(ev_boxes, key=lambda b: float(b.get("score") or 0.0))
        best_box = best_b.get("box", [0, 0, w // 2, h // 2])
        tdraw.rectangle(best_box, outline="#10B981", width=5)
        ev_img.crop(best_box).save(os.path.join(d_tree, "target_subquadrant_crop.png"))
    tree_vis.save(os.path.join(d_tree, "I_star_quadtree_grid_overlay.png"))

    # 3. Stage 3: SigLIP-2 Ranked Patches
    d_topk = os.path.join(sample_dir, "03_stage3_siglip_topk")
    os.makedirs(d_topk, exist_ok=True)
    patches = telemetry.get("stage3_boxes", [])
    sorted_patches = sorted(patches, key=lambda p: float(p.get("score") or 0.0), reverse=True)
    k_retain = min(len(sorted_patches), 20)
    for rank, p in enumerate(sorted_patches[:k_retain + 3], 1):
        p_img_idx, p_box = p.get("img_idx", 0), p.get("box", [0, 0, 100, 100])
        score = float(p.get("score") or 0.0)
        status = "retained" if rank <= k_retain else "pruned"
        color = "#10B981" if rank <= k_retain else "#EF4444"
        if 0 <= p_img_idx < len(images):
            patch_crop = images[p_img_idx].crop(p_box)
            labeled = draw_labeled_box(patch_crop, [0, 0, patch_crop.width - 1, patch_crop.height - 1],
                                       f"#{rank} S={score:.2f} {status.upper()}", color=color)
            labeled.save(os.path.join(d_topk, f"rank_{rank:02d}_score_{score:.2f}_{status}.png"))

    # 4. Stage 4: Containment Pair
    d_scr = os.path.join(sample_dir, "04_stage4_screening")
    os.makedirs(d_scr, exist_ok=True)
    s5_boxes = telemetry.get("stage5_boxes", [])
    if s5_boxes and len(ev_boxes) > 1:
        parent_box = ev_boxes[0].get("box")
        child_box = s5_boxes[0].get("box")
        if parent_box and child_box:
            images[ev_idx].crop(parent_box).save(os.path.join(d_scr, "coarse_parent_A_pruned.png"))
            images[ev_idx].crop(child_box).save(os.path.join(d_scr, "fine_child_B_retained.png"))

    # 5. Stage 6: Clean Context Crops & Voting
    d_fov = os.path.join(sample_dir, "05_stage6_foveated_crops")
    os.makedirs(d_fov, exist_ok=True)
    for c_i, c in enumerate(telemetry.get("stage6_crops", [])):
        c_img_idx = c.get("img_idx", 0)
        c_box = c.get("expanded_box") or c.get("box")
        if c_box and 0 <= c_img_idx < len(images):
            crop_clean = images[c_img_idx].crop(c_box)
            crop_clean.save(os.path.join(d_fov, f"crop_{c_i:02d}_clean_1.25x.png"))
            # Bbox overlay
            orig_b = c.get("box")
            if orig_b:
                rel_box = [orig_b[0] - c_box[0], orig_b[1] - c_box[1],
                           orig_b[2] - c_box[0], orig_b[3] - c_box[1]]
                draw_labeled_box(crop_clean, rel_box, "Target BBox", color="#EF4444").save(
                    os.path.join(d_fov, f"crop_{c_i:02d}_bbox_overlay.png")
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Save pipeline figures and crops on 2-3 samples.")
    parser.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--siglip_name", type=str, default=os.environ.get("SIGLIP_NAME", "google/siglip2-so400m-patch16-512"))
    parser.add_argument("--nlp_model", type=str, default=os.environ.get("NLP_MODEL", "en_core_web_sm"))
    parser.add_argument("--dataset_path", type=str, default=os.environ.get("DATASET_PATH", "data/STASHBench/annotations_part1234_merged.json"))
    parser.add_argument("--image_dir", type=str, default=os.environ.get("IMAGE_DIR", "data/STASHBench/pool"))
    parser.add_argument("--output_dir", type=str, default="experiments/figures_data")
    parser.add_argument("--sample_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--pool_mode", type=str, choices=["N_1", "N_5", "N_10", "N_20", "crop"], default="N_5")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load dataset
    print(f"[1/4] Loading dataset from '{args.dataset_path}' for samples {args.sample_indices} ({args.pool_mode})...")
    if not os.path.exists(args.dataset_path) and args.dry_run:
        print("[Dry Run] Creating synthetic mock sample for offline dry-run test...")
        eval_tasks = [(s_idx, {"sample_id": f"mock_s{s_idx}", "question": "What brand text is on the jacket sleeve?"},
                       args.pool_mode, {"images": [{"image_id": f"img_{s_idx}"}], "gt_index": 1})
                      for s_idx in args.sample_indices]
    else:
        all_samples, all_tasks = load_and_slice_dataset(args.dataset_path, args.pool_mode)
        eval_tasks = [t for t in all_tasks if t[0] in args.sample_indices]
        if not eval_tasks and all_tasks:
            eval_tasks = all_tasks[:len(args.sample_indices)]

    print(f"[2/4] Selected {len(eval_tasks)} sample tasks for pipeline asset saving.")

    # 2. Load engines
    engines = None
    if not args.dry_run:
        print("[3/4] Initializing MSSearch V2 canonical engines...")
        engines = load_our_engines(
            model_path=args.model_path,
            siglip_name=args.siglip_name,
            nlp_model=args.nlp_model,
            config_path=args.config,
            device=args.device,
        )

    # 3. Execution loop
    print(f"[4/4] Executing and exporting assets to '{args.output_dir}'...")
    for idx, (s_idx, sample, pool_name, pool_info) in enumerate(eval_tasks):
        sample_id = sample.get("sample_id", f"sample_{s_idx}")
        cand_meta = pool_info.get("images", [])
        gt_idx = pool_info.get("gt_index")
        gt_img_idx = (gt_idx - 1) if gt_idx is not None else 0

        sample_dir = os.path.join(args.output_dir, f"sample_{s_idx}_{sample_id}_{pool_name}")
        os.makedirs(sample_dir, exist_ok=True)

        if args.dry_run:
            images = [Image.new("RGB", (1024, 768), color=(40 + i * 30, 80 + i * 20, 120 + i * 10)) for i in range(len(cand_meta) or 3)]
            resp = {
                "primary_answer": "epio MIX",
                "telemetry": {
                    "primary_evidence_image": gt_img_idx,
                    "target_object": "jacket sleeve logo",
                    "stage3_boxes": [{"img_idx": gt_img_idx, "box": [512, 384, 1024, 768], "score": 0.94},
                                     {"img_idx": gt_img_idx, "box": [0, 0, 512, 384], "score": 0.88},
                                     {"img_idx": 1, "box": [0, 0, 512, 384], "score": 0.28}],
                    "stage5_boxes": [{"img_idx": gt_img_idx, "box": [600, 450, 850, 650]}],
                    "stage6_crops": [{"img_idx": gt_img_idx, "box": [620, 480, 800, 600],
                                      "expanded_box": [580, 440, 840, 640], "final_answer": "epio MIX"}]
                }
            }
        else:
            images = [resolve_image(m, args.image_dir) for m in cand_meta]
            resp = run_our_pipeline(engines, images, sample, s_idx)

        # Save all intermediate crops and overlays
        save_stage_assets(sample_dir, sample, cand_meta, images, resp, gt_img_idx)

        # Save metadata summary
        summary = {
            "sample_index": s_idx, "sample_id": sample_id, "pool_name": pool_name,
            "question": sample.get("question"), "ground_truth": sample.get("answer"),
            "prediction": resp.get("primary_answer"),
            "primary_evidence_image": resp.get("telemetry", {}).get("primary_evidence_image"),
            "timings_sec": resp.get("telemetry", {}).get("timings_sec", {}),
            "stage6_extractions": resp.get("telemetry", {}).get("stage6_crops", [])
        }
        with open(os.path.join(sample_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"[{idx + 1}/{len(eval_tasks)}] Saved pipeline figure assets -> '{sample_dir}'")

    print("\n[Done] Pipeline figure assets successfully generated.")


if __name__ == "__main__":
    main()
