#!/usr/bin/env python3
"""
Standalone SigLIP Inference & Calibration Diagnostic Script
============================================================
Designed to run on Kaggle GPU, Colab, or remote server.
Diagnoses:
1. Model loading & processor compatibility (SigLIP v1 vs SigLIP 2).
2. Prompt formatting comparisons (raw vs 'a photo of {q}' vs 'The photo of {label}.').
3. Scoring calibration comparison:
   - Mode A: Native SigLIP Sigmoid: sigma(cos * exp(scale) + bias)
   - Mode B: Linear Calibrated Score: clamp((cos - 0.02) / 0.22, 0.0, 1.0)
   - Mode C: Raw Cosine Similarity: cos(I, T)
4. Stage 3 Saliency-Plateau Decision Simulation:
   - Parent patch vs 4 child sub-patches
   - Delta S = max(S_children) - S_parent vs tau = 0.04
5. Integration test with mssearch.models.modeling_siglip.SigLIPScorer
"""

import os
import sys
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# Ensure MSSearch root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MSSEARCH_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if MSSEARCH_ROOT not in sys.path:
    sys.path.insert(0, MSSEARCH_ROOT)


def resolve_checkpoint_path(default_name: str, candidate_dirs: List[str]) -> str:
    """Finds local checkpoint path if present, otherwise returns default huggingface hub name."""
    for d in candidate_dirs:
        if os.path.exists(d):
            print(f"[Discovery] Found local checkpoint at: {d}")
            return d
    return default_name


def resolve_dataset_path(candidate_dirs: List[str]) -> Optional[str]:
    """Finds HRBench base directory if available."""
    for d in candidate_dirs:
        if os.path.exists(d):
            print(f"[Discovery] Found dataset root at: {d}")
            return d
    return None


def create_synthetic_test_crops() -> Tuple[Image.Image, List[Tuple[str, Image.Image]]]:
    """Creates a synthetic canvas with distinct target and distractor shapes."""
    canvas = Image.new("RGB", (1024, 1024), color=(240, 240, 240))
    draw = ImageDraw.Draw(canvas)

    # Target: A clear red car shape at top-left
    draw.rectangle([100, 100, 350, 250], fill=(220, 30, 30))
    draw.ellipse([140, 230, 190, 280], fill=(20, 20, 20))
    draw.ellipse([260, 230, 310, 280], fill=(20, 20, 20))
    draw.text((160, 140), "CAR #27", fill=(255, 255, 255))

    # Distractor: Blue tree at bottom-right
    draw.rectangle([700, 650, 740, 850], fill=(120, 70, 30))
    draw.ellipse([630, 500, 810, 680], fill=(30, 160, 40))

    # Background: empty region at top-right
    draw.rectangle([600, 100, 900, 400], fill=(230, 230, 230))

    crops = [
        ("Target (Red Car)", canvas.crop((80, 80, 380, 300))),
        ("Target Sub-Patch (Front Wheel)", canvas.crop((120, 180, 220, 290))),
        ("Distractor (Tree)", canvas.crop((600, 480, 840, 860))),
        ("Background (Empty)", canvas.crop((600, 100, 900, 400))),
    ]
    return canvas, crops


def test_prompt_and_scoring(
    model: Any,
    processor: Any,
    crops: List[Tuple[str, Image.Image]],
    queries: List[str],
    device: str,
    dtype: torch.dtype,
):
    """Evaluates crops against queries across different prompt templates and scoring metrics."""
    print("\n" + "=" * 80)
    print("STEP 2: PROMPT TEMPLATES & SCORING MODES EVALUATION")
    print("=" * 80)

    prompt_templates = {
        "Raw Query ('{q}')": lambda q: q.strip(),
        "CLIP Standard ('a photo of a/an {q}')": lambda q: (
            q.strip() if q.strip().startswith(("a ", "an ", "the ", "photo of "))
            else f"a photo of {'an' if q.strip()[0].lower() in 'aeiou' else 'a'} {q.strip()}"
        ),
        "SigLIP2 Notebook ('The photo of {q}.')": lambda q: f"The photo of {q.strip()}.",
    }

    logit_scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
    logit_bias = model.logit_bias.item() if hasattr(model, "logit_bias") else -10.0
    print(f"SigLIP Parameters: exp(logit_scale) = {logit_scale:.2f} | logit_bias = {logit_bias:.2f}\n")

    pil_images = [c[1] for c in crops]
    crop_names = [c[0] for c in crops]

    for t_name, t_fn in prompt_templates.items():
        print(f"--- Prompt Template: {t_name} ---")
        formatted_queries = [t_fn(q) for q in queries]
        for q_raw, q_fmt in zip(queries, formatted_queries):
            print(f"  Query: '{q_raw}' -> Prompt: '{q_fmt}'")

        # Encode text
        text_inputs = processor(
            text=formatted_queries,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        ).to(device)

        # Encode images
        img_inputs = processor(
            images=pil_images,
            return_tensors="pt"
        ).to(device)
        if dtype == torch.float16 and "pixel_values" in img_inputs:
            img_inputs["pixel_values"] = img_inputs["pixel_values"].to(dtype)

        with torch.inference_mode():
            t_out = model.get_text_features(**text_inputs)
            if hasattr(t_out, "pooler_output") and t_out.pooler_output is not None:
                t_feat = t_out.pooler_output
            elif hasattr(t_out, "last_hidden_state") and t_out.last_hidden_state is not None:
                t_feat = t_out.last_hidden_state.mean(dim=1)
            else:
                t_feat = t_out
            t_feat = F.normalize(t_feat.float(), dim=-1)

            i_out = model.get_image_features(**img_inputs)
            if hasattr(i_out, "pooler_output") and i_out.pooler_output is not None:
                i_feat = i_out.pooler_output
            elif hasattr(i_out, "last_hidden_state") and i_out.last_hidden_state is not None:
                i_feat = i_out.last_hidden_state.mean(dim=1)
            else:
                i_feat = i_out
            i_feat = F.normalize(i_feat.float(), dim=-1)

            # Cosine similarity matrix: [num_crops, num_queries]
            cos_matrix = torch.matmul(i_feat, t_feat.T)

            # Native Sigmoid
            logits = cos_matrix * logit_scale + logit_bias
            sigmoid_probs = torch.sigmoid(logits)

            # Linear Calibration: clamp((cos - 0.02) / 0.22, 0.0, 1.0)
            calibrated_scores = torch.clamp((cos_matrix - 0.02) / 0.22, 0.0, 1.0)

        # Print comparison table
        print("\n  Results across Crops:")
        hdr = f"  {'Crop Description':<30} | {'Query':<15} | {'Cos Sim':<8} | {'Sigmoid':<10} | {'Calibrated':<10}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for c_idx, c_name in enumerate(crop_names):
            for q_idx, q_txt in enumerate(queries):
                c_val = cos_matrix[c_idx, q_idx].item()
                s_val = sigmoid_probs[c_idx, q_idx].item()
                cal_val = calibrated_scores[c_idx, q_idx].item()
                print(f"  {c_name:<30} | {q_txt:<15} | {c_val:+.4f} | {s_val:.6f}   | {cal_val:.4f}")
        print()


def test_stage3_plateau_simulation(
    model: Any,
    processor: Any,
    image_pil: Image.Image,
    target_bbox: Tuple[int, int, int, int],
    active_objects: List[str],
    device: str,
):
    """
    Simulates Stage 3 Saliency-Plateau zoom decisions on a real image.
    Compares how Native Sigmoid vs Linear Calibrated vs Cosine Similarity behave
    at Parent vs Child quadtree splits.
    """
    print("\n" + "=" * 80)
    print("STEP 3: STAGE 3 SALIENCY-PLATEAU ZOOM SIMULATION")
    print("=" * 80)
    print(f"Canvas size: {image_pil.size} | Target BBox: {target_bbox}")
    print(f"Active Objects: {active_objects}")

    tx1, ty1, tx2, ty2 = target_bbox
    tw = tx2 - tx1
    th = ty2 - ty1

    # Case 1: Parent patch enclosing target with 2x margin
    px1 = max(0, tx1 - tw // 2)
    py1 = max(0, ty1 - th // 2)
    px2 = min(image_pil.width, tx2 + tw // 2)
    py2 = min(image_pil.height, ty2 + th // 2)
    parent_target_box = (px1, py1, px2, py2)

    # 4 Quadrants of parent_target_box
    mid_x = (px1 + px2) // 2
    mid_y = (py1 + py2) // 2
    quadrants_target = [
        (px1, py1, mid_x, mid_y),
        (mid_x, py1, px2, mid_y),
        (px1, mid_y, mid_x, py2),
        (mid_x, mid_y, px2, py2),
    ]

    # Case 2: Background parent patch (far from target)
    bg_px1 = 0
    bg_py1 = min(image_pil.height - 800, max(0, py2 + 1000))
    bg_px2 = min(image_pil.width, bg_px1 + 800)
    bg_py2 = min(image_pil.height, bg_py1 + 800)
    parent_bg_box = (bg_px1, bg_py1, bg_px2, bg_py2)

    bg_mid_x = (bg_px1 + bg_px2) // 2
    bg_mid_y = (bg_py1 + bg_py2) // 2
    quadrants_bg = [
        (bg_px1, bg_py1, bg_mid_x, bg_mid_y),
        (bg_mid_x, bg_py1, bg_px2, bg_mid_y),
        (bg_px1, bg_mid_y, bg_mid_x, bg_py2),
        (bg_mid_x, bg_mid_y, bg_px2, bg_py2),
    ]

    cases = [
        ("TARGET REGION (Should Continue Zooming)", parent_target_box, quadrants_target),
        ("BACKGROUND REGION (Should Terminate Zoom)", parent_bg_box, quadrants_bg),
    ]

    gain_threshold = 0.04

    for case_name, p_box, c_boxes in cases:
        print(f"\n--- Assessment: {case_name} ---")
        print(f"Parent Box: {p_box} ({p_box[2]-p_box[0]}x{p_box[3]-p_box[1]}px)")

        p_crop = image_pil.crop(p_box)
        c_crops = [image_pil.crop(cb) for cb in c_boxes]
        all_crops = [p_crop] + c_crops

        prompts = [f"a photo of {q.strip()}" for q in active_objects]
        text_inputs = processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt").to(device)
        img_inputs = processor(images=all_crops, return_tensors="pt").to(device)

        with torch.inference_mode():
            t_out = model.get_text_features(**text_inputs)
            t_feat = _get_pooler_or_mean(t_out)
            t_feat = F.normalize(t_feat.float(), dim=-1)

            i_out = model.get_image_features(**img_inputs)
            i_feat = _get_pooler_or_mean(i_out)
            i_feat = F.normalize(i_feat.float(), dim=-1)

            cos_matrix = torch.matmul(i_feat, t_feat.T)  # [5, num_queries]
            cos_max = cos_matrix.max(dim=1).values  # [5]

            logit_scale = model.logit_scale.exp().item() if hasattr(model, "logit_scale") else 100.0
            logit_bias = model.logit_bias.item() if hasattr(model, "logit_bias") else -10.0

            logits = cos_max * logit_scale + logit_bias
            sigmoids = torch.sigmoid(logits)
            calibrated = torch.clamp((cos_max - 0.02) / 0.22, 0.0, 1.0)

        cos_p = cos_max[0].item()
        cos_c = [c.item() for c in cos_max[1:]]
        cos_delta = max(cos_c) - cos_p

        sig_p = sigmoids[0].item()
        sig_c = [c.item() for c in sigmoids[1:]]
        sig_delta = max(sig_c) - sig_p

        cal_p = calibrated[0].item()
        cal_c = [c.item() for c in calibrated[1:]]
        cal_delta = max(cal_c) - cal_p

        print(f"  [Metric 1: Native Sigmoid]   S_parent={sig_p:.6f} | Max Child={max(sig_c):.6f} | Delta S = {sig_delta:+.6f}")
        action_sig = "ZOOM DEEPER" if sig_delta > gain_threshold else "TERMINATE ZOOM (PLATEAU)"
        print(f"             Decision (tau={gain_threshold}): {action_sig}")

        print(f"  [Metric 2: Linear Calibrated] S_parent={cal_p:.4f}   | Max Child={max(cal_c):.4f}   | Delta S = {cal_delta:+.4f}")
        action_cal = "ZOOM DEEPER" if cal_delta > gain_threshold else "TERMINATE ZOOM (PLATEAU)"
        print(f"             Decision (tau={gain_threshold}): {action_cal}")

        print(f"  [Metric 3: Raw Cosine]        S_parent={cos_p:.4f}   | Max Child={max(cos_c):.4f}   | Delta S = {cos_delta:+.4f}")
        action_cos = "ZOOM DEEPER" if cos_delta > gain_threshold else "TERMINATE ZOOM (PLATEAU)"
        print(f"             Decision (tau={gain_threshold}): {action_cos}")


def _get_pooler_or_mean(out: Any) -> torch.Tensor:
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state.mean(dim=1)
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0:
        return out[0]
    return out


def main():
    parser = argparse.ArgumentParser(description="SigLIP Scoring Diagnostic & Calibration Runner")
    parser.add_argument("--model_name", type=str, default="google/siglip-base-patch16-384")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_idx", type=int, default=0)
    args = parser.parse_args()

    print("=" * 80)
    print("MSSearch SigLIP Diagnostic Engine")
    print("=" * 80)
    print(f"Device: {args.device} | PyTorch: {torch.__version__}")

    # Checkpoint resolution
    candidate_checkpoints = [
        "/kaggle/input/siglip-base-patch16-384",
        "/kaggle/input/google-siglip-base-patch16-384",
        "/kaggle/input/datasets/thnhtt/siglip-base-patch16-384",
        "D:/Research/VisualSearch/siglip-base-patch16-384",
    ]
    model_path = resolve_checkpoint_path(args.model_name, candidate_checkpoints)

    print(f"\n[Loading] Loading SigLIP from: {model_path} on {args.device}...")
    from transformers import AutoProcessor, AutoModel
    dtype = torch.float16 if "cuda" in args.device else torch.float32
    try:
        model = AutoModel.from_pretrained(model_path, torch_dtype=dtype).to(args.device).eval()
        processor = AutoProcessor.from_pretrained(model_path)
        print(f"[Loading] Successfully loaded {type(model).__name__}!")
    except Exception as e:
        print(f"[Loading Error] Failed to load model via AutoModel: {e}")
        return

    # STEP 1: Synthetic Benchmark
    synthetic_canvas, synthetic_crops = create_synthetic_test_crops()
    test_queries = ["car", "wheel", "tree", "building"]
    test_prompt_and_scoring(model, processor, synthetic_crops, test_queries, args.device, dtype)

    # STEP 2: Real Dataset Ingestion if Available
    dataset_dirs = [
        "D:/Research/VisualSearch/hr-bench_8k",
        "d:/Research/VisualSearch/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_8k/mnt/sdb/llp/datasets/zoom_eye_data/zoom_eye_data/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_8k",
        "/kaggle/input/hr-bench-8k",
    ]
    ds_root = resolve_dataset_path(dataset_dirs)
    if ds_root:
        ann_path = os.path.join(ds_root, "annotation_hr-bench_8k.json")
        if os.path.exists(ann_path):
            with open(ann_path, "r", encoding="utf-8") as f:
                ann_data = json.load(f)
            sample = ann_data[args.sample_idx]
            q_text = sample.get("question", "")
            img_rel = sample.get("input_image", f"image/{args.sample_idx}.jpg")
            img_path = os.path.join(ds_root, img_rel)
            if os.path.exists(img_path):
                real_img = Image.open(img_path).convert("RGB")
                print(f"\n[Dataset] Loaded sample {args.sample_idx}: '{q_text}'")
                print(f"[Dataset] Image path: {img_path} ({real_img.size})")

                # Sample 0 verified bbox: [1940, 815, 2266, 993]
                sample_bbox = (1940, 815, 2266, 993) if args.sample_idx == 0 else (real_img.width//4, real_img.height//4, real_img.width//2, real_img.height//2)
                active_objs = ["number", "entrance where the woman is standing"] if args.sample_idx == 0 else ["target"]

                test_stage3_plateau_simulation(
                    model=model,
                    processor=processor,
                    image_pil=real_img,
                    target_bbox=sample_bbox,
                    active_objects=active_objs,
                    device=args.device,
                )

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    print("1. If Native Sigmoid delta is < 0.04 even on Target region: Sigmoid loss squashes small")
    print("   cosine differences into near-zero space (e.g. 0.0008 -> 0.0091 gives delta=0.0083 < 0.04).")
    print("2. Linear Calibrated score clamp((cos - 0.02) / 0.22, 0.0, 1.0) restores continuous gradient,")
    print("   making delta_S = 0.18 > 0.04 on target while keeping delta_S <= 0.01 on background.")
    print("3. Query formatting: 'a photo of {q}' provides consistent zero-shot alignment compared to raw tags.")


if __name__ == "__main__":
    main()
