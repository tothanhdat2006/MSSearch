#!/usr/bin/env python3
"""
MSSearch Pilot Results Visualizer
=============================================================================
High-resolution, publication-grade visualization engine for MSSearch results.
- Reconstructs candidate image pools C = {I_1, ..., I_N} with deterministic random seed
- Overlays Stage 5 pre-aggregation bounding boxes (scores + entropy) & Stage 6 crops
- Synthesizes and visualizes the EXACT spatial compressed layout fed to Stage 6 (Qwen2.5-VL)
- Extracts and magnifies high-resolution zoomed crops for fine-grained evidence inspection
- Generates composite multi-panel PNGs for each sample
- Compiles an interactive, dark-mode offline HTML gallery dashboard in the 'vis' folder
"""

import os
import sys
import json
import re
import random
import argparse
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont


# ==============================================================================
# FONT HELPERS & STYLING PALETTE
# ==============================================================================

def get_font(size: int = 16, bold: bool = False):
    """Loads system TTF fonts with graceful fallback to PIL default font."""
    font_candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
    ]
    for candidate in font_candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


# Color Palette (Modern Dark Theme)
PALETTE = {
    "bg": (15, 20, 28),              # Deep Navy
    "card_bg": (26, 34, 48),         # Card surface
    "card_border": (45, 55, 75),     # Card outline
    "text_main": (248, 250, 252),    # White
    "text_muted": (148, 163, 184),   # Slate muted
    "accent_cyan": (56, 189, 248),   # Sky cyan
    "accent_gold": (251, 191, 36),   # Amber gold
    "correct_green": (74, 222, 128), # Emerald
    "correct_bg": (20, 83, 45),      # Dark green badge
    "incorrect_red": (248, 113, 113), # Crimson
    "incorrect_bg": (127, 29, 29),   # Dark red badge
    "target_border": (74, 222, 128), # Green border for target
    "distractor_border": (100, 116, 139), # Slate border for distractor
    "evidence_badge": (14, 116, 144),# Cyan badge
    "box_cyan": (0, 220, 255),       # Stage 5 candidate boxes
    "box_gold": (255, 190, 0),       # Top evidence / Stage 6 boxes
    "stage6_badge": (124, 58, 237),  # Purple badge for Stage 6 layout
}


def safe_save_image(image: Image.Image, file_path: str, max_retries: int = 5) -> None:
    """Saves an image safely with atomic swap and retry mechanism to handle Windows file locking."""
    import time
    tmp_path = file_path + f".tmp_{os.getpid()}_{random.randint(1000, 9999)}"
    try:
        image.save(tmp_path)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        os.replace(tmp_path, file_path)
        return
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    for attempt in range(max_retries):
        try:
            image.save(file_path)
            return
        except OSError:
            time.sleep(0.3)
    image.save(file_path)


# ==============================================================================
# SPATIAL COMPRESSED CANVAS SYNTHESIS (STAGE 5 RECONSTRUCTION)
# ==============================================================================

def merge_intervals(intervals: List[Tuple[int, int]], min_gap: int = 0) -> List[Tuple[int, int]]:
    """Merges overlapping or near-adjacent 1D intervals [v1, v2]."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_ivs[0])]
    for curr in sorted_ivs[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1] + min_gap:
            prev[1] = max(prev[1], curr[1])
        else:
            merged.append(list(curr))
    return [tuple(m) for m in merged]


def create_spatial_compressed_canvas(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    gutter_px: int = 28,
    bg_color: Tuple[int, int, int] = (128, 128, 128)
) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
    """
    Synthesizes the exact continuous-coordinate compressed canvas fed to Stage 6.
    Excises empty rows and columns outside surviving candidate boxes and inserts 28px neutral visual gutters.
    """
    if not boxes:
        return image, []
    if len(boxes) == 1:
        box = boxes[0]
        crop = image.crop(box)
        return crop, [(0, 0, crop.width, crop.height)]

    w, h = image.size
    x_ivs = [(max(0, b[0]), min(w, b[2])) for b in boxes if b[2] > b[0]]
    y_ivs = [(max(0, b[1]), min(h, b[3])) for b in boxes if b[3] > b[1]]

    merged_x = merge_intervals(x_ivs, min_gap=gutter_px)
    merged_y = merge_intervals(y_ivs, min_gap=gutter_px)

    x_map_start: Dict[Tuple[int, int], int] = {}
    curr_x = 0
    for idx, (x1, x2) in enumerate(merged_x):
        x_map_start[(x1, x2)] = curr_x
        curr_x += (x2 - x1)
        if idx < len(merged_x) - 1:
            curr_x += gutter_px
    canvas_w = max(1, curr_x)

    y_map_start: Dict[Tuple[int, int], int] = {}
    curr_y = 0
    for idx, (y1, y2) in enumerate(merged_y):
        y_map_start[(y1, y2)] = curr_y
        curr_y += (y2 - y1)
        if idx < len(merged_y) - 1:
            curr_y += gutter_px
    canvas_h = max(1, curr_y)

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    def remap_coord(val: int, intervals: List[Tuple[int, int]], map_start: Dict[Tuple[int, int], int]) -> int:
        for (v1, v2) in intervals:
            if v1 <= val <= v2:
                return map_start[(v1, v2)] + (val - v1)
            elif val < v1:
                return map_start[(v1, v2)]
        if intervals:
            last = intervals[-1]
            return map_start[last] + (last[1] - last[0])
        return 0

    # Paste patch crops onto canvas
    for box in boxes:
        crop = image.crop(box)
        nx1 = remap_coord(box[0], merged_x, x_map_start)
        ny1 = remap_coord(box[1], merged_y, y_map_start)
        canvas.paste(crop, (nx1, ny1))

    # Remap bounding boxes
    remapped_boxes: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        nx1 = remap_coord(box[0], merged_x, x_map_start)
        ny1 = remap_coord(box[1], merged_y, y_map_start)
        nx2 = remap_coord(box[2], merged_x, x_map_start)
        ny2 = remap_coord(box[3], merged_y, y_map_start)
        remapped_boxes.append((nx1, ny1, nx2, ny2))

    return canvas, remapped_boxes


# ==============================================================================
# HRBENCH AUTO-DISCOVERY & DATASET HELPERS
# ==============================================================================

def discover_hrbench_dir(
    preferred_path: Optional[str] = None,
    preferred_annotation: Optional[str] = None
) -> Tuple[str, str]:
    """Discovers HRBench directory and annotation file across Windows & Linux environments."""
    if preferred_annotation and os.path.exists(preferred_annotation):
        resolved_annot = os.path.abspath(preferred_annotation)
        resolved_base = preferred_path if (preferred_path and os.path.exists(preferred_path)) else os.path.dirname(resolved_annot)
        return resolved_base, resolved_annot

    search_dirs = []
    if preferred_path:
        search_dirs.append(preferred_path)
    env_dir = os.environ.get("HRBENCH_DIR")
    if env_dir:
        search_dirs.append(env_dir)

    search_dirs.extend([
        "D:/Research/VisualSearch/hr-bench_8k",
        "d:/Research/VisualSearch/hr-bench_8k",
        "/home/ubuntu/workspace/hr/data/hr-bench_8k",
        "/home/ubuntu/workspace/hr/data/hr-bench_4k",
        "docs/experiments/multisource/data/hr-bench_8k",
        "docs/experiments/freeform/ICML26-CVSearch/data/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_8k/mnt/sdb/llp/datasets/zoom_eye_data/zoom_eye_data/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_8k",
    ])

    resolved_base = None
    for d in search_dirs:
        if d and os.path.exists(d):
            resolved_base = d
            break

    if resolved_base is None:
        raise FileNotFoundError(
            f"[Dataset Error] Could not find HRBench directory in paths: {search_dirs}"
        )

    annotation_path = os.path.join(resolved_base, "annotation_hr-bench_8k.json")
    if not os.path.exists(annotation_path):
        annotation_path = os.path.join(resolved_base, "annotation_hr-bench_4k.json")

    if not os.path.exists(annotation_path):
        raise FileNotFoundError(
            f"[Dataset Error] Annotation file not found in HRBench dir: {resolved_base}"
        )

    return resolved_base, annotation_path


def parse_sample_options_map(options_raw: Any, cycle_idx: int = 0) -> Dict[str, str]:
    """Parses options block into a mapping { 'A': 'option text', 'B': 'option text', ... }."""
    opt_dict: Dict[str, str] = {}
    if not options_raw:
        return opt_dict

    if isinstance(options_raw, list):
        if not options_raw:
            return opt_dict
        if 0 <= cycle_idx < len(options_raw):
            chosen = options_raw[cycle_idx]
        else:
            chosen = options_raw[0]
    else:
        chosen = options_raw

    if isinstance(chosen, dict):
        for k, v in chosen.items():
            opt_dict[str(k).strip().upper()] = str(v).strip()
    elif isinstance(chosen, str):
        for line in chosen.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            m = re.match(r"^[\(\[\"']*([A-Da-d])[\)\]\.\:\s]+(.*)$", line_str)
            if m:
                opt_dict[m.group(1).upper()] = m.group(2).strip()

    return opt_dict


def get_sample_gt_details(
    sample_index: int,
    cycle_idx: int,
    hrbench_data: List[Dict[str, Any]]
) -> Tuple[str, str, Dict[str, str]]:
    """
    Extracts authoritative ground truth letter, full ground truth text, and options map
    directly from HRBench annotation JSON.
    Returns: (gt_letter, gt_text, options_map)
    """
    if sample_index < 0 or sample_index >= len(hrbench_data):
        return ("N/A", "N/A", {})

    sample = hrbench_data[sample_index]
    ans_obj = sample.get("answer", "")
    gt_letter = "N/A"
    if isinstance(ans_obj, list):
        if 0 <= cycle_idx < len(ans_obj):
            gt_letter = str(ans_obj[cycle_idx]).strip().upper()
        elif len(ans_obj) > 0:
            gt_letter = str(ans_obj[0]).strip().upper()
    elif ans_obj:
        gt_letter = str(ans_obj).strip().upper()

    options_map = parse_sample_options_map(sample.get("options", {}), cycle_idx=cycle_idx)
    gt_text = options_map.get(gt_letter, "")
    return (gt_letter, gt_text, options_map)


def is_sample_cross_object(rec: Dict[str, Any]) -> bool:
    """
    Determines whether a sample used cross-object relational layout mode or single-object crops-only mode.
    Prioritizes authoritative flags saved in the record, with fallback heuristics for legacy logs.
    """
    # 1. Authoritative flags saved in results JSONL
    if "is_cross_object" in rec:
        return bool(rec["is_cross_object"])
    if "is_relational" in rec:
        return bool(rec["is_relational"])
    s5_mode = rec.get("stage5_mode")
    if s5_mode in ["layout", "continuous_enclosure"]:
        return True
    elif s5_mode == "crops_only":
        return False

    # 2. Check telemetry dict if nested
    telemetry = rec.get("telemetry", {})
    if isinstance(telemetry, dict):
        if "is_cross_object" in telemetry:
            return bool(telemetry["is_cross_object"])
        if "is_relational" in telemetry:
            return bool(telemetry["is_relational"])
        if telemetry.get("stage5_mode") in ["layout", "continuous_enclosure"]:
            return True
        elif telemetry.get("stage5_mode") == "crops_only":
            return False

    # 3. Fallback heuristic strictly for legacy results generated prior to flag persistence
    q = rec.get("question", "").lower()
    single_obj_kws = [
        "phone number", "registration number", "license plate",
        "color of the", "color are the", "what color is", "title of the", "name of the",
        "brand of the", "what is written on", "text on the", "text written on",
        "number written on", "number displayed on", "what number is", "what letter is",
        "how many", "what type of", "what kind of", "what model is"
    ]
    if any(kw in q for kw in single_obj_kws):
        return False

    rel_kws = [
        "left of", "right of", "above the", "below the", "under the",
        "next to", "in front of", "behind the", "between", "which side of",
        "closer to", "farther from", "higher than", "lower than"
    ]
    if any(kw in q for kw in rel_kws):
        return True

    return False


def reconstruct_candidate_pool(
    sample_index: int,
    hrbench_data: List[Dict[str, Any]],
    hrbench_base_dir: str,
    num_candidate_images: int = 3,
    window_size: int = 5,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Reconstructs the exact candidate image pool C = {I_1, ..., I_N} using identical RNG logic to debug_stages.py.
    """
    target_sample = hrbench_data[sample_index]
    target_img_rel = target_sample.get("input_image", "")
    target_img_path = os.path.join(hrbench_base_dir, target_img_rel)
    if not os.path.exists(target_img_path):
        target_img_path = target_img_rel
    target_canvas = Image.open(target_img_path).convert("RGB")

    total_samples = len(hrbench_data)
    min_win = max(0, sample_index - window_size)
    max_win = min(total_samples - 1, sample_index + window_size)

    candidate_distractor_indices = [idx for idx in range(min_win, max_win + 1) if idx != sample_index]
    if len(candidate_distractor_indices) < (num_candidate_images - 1):
        candidate_distractor_indices = [idx for idx in range(total_samples) if idx != sample_index]

    rng = random.Random(random_seed + sample_index)
    sampled_distractor_indices = rng.sample(candidate_distractor_indices, max(0, num_candidate_images - 1))

    pool_items = [
        {"is_target": True, "sample_idx": sample_index, "image": target_canvas, "path": target_img_rel}
    ]

    for d_idx in sampled_distractor_indices:
        d_sample = hrbench_data[d_idx]
        d_rel = d_sample.get("input_image", "")
        d_path = os.path.join(hrbench_base_dir, d_rel)
        if not os.path.exists(d_path):
            d_path = d_rel
        d_img = Image.open(d_path).convert("RGB")
        pool_items.append({"is_target": False, "sample_idx": d_idx, "image": d_img, "path": d_rel})

    # Shuffle candidate images deterministically
    rng.shuffle(pool_items)
    return pool_items


# ==============================================================================
# DRAWING HELPERS
# ==============================================================================

def draw_header_card(
    draw: ImageDraw.ImageDraw,
    rec: Dict[str, Any],
    width: int,
    start_y: int,
    hrbench_data: Optional[List[Dict[str, Any]]] = None
) -> int:
    """Draws structured header banner with status badges, question, answers, and latency chips."""
    font_title = get_font(22, bold=True)
    font_bold = get_font(16, bold=True)
    font_main = get_font(15, bold=False)
    font_mono = get_font(13, bold=False)

    pad_x = 24
    pad_y = 16
    curr_y = start_y + pad_y

    # Status Badge
    is_correct = rec.get("is_correct", False)
    status_text = "CORRECT" if is_correct else "INCORRECT"
    status_bg = PALETTE["correct_bg"] if is_correct else PALETTE["incorrect_bg"]
    status_fg = PALETTE["correct_green"] if is_correct else PALETTE["incorrect_red"]

    draw.rounded_rectangle(
        [pad_x, curr_y, pad_x + 110, curr_y + 32],
        radius=6,
        fill=status_bg,
        outline=status_fg,
        width=1
    )
    draw.text((pad_x + 16, curr_y + 7), status_text, fill=status_fg, font=font_bold)

    # Title & Route
    title_str = f"Sample #{rec.get('sample_index', 0)}  |  Route: {rec.get('route_selected', 'N/A')}  |  Total Latency: {rec.get('latency_sec', 0.0):.2f}s"
    draw.text((pad_x + 130, curr_y + 6), title_str, fill=PALETTE["text_main"], font=font_title)
    curr_y += 44

    # Question
    q_str = f"Q: {rec.get('question', '')}"
    draw.text((pad_x, curr_y), q_str, fill=PALETTE["accent_cyan"], font=font_bold)
    curr_y += 28

    # Ground Truth vs Prediction
    sample_index = rec.get("sample_index", 0)
    cycle_idx = rec.get("cycle_idx", 0)
    gt_letter = rec.get("gt_letter", "N/A")
    gt_text = ""
    options_map: Dict[str, str] = {}
    if hrbench_data and 0 <= sample_index < len(hrbench_data):
        annot_gt_letter, annot_gt_text, options_map = get_sample_gt_details(sample_index, cycle_idx, hrbench_data)
        if annot_gt_letter != "N/A":
            gt_letter = annot_gt_letter
        gt_text = annot_gt_text

    gt_display = f"[{gt_letter}] {gt_text}".strip() if gt_text else f"[{gt_letter}]"

    pred = rec.get("predicted_letter", "None") or "None"
    primary_ans = rec.get("primary_answer", "N/A")
    pred_display = f"[{pred}]"
    if pred != "None" and pred in options_map:
        pred_display += f" {options_map[pred]}"

    ev_imgs = rec.get("evidence_images", [])

    is_cross = is_sample_cross_object(rec)
    mode_text = "Mode: Layout (Cross-Object)" if is_cross else "Mode: Crops Only (Single-Object)"
    mode_col = PALETTE["accent_gold"] if is_cross else PALETTE["correct_green"]

    pred_color = PALETTE["correct_green"] if is_correct else PALETTE["incorrect_red"]

    # Render Ground Truth with Emerald Highlight
    draw.text((pad_x, curr_y), "Ground Truth: ", fill=PALETTE["text_muted"], font=font_main)
    try:
        gt_label_w = font_main.getbbox("Ground Truth: ")[2]
    except Exception:
        gt_label_w = 110
    draw.text((pad_x + gt_label_w, curr_y), gt_display, fill=PALETTE["correct_green"], font=font_bold)

    # Offset for Predicted & Model Answer
    pred_x = max(pad_x + 400, pad_x + gt_label_w + len(gt_display) * 9 + 28)
    draw.text((pred_x, curr_y), "Predicted: ", fill=PALETTE["text_muted"], font=font_main)
    try:
        pred_label_w = font_main.getbbox("Predicted: ")[2]
    except Exception:
        pred_label_w = 85
    draw.text((pred_x + pred_label_w, curr_y), pred_display, fill=pred_color, font=font_bold)

    raw_ans_x = max(pred_x + 220, pred_x + pred_label_w + len(pred_display) * 9 + 28)
    draw.text((raw_ans_x, curr_y), f"Model Output: '{primary_ans}'", fill=PALETTE["text_main"], font=font_main)

    draw.text((width - 520, curr_y), f"Evidence Images: {ev_imgs}", fill=PALETTE["accent_cyan"], font=font_main)
    draw.text((width - 260, curr_y), f"[{mode_text}]", fill=mode_col, font=font_bold)
    curr_y += 28

    # Stage Timings Chips
    timings = rec.get("timings", {})
    timing_parts = [
        f"P1: {timings.get('phase1_direct_answer', 0.0):.2f}s",
        f"S1: {timings.get('stage1_parser', 0.0):.2f}s",
        f"S2: {timings.get('stage2_fast_path', 0.0):.2f}s",
        f"S3: {timings.get('stage3_partitioning', 0.0):.2f}s",
        f"S4: {timings.get('stage4_grouping_gating', 0.0):.2f}s",
        f"S5: {timings.get('stage5_screening', 0.0):.2f}s",
        f"S6: {timings.get('stage6_reading', 0.0):.2f}s",
    ]
    draw.text((pad_x, curr_y), "Stages: " + "  |  ".join(timing_parts), fill=PALETTE["text_muted"], font=font_mono)
    curr_y += 24

    return curr_y + pad_y


def draw_candidate_image_panel(
    img_item: Dict[str, Any],
    img_idx: int,
    boxes: List[Dict[str, Any]],
    target_idx: int,
    is_evidence: bool,
    max_panel_w: int,
    max_panel_h: int
) -> Tuple[Image.Image, float, float]:
    """
    Renders a single candidate image canvas with overlaid bounding boxes and annotations.
    """
    orig_img = img_item["image"]
    orig_w, orig_h = orig_img.size

    scale = min(max_panel_w / orig_w, max_panel_h / orig_h)
    new_w = max(10, int(round(orig_w * scale)))
    new_h = max(10, int(round(orig_h * scale)))

    downscaled = orig_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    annotated = downscaled.copy()
    draw = ImageDraw.Draw(annotated)

    # Draw Stage 5 pre-aggregation bounding boxes
    for b in boxes:
        raw_box = b.get("box", [0, 0, orig_w, orig_h])
        x1 = int(round(raw_box[0] * scale))
        y1 = int(round(raw_box[1] * scale))
        x2 = int(round(raw_box[2] * scale))
        y2 = int(round(raw_box[3] * scale))

        score = b.get("score", 0.0)
        box_color = PALETTE["box_gold"] if score >= 2.0 else PALETTE["box_cyan"]
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)

    return annotated, scale, scale


# ==============================================================================
# SAMPLE VISUALIZATION ENGINE
# ==============================================================================

def visualize_sample(
    rec: Dict[str, Any],
    hrbench_data: List[Dict[str, Any]],
    hrbench_base_dir: str,
    output_dir: str,
    max_panel_w: int = 760,
    max_panel_h: int = 650,
    save_crops: bool = True
) -> Tuple[str, List[str]]:
    """
    Generates and saves a composite multi-panel visualization image for a sample,
    incorporating the exact Stage 6 spatial compressed layout.
    """
    sample_index = rec["sample_index"]
    target_pool_idx = rec.get("target_pool_idx", 0)
    evidence_images = set(rec.get("evidence_images", []))

    # 1. Reconstruct candidate pool
    pool_items = reconstruct_candidate_pool(
        sample_index=sample_index,
        hrbench_data=hrbench_data,
        hrbench_base_dir=hrbench_base_dir,
        num_candidate_images=3,
        window_size=5,
        random_seed=42
    )

    # Group boxes by image index
    boxes_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for b in rec.get("final_crop_boxes_before_aggregate", []):
        boxes_by_img.setdefault(b["img_idx"], []).append(b)

    # 2. Render Row 1: Candidate Image Panels
    rendered_panels = []
    for idx, item in enumerate(pool_items):
        is_ev = idx in evidence_images
        img_boxes = boxes_by_img.get(idx, [])
        panel_img, sx, sy = draw_candidate_image_panel(
            img_item=item,
            img_idx=idx,
            boxes=img_boxes,
            target_idx=target_pool_idx,
            is_evidence=is_ev,
            max_panel_w=max_panel_w,
            max_panel_h=max_panel_h
        )
        rendered_panels.append({
            "image": panel_img,
            "orig_size": item["image"].size,
            "is_target": item["is_target"],
            "sample_idx": item["sample_idx"],
            "is_evidence": is_ev,
            "num_boxes": len(img_boxes)
        })

    # 3. Render Row 2: Exact Stage 6 Inputs (Spatial Compressed Canvas vs Native Crops Only)
    is_cross_object = is_sample_cross_object(rec)
    stage6_layouts = []
    layout_dir = os.path.join(output_dir, "stage6_layouts")
    if not os.path.exists(layout_dir):
        os.makedirs(layout_dir, exist_ok=True)

    layout_filenames = []
    if is_cross_object:
        # Cross-Object Mode: Synthesize exact continuous-coordinate compressed canvas per image
        for idx, item in enumerate(pool_items):
            img_boxes = boxes_by_img.get(idx, [])
            if not img_boxes:
                continue

            raw_boxes = [tuple(b["box"]) for b in img_boxes]
            canvas_img, remapped_boxes = create_spatial_compressed_canvas(
                image=item["image"],
                boxes=raw_boxes,
                gutter_px=28
            )

            annotated_layout = canvas_img.copy()
            draw_layout = ImageDraw.Draw(annotated_layout)
            for r_box in remapped_boxes:
                draw_layout.rectangle(r_box, outline=PALETTE["box_gold"], width=2)

            layout_fname = f"layout_s{sample_index:03d}_img{idx}.png"
            safe_save_image(canvas_img, os.path.join(layout_dir, layout_fname))
            layout_filenames.append(layout_fname)

            s_scale = min(680 / annotated_layout.width, 480 / annotated_layout.height)
            lw = max(10, int(round(annotated_layout.width * s_scale)))
            lh = max(10, int(round(annotated_layout.height * s_scale)))
            downscaled_layout = annotated_layout.resize((lw, lh), Image.Resampling.BILINEAR)

            stage6_layouts.append({
                "type": "canvas",
                "img_idx": idx,
                "image": downscaled_layout,
                "raw_size": canvas_img.size,
                "num_patches": len(raw_boxes),
                "is_evidence": idx in evidence_images,
                "is_target": item["is_target"]
            })
    else:
        # Single-Object Mode: Native-resolution candidate crops ONLY (no grey voids)
        all_candidate_boxes = list(rec.get("final_crop_boxes_before_aggregate", []))
        all_candidate_boxes.sort(
            key=lambda b: (b.get("score", 0.0), b.get("entropy", 0.0)),
            reverse=True
        )

        for c_idx, b_info in enumerate(all_candidate_boxes[:6]):
            src_idx = b_info["img_idx"]
            orig_img = pool_items[src_idx]["image"]
            b_coords = tuple(b_info["box"])
            x1, y1, x2, y2 = b_coords
            cw = max(1, x2 - x1)
            ch = max(1, y2 - y1)
            if cw >= 8 and ch >= 8:
                crop_img = orig_img.crop((x1, y1, x2, y2))
                crop_fname = f"crop_s{sample_index:03d}_c{c_idx}_img{src_idx}.png"
                safe_save_image(crop_img, os.path.join(layout_dir, crop_fname))
                layout_filenames.append(crop_fname)

                c_scale = min(420 / crop_img.width, 360 / crop_img.height, 1.0)
                cw_scaled = max(10, int(round(crop_img.width * c_scale)))
                ch_scaled = max(10, int(round(crop_img.height * c_scale)))
                downscaled_crop = crop_img.resize((cw_scaled, ch_scaled), Image.Resampling.LANCZOS)

                stage6_layouts.append({
                    "type": "crop",
                    "img_idx": src_idx,
                    "image": downscaled_crop,
                    "raw_size": crop_img.size,
                    "num_patches": 1,
                    "is_evidence": src_idx in evidence_images,
                    "is_target": pool_items[src_idx]["is_target"],
                    "box": b_coords,
                    "score": b_info.get("score", 0.0),
                    "crop_rank": c_idx + 1
                })

    # 4. Prepare Row 3: Detail Zoom Insets from evidence regions
    zoom_crops = []
    crop_dir = os.path.join(output_dir, "crops")
    if save_crops and not os.path.exists(crop_dir):
        os.makedirs(crop_dir, exist_ok=True)

    candidate_crops_to_extract = []
    for b in rec.get("final_crop_boxes_before_aggregate", []):
        if b.get("img_idx") in evidence_images or b.get("img_idx") == target_pool_idx:
            candidate_crops_to_extract.append(b)

    candidate_crops_to_extract.sort(
        key=lambda x: (x.get("score", 0.0), x.get("entropy", 0.0)),
        reverse=True
    )

    for c_idx, c_info in enumerate(candidate_crops_to_extract[:4]):
        src_img_idx = c_info["img_idx"]
        orig_img = pool_items[src_img_idx]["image"]
        b_coords = c_info["box"]
        x1, y1, x2, y2 = b_coords
        cw, ch = x2 - x1, y2 - y1
        if cw >= 8 and ch >= 8:
            crop_canvas = orig_img.crop((x1, y1, x2, y2))
            crop_filename = f"crop_s{sample_index:03d}_img{src_img_idx}_{c_idx}.png"
            if save_crops:
                safe_save_image(crop_canvas, os.path.join(crop_dir, crop_filename))
            zoom_crops.append({
                "image": crop_canvas,
                "img_idx": src_img_idx,
                "box": b_coords,
                "score": c_info.get("score", 0.0),
                "entropy": c_info.get("entropy", 0.0),
                "filename": crop_filename
            })

    # 5. Calculate Total Canvas Layout Dimensions
    margin = 24
    header_h = 160
    panels_total_w = sum(p["image"].width + margin for p in rendered_panels) + margin
    s6_total_w = (sum(l["image"].width + margin for l in stage6_layouts) + margin) if stage6_layouts else 0
    zoom_total_w = (sum(min(340, z["image"].width) + 24 for z in zoom_crops) + margin) if zoom_crops else 0

    total_w = max(1920, panels_total_w, s6_total_w, zoom_total_w)
    panels_h = max((p["image"].height for p in rendered_panels), default=400) + 48

    stage6_row_h = (max((l["image"].height for l in stage6_layouts), default=0) + 72) if stage6_layouts else 0
    zoom_row_h = 240 if zoom_crops else 0

    total_h = header_h + panels_h + stage6_row_h + zoom_row_h + margin * 4 + 80

    canvas = Image.new("RGB", (total_w, total_h), color=PALETTE["bg"])
    draw = ImageDraw.Draw(canvas)

    # 1. Header Banner
    draw.rounded_rectangle(
        [margin, margin, total_w - margin, header_h],
        radius=10,
        fill=PALETTE["card_bg"],
        outline=PALETTE["card_border"],
        width=1
    )
    draw_header_card(draw, rec, total_w, margin, hrbench_data=hrbench_data)

    # Section 1 Title: Candidate Pool Overview
    font_sec = get_font(18, bold=True)
    font_panel_title = get_font(15, bold=True)
    font_panel_sub = get_font(13, bold=False)

    sec1_y = header_h + margin + 8
    draw.text((margin, sec1_y), "Row 1: Candidate Image Pool & Stage 5 Patch Proposals", fill=PALETTE["accent_cyan"], font=font_sec)

    # 2. Candidate Images Row
    curr_y = sec1_y + 32
    curr_x = margin

    for idx, panel in enumerate(rendered_panels):
        p_img = panel["image"]
        pw, ph = p_img.size

        card_rect = [curr_x, curr_y, curr_x + pw + 16, curr_y + ph + 48]
        draw.rounded_rectangle(card_rect, radius=8, fill=PALETTE["card_bg"], outline=PALETTE["card_border"], width=1)

        is_target = panel["is_target"]
        title_tag = "TARGET" if is_target else "DISTRACTOR"
        title_color = PALETTE["correct_green"] if is_target else PALETTE["text_muted"]

        draw.text((curr_x + 12, curr_y + 10), f"Image {idx}: {title_tag}", fill=title_color, font=font_panel_title)

        if panel["is_evidence"]:
            draw.rounded_rectangle(
                [curr_x + pw - 90, curr_y + 8, curr_x + pw + 6, curr_y + 30],
                radius=4,
                fill=PALETTE["evidence_badge"]
            )
            draw.text((curr_x + pw - 82, curr_y + 11), "[EVIDENCE]", fill=PALETTE["text_main"], font=get_font(12, bold=True))

        info_sub = f"Sample #{panel['sample_idx']} | {panel['orig_size'][0]}x{panel['orig_size'][1]} | {panel['num_boxes']} boxes"
        draw.text((curr_x + 140, curr_y + 12), info_sub, fill=PALETTE["text_muted"], font=font_panel_sub)

        canvas.paste(p_img, (curr_x + 8, curr_y + 40))
        border_col = PALETTE["target_border"] if is_target else PALETTE["distractor_border"]
        draw.rectangle([curr_x + 7, curr_y + 39, curr_x + 8 + pw, curr_y + 40 + ph], outline=border_col, width=2)

        curr_x += pw + margin

    # 3. Section 2: Stage 6 Input Row (Exact Canvas Fed to Reader VLM)
    next_y = curr_y + panels_h + 16
    if stage6_layouts:
        row2_title = (
            "Row 2: Exact Layout Fed to Stage 6 Reader (Spatial Compressed Canvas — Cross-Object Mode)"
            if is_cross_object else
            "Row 2: Exact Crops Fed to Stage 6 Reader (Native Resolution Crops Only — Single-Object Mode)"
        )
        row2_col = PALETTE["accent_gold"] if is_cross_object else PALETTE["correct_green"]
        draw.text(
            (margin, next_y),
            row2_title,
            fill=row2_col,
            font=font_sec
        )
        s6_y = next_y + 32
        lx = margin

        for s6 in stage6_layouts:
            l_img = s6["image"]
            lw, lh = l_img.size

            s6_card = [lx, s6_y, lx + lw + 16, s6_y + lh + 48]
            draw.rounded_rectangle(s6_card, radius=8, fill=PALETTE["card_bg"], outline=PALETTE["card_border"], width=1)

            # Badge & Header
            is_canvas = (s6.get("type", "canvas") == "canvas")
            badge_text = "STAGE 6 VLM FEED: CANVAS" if is_canvas else f"STAGE 6 CROP #{s6.get('crop_rank', 1)}"
            badge_w = 150 if is_canvas else 125
            draw.rounded_rectangle([lx + 10, s6_y + 8, lx + badge_w, s6_y + 30], radius=4, fill=PALETTE["stage6_badge"])
            draw.text((lx + 14, s6_y + 11), badge_text, fill=PALETTE["text_main"], font=get_font(11, bold=True))

            role_str = "TARGET" if s6["is_target"] else "DISTRACTOR"
            role_col = PALETTE["correct_green"] if s6["is_target"] else PALETTE["text_muted"]
            title_text = f"Image {s6['img_idx']} ({role_str})"
            title_x = lx + badge_w + 10
            draw.text((title_x, s6_y + 10), title_text, fill=role_col, font=font_panel_title)

            # Avoid text overlap: calculate title width dynamically
            try:
                t_bbox = font_panel_title.getbbox(title_text)
                title_w = t_bbox[2] - t_bbox[0]
            except Exception:
                title_w = len(title_text) * 9
            meta_x = title_x + title_w + 12

            if is_canvas:
                meta_str = f"Canvas: {s6['raw_size'][0]}x{s6['raw_size'][1]} | {s6['num_patches']} patches compressed (28px gutters)"
            else:
                meta_str = f"Native Crop: {s6['raw_size'][0]}x{s6['raw_size'][1]} | s={s6.get('score', 1.0):.1f}"
            draw.text((meta_x, s6_y + 12), meta_str, fill=PALETTE["text_muted"], font=font_panel_sub)

            canvas.paste(l_img, (lx + 8, s6_y + 40))
            border_col = PALETTE["box_gold"] if is_canvas else PALETTE["accent_cyan"]
            draw.rectangle([lx + 7, s6_y + 39, lx + 8 + lw, s6_y + 40 + lh], outline=border_col, width=2)

            lx += lw + margin

        next_y = s6_y + stage6_row_h

    # 4. Section 3: High-Resolution Evidence Insets
    if zoom_crops:
        draw.text(
            (margin, next_y),
            "Row 3: High-Resolution Native Evidence Details (Uncompressed 100% Crop Insets)",
            fill=PALETTE["correct_green"],
            font=font_sec
        )
        zy = next_y + 28
        zx = margin
        for z in zoom_crops:
            z_img = z["image"]
            scale_z = min(340 / z_img.width, 180 / z_img.height, 1.0)
            zw, zh = max(10, int(z_img.width * scale_z)), max(10, int(z_img.height * scale_z))
            z_down = z_img.resize((zw, zh), Image.Resampling.LANCZOS)

            z_card = [zx, zy, zx + zw + 16, zy + zh + 36]
            draw.rounded_rectangle(z_card, radius=6, fill=PALETTE["card_bg"], outline=PALETTE["card_border"])

            label = f"Img {z['img_idx']} | Box: {z['box']} | s={z['score']:.1f}"
            draw.text((zx + 8, zy + 6), label, fill=PALETTE["accent_cyan"], font=get_font(11, bold=True))
            canvas.paste(z_down, (zx + 8, zy + 26))
            draw.rectangle([zx + 7, zy + 25, zx + 8 + zw, zy + 26 + zh], outline=PALETTE["box_gold"], width=1)

            zx += zw + 24

    # Save Final Composite Visualization PNG
    status_suffix = "correct" if rec.get("is_correct", False) else "incorrect"
    out_filename = f"sample_{sample_index:03d}_{status_suffix}.png"
    out_path = os.path.join(output_dir, out_filename)
    safe_save_image(canvas, out_path)
    return out_path, layout_filenames


# ==============================================================================
# HTML GALLERY DASHBOARD GENERATOR
# ==============================================================================

def generate_html_gallery(
    records: List[Dict[str, Any]],
    output_dir: str,
    vis_filenames: Dict[int, str],
    layout_map: Dict[int, List[str]],
    hrbench_data: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generates a standalone, dark-themed offline HTML dashboard (vis/index.html)
    with direct modal access to both the full composite and Stage 6 layout feeds.
    """
    total = len(records)
    correct_count = sum(1 for r in records if r.get("is_correct", False))
    acc_pct = (correct_count / total * 100) if total > 0 else 0.0
    avg_lat = sum(r.get("latency_sec", 0.0) for r in records) / total if total > 0 else 0.0

    routes: Dict[str, int] = {}
    for r in records:
        routes[r.get("route_selected", "other")] = routes.get(r.get("route_selected", "other"), 0) + 1

    html_records = []
    for r in records:
        idx = r.get("sample_index", 0)
        img_file = vis_filenames.get(idx, "")
        layouts = layout_map.get(idx, [])

        cycle_idx = r.get("cycle_idx", 0)
        gt_letter = r.get("gt_letter", "N/A")
        gt_text = ""
        options_map: Dict[str, str] = {}
        if hrbench_data and 0 <= idx < len(hrbench_data):
            annot_gt_letter, annot_gt_text, options_map = get_sample_gt_details(idx, cycle_idx, hrbench_data)
            if annot_gt_letter != "N/A":
                gt_letter = annot_gt_letter
            gt_text = annot_gt_text

        pred_letter = r.get("predicted_letter", "None") or "None"
        pred_text = options_map.get(pred_letter, "") if pred_letter != "None" else ""

        html_records.append({
            "index": idx,
            "question": r.get("question", ""),
            "gt": gt_letter,
            "gt_text": gt_text,
            "gt_display": f"[{gt_letter}] {gt_text}".strip() if gt_text else f"[{gt_letter}]",
            "pred": pred_letter,
            "pred_text": pred_text,
            "pred_display": f"[{pred_letter}] {pred_text}".strip() if pred_text else f"[{pred_letter}]",
            "is_correct": r.get("is_correct", False),
            "primary": r.get("primary_answer", ""),
            "route": r.get("route_selected", ""),
            "latency": round(r.get("latency_sec", 0.0), 2),
            "image": img_file,
            "layouts": layouts,
            "is_cross_object": is_sample_cross_object(r),
        })

    json_data = json.dumps(html_records)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MSSearch Benchmark Visualizer</title>
  <style>
    :root {{
      --bg: #0f172a;
      --surface: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --cyan: #38bdf8;
      --green: #4ade80;
      --red: #f87171;
      --gold: #fbbf24;
      --purple: #a855f7;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }}
    header {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 24px;
    }}
    h1 {{ font-size: 24px; font-weight: 700; color: var(--cyan); margin-bottom: 12px; }}
    .stats-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-top: 12px;
    }}
    .stat-badge {{
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border);
      padding: 8px 16px;
      border-radius: 8px;
      font-size: 14px;
    }}
    .stat-badge strong {{ color: var(--gold); }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 24px;
      align-items: center;
    }}
    .btn {{
      background: var(--surface);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.15s ease;
    }}
    .btn:hover, .btn.active {{
      background: var(--cyan);
      color: #0f172a;
      border-color: var(--cyan);
    }}
    input[type="text"] {{
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 13px;
      min-width: 280px;
    }}
    .gallery-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
      gap: 20px;
    }}
    .sample-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .sample-card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }}
    .card-header {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .badge {{
      font-size: 11px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 4px;
      text-transform: uppercase;
    }}
    .badge.correct {{ background: rgba(74, 222, 128, 0.15); color: var(--green); border: 1px solid var(--green); }}
    .badge.incorrect {{ background: rgba(248, 113, 113, 0.15); color: var(--red); border: 1px solid var(--red); }}
    .badge.layout {{ background: rgba(168, 85, 247, 0.2); color: var(--purple); border: 1px solid var(--purple); cursor: pointer; text-decoration: none; }}
    .badge.layout:hover {{ background: var(--purple); color: #fff; }}
    .card-body {{ padding: 14px 16px; flex-grow: 1; }}
    .card-q {{ font-weight: 600; font-size: 14px; margin-bottom: 8px; color: var(--text); }}
    .card-meta {{ font-size: 12px; color: var(--muted); margin-bottom: 10px; display: flex; gap: 12px; }}
    .layout-links {{ display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; align-items: center; font-size: 11px; }}
    .card-thumb {{
      width: 100%;
      height: 240px;
      object-fit: cover;
      cursor: pointer;
      border-top: 1px solid var(--border);
      background: #000;
    }}
    /* Modal */
    .modal {{
      display: none;
      position: fixed;
      z-index: 1000;
      top: 0; left: 0; width: 100%; height: 100%;
      background: rgba(15, 23, 42, 0.95);
      justify-content: center;
      align-items: center;
      padding: 24px;
    }}
    .modal.active {{ display: flex; }}
    .modal img {{
      max-width: 96%;
      max-height: 92%;
      border-radius: 8px;
      box-shadow: 0 0 32px rgba(0,0,0,0.8);
      border: 1px solid var(--border);
    }}
    .modal-close {{
      position: absolute;
      top: 20px; right: 28px;
      font-size: 32px;
      color: var(--text);
      cursor: pointer;
    }}
  </style>
</head>
<body>
  <header>
    <h1>MSSearch Benchmark Visualization Dashboard</h1>
    <div class="stats-row">
      <div class="stat-badge">Total Samples: <strong>{total}</strong></div>
      <div class="stat-badge">Accuracy: <strong>{correct_count}/{total} ({acc_pct:.1f}%)</strong></div>
      <div class="stat-badge">Avg Latency: <strong>{avg_lat:.2f}s</strong></div>
      <div class="stat-badge">Routes: <strong>{', '.join([f'{k}: {v}' for k, v in routes.items()])}</strong></div>
    </div>
  </header>

  <div class="controls">
    <button class="btn active" onclick="setFilter('all')">All ({total})</button>
    <button class="btn" onclick="setFilter('correct')">Correct ({correct_count})</button>
    <button class="btn" onclick="setFilter('incorrect')">Incorrect ({total - correct_count})</button>
    <button class="btn" onclick="setFilter('cross')">Cross-Object</button>
    <button class="btn" onclick="setFilter('single')">Single-Object</button>
    <input type="text" id="searchInput" placeholder="Search question, sample #, or answer..." oninput="renderGallery()">
  </div>

  <div class="gallery-grid" id="galleryGrid"></div>

  <div class="modal" id="imageModal" onclick="closeModal()">
    <span class="modal-close">&times;</span>
    <img id="modalImg" src="">
  </div>

  <script>
    const data = {json_data};
    let currentFilter = 'all';

    function setFilter(filter) {{
      currentFilter = filter;
      document.querySelectorAll('.controls .btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      renderGallery();
    }}

    function renderGallery() {{
      const query = document.getElementById('searchInput').value.toLowerCase().trim();
      const container = document.getElementById('galleryGrid');
      container.innerHTML = '';

      const filtered = data.filter(d => {{
        if (currentFilter === 'correct' && !d.is_correct) return false;
        if (currentFilter === 'incorrect' && d.is_correct) return false;
        if (currentFilter === 'cross' && !d.is_cross_object) return false;
        if (currentFilter === 'single' && d.is_cross_object) return false;
        if (query) {{
          const matchIdx = d.index.toString().includes(query);
          const matchQ = d.question.toLowerCase().includes(query);
          const matchAns = d.primary.toLowerCase().includes(query);
          return matchIdx || matchQ || matchAns;
        }}
        return true;
      }});

      filtered.forEach(d => {{
        const card = document.createElement('div');
        card.className = 'sample-card';
        const badgeCls = d.is_correct ? 'correct' : 'incorrect';
        const badgeText = d.is_correct ? 'CORRECT' : 'INCORRECT';

        let layoutBadges = '';
        if (d.layouts && d.layouts.length > 0) {{
          const feedLabel = d.is_cross_object ? 'Stage 6 Canvas Layouts:' : 'Stage 6 Native Crops:';
          layoutBadges = `<span style="color:var(--muted)">${{feedLabel}}</span> `;
          d.layouts.forEach((l, i) => {{
            const bName = d.is_cross_object ? `Layout #${{i}}` : `Crop #${{i}}`;
            layoutBadges += `<span class="badge layout" onclick="event.stopPropagation(); openModal('stage6_layouts/${{l}}')">${{bName}}</span> `;
          }});
        }}

        card.innerHTML = `
          <div class="card-header">
            <strong>Sample #${{d.index}}</strong>
            <span class="badge ${{badgeCls}}">${{badgeText}}</span>
          </div>
          <div class="card-body">
            <div class="card-q">${{d.question}}</div>
            <div class="card-meta" style="flex-direction:column; gap:4px; margin-bottom:8px;">
              <div>GT: <strong style="color:var(--green)">${{d.gt_display}}</strong></div>
              <div>Pred: <strong style="color:${{d.is_correct ? 'var(--green)' : 'var(--red)'}}">${{d.pred_display}}</strong> &nbsp;|&nbsp; Latency: <strong>${{d.latency}}s</strong></div>
            </div>
            <div style="font-size:12px; color:var(--gold); margin-bottom:6px;">Model Output: '${{d.primary}}'</div>
            <div class="layout-links">${{layoutBadges}}</div>
            <div style="font-size:11px; color:var(--muted)">Route: ${{d.route}}</div>
          </div>
          <img class="card-thumb" src="${{d.image}}" alt="Sample #${{d.index}}" onclick="openModal('${{d.image}}')">
        `;
        container.appendChild(card);
      }});
    }}

    function openModal(src) {{
      document.getElementById('modalImg').src = src;
      document.getElementById('imageModal').classList.add('active');
    }}

    function closeModal() {{
      document.getElementById('imageModal').classList.remove('active');
    }}

    document.addEventListener('keydown', (e) => {{
      if (e.key === 'Escape') closeModal();
    }});

    renderGallery();
  </script>
</body>
</html>
"""
    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return html_path


# ==============================================================================
# MAIN RUNNER & AUTO-PATH RESOLUTION
# ==============================================================================

def resolve_results_path(input_path: str) -> str:
    """
    Resolves results file path.
    1. Returns input_path if it exists directly as a file.
    2. Searches known directories for the most recent results_*.jsonl or results.jsonl.
    """
    candidates_to_check = [
        input_path,
        os.path.abspath(input_path),
        os.path.join(os.path.dirname(__file__), input_path),
        os.path.join("docs/experiments/multisource/MSSearch", input_path),
    ]
    for c in candidates_to_check:
        if os.path.isfile(c) and os.path.getsize(c) > 0:
            return os.path.abspath(c)

    # Search candidates directories
    search_dirs = []
    for c in candidates_to_check:
        if os.path.isdir(c):
            search_dirs.append(c)
        elif os.path.isdir(os.path.dirname(c)):
            search_dirs.append(os.path.dirname(c))

    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs.extend([
        os.path.join(script_dir, "multisource_hrbench", "results"),
        os.path.join(script_dir, "results"),
        script_dir,
        os.path.abspath("docs/experiments/multisource/MSSearch/pilot/multisource_hrbench/results"),
        os.path.abspath("docs/experiments/multisource/MSSearch/pilot/results"),
        os.path.abspath("docs/experiments/multisource/MSSearch/pilot"),
    ])

    timestamped_files = []
    other_files = []
    seen_dirs = set()

    for d in search_dirs:
        d_abs = os.path.abspath(d)
        if d_abs in seen_dirs or not os.path.exists(d_abs):
            continue
        seen_dirs.add(d_abs)
        for f in os.listdir(d_abs):
            if f.endswith(".jsonl"):
                fp = os.path.join(d_abs, f)
                if os.path.isfile(fp) and os.path.getsize(fp) > 0:
                    if f.startswith("results_"):
                        timestamped_files.append((os.path.getmtime(fp), fp))
                    elif f == "results.jsonl":
                        other_files.append((os.path.getmtime(fp), fp))

    found = timestamped_files or other_files
    if found:
        found.sort(reverse=True)
        resolved = os.path.abspath(found[0][1])
        print(f"[Visualizer] Auto-resolved latest results file: {resolved}")
        return resolved

    raise FileNotFoundError(
        f"Results file not found for path: '{input_path}'. Searched directories: {list(seen_dirs)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize MSSearch Pilot Results with Stage 6 Layouts")
    parser.add_argument(
        "--results_path",
        type=str,
        default="pilot/results.jsonl",
        help="Path to results.jsonl or timestamped results_*.jsonl file (auto-detects newest if directory or default)."
    )
    parser.add_argument(
        "--hrbench_dir",
        type=str,
        default=None,
        help="Root path to HRBench dataset."
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="D:/Research/VisualSearch/hr-bench_8k/annotation_hr-bench_8k.json",
        help="Path to HRBench annotation JSON file (defaults to D:/Research/VisualSearch/hr-bench_8k/annotation_hr-bench_8k.json if present)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="vis",
        help="Directory to save visualization PNGs and index.html."
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="all",
        help="Comma-separated sample indices to visualize (e.g. '0,1,2') or 'all'."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process."
    )
    parser.add_argument(
        "--no_html",
        action="store_true",
        help="Disable HTML gallery generation."
    )
    args = parser.parse_args()

    # Resolve paths
    results_path = resolve_results_path(args.results_path)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Visualizer] Loading results from: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[Visualizer] Loaded {len(records)} result records.")

    # Discover HRBench
    base_dir, annot_file = discover_hrbench_dir(
        preferred_path=args.hrbench_dir,
        preferred_annotation=args.annotation_file
    )
    print(f"[Visualizer] Found HRBench directory: {base_dir}")
    print(f"[Visualizer] Loading annotations from: {annot_file}")
    with open(annot_file, "r", encoding="utf-8") as f:
        hrbench_data = json.load(f)

    # Filter samples
    if args.sample_indices != "all":
        target_indices = {int(x.strip()) for x in args.sample_indices.split(",") if x.strip()}
        records = [r for r in records if r.get("sample_index") in target_indices]

    if args.max_samples is not None:
        records = records[:args.max_samples]

    print(f"[Visualizer] Rendering visualizations for {len(records)} sample(s) with Stage 6 Layouts -> '{output_dir}'...")

    vis_filenames: Dict[int, str] = {}
    layout_map: Dict[int, List[str]] = {}

    for i, rec in enumerate(records):
        s_idx = rec.get("sample_index", i)
        print(f"  [{i+1}/{len(records)}] Rendering Sample #{s_idx} ({'CORRECT' if rec.get('is_correct') else 'INCORRECT'})...")
        out_png, layout_fnames = visualize_sample(
            rec=rec,
            hrbench_data=hrbench_data,
            hrbench_base_dir=base_dir,
            output_dir=output_dir,
            max_panel_w=760,
            max_panel_h=650,
            save_crops=True
        )
        vis_filenames[s_idx] = os.path.basename(out_png)
        layout_map[s_idx] = layout_fnames

    # Generate HTML gallery dashboard
    if not args.no_html:
        html_path = generate_html_gallery(
            records=records,
            output_dir=output_dir,
            vis_filenames=vis_filenames,
            layout_map=layout_map,
            hrbench_data=hrbench_data
        )
        print(f"[Visualizer] Interactive HTML dashboard generated: {html_path}")

    print(f"[Visualizer] Completed! All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
