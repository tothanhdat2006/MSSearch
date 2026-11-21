"""
Visualization and Qualitative Telemetry Engine for STASH-Search.
Generates multi-stage bounding box overlays, saves native-resolution candidate crops,
builds publication-grade pipeline overview composites, and exports structured JSON traces.
Built with pure Pillow (PIL) for headless cross-platform execution without external GUI dependencies.
"""

import os
import json
import math
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image, ImageDraw, ImageFont

from .config import STASHConfig
from .stages.quadtree_partitioning import BoundedPatch
from .stages.visual_screening import CandidateExtraction

# Cohesive publication color palette (RGB)
COLOR_EVIDENCE_WINNER = (46, 204, 113)     # Emerald Green
COLOR_SCREENED_CROP   = (52, 152, 219)     # Ocean Blue
COLOR_FUSED_BOX       = (155, 89, 182)     # Royal Purple
COLOR_QUADTREE_LEAF   = (26, 188, 156)     # Teal / Aqua
COLOR_GT_HIGHLIGHT    = (241, 196, 15)     # Warm Gold
COLOR_PRUNED_BOX      = (231, 76, 60)      # Muted Red / Coral
COLOR_BACKGROUND_CARD = (248, 249, 250)    # Light Gray
COLOR_HEADER_BG       = (27, 42, 53)       # Deep Slate / Navy
COLOR_TEXT_PRIMARY    = (33, 37, 41)       # Dark Charcoal
COLOR_TEXT_SECONDARY  = (108, 117, 125)    # Medium Gray


def _get_default_font(size: int = 12) -> ImageFont.ImageFont:
    """Returns PIL default font or attempts to scale if possible."""
    try:
        # Windows standard fonts
        for font_name in ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"]:
            try:
                return ImageFont.truetype(font_name, size=size)
            except Exception:
                continue
    except Exception:
        pass
    return ImageFont.load_default()


def draw_box_with_badge(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    label: Optional[str] = None,
    width: int = 3,
    font: Optional[ImageFont.ImageFont] = None
) -> None:
    """
    Renders bounding box with customizable line width and a high-contrast label pill badge.
    """
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

    if label:
        if font is None:
            font = _get_default_font(12)

        # Estimate text bbox
        try:
            bbox = font.getbbox(label)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(label) * 7
            text_h = 12

        pad_x, pad_y = 4, 2
        pill_x1 = max(0, x1)
        pill_y1 = max(0, y1 - text_h - 2 * pad_y)
        pill_x2 = pill_x1 + text_w + 2 * pad_x
        pill_y2 = pill_y1 + text_h + 2 * pad_y

        # Draw pill background and text
        draw.rectangle([pill_x1, pill_y1, pill_x2, pill_y2], fill=color)
        draw.text((pill_x1 + pad_x, pill_y1 + pad_y), label, fill=(255, 255, 255), font=font)


def render_stage2_quadtree_overlay(
    image: Image.Image,
    patches: List[BoundedPatch],
    img_idx: int,
    box_width: int = 2
) -> Image.Image:
    """
    Renders quadtree candidate leaf boxes color-coded by visual alignment score.
    """
    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    font = _get_default_font(11)

    img_patches = [p for p in patches if p.img_idx == img_idx]
    for p in img_patches:
        score = p.alignment_score if p.alignment_score is not None else 0.0
        label = f"S:{score:.3f}"
        draw_box_with_badge(
            draw=draw,
            box=p.box,
            color=COLOR_QUADTREE_LEAF,
            label=label,
            width=box_width,
            font=font
        )
    return img_copy


def render_stage3_screened_overlay(
    image: Image.Image,
    extractions: List[CandidateExtraction],
    img_idx: int,
    box_width: int = 3
) -> Image.Image:
    """
    Renders retained, expanded candidate crop boxes evaluated by MLLM.
    """
    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    font = _get_default_font(12)

    img_extractions = [e for e in extractions if e.img_idx == img_idx]
    for rank, e in enumerate(img_extractions):
        label = f"#{rank} S:{e.alignment_score:.2f} Ans:'{e.final_answer[:12]}'"
        draw_box_with_badge(
            draw=draw,
            box=e.expanded_box,
            color=COLOR_SCREENED_CROP,
            label=label,
            width=box_width,
            font=font
        )
    return img_copy


def render_stage4_evidence_overlay(
    image: Image.Image,
    winning_extraction: Optional[CandidateExtraction],
    other_extractions: List[CandidateExtraction],
    img_idx: int,
    is_gt_image: bool = False
) -> Image.Image:
    """
    Renders supporting evidence image with winning crop highlighted in emerald green.
    """
    img_copy = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    font = _get_default_font(13)

    # First draw other non-winning candidate boxes lightly
    img_others = [e for e in other_extractions if e.img_idx == img_idx and e != winning_extraction]
    for e in img_others:
        draw_box_with_badge(
            draw=draw,
            box=e.expanded_box,
            color=(180, 190, 200),
            label=None,
            width=1,
            font=font
        )

    # Draw winning crop with prominent green border and badge
    if winning_extraction and winning_extraction.img_idx == img_idx:
        badge = f"[WINNER] S:{winning_extraction.alignment_score:.3f} | '{winning_extraction.final_answer}'"
        draw_box_with_badge(
            draw=draw,
            box=winning_extraction.expanded_box,
            color=COLOR_EVIDENCE_WINNER,
            label=badge,
            width=4,
            font=font
        )

    # If this is the GT image, add top-left ground truth tag
    if is_gt_image:
        draw.rectangle([10, 10, 160, 36], fill=COLOR_GT_HIGHLIGHT)
        draw.text((16, 14), "GT EVIDENCE IMG", fill=(0, 0, 0), font=font)

    return img_copy


def generate_pipeline_overview_composite(
    gallery_images: List[Image.Image],
    sample_id: str,
    question: str,
    active_objects: List[str],
    extractions: List[CandidateExtraction],
    crops_to_read: List[Image.Image],
    primary_answer: str,
    primary_evidence_img: Optional[int],
    gt_answer: Optional[str] = None,
    gt_image_idx: Optional[int] = None,
    timings: Optional[Dict[str, float]] = None,
    consensus_details: Optional[Dict[str, Any]] = None
) -> Image.Image:
    """
    Constructs a multi-panel publication-grade overview composite image (approx 1600x1000px):
    - Top: Header bar with Query, Search Targets, Timings, Predictions vs Ground Truth.
    - Left: Gallery strip / Winning evidence image with visual search box overlays.
    - Right: Native-resolution candidate crops grid with scores and answers.
    - Bottom: SAC consensus cluster voting summary.
    """
    canvas_w, canvas_h = 1600, 1080
    composite = Image.new("RGB", (canvas_w, canvas_h), color=COLOR_BACKGROUND_CARD)
    draw = ImageDraw.Draw(composite)
    f_title = _get_default_font(18)
    f_body = _get_default_font(13)
    f_small = _get_default_font(11)

    # 1. Header Banner
    header_h = 140
    draw.rectangle([0, 0, canvas_w, header_h], fill=COLOR_HEADER_BG)
    draw.text((24, 16), f"STASH-Search Visual Pipeline Execution | Sample: {sample_id}", fill=(255, 255, 255), font=f_title)
    
    q_text = f"Query: \"{question}\""
    if len(q_text) > 130:
        q_text = q_text[:127] + "..."
    draw.text((24, 46), q_text, fill=(220, 230, 240), font=f_body)

    tgt_str = ", ".join(f"'{o}'" for o in active_objects) if active_objects else "None"
    draw.text((24, 70), f"Active Targets: [{tgt_str}]", fill=(241, 196, 15), font=f_body)

    lat_str = f"Latency: {timings.get('total_latency_sec', 0):.2f}s" if timings else ""
    draw.text((24, 94), f"Gallery Size: N={len(gallery_images)} images | {lat_str}", fill=(180, 200, 215), font=f_small)

    # Predictions vs GT Card (Top Right)
    status_corr = (gt_answer is not None and primary_answer.lower() == str(gt_answer).lower())
    status_img = (gt_image_idx is not None and primary_evidence_img == gt_image_idx)
    badge_color = COLOR_EVIDENCE_WINNER if (status_corr and status_img) else COLOR_PRUNED_BOX
    draw.rectangle([canvas_w - 380, 16, canvas_w - 20, header_h - 16], fill=badge_color)
    draw.text((canvas_w - 365, 24), f"Pred Answer: '{primary_answer}'", fill=(255, 255, 255), font=f_body)
    draw.text((canvas_w - 365, 48), f"GT Answer:   '{gt_answer}'", fill=(255, 255, 255), font=f_body)
    draw.text((canvas_w - 365, 72), f"Pred Img: #{primary_evidence_img} | GT Img: #{gt_image_idx}", fill=(255, 255, 255), font=f_body)
    tag = "PASS (JOINT MATCH)" if (status_corr and status_img) else "EVALUATION COMPLETE"
    draw.text((canvas_w - 365, 96), tag, fill=(255, 255, 255), font=f_title)

    # 2. Left Panel: Primary Evidence Image with Overlays
    panel_y = header_h + 16
    left_w = 860
    left_h = 880

    draw.rectangle([20, panel_y, 20 + left_w, panel_y + left_h], fill=(255, 255, 255), outline=(220, 225, 230), width=1)
    draw.text((36, panel_y + 12), "Supporting Evidence View (Full Image + Bounding Box Overlays)", fill=COLOR_TEXT_PRIMARY, font=f_body)

    ev_idx = primary_evidence_img if (primary_evidence_img is not None and primary_evidence_img < len(gallery_images)) else 0
    base_img = gallery_images[ev_idx]
    winning_e = extractions[0] if extractions else None
    if consensus_details and consensus_details.get("winning_extraction_idx") is not None:
        w_idx = consensus_details["winning_extraction_idx"]
        if w_idx < len(extractions):
            winning_e = extractions[w_idx]

    is_gt = (gt_image_idx is not None and ev_idx == gt_image_idx)
    overlay_img = render_stage4_evidence_overlay(
        image=base_img,
        winning_extraction=winning_e,
        other_extractions=extractions,
        img_idx=ev_idx,
        is_gt_image=is_gt
    )

    # Fit overlay image into left panel area
    avail_w = left_w - 32
    avail_h = left_h - 60
    scale = min(avail_w / max(1, overlay_img.width), avail_h / max(1, overlay_img.height))
    new_w, new_h = max(1, int(overlay_img.width * scale)), max(1, int(overlay_img.height * scale))
    disp_overlay = overlay_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    pos_x = 20 + (left_w - new_w) // 2
    pos_y = panel_y + 45 + (avail_h - new_h) // 2
    composite.paste(disp_overlay, (pos_x, pos_y))

    # 3. Right Panel: Native Resolution Inspected Crops
    right_x = 20 + left_w + 16
    right_w = canvas_w - right_x - 20
    draw.rectangle([right_x, panel_y, right_x + right_w, panel_y + left_h], fill=(255, 255, 255), outline=(220, 225, 230), width=1)
    draw.text((right_x + 16, panel_y + 12), "Native-Resolution Inspected Crops (Stage 3)", fill=COLOR_TEXT_PRIMARY, font=f_body)

    # Display up to top 6 crops in 2 columns x 3 rows
    crop_display_count = min(6, len(crops_to_read))
    crop_cell_w = (right_w - 48) // 2
    crop_cell_h = 160
    
    for c_idx in range(crop_display_count):
        col = c_idx % 2
        row = c_idx // 2
        cx = right_x + 16 + col * (crop_cell_w + 16)
        cy = panel_y + 45 + row * (crop_cell_h + 12)

        c_img = crops_to_read[c_idx]
        e = extractions[c_idx] if c_idx < len(extractions) else None

        # Draw cell container
        is_winner = (e == winning_e)
        border_col = COLOR_EVIDENCE_WINNER if is_winner else (200, 210, 220)
        draw.rectangle([cx, cy, cx + crop_cell_w, cy + crop_cell_h], outline=border_col, width=2 if is_winner else 1)

        # Thumbnail crop
        c_thumb_w = crop_cell_w - 12
        c_thumb_h = crop_cell_h - 40
        c_scale = min(c_thumb_w / max(1, c_img.width), c_thumb_h / max(1, c_img.height))
        cw, ch = max(1, int(c_img.width * c_scale)), max(1, int(c_img.height * c_scale))
        thumb = c_img.resize((cw, ch), Image.Resampling.LANCZOS)
        tx = cx + (crop_cell_w - cw) // 2
        ty = cy + (c_thumb_h - ch) // 2 + 4
        composite.paste(thumb, (tx, ty))

        # Label under crop
        score_val = e.alignment_score if e else 0.0
        ans_val = e.final_answer if e else "N/A"
        label_text = f"#{c_idx} Img:{e.img_idx if e else 0} S:{score_val:.2f} | '{ans_val[:14]}'"
        lbl_bg = COLOR_EVIDENCE_WINNER if is_winner else COLOR_HEADER_BG
        draw.rectangle([cx + 2, cy + crop_cell_h - 26, cx + crop_cell_w - 2, cy + crop_cell_h - 2], fill=lbl_bg)
        draw.text((cx + 6, cy + crop_cell_h - 22), label_text, fill=(255, 255, 255), font=f_small)

    # 4. Consensus Voting Summary (Bottom of right panel)
    cons_y = panel_y + 45 + 3 * (crop_cell_h + 12) + 10
    draw.rectangle([right_x + 16, cons_y, right_x + right_w - 16, panel_y + left_h - 16], fill=(245, 247, 250))
    draw.text((right_x + 24, cons_y + 10), "Source-Attributed Consensus (SAC) Clusters:", fill=COLOR_TEXT_PRIMARY, font=f_body)

    clusters = consensus_details.get("clusters", []) if consensus_details else []
    for c_i, c in enumerate(clusters[:4]):
        cy_line = cons_y + 36 + c_i * 24
        rep = c.get("representative", "")
        cnt = c.get("count", 1)
        wgt = c.get("weighted_score", 0.0)
        is_win = (rep == primary_answer)
        prefix = "★ [WINNER]" if is_win else "•"
        txt_col = (39, 174, 96) if is_win else COLOR_TEXT_SECONDARY
        draw.text((right_x + 28, cy_line), f"{prefix} \"{rep}\" : Support={cnt} | Weight={wgt:.2f}", fill=txt_col, font=f_small)

    return composite


def save_sample_visualization(
    vis_root_dir: str,
    benchmark_name: str,
    pool_mode: str,
    sample_id: str,
    gallery_images: List[Image.Image],
    question: str,
    active_objects: List[str],
    candidate_patches: List[BoundedPatch],
    extractions: List[CandidateExtraction],
    crops_to_read: List[Image.Image],
    primary_answer: str,
    primary_evidence_img: Optional[int],
    consensus_details: Optional[Dict[str, Any]] = None,
    timings: Optional[Dict[str, float]] = None,
    gt_answer: Optional[str] = None,
    gt_image_idx: Optional[int] = None,
    config: Optional[STASHConfig] = None
) -> Dict[str, Any]:
    """
    Main coordinator function to generate and save all visual artifacts and machine-readable traces.
    Creates structured directory under:
        <vis_root_dir>/<benchmark_name>/<pool_mode>/<clean_sample_id>/
    """
    clean_id = str(sample_id).replace("/", "_").replace("\\", "_")
    sample_dir = os.path.join(vis_root_dir, benchmark_name, pool_mode, clean_id)
    os.makedirs(sample_dir, exist_ok=True)

    # Subdirectories
    dir_inputs = os.path.join(sample_dir, "00_inputs")
    dir_stage2 = os.path.join(sample_dir, "01_stage2_partitioning")
    dir_stage3 = os.path.join(sample_dir, "02_stage3_screening")
    dir_crops  = os.path.join(dir_stage3, "crops")
    dir_stage4 = os.path.join(sample_dir, "03_stage4_consensus")
    for d in [dir_inputs, dir_stage2, dir_stage3, dir_crops, dir_stage4]:
        os.makedirs(d, exist_ok=True)

    # 1. Inputs: Save query info and gallery images
    query_info = {
        "sample_id": sample_id,
        "benchmark": benchmark_name,
        "pool_mode": pool_mode,
        "question": question,
        "active_objects": active_objects,
        "gt_answer": gt_answer,
        "gt_image_idx": gt_image_idx,
        "num_gallery_images": len(gallery_images),
    }
    with open(os.path.join(dir_inputs, "query.json"), "w", encoding="utf-8") as f:
        json.dump(query_info, f, indent=2)

    for i, img in enumerate(gallery_images):
        gt_tag = "_GT" if (gt_image_idx is not None and i == gt_image_idx) else ""
        img.save(os.path.join(dir_inputs, f"gallery_img_{i:02d}{gt_tag}.jpg"), quality=90)

    # 2. Stage 2: Partitioning overlays and patch metadata
    patches_meta = [
        {
            "img_idx": p.img_idx,
            "box": list(p.box),
            "depth": p.depth,
            "alignment_score": round(float(p.alignment_score), 4) if p.alignment_score is not None else None,
            "entropy": round(float(p.entropy), 4),
            "area": p.area,
        }
        for p in candidate_patches
    ]
    with open(os.path.join(dir_stage2, "partition_patches.json"), "w", encoding="utf-8") as f:
        json.dump(patches_meta, f, indent=2)

    for i, img in enumerate(gallery_images):
        overlay2 = render_stage2_quadtree_overlay(img, candidate_patches, img_idx=i)
        overlay2.save(os.path.join(dir_stage2, f"quadtree_overlay_img_{i:02d}.jpg"), quality=88)

    # 3. Stage 3: Screening overlays, crop images, and extractions metadata
    crops_metadata = []
    for idx, (c_img, e) in enumerate(zip(crops_to_read, extractions)):
        safe_ans = "".join(c for c in e.final_answer if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')[:16]
        c_filename = f"crop_{idx:02d}_img{e.img_idx:02d}_score{e.alignment_score:.3f}_{safe_ans}.jpg"
        c_path = os.path.join(dir_crops, c_filename)
        c_img.save(c_path, quality=95)

        crops_metadata.append({
            "crop_idx": idx,
            "img_idx": e.img_idx,
            "box": list(e.box),
            "expanded_box": list(e.expanded_box),
            "alignment_score": round(float(e.alignment_score), 4),
            "crop_area": e.crop_area,
            "raw_answer": e.raw_answer,
            "final_answer": e.final_answer,
            "saved_file": os.path.relpath(c_path, sample_dir)
        })

    with open(os.path.join(dir_stage3, "screening_crops.json"), "w", encoding="utf-8") as f:
        json.dump(crops_metadata, f, indent=2)

    for i, img in enumerate(gallery_images):
        overlay3 = render_stage3_screened_overlay(img, extractions, img_idx=i)
        overlay3.save(os.path.join(dir_stage3, f"screened_boxes_overlay_img_{i:02d}.jpg"), quality=88)

    # 4. Stage 4: Consensus details and winning evidence overlay
    if consensus_details:
        with open(os.path.join(dir_stage4, "consensus_clusters.json"), "w", encoding="utf-8") as f:
            json.dump(consensus_details, f, indent=2)

    winning_e = extractions[0] if extractions else None
    if consensus_details and consensus_details.get("winning_extraction_idx") is not None:
        w_idx = consensus_details["winning_extraction_idx"]
        if w_idx < len(extractions):
            winning_e = extractions[w_idx]

    ev_img_idx = primary_evidence_img if (primary_evidence_img is not None and primary_evidence_img < len(gallery_images)) else 0
    overlay4 = render_stage4_evidence_overlay(
        image=gallery_images[ev_img_idx],
        winning_extraction=winning_e,
        other_extractions=extractions,
        img_idx=ev_img_idx,
        is_gt_image=(gt_image_idx is not None and ev_img_idx == gt_image_idx)
    )
    overlay4.save(os.path.join(dir_stage4, f"evidence_winner_overlay_img_{ev_img_idx:02d}.jpg"), quality=90)

    # 5. Pipeline Overview Composite Figure
    overview_img = generate_pipeline_overview_composite(
        gallery_images=gallery_images,
        sample_id=sample_id,
        question=question,
        active_objects=active_objects,
        extractions=extractions,
        crops_to_read=crops_to_read,
        primary_answer=primary_answer,
        primary_evidence_img=primary_evidence_img,
        gt_answer=gt_answer,
        gt_image_idx=gt_image_idx,
        timings=timings,
        consensus_details=consensus_details
    )
    overview_path = os.path.join(sample_dir, "pipeline_overview.jpg")
    overview_img.save(overview_path, quality=90)

    # 6. Consolidated End-to-End Trace JSON
    trace = {
        "sample_id": sample_id,
        "benchmark": benchmark_name,
        "pool_mode": pool_mode,
        "question": question,
        "active_objects": active_objects,
        "gt_answer": gt_answer,
        "gt_image_idx": gt_image_idx,
        "predicted_answer": primary_answer,
        "predicted_evidence_image": primary_evidence_img,
        "is_correct_answer": (gt_answer is not None and primary_answer.lower() == str(gt_answer).lower()),
        "is_correct_image": (gt_image_idx is not None and primary_evidence_img == gt_image_idx),
        "timings_sec": timings or {},
        "stage2_patches_count": len(candidate_patches),
        "stage3_crops_count": len(extractions),
        "crops": crops_metadata,
        "consensus": consensus_details or {},
        "artifacts": {
            "pipeline_overview": os.path.relpath(overview_path, sample_dir),
            "evidence_overlay": f"03_stage4_consensus/evidence_winner_overlay_img_{ev_img_idx:02d}.jpg"
        }
    }
    with open(os.path.join(sample_dir, "pipeline_trace.json"), "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)

    return trace
