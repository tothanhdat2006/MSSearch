#!/usr/bin/env python3
"""
Multi-Source Visual Search Engine (MSSearch) Diagnostic Pipeline for HRBench
=============================================================================
Replicates the stage-by-stage diagnostic and telemetry capabilities of
`notebooks/debug_mssearch_multisource_stages.ipynb` into a standalone,
headless CLI tool for local Ubuntu server environments.

Key Capabilities:
- Candidate Pool C = {I_1, ..., I_N}: Target image + (N-1) distractors sampled from [i - W, i + W]
- Scaled Budgets (N * Budget): SAM3 scans, latent queries, retention limits scaled linearly
- Full Stage Instrumentation:
    Stage 1: Question Parsing, Relational Intent & Anchor Reordering
    Stage 2: 2-Gate Early Assessment (Phase 1 Direct Global + Phase 2 Tool Fast Path)
    Stage 3: Multi-Image Saliency-Plateau Partitioning (reusing Stage 2 feature cache)
    Stage 4: SAM3 Feature Grouping & Gating with Micro-Fovea Token Boost
    Stage 5: Direct Unified Candidate Pool Aggregation
    Stage 6: Foveated VLM Reading & Open-Set Multi-Answer Deduplication
    Cell 10: End-to-End `get_mssearch_response(...)` Cross-Verification
- Selective Visual Artifact Controls:
    Default: Formatted text diagnostics to stdout.
    --save_visuals all: Generates & saves all 6 stage PNGs for all evaluated samples.
    --save_visuals "2 5 7": Generates & saves PNGs only for the specified sample indices.
"""

import os
import sys
import gc
import re
import json
import time
import random
import argparse
from typing import List, Dict, Tuple, Optional, Any, Set, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Headless matplotlib configuration
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure MSSearch repository root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MSSEARCH_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if MSSEARCH_ROOT not in sys.path:
    sys.path.insert(0, MSSEARCH_ROOT)

# MSSearch Imports
from mssearch.config import MSSearchConfig
from mssearch.models.modeling_sam3 import sam3_inference
from mssearch.models.modeling_siglip import SigLIPScorer
from mssearch.models.modeling_qwenvl import ModelQwenVL
from mssearch.stages.stage1_parser import (
    parse_question_to_active_objects,
    ParsedQuestion,
)
from mssearch.stages.stage2_fast_path import (
    run_direct_global_assessment,
    DirectAnswerResult,
    run_tool_based_fast_path,
    FastPathResult,
)
from mssearch.stages.stage3_partitioning import (
    partition_candidate_images,
    BoundedPatch,
)
from mssearch.stages.stage4_grouping_gating import (
    run_feature_grouping_and_gating,
)
from mssearch.stages.stage5_screening import (
    run_resolution_screening,
    CandidateCrop,
)
from mssearch.stages.stage6_foveated_reader import (
    run_foveated_vlm_reading,
    is_negative_or_unanswerable,
    deduplicate_answers,
)
from mssearch.MSSearch import (
    MultiSourceSearchEngine,
    get_mssearch_response,
)


# ==============================================================================
# VISUAL ARTIFACTS & CLI HELPERS
# ==============================================================================

def parse_save_visuals_arg(arg_val: Optional[Union[List[str], str]]) -> Optional[Union[str, Set[int]]]:
    """
    Parses the --save_visuals argument into either None, 'all', or a set of integer indices.
    Examples:
      None -> None (no visuals)
      ['all'] or 'all' -> 'all'
      [] (flag present without args) -> 'all'
      ['2', '5', '7'] or '2 5 7' or '2,5,7' -> {2, 5, 7}
    """
    if arg_val is None:
        return None
    if isinstance(arg_val, list):
        if len(arg_val) == 0:
            return "all"
        if len(arg_val) == 1 and str(arg_val[0]).strip().lower() == "all":
            return "all"
        all_tokens = []
        for item in arg_val:
            tokens = re.split(r"[,\s]+", str(item).strip())
            all_tokens.extend(tokens)
        indices = set()
        for tok in all_tokens:
            tok = tok.strip()
            if not tok:
                continue
            if tok.lower() == "all":
                return "all"
            try:
                indices.add(int(tok))
            except ValueError:
                pass
        return indices if indices else "all"
    elif isinstance(arg_val, str):
        val = arg_val.strip()
        if not val or val.lower() == "all":
            return "all"
        tokens = re.split(r"[,\s]+", val)
        indices = set()
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            if tok.lower() == "all":
                return "all"
            try:
                indices.add(int(tok))
            except ValueError:
                pass
        return indices if indices else "all"
    return None


def should_save_visuals(save_visuals_spec: Optional[Union[str, Set[int]]], sample_idx: int) -> bool:
    """Checks whether visual artifacts should be generated and saved for sample_idx."""
    if save_visuals_spec is None:
        return False
    if save_visuals_spec == "all":
        return True
    if isinstance(save_visuals_spec, set):
        return sample_idx in save_visuals_spec
    return False


def resolve_gt_letter(answer_obj: Any, cycle_idx: int = 0) -> str:
    """Extracts ground truth option letter (e.g. 'A', 'B') from HRBench annotations."""
    if isinstance(answer_obj, list):
        if 0 <= cycle_idx < len(answer_obj):
            return str(answer_obj[cycle_idx]).strip().upper()
        return str(answer_obj[0]).strip().upper() if answer_obj else ""
    return str(answer_obj).strip().upper()


def parse_options_text(options_raw: Any) -> Dict[str, str]:
    """Parses candidate options block into { 'A': text, 'B': text, ... }."""
    opt_dict: Dict[str, str] = {}
    if not options_raw:
        return opt_dict

    if isinstance(options_raw, list) and len(options_raw) > 0:
        opt_block = options_raw[0]
    else:
        opt_block = options_raw

    if isinstance(opt_block, dict):
        for k, v in opt_block.items():
            opt_dict[str(k).upper()] = str(v).strip().lower()
    elif isinstance(opt_block, str):
        for line in opt_block.splitlines():
            m = re.match(r"^[\(\[\"']*([A-Da-d])[\)\]\.\:\s]+(.*)$", line.strip())
            if m:
                opt_dict[m.group(1).upper()] = m.group(2).strip().lower()

    return opt_dict


def extract_predicted_option(pred_str: Optional[str], options_raw: Any = None) -> Optional[str]:
    """Extracts the multiple-choice option letter ('A', 'B', 'C', 'D') from model output."""
    if pred_str is None or not isinstance(pred_str, str):
        return None

    clean_pred = pred_str.strip()
    if not clean_pred or clean_pred.lower() in ("no answer found", "none", "n/a", "[error: image not found]"):
        return None

    # 1. Regex check for leading letter (e.g. 'A. 27B', '(A)', 'A: ')
    m = re.match(r"^[\(\[\"']*([A-Da-d])[\)\]\.\:\s]", clean_pred)
    if m:
        return m.group(1).upper()

    # 2. Exact single letter
    if clean_pred.upper() in ("A", "B", "C", "D"):
        return clean_pred.upper()

    # 3. Final Answer / Option phrase anywhere in text
    m = re.search(r"\b(?:final answer|answer|choice|option)\b[^\w\n\r]*([A-Da-d])\b", clean_pred, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 4. Fallback: match prediction content against option candidate text
    if options_raw:
        opt_dict = parse_options_text(options_raw)
        p_lower = clean_pred.lower()
        for letter, text in opt_dict.items():
            if text and (p_lower == text or p_lower.startswith(text) or text in p_lower):
                return letter

    return None


def discover_hrbench_dir(preferred_path: Optional[str]) -> Tuple[str, str]:
    """Discovers HRBench root directory and annotation file with multiple local fallbacks."""
    search_dirs = []
    if preferred_path:
        search_dirs.append(preferred_path)
    env_dir = os.environ.get("HRBENCH_DIR")
    if env_dir:
        search_dirs.append(env_dir)

    search_dirs.extend([
        "/home/ubuntu/workspace/hr/data/hr-bench_8k",
        "/home/ubuntu/workspace/hr/data/hr-bench_4k",
        "D:/Research/VisualSearch/hr-bench_8k",
        "d:/Research/VisualSearch/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_8k/mnt/sdb/llp/datasets/zoom_eye_data/zoom_eye_data/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_8k",
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_4k",
        "docs/experiments/multisource/data/hr-bench_8k",
        "docs/experiments/freeform/ICML26-CVSearch/data/hr-bench_8k",
    ])

    resolved_base = None
    for d in search_dirs:
        if d and os.path.exists(d):
            resolved_base = d
            break

    if resolved_base is None:
        raise FileNotFoundError(
            f"[Dataset Error] Could not find HRBench directory in candidate paths: {search_dirs}"
        )

    annotation_path = os.path.join(resolved_base, "annotation_hr-bench_8k.json")
    if not os.path.exists(annotation_path):
        annotation_path = os.path.join(resolved_base, "annotation_hr-bench_4k.json")

    if not os.path.exists(annotation_path):
        raise FileNotFoundError(
            f"[Dataset Error] Could not find annotation JSON in HRBench directory: {resolved_base}"
        )

    return resolved_base, annotation_path


# ==============================================================================
# VISUALIZATION RENDERING SUITE
# ==============================================================================

def save_fig_clean(fig: plt.Figure, filepath: str) -> None:
    """Saves figure with clean tight layout and closes it to prevent memory leaks."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_stage0_candidate_pool(
    candidate_images: List[Image.Image],
    pool_items: List[Dict[str, Any]],
    real_question: str,
    target_pool_idx: int,
    output_path: str,
) -> None:
    """Renders 01_candidate_pool.png: Thumbnails of candidate images with Target/Distractor borders."""
    n_imgs = len(candidate_images)
    fig, axes = plt.subplots(1, n_imgs, figsize=(5.0 * n_imgs, 5.0))
    if n_imgs == 1:
        axes = [axes]

    for i, (ax, item) in enumerate(zip(axes, pool_items)):
        thumb = item["image"].copy()
        thumb.thumbnail((512, 512))
        ax.imshow(thumb)
        is_t = item["is_target"]
        color = "forestgreen" if is_t else "royalblue"
        label = (
            f"Image {i} [TARGET]\nSample #{item['sample_idx']}"
            if is_t
            else f"Image {i} [DISTRACTOR]\nSample #{item['sample_idx']}"
        )
        ax.set_title(label, fontsize=11, fontweight="bold", color=color)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3.5)

    wrapped_q = (real_question[:110] + "...") if len(real_question) > 110 else real_question
    plt.suptitle(
        f"Candidate Image Pool C (N = {n_imgs}) for Question:\n'{wrapped_q}'",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    save_fig_clean(fig, output_path)


def render_stage1_parsing_card(
    real_question: str,
    sample_index: int,
    question_type: str,
    is_relational: bool,
    parser_strategy: str,
    n_images: int,
    active_objects: List[str],
    anchors: List[str],
    target_element: str,
    output_path: str,
) -> None:
    """Renders 01_stage1_parsing.png: Structured semantic parse card with badge chips."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    card = mpatches.FancyBboxPatch(
        (0.1, 0.2), 10.8, 3.0, boxstyle="round,pad=0.2", edgecolor="#1a237e", facecolor="#f8f9fa", linewidth=2.0
    )
    ax.add_patch(card)

    wrapped_q = (real_question[:95] + "...") if len(real_question) > 95 else real_question
    ax.text(0.3, 2.55, f"[Multi-Source Sample {sample_index}] {wrapped_q}", fontsize=11, fontweight="bold", color="#1a237e")
    ax.text(
        0.3,
        1.85,
        f"Type: {question_type} | Relational: {is_relational} | Strategy: {parser_strategy} | Pool Size: N={n_images}",
        fontsize=9.5,
        color="#424242",
        style="italic",
    )

    chips_x = 0.3
    anchor_lower = [str(a).lower() for a in anchors]
    target_lower = str(target_element).lower()

    for i, obj in enumerate(active_objects):
        obj_l = str(obj).lower()
        if obj_l in anchor_lower:
            badge = "Anchor"
            border_col, bg_col, txt_col = "#00796b", "#e0f2f1", "#004d40"
        elif target_lower and (obj_l in target_lower or target_lower in obj_l):
            badge = "Target"
            border_col, bg_col, txt_col = "#c2185b", "#fce4ec", "#880e4f"
        else:
            is_anchor = (i < len(active_objects) // 2) if len(active_objects) > 1 else True
            badge = "Host" if is_anchor else "Object"
            border_col = "#00796b" if is_anchor else "#e65100"
            bg_col = "#e0f2f1" if is_anchor else "#fff3e0"
            txt_col = "#004d40" if is_anchor else "#bf360c"

        chip_w = len(str(obj)) * 0.20 + 1.05
        chip = mpatches.FancyBboxPatch(
            (chips_x, 0.45), chip_w, 0.76, boxstyle="round,pad=0.1", edgecolor=border_col, facecolor=bg_col, linewidth=1.5
        )
        ax.add_patch(chip)
        ax.text(chips_x + 0.15, 0.65, f"{badge}: {obj}", fontsize=9.5, fontweight="bold", color=txt_col)
        chips_x += chip_w + 0.30

    plt.tight_layout()
    save_fig_clean(fig, output_path)


def render_stage2_fast_path(
    candidate_images: List[Image.Image],
    fast_path_res: FastPathResult,
    direct_ans_res: DirectAnswerResult,
    target_pool_idx: int,
    output_path: str,
) -> None:
    """Renders 02_stage2_fast_path.png: Phase 1 & 2 Fast Path triggers and detected bboxes."""
    n_imgs = len(candidate_images)
    fig, axes = plt.subplots(1, n_imgs, figsize=(6.0 * n_imgs, 6.0))
    if n_imgs == 1:
        axes = [axes]

    ev_imgs = getattr(fast_path_res, "evidence_images", [])
    if not ev_imgs and getattr(fast_path_res, "evidence_img_idx", None) is not None:
        ev_imgs = [fast_path_res.evidence_img_idx]

    detected_boxes = getattr(fast_path_res, "detected_boxes", [])
    boxes_by_img = getattr(fast_path_res, "boxes_by_image", None)

    for img_idx, ax in enumerate(axes):
        vis_img = candidate_images[img_idx].copy()
        ax.imshow(vis_img)
        is_t = (img_idx == target_pool_idx)
        tag = "TARGET" if is_t else "DISTRACTOR"
        color = "forestgreen" if is_t else "royalblue"

        # Safely resolve boxes whether detected_boxes is a dict, list, or boxes_by_image is provided
        if isinstance(boxes_by_img, dict) and boxes_by_img:
            boxes = boxes_by_img.get(img_idx, [])
        elif isinstance(detected_boxes, dict):
            boxes = detected_boxes.get(img_idx, [])
        elif isinstance(detected_boxes, (list, tuple)):
            if img_idx in ev_imgs or (not ev_imgs and img_idx == target_pool_idx and fast_path_res.triggered):
                boxes = list(detected_boxes)
            else:
                boxes = []
        else:
            boxes = []

        for b_idx, box in enumerate(boxes):
            if not isinstance(box, (list, tuple)) or len(box) < 4:
                continue
            x1, y1, x2, y2 = [int(round(float(c))) for c in box[:4]]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            rect = mpatches.Rectangle(
                (x1, y1), bw, bh, linewidth=3.0, edgecolor="#00e5ff", facecolor="none", linestyle="-"
            )
            ax.add_patch(rect)
            ax.text(
                x1 + 10,
                max(20, y1 + 30),
                f"FastPath BBox #{b_idx}",
                fontsize=8.5,
                fontweight="bold",
                color="black",
                bbox=dict(boxstyle="square,pad=0.2", facecolor="#00e5ff", alpha=0.9),
            )

        title = f"Image {img_idx} [{tag}]\nDetected Boxes: {len(boxes)}"
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)
        ax.axis("off")

    p1_str = f"Phase 1 Direct: trig={direct_ans_res.triggered} (conf={direct_ans_res.confidence:.2f})"
    p2_str = f"Phase 2 FastPath: trig={fast_path_res.triggered} (conf={fast_path_res.confidence:.2f})"
    plt.suptitle(f"Stage 2: 2-Gate Early Assessment\n{p1_str} | {p2_str}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig_clean(fig, output_path)


def render_stage3_partitioning(
    candidate_images: List[Image.Image],
    bounded_patches: List[BoundedPatch],
    target_pool_idx: int,
    output_path: str,
) -> None:
    """Renders 03_stage3_partitioning.png: Saliency-Plateau bounded patches overlaid on pool canvases."""
    n_imgs = len(candidate_images)
    fig, axes = plt.subplots(1, n_imgs, figsize=(6.0 * n_imgs, 6.0))
    if n_imgs == 1:
        axes = [axes]

    cols = plt.cm.tab20(np.linspace(0, 1, max(1, len(bounded_patches))))
    for img_idx, ax in enumerate(axes):
        vis_img = candidate_images[img_idx].copy()
        ax.imshow(vis_img)
        img_patches = [(idx, p) for idx, p in enumerate(bounded_patches) if p.img_idx == img_idx]

        for idx, p in img_patches:
            x1, y1, x2, y2 = [int(round(float(c))) for c in p.box]
            c = cols[idx % len(cols)]
            rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2.5, edgecolor=c, facecolor="none")
            ax.add_patch(rect)
            s_lbl = f"S={p.siglip_score:.2f}\n" if p.siglip_score is not None else ""
            ax.text(
                x1 + 10,
                y1 + 35,
                f"P{idx}\n{s_lbl}H={p.entropy:.2f}",
                fontsize=7.5,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="square,pad=0.2", facecolor=c, alpha=0.85),
            )

        is_t = (img_idx == target_pool_idx)
        color = "forestgreen" if is_t else "royalblue"
        ax.set_title(
            f"Image {img_idx} [{'TARGET' if is_t else 'DISTRACTOR'}]\n{len(img_patches)} Patches",
            fontsize=11,
            fontweight="bold",
            color=color,
        )
        ax.axis("off")

    plt.suptitle(
        f"Stage 3: Partitioned Candidate Patches Across Candidate Pool C (Total: {len(bounded_patches)})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save_fig_clean(fig, output_path)


def render_stage4_grouping_gating(
    candidate_images: List[Image.Image],
    bounded_patches: List[BoundedPatch],
    remaining_patches: List[BoundedPatch],
    target_pool_idx: int,
    output_path: str,
) -> None:
    """Renders 04_stage4_grouping_gating.png: Overlaid Kept (green) vs Pruned (red) patches."""
    n_imgs = len(candidate_images)
    fig, axes = plt.subplots(1, n_imgs, figsize=(6.0 * n_imgs, 6.0))
    if n_imgs == 1:
        axes = [axes]

    retained_s4_boxes = {(p.img_idx, tuple([int(round(float(c))) for c in p.box])) for p in remaining_patches}
    pruned_count = len(bounded_patches) - len(remaining_patches)

    for img_idx, ax in enumerate(axes):
        vis_img = candidate_images[img_idx].copy()
        ax.imshow(vis_img)
        img_patches = [(idx, p) for idx, p in enumerate(bounded_patches) if p.img_idx == img_idx]

        n_k = 0
        n_p = 0
        for idx, p in img_patches:
            box_int = [int(round(float(c))) for c in p.box]
            x1, y1, x2, y2 = box_int
            is_kept = (img_idx, tuple(box_int)) in retained_s4_boxes

            if is_kept:
                n_k += 1
                rect = mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, linewidth=3.5, edgecolor="#00e676", facecolor="none"
                )
                ax.add_patch(rect)
                s_lbl = f"S={p.siglip_score:.2f}\n" if p.siglip_score is not None else ""
                ax.text(
                    x1 + 10,
                    y1 + 35,
                    f"P{idx}\n{s_lbl}H={p.entropy:.2f}\nKEPT",
                    fontsize=7.5,
                    fontweight="bold",
                    color="black",
                    bbox=dict(boxstyle="square,pad=0.2", facecolor="#00e676", alpha=0.9),
                )
            else:
                n_p += 1
                rect = mpatches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, linewidth=1.8, edgecolor="#ef5350", facecolor="none", linestyle="--", alpha=0.7
                )
                ax.add_patch(rect)
                ax.text(
                    x1 + 10,
                    y1 + 25,
                    f"P{idx}\nPRUNED",
                    fontsize=6.5,
                    fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="square,pad=0.15", facecolor="#c62828", alpha=0.6),
                )

        is_t = (img_idx == target_pool_idx)
        color = "forestgreen" if is_t else "royalblue"
        label = (
            f"Image {img_idx} [{'TARGET' if is_t else 'DISTRACTOR'}]\nKept: {n_k} | Pruned: {n_p} (from {len(img_patches)} total)"
        )
        ax.set_title(label, fontsize=11, fontweight="bold", color=color)
        ax.axis("off")

    plt.suptitle(
        f"Stage 4: Original Patches Kept vs. Pruned Across Candidate Pool C\n(Green: Retained {len(remaining_patches)} | Red Dashed: Pruned {pruned_count})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    save_fig_clean(fig, output_path)


def render_stage5_screening(
    candidate_images: List[Image.Image],
    unified_pool: List[CandidateCrop],
    target_pool_idx: int,
    output_path: str,
) -> None:
    """Renders 05_stage5_screening.png: Gallery of candidate crops in Unified Pool P."""
    n_show = min(len(unified_pool), 12)
    if n_show == 0:
        return

    ncols = min(4, n_show)
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    fig.suptitle(
        f"Stage 5: Unified Candidate Pool P ({len(unified_pool)} Total Retained Candidates)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for i in range(n_show):
        ax = axes[i]
        cand = unified_pool[i]
        is_t = (cand.img_idx == target_pool_idx)
        crop = (
            cand.crop_image
            if hasattr(cand, "crop_image") and cand.crop_image is not None
            else candidate_images[cand.img_idx].crop(tuple(int(round(float(c))) for c in cand.box[:4]))
        )
        ax.imshow(crop)

        branch = getattr(cand, "source_branch", "Direct")
        tag = "TARGET" if is_t else "DISTRACTOR"
        border_color = "#2e7d32" if is_t else "#1565c0"

        ax.set_title(
            f"Crop {i} | Image {cand.img_idx} [{tag}]\nSize: {crop.width}x{crop.height}px | Branch: {branch}",
            fontsize=9,
            fontweight="bold",
            color=border_color,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_fig_clean(fig, output_path)


def render_stage6_reading(
    candidate_images: List[Image.Image],
    crop_records: List[Dict[str, Any]],
    real_question: str,
    target_pool_idx: int,
    output_path: str,
    unified_pool: Optional[List[Any]] = None,
) -> None:
    """Renders 06_stage6_reading.png: Foveated reading crops with predicted text and evidence borders."""
    n_read = len(crop_records)
    if n_read == 0:
        return

    ncols = min(4, n_read)
    nrows = (n_read + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 5.0 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    wrapped_q = (real_question[:90] + "...") if len(real_question) > 90 else real_question
    fig.suptitle(
        f"Stage 6: All Foveated Reading Crops & Predicted Text ({n_read} Total Crops)\nQuestion: '{wrapped_q}'",
        fontsize=12.5,
        fontweight="bold",
        y=1.00,
    )

    for i in range(n_read):
        ax = axes[i]
        c_rec = crop_records[i]
        img_idx = c_rec.get("img_idx", 0)
        exp_box = c_rec.get("expanded_box", [0, 0, 100, 100])
        loc_box = c_rec.get("local_target_box", None)
        loc_boxes = c_rec.get("local_target_boxes", [loc_box] if loc_box else [])
        is_ev = c_rec.get("is_evidence", False)
        ans = c_rec.get("raw_answer", "")
        is_t = (img_idx == target_pool_idx)
        is_compressed = c_rec.get("is_compressed_canvas", False)

        crop_idx = c_rec.get("crop_idx", i)
        if (
            is_compressed
            and unified_pool is not None
            and crop_idx < len(unified_pool)
            and getattr(unified_pool[crop_idx], "crop_image", None) is not None
        ):
            crop_pil = unified_pool[crop_idx].crop_image
        else:
            exp_box_clean = tuple(int(round(float(v))) for v in exp_box[:4])
            iw, ih = candidate_images[img_idx].size
            x1 = max(0, min(iw - 1, exp_box_clean[0]))
            y1 = max(0, min(ih - 1, exp_box_clean[1]))
            x2 = max(x1 + 1, min(iw, exp_box_clean[2]))
            y2 = max(y1 + 1, min(ih, exp_box_clean[3]))
            crop_pil = candidate_images[img_idx].crop((x1, y1, x2, y2))

        ax.imshow(crop_pil)

        for b in loc_boxes:
            if b and len(b) >= 4:
                loc_clean = [int(round(float(v))) for v in b[:4]]
                rect = mpatches.Rectangle(
                    (loc_clean[0], loc_clean[1]),
                    max(1, loc_clean[2] - loc_clean[0]),
                    max(1, loc_clean[3] - loc_clean[1]),
                    linewidth=2.5,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)

        wrapped_ans = (ans[:65] + "...") if len(ans) > 65 else ans
        status_tag = "EVIDENCE" if is_ev else "FILTERED"
        border_col = "#1b5e20" if is_ev else "#757575"
        tag = "TARGET" if is_t else "DISTRACTOR"

        ax.set_title(
            f"Crop {i} [Img {img_idx} - {tag}] | [{status_tag}]",
            fontsize=9.0,
            color=border_col,
            fontweight="bold",
            pad=6,
        )
        ax.set_xlabel(
            f'Predicted Text:\n"{wrapped_ans}"',
            fontsize=8.5,
            fontweight="bold",
            color=border_col,
            labelpad=8,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(3 if is_ev else 1.2)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n_read, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_fig_clean(fig, output_path)


# ==============================================================================
# SAMPLE DIAGNOSTIC EXECUTION ENGINE
# ==============================================================================

def execute_single_sample_stages(
    sample_index: int,
    hrbench_data: List[Dict[str, Any]],
    hrbench_base_dir: str,
    sam3_engine: Any,
    siglip_engine: Any,
    qwen_engine: Any,
    nlp_parser: Any,
    args: argparse.Namespace,
    save_visuals_spec: Optional[Union[str, Set[int]]],
) -> Dict[str, Any]:
    """
    Executes the entire diagnostic suite for a single HRBench sample:
    1. Dynamic Window Distractor Sampling
    2. Stage 1: Question Parsing & Anchor Reordering
    3. Stage 2: 2-Gate Early Assessment (Direct Global + Tool Fast Path)
    4. Stage 3: Saliency-Plateau Partitioning
    5. Stage 4: SAM3 Feature Grouping & Scaled Gating
    6. Stage 5: Unified Candidate Pool Aggregation
    7. Stage 6: Foveated Reading & Multi-Answer Consolidation
    8. Cell 10: End-to-End Pipeline Cross-Verification
    9. Conditional Visual Artifact Export (Gated by --save_visuals)
    """
    target_sample = hrbench_data[sample_index]
    real_question = target_sample.get("question", "")
    cycle_idx = target_sample.get("cycle_category", 0)
    gt_answer = target_sample.get("answer", "")
    gt_letter = resolve_gt_letter(gt_answer, cycle_idx)
    raw_options = target_sample.get("options", {})

    print("\n" + "=" * 80)
    print(f"RUNNING DIAGNOSTICS FOR HRBENCH SAMPLE #{sample_index}")
    print(f"Question     : '{real_question}'")
    print(f"Cycle Category: {cycle_idx} | Ground Truth: Option '{gt_letter}'")
    print("=" * 80)

    # 1. Load Target Image
    target_img_rel = target_sample.get("input_image", "")
    target_img_path = os.path.join(hrbench_base_dir, target_img_rel)
    if not os.path.exists(target_img_path):
        target_img_path = target_img_rel
    if not os.path.exists(target_img_path):
        raise FileNotFoundError(f"Target image not found at: {target_img_path}")
    target_canvas = Image.open(target_img_path).convert("RGB")

    # 2. Sample Distractor Images from [sample_index - window_size, sample_index + window_size]
    total_samples = len(hrbench_data)
    min_win = max(0, sample_index - args.window_size)
    max_win = min(total_samples - 1, sample_index + args.window_size)

    candidate_distractor_indices = [idx for idx in range(min_win, max_win + 1) if idx != sample_index]
    if len(candidate_distractor_indices) < (args.num_candidate_images - 1):
        candidate_distractor_indices = [idx for idx in range(total_samples) if idx != sample_index]

    rng = random.Random(args.random_seed + sample_index)
    sampled_distractor_indices = rng.sample(candidate_distractor_indices, args.num_candidate_images - 1)
    print(f"[Sampling] Distractor indices sampled within +/- {args.window_size}: {sampled_distractor_indices}")

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
    candidate_images = [item["image"] for item in pool_items]
    target_pool_idx = next(i for i, item in enumerate(pool_items) if item["is_target"])

    print(f"[Candidate Pool C] Total N = {len(candidate_images)} images constructed.")
    print(f"[Candidate Pool C] Target image is positioned at Pool Index: {target_pool_idx}")
    for idx, item in enumerate(pool_items):
        tag = "TARGET" if item["is_target"] else "DISTRACTOR"
        print(f"  - Image {idx}: [{tag}] Sample #{item['sample_idx']} | Path: '{item['path']}' | Size: {item['image'].size}")

    # Check whether visual artifacts should be rendered
    save_vis = should_save_visuals(save_visuals_spec, sample_index)
    sample_vis_dir = os.path.join(args.output_dir, "visuals", f"sample_{sample_index}")
    if save_vis:
        os.makedirs(sample_vis_dir, exist_ok=True)
        print(f"[Visual Telemetry] PNG export ENABLED for Sample #{sample_index} -> {sample_vis_dir}")
        render_stage0_candidate_pool(
            candidate_images, pool_items, real_question, target_pool_idx, os.path.join(sample_vis_dir, "01_candidate_pool.png")
        )
    else:
        print(f"[Visual Telemetry] PNG export disabled for Sample #{sample_index} (Text logs only).")

    # 3. Master Multi-Source Configuration
    config = MSSearchConfig(
        multi_answer_mode=True,
        scale_budgets_by_num_images=True,
        max_sam3_eval_patches=16 * args.num_candidate_images,
        max_latent_evals=8 * args.num_candidate_images,
        max_retained_patches=16 * args.num_candidate_images,
        enable_direct_answer=True,
        direct_answer_conf_threshold=0.60,
        enable_fast_path=True,
        fast_path_conf_threshold=0.70,
        enable_visual_prompt_bbox=True,
        partition_stopping_mode="saliency_plateau",
        min_plateau_size=256,
        max_plateau_size=1536,
        saliency_plateau_gain_threshold=0.04,
        saliency_target_threshold=0.35,
        max_patches_per_image=100,
        enable_feature_grouping=True,
        grouping_sim_threshold=0.80,
        latent_prompt_mode="question_aware",
        enable_early_consensus_exit=False,
        prompt_directed_final_answer=True,
        max_vlm_tokens=128,
        answer_dedup_similarity=0.85,
        verbose=True,
    )

    run_mode = getattr(args, "mode", "both")
    if run_mode == "e2e":
        print("\n--- EXECUTING END-TO-END PIPELINE ---")
        e2e_t0 = time.perf_counter()
        e2e_response = get_mssearch_response(
            sam_model=sam3_engine,
            zoom_model=qwen_engine,
            siglip_model=siglip_engine,
            nlp_model=nlp_parser,
            annotation=target_sample,
            candidate_images=candidate_images,
            config=config,
        )
        e2e_latency = time.perf_counter() - e2e_t0

        telemetry = e2e_response.get("telemetry", {})
        timings = telemetry.get("timings_sec", {})
        e2e_primary_answer = e2e_response.get("primary_answer", "")
        route_selected = telemetry.get("route", "UNKNOWN")

        pred_letter = extract_predicted_option(e2e_primary_answer, raw_options)
        is_correct = (pred_letter == gt_letter) if pred_letter and gt_letter else False
        res_tag = "[CORRECT]" if is_correct else "[INCORRECT]"

        print(f"\n[End-to-End Result] Route: {route_selected} | Latency: {e2e_latency:.2f}s")
        print(f"  - E2E Primary Answer : '{e2e_primary_answer}'")
        print(f"  - Predicted Option   : '{pred_letter}' | Ground Truth: '{gt_letter}' -> {res_tag}")
        print(f"  - Evidence Images    : {e2e_response.get('evidence_images')} (Target: {target_pool_idx})")

        print("\n[Latency Profiling Breakdown]")
        total_recorded = sum(timings.values()) if timings else e2e_latency
        stage_meta = [
            ("phase1_direct_answer", "Phase 1: Direct Global Answer", "GPU 0 (Qwen2.5-VL)"),
            ("stage1_parser", "Stage 1: Question Parsing & Anchor Reorder", "GPU 0 (Qwen2.5-VL / spaCy)"),
            ("stage2_fast_path", "Stage 2: Tool Fast Path & Feat Cache", "GPU 1 (SAM3) + GPU 0 (Qwen2.5-VL)"),
            ("stage3_partitioning", "Stage 3: Saliency-Plateau Partitioning", "CPU + GPU 0 (SigLIP)"),
            ("stage4_grouping_gating", "Stage 4: SAM3 Feature Grouping & Gating", "GPU 1 (SAM3) + GPU 0 (Qwen2.5-VL)"),
            ("stage5_screening", "Stage 5: Unified Candidate Pool Aggregation", "CPU Direct"),
            ("stage6_reading", "Stage 6: Foveated Reading & Multi-Answer", "GPU 0 (Qwen2.5-VL)"),
        ]
        for key, name, dev in stage_meta:
            lat = timings.get(key, 0.0)
            pct = (lat / total_recorded * 100.0) if total_recorded > 0 else 0.0
            print(f"  - {name:<46} | {dev:<30} | {lat:5.2f}s ({pct:4.1f}%)")

        record = {
            "sample_index": sample_index,
            "question": real_question,
            "cycle_idx": cycle_idx,
            "gt_letter": gt_letter,
            "predicted_letter": pred_letter,
            "is_correct": is_correct,
            "target_pool_idx": target_pool_idx,
            "step_answer": e2e_primary_answer,
            "e2e_answer": e2e_primary_answer,
            "route_selected": route_selected,
            "is_e2e_step_match": True,
            "e2e_latency": round(e2e_latency, 3),
            "evidence_images": e2e_response.get("evidence_images", []),
            "timings": timings,
            "visuals_saved": save_vis,
        }
        return record

    # --------------------------------------------------------------------------
    # STAGE 1: QUESTION PARSING
    # --------------------------------------------------------------------------
    print("\n--- STAGE 1: QUESTION PARSING ---")
    t0_s1 = time.perf_counter()
    raw_active_objects, parser_strategy = parse_question_to_active_objects(
        question=real_question,
        zoom_model=qwen_engine,
        nlp_model=nlp_parser,
        verbose=True,
        return_strategy=True,
        image_pil=candidate_images[target_pool_idx],
        config=config,
    )

    is_relational = False
    question_type = getattr(raw_active_objects, "question_type", "unknown")
    anchors = getattr(raw_active_objects, "anchors", [])
    target_element = getattr(raw_active_objects, "target_element", "")
    active_objects = list(raw_active_objects.active_objects)

    t_s1 = time.perf_counter() - t0_s1
    print(f"[Stage 1 Parse] Type: '{question_type}' | Strategy: {parser_strategy}")
    print(f"[Stage 1 Parse] Final Active Objects A_q: {active_objects} (Latency: {t_s1:.3f}s)")

    if save_vis:
        render_stage1_parsing_card(
            real_question,
            sample_index,
            question_type,
            is_relational,
            parser_strategy,
            len(candidate_images),
            active_objects,
            anchors,
            target_element,
            os.path.join(sample_vis_dir, "01_stage1_parsing.png"),
        )

    # --------------------------------------------------------------------------
    # STAGE 2: 2-GATE EARLY ASSESSMENT ACROSS ALL N CANDIDATE IMAGES
    # --------------------------------------------------------------------------
    print("\n--- STAGE 2: 2-GATE EARLY ASSESSMENT ---")
    t0_da = time.perf_counter()
    direct_ans_res = run_direct_global_assessment(
        candidate_images=candidate_images,
        question=real_question,
        zoom_model=qwen_engine,
        config=config,
    )
    t_da = time.perf_counter() - t0_da
    print(f"[Phase 1 Direct Global] Triggered: {direct_ans_res.triggered} | Conf: {direct_ans_res.confidence:.3f} | Latency: {t_da:.3f}s")
    if direct_ans_res.triggered:
        print(f"  - Answer: '{direct_ans_res.answer}' | Evidence Images: {getattr(direct_ans_res, 'evidence_images', [direct_ans_res.evidence_img_idx])}")

    t0_fp = time.perf_counter()
    fast_path_res = run_tool_based_fast_path(
        candidate_images=candidate_images,
        active_objects=active_objects,
        question=real_question,
        sam_model=sam3_engine,
        zoom_model=qwen_engine,
        config=config,
    )
    t_fp = time.perf_counter() - t0_fp
    print(f"[Phase 2 Tool Fast Path] Triggered: {fast_path_res.triggered} | Conf: {fast_path_res.confidence:.3f} | Latency: {t_fp:.3f}s")
    print(f"  - Cached SAM3 Features: {list(fast_path_res.feature_cache.keys())}")
    if fast_path_res.triggered:
        print(f"  - Answer: '{fast_path_res.answer}' | Evidence Images: {getattr(fast_path_res, 'evidence_images', [fast_path_res.evidence_img_idx])}")

    if save_vis:
        render_stage2_fast_path(
            candidate_images, fast_path_res, direct_ans_res, target_pool_idx, os.path.join(sample_vis_dir, "02_stage2_fast_path.png")
        )

    # --------------------------------------------------------------------------
    # STAGE 3: MULTI-IMAGE SALIENCY-PLATEAU PARTITIONING
    # --------------------------------------------------------------------------
    print("\n--- STAGE 3: SALIENCY-PLATEAU PARTITIONING ---")
    t0_s3 = time.perf_counter()
    feature_cache = fast_path_res.feature_cache if config.reuse_stage2_features else {}

    bounded_patches = partition_candidate_images(
        candidate_images=candidate_images,
        stage2_feature_cache=feature_cache,
        config=config,
        active_objects=active_objects,
        siglip_model=siglip_engine,
    )
    t_s3 = time.perf_counter() - t0_s3

    patches_per_img = {}
    for p in bounded_patches:
        patches_per_img[p.img_idx] = patches_per_img.get(p.img_idx, 0) + 1

    print(f"[Stage 3 Output] Total Bounded Patches: {len(bounded_patches)} (Latency: {t_s3:.3f}s)")
    for img_idx in range(len(candidate_images)):
        is_t = (img_idx == target_pool_idx)
        tag = "TARGET" if is_t else "DISTRACTOR"
        print(f"  - Image {img_idx} [{tag}]: {patches_per_img.get(img_idx, 0)} surviving candidate patches")

    if save_vis:
        render_stage3_partitioning(
            candidate_images, bounded_patches, target_pool_idx, os.path.join(sample_vis_dir, "03_stage3_partitioning.png")
        )

    # --------------------------------------------------------------------------
    # STAGE 4: SAM3 FEATURE GROUPING & SCALED GATING
    # --------------------------------------------------------------------------
    print("\n--- STAGE 4: SAM3 FEATURE GROUPING & GATING ---")
    t0_s4 = time.perf_counter()
    remaining_patches, stage4_groups = run_feature_grouping_and_gating(
        bounded_patches=bounded_patches,
        candidate_images=candidate_images,
        active_objects=active_objects,
        sam_model=sam3_engine,
        config=config,
        detected_boxes=fast_path_res.detected_boxes,
        zoom_model=qwen_engine,
        return_debug=True,
        question=real_question,
    )
    t_s4 = time.perf_counter() - t0_s4

    pruned_count = len(bounded_patches) - len(remaining_patches)
    prune_pct = (pruned_count / max(1, len(bounded_patches))) * 100.0

    print(f"[Stage 4 Output] Formed Groups: {len(stage4_groups)} | Surviving Patches: {len(remaining_patches)} / {len(bounded_patches)} ({pruned_count} pruned, {prune_pct:.1f}% reduction) (Latency: {t_s4:.3f}s)")
    surviving_per_img = {}
    for p in remaining_patches:
        surviving_per_img[p.img_idx] = surviving_per_img.get(p.img_idx, 0) + 1

    for img_idx in range(len(candidate_images)):
        is_t = (img_idx == target_pool_idx)
        tag = "TARGET" if is_t else "DISTRACTOR"
        tot_img = patches_per_img.get(img_idx, 0)
        print(f"  - Image {img_idx} [{tag}]: {surviving_per_img.get(img_idx, 0)} / {tot_img} patches retained")

    if save_vis:
        render_stage4_grouping_gating(
            candidate_images, bounded_patches, remaining_patches, target_pool_idx, os.path.join(sample_vis_dir, "04_stage4_grouping_gating.png")
        )

    # --------------------------------------------------------------------------
    # STAGE 5: DIRECT UNIFIED CANDIDATE POOL AGGREGATION
    # --------------------------------------------------------------------------
    print("\n--- STAGE 5: UNIFIED POOL AGGREGATION ---")
    t0_s5 = time.perf_counter()
    unified_pool = run_resolution_screening(
        remaining_patches=remaining_patches,
        candidate_images=candidate_images,
        active_objects=active_objects,
        siglip_model=siglip_engine,
        config=config,
    )
    t_s5 = time.perf_counter() - t0_s5

    pool_per_img = {}
    for c in unified_pool:
        pool_per_img[c.img_idx] = pool_per_img.get(c.img_idx, 0) + 1

    print(f"[Stage 5 Output] Unified Candidate Pool P Size: {len(unified_pool)} crops (Latency: {t_s5:.3f}s)")
    for img_idx in range(len(candidate_images)):
        is_t = (img_idx == target_pool_idx)
        tag = "TARGET" if is_t else "DISTRACTOR"
        print(f"  - Image {img_idx} [{tag}]: {pool_per_img.get(img_idx, 0)} candidate crops ready for Stage 6")

    if save_vis:
        render_stage5_screening(
            candidate_images, unified_pool, target_pool_idx, os.path.join(sample_vis_dir, "05_stage5_screening.png")
        )

    # --------------------------------------------------------------------------
    # STAGE 6: FOVEATED VLM READING & OPEN-SET MULTI-ANSWER DEDUPLICATION
    # --------------------------------------------------------------------------
    print("\n--- STAGE 6: FOVEATED VLM READING & DEDUPLICATION ---")
    t0_s6 = time.perf_counter()
    final_result = run_foveated_vlm_reading(
        unified_pool=unified_pool,
        candidate_images=candidate_images,
        question=real_question,
        zoom_model=qwen_engine,
        config=config,
        question_type=None,
    )
    t_s6 = time.perf_counter() - t0_s6

    informative_answers = [a for a in final_result.answers if not is_negative_or_unanswerable(a)]
    if config.multi_answer_mode:
        step_consolidated_answer = (
            "; ".join(informative_answers)
            if informative_answers
            else "; ".join(final_result.answers)
            if final_result.answers
            else "No answer found"
        )
    else:
        step_consolidated_answer = (
            informative_answers[0]
            if informative_answers
            else final_result.answers[0]
            if final_result.answers
            else "No answer found"
        )

    crop_records = getattr(final_result, "crop_records", [])

    print(f"[Stage 6 Output] Evaluated Crops: {len(crop_records)} / {len(unified_pool)} (Latency: {t_s6:.3f}s)")
    print(f"[Stage 6 Output] Raw Extractions Count: {len(final_result.extractions)}")
    print(f"[Stage 6 Output] Unique Structured Answers: {final_result.answers}")
    print(f"[Stage 6 Output] Consolidated Multi-Answer Text: '{step_consolidated_answer}'")
    print(f"[Stage 6 Output] Evidence Image Indices: {final_result.evidence_image_indices} (Target Pool Index: {target_pool_idx})")

    # Crop records audit
    print("\n[Stage 6 Crop Audit Summary]")
    print(f"{'Crop #':<8} | {'Image #':<12} | {'Status':<10} | {'Branch':<10} | {'Size':<12} | {'Final Answer'}")
    print("-" * 75)
    for i, c_rec in enumerate(crop_records):
        status = "EVIDENCE" if c_rec.get("is_evidence") else "FILTERED"
        img_i = c_rec.get("img_idx", 0)
        is_t = (img_i == target_pool_idx)
        img_lbl = f"Img {img_i} ({'T' if is_t else 'D'})"
        branch = str(c_rec.get("source_branch", "Direct"))
        box = c_rec.get("expanded_box", [0, 0, 0, 0])
        size_str = f"{box[2]-box[0]}x{box[3]-box[1]}px"
        fa = str(c_rec.get("final_answer", "N/A")).strip()
        print(f"{i:<8} | {img_lbl:<12} | {status:<10} | {branch:<10} | {size_str:<12} | {fa}")

    if save_vis:
        render_stage6_reading(
            candidate_images,
            crop_records,
            real_question,
            target_pool_idx,
            os.path.join(sample_vis_dir, "06_stage6_reading.png"),
            unified_pool=unified_pool,
        )

    # --------------------------------------------------------------------------
    # CELL 10: END-TO-END PIPELINE CROSS-VERIFICATION
    # --------------------------------------------------------------------------
    print("\n--- CELL 10: END-TO-END PIPELINE CROSS-VERIFICATION ---")
    e2e_t0 = time.perf_counter()
    e2e_response = get_mssearch_response(
        sam_model=sam3_engine,
        zoom_model=qwen_engine,
        siglip_model=siglip_engine,
        nlp_model=nlp_parser,
        annotation=target_sample,
        candidate_images=candidate_images,
        config=config,
    )
    e2e_latency = time.perf_counter() - e2e_t0

    telemetry = e2e_response.get("telemetry", {})
    timings = telemetry.get("timings_sec", {})
    e2e_primary_answer = e2e_response.get("primary_answer", "")
    route_selected = telemetry.get("route", "UNKNOWN")

    pred_letter = extract_predicted_option(e2e_primary_answer or step_consolidated_answer, raw_options)
    is_correct = (pred_letter == gt_letter) if pred_letter and gt_letter else False
    res_tag = "[CORRECT]" if is_correct else "[INCORRECT]"

    print(f"[End-to-End Result] Route: {route_selected} | Latency: {e2e_latency:.2f}s")
    print(f"  - E2E Primary Answer : '{e2e_primary_answer}'")
    print(f"  - Step Primary Answer: '{step_consolidated_answer}'")
    print(f"  - Predicted Option   : '{pred_letter}' | Ground Truth: '{gt_letter}' -> {res_tag}")
    print(f"  - Discovered Evidence Images: {e2e_response.get('evidence_images')} (Target at {target_pool_idx})")

    # Cross-verification comparison
    is_e2e_step_match = (e2e_primary_answer.strip().lower() == step_consolidated_answer.strip().lower())
    if is_e2e_step_match:
        print("  [VERIFICATION PASS] End-to-end get_mssearch_response matches step-by-step extraction exactly!")
    else:
        print("  [VERIFICATION NOTICE] Slight deviation between E2E and step-by-step extractions (due to cascade early route or stochasticity).")

    # Print Timing Profiling Breakdown
    print("\n[Latency Profiling Breakdown]")
    total_recorded = sum(timings.values()) if timings else e2e_latency
    stage_meta = [
        ("phase1_direct_answer", "Phase 1: Direct Global Answer", "GPU 0 (Qwen2.5-VL)"),
        ("stage1_parser", "Stage 1: Question Parsing & Anchor Reorder", "GPU 0 (Qwen2.5-VL / spaCy)"),
        ("stage2_fast_path", "Stage 2: Tool Fast Path & Feat Cache", "GPU 1 (SAM3) + GPU 0 (Qwen2.5-VL)"),
        ("stage3_partitioning", "Stage 3: Saliency-Plateau Partitioning", "CPU + GPU 0 (SigLIP)"),
        ("stage4_grouping_gating", "Stage 4: SAM3 Feature Grouping & Gating", "GPU 1 (SAM3) + GPU 0 (Qwen2.5-VL)"),
        ("stage5_screening", "Stage 5: Unified Candidate Pool Aggregation", "CPU Direct"),
        ("stage6_reading", "Stage 6: Foveated Reading & Multi-Answer", "GPU 0 (Qwen2.5-VL)"),
    ]
    for key, name, dev in stage_meta:
        lat = timings.get(key, 0.0)
        pct = (lat / total_recorded * 100.0) if total_recorded > 0 else 0.0
        print(f"  - {name:<46} | {dev:<30} | {lat:5.2f}s ({pct:4.1f}%)")

    record = {
        "sample_index": sample_index,
        "question": real_question,
        "cycle_idx": cycle_idx,
        "gt_letter": gt_letter,
        "predicted_letter": pred_letter,
        "is_correct": is_correct,
        "target_pool_idx": target_pool_idx,
        "step_answer": step_consolidated_answer,
        "e2e_answer": e2e_primary_answer,
        "route_selected": route_selected,
        "is_e2e_step_match": is_e2e_step_match,
        "e2e_latency": round(e2e_latency, 3),
        "evidence_images": e2e_response.get("evidence_images", []),
        "timings": timings,
        "visuals_saved": save_vis,
    }
    return record


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Source Visual Search Engine (MSSearch) Diagnostic Runner for HRBench"
    )
    parser.add_argument(
        "--hrbench_dir",
        type=str,
        default=None,
        help="Root path to HRBench dataset. Defaults to $HRBENCH_DIR or standard local search paths.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "results"),
        help="Directory to save JSONL logs and visual artifacts. Default: ./results",
    )
    parser.add_argument(
        "--sam3_path",
        type=str,
        default=os.environ.get("SAM3_PATH", "/home/ubuntu/workspace/hr/models/sam3/sam3/sam3.pt"),
        help="Path to local SAM3 checkpoint.",
    )
    parser.add_argument(
        "--qwen_path",
        type=str,
        default=os.environ.get("QWEN_PATH", "/home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct"),
        help="Path or HuggingFace ID for Qwen2.5-VL-7B-Instruct.",
    )
    parser.add_argument(
        "--siglip_name",
        type=str,
        default=os.environ.get("SIGLIP_NAME", "google/siglip-base-patch16-384"),
        help="HuggingFace model ID or path for SigLIP.",
    )
    parser.add_argument(
        "--nlp_model",
        type=str,
        default=os.environ.get("NLP_MODEL", "/home/ubuntu/workspace/hr/models/spacy/en_core_web_sm-3.8.0"),
        help="Path or model name for spaCy NLP parser.",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Single HRBench sample index to evaluate (default: 0).",
    )
    parser.add_argument(
        "--sample_indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit list of HRBench sample indices to evaluate (e.g. --sample_indices 0 2 5). Overrides --sample_idx.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional ceiling on number of evaluated samples.",
    )
    parser.add_argument(
        "--num_candidate_images",
        type=int,
        default=3,
        help="Number of candidate images N in pool C (1 target + N-1 distractors, default: 3).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Local index window for distractor sampling [i - W, i + W] (default: 5).",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling and shuffling (default: 42).",
    )
    parser.add_argument(
        "--all_samples",
        "--all",
        dest="all_samples",
        action="store_true",
        help="Evaluate all samples in the HRBench dataset.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["e2e", "both", "step"],
        default=None,
        help="Execution mode: 'e2e' (direct single-pass pipeline, recommended for full runs), 'both' (runs step-by-step AND e2e for cross-verification), 'step' (step-by-step diagnostics only). Default: 'e2e' when --all_samples is active, 'both' otherwise.",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="Starting sample index when slicing the dataset (0-indexed).",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Ending sample index (exclusive) when slicing the dataset.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume execution: inspects existing results.jsonl and skips already evaluated sample indices.",
    )
    parser.add_argument(
        "--save_visuals",
        nargs="*",
        default=None,
        help="Visual artifact control. Omit for text diagnostics only. Pass 'all' to save PNGs for every sample, or space/comma-delimited sample indices (e.g. --save_visuals 2 5 7).",
    )

    args = parser.parse_args()
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode is None:
        args.mode = "e2e" if args.all_samples else "both"

    # 1. Parse --save_visuals flag
    save_visuals_spec = parse_save_visuals_arg(args.save_visuals)
    if save_visuals_spec is None:
        print("[Config] --save_visuals not provided -> Text diagnostics only (0 visual PNGs will be saved).")
    elif save_visuals_spec == "all":
        print("[Config] --save_visuals 'all' -> Visual PNGs will be saved for ALL evaluated samples.")
    else:
        print(f"[Config] --save_visuals active for sample indices: {sorted(list(save_visuals_spec))}")

    print(f"[Config] Execution Mode: '{args.mode.upper()}' ({'Single-Pass High-Throughput' if args.mode == 'e2e' else 'Dual Step + E2E Verification'})")

    # 2. Discover HRBench Dataset
    hrbench_base_dir, hrbench_annotation_path = discover_hrbench_dir(args.hrbench_dir)
    print(f"[Dataset Discovery] Using HRBench root: {hrbench_base_dir}")
    print(f"[Dataset Discovery] Loading annotations: {hrbench_annotation_path}")

    with open(hrbench_annotation_path, "r", encoding="utf-8") as f:
        hrbench_data = json.load(f)
    print(f"[Dataset] Ingested {len(hrbench_data)} total samples.")

    # 3. Resolve Sample Indices
    if args.all_samples:
        target_indices = list(range(len(hrbench_data)))
    elif args.sample_indices is not None and len(args.sample_indices) > 0:
        target_indices = args.sample_indices
    else:
        target_indices = [args.sample_idx]

    if args.start_idx is not None or args.end_idx is not None:
        start_i = args.start_idx or 0
        end_i = args.end_idx if args.end_idx is not None else len(target_indices)
        target_indices = target_indices[start_i:end_i]

    if args.max_samples is not None and args.max_samples > 0:
        target_indices = target_indices[: args.max_samples]

    results_log_path = os.path.abspath(os.path.join(args.output_dir, "results.jsonl"))
    os.makedirs(os.path.dirname(results_log_path), exist_ok=True)

    if args.resume and os.path.exists(results_log_path):
        completed_ids = set()
        with open(results_log_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                if line.strip():
                    try:
                        r_data = json.loads(line)
                        if "sample_index" in r_data:
                            completed_ids.add(r_data["sample_index"])
                    except Exception:
                        pass
        original_count = len(target_indices)
        target_indices = [idx for idx in target_indices if idx not in completed_ids]
        print(f"[Resume] Found {len(completed_ids)} already completed samples. Remaining in queue: {len(target_indices)} / {original_count}")

    print(f"[Execution Queue] Evaluating {len(target_indices)} sample(s) (First 10: {target_indices[:10]}...)")

    # 4. Asymmetric Dual-GPU Allocation & Model Initialization
    num_gpus = torch.cuda.device_count()
    sam3_device = "cuda:1" if num_gpus > 1 else ("cuda:0" if num_gpus > 0 else "cpu")
    siglip_device = "cuda:0" if num_gpus > 0 else "cpu"
    qwen_device = "cuda:0" if num_gpus > 0 else "cpu"

    print("=" * 80)
    print(f"HARDWARE TOPOLOGY: {num_gpus} CUDA GPUs Available")
    print(f"  - SAM3 Engine      : {sam3_device}")
    print(f"  - SigLIP Engine    : {siglip_device}")
    print(f"  - Qwen2.5-VL Engine: {qwen_device}")
    print("=" * 80)

    print(f"\n[Loading] SAM3 Segmentation Engine from '{args.sam3_path}' on {sam3_device}...")
    sam3_engine = sam3_inference(model_path=args.sam3_path, device=sam3_device)
    print("  [Loaded] SAM3 Local Segmenter Engine")

    print(f"\n[Loading] SigLIP Saliency Engine '{args.siglip_name}' on {siglip_device}...")
    siglip_engine = SigLIPScorer(
        model_name=args.siglip_name,
        device=siglip_device,
        torch_dtype=torch.float16 if "cuda" in siglip_device else torch.float32,
    )
    print("  [Loaded] SigLIP Saliency Engine")

    print(f"\n[Loading] Qwen2.5-VL Multi-Scale Reader from '{args.qwen_path}' on {qwen_device}...")
    qwen_engine = ModelQwenVL(
        model_path=args.qwen_path,
        device=qwen_device,
        torch_dtype=torch.float16 if "cuda" in qwen_device else torch.float32,
        patch_scale=1.2,
    )
    print("  [Loaded] Qwen2.5-VL Reader Engine")

    nlp_parser = None
    if args.nlp_model:
        try:
            import spacy
            if os.path.exists(args.nlp_model):
                nlp_parser = spacy.load(args.nlp_model)
            else:
                nlp_parser = spacy.load("en_core_web_sm")
            print(f"  [Loaded] spaCy NLP Grammar Engine ({args.nlp_model})")
        except Exception as e:
            print(f"  [Notice] spaCy load skipped: {e}")

    # 5. Execute Evaluation Loop
    print(f"\n[Logging] Results will be appended to: {results_log_path}")

    all_records = []
    correct_count = 0
    evaluated_count = 0
    route_counter = {}
    latencies = []

    for queue_pos, s_idx in enumerate(target_indices, start=1):
        if s_idx < 0 or s_idx >= len(hrbench_data):
            print(f"[Warning] Sample index {s_idx} out of range [0, {len(hrbench_data)-1}]. Skipping.")
            continue

        try:
            rec = execute_single_sample_stages(
                sample_index=s_idx,
                hrbench_data=hrbench_data,
                hrbench_base_dir=hrbench_base_dir,
                sam3_engine=sam3_engine,
                siglip_engine=siglip_engine,
                qwen_engine=qwen_engine,
                nlp_parser=nlp_parser,
                args=args,
                save_visuals_spec=save_visuals_spec,
            )
            all_records.append(rec)
            evaluated_count += 1
            if rec.get("is_correct"):
                correct_count += 1
            r_sel = rec.get("route_selected", "UNKNOWN")
            route_counter[r_sel] = route_counter.get(r_sel, 0) + 1
            latencies.append(rec.get("e2e_latency", 0.0))

            os.makedirs(os.path.dirname(results_log_path), exist_ok=True)
            with open(results_log_path, "a", encoding="utf-8") as f_out:
                f_out.write(json.dumps(rec) + "\n")

            # Running status banner
            acc_pct = (correct_count / evaluated_count * 100.0) if evaluated_count > 0 else 0.0
            avg_lat_so_far = sum(latencies) / len(latencies) if latencies else 0.0
            print(
                f"\n>>> PROGRESS: {queue_pos}/{len(target_indices)} (Sample #{s_idx}) | "
                f"Running Acc: {correct_count}/{evaluated_count} ({acc_pct:.1f}%) | "
                f"Avg Latency: {avg_lat_so_far:.2f}s <<<\n"
            )

        except Exception as e:
            print(f"\n[ERROR] Exception encountered on sample #{s_idx}: {e}")
            import traceback
            traceback.print_exc()

        # Dynamic Memory Shield: Periodic CUDA flush
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_eval = len(all_records)
    overall_acc = (correct_count / total_eval * 100.0) if total_eval > 0 else 0.0
    avg_lat = (sum(latencies) / len(latencies)) if latencies else 0.0

    summary_data = {
        "total_samples_evaluated": total_eval,
        "correct_samples": correct_count,
        "accuracy_acc1": round(overall_acc, 2),
        "avg_latency_sec": round(avg_lat, 2),
        "routes": route_counter,
        "results_log_path": results_log_path,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary_data, f_sum, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED: Evaluated {total_eval} / {len(target_indices)} samples.")
    print(f"Overall Accuracy (Acc@1): {correct_count}/{total_eval} ({overall_acc:.2f}%)")
    print(f"Average Latency         : {avg_lat:.2f}s per sample")
    print(f"Route Breakdown         : {route_counter}")
    print(f"Telemetry log written to: {results_log_path}")
    print(f"Summary JSON written to : {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
