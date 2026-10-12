"""
Stage 6: Foveated VLM Reading with 1.25x BBox Context Expansion & Answer Deduplication.
Expands each candidate patch independently by 25% margin (no destructive bbox merging),
queries Qwen2.5-VL exhaustively across all evidence patches,
and deduplicates extracted target answers via semantic similarity into Answer Memory M.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from difflib import SequenceMatcher
from PIL import Image, ImageDraw

from ..config import MSSearchConfig
from .stage5_screening import CandidateCrop
from .stage2_fast_path import render_visual_prompt_bbox


@dataclass
class EvidenceExtraction:
    """Records evidence found in a single candidate crop."""
    img_idx: int
    box: Tuple[int, int, int, int]
    expanded_box: Tuple[int, int, int, int]
    raw_answer: str
    source_branch: str


@dataclass
class FinalSearchResult:
    """Final output container for the Dual-Path Visual Search Engine."""
    answers: List[str]
    evidence_image_indices: List[int]
    extractions: List[EvidenceExtraction] = field(default_factory=list)


def expand_bounding_box(
    box: Any,
    img_w: int,
    img_h: int,
    expansion_factor: float = 1.25
) -> Tuple[int, int, int, int]:
    """
    Independently expands a bounding box by expansion_factor (e.g. 1.25x = 25% margin)
    clamped to image boundary, guaranteed pure Python ints.
    """
    if hasattr(box, "tolist"):
        box = box.tolist()
    x1, y1, x2, y2 = [int(round(float(c))) for c in box]
    w = x2 - x1
    h = y2 - y1

    delta_w = int(round((expansion_factor - 1.0) / 2.0 * w))
    delta_h = int(round((expansion_factor - 1.0) / 2.0 * h))

    ex1 = max(0, x1 - delta_w)
    ey1 = max(0, y1 - delta_h)
    ex2 = min(img_w, x2 + delta_w)
    ey2 = min(img_h, y2 + delta_h)

    return (int(ex1), int(ey1), int(ex2), int(ey2))



def is_negative_or_unanswerable(text: str) -> bool:
    """Detects whether VLM output indicates no answer / absence of target."""
    cleaned = text.strip().lower()
    negative_phrases = [
        "cannot answer", "not visible", "not clearly visible",
        "no information", "cannot determine", "not enough information",
        "unable to see", "none", "not present", "n/a", "cannot be seen"
    ]
    return any(p in cleaned for p in negative_phrases) or len(cleaned) < 2


def deduplicate_answers(answers: List[str], similarity_threshold: float = 0.85, verbose: bool = False) -> List[str]:
    """
    Deduplicates answer strings via Levenshtein / SequenceMatcher semantic similarity.
    Keeps the most descriptive, higher-entropy answer among near-duplicates.
    """
    unique_answers: List[str] = []

    for ans in answers:
        norm_ans = ans.strip()
        if not norm_ans or is_negative_or_unanswerable(norm_ans):
            continue

        is_duplicate = False
        for idx, existing in enumerate(unique_answers):
            ratio = SequenceMatcher(None, norm_ans.lower(), existing.lower()).ratio()
            if ratio >= similarity_threshold:
                is_duplicate = True
                if verbose:
                    print(f"[Stage 6][Deduplication] Near-duplicate detected (sim={ratio:.2f}): '{norm_ans}' vs '{existing}'. Merging.")
                # Keep whichever string is more descriptive
                if len(norm_ans) > len(existing):
                    unique_answers[idx] = norm_ans
                break

        if not is_duplicate:
            unique_answers.append(norm_ans)

    return unique_answers


def run_foveated_vlm_reading(
    unified_pool: List[CandidateCrop],
    candidate_images: List[Image.Image],
    question: str,
    zoom_model,
    config: MSSearchConfig
) -> FinalSearchResult:
    """
    Executes Foveated VLM Reading with 1.25x padding and builds Answer Memory M.
    Exhaustively searches all candidate crops (zero early stopping) to support hidden K.
    """
    extractions: List[EvidenceExtraction] = []
    evidence_indices: set = set()

    if config.verbose:
        print(f"[Stage 6] Commencing exhaustive foveated reading across {len(unified_pool)} candidate crops (1.25x bbox expansion)...")

    for c_idx, item in enumerate(unified_pool):
        img_idx = item.img_idx
        img_pil = candidate_images[img_idx]
        img_w, img_h = img_pil.size

        # 1. Expand bounding box independently by 1.25x
        exp_box = expand_bounding_box(
            item.box, img_w, img_h, expansion_factor=config.bbox_expansion_factor
        )

        foveated_crop = img_pil.crop(exp_box)
        fcw, fch = foveated_crop.size
        if fcw < 8 or fch < 8:
            continue

        # 2. Apply Visual and Textual Prompting if configured
        vlm_input_crop = foveated_crop
        x1, y1 = exp_box[0], exp_box[1]
        local_target_box = (
            item.box[0] - x1, item.box[1] - y1,
            item.box[2] - x1, item.box[3] - y1
        )

        if config.enable_visual_prompt_bbox:
            vlm_input_crop = render_visual_prompt_bbox(
                foveated_crop,
                [local_target_box],
                color=config.visual_prompt_bbox_color,
                width=config.visual_prompt_bbox_width
            )

        text_coords_hint = ""
        if config.enable_textual_prompt_coords:
            text_coords_hint = f" Focal region: [{local_target_box[0]}, {local_target_box[1]}, {local_target_box[2]}, {local_target_box[3]}]."

        # 3. Query VLM with full question q
        prompt = f"{question}{text_coords_hint}"
        raw_answer = ""
        if zoom_model is not None:
            try:
                if hasattr(zoom_model, "generate_response"):
                    raw_answer = zoom_model.generate_response(image_pil=vlm_input_crop, question=prompt)
                elif hasattr(zoom_model, "free_form_using_nodes"):
                    raw_answer = zoom_model.free_form_using_nodes(image_pil=vlm_input_crop, question=prompt, searched_nodes=[])
                elif hasattr(zoom_model, "generate"):
                    raw_answer = zoom_model.generate(image_pil=vlm_input_crop, question=prompt)
                else:
                    raw_answer = str(zoom_model(image_pil=vlm_input_crop, question=prompt))
            except Exception as e:
                if config.verbose:
                    print(f"[Stage 6][Error Fallback] Crop {c_idx} (Image {img_idx}) VLM generation failed ({e}); skipping crop.")
                continue
        else:
            if config.verbose:
                print(f"[Stage 6][Fallback] Zoom VLM model absent; cannot read crop {c_idx}.")
            continue

        if not is_negative_or_unanswerable(raw_answer):
            extractions.append(
                EvidenceExtraction(
                    img_idx=img_idx,
                    box=item.box,
                    expanded_box=exp_box,
                    raw_answer=raw_answer,
                    source_branch=item.source_branch
                )
            )
            evidence_indices.add(img_idx)
            if config.verbose:
                print(f"[Stage 6][Evidence Found] Crop {c_idx} (Image {img_idx}, {item.source_branch}): '{raw_answer}'")
        elif config.verbose:
            print(f"[Stage 6][Negative Filtered] Crop {c_idx} (Image {img_idx}) returned unanswerable response: '{raw_answer}'")

    # 4. Semantic Deduplication of Answer Memory
    raw_ans_list = [e.raw_answer for e in extractions]
    final_dedup_answers = deduplicate_answers(
        raw_ans_list,
        similarity_threshold=config.answer_dedup_similarity,
        verbose=config.verbose
    )

    if config.verbose:
        print(f"[Stage 6] Reading complete: Extracted {len(extractions)} evidence items, deduplicated to {len(final_dedup_answers)} unique answers: {final_dedup_answers} (Evidence images: {sorted(list(evidence_indices))}).")

    return FinalSearchResult(
        answers=final_dedup_answers,
        evidence_image_indices=sorted(list(evidence_indices)),
        extractions=extractions
    )

