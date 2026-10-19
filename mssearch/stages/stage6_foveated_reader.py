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
    final_answer: str = ""


import re

def extract_final_answer(text: str) -> str:
    """
    Extracts the clean, concise final answer from a model response concluding with:
    'Final Answer: <concise answer>' or inside '<answer>...</answer>' tags.
    Falls back gracefully to the raw response if no structured demarcation is found.
    """
    if not text:
        return ""
    # Match 'Final Answer:', '**Final Answer:**', or 'Final Answer is:'
    m = re.search(r'(?:\*\*|__)?Final Answer(?:\*\*|__)?\s*(?:is|:)?\s*([^\n\r]+)', text, flags=re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        val = re.sub(r'^[\*\[\"\'\s]+|[\*\]\"\'\.\s]+$', '', val).strip()
        if val:
            return val
    # Fallback to <answer> tags if present
    m = re.search(r'<answer>([\s\S]*?)</answer>', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()

@dataclass
class FinalSearchResult:
    """Final output container for the Dual-Path Visual Search Engine."""
    answers: List[str]
    evidence_image_indices: List[int]
    extractions: List[EvidenceExtraction] = field(default_factory=list)
    crop_records: List[Dict[str, Any]] = field(default_factory=list)


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
    """
    Detects whether VLM output indicates absence / refusal / non-visibility of target object,
    e.g. 'none', 'nothing', 'no person', 'no mailbox', 'The image does not show a mailbox', 'There is no mailbox visible', etc.
    Preserves valid answers like numbers, single characters, and road/store signs ('No Entry', 'No Parking').
    """
    cleaned = text.strip().rstrip(".").rstrip(",")
    if not cleaned:
        return True
    cleaned_lower = cleaned.lower()

    # 1. Standalone negative words / short phrases
    standalone_refusals = {
        "none", "nothing", "null", "unknown", "na", "n/a", "nil", "empty",
        "no answer", "no answer found", "unanswerable", "cannot answer",
        "not found", "not visible", "no text", "no translation needed",
        "no visible text", "no target", "no object", "none visible",
        "not applicable", "no information", "not enough information",
        "not present", "not clearly visible", "not depicted", "unclear"
    }
    if cleaned_lower in standalone_refusals:
        return True

    # 2. Short absence phrases: 'no <object>' (e.g. 'no mailbox', 'no person', 'no box, no clock')
    # Negative lookahead protects road/store sign text like 'no entry', 'no parking', 'no turn'
    if re.match(r"^no\s+(?!(?:entry|parking|turn|stopping|standing|trespassing|smoking|littering)\b)[a-z0-9_,\s]+$", cleaned_lower):
        return True

    # 3. Contextual absence and refusal regex patterns
    negative_patterns = [
        r"does not (?:show|contain|depict|feature|display|include)\b",
        r"doesn't (?:show|contain|depict|feature|display|include)\b",
        r"did not (?:show|contain|depict|feature|display|include)\b",
        r"didn't (?:show|contain|depict|feature|display|include)\b",
        r"there is no\b",
        r"there are no\b",
        r"there's no\b",
        r"no (?:\w+\s+){0,5}(?:is |are )?visible\b",
        r"no visible\b",
        r"not visible\b",
        r"not clearly visible\b",
        r"cannot be seen\b",
        r"can't be seen\b",
        r"cannot see\b",
        r"can't see\b",
        r"unable to see\b",
        r"cannot find\b",
        r"can't find\b",
        r"unable to find\b",
        r"cannot be found\b",
        r"cannot answer\b",
        r"can't answer\b",
        r"unable to answer\b",
        r"cannot determine\b",
        r"can't determine\b",
        r"unable to determine\b",
        r"cannot be determined\b",
        r"no information\b",
        r"not enough information\b",
        r"not present\b",
        r"is absent\b",
        r"are absent\b",
        r"no indication of\b",
        r"no sign of\b",
        r"not depicted\b",
        r"not in the (?:image|frame|crop|picture|photo)\b",
        r"outside the (?:image|frame|crop|picture|photo)\b",
        r"not visible in the (?:image|frame|crop|picture|photo)\b",
        r"not visible in this (?:image|frame|crop|picture|photo)\b",
        r"image (?:you provided )?does not show\b",
        r"image (?:you provided )?does not contain\b",
        r"image does not show\b",
        r"image does not contain\b",
        r"\bnone visible\b",
        r"\bnot found\b",
        r"no (?:\w+\s+){0,5}(?:found|detected|seen|present)\b",
        r"no (?:\w+\s+){0,5}(?:or \w+\s+){0,3}present\b",
    ]

    for pat in negative_patterns:
        if re.search(pat, cleaned_lower):
            return True

    return False


def deduplicate_answers(answers: List[str], similarity_threshold: float = 0.85, verbose: bool = False) -> List[str]:
    """
    Deduplicates answer strings via Levenshtein / SequenceMatcher semantic similarity.
    Keeps the most descriptive, higher-entropy answer among near-duplicates,
    and returns unique answers sorted by support count (descending consensus).
    """
    valid_answers = [ans.strip() for ans in answers if ans.strip() and not is_negative_or_unanswerable(ans)]
    if not valid_answers:
        return []

    unique_clusters: List[Dict[str, Any]] = []

    for ans in valid_answers:
        matched = False
        for cluster in unique_clusters:
            rep = cluster["rep"]
            ratio = SequenceMatcher(None, ans.lower(), rep.lower()).ratio()
            if ratio >= similarity_threshold:
                matched = True
                cluster["count"] += 1
                if verbose:
                    print(f"[Stage 6][Deduplication] Near-duplicate detected (sim={ratio:.2f}): '{ans}' vs '{rep}'. Merging.")
                # Keep whichever string is more descriptive
                if len(ans) > len(rep):
                    cluster["rep"] = ans
                break

        if not matched:
            unique_clusters.append({"rep": ans, "count": 1})

    # Rank clusters: highest consensus support count first, then by length
    unique_clusters.sort(key=lambda c: (c["count"], len(c["rep"])), reverse=True)
    return [c["rep"] for c in unique_clusters]


def run_foveated_vlm_reading(
    unified_pool: List[CandidateCrop],
    candidate_images: List[Image.Image],
    question: str,
    zoom_model,
    config: MSSearchConfig,
    is_relational: Optional[bool] = None,
    question_type: Optional[str] = None,
    **kwargs
) -> FinalSearchResult:
    """
    Executes Foveated VLM Reading with 1.25x padding and builds Answer Memory M.
    Exhaustively searches all candidate crops (zero early stopping) to support hidden K.
    """
    if zoom_model is None:
        raise RuntimeError(
            "[Stage 6 Error] zoom_model (Qwen2.5-VL) is None. Stage 6 Foveated Reading requires a loaded VLM model; fallback and dry-run execution are strictly disabled."
        )

    extractions: List[EvidenceExtraction] = []
    evidence_indices: set = set()
    crop_records: List[Dict[str, Any]] = []

    if config.verbose:
        print(f"[Stage 6] Commencing exhaustive foveated reading across {len(unified_pool)} candidate crops (1.25x bbox expansion)...")

    # Determine relational intent if not explicitly passed
    if is_relational is None:
        if getattr(config, "enable_query_adaptive_dual_scale", True):
            from .stage1_parser import detect_relational_intent
            is_relational = detect_relational_intent(question)
        else:
            is_relational = False

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

        # Query-Adaptive Dual-Scale Context:
        # If question queries spatial relations or orientation, provide [Global with BBox, Local Zoomed Crop]
        vlm_image_input: Union[Image.Image, List[Image.Image]] = vlm_input_crop
        if is_relational:
            # Annotate focus box on global image
            global_annotated = render_visual_prompt_bbox(
                img_pil,
                [item.box],
                color=config.visual_prompt_bbox_color,
                width=max(2, img_w // 200)
            )
            # Downsample global image if excessively large to protect token budget
            max_global_dim = 768
            if img_w > max_global_dim or img_h > max_global_dim:
                scale = max_global_dim / max(img_w, img_h)
                global_annotated = global_annotated.resize(
                    (int(round(img_w * scale)), int(round(img_h * scale))),
                    Image.Resampling.BILINEAR
                )
            vlm_image_input = [global_annotated, vlm_input_crop]
            if config.verbose and c_idx == 0:
                print(f"[Stage 6][Dual-Scale] Spatial/relational query detected; using Dual-Scale [Global with BBox, Local Crop] mode.")

        # 3. Query VLM with prompt-directed Final Answer directive & negative escape grounding
        if getattr(config, "prompt_directed_final_answer", True):
            if is_relational:
                prompt = (
                    f"{question}{text_coords_hint}\n\n"
                    "Carefully inspect the zoomed crop and the red focus rectangle on the main image.\n"
                    "If the target object or requested text/information is NOT clearly visible in this crop, output:\n"
                    "Final Answer: None\n\n"
                    "Otherwise, answer concisely based strictly on what is visible in the focal region, concluding with:\n"
                    "Final Answer: <concise answer>"
                )
            else:
                prompt = (
                    f"{question}{text_coords_hint}\n\n"
                    "Carefully inspect this crop. If the target object or requested text/information is NOT clearly visible in this crop, output:\n"
                    "Final Answer: None\n\n"
                    "Otherwise, answer concisely based strictly on what is visible in this crop, concluding with:\n"
                    "Final Answer: <concise answer>"
                )
        else:
            prompt = (
                f"{question}{text_coords_hint}\n\n"
                "If the target object or requested text/information is NOT clearly visible in this crop, output: Final Answer: None"
            )

        raw_answer = ""
        max_tokens = getattr(config, "max_vlm_tokens", 128)
        try:
            if hasattr(zoom_model, "generate_response"):
                raw_answer = zoom_model.generate_response(image_pil=vlm_image_input, question=prompt, max_new_tokens=max_tokens)
            elif hasattr(zoom_model, "free_form_using_nodes"):
                raw_answer = zoom_model.free_form_using_nodes(image_pil=vlm_image_input, question=prompt, searched_nodes=[], max_new_tokens=max_tokens)
            elif hasattr(zoom_model, "generate"):
                raw_answer = zoom_model.generate(image_pil=vlm_image_input, question=prompt, max_new_tokens=max_tokens)
            else:
                raw_answer = str(zoom_model(image_pil=vlm_image_input, question=prompt))
        except Exception as e:
            if config.verbose:
                print(f"[Stage 6][Error Fallback] Crop {c_idx} (Image {img_idx}) VLM generation failed ({e}); skipping crop.")
            continue

        final_ans = extract_final_answer(raw_answer)
        is_ev = not is_negative_or_unanswerable(final_ans if final_ans else raw_answer)

        # Collect detailed debug crop record for notebook visualization
        crop_records.append({
            "crop_idx": c_idx,
            "img_idx": img_idx,
            "box": list(item.box),
            "expanded_box": list(exp_box),
            "local_target_box": list(local_target_box),
            "raw_answer": raw_answer,
            "final_answer": final_ans,
            "is_evidence": is_ev,
            "source_branch": item.source_branch,
            "is_dual_scale": is_relational,
            "question_type": question_type
        })

        if is_ev:
            extractions.append(
                EvidenceExtraction(
                    img_idx=img_idx,
                    box=item.box,
                    expanded_box=exp_box,
                    raw_answer=raw_answer,
                    source_branch=item.source_branch,
                    final_answer=final_ans
                )
            )
            evidence_indices.add(img_idx)
            if config.verbose:
                print(f"[Stage 6][Evidence Found] Crop {c_idx} (Image {img_idx}, {item.source_branch}): Final Answer: '{final_ans}' (Raw: '{raw_answer[:60]}...')")

            # Fix B: Collision-Free Early Stopping comparing extracted core answers (no preamble collision)
            if getattr(config, "enable_early_consensus_exit", True) and len(extractions) >= 2:
                req_consensus = getattr(config, "early_consensus_count", 2)
                thresh = getattr(config, "early_consensus_threshold", 0.75)
                cur_core = (extractions[-1].final_answer or extract_final_answer(raw_answer)).strip().lower()

                matches = []
                for prev in extractions[:-1]:
                    prev_core = (prev.final_answer or extract_final_answer(prev.raw_answer)).strip().lower()
                    ratio = SequenceMatcher(None, cur_core, prev_core).ratio()
                    # Check substring match only if substantive length (>= 3 chars)
                    has_sub = (len(cur_core) >= 3 and cur_core in prev_core) or (len(prev_core) >= 3 and prev_core in cur_core)
                    if ratio >= thresh or has_sub:
                        matches.append(prev)

                if len(matches) + 1 >= req_consensus:
                    if config.verbose:
                        print(f"[Stage 6][Early Exit] Consensus confirmed ({len(matches) + 1} matching crops) on '{final_ans}'. Halting remaining {len(unified_pool) - c_idx - 1} crops to optimize throughput.")
                    break
        elif config.verbose:
            print(f"[Stage 6][Negative Filtered] Crop {c_idx} (Image {img_idx}) returned unanswerable response: '{raw_answer}'")

    # 4. Semantic Deduplication of Answer Memory using clean final answers
    clean_ans_list = [e.final_answer if e.final_answer else e.raw_answer for e in extractions]
    final_dedup_answers = deduplicate_answers(
        clean_ans_list,
        similarity_threshold=config.answer_dedup_similarity,
        verbose=config.verbose
    )

    if config.verbose:
        print(f"[Stage 6] Reading complete: Extracted {len(extractions)} evidence items, deduplicated to {len(final_dedup_answers)} unique answers: {final_dedup_answers} (Evidence images: {sorted(list(evidence_indices))}).")

    return FinalSearchResult(
        answers=final_dedup_answers,
        evidence_image_indices=sorted(list(evidence_indices)),
        extractions=extractions,
        crop_records=crop_records
    )

