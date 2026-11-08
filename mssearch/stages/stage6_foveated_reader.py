"""
Stage 6: Foveated VLM Reading with 1.25x BBox Context Expansion & Answer Deduplication.
Expands each candidate patch independently by 25% margin (no destructive bbox merging),
queries Qwen2.5-VL exhaustively across all evidence patches,
and deduplicates extracted target answers via semantic similarity into Answer Memory M.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from PIL import Image

from ..config import MSSearchConfig
from .stage5_screening import CandidateCrop
from .utils_stage6 import (
    EvidenceExtraction,
    FinalSearchResult,
    extract_final_answer,
    expand_bounding_box,
    is_negative_or_unanswerable,
    split_attribute_answer,
    deduplicate_answers,
    resolve_by_majority_voting,
)

__all__ = [
    "EvidenceExtraction",
    "FinalSearchResult",
    "extract_final_answer",
    "expand_bounding_box",
    "is_negative_or_unanswerable",
    "split_attribute_answer",
    "deduplicate_answers",
    "resolve_by_majority_voting",
    "run_foveated_vlm_reading",
]


def run_foveated_vlm_reading(
    unified_pool: List[CandidateCrop],
    candidate_images: List[Image.Image],
    question: str,
    zoom_model,
    config: MSSearchConfig,
    question_type: Optional[str] = None,
    **kwargs
) -> FinalSearchResult:
    """
    Executes Foveated VLM Reading with 1.25x padding and builds Answer Memory M.
    Reads all prepared crops with batched VLM execution (batch size 4 or 8).
    """
    if zoom_model is None:
        raise RuntimeError(
            "[Stage 6 Error] zoom_model (Qwen2.5-VL) is None. Stage 6 Foveated Reading requires a loaded VLM model; fallback and dry-run execution are strictly disabled."
        )

    extractions: List[EvidenceExtraction] = []
    evidence_indices: set = set()
    crop_records: List[Dict[str, Any]] = []

    if config.verbose:
        print(f"[Stage 6] Commencing foveated reading across {len(unified_pool)} candidate crops (1.25x bbox expansion, batch_size={getattr(config, 'vlm_batch_size', 4)})...")

    # 1. Prepare and pre-filter all crop inputs
    prepared_crops: List[Dict[str, Any]] = []

    for c_idx, item in enumerate(unified_pool):
        img_idx = item.img_idx
        img_pil = candidate_images[img_idx]
        img_w, img_h = img_pil.size

        if getattr(item, "is_compressed_canvas", False):
            # Spatial compressed canvas contains all retained patches with empty rows/cols excised
            foveated_crop = item.crop_image
            exp_box = item.box
            local_target_boxes = getattr(item, "remapped_boxes", []) or []
            if local_target_boxes:
                local_target_box = local_target_boxes[0]
            else:
                fcw, fch = foveated_crop.size
                local_target_box = (0, 0, fcw, fch)
                local_target_boxes = [local_target_box]
        else:
            # Expand bounding box independently by 1.25x
            exp_box = expand_bounding_box(
                item.box, img_w, img_h, expansion_factor=config.bbox_expansion_factor
            )

            foveated_crop = img_pil.crop(exp_box)
            fcw, fch = foveated_crop.size
            if fcw < 8 or fch < 8:
                continue

            x1, y1 = exp_box[0], exp_box[1]
            local_target_box = (
                item.box[0] - x1, item.box[1] - y1,
                item.box[2] - x1, item.box[3] - y1
            )
            local_target_boxes = [local_target_box]

        vlm_image_input: Image.Image = foveated_crop

        # VLM Answer Prompt (<image>\n<question> matching CVSearch standard)
        clean_q = question[len("<image>\n"):].strip() if question.startswith("<image>\n") else question.strip()
        prompt = f"<image>\n{clean_q}"

        prepared_crops.append({
            "c_idx": c_idx,
            "item": item,
            "img_idx": img_idx,
            "exp_box": exp_box,
            "local_target_box": local_target_box,
            "local_target_boxes": local_target_boxes,
            "vlm_image_input": vlm_image_input,
            "prompt": prompt
        })

    # 2. Batched Generation across Candidate Crops
    vlm_batch_size = max(1, getattr(config, "vlm_batch_size", 4))
    use_batching = hasattr(zoom_model, "batch_generate_response") and vlm_batch_size > 1
    max_tokens = getattr(config, "max_vlm_tokens", 512)

    for b_start in range(0, len(prepared_crops), vlm_batch_size):
        chunk = prepared_crops[b_start : b_start + vlm_batch_size]

        if use_batching and len(chunk) > 1:
            chunk_inputs = [c["vlm_image_input"] for c in chunk]
            chunk_prompts = [c["prompt"] for c in chunk]
            try:
                raw_answers = zoom_model.batch_generate_response(
                    chunk_inputs,
                    chunk_prompts,
                    max_new_tokens=max_tokens,
                    batch_size=vlm_batch_size
                )
            except Exception as e:
                if config.verbose:
                    print(f"[Stage 6][Batched Fallback] Batch generation failed ({e}); falling back to sequential.")
                raw_answers = []
                for c in chunk:
                    try:
                        raw_answers.append(zoom_model.generate_response(image_pil=c["vlm_image_input"], question=c["prompt"], max_new_tokens=max_tokens))
                    except Exception as err:
                        if config.verbose or b_start == 0:
                            print(f"[Stage 6][Error Fallback] Crop {c['c_idx']} (Image {c['img_idx']}) generation failed: {err}")
                        raw_answers.append("")
        else:
            raw_answers = []
            for c in chunk:
                try:
                    if hasattr(zoom_model, "generate_response"):
                        ans = zoom_model.generate_response(image_pil=c["vlm_image_input"], question=c["prompt"], max_new_tokens=max_tokens)
                    elif hasattr(zoom_model, "free_form_using_nodes"):
                        ans = zoom_model.free_form_using_nodes(image_pil=c["vlm_image_input"], question=c["prompt"], searched_nodes=[], max_new_tokens=max_tokens)
                    elif hasattr(zoom_model, "generate"):
                        ans = zoom_model.generate(image_pil=c["vlm_image_input"], question=c["prompt"], max_new_tokens=max_tokens)
                    else:
                        ans = str(zoom_model(image_pil=c["vlm_image_input"], question=c["prompt"]))
                    raw_answers.append(ans)
                except Exception as e:
                    if config.verbose or b_start == 0:
                        print(f"[Stage 6][Error Fallback] Crop {c['c_idx']} (Image {c['img_idx']}) generation failed: {e}")
                    raw_answers.append("")

        for c, raw_answer in zip(chunk, raw_answers):
            if not raw_answer:
                continue

            c_idx = c["c_idx"]
            item = c["item"]
            img_idx = c["img_idx"]
            exp_box = c["exp_box"]
            local_target_box = c["local_target_box"]
            local_target_boxes = c.get("local_target_boxes", [local_target_box] if local_target_box else [])

            final_ans = extract_final_answer(raw_answer)
            is_ev = bool(final_ans) and not is_negative_or_unanswerable(final_ans)

            crop_records.append({
                "crop_idx": c_idx,
                "img_idx": img_idx,
                "box": list(item.box),
                "expanded_box": list(exp_box),
                "local_target_box": list(local_target_box) if local_target_box is not None else [],
                "local_target_boxes": [list(b) for b in local_target_boxes] if local_target_boxes else [],
                "raw_answer": raw_answer,
                "final_answer": final_ans,
                "is_evidence": is_ev,
                "source_branch": item.source_branch,
                "is_compressed_canvas": getattr(item, "is_compressed_canvas", False),
                "is_dual_scale": False,
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
                        final_answer=final_ans,
                        crop_score=float(getattr(item, "score", 0.0) or 0.0)
                    )
                )
                evidence_indices.add(img_idx)
                if config.verbose:
                    print(f"[Stage 6][Evidence Found] Crop {c_idx} (Image {img_idx}, {item.source_branch}): Answer: '{final_ans}' (Raw: '{raw_answer[:60]}...')")

            elif config.verbose:
                print(f"[Stage 6][Negative Filtered] Crop {c_idx} (Image {img_idx}) returned unanswerable response: '{raw_answer}'")


    # 4. Semantic Deduplication of Answer Memory using clean final answers & Majority Voting
    if extractions:
        winner_ans, _, ranked_answers = resolve_by_majority_voting(
            extractions,
            similarity_thresh=getattr(config, "answer_dedup_similarity", 0.75),
            verbose=config.verbose
        )
        final_dedup_answers = ranked_answers if ranked_answers else []
    else:
        final_dedup_answers = []

    # 5. Strict Evidence Pruning: Retain only image indices that yielded affirmative, non-negative answers
    valid_evidence_indices: List[int] = []
    for e in extractions:
        cand = e.final_answer
        if cand and not is_negative_or_unanswerable(cand):
            if e.img_idx not in valid_evidence_indices:
                valid_evidence_indices.append(e.img_idx)

    if config.verbose:
        print(f"[Stage 6] Reading complete: Extracted {len(extractions)} evidence items, deduplicated to {len(final_dedup_answers)} unique answers: {final_dedup_answers} (Evidence images: {valid_evidence_indices}).")

    return FinalSearchResult(
        answers=final_dedup_answers,
        evidence_image_indices=valid_evidence_indices,
        extractions=extractions,
        crop_records=crop_records
    )
