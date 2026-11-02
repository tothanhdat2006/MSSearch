"""
Stage 6: Foveated VLM Reading with 1.25x BBox Context Expansion & Answer Deduplication.
Expands each candidate patch independently by 25% margin (no destructive bbox merging),
queries Qwen2.5-VL exhaustively across all evidence patches,
and deduplicates extracted target answers via semantic similarity into Answer Memory M.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from difflib import SequenceMatcher
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
    is_singular_query,
)

__all__ = [
    "EvidenceExtraction",
    "FinalSearchResult",
    "extract_final_answer",
    "expand_bounding_box",
    "is_negative_or_unanswerable",
    "split_attribute_answer",
    "deduplicate_answers",
    "is_singular_query",
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
    Supports batched VLM execution (batch size 4 or 8) and query-adaptive dynamic halting.
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
    answer_mode = getattr(config, "answer_type", "free_form")

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
        if getattr(config, "prompt_directed_final_answer", False):
            if answer_mode == "attribute_list":
                prompt = (
                    f"<image>\n{clean_q}\n\n"
                    "State ONLY the specific attribute values asked in the question (e.g. brand, color, type, text, name).\n"
                    "Do NOT write full sentences, conversational filler, or explanations.\n"
                    "If multiple values are present, separate them with commas (e.g. <answer>item1, item2</answer>).\n"
                    "If the target object or requested information is NOT clearly visible or identifiable in this crop, output strictly:\n"
                    "<answer>None</answer>\n\n"
                    "Otherwise, output the exact attributes inside <answer>...</answer> tags:\n"
                    "<answer>concise answer</answer>"
                )
            elif answer_mode in ["option_single", "option_list", "multiple_choice", "mc", "Multiple Choice"] or "Options:\n" in clean_q:
                prompt = (
                    f"<image>\n{clean_q}\n\n"
                    "Select the best option from the choices above that answers the question.\n"
                    "State ONLY the option letter and concise choice (e.g. <answer>(A)</answer> or <answer>A. choice text</answer>).\n"
                    "Do NOT write conversational filler, reasoning, or explanations.\n"
                    "If the target object or requested information is NOT clearly visible in this crop, output strictly:\n"
                    "<answer>None</answer>\n\n"
                    "Otherwise, output the option letter inside <answer>...</answer> tags:\n"
                    "<answer>(A)</answer>"
                )
            else:
                prompt = (
                    f"<image>\n{clean_q}\n\n"
                    "State ONLY the direct factual answer (1 to 5 words). Do NOT write conversational filler or explanations.\n"
                    "If the target object or requested text/information is NOT clearly visible in this crop, output strictly:\n"
                    "<answer>None</answer>\n\n"
                    "Otherwise, output the concise answer inside <answer>...</answer> tags:\n"
                    "<answer>concise answer</answer>"
                )
        else:
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
    max_tokens = getattr(config, "max_vlm_tokens", 128)
    is_singular = is_singular_query(question, question_type)
    early_exit_halted = False

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
                    except Exception:
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
                    if config.verbose:
                        print(f"[Stage 6][Error Fallback] Crop {c['c_idx']} (Image {c['img_idx']}) generation failed ({e}); skipping.")
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
                        final_answer=final_ans
                    )
                )
                evidence_indices.add(img_idx)
                if config.verbose:
                    print(f"[Stage 6][Evidence Found] Crop {c_idx} (Image {img_idx}, {item.source_branch}): <answer>: '{final_ans}' (Raw: '{raw_answer[:60]}...')")

                # Consensus stopping & query-adaptive dynamic halting
                enable_halting = getattr(config, "enable_early_consensus_exit", True) or (
                    getattr(config, "enable_query_adaptive_halting", True) and is_singular
                )
                if enable_halting and len(extractions) >= 2:
                    req_consensus = getattr(config, "early_consensus_count", 2)
                    thresh = getattr(config, "early_consensus_threshold", 0.75)
                    cur_core = (extractions[-1].final_answer or extract_final_answer(raw_answer)).strip().lower()

                    matches = []
                    for prev in extractions[:-1]:
                        prev_core = (prev.final_answer or extract_final_answer(prev.raw_answer)).strip().lower()
                        ratio = SequenceMatcher(None, cur_core, prev_core).ratio()
                        has_sub = (len(cur_core) >= 3 and cur_core in prev_core) or (len(prev_core) >= 3 and prev_core in cur_core)
                        if ratio >= thresh or has_sub:
                            matches.append(prev)

                    if len(matches) + 1 >= req_consensus:
                        if config.verbose:
                            print(f"[Stage 6][Early Exit] Consensus confirmed ({len(matches) + 1} matching crops) on '{final_ans}'. Halting remaining {len(prepared_crops) - c_idx - 1} crops to optimize throughput.")
                        early_exit_halted = True
                        break
            elif config.verbose:
                print(f"[Stage 6][Negative Filtered] Crop {c_idx} (Image {img_idx}) returned unanswerable response: '{raw_answer}'")

        if early_exit_halted:
            break

    # 4. Semantic Deduplication of Answer Memory using clean final answers
    clean_ans_list = [e.final_answer for e in extractions if e.final_answer]
    final_dedup_answers = deduplicate_answers(
        clean_ans_list,
        similarity_threshold=config.answer_dedup_similarity,
        verbose=config.verbose
    )

    # 5. Strict Evidence Pruning: Retain only image indices that yielded affirmative, non-negative answers
    valid_evidence_indices: set = set()
    for e in extractions:
        cand = e.final_answer
        if cand and not is_negative_or_unanswerable(cand):
            valid_evidence_indices.add(e.img_idx)

    if config.verbose:
        print(f"[Stage 6] Reading complete: Extracted {len(extractions)} evidence items, deduplicated to {len(final_dedup_answers)} unique answers: {final_dedup_answers} (Evidence images: {sorted(list(valid_evidence_indices))}).")

    return FinalSearchResult(
        answers=final_dedup_answers,
        evidence_image_indices=sorted(list(valid_evidence_indices)),
        extractions=extractions,
        crop_records=crop_records
    )
