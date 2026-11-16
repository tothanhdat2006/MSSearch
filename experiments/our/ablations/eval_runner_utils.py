"""
Shared utilities for MSSearch ablation evaluations.
Provides robust telemetry serialization, resumption tracking,
and OOM-resilient multi-image VLM inference with dynamic pixel budgets.
"""

import os
import gc
import json
import re
from typing import List, Dict, Any, Tuple, Optional
import torch
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None


def load_completed_keys(results_file: str) -> set:
    """Extracts completed task identifiers from an existing results file for atomic resumption."""
    completed = set()
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed.add(f"{rec.get('sample_id')}_{rec.get('pool_name')}")
                    except Exception:
                        pass
    return completed


def resolve_ground_truth(sample: Dict[str, Any]) -> List[str]:
    """Extracts a normalized ground-truth string list from diverse sample formats."""
    ans_field = sample.get("answer")
    if isinstance(ans_field, dict):
        return ans_field.get("list", [])
    elif isinstance(ans_field, list):
        return ans_field
    elif isinstance(ans_field, str):
        return [ans_field]
    return []


def format_multi_image_prompt(num_images: int, question: str) -> str:
    """Formats exact multi-image prompt requesting valid JSON image_ids and answer (v3 format)."""
    image_lines = "\n".join(f"Image {i}: <image>" for i in range(1, num_images + 1))
    return (
        f"{image_lines}\n\n"
        f"Question: {question}\n\n"
        "Identify the image(s) containing sufficient visual evidence\n"
        "to answer the question. Select only images that actually\n"
        "support the answer. Do not assume multiple images are relevant.\n\n"
        "Return only a valid JSON object with exactly two fields:\n"
        f'"image_ids": a list of one or more image numbers (integers from 1 to {num_images}) for the selected images\n'
        '"answer": a short, non-empty answer to the question\n'
        'If none of the images contains sufficient visual evidence, return {"image_ids": [], "answer": "not found"}'
    )


def parse_baseline_json_response(
    raw_resp: str, candidate_meta: Optional[List[Dict[str, Any]]] = None
) -> Tuple[List[Any], List[Any], List[int], str]:
    """Extracts (raw_image_ids, actual_image_ids, 1-based pred_indices, pred_answer)."""
    cleaned = raw_resp.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    json_str = match.group(1) if match else None
    if not json_str:
        match = re.search(r"(\{.*?\})", cleaned, re.DOTALL)
        json_str = match.group(1) if match else cleaned

    raw_ids, pred_ans = [], ""
    try:
        data = json.loads(json_str)
        val = data.get("image_ids") if data.get("image_ids") is not None else data.get("image_id")
        raw_ids = val if isinstance(val, list) else ([val] if val is not None else [])
        pred_ans = str(data.get("answer", "")).strip()
    except Exception:
        ids_m = re.search(r"[\"']image_ids?[\"']\s*:\s*\[([^\]]*)\]", raw_resp)
        if ids_m:
            raw_ids = [x.strip().strip("\"'") for x in ids_m.group(1).split(",") if x.strip()]
        else:
            id_m = re.search(r"[\"']image_ids?[\"']\s*:\s*([0-9]+|[\"'][^\"']+[\"'])", raw_resp)
            if id_m:
                raw_ids = [id_m.group(1).strip("\"'")]
        ans_m = re.search(r"[\"']answer[\"']\s*:\s*[\"']([^\"']+)[\"']", raw_resp)
        pred_ans = ans_m.group(1).strip() if ans_m else raw_resp.strip()
    pred_indices, actual_image_ids = [], []
    if candidate_meta:
        for r_id in raw_ids:
            try:
                idx_int = int(r_id)
            except (ValueError, TypeError):
                idx_int = None
            if idx_int is not None and 1 <= idx_int <= len(candidate_meta):
                pred_indices.append(idx_int)
                actual_image_ids.append(candidate_meta[idx_int - 1].get("image_id"))
            else:
                for idx, c in enumerate(candidate_meta):
                    if str(c.get("image_id")) == str(r_id):
                        pred_indices.append(idx + 1)
                        actual_image_ids.append(c.get("image_id"))
                        break
    else:
        actual_image_ids = raw_ids
        for r_id in raw_ids:
            try:
                idx_int = int(r_id)
                if idx_int > 0:
                    pred_indices.append(idx_int)
            except (ValueError, TypeError):
                pass
    return raw_ids, actual_image_ids, pred_indices, pred_ans or raw_resp.strip()


def run_qwen_direct_multi_image(
    engine: Any,
    images: List[Image.Image],
    question: str,
    max_new_tokens: int = 512,
) -> Tuple[str, List[Any], List[int]]:
    """
    Executes multi-image direct baseline forward pass with uncapped model resolution
    and multi-stage OOM recovery (progressive downscaling and sequential fallback).
    """
    n_imgs = max(1, len(images))
    prompt_text = format_multi_image_prompt(n_imgs, question)
    content = []
    parts = prompt_text.split("<image>")
    for i, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if i < len(parts) - 1:
            content.append({"type": "image"})
    messages = [{"role": "user", "content": content}]
    full_prompt = engine.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _attempt_generate(imgs, max_px: Optional[int] = None):
        call_kwargs = {}
        if max_px is not None and hasattr(engine.processor, "image_processor") and hasattr(engine.processor.image_processor, "max_pixels"):
            call_kwargs["max_pixels"] = max_px
            call_kwargs["min_pixels"] = getattr(engine.processor.image_processor, "min_pixels", 28 * 28)
        inputs = engine.processor(text=[full_prompt], images=imgs, return_tensors="pt", padding=True, **call_kwargs).to(engine.device)
        with torch.inference_mode():
            out_ids = engine.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out_ids[0][len(inputs.input_ids[0]):]
        del inputs, out_ids
        return engine.processor.tokenizer.decode(trimmed, skip_special_tokens=True).strip()

    try:
        raw_res = _attempt_generate(images)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        print(f"[OOM Warning] Multi-image forward OOM with N={n_imgs}. Retrying with 50% downscaling...")
        try:
            downscaled = [ImageOps.exif_transpose(im).resize((max(1, im.width // 2), max(1, im.height // 2))) for im in images]
            raw_res = _attempt_generate(downscaled)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"[OOM Fallback] Sequential one-by-one forward fallback activated.")
            answers = []
            for idx, im in enumerate(images):
                single_prompt = engine.get_prompt_from_qs(f"<image>\n{question}")
                inp = engine.processor(text=[single_prompt], images=[im], return_tensors="pt").to(engine.device)
                with torch.inference_mode():
                    out = engine.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
                ans = engine.processor.tokenizer.decode(out[0][len(inp.input_ids[0]):], skip_special_tokens=True).strip()
                answers.append(ans)
                del inp, out; torch.cuda.empty_cache()
            best_idx = 0
            for idx, a in enumerate(answers):
                if a and not any(neg in a.lower() for neg in ["not found", "cannot determine", "no answer", "none"]):
                    best_idx = idx
                    break
            return (answers[best_idx] if answers else "", [best_idx], [best_idx + 1])

    candidates = [{"image_id": i} for i in range(1, n_imgs + 1)]
    _, _, pred_indices, pred_ans = parse_baseline_json_response(raw_res, candidates)
    pred_0based = [(p - 1) for p in pred_indices if p > 0]
    return pred_ans, pred_0based, pred_indices


def run_joint_crop_reading(
    unified_pool: List[Any],
    candidate_images: List[Image.Image],
    question: str,
    zoom_model: Any,
    config: Any,
    question_type: Optional[str] = None,
    **kwargs,
):
    """
    Ablation "w/o Source-Attributed Consensus" (paper Sec. 4.4). Drop-in replacement for the Stage 6
    `run_foveated_vlm_reading`: prepares the same 1.25x-expanded native-resolution crops as Stage 6, but feeds
    ALL of them to the MLLM in ONE multi-image call (same JSON prompt as the `qwen` direct baseline) instead of
    reading each crop independently and voting. The MLLM's answer is the final answer and the source image of the
    first crop it selects is I*; no answer clustering / voting is applied. If the MLLM gives an informative answer
    but selects no crop, the highest-scoring crop is used for I* (recorded as `selection_fallback`).
    """
    from mssearch.stages.utils_stage6 import (
        EvidenceExtraction, FinalSearchResult, expand_bounding_box,
        extract_final_answer, is_negative_or_unanswerable,
    )
    crops, metas = [], []
    for c_idx, item in enumerate(unified_pool):
        img_pil = candidate_images[item.img_idx]
        img_w, img_h = img_pil.size
        if getattr(item, "is_already_expanded", False):
            exp_box = tuple(int(v) for v in item.box)
            crop = item.crop_image if item.crop_image is not None else img_pil.crop(exp_box)
        else:
            exp_box = expand_bounding_box(item.box, img_w, img_h, expansion_factor=config.bbox_expansion_factor)
            crop = img_pil.crop(exp_box)
        if crop.size[0] < 8 or crop.size[1] < 8:
            continue
        crops.append(crop)
        metas.append({"crop_idx": c_idx, "item": item, "img_idx": int(item.img_idx), "exp_box": exp_box})
    if not crops:
        return FinalSearchResult(answers=[], evidence_image_indices=[])

    clean_q = question[len("<image>\n"):].strip() if question.startswith("<image>\n") else question.strip()
    pred_ans, sel, _ = run_qwen_direct_multi_image(
        zoom_model, crops, clean_q, max_new_tokens=getattr(config, "max_vlm_tokens", 512)
    )
    sel = list(dict.fromkeys(i for i in sel if 0 <= i < len(metas)))
    final_ans = extract_final_answer(pred_ans)
    informative = bool(final_ans) and not is_negative_or_unanswerable(final_ans)
    fallback = informative and not sel
    if fallback:
        sel = [max(range(len(metas)), key=lambda i: float(getattr(metas[i]["item"], "score", 0.0) or 0.0))]
    sel_set = set(sel) if informative else set()

    crop_records = []
    for i, m in enumerate(metas):
        chosen = i in sel_set
        crop_records.append({
            "crop_idx": m["crop_idx"], "img_idx": m["img_idx"],
            "box": list(m["item"].box), "expanded_box": list(m["exp_box"]),
            "raw_answer": pred_ans if chosen else "", "final_answer": final_ans if chosen else "",
            "is_evidence": chosen, "source_branch": m["item"].source_branch,
            "joint_reading": True, "selected_by_mllm": chosen and not fallback,
            "selection_fallback": chosen and fallback, "question_type": question_type,
        })

    extractions, evidence = [], []
    if informative:
        for i in sel:
            if metas[i]["img_idx"] not in evidence:
                evidence.append(metas[i]["img_idx"])
        top = metas[sel[0]]
        extractions.append(EvidenceExtraction(
            img_idx=top["img_idx"], box=tuple(int(v) for v in top["item"].box), expanded_box=tuple(top["exp_box"]),
            raw_answer=pred_ans, source_branch="JointReading(no-consensus)", final_answer=final_ans,
            crop_score=float(getattr(top["item"], "score", 0.0) or 0.0),
        ))
    if getattr(config, "verbose", False):
        print(f"[Stage 6 (no consensus)] Joint reading of {len(crops)} crops -> answer '{final_ans}' | "
              f"selected crops {sel if informative else []}{' (fallback: top-score crop)' if fallback else ''} | evidence images {evidence}")
    return FinalSearchResult(
        answers=[final_ans] if informative else [],
        evidence_image_indices=evidence,
        extractions=extractions,
        crop_records=crop_records,
        voting_scores=[],
    )


try:
    from eval_metrics import format_stage6_answers, format_voting_scores
except ImportError:
    from experiments.our.eval_metrics import format_stage6_answers, format_voting_scores
