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

    return raw_ids, actual_image_ids, pred_indices, pred_ans or raw_resp.strip()


def run_qwen_direct_multi_image(
    engine: Any,
    images: List[Image.Image],
    question: str,
    max_new_tokens: int = 128,
) -> Tuple[str, List[Any], List[int]]:
    """
    Executes multi-image direct baseline forward pass with dynamic pixel budgeting
    and multi-stage OOM recovery (progressive downscaling and sequential fallback).
    """
    n_imgs = max(1, len(images))
    prompt_text = format_multi_image_prompt(n_imgs, question)

    # Dynamic pixel ceiling: caps visual tokens to avoid token explosion on large N
    target_budget_pixels = 4_000_000
    per_img_max_pixels = max(128 * 128, min(1024 * 1024, target_budget_pixels // n_imgs))

    content = []
    parts = prompt_text.split("<image>")
    for i, part in enumerate(parts):
        if part:
            content.append({"type": "text", "text": part})
        if i < len(parts) - 1:
            content.append({"type": "image"})
    messages = [{"role": "user", "content": content}]
    full_prompt = engine.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _attempt_generate(imgs, max_px):
        call_kwargs = {}
        if hasattr(engine.processor, "image_processor") and hasattr(engine.processor.image_processor, "max_pixels"):
            call_kwargs["max_pixels"] = max_px
            call_kwargs["min_pixels"] = getattr(engine.processor.image_processor, "min_pixels", 28 * 28)
        inputs = engine.processor(text=[full_prompt], images=imgs, return_tensors="pt", padding=True, **call_kwargs).to(engine.device)
        with torch.inference_mode():
            out_ids = engine.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out_ids[0][len(inputs.input_ids[0]):]
        del inputs, out_ids
        return engine.processor.tokenizer.decode(trimmed, skip_special_tokens=True).strip()

    try:
        raw_res = _attempt_generate(images, per_img_max_pixels)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        print(f"[OOM Warning] Multi-image forward OOM with N={n_imgs}. Retrying with 50% downscaling...")
        try:
            downscaled = [ImageOps.exif_transpose(im).resize((max(1, im.width // 2), max(1, im.height // 2))) for im in images]
            raw_res = _attempt_generate(downscaled, per_img_max_pixels // 4)
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
            return (answers[0] if answers else "", [0], [1])

    _, _, pred_indices, pred_ans = parse_baseline_json_response(raw_res)
    pred_0based = [(p - 1) for p in pred_indices if p > 0]
    return pred_ans, pred_0based, pred_indices
