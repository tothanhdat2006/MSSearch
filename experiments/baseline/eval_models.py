import os
import torch
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
from difflib import SequenceMatcher
try:
    from eval_metrics import is_negative_or_unanswerable
except ModuleNotFoundError:
    from experiments.baseline.eval_metrics import is_negative_or_unanswerable

def create_vertical_panel_canvas(images: List[Image.Image]) -> Image.Image:
    """Scales images to max width of the batch, stacking vertically with 20px white gap and 2px borders."""
    if not images:
        return Image.new("RGB", (100, 100), (255, 255, 255))
    target_width = max(img.width for img in images)
    gap_px, border_px, banner_h = 20, 2, 30
    resized_images = []
    for img in images:
        scale = target_width / img.width
        new_h = max(1, int(round(img.height * scale)))
        resized_images.append(img.resize((target_width, new_h), Image.Resampling.BILINEAR))

    total_height = sum(img.height + gap_px + banner_h for img in resized_images)
    canvas = Image.new("RGB", (target_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    cur_y = 0
    for idx, img in enumerate(resized_images):
        draw.text((10, cur_y + 6), f"Candidate Image {idx + 1}", fill=(0, 0, 0))
        cur_y += banner_h
        draw.rectangle([0, cur_y, target_width - 1, cur_y + border_px], fill=(200, 200, 200))
        cur_y += border_px
        canvas.paste(img, (0, cur_y))
        cur_y += img.height
        draw.rectangle([0, cur_y, target_width - 1, cur_y + border_px], fill=(200, 200, 200))
        cur_y += (gap_px - border_px)
    return canvas

def load_vlm_engine(
    model_path_or_name: str,
    device: str = "cuda:0",
    attn_implementation: str = "sdpa",
    min_pixels: int = 3136,
    max_pixels: int = 1048576,
) -> Any:
    """Loads appropriate VLM engine from local directory or Hugging Face repository."""
    from mssearch.models import ModelQwenVL, ModelInternvl, ModelYiVL, ModelLlavaHR, ModelGlobalLocal
    m_str = str(model_path_or_name).lower()

    if "internvl" in m_str:
        return ModelInternvl(model_path=model_path_or_name, device=device)
    elif "yi-vl" in m_str or "yi_vl" in m_str:
        return ModelYiVL(model_path=model_path_or_name, device=device)
    elif "llava-hr" in m_str or "llava_hr" in m_str:
        return ModelLlavaHR(model_path=model_path_or_name, device=device)
    elif "llava-v1.6" in m_str or "llava_v1.6" in m_str or "vicuna" in m_str:
        try:
            return ModelGlobalLocal(model_path=model_path_or_name, device=device)
        except Exception:
            return ModelQwenVL(model_path=model_path_or_name, device=device, attn_implementation=attn_implementation, min_pixels=min_pixels, max_pixels=max_pixels)
    else:
        # Defaults to QwenVL/AutoProcessor engine (handles Qwen2.5-VL, Qwen3-VL, LLaVA-OneVision)
        return ModelQwenVL(model_path=model_path_or_name, device=device, attn_implementation=attn_implementation, min_pixels=min_pixels, max_pixels=max_pixels)

def format_baseline_prompt(num_images: int, question: str) -> str:
    """Formats exact multi-image prompt requesting valid JSON image_id and answer."""
    lines = [f"Image {idx + 1}: <image>\n" for idx in range(num_images)]
    lines.append(f"\nQuestion: {question}\n\n")
    lines.append("Identify the image that contains the evidence and answer the question.\n")
    lines.append("Return only valid JSON:\n")
    lines.append('{"image_id": 1, "answer": "your answer"}')
    return "".join(lines)

def parse_baseline_json_response(raw_resp: str, candidate_meta: List[Dict[str, Any]] = None) -> Tuple[Any, Any, Any, str]:
    """Extracts (raw_image_id, actual_image_id, image_index_1based, free_form_answer)."""
    import re, json
    cleaned = raw_resp.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    json_str = match.group(1) if match else None
    if not json_str:
        match = re.search(r"(\{.*?\})", cleaned, re.DOTALL)
        json_str = match.group(1) if match else cleaned

    pred_img_id = None
    pred_ans = ""
    try:
        data = json.loads(json_str)
        pred_img_id = data.get("image_id")
        pred_ans = str(data.get("answer", "")).strip()
    except Exception:
        id_m = re.search(r"[\"']image_id[\"']\s*:\s*([0-9]+|[\"'][^\"']+[\"'])", raw_resp)
        ans_m = re.search(r"[\"']answer[\"']\s*:\s*[\"']([^\"']+)[\"']", raw_resp)
        if id_m:
            raw_id = id_m.group(1).strip("\"'")
            try:
                pred_img_id = int(raw_id)
            except ValueError:
                pred_img_id = raw_id
        pred_ans = ans_m.group(1).strip() if ans_m else raw_resp.strip()

    pred_img_idx = None
    actual_img_id = None
    if candidate_meta:
        if isinstance(pred_img_id, int) and 1 <= pred_img_id <= len(candidate_meta):
            pred_img_idx = pred_img_id
            actual_img_id = candidate_meta[pred_img_id - 1].get("image_id")
        elif pred_img_id is not None:
            for idx, c in enumerate(candidate_meta):
                if str(c.get("image_id")) == str(pred_img_id):
                    pred_img_idx = idx + 1
                    actual_img_id = c.get("image_id")
                    break
    if not actual_img_id and pred_img_id is not None:
        actual_img_id = pred_img_id
    if not pred_ans:
        pred_ans = raw_resp.strip()
    return pred_img_id, actual_img_id, pred_img_idx, pred_ans

def run_vlm_baseline_inference(
    engine: Any,
    images: List[Image.Image],
    question: str,
    eval_strategy: str = "multi_image",
    max_new_tokens: int = 128
) -> Tuple[str, List[str]]:
    """Dispatches multi-image or one-by-one inference across candidate images."""
    if eval_strategy == "one_by_one":
        return _run_one_by_one(engine, images, question, max_new_tokens)

    # Native Multi-Image Route for Qwen / LLaVA-OV
    if hasattr(engine, "processor") and hasattr(engine, "model") and hasattr(engine, "get_prompt_from_qs"):
        prompt_str = format_baseline_prompt(len(images), question)
        full_prompt = engine.get_prompt_from_qs(prompt_str)
        try:
            inputs = engine.processor(text=[full_prompt], images=[images], return_tensors="pt", padding=True).to(engine.device)
            with torch.inference_mode():
                out_ids = engine.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            trimmed = out_ids[0][len(inputs.input_ids[0]):]
            res = engine.processor.tokenizer.decode(trimmed, skip_special_tokens=True).strip()
            return res, [res]
        except Exception:
            pass  # Fall through to vertical composite canvas

    # InternVL Chat Route
    if hasattr(engine, "model") and hasattr(engine, "tokenizer") and hasattr(getattr(engine, "model", None), "chat"):
        from mssearch.models.modeling_internvl import load_image, IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN
        pixel_vals, num_patches_list = [], []
        for img in images:
            pv = load_image(img, input_size=(448, 448), max_num=12)
            pixel_vals.append(pv.to(engine.dtype))
            num_patches_list.append(pv.shape[0])
        full_pv = torch.cat(pixel_vals, dim=0).to(engine.device)

        chat_prompt = format_baseline_prompt(len(images), question)
        for num_p in num_patches_list:
            tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * engine.model.num_image_token * num_p + IMG_END_TOKEN
            chat_prompt = chat_prompt.replace("<image>", tokens, 1)

        resp = engine.model.chat(engine.tokenizer, full_pv, chat_prompt, dict(max_new_tokens=max_new_tokens, do_sample=False))
        return resp.strip(), [resp.strip()]

    # Single-Image / Composite Canvas Route (LLaVA-v1.6, LLaVA-HR, Yi-VL)
    canvas = create_vertical_panel_canvas(images)
    composite_qs = (
        f"Image 1 to Image {len(images)} are stacked vertically in the provided image.\n\n"
        f"Question: {question}\n\n"
        f"Identify the image that contains the evidence and answer the question.\n"
        f"Return only valid JSON:\n"
        f'{{"image_id": 1, "answer": "your answer"}}'
    )
    if hasattr(engine, "generate_response"):
        resp = engine.generate_response(image_pil=canvas, question=composite_qs, max_new_tokens=max_new_tokens)
    else:
        resp = str(engine(canvas, composite_qs))
    return resp.strip(), [resp.strip()]

def _run_one_by_one(engine: Any, images: List[Image.Image], question: str, max_new_tokens: int) -> Tuple[str, List[str]]:
    answers = []
    for img in images:
        if hasattr(engine, "generate_response"):
            ans = engine.generate_response(image_pil=img, question=question, max_new_tokens=max_new_tokens)
        else:
            ans = engine.processor.tokenizer.decode(engine.model.generate(
                **engine.processor(text=[f"<image>\n{question}"], images=[img], return_tensors="pt").to(engine.device),
                max_new_tokens=max_new_tokens
            )[0], skip_special_tokens=True)
        if not is_negative_or_unanswerable(ans):
            answers.append(ans.strip())

    # Semicolon deduplicated consolidation
    unique_answers = []
    for a in answers:
        if not any(SequenceMatcher(None, a.lower(), u.lower()).ratio() >= 0.85 for u in unique_answers):
            unique_answers.append(a)

    consolidated = "; ".join(unique_answers) if unique_answers else "None"
    return consolidated, unique_answers
