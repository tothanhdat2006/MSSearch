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

def load_vlm_engine(model_path_or_name: str, device: str = "cuda:0", attn_implementation: str = "sdpa") -> Any:
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
            return ModelQwenVL(model_path=model_path_or_name, device=device, attn_implementation=attn_implementation)
    else:
        # Defaults to QwenVL/AutoProcessor engine (handles Qwen2.5-VL, Qwen3-VL, LLaVA-OneVision)
        return ModelQwenVL(model_path=model_path_or_name, device=device, attn_implementation=attn_implementation)

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
        prompt_parts = [f"Candidate Image {idx + 1}:\n<image>\n" for idx in range(len(images))]
        prompt_parts.append(f"\n{question}")
        full_prompt = engine.get_prompt_from_qs("".join(prompt_parts))
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

        prompt_parts = [f"Candidate Image {idx + 1}:\n<image>\n" for idx in range(len(images))]
        prompt_parts.append(f"\n{question}")
        chat_prompt = "".join(prompt_parts)
        for num_p in num_patches_list:
            tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * engine.model.num_image_token * num_p + IMG_END_TOKEN
            chat_prompt = chat_prompt.replace("<image>", tokens, 1)

        resp = engine.model.chat(engine.tokenizer, full_pv, chat_prompt, dict(max_new_tokens=max_new_tokens, do_sample=False))
        return resp.strip(), [resp.strip()]

    # Single-Image / Composite Canvas Route (LLaVA-v1.6, LLaVA-HR, Yi-VL)
    canvas = create_vertical_panel_canvas(images)
    if hasattr(engine, "generate_response"):
        resp = engine.generate_response(image_pil=canvas, question=question, max_new_tokens=max_new_tokens)
    else:
        resp = str(engine(canvas, question))
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
