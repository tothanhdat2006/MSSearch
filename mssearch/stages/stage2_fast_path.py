"""
Stage 2: Tool-Based Fast Path vs Visual Search Routing.
Runs SAM3 full-canvas detection across N candidate images for Active Objects A_q.
If all |A_q| objects are detected with high VLM existence confidence (> 0.70),
generates the answer immediately and caches vision features for Stage 3 reuse.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw
import numpy as np
try:
    import torch
except ImportError:
    torch = None

from ..config import MSSearchConfig


@dataclass
class DirectAnswerResult:
    """Telemetry and outcome container for Phase 1 Direct Global Answer."""
    triggered: bool
    answer: Optional[str] = None
    raw_response: Optional[str] = None
    confidence: float = 0.0
    evidence_img_idx: Optional[int] = None
    answers: List[str] = field(default_factory=list)
    evidence_images: List[int] = field(default_factory=list)


@dataclass
class FastPathResult:
    """Telemetry and outcome container for Stage 2 Fast Path."""
    triggered: bool
    answer: Optional[str] = None
    raw_response: Optional[str] = None
    evidence_img_idx: Optional[int] = None
    confidence: float = 0.0
    detected_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Caches dense SAM3 feature maps {img_idx: np.ndarray [C, H_f, W_f]} for zero-overhead reuse in Stage 3
    feature_cache: Dict[int, np.ndarray] = field(default_factory=dict)
    answers: List[str] = field(default_factory=list)
    evidence_images: List[int] = field(default_factory=list)


def compute_composite_bbox(
    boxes: List[Any],
    img_w: int,
    img_h: int,
    margin_ratio: float = 0.10
) -> Tuple[int, int, int, int]:
    """Computes the union bounding box enclosing all detected targets with safety margin, guaranteed pure Python ints."""
    clean_boxes = []
    for b in boxes:
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().tolist()
        elif isinstance(b, np.ndarray):
            b = b.tolist()
        clean_boxes.append([float(coord) for coord in b])

    x1 = min(b[0] for b in clean_boxes)
    y1 = min(b[1] for b in clean_boxes)
    x2 = max(b[2] for b in clean_boxes)
    y2 = max(b[3] for b in clean_boxes)

    pad_w = (x2 - x1) * margin_ratio
    pad_h = (y2 - y1) * margin_ratio

    cx1 = max(0, int(round(x1 - pad_w)))
    cy1 = max(0, int(round(y1 - pad_h)))
    cx2 = min(img_w, int(round(x2 + pad_w)))
    cy2 = min(img_h, int(round(y2 + pad_h)))

    return (int(cx1), int(cy1), int(cx2), int(cy2))


def render_visual_prompt_bbox(
    crop_img: Image.Image,
    local_boxes: List[Any],
    color: str = "red",
    width: int = 3
) -> Image.Image:
    """Draws colored bounding boxes on the composite crop as visual prompts for the VLM."""
    annotated = crop_img.copy()
    draw = ImageDraw.Draw(annotated)
    for box in local_boxes:
        if torch is not None and isinstance(box, torch.Tensor):
            box = box.detach().cpu().tolist()
        elif isinstance(box, np.ndarray):
            box = box.tolist()
        clean_box = [int(round(float(coord))) for coord in box]
        draw.rectangle(clean_box, outline=color, width=width)
    return annotated


def _query_vlm_response(zoom_model, image_pil: Image.Image, question: str, max_new_tokens: int = 128) -> str:
    """Safely invokes VLM response generation across model APIs with resolution safety shield."""
    safe_img = image_pil
    if max(image_pil.width, image_pil.height) > 1024:
        scale = 1024.0 / max(image_pil.width, image_pil.height)
        safe_img = image_pil.resize(
            (int(round(image_pil.width * scale)), int(round(image_pil.height * scale))),
            Image.Resampling.LANCZOS
        )
    if hasattr(zoom_model, "generate_response"):
        return zoom_model.generate_response(image_pil=safe_img, question=question, max_new_tokens=max_new_tokens)
    elif hasattr(zoom_model, "free_form_using_nodes"):
        return zoom_model.free_form_using_nodes(image_pil=safe_img, question=question, searched_nodes=[], max_new_tokens=max_new_tokens)
    elif hasattr(zoom_model, "generate"):
        return zoom_model.generate(image_pil=safe_img, question=question, max_new_tokens=max_new_tokens)
    return str(zoom_model(image_pil=safe_img, question=question))


def _query_vlm_existence(zoom_model, image_pil: Image.Image, query: str) -> float:
    """Safely invokes VLM existence confidence query across model APIs with resolution safety shield."""
    if hasattr(zoom_model, "get_confidence_value"):
        # Physical resolution shield: clamp image to max 1024px to prevent 27.79 GiB SDPA OOM
        safe_img = image_pil
        if max(image_pil.width, image_pil.height) > 1024:
            scale = 1024.0 / max(image_pil.width, image_pil.height)
            safe_img = image_pil.resize(
                (int(round(image_pil.width * scale)), int(round(image_pil.height * scale))),
                Image.Resampling.LANCZOS
            )
        for kwarg in [{"node": None}, {"node_list": None}, {}]:
            try:
                conf = zoom_model.get_confidence_value(
                    image_pil=safe_img,
                    confidence_type="existence",
                    input_ele=query,
                    **kwarg
                )
                if isinstance(conf, torch.Tensor):
                    conf = conf.item()
                return float(conf)
            except TypeError:
                continue
            except Exception as e:
                print(f"[Stage 2][VLM Confidence Notice] {e}")
                return 0.0
    return 0.0


def run_direct_global_assessment(
    candidate_images: List[Image.Image],
    question: str,
    zoom_model,
    config: MSSearchConfig
) -> DirectAnswerResult:
    """
    Phase 1: Probes whether the question can be confidently answered directly from
    the root global image without tool invocation or multiscale decomposition.
    Uses Yes/No answering confidence probing:
      c_q = 2 * (Softmax(logits_Yes/No)[Yes] - 0.5) in [-1, 1].
    If c_q > config.direct_answer_conf_threshold, generates concise answer directly.
    """
    if not getattr(config, "enable_direct_answer", True) or zoom_model is None:
        return DirectAnswerResult(triggered=False)

    threshold = getattr(config, "direct_answer_conf_threshold", 0.60)
    if config.verbose:
        print(f"[Phase 1][Direct Answer] Probing {len(candidate_images)} images with answering confidence (Threshold = {threshold:.2f})...")

    from .stage6_foveated_reader import is_negative_or_unanswerable, extract_final_answer, deduplicate_answers

    collected_direct: List[Dict[str, Any]] = []

    for img_idx, img_pil in enumerate(candidate_images):
        # 1. Probe answering confidence c_q on the root view (clamped to 1024px for safety)
        conf = 0.0
        if hasattr(zoom_model, "get_confidence_value"):
            try:
                safe_img = img_pil
                if max(img_pil.width, img_pil.height) > 1024:
                    scale = 1024.0 / max(img_pil.width, img_pil.height)
                    safe_img = img_pil.resize(
                        (int(round(img_pil.width * scale)), int(round(img_pil.height * scale))),
                        Image.Resampling.BILINEAR
                    )
                val = zoom_model.get_confidence_value(
                    node=[],
                    image_pil=safe_img,
                    confidence_type="answering",
                    input_ele=question,
                    root_anyres=False
                )
                if isinstance(val, torch.Tensor):
                    val = val.item()
                conf = float(val)
            except Exception as e:
                if config.verbose:
                    print(f"[Phase 1][Direct Answer Notice] Image {img_idx} confidence probing error ({e})")
                conf = 0.0

        if config.verbose:
            print(f"[Phase 1][Direct Answer] Image {img_idx}: answering confidence = {conf:.3f} (Threshold = {threshold:.2f})")

        # 2. Check threshold gate
        if conf > threshold:
            if getattr(config, "prompt_directed_final_answer", True):
                prompt = (
                    f"{question}\n\n"
                    "If the target object or requested text/information is NOT clearly visible in this crop, output:\n"
                    "Final Answer: None\n\n"
                    "Otherwise, answer the question concisely. Conclude your response with:\n"
                    "Final Answer: <concise answer>"
                )
            else:
                prompt = (
                    f"{question}\n\n"
                    "If the target object or requested text/information is NOT clearly visible in this crop, output: Final Answer: None"
                )

            raw_answer = _query_vlm_response(zoom_model, img_pil, prompt, max_new_tokens=getattr(config, "max_vlm_tokens", 128))
            final_ans = extract_final_answer(raw_answer) if getattr(config, "prompt_directed_final_answer", True) else raw_answer

            # Safety check: bypass if answer is a refusal / unanswerable
            if is_negative_or_unanswerable(raw_answer) or (final_ans and is_negative_or_unanswerable(final_ans)):
                if config.verbose:
                    print(f"[Phase 1][Direct Answer Refusal] Global response was uninformative ('{raw_answer}'); bypassing Phase 1.")
                continue

            if getattr(config, "multi_answer_mode", False) and len(candidate_images) > 1:
                if config.verbose:
                    print(f"[Phase 1][Direct Answer Multi-Hit] Confirmed on image {img_idx} (conf={conf:.3f}): '{final_ans}'. Continuing scan for remaining candidate images...")
                collected_direct.append({
                    "answer": final_ans,
                    "raw_response": raw_answer,
                    "confidence": conf,
                    "img_idx": img_idx
                })
                continue

            if config.verbose:
                print(f"[Phase 1][Direct Answer Triggered] Confirmed on image {img_idx} (conf={conf:.3f}): '{final_ans}'. Bypassing all deep search stages!")
            return DirectAnswerResult(
                triggered=True,
                answer=final_ans,
                raw_response=raw_answer,
                confidence=conf,
                evidence_img_idx=img_idx,
                answers=[final_ans],
                evidence_images=[img_idx]
            )

    if getattr(config, "multi_answer_mode", False) and collected_direct:
        raw_ans_list = [r["answer"] for r in collected_direct]
        ev_imgs = [r["img_idx"] for r in collected_direct]
        raw_resps = [r["raw_response"] for r in collected_direct]
        max_c = max(r["confidence"] for r in collected_direct)
        sim_thresh = getattr(config, "answer_dedup_similarity", 0.85)
        ans_list = deduplicate_answers(raw_ans_list, similarity_threshold=sim_thresh, verbose=config.verbose)
        if not ans_list:
            ans_list = raw_ans_list
        combined_ans = "; ".join(ans_list)
        if config.verbose:
            print(f"[Phase 1][Direct Answer Multi-Complete] Found {len(ans_list)} direct answers across images {ev_imgs}: '{combined_ans}'.")
        return DirectAnswerResult(
            triggered=True,
            answer=combined_ans,
            raw_response="\n---\n".join(raw_resps),
            confidence=max_c,
            evidence_img_idx=ev_imgs[0],
            answers=ans_list,
            evidence_images=ev_imgs
        )

    return DirectAnswerResult(triggered=False)


def run_tool_based_fast_path(
    candidate_images: List[Image.Image],
    active_objects: List[str],
    question: str,
    sam_model,
    zoom_model,
    config: MSSearchConfig
) -> FastPathResult:
    """
    Executes the Tool-Based Fast Path across the pool of candidate images.
    """
    feature_cache: Dict[int, np.ndarray] = {}

    if not config.enable_fast_path:
        return FastPathResult(triggered=False, feature_cache=feature_cache)

    if sam_model is None:
        raise RuntimeError("[Stage 2 Error] sam_model (SAM3) is None. Tool Fast Path requires a loaded SAM3 model; fallback execution is strictly disabled.")
    if zoom_model is None:
        raise RuntimeError("[Stage 2 Error] zoom_model (Qwen2.5-VL) is None. Tool Fast Path requires a loaded VLM model; fallback execution is strictly disabled.")

    if config.verbose:
        print(f"[Stage 2] Scanning {len(candidate_images)} candidate images for Active Objects {active_objects} (Fast Path Threshold: {config.fast_path_conf_threshold})...")

    collected_fp: List[Dict[str, Any]] = []

    for img_idx, img_pil in enumerate(candidate_images):
        img_w, img_h = img_pil.size

        # 1. SAM3 Full-Canvas Batched Detection
        try:
            with torch.inference_mode():
                backbone_out, processed_results, text_id = sam_model.batch_inference(
                    img_pil, active_objects
                )
        except Exception as e:
            if config.verbose:
                print(f"[Stage 2][Error Fallback] SAM3 canvas inference failed on image {img_idx} ({e}); continuing.")
            continue

        # 2. Extract and cache vision feature map for Stage 3 reuse (Zero-recompute invariant)
        if config.reuse_stage2_features and 'vision_features' in backbone_out:
            feat = backbone_out['vision_features']
            if isinstance(feat, torch.Tensor):
                feat_np = feat.detach().cpu().float().numpy().squeeze(0)
            else:
                feat_np = np.squeeze(feat, axis=0)
            feature_cache[img_idx] = feat_np

        # 3. Check detection coverage across all Active Objects
        all_objects_found = True
        matched_boxes_by_obj: List[Tuple[int, int, int, int]] = []

        for tid in text_id:
            res_for_t = processed_results.get(tid, {}) if isinstance(processed_results, dict) else {}
            boxes = res_for_t.get('boxes', [])
            scores = res_for_t.get('scores', [])

            # Convert to CPU Python lists
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.detach().cpu().tolist()
            elif isinstance(boxes, np.ndarray):
                boxes = boxes.tolist()

            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().tolist()
            elif isinstance(scores, np.ndarray):
                scores = scores.tolist()

            confident_boxes = []
            for b, s in zip(boxes, scores):
                s_val = float(s) if not isinstance(s, (list, tuple)) else float(s[0])
                if s_val >= config.sam3_group_det_threshold:
                    if (torch is not None and isinstance(b, torch.Tensor)) or isinstance(b, np.ndarray):
                        b = b.tolist()
                    b_int = tuple(int(round(float(c))) for c in b)
                    confident_boxes.append(b_int)

            if len(confident_boxes) == 0:
                all_objects_found = False
                break
            else:
                matched_boxes_by_obj.append(confident_boxes[0])  # Take highest scoring detection

        if not all_objects_found or len(matched_boxes_by_obj) < len(active_objects):
            if config.verbose:
                print(f"[Stage 2] Image {img_idx}: Detected {len(matched_boxes_by_obj)}/{len(active_objects)} objects -> missing targets, fast path bypass.")
            continue

        # 4. Construct composite crop encompassing all active target detections (guaranteed pure ints)
        composite_box = compute_composite_bbox(matched_boxes_by_obj, img_w, img_h)
        crop_pil = img_pil.crop(composite_box)
        cw, ch = crop_pil.size
        if cw < 8 or ch < 8:
            continue

        # Compute coordinates relative to crop origin (guaranteed pure ints)
        x_min, y_min = composite_box[0], composite_box[1]
        local_boxes = [
            (
                int(round(float(b[0]) - x_min)),
                int(round(float(b[1]) - y_min)),
                int(round(float(b[2]) - x_min)),
                int(round(float(b[3]) - y_min))
            )
            for b in matched_boxes_by_obj
        ]

        # 5. Apply Visual Prompt Bounding Box (controlled by config flag)
        vlm_crop = crop_pil
        if config.enable_visual_prompt_bbox:
            vlm_crop = render_visual_prompt_bbox(
                crop_pil,
                local_boxes,
                color=config.visual_prompt_bbox_color,
                width=config.visual_prompt_bbox_width
            )

        # 6. Build Textual Prompt Coordinates (controlled by config flag)
        text_coords_hint = ""
        if config.enable_textual_prompt_coords:
            coord_strs = [
                f"{obj} at [{b[0]}, {b[1]}, {b[2]}, {b[3]}]"
                for obj, b in zip(active_objects, local_boxes)
            ]
            text_coords_hint = f" Specified visual targets: {'; '.join(coord_strs)}."

        # 7. Query VLM Existence Confidence
        existence_query = f"{', '.join(active_objects)}.{text_coords_hint}"
        existence_conf = _query_vlm_existence(zoom_model, vlm_crop, existence_query)

        if config.verbose:
            print(f"[Stage 2] Image {img_idx}: All targets detected! VLM existence conf = {existence_conf:.3f} (Threshold = {config.fast_path_conf_threshold:.2f})")

        # 8. Fast Path Decision Gate (> 0.70)
        if existence_conf > config.fast_path_conf_threshold:
            # High existence confidence confirmed -> Generate final answer immediately!
            if getattr(config, "prompt_directed_final_answer", True):
                answer_prompt = (
                    f"{question}{text_coords_hint}\n\n"
                    "If the target object or requested text/information is NOT clearly visible in this crop, output:\n"
                    "Final Answer: None\n\n"
                    "Otherwise, answer the question concisely. Conclude your response with:\n"
                    "Final Answer: <concise answer>"
                )
            else:
                answer_prompt = (
                    f"{question}{text_coords_hint}\n\n"
                    "If the target object or requested text/information is NOT clearly visible in this crop, output: Final Answer: None"
                )
            answer_text = _query_vlm_response(zoom_model, vlm_crop, answer_prompt, max_new_tokens=config.max_vlm_tokens)

            from .stage6_foveated_reader import is_negative_or_unanswerable, extract_final_answer, deduplicate_answers
            final_ans = extract_final_answer(answer_text) if getattr(config, "prompt_directed_final_answer", True) else answer_text
            if is_negative_or_unanswerable(answer_text) or (final_ans and is_negative_or_unanswerable(final_ans)):
                if config.verbose:
                    print(f"[Stage 2][Negative Fast Path] Fast path answer was uninformative/refusal ('{answer_text}'); bypassing fast path.")
                continue

            if getattr(config, "multi_answer_mode", False) and len(candidate_images) > 1:
                if config.verbose:
                    print(f"[Stage 2][Fast Path Multi-Hit] Confirmed on image {img_idx} (conf={existence_conf:.3f}): '{final_ans}'. Continuing scan for remaining candidate images...")
                collected_fp.append({
                    "answer": final_ans,
                    "raw_response": answer_text,
                    "img_idx": img_idx,
                    "confidence": float(existence_conf),
                    "boxes": matched_boxes_by_obj
                })
                continue

            if config.verbose:
                print(f"[Stage 2][Non-Fallback Early Exit] Instant answer confirmed on image {img_idx}: '{final_ans}'. Exiting in 1 step!")
            return FastPathResult(
                triggered=True,
                answer=final_ans,
                raw_response=answer_text,
                evidence_img_idx=img_idx,
                confidence=float(existence_conf),
                detected_boxes=matched_boxes_by_obj,
                feature_cache=feature_cache,
                answers=[final_ans],
                evidence_images=[img_idx]
            )
        else:
            if config.verbose:
                print(f"[Stage 2][Fallback Trigger] Existence confidence {existence_conf:.3f} did not exceed {config.fast_path_conf_threshold:.2f}; falling back from instant exit.")

    if getattr(config, "multi_answer_mode", False) and collected_fp:
        raw_ans_list = [r["answer"] for r in collected_fp]
        ev_imgs = [r["img_idx"] for r in collected_fp]
        raw_resps = [r["raw_response"] for r in collected_fp]
        all_boxes = [b for r in collected_fp for b in r["boxes"]]
        max_c = max(r["confidence"] for r in collected_fp)
        sim_thresh = getattr(config, "answer_dedup_similarity", 0.85)
        ans_list = deduplicate_answers(raw_ans_list, similarity_threshold=sim_thresh, verbose=config.verbose)
        if not ans_list:
            ans_list = raw_ans_list
        combined_ans = "; ".join(ans_list)
        if config.verbose:
            print(f"[Stage 2][Fast Path Multi-Complete] Found {len(ans_list)} fast path answers across images {ev_imgs}: '{combined_ans}'.")
        return FastPathResult(
            triggered=True,
            answer=combined_ans,
            raw_response="\n---\n".join(raw_resps),
            evidence_img_idx=ev_imgs[0],
            confidence=max_c,
            detected_boxes=all_boxes,
            feature_cache=feature_cache,
            answers=ans_list,
            evidence_images=ev_imgs
        )

    if config.verbose:
        print(f"[Stage 2][Fallback to Stage 3] Fast Path did not trigger across all candidate images. Switching to Visual Search Branch. (Cached {len(feature_cache)} feature maps).")

    return FastPathResult(triggered=False, feature_cache=feature_cache)


# Backward-compatible alias
run_fast_path_check = run_tool_based_fast_path
