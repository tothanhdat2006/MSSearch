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
class FastPathResult:
    """Telemetry and outcome container for Stage 2 Fast Path."""
    triggered: bool
    answer: Optional[str] = None
    evidence_img_idx: Optional[int] = None
    confidence: float = 0.0
    detected_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Caches dense SAM3 feature maps {img_idx: np.ndarray [C, H_f, W_f]} for zero-overhead reuse in Stage 3
    feature_cache: Dict[int, np.ndarray] = field(default_factory=dict)


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

    if not config.enable_fast_path or sam_model is None or zoom_model is None:
        if config.verbose:
            print("[Stage 2][Fallback] Tool-Based Fast Path disabled or models missing; routing directly to Stage 3 Visual Search Branch.")
        return FastPathResult(triggered=False, feature_cache=feature_cache)

    if config.verbose:
        print(f"[Stage 2] Scanning {len(candidate_images)} candidate images for Active Objects {active_objects} (Fast Path Threshold: {config.fast_path_conf_threshold})...")

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
            concise_hint = " Answer concisely with just the direct answer in a few words." if getattr(config, "enable_concise_reading_prompt", True) else ""
            answer_prompt = f"{question}{text_coords_hint}{concise_hint}"
            answer_text = _query_vlm_response(zoom_model, vlm_crop, answer_prompt, max_new_tokens=config.max_vlm_tokens)

            if config.verbose:
                print(f"[Stage 2][Non-Fallback Early Exit] Instant answer confirmed on image {img_idx}: '{answer_text}'. Exiting in 1 step!")
            return FastPathResult(
                triggered=True,
                answer=answer_text,
                evidence_img_idx=img_idx,
                confidence=float(existence_conf),
                detected_boxes=matched_boxes_by_obj,
                feature_cache=feature_cache
            )
        else:
            if config.verbose:
                print(f"[Stage 2][Fallback Trigger] Existence confidence {existence_conf:.3f} did not exceed {config.fast_path_conf_threshold:.2f}; falling back from instant exit.")

    if config.verbose:
        print(f"[Stage 2][Fallback to Stage 3] Fast Path did not trigger across all candidate images. Switching to Visual Search Branch. (Cached {len(feature_cache)} feature maps).")

    return FastPathResult(triggered=False, feature_cache=feature_cache)


# Backward-compatible alias
run_fast_path_check = run_tool_based_fast_path
