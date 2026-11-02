"""
Stage 2: Tool-Based Fast Path vs Visual Search Routing.
Runs SAM3 full-canvas detection across N candidate images for Active Objects A_q.
If all |A_q| objects are detected with high VLM existence confidence (> 0.70),
generates the answer immediately and caches vision features for Stage 3 reuse.
"""

from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ..config import MSSearchConfig
from .utils_stage2 import (
    DirectAnswerResult,
    FastPathResult,
    compute_composite_bbox,
    render_visual_prompt_bbox,
    clamp_image_to_safe_resolution,
    _query_vlm_response,
    _query_vlm_existence,
    build_direct_answer_prompt,
)

__all__ = [
    "DirectAnswerResult",
    "FastPathResult",
    "compute_composite_bbox",
    "render_visual_prompt_bbox",
    "run_direct_global_assessment",
    "run_tool_based_fast_path",
]


def run_direct_global_assessment(
    candidate_images: List[Image.Image],
    question: str,
    zoom_model,
    config: MSSearchConfig,
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
    max_observed_conf: float = 0.0

    # Batch probe answering confidence c_q across all candidate images if supported
    vlm_batch_size = max(1, getattr(config, "vlm_batch_size", 4))
    conf_scores: Optional[List[float]] = None
    if hasattr(zoom_model, "batch_get_confidence_values") and vlm_batch_size > 1 and len(candidate_images) > 1:
        safe_imgs = [clamp_image_to_safe_resolution(img, 1024) for img in candidate_images]
        try:
            conf_scores = zoom_model.batch_get_confidence_values(
                image_list=safe_imgs,
                confidence_type="answering",
                input_ele=question,
                batch_size=vlm_batch_size,
            )
        except Exception as e:
            if config.verbose:
                print(f"[Phase 1][Batched Probing Fallback] Batch answering probe failed ({e}); using sequential probe.")
            conf_scores = None

    for img_idx, img_pil in enumerate(candidate_images):
        if conf_scores is not None and img_idx < len(conf_scores):
            conf = float(conf_scores[img_idx])
        else:
            conf = 0.0
            if hasattr(zoom_model, "get_confidence_value"):
                try:
                    safe_img = clamp_image_to_safe_resolution(img_pil, 1024)
                    val = zoom_model.get_confidence_value(
                        node=[],
                        image_pil=safe_img,
                        confidence_type="answering",
                        input_ele=question,
                        root_anyres=False,
                    )
                    if isinstance(val, torch.Tensor):
                        val = val.item()
                    conf = float(val)
                except Exception as e:
                    if config.verbose:
                        print(f"[Phase 1][Direct Answer Notice] Image {img_idx} confidence probing error ({e})")
                    conf = 0.0

        max_observed_conf = max(max_observed_conf, conf)
        if config.verbose:
            print(f"[Phase 1][Direct Answer] Image {img_idx}: answering confidence = {conf:.3f} (Threshold = {threshold:.2f})")

        # Check threshold gate
        if conf > threshold:
            prompt = build_direct_answer_prompt(question, config)
            raw_answer = _query_vlm_response(zoom_model, img_pil, prompt, max_new_tokens=getattr(config, "max_vlm_tokens", 128))
            final_ans = extract_final_answer(raw_answer)

            if is_negative_or_unanswerable(raw_answer) or (final_ans and is_negative_or_unanswerable(final_ans)):
                if config.verbose:
                    print(f"[Phase 1][Direct Answer Refusal] Global response was uninformative ('{raw_answer}'); bypassing Phase 1.")
                continue

            if len(candidate_images) > 1:
                if config.verbose:
                    print(f"[Phase 1][Direct Answer Confirmed] Image {img_idx} (conf={conf:.3f}): '{final_ans}'. Continuing scan...")
                collected_direct.append({
                    "answer": final_ans,
                    "raw_response": raw_answer,
                    "confidence": conf,
                    "img_idx": img_idx,
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
                evidence_images=[img_idx],
            )

    if collected_direct:
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
            print(f"[Phase 1][Direct Answer Complete] Found {len(ans_list)} direct answers across images {ev_imgs}: '{combined_ans}'.")
        return DirectAnswerResult(
            triggered=True,
            answer=combined_ans,
            raw_response="\n---\n".join(raw_resps),
            confidence=max_c,
            evidence_img_idx=ev_imgs[0],
            answers=ans_list,
            evidence_images=ev_imgs,
        )

    return DirectAnswerResult(triggered=False, confidence=max_observed_conf)


def run_tool_based_fast_path(
    candidate_images: List[Image.Image],
    active_objects: List[str],
    question: str,
    sam_model,
    zoom_model,
    config: MSSearchConfig,
    active_img_indices: Optional[List[int]] = None,
) -> FastPathResult:
    """
    Executes the Tool-Based Fast Path across the pool of candidate images.
    If active_img_indices is provided, only scans those image indices.
    """
    feature_cache: Dict[int, np.ndarray] = {}

    if not config.enable_fast_path:
        return FastPathResult(triggered=False, feature_cache=feature_cache)

    if sam_model is None:
        raise RuntimeError("[Stage 2 Error] sam_model (SAM3) is None. Tool Fast Path requires a loaded SAM3 model.")
    if zoom_model is None:
        raise RuntimeError("[Stage 2 Error] zoom_model (Qwen2.5-VL) is None. Tool Fast Path requires a loaded VLM model.")

    target_indices = active_img_indices if active_img_indices is not None else list(range(len(candidate_images)))
    if config.verbose:
        print(f"[Stage 2] Scanning {len(target_indices)} candidate images (indices: {target_indices}) for Active Objects {active_objects} (Fast Path Threshold: {config.fast_path_conf_threshold})...")

    collected_fp: List[Dict[str, Any]] = []
    max_observed_existence_conf: float = 0.0
    targets_detected_on_any_image: bool = False

    for img_idx, img_pil in enumerate(candidate_images):
        if active_img_indices is not None and img_idx not in active_img_indices:
            continue
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

        # 2. Extract and cache vision feature map for Stage 3 reuse
        if config.reuse_stage2_features and 'vision_features' in backbone_out:
            feat = backbone_out['vision_features']
            if isinstance(feat, torch.Tensor):
                feat_np = feat.detach().cpu().float().numpy().squeeze(0)
            else:
                feat_np = np.squeeze(feat, axis=0)
            feature_cache[img_idx] = feat_np

        del backbone_out
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Check detection coverage across all Active Objects
        all_objects_found = True
        matched_boxes_by_obj: List[Tuple[int, int, int, int]] = []

        for tid in text_id:
            res_for_t = processed_results.get(tid, {}) if isinstance(processed_results, dict) else {}
            boxes = res_for_t.get('boxes', [])
            scores = res_for_t.get('scores', [])

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
                matched_boxes_by_obj.append(confident_boxes[0])

        if not all_objects_found or len(matched_boxes_by_obj) < len(active_objects):
            if config.verbose:
                print(f"[Stage 2] Image {img_idx}: Detected {len(matched_boxes_by_obj)}/{len(active_objects)} objects -> missing targets, fast path bypass.")
            del processed_results
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        targets_detected_on_any_image = True

        # 4. Construct composite crop encompassing all active target detections
        composite_box = compute_composite_bbox(matched_boxes_by_obj, img_w, img_h)
        crop_pil = img_pil.crop(composite_box)
        cw, ch = crop_pil.size
        if cw < 8 or ch < 8:
            del processed_results
            del crop_pil
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        x_min, y_min = composite_box[0], composite_box[1]
        local_boxes = [
            (
                int(round(float(b[0]) - x_min)),
                int(round(float(b[1]) - y_min)),
                int(round(float(b[2]) - x_min)),
                int(round(float(b[3]) - y_min)),
            )
            for b in matched_boxes_by_obj
        ]

        # 5. Visual prompt annotation if enabled
        vlm_crop = crop_pil
        if config.enable_visual_prompt_bbox:
            vlm_crop = render_visual_prompt_bbox(
                crop_pil,
                local_boxes,
                color=config.visual_prompt_bbox_color,
                width=config.visual_prompt_bbox_width,
            )

        # 6. Textual coordinates prompt hint if enabled
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
        if existence_conf > max_observed_existence_conf:
            max_observed_existence_conf = float(existence_conf)

        if config.verbose:
            print(f"[Stage 2] Image {img_idx}: All targets detected! VLM existence conf = {existence_conf:.3f} (Threshold = {config.fast_path_conf_threshold:.2f})")

        # 8. Fast Path Decision Gate
        if existence_conf > config.fast_path_conf_threshold:
            answer_prompt = build_direct_answer_prompt(f"{question}{text_coords_hint}", config)
            answer_text = _query_vlm_response(zoom_model, vlm_crop, answer_prompt, max_new_tokens=config.max_vlm_tokens)

            from .stage6_foveated_reader import is_negative_or_unanswerable, extract_final_answer, deduplicate_answers
            final_ans = extract_final_answer(answer_text)
            if is_negative_or_unanswerable(answer_text) or (final_ans and is_negative_or_unanswerable(final_ans)):
                if config.verbose:
                    print(f"[Stage 2][Negative Fast Path] Fast path answer was uninformative/refusal ('{answer_text}'); bypassing fast path.")
                del processed_results
                del vlm_crop
                del crop_pil
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if len(candidate_images) > 1:
                if config.verbose:
                    print(f"[Stage 2][Fast Path Confirmed] Image {img_idx} (conf={existence_conf:.3f}): '{final_ans}'. Continuing scan...")
                collected_fp.append({
                    "answer": final_ans,
                    "raw_response": answer_text,
                    "img_idx": img_idx,
                    "confidence": float(existence_conf),
                    "boxes": matched_boxes_by_obj,
                })
                del processed_results
                del vlm_crop
                del crop_pil
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                evidence_images=[img_idx],
                boxes_by_image={img_idx: matched_boxes_by_obj},
            )
        else:
            if config.verbose:
                print(f"[Stage 2][Fallback Trigger] Existence confidence {existence_conf:.3f} did not exceed {config.fast_path_conf_threshold:.2f}; falling back from instant exit.")
            del processed_results
            del vlm_crop
            del crop_pil
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    if collected_fp:
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
            print(f"[Stage 2][Fast Path Complete] Found {len(ans_list)} fast path answers across images {ev_imgs}: '{combined_ans}'.")
        return FastPathResult(
            triggered=True,
            answer=combined_ans,
            raw_response="\n---\n".join(raw_resps),
            evidence_img_idx=ev_imgs[0],
            confidence=max_c,
            detected_boxes=all_boxes,
            feature_cache=feature_cache,
            answers=ans_list,
            evidence_images=ev_imgs,
            boxes_by_image={r["img_idx"]: r["boxes"] for r in collected_fp},
        )

    if config.verbose:
        print(f"[Stage 2][Fallback to Stage 3] Fast Path did not trigger across candidate images. Switching to Visual Search Branch. (Cached {len(feature_cache)} feature maps).")

    bypass_reason = "low_existence_confidence" if targets_detected_on_any_image else "targets_not_detected"
    conf_out = max_observed_existence_conf if targets_detected_on_any_image else 0.0
    return FastPathResult(
        triggered=False,
        confidence=conf_out,
        bypass_reason=bypass_reason,
        feature_cache=feature_cache,
    )
