"""
Stage 2 Utilities: Data Structures, Geometry Calculations, and VLM Probing Helpers.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw
import numpy as np

import torch

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
    bypass_reason: Optional[str] = None
    detected_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Caches dense SAM3 feature maps {img_idx: np.ndarray [C, H_f, W_f]} for zero-overhead reuse in Stage 3
    feature_cache: Dict[int, np.ndarray] = field(default_factory=dict)
    answers: List[str] = field(default_factory=list)
    evidence_images: List[int] = field(default_factory=list)
    boxes_by_image: Dict[int, List[Tuple[int, int, int, int]]] = field(default_factory=dict)


def clamp_image_to_safe_resolution(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Clamps image to max dimension (default 1024px) using BILINEAR downsampling to prevent memory spikes."""
    if max(img.width, img.height) > max_dim:
        scale = float(max_dim) / max(img.width, img.height)
        return img.resize(
            (int(round(img.width * scale)), int(round(img.height * scale))),
            Image.Resampling.BILINEAR
        )
    return img


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
    safe_img = clamp_image_to_safe_resolution(image_pil, max_dim=1024)
    if hasattr(zoom_model, "generate_response"):
        resp = zoom_model.generate_response(image_pil=safe_img, question=question, max_new_tokens=max_new_tokens)
    elif hasattr(zoom_model, "free_form_using_nodes"):
        resp = zoom_model.free_form_using_nodes(image_pil=safe_img, question=question, searched_nodes=[], max_new_tokens=max_new_tokens)
    elif hasattr(zoom_model, "generate"):
        resp = zoom_model.generate(image_pil=safe_img, question=question, max_new_tokens=max_new_tokens)
    else:
        resp = str(zoom_model(image_pil=safe_img, question=question))
    return resp


def _query_vlm_existence(zoom_model, image_pil: Image.Image, query: str) -> float:
    """Invokes VLM existence confidence query with resolution safety shield."""
    if hasattr(zoom_model, "get_confidence_value"):
        safe_img = clamp_image_to_safe_resolution(image_pil, max_dim=1024)
        try:
            conf = zoom_model.get_confidence_value(
                node=None,
                image_pil=safe_img,
                confidence_type="existence",
                input_ele=query
            )
            if isinstance(conf, torch.Tensor):
                conf = conf.item()
            return float(conf)
        except Exception as e:
            print(f"[Stage 2][VLM Confidence Notice] {e}")
            return 0.0
    return 0.0


def build_direct_answer_prompt(question: str, config: Optional[MSSearchConfig] = None) -> str:
    """Constructs clean concise prompt (<image>\\n<question>) for Phase 1 Direct Answer."""
    clean_q = question[len("<image>\n"):].strip() if question.startswith("<image>\n") else question.strip()
    return f"<image>\n{clean_q}"
