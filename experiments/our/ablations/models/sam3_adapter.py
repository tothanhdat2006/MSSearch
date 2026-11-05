"""
Adapter wrapping SAM3 inference engine for MSSearch ablation suite.
"""

from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from .base import BaseDetector


class SAM3Detector(BaseDetector):
    """Wraps SAM3 open-vocabulary detection engine conforming to BaseDetector."""

    def __init__(self, model_path: str = "", device: Optional[str] = None, engine: Optional[Any] = None, **kwargs):
        self.model_path = model_path
        self.device = device
        if engine is not None:
            self.engine = engine
        else:
            try:
                from mssearch.models.modeling_sam3 import sam3_inference
                self.engine = sam3_inference(model_path=model_path, device=device, **kwargs)
            except ImportError:
                # Stub fallback when local dependencies (e.g. sam3 / spacy) are not present
                self.engine = None

    def batch_inference(
        self,
        image: Image.Image,
        text_prompts: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], List[int]]:
        if self.engine is None:
            return None, {i: {"boxes": [], "scores": []} for i in range(1, len(text_prompts) + 1)}, list(range(1, len(text_prompts) + 1))
        return self.engine.batch_inference(image, text_prompts)

    def batch_inference_multi_images(
        self,
        images: List[Image.Image],
        text_prompts: List[str],
        batch_size: int = 4,
    ) -> List[Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], List[int]]]:
        if hasattr(self.engine, "batch_inference_multi_images"):
            return self.engine.batch_inference_multi_images(images, text_prompts, batch_size=batch_size)
        return super().batch_inference_multi_images(images, text_prompts, batch_size=batch_size)
