"""
Adapter wrapping SAM3 inference engine for MSSearch ablation suite.
"""

from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
from .base import BaseDetector


class SAM3Detector(BaseDetector):
    """Wraps SAM3 open-vocabulary detection engine conforming to BaseDetector."""

    def __init__(self, model_path: str, device: Optional[str] = None, **kwargs):
        from mssearch.models.modeling_sam3 import sam3_inference
        self.model_path = model_path
        self.device = device
        self.engine = sam3_inference(model_path=model_path, device=device, **kwargs)

    def batch_inference(
        self,
        image: Image.Image,
        text_prompts: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], List[int]]:
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
