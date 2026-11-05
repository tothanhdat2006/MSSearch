"""
Grounding DINO detector adapter for MSSearch ablation suite.
Uses HuggingFace AutoModelForZeroShotObjectDetection / AutoProcessor.
"""

from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torch
from .base import BaseDetector


class GroundingDinoDetector(BaseDetector):
    """
    Open-vocabulary detector adapter using Grounding DINO
    (IDEA-Research/grounding-dino-base or grounding-dino-tiny).
    """

    def __init__(
        self,
        model_name_or_path: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda:0",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        self.model_name = model_name_or_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)

        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Loads HuggingFace Grounding DINO processor and model."""
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        print(f"[GroundingDinoDetector] Loading '{self.model_name}' on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        ).to(self.device).eval()

    @torch.inference_mode()
    def batch_inference(
        self,
        image: Image.Image,
        text_prompts: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], List[int]]:
        """
        Runs Grounding DINO on image for each text prompt in text_prompts.
        Returns: (backbone_out, processed_results, text_id)
        """
        if not text_prompts:
            return None, {}, []

        w, h = image.size
        processed_results: Dict[int, Dict[str, Any]] = {}
        text_id = list(range(1, len(text_prompts) + 1))

        for idx, prompt in enumerate(text_prompts, start=1):
            clean_text = prompt.strip().rstrip(".") + "."
            inputs = self.processor(images=image, text=clean_text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.dtype == torch.float16 and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            outputs = self.model(**inputs)
            target_sizes = torch.tensor([[h, w]], device=self.device)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )[0]

            boxes_t = results["boxes"]
            scores_t = results["scores"]

            boxes = boxes_t.detach().cpu().tolist() if hasattr(boxes_t, "tolist") else list(boxes_t)
            scores = scores_t.detach().cpu().tolist() if hasattr(scores_t, "tolist") else list(scores_t)

            # ponytail: clamp pixel coordinates to [0, W] and [0, H]
            clamped_boxes = []
            for b in boxes:
                x1 = max(0, min(w, int(round(float(b[0])))))
                y1 = max(0, min(h, int(round(float(b[1])))))
                x2 = max(0, min(w, int(round(float(b[2])))))
                y2 = max(0, min(h, int(round(float(b[3])))))
                clamped_boxes.append([x1, y1, x2, y2])

            processed_results[idx] = {
                "boxes": clamped_boxes,
                "scores": [float(s) for s in scores],
            }

        return None, processed_results, text_id

    @torch.inference_mode()
    def batch_inference_multi_images(
        self,
        images: List[Image.Image],
        text_prompts: List[str],
        batch_size: int = 4,
    ) -> List[Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], List[int]]]:
        if not images:
            return []
        results = []
        for img in images:
            try:
                results.append(self.batch_inference(img, text_prompts))
            except Exception:
                results.append((None, {}, list(range(1, len(text_prompts) + 1))))
        return results
