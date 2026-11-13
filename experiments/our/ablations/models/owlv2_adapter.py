"""
OWLv2 detector adapter for MSSearch ablation suite.
Uses Google OWLv2 (google/owlv2-base-patch16-ensemble) via Hugging Face Transformers.
Extracts both dense ViT patch tokens reshaped to (1, C, H_f, W_f) for Stage 3 SGAP
and grounded open-vocabulary object detection boxes for Stage 4 Feature Grouping.
"""

from typing import List, Tuple, Dict, Any, Optional
import math
from PIL import Image
import torch
import numpy as np

from .base import BaseDetector


class OWLv2Detector(BaseDetector):
    """
    Open-vocabulary detector adapter using Google OWLv2
    (google/owlv2-base-patch16-ensemble or google/owlv2-large-patch14-ensemble).
    """

    def __init__(
        self,
        model_name_or_path: str = "google/owlv2-base-patch16-ensemble",
        device: str = "cuda:0",
        torch_dtype: Optional[torch.dtype] = None,
        box_threshold: float = 0.3,
        lazy_load: bool = False,
        **kwargs,
    ):
        self.model_name = model_name_or_path or "google/owlv2-base-patch16-ensemble"
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.box_threshold = box_threshold

        self.processor = None
        self.model = None
        self.stub_mode = False

        if not lazy_load:
            self._ensure_loaded()

    def _load_model(self):
        """Loads OWLv2 processor and model."""
        if self.model_name == "stub":
            self.stub_mode = True
            return

        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        print(f"[OWLv2Detector] Loading '{self.model_name}' on {self.device} ({self.dtype})...")
        try:
            self.processor = Owlv2Processor.from_pretrained(self.model_name)
            self.model = Owlv2ForObjectDetection.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
            ).to(self.device).eval()
        except Exception as e:
            print(f"[OWLv2Detector][Warning] Could not load '{self.model_name}' ({e}). Activating stub mode.")
            self.stub_mode = True

    def _ensure_loaded(self):
        if self.processor is None or self.model is None:
            if not self.stub_mode:
                self._load_model()

    @torch.inference_mode()
    def batch_inference(
        self,
        image: Image.Image,
        text_prompts: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], List[int]]:
        """
        Runs OWLv2 on a single PIL image for text_prompts.
        Returns:
            Tuple of (backbone_out, processed_results, text_id)
        """
        if not text_prompts:
            return None, {}, []

        text_id = list(range(1, len(text_prompts) + 1))
        if self.model_name == "stub" or self.stub_mode:
            return None, {i: {"boxes": [], "scores": []} for i in text_id}, text_id

        self._ensure_loaded()
        w, h = image.size
        processed_results: Dict[int, Dict[str, Any]] = {i: {"boxes": [], "scores": []} for i in text_id}
        backbone_out = None

        try:
            # Hugging Face Owlv2Processor expects text as list of lists of queries for batched queries per image
            inputs = self.processor(text=[text_prompts], images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if "pixel_values" in inputs and self.dtype in (torch.float16, torch.bfloat16):
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            # 1. Extract ViT Patch Tokens & Reshape to (1, C, H_f, W_f) for Stage 3 SGAP
            try:
                vision_outputs = self.model.owlv2.vision_model(pixel_values=inputs["pixel_values"])
                patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]  # Exclude [CLS]
                B_tok, N_tok, C_tok = patch_tokens.shape
                grid_size = int(math.isqrt(N_tok))
                if grid_size > 0:
                    feat = patch_tokens[:, :grid_size * grid_size, :].permute(0, 2, 1).reshape(B_tok, C_tok, grid_size, grid_size)
                    # The processor pads to a square at the bottom/right; keep only the image region
                    valid_h = max(1, round(grid_size * h / max(w, h)))
                    valid_w = max(1, round(grid_size * w / max(w, h)))
                    backbone_out = {"vision_features": feat[:, :, :valid_h, :valid_w]}
            except Exception as e:
                backbone_out = None

            # 2. Extract Detections
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([[h, w]], device=self.device)
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=self.box_threshold
            )[0]

            boxes_t = results.get("boxes", [])
            scores_t = results.get("scores", [])
            labels_t = results.get("labels", [])

            boxes = boxes_t.detach().cpu().tolist() if hasattr(boxes_t, "tolist") else list(boxes_t)
            scores = scores_t.detach().cpu().tolist() if hasattr(scores_t, "tolist") else list(scores_t)
            labels = labels_t.detach().cpu().tolist() if hasattr(labels_t, "tolist") else list(labels_t)

            for box, score, label_idx in zip(boxes, scores, labels):
                l_idx = int(label_idx)
                # Map 0-based label index to 1-based text_id
                tid = l_idx + 1
                if tid in processed_results:
                    x1 = max(0, min(w, int(round(float(box[0])))))
                    y1 = max(0, min(h, int(round(float(box[1])))))
                    x2 = max(0, min(w, int(round(float(box[2])))))
                    y2 = max(0, min(h, int(round(float(box[3])))))
                    if x2 > x1 and y2 > y1:
                        processed_results[tid]["boxes"].append([x1, y1, x2, y2])
                        processed_results[tid]["scores"].append(float(score))

        except Exception as e:
            print(f"[OWLv2Detector][Warning] Inference failed ({e}). Returning empty detections.")

        return backbone_out, processed_results, text_id

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
