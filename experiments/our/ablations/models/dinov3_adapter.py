"""
DINOv3 (facebook/dinov3-vitl16-pretrain-lvd1689m) detector/backbone adapter
for MSSearch ablation suite.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torch
from .base import BaseDetector


class DINOv3Detector(BaseDetector):
    """
    Self-supervised vision transformer adapter using DINOv3-ViT-L/16.
    Extracts dense visual patch representations and generates saliency-guided
    region proposals for open-set visual queries.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: str = "cuda:0",
        torch_dtype: Optional[torch.dtype] = None,
        box_threshold: float = 0.20,
        token: Optional[str] = None,
        **kwargs,
    ):
        self.model_path = model_name_or_path or os.environ.get(
            "DINOV3_PATH", "facebook/dinov3-vitl16-pretrain-lvd1689m"
        )
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch_dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        self.box_threshold = box_threshold
        self.token = token or os.environ.get("HF_TOKEN", None)

        self.processor = None
        self.model = None
        self.stub_mode = False
        self._init_backend()

    def _init_backend(self):
        """Loads DINOv3 model or activates stub mode on missing gated credentials."""
        if self.model_path == "stub":
            self.stub_mode = True
            return

        try:
            from transformers import AutoImageProcessor, AutoModel
            print(f"[DINOv3Detector] Loading '{self.model_path}' on {self.device}...")
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_path,
                token=self.token,
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                token=self.token,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            ).to(self.device).eval()
        except Exception as e:
            print(f"[DINOv3Detector][Warning] Could not load weights ({e}). Stub proposal mode active.")
            self.stub_mode = True

    @torch.inference_mode()
    def batch_inference(
        self,
        image: Image.Image,
        text_prompts: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], List[int]]:
        """
        Runs DINOv3 ViT-L/16 feature extraction and derives region proposals.
        Returns:
            Tuple of (backbone_out, processed_results, text_id)
        """
        text_ids = list(range(1, len(text_prompts) + 1))
        w, h = image.size

        if self.stub_mode or self.model is None or self.processor is None:
            processed_results: Dict[int, Dict[str, Any]] = {
                tid: {"boxes": [], "scores": [], "labels": []} for tid in text_ids
            }
            return None, processed_results, text_ids

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        # Dense patch representations for Stage 2/3 feature reuse
        last_hidden = outputs.last_hidden_state  # [1, num_tokens, D]
        backbone_out = {"vision_features": last_hidden}

        # Derive region proposals from ViT patch activation energy
        # Exclude CLS token if present (num_patches = num_tokens - 1 or num_tokens)
        tokens = last_hidden[0]
        if tokens.size(0) > 1:
            patch_tokens = tokens[1:] if tokens.size(0) % 2 != 0 else tokens
        else:
            patch_tokens = tokens

        energy = patch_tokens.norm(dim=-1).float()  # [N_patches]
        energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)

        side = int(energy.size(0) ** 0.5)
        boxes: List[List[float]] = []
        scores: List[float] = []

        if side > 1:
            grid = energy[: side * side].view(side, side)
            peak_mask = grid > self.box_threshold
            y_indices, x_indices = torch.where(peak_mask)

            if len(x_indices) > 0:
                # Group connected high-energy patches into bounding boxes
                x_min = float(x_indices.min().item()) / side * w
                x_max = float(x_indices.max().item() + 1) / side * w
                y_min = float(y_indices.min().item()) / side * h
                y_max = float(y_indices.max().item() + 1) / side * h
                mean_score = float(grid[peak_mask].mean().item())

                boxes.append([x_min, y_min, x_max, y_max])
                scores.append(round(mean_score, 4))

        # Replicate candidate proposals across active prompt indices
        processed_results = {}
        for idx, tid in enumerate(text_ids):
            label = text_prompts[idx] if idx < len(text_prompts) else "object"
            processed_results[tid] = {
                "boxes": [list(b) for b in boxes],
                "scores": list(scores),
                "labels": [label] * len(boxes),
            }

        return backbone_out, processed_results, text_ids
