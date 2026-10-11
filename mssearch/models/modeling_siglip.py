"""
Lightweight SigLIP / CLIP Scorer for Branch B (Low-Resolution Screening).
Runs on GPU 0 with fp16 precision to preserve memory headroom alongside Qwen2.5-VL.
"""

import os
from typing import List, Union, Optional
from PIL import Image
import torch
import torch.nn.functional as F


class SigLIPScorer:
    """
    SigLIP (or CLIP) inference wrapper for zero-shot text-image cosine similarity scoring.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-384",
        device: str = "cuda:0",
        torch_dtype=torch.float16,
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch_dtype if torch.cuda.is_available() else torch.float32
        self.model = None
        self.processor = None
        self.is_siglip = "siglip" in model_name.lower()

    def _ensure_loaded(self):
        """Lazy loader to prevent unneeded GPU allocation until Stage 5 is invoked."""
        if self.model is None:
            print(f"[SigLIPScorer] Loading '{self.model_name}' on {self.device} ({self.dtype})...")
            if self.is_siglip:
                from transformers import AutoModel, AutoProcessor
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True
                ).to(self.device).eval()
                self.processor = AutoProcessor.from_pretrained(self.model_name)
            else:
                from transformers import CLIPModel, CLIPProcessor
                self.model = CLIPModel.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True
                ).to(self.device).eval()
                self.processor = CLIPProcessor.from_pretrained(self.model_name)

    @torch.inference_mode()
    def score_crops(
        self,
        crops: List[Image.Image],
        text_queries: List[str]
    ) -> List[float]:
        """
        Computes the maximum cosine alignment score across the active object queries.
        
        Args:
            crops: List of PIL images (low-res candidate crops <= 336px).
            text_queries: List of active object strings (e.g. ['car', 'sign']).
            
        Returns:
            List of float scores in [0.0, 1.0], one per crop.
        """
        if not crops or not text_queries:
            return []

        self._ensure_loaded()

        # Format prompt according to model architecture
        prompts = [
            q if self.is_siglip else f"a photo of a {q}"
            for q in text_queries
        ]

        # Extract text embeddings: [num_queries, D]
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        if self.is_siglip:
            text_features = self.model.get_text_features(**text_inputs)
        else:
            text_features = self.model.get_text_features(text_inputs["input_ids"])

        text_features = F.normalize(text_features.float(), dim=-1)

        # Extract image embeddings: [num_crops, D]
        img_inputs = self.processor(
            images=crops,
            return_tensors="pt"
        ).to(self.device)

        if self.dtype == torch.float16 and "pixel_values" in img_inputs:
            img_inputs["pixel_values"] = img_inputs["pixel_values"].to(self.dtype)

        if self.is_siglip:
            image_features = self.model.get_image_features(**img_inputs)
        else:
            image_features = self.model.get_image_features(img_inputs["pixel_values"])

        image_features = F.normalize(image_features.float(), dim=-1)

        # Pairwise cosine similarities: [num_crops, num_queries]
        sim_matrix = torch.matmul(image_features, text_features.T)

        # For each crop, take the maximum alignment over all active objects
        max_scores = sim_matrix.max(dim=1).values.cpu().tolist()
        return [max(0.0, float(s)) for s in max_scores]

    def score_single(self, crop: Image.Image, text_queries: List[str]) -> float:
        """Helper to score a single crop."""
        scores = self.score_crops([crop], text_queries)
        return scores[0] if scores else 0.0
