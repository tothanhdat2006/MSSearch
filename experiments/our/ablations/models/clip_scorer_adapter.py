"""
CLIP Scorer adapter for MSSearch ablation suite (CLIP-ViT-B/32, CLIP-ViT-L/14).
Uses HuggingFace AutoModel / AutoProcessor with calibrated zero-shot screening.
"""

from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import torch
from .base import BaseVisualScorer


def format_clip_prompts(text_queries: List[str]) -> List[str]:
    """Prefixes prompt queries with natural language templates for standard CLIP."""
    prompts = []
    for q in text_queries:
        q_clean = q.strip()
        if q_clean.startswith(("a ", "an ", "the ", "photo of ")):
            prompts.append(q_clean)
        else:
            art = "an" if q_clean[0].lower() in "aeiou" else "a"
            prompts.append(f"a photo of {art} {q_clean}")
    return prompts


class CLIPVisualScorer(BaseVisualScorer):
    """
    Zero-shot text-image screening scorer for CLIP backbones:
    - CLIP-ViT-B/32 (openai/clip-vit-base-patch32)
    - CLIP-ViT-L/14 (openai/clip-vit-large-patch14)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda:0",
        torch_dtype: Optional[torch.dtype] = None,
        calib_center: float = 0.20,
        calib_scale: float = 12.0,
        **kwargs,
    ):
        self.model_name = kwargs.get("model_path", model_name)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.calib_center = calib_center
        self.calib_scale = calib_scale

        self.model = None
        self.processor = None
        self._text_embed_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    def _ensure_loaded(self):
        if self.model is None:
            from transformers import AutoModel, AutoProcessor
            print(f"[CLIPVisualScorer] Loading '{self.model_name}' on {self.device} ({self.dtype})...")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(self.model_name)

    def clear_text_cache(self):
        self._text_embed_cache.clear()

    def get_text_features(self, text_queries: List[str]) -> torch.Tensor:
        """Computes and caches L2-normalized text feature embeddings."""
        key = tuple(text_queries)
        if key in self._text_embed_cache:
            return self._text_embed_cache[key]

        prompts = format_clip_prompts(text_queries)
        text_inputs = self.processor(text=prompts, padding=True, return_tensors="pt")
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        text_embeds = self.model.get_text_features(**text_inputs)
        if hasattr(text_embeds, "pooler_output") and text_embeds.pooler_output is not None:
            text_embeds = text_embeds.pooler_output
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        self._text_embed_cache[key] = text_embeds
        return text_embeds

    @torch.inference_mode()
    def score_crops(
        self,
        crops: List[Image.Image],
        text_queries: List[str],
        batch_size: int = 32,
    ) -> List[float]:
        """
        Scores candidate image crops against text queries using calibrated CLIP cosine similarity.
        Returns: list of float scores in [0.0, 1.0].
        """
        if not crops or not text_queries:
            return []

        self._ensure_loaded()
        text_embeds = self.get_text_features(text_queries)
        scores: List[float] = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            img_inputs = self.processor(images=batch_crops, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to(self.device)
            if self.dtype == torch.float16:
                pixel_values = pixel_values.to(self.dtype)

            img_embeds = self.model.get_image_features(pixel_values=pixel_values)
            if hasattr(img_embeds, "pooler_output") and img_embeds.pooler_output is not None:
                img_embeds = img_embeds.pooler_output
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

            # Pure cosine similarity matrix: [B, Q]
            cos_sim = torch.matmul(img_embeds, text_embeds.t())
            max_sim = cos_sim.max(dim=-1).values
            scores.extend(max_sim.cpu().tolist())

        # ponytail: clamp bounds to [0.0, 1.0]
        return [float(min(1.0, max(0.0, s))) for s in scores]
