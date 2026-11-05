"""
EVA-02-L/14 Visual Scorer adapter for MSSearch ablation suite.
Supports open_clip, timm, and HuggingFace AutoModel backbones.
"""

import os
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import torch
from .base import BaseVisualScorer


class EVA02VisualScorer(BaseVisualScorer):
    """
    Zero-shot text-image screening scorer using EVA-02-L/14
    (e.g., EVA02-L-14-336 or QuanSun/EVA-CLIP).
    """

    def __init__(
        self,
        model_name: str = "timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k",
        pretrained: str = "merged2b_s6b_b61k",
        device: str = "cuda:0",
        torch_dtype: Optional[torch.dtype] = None,
        calib_center: float = 0.20,
        calib_scale: float = 12.0,
        **kwargs,
    ):
        self.model_name = kwargs.get("model_path", model_name)
        self.pretrained = pretrained
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch_dtype or (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.calib_center = calib_center
        self.calib_scale = calib_scale

        self.backend = None  # "open_clip" or "hf"
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._text_embed_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    def _ensure_loaded(self):
        if self.model is not None:
            return

        # 1. Try loading via open_clip
        try:
            import open_clip  # type: ignore
            clip_id = f"hf-hub:{self.model_name}" if "/" in self.model_name and not self.model_name.startswith("hf-hub:") and not os.path.isdir(self.model_name) else self.model_name
            print(f"[EVA02VisualScorer] Loading via open_clip: '{clip_id}' ({self.pretrained}) on {self.device}...")
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                clip_id,
                pretrained=self.pretrained,
                device=self.device,
            )
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(clip_id)
            self.backend = "open_clip"
            return
        except Exception as e_clip:
            print(f"[EVA02VisualScorer] open_clip unavailable or failed ({e_clip}); falling back to HuggingFace...")

        # 2. Try loading via HuggingFace AutoModel / AutoProcessor
        from transformers import AutoModel, AutoProcessor
        hf_repo = self.model_name if self.model_name else "timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"
        print(f"[EVA02VisualScorer] Loading via HuggingFace: '{hf_repo}' on {self.device}...")
        self.model = AutoModel.from_pretrained(
            hf_repo,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.preprocess = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)
        self.backend = "hf"

    def clear_text_cache(self):
        self._text_embed_cache.clear()

    def get_text_features(self, text_queries: List[str]) -> torch.Tensor:
        key = tuple(text_queries)
        if key in self._text_embed_cache:
            return self._text_embed_cache[key]

        prompts = [f"a photo of a {q.strip()}" if not q.strip().startswith("a ") else q.strip() for q in text_queries]

        if self.backend == "open_clip":
            text_tokens = self.tokenizer(prompts).to(self.device)
            with torch.inference_mode():
                text_embeds = self.model.encode_text(text_tokens)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        else:
            inputs = self.preprocess(text=prompts, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                text_embeds = self.model.get_text_features(**inputs)
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
        Computes maximum text-image alignment probability using EVA-02-L/14.
        Returns: list of float scores in [0.0, 1.0].
        """
        if not crops or not text_queries:
            return []

        self._ensure_loaded()
        text_embeds = self.get_text_features(text_queries)
        scores: List[float] = []

        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]

            if self.backend == "open_clip":
                img_tensors = torch.stack([self.preprocess(c) for c in batch_crops]).to(self.device)
                img_embeds = self.model.encode_image(img_tensors)
            else:
                inputs = self.preprocess(images=batch_crops, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                img_embeds = self.model.get_image_features(**inputs)
                if hasattr(img_embeds, "pooler_output") and img_embeds.pooler_output is not None:
                    img_embeds = img_embeds.pooler_output

            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

            cos_sim = torch.matmul(img_embeds, text_embeds.t())
            max_sim = cos_sim.max(dim=-1).values

            probs = torch.sigmoid((max_sim - self.calib_center) * self.calib_scale)
            scores.extend(probs.cpu().tolist())

        # ponytail: clamp bounds to [0.0, 1.0]
        return [float(min(1.0, max(0.0, s))) for s in scores]
