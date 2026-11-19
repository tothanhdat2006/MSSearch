"""
SigLIP / CLIP Dual-Encoder Implementation for Stage 2 & 3 Visual Alignment Scoring.
"""

import os
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, AutoConfig

from .base import BaseDualEncoder


class SigLIPDualEncoder(BaseDualEncoder):
    """
    Zero-shot vision-language alignment scorer using official HuggingFace
    SigLIP-2 / CLIP pipelines with normalized feature dot-products.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch16-512",
        device: str = "cuda:0",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        if torch_dtype is not None:
            self.dtype = torch_dtype
        else:
            self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None
        self.is_siglip = "siglip" in self.model_name.lower()
        self._text_embed_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    def _ensure_loaded(self) -> None:
        """Lazy loader to prevent unnecessary GPU memory consumption until scoring is needed."""
        if self.model is None:
            hf_offline = os.environ.get("HF_HUB_OFFLINE", "0") in ("1", "true", "True") or \
                         os.environ.get("TRANSFORMERS_OFFLINE", "0") in ("1", "true", "True")
            local_files_only = hf_offline or os.path.isdir(str(self.model_name))

            cfg = AutoConfig.from_pretrained(self.model_name, local_files_only=local_files_only)
            if hasattr(cfg, "text_config"):
                cfg.text_config.bos_token_id = None
                cfg.text_config.eos_token_id = None
            cfg.bos_token_id = None
            cfg.eos_token_id = None

            self.model = AutoModel.from_pretrained(
                self.model_name,
                config=cfg,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                local_files_only=local_files_only,
            ).to(self.device).eval()

            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=local_files_only
            )

    def clear_text_cache(self) -> None:
        """Clears cached text embeddings."""
        self._text_embed_cache.clear()

    def _get_text_features(self, text_queries: List[str]) -> torch.Tensor:
        """Computes and caches normalized text embeddings for the active target set."""
        key = tuple(text_queries)
        if key in self._text_embed_cache:
            return self._text_embed_cache[key]

        prompts = [q.strip() for q in text_queries if q.strip()]
        if not self.is_siglip:
            # Natural language prefix for CLIP
            prompts = [
                f"a photo of {q}" if not q.startswith(("a ", "an ", "the ", "photo of ")) else q
                for q in prompts
            ]

        text_inputs = self.processor(
            text=prompts,
            padding="max_length" if self.is_siglip else True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        text_embeds = self.model.get_text_features(**text_inputs)
        if hasattr(text_embeds, "pooler_output") and text_embeds.pooler_output is not None:
            text_embeds = text_embeds.pooler_output
        elif hasattr(text_embeds, "last_hidden_state") and text_embeds.last_hidden_state is not None:
            text_embeds = text_embeds.last_hidden_state.mean(dim=1)

        # L2 normalize
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        self._text_embed_cache[key] = text_embeds
        return text_embeds

    @torch.inference_mode()
    def score_crops(
        self,
        crops: List[Image.Image],
        text_queries: List[str],
        batch_size: int = 32
    ) -> List[float]:
        """
        Computes maximum alignment score S(p, O_Q) = max_{o in O_Q} s(p, o)
        across crops for the given text queries.
        """
        if not crops or not text_queries:
            return []

        self._ensure_loaded()
        text_embeds = self._get_text_features(text_queries)  # [num_targets, D]

        logit_scale = self.model.logit_scale.exp() if hasattr(self.model, "logit_scale") else 1.0
        logit_bias = self.model.logit_bias if hasattr(self.model, "logit_bias") else 0.0

        scores: List[float] = []
        for i in range(0, len(crops), batch_size):
            batch_crops = crops[i : i + batch_size]
            img_inputs = self.processor(images=batch_crops, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"].to(self.device)
            if self.dtype == torch.float16 and pixel_values.dtype != torch.float16:
                pixel_values = pixel_values.to(self.dtype)

            img_embeds = self.model.get_image_features(pixel_values=pixel_values)
            if hasattr(img_embeds, "pooler_output") and img_embeds.pooler_output is not None:
                img_embeds = img_embeds.pooler_output
            elif hasattr(img_embeds, "last_hidden_state") and img_embeds.last_hidden_state is not None:
                img_embeds = img_embeds.last_hidden_state.mean(dim=1)

            # L2 normalize
            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)  # [B, D]

            logits = (img_embeds @ text_embeds.T) * logit_scale + logit_bias  # [B, num_targets]
            if self.is_siglip:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.clamp((logits + 1.0) / 2.0, 0.0, 1.0)

            max_probs, _ = probs.max(dim=-1)
            scores.extend(max_probs.cpu().tolist())

        return scores
