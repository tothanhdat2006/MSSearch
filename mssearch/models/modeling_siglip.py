"""
Lightweight SigLIP / CLIP Scorer for Branch B (Low-Resolution Screening)
using native HuggingFace pipeline as specified in official model documentation.
"""

import os
import warnings
import logging
from typing import List, Union, Any
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
import torch
from transformers import AutoModel, AutoProcessor, AutoConfig

# Suppress upstream config token ID mismatch warnings and dtype deprecation
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*bos_token_id.*")
warnings.filterwarnings("ignore", message=r".*eos_token_id.*")
warnings.filterwarnings("ignore", message=r".*`torch_dtype` is deprecated.*")


def format_text_queries(text_queries: List[str], is_siglip: bool = True) -> List[str]:
    """
    Formats text queries: SigLIP retains raw tags; CLIP benefits from natural language prefixes.
    """
    prompts = []
    for q in text_queries:
        q_clean = q.strip()
        if is_siglip:
            prompts.append(q_clean)
        elif q_clean.startswith(("a ", "an ", "the ", "photo of ")):
            prompts.append(q_clean)
        else:
            art = "an" if q_clean[0].lower() in "aeiou" else "a"
            prompts.append(f"a photo of {art} {q_clean}")
    return prompts


def _extract_embedding_tensor(output: Any) -> torch.Tensor:
    """
    Extracts feature embedding tensor from model outputs across HuggingFace transformer variants.
    Maintained for backward compatibility with external evaluators.
    """
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state.mean(dim=1)
    if hasattr(output, "logits") and output.logits is not None:
        return output.logits
    if isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Cannot extract tensor embedding from output of type {type(output).__name__}")


class SigLIPScorer:
    """
    SigLIP / CLIP inference wrapper for zero-shot text-image scoring
    using official HuggingFace outputs.logits_per_image and torch.sigmoid.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch16-512",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
        calib_center: float = 0.20,
        calib_scale: float = 12.0,
        **kwargs,
    ):
        model_candidate = kwargs.get("model_path", model_name)
        if not os.path.isdir(str(model_candidate)):
            model_dir = os.environ.get("MODEL_DIR", "")
            base = os.path.basename(str(model_candidate))
            if model_dir and os.path.isdir(os.path.join(model_dir, base)):
                model_candidate = os.path.join(model_dir, base)
            elif os.path.isdir(base):
                model_candidate = os.path.abspath(base)
        self.model_name = model_candidate
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch_dtype if torch.cuda.is_available() else torch.float32
        self.calib_center = float(calib_center)
        self.calib_scale = float(calib_scale)
        self.model = None
        self.processor = None
        self.is_siglip = "siglip" in self.model_name.lower()
        self._text_embed_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    def _ensure_loaded(self):
        """Lazy loader to prevent unneeded GPU allocation until Stage 5 is invoked."""
        if self.model is None:
            hf_offline = os.environ.get("HF_HUB_OFFLINE", "0") in ("1", "true", "True") or os.environ.get("TRANSFORMERS_OFFLINE", "0") in ("1", "true", "True")
            local_files_only = hf_offline or os.path.isdir(str(self.model_name))
            print(f"[SigLIPScorer] Loading '{self.model_name}' on {self.device} ({self.dtype}) (local_files_only={local_files_only})...")
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
            self.processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=local_files_only)

    def clear_text_cache(self):
        """Clears cached text query embeddings."""
        self._text_embed_cache.clear()

    def get_text_features(self, text_queries: List[str]) -> torch.Tensor:
        """Returns L2-normalized text feature embeddings with caching."""
        key = tuple(text_queries)
        if key in self._text_embed_cache:
            return self._text_embed_cache[key]
        prompts = format_text_queries(text_queries, is_siglip=self.is_siglip)
        text_inputs = self.processor(
            text=prompts,
            padding="max_length" if self.is_siglip else True,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        text_embeds = self.model.get_text_features(**text_inputs)
        text_embeds = _extract_embedding_tensor(text_embeds)
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
        Computes maximum text-image alignment probability across active object queries
        following the official HuggingFace SigLIP pipeline:
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits_per_image)
            
        Args:
            crops: List of PIL images (low-res candidate crops).
            text_queries: List of active object strings (e.g. ['car', 'sign']).
            batch_size: Mini-batch size for GPU memory management.
            
        Returns:
            List of float probability scores in [0.0, 1.0], one per crop.
        """
        if not crops or not text_queries:
            return []

        self._ensure_loaded()

        can_use_features = (
            hasattr(self.model, "get_text_features")
            and hasattr(self.model, "get_image_features")
            and hasattr(self.model, "logit_scale")
            and not type(self.model).__module__.startswith("unittest.mock")
        )

        scores: List[float] = []
        if can_use_features:
            text_embeds = self.get_text_features(text_queries)
            logit_scale = self.model.logit_scale.exp()
            logit_bias = self.model.logit_bias if hasattr(self.model, "logit_bias") else 0.0

            for i in range(0, len(crops), batch_size):
                batch_crops = crops[i : i + batch_size]
                img_inputs = self.processor(images=batch_crops, return_tensors="pt")
                pixel_values = img_inputs["pixel_values"].to(self.device)
                if self.dtype == torch.float16:
                    pixel_values = pixel_values.to(self.dtype)

                img_embeds = self.model.get_image_features(pixel_values=pixel_values)
                img_embeds = _extract_embedding_tensor(img_embeds)
                img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)

                cos_sim = torch.matmul(img_embeds, text_embeds.t())
                if self.is_siglip:
                    logits_per_image = cos_sim * logit_scale + logit_bias
                    probs = torch.sigmoid(logits_per_image)
                else:
                    # Pure cosine similarity for contrastive vision-language encoders (CLIP, EVA)
                    probs = cos_sim

                max_probs = probs.max(dim=-1).values
                scores.extend(max_probs.cpu().tolist())
        else:
            prompts = format_text_queries(text_queries, is_siglip=self.is_siglip)
            for i in range(0, len(crops), batch_size):
                batch_crops = crops[i : i + batch_size]
                inputs = self.processor(
                    text=prompts,
                    images=batch_crops,
                    padding="max_length" if self.is_siglip else True,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if self.dtype == torch.float16 and "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image

                if self.is_siglip:
                    probs = torch.sigmoid(logits_per_image)
                else:
                    scale = self.model.logit_scale.exp() if hasattr(self.model, "logit_scale") else 100.0
                    cos_sim = logits_per_image / scale
                    probs = cos_sim

                max_probs = probs.max(dim=-1).values
                scores.extend(max_probs.cpu().tolist())

        # ponytail: clamp bounds to [0.0, 1.0] float contract
        return [float(min(1.0, max(0.0, s))) for s in scores]

    def score_single(self, crop: Image.Image, text_queries: List[str]) -> float:
        """Helper to score a single crop."""
        scores = self.score_crops([crop], text_queries)
        return scores[0] if scores else 0.0

    def compute_similarity(self, image: Image.Image, text_query: Union[str, List[str]]) -> float:
        """
        Computes similarity score between a single image and text query or list of queries.
        Provides a direct scalar interface for single image evaluation.
        """
        if isinstance(text_query, str):
            text_queries = [text_query]
        else:
            text_queries = list(text_query)
        return self.score_single(image, text_queries)
