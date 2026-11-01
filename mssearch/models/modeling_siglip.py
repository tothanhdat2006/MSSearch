"""
Lightweight SigLIP / CLIP Scorer for Branch B (Low-Resolution Screening).
Runs on GPU 0 with fp16 precision to preserve memory headroom alongside Qwen2.5-VL.
"""

import warnings
import logging
from typing import List, Union, Any
from PIL import Image
import torch
import torch.nn.functional as F

# Suppress harmless upstream config token ID mismatch warnings and dtype deprecation
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*bos_token_id.*")
warnings.filterwarnings("ignore", message=r".*eos_token_id.*")
warnings.filterwarnings("ignore", message=r".*`torch_dtype` is deprecated.*")


def format_text_queries(text_queries: List[str], is_siglip: bool = True) -> List[str]:
    """
    Formats text queries: SigLIP performs best with raw tags; CLIP benefits from natural language prefixes.
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
    Extracts the feature embedding tensor from model outputs across HuggingFace transformer variants.
    Handles BaseModelOutputWithPooling, SiglipOutput, tuples, and raw torch.Tensor.
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
    SigLIP (or CLIP) inference wrapper for zero-shot text-image cosine similarity scoring.
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-384",
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
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
            logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
            if self.is_siglip:
                from transformers import SiglipModel, SiglipProcessor, AutoConfig
                cfg = AutoConfig.from_pretrained(self.model_name)
                if hasattr(cfg, "text_config"):
                    cfg.text_config.bos_token_id = None
                    cfg.text_config.eos_token_id = None
                cfg.bos_token_id = None
                cfg.eos_token_id = None

                self.model = SiglipModel.from_pretrained(
                    self.model_name,
                    config=cfg,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                ).to(self.device).eval()
                self.processor = SiglipProcessor.from_pretrained(self.model_name)
            else:
                from transformers import CLIPModel, CLIPProcessor
                self.model = CLIPModel.from_pretrained(
                    self.model_name,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
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

        prompts = format_text_queries(text_queries, is_siglip=self.is_siglip)

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

        text_features = _extract_embedding_tensor(text_features)
        text_features = F.normalize(text_features.float(), dim=-1)

        # Extract image embeddings: [num_crops, D]
        img_inputs = self.processor(
            images=crops,
            return_tensors="pt"
        ).to(self.device)

        if self.dtype == torch.float16 and "pixel_values" in img_inputs:
            img_inputs["pixel_values"] = img_inputs["pixel_values"].to(self.dtype)

        # 1. Global pooled feature extraction
        if self.is_siglip:
            image_features = self.model.get_image_features(**img_inputs)
        else:
            image_features = self.model.get_image_features(img_inputs["pixel_values"])

        image_features = _extract_embedding_tensor(image_features)
        image_features = F.normalize(image_features.float(), dim=-1)

        # Pairwise global pooled cosine similarities: [num_crops, num_queries]
        pooled_sim = torch.matmul(image_features, text_features.T)

        # 2. Dense Spatial Token-Max Matching (CLIP only, requires linear visual_projection layer)
        # SigLIP uses multihead attention pooling (MAP) without a standalone token projection layer.
        dense_sim = None
        if not self.is_siglip and hasattr(self.model, "visual_projection") and self.model.visual_projection is not None:
            vision_model = getattr(self.model, "vision_model", None)
            if vision_model is not None:
                vis_out = vision_model(pixel_values=img_inputs["pixel_values"])
                if hasattr(vis_out, "last_hidden_state") and vis_out.last_hidden_state is not None:
                    spatial_tokens = vis_out.last_hidden_state.float()
                    spatial_tokens = self.model.visual_projection(spatial_tokens)
                    if spatial_tokens.shape[-1] == text_features.shape[-1]:
                        spatial_tokens = F.normalize(spatial_tokens, dim=-1)
                        token_sims = torch.einsum("btd,qd->btq", spatial_tokens, text_features)
                        dense_sim = token_sims.max(dim=1).values

        if dense_sim is not None:
            sim_matrix = torch.maximum(pooled_sim, dense_sim)
        else:
            sim_matrix = pooled_sim

        # 3. Dynamic Range Calibration / SigLIP Native Sigmoid Probability
        if self.is_siglip and hasattr(self.model, "logit_scale") and hasattr(self.model, "logit_bias"):
            logits = sim_matrix * self.model.logit_scale.exp() + self.model.logit_bias
            probs = torch.sigmoid(logits)
            raw_max = probs.max(dim=1).values
        else:
            # CLIP fallback: maps raw contrastive range [0.02, 0.25] to [0.0, 1.0]
            raw_max = sim_matrix.max(dim=1).values
            raw_max = torch.clamp((raw_max - 0.02) / 0.22, 0.0, 1.0)
        return [float(min(1.0, max(0.0, s))) for s in raw_max.cpu().tolist()]

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
