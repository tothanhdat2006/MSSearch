"""
Lightweight SigLIP / CLIP Scorer for Branch B (Low-Resolution Screening).
Runs on GPU 0 with fp16 precision to preserve memory headroom alongside Qwen2.5-VL.
"""

import os
import warnings
import logging
from typing import List, Union, Optional, Any, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Suppress harmless upstream config token ID mismatch warnings and dtype deprecation
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*bos_token_id.*")
warnings.filterwarnings("ignore", message=r".*eos_token_id.*")
warnings.filterwarnings("ignore", message=r".*`torch_dtype` is deprecated.*")


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
    if isinstance(output, (tuple, list)) and len(output) > 0:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Cannot extract tensor embedding from output of type {type(output).__name__}")


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
        self._model = None
        self._processor = None
        self.is_siglip = "siglip" in model_name.lower()

    @property
    def model(self):
        if self._model is None:
            self._ensure_loaded()
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def processor(self):
        if self._processor is None:
            self._ensure_loaded()
        return self._processor

    @processor.setter
    def processor(self, value):
        self._processor = value

    def _ensure_loaded(self):
        """Lazy loader to prevent unneeded GPU allocation until invoked."""
        if self._model is None:
            print(f"[SigLIPScorer] Loading '{self.model_name}' on {self.device} ({self.dtype})...")
            # Mute configuration warnings during checkpoint config ingestion
            logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

            offline_env = (
                os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
                or os.environ.get("HF_HUB_OFFLINE", "0") == "1"
            )
            local_files_only = os.path.exists(self.model_name) or offline_env

            if self.is_siglip:
                try:
                    from transformers import SiglipModel, SiglipProcessor, AutoConfig
                    try:
                        cfg = AutoConfig.from_pretrained(self.model_name, local_files_only=local_files_only)
                    except Exception:
                        cfg = AutoConfig.from_pretrained(self.model_name, local_files_only=False)
                        local_files_only = False

                    if hasattr(cfg, "text_config"):
                        cfg.text_config.bos_token_id = None
                        cfg.text_config.eos_token_id = None
                    cfg.bos_token_id = None
                    cfg.eos_token_id = None

                    try:
                        loaded_model = SiglipModel.from_pretrained(
                            self.model_name,
                            config=cfg,
                            dtype=self.dtype,
                            low_cpu_mem_usage=True,
                            local_files_only=local_files_only,
                        ).to(self.device).eval()
                    except (TypeError, ValueError, Exception):
                        try:
                            loaded_model = SiglipModel.from_pretrained(
                                self.model_name,
                                config=cfg,
                                torch_dtype=self.dtype,
                                low_cpu_mem_usage=True,
                                local_files_only=local_files_only,
                            ).to(self.device).eval()
                        except Exception:
                            loaded_model = SiglipModel.from_pretrained(
                                self.model_name,
                                config=cfg,
                                torch_dtype=self.dtype,
                                low_cpu_mem_usage=True,
                                local_files_only=False,
                            ).to(self.device).eval()

                    try:
                        self._processor = SiglipProcessor.from_pretrained(self.model_name, local_files_only=local_files_only)
                    except Exception:
                        self._processor = SiglipProcessor.from_pretrained(self.model_name, local_files_only=False)
                    self._model = loaded_model
                except Exception:
                    from transformers import AutoModel, AutoProcessor
                    try:
                        loaded_model = AutoModel.from_pretrained(
                            self.model_name,
                            dtype=self.dtype,
                            low_cpu_mem_usage=True,
                            local_files_only=local_files_only,
                        ).to(self.device).eval()
                    except Exception:
                        loaded_model = AutoModel.from_pretrained(
                            self.model_name,
                            torch_dtype=self.dtype,
                            low_cpu_mem_usage=True,
                            local_files_only=False,
                        ).to(self.device).eval()

                    try:
                        self._processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=local_files_only)
                    except Exception:
                        self._processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=False)
                    self._model = loaded_model
            else:
                from transformers import CLIPModel, CLIPProcessor
                try:
                    loaded_model = CLIPModel.from_pretrained(
                        self.model_name,
                        dtype=self.dtype,
                        low_cpu_mem_usage=True,
                        local_files_only=local_files_only,
                    ).to(self.device).eval()
                except Exception:
                    loaded_model = CLIPModel.from_pretrained(
                        self.model_name,
                        torch_dtype=self.dtype,
                        low_cpu_mem_usage=True,
                        local_files_only=False,
                    ).to(self.device).eval()

                try:
                    self._processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=local_files_only)
                except Exception:
                    self._processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=False)
                self._model = loaded_model

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

        # Format prompt with descriptive templates to avoid zero-shot vocabulary shift
        prompts = [
            q if q.startswith(("a ", "an ", "the ", "photo of")) else f"a photo of {q}"
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

        # 2. Dense Spatial Token-Max Matching (prevents micro-targets from being diluted)
        dense_sim = None
        try:
            vision_model = getattr(self.model, "vision_model", None)
            if vision_model is not None:
                vis_out = vision_model(pixel_values=img_inputs["pixel_values"])
                if hasattr(vis_out, "last_hidden_state") and vis_out.last_hidden_state is not None:
                    spatial_tokens = vis_out.last_hidden_state.float()  # [num_crops, num_tokens, D_vis]
                    if hasattr(self.model, "visual_projection") and self.model.visual_projection is not None:
                        spatial_tokens = self.model.visual_projection(spatial_tokens)
                    if spatial_tokens.shape[-1] == text_features.shape[-1]:
                        spatial_tokens = F.normalize(spatial_tokens, dim=-1)
                        # [num_crops, num_tokens, num_queries] -> max over tokens -> [num_crops, num_queries]
                        token_sims = torch.einsum("btd,qd->btq", spatial_tokens, text_features)
                        dense_sim = token_sims.max(dim=1).values
        except Exception:
            dense_sim = None

        if dense_sim is not None:
            # Fuse global gestalt and peak local token alignment
            sim_matrix = torch.maximum(pooled_sim, dense_sim)
        else:
            sim_matrix = pooled_sim

        # 3. Dynamic Range Calibration (maps raw contrastive range [0.02, 0.25] to [0.0, 1.0])
        # In zero-shot vision-language space, raw orthogonal cross-modal cos is ~0.00-0.03 for noise,
        # and ~0.15-0.30 for salient semantic matches.
        raw_max = sim_matrix.max(dim=1).values  # [num_crops]
        calibrated_scores = torch.clamp((raw_max - 0.02) / 0.22, 0.0, 1.0)
        return [float(min(1.0, max(0.0, s))) for s in calibrated_scores.cpu().tolist()]

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

    @torch.inference_mode()
    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """Extracts L2-normalized image embedding(s) as float32 numpy array."""
        self._ensure_loaded()
        single = isinstance(images, Image.Image)
        img_list = [images] if single else images
        if not img_list:
            return np.empty((0, 768), dtype=np.float32)

        inputs = self.processor(images=img_list, return_tensors="pt").to(self.device)
        if self.dtype == torch.float16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        if self.is_siglip:
            feats = self.model.get_image_features(**inputs)
        else:
            feats = self.model.get_image_features(inputs["pixel_values"])

        feats = _extract_embedding_tensor(feats)
        feats = F.normalize(feats.float(), dim=-1).cpu().numpy().astype(np.float32)
        return feats[0] if single else feats

    @torch.inference_mode()
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Extracts L2-normalized text embedding(s) as float32 numpy array."""
        self._ensure_loaded()
        single = isinstance(texts, str)
        text_list = [texts] if single else texts
        if not text_list:
            return np.empty((0, 768), dtype=np.float32)

        inputs = self.processor(text=text_list, padding=True, return_tensors="pt").to(self.device)
        if self.is_siglip:
            feats = self.model.get_text_features(**inputs)
        else:
            feats = self.model.get_text_features(inputs["input_ids"])

        feats = _extract_embedding_tensor(feats)
        feats = F.normalize(feats.float(), dim=-1).cpu().numpy().astype(np.float32)
        return feats[0] if single else feats

    @torch.inference_mode()
    def extract_dense_spatial_token_saliency(
        self,
        crop_img: Image.Image,
        text_queries: List[str],
        top_k_tokens: int = 3,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Extracts unpooled spatial ViT token grid similarities against text_queries.
        Returns:
            s_dense: Top-k localized spatial token cosine peak concentration
            s_pool: Global average-pooled cosine similarity
            token_heatmap: 2D spatial similarity map (e.g. 24x24)
        """
        self._ensure_loaded()
        if not text_queries:
            return 0.0, 0.0, np.zeros((24, 24), dtype=np.float32)

        prompts = [
            q if q.startswith(("a ", "an ", "the ", "photo of")) else f"a photo of {q}"
            for q in text_queries
        ]

        inputs = self.processor(
            text=prompts,
            images=crop_img,
            return_tensors="pt",
            padding="max_length",
            max_length=64
        ).to(self.device)

        if self.dtype == torch.float16 and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        # 1. Text embedding extraction
        if self.is_siglip:
            text_features = self.model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        else:
            text_features = self.model.get_text_features(inputs.input_ids)
        text_features = _extract_embedding_tensor(text_features)
        text_features = F.normalize(text_features.float(), dim=-1)

        # 2. Vision forward
        vision_model = getattr(self.model, "vision_model", None)
        if vision_model is None:
            img_feats = self.model.get_image_features(pixel_values=inputs.pixel_values)
            img_feats = F.normalize(_extract_embedding_tensor(img_feats).float(), dim=-1)
            sim = torch.matmul(img_feats, text_features.t()).squeeze(0).max().item()
            return float(sim), float(sim), np.full((24, 24), sim, dtype=np.float32)

        vis_out = vision_model(pixel_values=inputs.pixel_values, output_hidden_states=True)
        spatial_tokens = vis_out.last_hidden_state.float()
        if hasattr(self.model, "visual_projection") and self.model.visual_projection is not None:
            spatial_tokens = self.model.visual_projection(spatial_tokens)
        spatial_tokens = F.normalize(spatial_tokens, dim=-1)

        # 3. Global pooled vector
        if hasattr(self.model, "get_image_features"):
            pooled_vision = self.model.get_image_features(pixel_values=inputs.pixel_values)
            pooled_vision = _extract_embedding_tensor(pooled_vision)
        else:
            pooled_vision = getattr(vis_out, "pooler_output", None)
            if pooled_vision is None:
                pooled_vision = spatial_tokens.mean(dim=1)
        pooled_vision = F.normalize(pooled_vision.float(), dim=-1)

        pool_sim = torch.matmul(pooled_vision, text_features.t()).squeeze(0)
        s_pool = float(pool_sim.max().item())

        # 4. Spatial token grid
        spatial_sim = torch.matmul(spatial_tokens.squeeze(0), text_features.t())
        max_spatial_sim, _ = spatial_sim.max(dim=-1)

        num_tokens = max_spatial_sim.shape[0]
        grid_dim = int(np.round(np.sqrt(num_tokens)))
        token_grid = max_spatial_sim.view(grid_dim, grid_dim).cpu().float().numpy()

        flat_sorted = np.sort(max_spatial_sim.cpu().float().numpy())[::-1]
        s_dense = float(np.mean(flat_sorted[:top_k_tokens]))

        return s_dense, s_pool, token_grid



