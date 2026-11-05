"""
Ablation models and adapters for MSSearch.
"""

from typing import Optional
from .base import BaseDetector, BaseVisualScorer
from .sam3_adapter import SAM3Detector
from .grounding_dino_adapter import GroundingDinoDetector
from .dinox_adapter import DinoXDetector
from .clip_scorer_adapter import CLIPVisualScorer
from .eva02_scorer_adapter import EVA02VisualScorer

__all__ = [
    "BaseDetector",
    "BaseVisualScorer",
    "SAM3Detector",
    "GroundingDinoDetector",
    "DinoXDetector",
    "CLIPVisualScorer",
    "EVA02VisualScorer",
    "load_detector",
    "load_visual_scorer",
]


def load_detector(
    detector_type: str = "sam3",
    model_path_or_name: Optional[str] = None,
    device: str = "cuda:0",
    **kwargs,
) -> BaseDetector:
    """
    Factory for open-vocabulary detection engines:
    - 'sam3': SAM3 baseline (weights at sam3_path)
    - 'grounding_dino': Grounding DINO (HF: IDEA-Research/grounding-dino-base)
    - 'dinox': DINO-X (weights or API)
    """
    det_lower = str(detector_type).lower().strip()
    if det_lower in ("sam3", "sam_3"):
        path = model_path_or_name or "/data/shared/cvpr/dat/models/sam3/sam3.pt"
        return SAM3Detector(model_path=path, device=device, **kwargs)
    elif det_lower in ("grounding_dino", "groundingdino", "gdino"):
        name = model_path_or_name or "IDEA-Research/grounding-dino-base"
        return GroundingDinoDetector(model_name_or_path=name, device=device, **kwargs)
    elif det_lower in ("dinox", "dino_x", "dino-x"):
        name = model_path_or_name or ""
        return DinoXDetector(model_name_or_path=name, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown detector_type '{detector_type}'. Supported: 'sam3', 'grounding_dino', 'dinox'.")


def load_visual_scorer(
    scorer_type: str = "siglip",
    model_name_or_path: Optional[str] = None,
    device: str = "cuda:0",
    **kwargs,
) -> BaseVisualScorer:
    """
    Factory for zero-shot text-image screening scorers:
    - 'siglip': SigLIP baseline (google/siglip-base-patch16-384)
    - 'clip_b32': CLIP-ViT-B/32 (openai/clip-vit-base-patch32)
    - 'clip_l14': CLIP-ViT-L/14 (openai/clip-vit-large-patch14)
    - 'eva02_l14': EVA-02-L/14 (EVA02-L-14-336 or QuanSun/EVA-CLIP)
    """
    sc_lower = str(scorer_type).lower().strip().replace("-", "_")
    if "siglip" in sc_lower:
        from mssearch.models.modeling_siglip import SigLIPScorer
        name = model_name_or_path or "google/siglip-base-patch16-384"
        return SigLIPScorer(model_name=name, device=device, **kwargs)
    elif sc_lower in ("clip_b32", "clip_vit_b32", "clip_b_32"):
        name = model_name_or_path or "openai/clip-vit-base-patch32"
        return CLIPVisualScorer(model_name=name, device=device, **kwargs)
    elif sc_lower in ("clip_l14", "clip_vit_l14", "clip_l_14", "clip"):
        name = model_name_or_path or "openai/clip-vit-large-patch14"
        return CLIPVisualScorer(model_name=name, device=device, **kwargs)
    elif sc_lower in ("eva02_l14", "eva02", "eva_02_l14", "eva_clip"):
        name = model_name_or_path or "EVA02-L-14-336"
        return EVA02VisualScorer(model_name=name, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown scorer_type '{scorer_type}'. Supported: 'siglip', 'clip_b32', 'clip_l14', 'eva02_l14'.")
