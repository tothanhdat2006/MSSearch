"""
Ablation models and adapters for MSSearch.
"""

from typing import Optional
from .base import BaseDetector, BaseVisualScorer
from .sam3_adapter import SAM3Detector
from .grounding_dino_adapter import GroundingDinoDetector
from .florence2_adapter import Florence2Detector
from .clip_scorer_adapter import CLIPVisualScorer
from .eva02_scorer_adapter import EVA02VisualScorer

try:
    from mssearch.models.modeling_siglip import SigLIPScorer
    BaseVisualScorer.register(SigLIPScorer)
except ImportError:
    pass

__all__ = [
    "BaseDetector",
    "BaseVisualScorer",
    "SAM3Detector",
    "GroundingDinoDetector",
    "Florence2Detector",
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
    - 'florence2': Florence-2-large (HF: florence-community/Florence-2-large)
    """
    det_lower = str(detector_type).lower().strip().replace("-", "_")
    if det_lower in ("none", "null", "no", "false", ""):
        return None
    elif det_lower in ("sam3", "sam_3"):
        path = model_path_or_name or "/data/shared/cvpr/dat/models/sam3/sam3.pt"
        return SAM3Detector(model_path=path, device=device, **kwargs)
    elif det_lower in ("grounding_dino", "groundingdino", "gdino"):
        name = model_path_or_name or "IDEA-Research/grounding-dino-base"
        return GroundingDinoDetector(model_name_or_path=name, device=device, **kwargs)
    elif det_lower in ("florence2", "florence_2", "florence", "florence2_large"):
        name = model_path_or_name or "florence-community/Florence-2-large"
        return Florence2Detector(model_name_or_path=name, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown detector_type '{detector_type}'. Supported: 'sam3', 'grounding_dino', 'florence2'.")


def load_visual_scorer(
    scorer_type: str = "siglip",
    model_name_or_path: Optional[str] = None,
    device: str = "cuda:0",
    **kwargs,
) -> BaseVisualScorer:
    """
    Factory for zero-shot text-image screening scorers:
    - 'siglip': SigLIP baseline (google/siglip-base-patch16-384)
    - 'siglip2_384': SigLIP2-SO400M 384px (google/siglip2-so400m-patch16-384)
    - 'siglip2_512': SigLIP2-SO400M 512px (google/siglip2-so400m-patch16-512)
    - 'clip_b32': CLIP-ViT-B/32 (openai/clip-vit-base-patch32)
    - 'clip_l14': CLIP-ViT-L/14 (openai/clip-vit-large-patch14)
    - 'eva02_l14': EVA-02-L/14 (timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k)
    """
    sc_lower = str(scorer_type).lower().strip().replace("-", "_")
    if sc_lower in ("siglip2_384", "siglip2_so400m_384", "siglip2_so400m_patch16_384"):
        from mssearch.models.modeling_siglip import SigLIPScorer
        name = model_name_or_path or "google/siglip2-so400m-patch16-384"
        return SigLIPScorer(model_name=name, device=device, **kwargs)
    elif sc_lower in ("siglip2_512", "siglip2_so400m_512", "siglip2_so400m_patch16_512"):
        from mssearch.models.modeling_siglip import SigLIPScorer
        name = model_name_or_path or "google/siglip2-so400m-patch16-512"
        return SigLIPScorer(model_name=name, device=device, **kwargs)
    elif sc_lower in ("siglip_base", "siglip_base_patch16_384"):
        from mssearch.models.modeling_siglip import SigLIPScorer
        name = model_name_or_path or "google/siglip-base-patch16-384"
        return SigLIPScorer(model_name=name, device=device, **kwargs)
    elif "siglip" in sc_lower:
        from mssearch.models.modeling_siglip import SigLIPScorer
        name = model_name_or_path or "google/siglip2-so400m-patch16-512"
        return SigLIPScorer(model_name=name, device=device, **kwargs)
    elif sc_lower in ("clip_b32", "clip_vit_b32", "clip_b_32"):
        name = model_name_or_path or "openai/clip-vit-base-patch32"
        return CLIPVisualScorer(model_name=name, device=device, **kwargs)
    elif sc_lower in ("clip_l14", "clip_vit_l14", "clip_l_14", "clip"):
        name = model_name_or_path or "openai/clip-vit-large-patch14"
        return CLIPVisualScorer(model_name=name, device=device, **kwargs)
    elif sc_lower in ("eva02_l14", "eva02", "eva_02_l14", "eva_clip", "eva02_336"):
        name = model_name_or_path or "timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"
        return EVA02VisualScorer(model_name=name, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown scorer_type '{scorer_type}'. Supported: 'siglip', 'siglip2_384', 'siglip2_512', 'clip_b32', 'clip_l14', 'eva02_l14'.")
