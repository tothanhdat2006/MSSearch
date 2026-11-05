"""Model adapters and backbone engines for MSSearch."""

from typing import Any

__all__ = [
    "ModelQwenVL",
    "ModelGlobalLocal",
    "ModelInternvl",
    "ModelYiVL",
    "ModelLlavaHR",
    "sam3_inference",
    "SigLIPScorer",
]

def __getattr__(name: str) -> Any:
    if name == "ModelQwenVL":
        from .modeling_qwenvl import ModelQwenVL
        return ModelQwenVL
    elif name == "ModelGlobalLocal":
        from .modeling_llava import ModelGlobalLocal
        return ModelGlobalLocal
    elif name == "ModelInternvl":
        from .modeling_internvl import ModelInternvl
        return ModelInternvl
    elif name == "ModelYiVL":
        from .modeling_yivl import ModelYiVL
        return ModelYiVL
    elif name == "ModelLlavaHR":
        from .modeling_llava_hr import ModelLlavaHR
        return ModelLlavaHR
    elif name == "sam3_inference":
        from .modeling_sam3 import sam3_inference
        return sam3_inference
    elif name == "SigLIPScorer":
        from .modeling_siglip import SigLIPScorer
        return SigLIPScorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
