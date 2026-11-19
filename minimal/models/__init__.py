"""
Model exports and factory functions for STASH-Search Minimal.
Matches mssearch/models conventions.
"""

from typing import Any
from .base import BaseDualEncoder, BaseMLLM

__all__ = [
    "BaseDualEncoder",
    "BaseMLLM",
    "load_mllm",
    "ModelQwenVL",
    "ModelGlobalLocal",
    "ModelInternvl",
    "SigLIPDualEncoder",
]


def __getattr__(name: str) -> Any:
    if name == "load_mllm":
        from .factory import load_mllm
        return load_mllm
    elif name in ("ModelQwenVL", "UnifiedMLLM"):
        from .modeling_qwenvl import ModelQwenVL
        return ModelQwenVL
    elif name == "ModelGlobalLocal":
        from .modeling_llava import ModelGlobalLocal
        return ModelGlobalLocal
    elif name == "ModelInternvl":
        from .modeling_internvl import ModelInternvl
        return ModelInternvl
    elif name == "SigLIPDualEncoder":
        from .dual_encoder import SigLIPDualEncoder
        return SigLIPDualEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
