"""
STASH-Search: Source-Attributed Spatial-Heuristic Visual Search.
Minimal, production-ready implementation.
"""

from .config import STASHConfig
from .engine import STASHSearchEngine
from .models.base import BaseMLLM, BaseDualEncoder

try:
    from .models.dual_encoder import SigLIPDualEncoder
    from .models.modeling_qwenvl import ModelQwenVL, UnifiedMLLM
except ImportError:
    SigLIPDualEncoder = None
    ModelQwenVL = None
    UnifiedMLLM = None

__all__ = [
    "STASHConfig",
    "STASHSearchEngine",
    "BaseMLLM",
    "BaseDualEncoder",
    "SigLIPDualEncoder",
    "ModelQwenVL",
    "UnifiedMLLM",
]
