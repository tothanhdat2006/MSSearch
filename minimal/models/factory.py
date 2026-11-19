"""
Model Factory for STASH-Search Minimal.
Dynamically dispatches and loads MLLM backbones matching mssearch/models conventions.
"""

from typing import Optional
from .base import BaseMLLM


def load_mllm(
    model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: str = "cuda:0",
    **kwargs
) -> BaseMLLM:
    """
    Factory function to instantiate MLLM backbones supported in STASH-Search:
    - Qwen3-VL-8B / Qwen3-VL-32B: ModelQwenVL
    - LLaVA-OneVision-7B: ModelGlobalLocal
    - InternVL3-8B: ModelInternvl
    """
    name_lower = str(model_name_or_path).lower()

    if "internvl" in name_lower:
        from .modeling_internvl import ModelInternvl
        print(f"[ModelFactory] Initializing ModelInternvl: '{model_name_or_path}' on {device}...")
        return ModelInternvl(model_path=model_name_or_path, device=device, **kwargs)
    elif "llava" in name_lower:
        from .modeling_llava import ModelGlobalLocal
        print(f"[ModelFactory] Initializing ModelGlobalLocal (LLaVA): '{model_name_or_path}' on {device}...")
        return ModelGlobalLocal(model_path=model_name_or_path, device=device, **kwargs)
    else:
        from .modeling_qwenvl import ModelQwenVL
        print(f"[ModelFactory] Initializing ModelQwenVL: '{model_name_or_path}' on {device}...")
        return ModelQwenVL(model_name_or_path=model_name_or_path, device=device, **kwargs)
