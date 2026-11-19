"""
Abstract Base Classes and Interfaces for STASH-Search Models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from PIL import Image


class BaseDualEncoder(ABC):
    """Abstract interface for vision-language dual-encoders (e.g. SigLIP-2, CLIP)."""

    @abstractmethod
    def score_crops(
        self,
        crops: List[Image.Image],
        text_queries: List[str],
        batch_size: int = 32
    ) -> List[float]:
        """
        Computes maximum visual-semantic alignment probability S(p, O_Q)
        for each image patch across candidate query targets.

        Returns a list of float scores in [0.0, 1.0], one per crop.
        """
        pass


class BaseMLLM(ABC):
    """Abstract interface for Multimodal Large Language Models (e.g. Qwen, InternVL, LLaVA)."""

    @abstractmethod
    def generate_text(
        self,
        prompt_or_messages: Union[str, List[Dict[str, Any]]],
        max_new_tokens: int = 96
    ) -> str:
        """
        Pure text generation interface (zero visual image overhead).
        Used by Stage 1 Target Extraction for in-context semantic query parsing.
        """
        pass

    @abstractmethod
    def generate_response_batch(
        self,
        images: List[Image.Image],
        question: str,
        max_new_tokens: int = 512
    ) -> List[str]:
        """
        Batch evaluation of candidate native-resolution image crops.
        Evaluates each crop with the canonical prompt '<image>\\n{question}'.
        """
        pass
