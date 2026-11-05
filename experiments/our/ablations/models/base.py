"""
Base adapter interfaces for MSSearch ablation studies.
Defines contracts for interchangeable Detection and Visual Screening backbones.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image


class BaseDetector(ABC):
    """Abstract contract for open-vocabulary detection models in MSSearch (replacing SAM3)."""

    @abstractmethod
    def batch_inference(
        self,
        image: Image.Image,
        text_prompts: List[str],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], List[int]]:
        """
        Runs open-vocabulary detection on a single PIL image.

        Args:
            image: Candidate PIL image or cropped sub-patch.
            text_prompts: List of target object labels, e.g. ['car', 'person'].

        Returns:
            Tuple of (backbone_out, processed_results, text_id):
            - backbone_out: Optional Dict containing 'vision_features' or None.
            - processed_results: Dict mapping 1-based text_id to:
                {
                    'boxes': List of [x1, y1, x2, y2] in pixel coords [0, W] and [0, H],
                    'scores': List of float confidence scores in [0.0, 1.0]
                }
            - text_id: List of integer IDs [1, 2, ..., len(text_prompts)].
        """
        pass

    def batch_inference_multi_images(
        self,
        images: List[Image.Image],
        text_prompts: List[str],
        batch_size: int = 4,
    ) -> List[Tuple[Optional[Dict[str, Any]], Dict[int, Dict[str, Any]], List[int]]]:
        """
        Batched inference across multiple candidate crops.
        Default implementation iterates over images; subclasses may optimize with GPU collation.
        """
        results = []
        for img in images:
            try:
                results.append(self.batch_inference(img, text_prompts))
            except Exception:
                # ponytail: safe fallback to empty detections on crop failure
                results.append((None, {}, list(range(1, len(text_prompts) + 1))))
        return results


class BaseVisualScorer(ABC):
    """Abstract contract for zero-shot text-image scoring models in MSSearch (replacing SigLIP)."""

    @abstractmethod
    def score_crops(
        self,
        crops: List[Image.Image],
        text_queries: List[str],
        batch_size: int = 32,
    ) -> List[float]:
        """
        Computes maximum text-image alignment probability across active object queries.

        Args:
            crops: List of PIL images (candidate sub-patches or crops).
            text_queries: List of active target queries, e.g. ['car', 'license plate'].
            batch_size: Inference mini-batch size.

        Returns:
            List of float scores strictly bounded in [0.0, 1.0], one per crop.
        """
        pass

    def score_single(self, crop: Image.Image, text_queries: List[str]) -> float:
        """Helper to score a single crop against text queries."""
        scores = self.score_crops([crop], text_queries)
        return scores[0] if scores else 0.0

    def compute_similarity(self, image: Image.Image, text_query: Union[str, List[str]]) -> float:
        """Direct scalar alignment interface."""
        queries = [text_query] if isinstance(text_query, str) else list(text_query)
        return self.score_single(image, queries)
