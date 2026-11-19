"""
Configuration dataclass for STASH-Search.
Contains only active operational parameters aligned with the paper's specification.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import json
import os


@dataclass
class STASHConfig:
    """Operational parameters for the 4-stage STASH-Search pipeline."""

    # Stage 1: Target Extraction
    nlp_model_name: Optional[str] = "en_core_web_sm"

    # Stage 2: Query-Guided Adaptive Partitioning
    quadrant_overlap_ratio: float = 0.10             # \rho: 10% overlap margin along boundary cuts
    saliency_plateau_gain_threshold: float = 0.04    # \Delta\tau_S: minimum alignment gain to continue zoom
    min_plateau_size: int = 512                      # S_{floor}: termination scale floor (px)
    max_plateau_size: int = 1536                     # S_{ceil}: mandatory split scale ceiling (px)
    max_partition_depth: int = 4                     # D_{max}: maximum quadtree depth
    max_patches_per_image: int = 64                  # K_{partition}: retained candidate patches per image

    # Stage 3: Gallery-Level Visual Screening & Native-Resolution Inspection
    k: int = 20                                      # Base retention budget per image (B_{retain} = k * N)
    seam_overlap_threshold: float = 0.10             # \tau_{seam}: overlap threshold for atomic sibling re-fusion
    bbox_expansion_factor: float = 1.25              # \alpha_{expand}: contextual margin expansion factor
    vlm_batch_size: int = 4                          # B_{reader}: parallel reading batch size
    max_vlm_tokens: int = 512                        # T_{max}: maximum generated tokens per crop inspection

    # Stage 4: Source-Attributed Consensus (SAC)
    answer_dedup_similarity: float = 0.85            # \tau_{vote}: string pattern matching threshold

    # Models & Runtime Environment
    dual_encoder_name: str = "google/siglip2-so400m-patch16-512"
    mllm_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda:0"
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Converts configuration to standard dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "STASHConfig":
        """Builds configuration from dictionary, ignoring unrecognized extra keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save_json(self, path: str) -> None:
        """Serializes configuration to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "STASHConfig":
        """Loads configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
