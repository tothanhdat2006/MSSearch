"""
Stage 5: Unified Candidate Pool Aggregation.
Converts all surviving, ranked candidate patches from Stage 4 directly into CandidateCrop
objects ready for Stage 6 Foveated VLM Reading.
Branch B (<= 336px) SigLIP re-screening is purged: all surviving patches are preserved directly.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from PIL import Image

from ..config import MSSearchConfig
from .stage3_partitioning import BoundedPatch


@dataclass
class CandidateCrop:
    """Represents a validated candidate crop in the Unified Candidate Pool P."""
    img_idx: int
    box: Tuple[int, int, int, int]
    crop_image: Image.Image
    score: float
    source_branch: str = "Direct"

    @property
    def confidence_score(self) -> float:
        """Alias for score to ensure backward compatibility."""
        return self.score


def run_resolution_screening(
    remaining_patches: List[BoundedPatch],
    candidate_images: List[Image.Image],
    active_objects: Optional[List[str]] = None,
    siglip_model = None,
    config: Optional[MSSearchConfig] = None
) -> List[CandidateCrop]:
    """
    Directly aggregates surviving candidate patches into the Unified Candidate Pool P
    without redundant Branch B SigLIP screening.
    """
    unified_pool: List[CandidateCrop] = []

    for patch in remaining_patches:
        box_ints = tuple(int(round(float(c))) for c in patch.box)
        crop = candidate_images[patch.img_idx].crop(box_ints)
        cw, ch = crop.size
        if cw < 4 or ch < 4:
            continue

        unified_pool.append(
            CandidateCrop(
                img_idx=patch.img_idx,
                box=patch.box,
                crop_image=crop,
                score=1.0,
                source_branch="Direct"
            )
        )

    if config and getattr(config, "verbose", True):
        print(f"[Stage 5] Unified Candidate Pool P aggregated: {len(unified_pool)} total crops ready for foveated VLM reading (Branch B SigLIP screening purged).")

    return unified_pool
