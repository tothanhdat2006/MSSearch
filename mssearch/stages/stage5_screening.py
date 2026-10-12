"""
Stage 5: Resolution-Aware Dual-Branch Screening & Unified Candidate Pool Aggregation.
Routes surviving patches:
  - Branch A (High-Res: max(W, H) > 336px): Direct Pass-Through into candidate pool.
  - Branch B (Low-Res:  max(W, H) <= 336px): Screened via SigLIP (score >= 0.50).
Aggregates all kept crops into Unified Candidate Pool P.
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
    source_branch: str  # "Branch A (Direct)" or "Branch B (SigLIP)"


def run_resolution_screening(
    remaining_patches: List[BoundedPatch],
    candidate_images: List[Image.Image],
    active_objects: List[str],
    siglip_model,
    config: MSSearchConfig
) -> List[CandidateCrop]:
    """
    Executes Resolution-Aware Screening and builds the Unified Candidate Pool P.
    """
    unified_pool: List[CandidateCrop] = []

    # Separate patches into Branch A and Branch B candidates
    branch_a_patches: List[BoundedPatch] = []
    branch_b_patches: List[BoundedPatch] = []
    branch_b_crops: List[Image.Image] = []

    for patch in remaining_patches:
        x1, y1, x2, y2 = patch.box
        pw = x2 - x1
        ph = y2 - y1
        longest_side = max(pw, ph)

        box_ints = tuple(int(round(float(c))) for c in patch.box)
        crop = candidate_images[patch.img_idx].crop(box_ints)
        cw, ch = crop.size
        if cw < 4 or ch < 4:
            continue

        if longest_side > config.branch_b_size_cutoff:
            # Branch A: Direct Pass-Through (> 336px)
            unified_pool.append(
                CandidateCrop(
                    img_idx=patch.img_idx,
                    box=patch.box,
                    crop_image=crop,
                    score=1.0,
                    source_branch="Branch A (Direct)"
                )
            )
            branch_a_patches.append(patch)
        else:
            # Branch B: Low-Resolution Candidate (<= 336px)
            branch_b_patches.append(patch)
            branch_b_crops.append(crop)

    if config.verbose:
        print(f"[Stage 5][Branch A Direct] {len(branch_a_patches)} patches with max side > {config.branch_b_size_cutoff}px passed through directly into Unified Candidate Pool.")

    # Process Branch B in batched SigLIP scoring if any low-res crops exist
    if branch_b_crops:
        used_fallback = False
        if siglip_model is not None:
            try:
                b_scores = siglip_model.score_crops(branch_b_crops, active_objects)
            except Exception as e:
                used_fallback = True
                b_scores = [1.0] * len(branch_b_crops)
                if config.verbose:
                    print(f"[Stage 5][Branch B Fallback] SigLIP batch inference failed ({e}); conservatively keeping all {len(branch_b_crops)} low-res crops.")
        else:
            used_fallback = True
            b_scores = [1.0] * len(branch_b_crops)
            if config.verbose:
                print(f"[Stage 5][Branch B Fallback] SigLIP model absent; conservatively keeping all {len(branch_b_crops)} low-res crops.")

        b_kept = 0
        b_discarded = 0
        for patch, crop, score in zip(branch_b_patches, branch_b_crops, b_scores):
            if score >= config.branch_b_score_threshold:
                unified_pool.append(
                    CandidateCrop(
                        img_idx=patch.img_idx,
                        box=patch.box,
                        crop_image=crop,
                        score=score,
                        source_branch="Branch B (SigLIP)"
                    )
                )
                b_kept += 1
            else:
                b_discarded += 1

        if config.verbose and not used_fallback:
            print(f"[Stage 5][Branch B Non-Fallback] SigLIP scored {len(branch_b_crops)} low-res crops: {b_kept} kept (score >= {config.branch_b_score_threshold:.2f}), {b_discarded} discarded.")

    if config.verbose:
        print(f"[Stage 5] Unified Candidate Pool P aggregated: {len(unified_pool)} total crops ready for foveated VLM reading.")

    return unified_pool

