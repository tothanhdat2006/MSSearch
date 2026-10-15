"""
MSSearch: Multi-Source Visual Search Engine.
Refactored directly from CVSearch into a modular, ablation-ready Dual-Path architecture.

Features:
1. Multi-source image pool support (N=20 candidate 4K images with hidden K evidence cardinality).
2. Tool-Based Fast Path: instant canvas detection + VLM existence gating (>0.70) with visual & textual prompt flags.
3. Tri-factor spatial entropy-gated partitioning (Quadrants vs SAM3-SGAP [8, 16]) bounded to max(W, H) <= 1024px.
4. Zero-redundant feature reuse across Stage 2, Stage 3, and Stage 4.
5. Resolution-aware dual-branch screening (Branch A direct pass-through >336px; Branch B SigLIP <=336px).
6. Exhaustive foveated VLM reading with independent 1.25x padding and semantic answer deduplication.
"""

import os
import time
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

from .config import MSSearchConfig
from .stages import (
    parse_question_to_active_objects,
    run_tool_based_fast_path,
    partition_candidate_images,
    run_feature_grouping_and_gating,
    run_resolution_screening,
    run_foveated_vlm_reading,
    FastPathResult,
    FinalSearchResult
)


class MultiSourceSearchEngine:
    """
    Master coordinator for the Dual-Path Multi-Source Visual Search pipeline.
    """

    def __init__(
        self,
        sam_model=None,
        zoom_model=None,
        siglip_model=None,
        nlp_model=None,
        config: Optional[MSSearchConfig] = None
    ):
        self.sam_model = sam_model
        self.zoom_model = zoom_model
        self.siglip_model = siglip_model
        self.nlp_model = nlp_model
        self.config = config if config is not None else MSSearchConfig()

    def search(
        self,
        candidate_images: List[Image.Image],
        question: str,
        ic_examples: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-source visual search across N images.
        """
        t_start = time.perf_counter()
        if self.config.verbose:
            print(f"\n[MSSearch] ====================================================================")
            print(f"[MSSearch] Starting Multi-Source Visual Search across {len(candidate_images)} candidate images.")
            print(f"[MSSearch] Question: '{question}'")
            print(f"[MSSearch] Fast Path Enabled: {self.config.enable_fast_path} | Partition Mode: {self.config.partition_mode}")
            print(f"[MSSearch] ====================================================================")

        telemetry: Dict[str, Any] = {
            "num_candidate_images": len(candidate_images),
            "stages_executed": [],
            "timings_sec": {}
        }

        # ---------------------------------------------------------------------
        # Stage 1: Question Parsing & Active Object Set (A_q)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        active_objects = parse_question_to_active_objects(
            question=question,
            zoom_model=self.zoom_model,
            nlp_model=self.nlp_model,
            ic_examples=ic_examples,
            verbose=self.config.verbose
        )
        telemetry["timings_sec"]["stage1_parser"] = round(time.perf_counter() - t0, 3)
        telemetry["active_objects"] = active_objects
        telemetry["stages_executed"].append("stage1_parser")

        # ---------------------------------------------------------------------
        # Stage 2: Tool-Based Fast Path Canvas Check
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        fast_path_result: FastPathResult = run_tool_based_fast_path(
            candidate_images=candidate_images,
            active_objects=active_objects,
            question=question,
            sam_model=self.sam_model,
            zoom_model=self.zoom_model,
            config=self.config
        )
        telemetry["timings_sec"]["stage2_fast_path"] = round(time.perf_counter() - t0, 3)
        telemetry["stages_executed"].append("stage2_fast_path")

        if fast_path_result.triggered:
            # Early exit: answer confirmed in 1 step!
            telemetry["route"] = "tool_based_fast_path"
            telemetry["fast_path_confidence"] = fast_path_result.confidence
            telemetry["evidence_images"] = [fast_path_result.evidence_img_idx]
            telemetry["total_latency_sec"] = round(time.perf_counter() - t_start, 3)
            return {
                "answers": [fast_path_result.answer],
                "primary_answer": fast_path_result.answer,
                "evidence_images": [fast_path_result.evidence_img_idx],
                "extractions": [
                    {
                        "img_idx": fast_path_result.evidence_img_idx,
                        "box": list(fast_path_result.detected_boxes[0]) if fast_path_result.detected_boxes else [0, 0, 100, 100],
                        "expanded_box": list(fast_path_result.detected_boxes[0]) if fast_path_result.detected_boxes else [0, 0, 100, 100],
                        "raw_answer": fast_path_result.answer,
                        "source_branch": "Fast Path"
                    }
                ],
                "telemetry": telemetry
            }

        # ---------------------------------------------------------------------
        # Stage 3: Entropy-Gated Recursive Patch Partitioning & SAM3-SGAP
        # ---------------------------------------------------------------------
        telemetry["route"] = "visual_search_branch"
        t0 = time.perf_counter()
        # Reuses feature map extracted during Stage 2 canvas check
        feature_cache = fast_path_result.feature_cache if self.config.reuse_stage2_features else {}
        bounded_patches = partition_candidate_images(
            candidate_images=candidate_images,
            stage2_feature_cache=feature_cache,
            config=self.config
        )
        telemetry["timings_sec"]["stage3_partitioning"] = round(time.perf_counter() - t0, 3)
        telemetry["num_bounded_patches"] = len(bounded_patches)
        telemetry["stages_executed"].append("stage3_partitioning")

        # ---------------------------------------------------------------------
        # Stage 4: SAM3 Feature Grouping & Active Object Gating
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        remaining_patches, stage4_groups = run_feature_grouping_and_gating(
            bounded_patches=bounded_patches,
            candidate_images=candidate_images,
            active_objects=active_objects,
            sam_model=self.sam_model,
            config=self.config,
            detected_boxes=fast_path_result.detected_boxes,
            zoom_model=self.zoom_model,
            return_debug=True
        )
        telemetry["timings_sec"]["stage4_grouping_gating"] = round(time.perf_counter() - t0, 3)
        telemetry["num_remaining_patches"] = len(remaining_patches)
        telemetry["stage4_groups"] = stage4_groups
        telemetry["stage4_kept_groups"] = sum(1 for g in stage4_groups if g.get("status") == "KEPT")
        telemetry["stage4_pruned_groups"] = sum(1 for g in stage4_groups if g.get("status") == "PRUNED")
        telemetry["stages_executed"].append("stage4_grouping_gating")

        # ---------------------------------------------------------------------
        # Stage 5: Resolution-Aware Screening (Branch A vs Branch B)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        unified_pool = run_resolution_screening(
            remaining_patches=remaining_patches,
            candidate_images=candidate_images,
            active_objects=active_objects,
            siglip_model=self.siglip_model,
            config=self.config
        )
        telemetry["timings_sec"]["stage5_screening"] = round(time.perf_counter() - t0, 3)
        telemetry["num_unified_crops"] = len(unified_pool)
        telemetry["branch_a_count"] = sum(1 for c in unified_pool if "Branch A" in c.source_branch)
        telemetry["branch_b_count"] = sum(1 for c in unified_pool if "Branch B" in c.source_branch)
        telemetry["stages_executed"].append("stage5_screening")

        # ---------------------------------------------------------------------
        # Stage 6: Foveated VLM Reading with 1.25x Padding & Deduplication
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        final_result: FinalSearchResult = run_foveated_vlm_reading(
            unified_pool=unified_pool,
            candidate_images=candidate_images,
            question=question,
            zoom_model=self.zoom_model,
            config=self.config
        )
        telemetry["timings_sec"]["stage6_reading"] = round(time.perf_counter() - t0, 3)
        telemetry["evidence_images"] = final_result.evidence_image_indices
        telemetry["stage6_crops"] = getattr(final_result, "crop_records", [])
        telemetry["total_latency_sec"] = round(time.perf_counter() - t_start, 3)
        telemetry["stages_executed"].append("stage6_foveated_reading")

        # Prioritize informative answers over refusal / negative answers
        from .stages.stage6_foveated_reader import is_negative_or_unanswerable
        informative_answers = [
            a for a in final_result.answers if not is_negative_or_unanswerable(a)
        ]
        if informative_answers:
            primary_answer = informative_answers[0]
        elif final_result.answers:
            primary_answer = final_result.answers[0]
        else:
            primary_answer = "No answer found"

        return {
            "answers": final_result.answers,
            "primary_answer": primary_answer,
            "evidence_images": final_result.evidence_image_indices,
            "extractions": [
                {
                    "img_idx": e.img_idx,
                    "box": list(e.box),
                    "expanded_box": list(e.expanded_box),
                    "raw_answer": e.raw_answer,
                    "source_branch": e.source_branch
                }
                for e in final_result.extractions
            ],
            "telemetry": telemetry
        }


def get_mssearch_response(
    sam_model,
    zoom_model,
    siglip_model,
    nlp_model,
    annotation: dict,
    candidate_images: List[Image.Image],
    config: Optional[MSSearchConfig] = None,
    ic_examples: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Drop-in functional wrapper replacing CVSearch's get_cvsearch_response.
    """
    engine = MultiSourceSearchEngine(
        sam_model=sam_model,
        zoom_model=zoom_model,
        siglip_model=siglip_model,
        nlp_model=nlp_model,
        config=config
    )
    question = annotation.get('question', '')
    return engine.search(
        candidate_images=candidate_images,
        question=question,
        ic_examples=ic_examples
    )
