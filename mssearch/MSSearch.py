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
    format_question_with_options,
    run_direct_global_assessment,
    DirectAnswerResult,
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
        options: Optional[Any] = None,
        cycle_idx: int = 0,
        ic_examples: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-source visual search across N images.
        """
        t_start = time.perf_counter()

        # Determine whether candidate options should be added to the VLM reading prompt
        # (Controlled by config.include_options_in_prompt or config.answer_type != "free_form")
        should_add_options = (
            getattr(self.config, "include_options_in_prompt", False) or
            getattr(self.config, "answer_type", "free_form") in ["option_single", "multiple_choice", "mc", "Multiple Choice"]
        )
        if should_add_options and options:
            reading_question = format_question_with_options(question, options, cycle_idx=cycle_idx)
            if self.config.verbose:
                print(f"[MSSearch] Options Prompting ACTIVE ({getattr(self.config, 'answer_type', 'option_single')}): Candidate options appended to reading prompt.")
        else:
            reading_question = question
            if self.config.verbose:
                print(f"[MSSearch] Free-form mode ACTIVE: Prompt will NOT contain candidate options.")

        if self.config.verbose:
            print(f"\n[MSSearch] ====================================================================")
            print(f"[MSSearch] Starting Multi-Source Visual Search across {len(candidate_images)} candidate images.")
            print(f"[MSSearch] Base Question: '{question}'")
            print(f"[MSSearch] Options Flag: include_options_in_prompt={should_add_options} (answer_type='{getattr(self.config, 'answer_type', 'free_form')}')")
            print(f"[MSSearch] Direct Answer Enabled: {getattr(self.config, 'enable_direct_answer', True)} (Threshold: {getattr(self.config, 'direct_answer_conf_threshold', 0.60)})")
            print(f"[MSSearch] Fast Path Enabled: {self.config.enable_fast_path} (Threshold: {self.config.fast_path_conf_threshold}) | Partition Mode: {self.config.partition_mode}")
            print(f"[MSSearch] Multi-Answer Mode: {getattr(self.config, 'multi_answer_mode', False)}")
            print(f"[MSSearch] ====================================================================")

        # ---------------------------------------------------------------------
        # Multi-Source Budget Scaling & Early Exit Configuration
        # ---------------------------------------------------------------------
        n_candidates = len(candidate_images)
        if getattr(self.config, "scale_budgets_by_num_images", True) and n_candidates > 1:
            base_sam3 = getattr(self.config, "max_sam3_eval_patches", 16)
            base_latent = getattr(self.config, "max_latent_evals", 8)
            base_retained = getattr(self.config, "max_retained_patches", 16)
            self.config.max_sam3_eval_patches = base_sam3 * n_candidates
            self.config.max_latent_evals = base_latent * n_candidates
            self.config.max_latent_eval_groups = self.config.max_latent_evals
            self.config.max_retained_patches = base_retained * n_candidates
            if self.config.verbose:
                print(f"[MSSearch] Scaled budgets by N={n_candidates}: max_sam3={self.config.max_sam3_eval_patches}, max_latent={self.config.max_latent_evals}, max_retained={self.config.max_retained_patches}")

        if getattr(self.config, "multi_answer_mode", False):
            self.config.enable_early_consensus_exit = False

        # Structured Query Parsing (target_attribute_type, condition_val, target_object)
        from .stages.stage1_parser import parse_visual_query_structure
        target_attr_type, cond_val, target_obj = parse_visual_query_structure(
            question, getattr(self.config, "query_meta", None)
        )

        telemetry: Dict[str, Any] = {
            "num_candidate_images": len(candidate_images),
            "stages_executed": [],
            "timings_sec": {},
            "include_options_in_prompt": should_add_options,
            "answer_type": getattr(self.config, "answer_type", "free_form"),
            "options_available": bool(options),
            "multi_answer_mode": getattr(self.config, "multi_answer_mode", False),
            "target_attribute_type": target_attr_type,
            "condition_val": cond_val,
            "target_object": target_obj
        }

        # ---------------------------------------------------------------------
        # Phase 1: Direct Global Answer Gate (Logit Sufficiency Probing)
        # ---------------------------------------------------------------------
        if getattr(self.config, "enable_direct_answer", True):
            t0 = time.perf_counter()
            direct_result: DirectAnswerResult = run_direct_global_assessment(
                candidate_images=candidate_images,
                question=reading_question,
                zoom_model=self.zoom_model,
                config=self.config
            )
            telemetry["timings_sec"]["phase1_direct_answer"] = round(time.perf_counter() - t0, 3)
            telemetry["stages_executed"].append("phase1_direct_answer")

            if direct_result.triggered:
                telemetry["route"] = "direct_global_answer"
                telemetry["direct_answer_confidence"] = direct_result.confidence
                ev_imgs = direct_result.evidence_images if direct_result.evidence_images else [direct_result.evidence_img_idx]
                telemetry["evidence_images"] = ev_imgs
                telemetry["total_latency_sec"] = round(time.perf_counter() - t_start, 3)
                stage6_patch_boxes = [
                    [0, 0, int(candidate_images[idx].size[0]), int(candidate_images[idx].size[1])]
                    for idx in ev_imgs
                ]
                telemetry["stage6_patch_boxes"] = stage6_patch_boxes
                ans_list = direct_result.answers if direct_result.answers else [direct_result.answer]
                from .stages.stage6_foveated_reader import split_attribute_answer
                pred_list = []
                for a in ans_list:
                    tokens = split_attribute_answer(
                        a,
                        target_attribute_type=target_attr_type,
                        condition_val=cond_val,
                        target_object=target_obj
                    )
                    for t in tokens:
                        if t.lower() not in [x.lower() for x in pred_list]:
                            pred_list.append(t)
                is_attr_mode = (getattr(self.config, "answer_type", "free_form") == "attribute_list")
                primary_ans = ", ".join(pred_list) if (is_attr_mode and pred_list) else direct_result.answer
                return {
                    "answers": ans_list,
                    "prediction_list": pred_list if pred_list else ans_list,
                    "primary_answer": primary_ans,
                    "raw_vlm_response": direct_result.raw_response,
                    "evidence_images": ev_imgs,
                    "stage6_patch_boxes": stage6_patch_boxes,
                    "extractions": [
                        {
                            "img_idx": idx,
                            "box": [0, 0, int(candidate_images[idx].size[0]), int(candidate_images[idx].size[1])],
                            "expanded_box": [0, 0, int(candidate_images[idx].size[0]), int(candidate_images[idx].size[1])],
                            "raw_answer": direct_result.raw_response,
                            "final_answer": direct_result.answer,
                            "source_branch": "Direct Global Answer"
                        }
                        for idx in ev_imgs
                    ],
                    "telemetry": telemetry
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
            verbose=self.config.verbose,
            image_pil=candidate_images[0] if candidate_images else None,
            config=self.config
        )
        telemetry["timings_sec"]["stage1_parser"] = round(time.perf_counter() - t0, 3)
        telemetry["active_objects"] = list(active_objects)
        telemetry["parsed_question"] = active_objects.to_dict() if hasattr(active_objects, "to_dict") else {}
        telemetry["stages_executed"].append("stage1_parser")

        # ---------------------------------------------------------------------
        # Stage 2: Tool-Based Fast Path Canvas Check
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        fast_path_result: FastPathResult = run_tool_based_fast_path(
            candidate_images=candidate_images,
            active_objects=active_objects,
            question=reading_question,
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
            ev_imgs = fast_path_result.evidence_images if fast_path_result.evidence_images else [fast_path_result.evidence_img_idx]
            telemetry["evidence_images"] = ev_imgs
            telemetry["total_latency_sec"] = round(time.perf_counter() - t_start, 3)
            stage6_patch_boxes = [list(map(int, b)) for b in fast_path_result.detected_boxes] if fast_path_result.detected_boxes else []
            telemetry["stage6_patch_boxes"] = stage6_patch_boxes
            ans_list = fast_path_result.answers if fast_path_result.answers else [fast_path_result.answer]
            from .stages.stage6_foveated_reader import split_attribute_answer
            pred_list = []
            for a in ans_list:
                tokens = split_attribute_answer(
                    a,
                    target_attribute_type=target_attr_type,
                    condition_val=cond_val,
                    target_object=target_obj
                )
                for t in tokens:
                    if t.lower() not in [x.lower() for x in pred_list]:
                        pred_list.append(t)
            is_attr_mode = (getattr(self.config, "answer_type", "free_form") == "attribute_list")
            primary_ans = ", ".join(pred_list) if (is_attr_mode and pred_list) else fast_path_result.answer
            return {
                "answers": ans_list,
                "prediction_list": pred_list if pred_list else ans_list,
                "primary_answer": primary_ans,
                "raw_vlm_response": fast_path_result.raw_response or fast_path_result.answer,
                "evidence_images": ev_imgs,
                "stage6_patch_boxes": stage6_patch_boxes,
                "extractions": [
                    {
                        "img_idx": ev_imgs[0] if ev_imgs else 0,
                        "box": list(fast_path_result.detected_boxes[0]) if fast_path_result.detected_boxes else [0, 0, 100, 100],
                        "expanded_box": list(fast_path_result.detected_boxes[0]) if fast_path_result.detected_boxes else [0, 0, 100, 100],
                        "raw_answer": fast_path_result.raw_response or fast_path_result.answer,
                        "final_answer": fast_path_result.answer,
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
            config=self.config,
            active_objects=active_objects,
            siglip_model=self.siglip_model
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
            return_debug=True,
            question=question
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
        is_relational = getattr(active_objects, "is_relational", None)
        question_type = getattr(active_objects, "question_type", None)
        final_result: FinalSearchResult = run_foveated_vlm_reading(
            unified_pool=unified_pool,
            candidate_images=candidate_images,
            question=reading_question,
            zoom_model=self.zoom_model,
            config=self.config,
            is_relational=is_relational,
            question_type=question_type
        )
        telemetry["timings_sec"]["stage6_reading"] = round(time.perf_counter() - t0, 3)
        telemetry["evidence_images"] = final_result.evidence_image_indices
        stage6_patch_boxes = [list(map(int, c.box)) for c in unified_pool]
        telemetry["stage6_patch_boxes"] = stage6_patch_boxes
        telemetry["stage6_crops"] = getattr(final_result, "crop_records", [])
        telemetry["total_latency_sec"] = round(time.perf_counter() - t_start, 3)
        telemetry["stages_executed"].append("stage6_foveated_reading")

        # Prioritize informative answers over refusal / negative answers
        from .stages.stage6_foveated_reader import is_negative_or_unanswerable, split_attribute_answer
        informative_answers = [
            a for a in final_result.answers if not is_negative_or_unanswerable(a)
        ]
        is_multi = getattr(self.config, "multi_answer_mode", False)
        is_attr_mode = (getattr(self.config, "answer_type", "free_form") == "attribute_list")

        if informative_answers:
            answers_out = informative_answers
        elif final_result.answers:
            answers_out = final_result.answers
        else:
            answers_out = []

        pred_list = []
        for a in answers_out:
            tokens = split_attribute_answer(
                a,
                target_attribute_type=target_attr_type,
                condition_val=cond_val,
                target_object=target_obj
            )
            for t in tokens:
                if t.lower() not in [x.lower() for x in pred_list]:
                    pred_list.append(t)

        if is_attr_mode and pred_list:
            primary_answer = ", ".join(pred_list)
        elif informative_answers:
            primary_answer = "; ".join(informative_answers) if is_multi else informative_answers[0]
        elif final_result.answers:
            primary_answer = "; ".join(final_result.answers) if is_multi else final_result.answers[0]
        else:
            primary_answer = "No answer found"

        # Find the raw VLM response corresponding to the primary answer
        raw_vlm_resp = None
        for e in final_result.extractions:
            cand_clean = e.final_answer if e.final_answer else e.raw_answer
            if cand_clean in informative_answers or cand_clean == primary_answer:
                raw_vlm_resp = e.raw_answer
                break
        if raw_vlm_resp is None and final_result.extractions:
            raw_vlm_resp = final_result.extractions[0].raw_answer

        return {
            "answers": answers_out,
            "prediction_list": pred_list if pred_list else answers_out,
            "primary_answer": primary_answer,
            "raw_vlm_response": raw_vlm_resp,
            "evidence_images": final_result.evidence_image_indices,
            "stage6_patch_boxes": stage6_patch_boxes,
            "extractions": [
                {
                    "img_idx": e.img_idx,
                    "box": list(e.box),
                    "expanded_box": list(e.expanded_box),
                    "raw_answer": e.raw_answer,
                    "final_answer": e.final_answer,
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
    options = annotation.get('options', None)
    cycle_idx = annotation.get('cycle_category', 0)
    try:
        cycle_idx = int(cycle_idx)
    except Exception:
        cycle_idx = 0

    # Respect annotation['answer_type'] if specified (aligned with CVSearch perform_CVSearch.py)
    if 'answer_type' in annotation and engine.config is not None:
        anno_type = annotation['answer_type']
        if anno_type in ["option_single", "multiple_choice", "mc", "Multiple Choice"]:
            engine.config.include_options_in_prompt = True
            engine.config.answer_type = anno_type
        elif anno_type == "free_form":
            engine.config.include_options_in_prompt = False
            engine.config.answer_type = "free_form"

    if 'query' in annotation and engine.config is not None:
        engine.config.query_meta = annotation['query']

    return engine.search(
        candidate_images=candidate_images,
        question=question,
        options=options,
        cycle_idx=cycle_idx,
        ic_examples=ic_examples
    )
