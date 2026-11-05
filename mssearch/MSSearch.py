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
from .stages.stage3_partitioning import BoundedPatch


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

        import copy

        # Determine logging modes
        is_quiet = getattr(self.config, "log_level", "summary") == "quiet" or not getattr(self.config, "verbose", True)
        is_summary = getattr(self.config, "stage_summary_only", True) and not is_quiet
        is_verbose = (getattr(self.config, "log_level", "summary") == "verbose") and not is_summary and not is_quiet

        # Pass a quiet config to sub-stages when stage_summary_only is active to suppress internal loop chatter
        sub_config = copy.copy(self.config)
        if is_summary:
            sub_config.verbose = False

        # Determine whether candidate options should be added to the VLM reading prompt
        # (Controlled by config.include_options_in_prompt or config.answer_type != "free_form")
        is_mc = getattr(self.config, "answer_type", "free_form") in [
            "option_single", "option_list", "multiple_choice", "mc", "Multiple Choice"
        ]
        should_add_options = (
            getattr(self.config, "include_options_in_prompt", False) or
            is_mc
        )
        if should_add_options and options:
            reading_question = format_question_with_options(question, options, cycle_idx=cycle_idx)
            if is_verbose:
                print(f"[MSSearch] Options Prompting ACTIVE ({getattr(self.config, 'answer_type', 'option_single')}): Candidate options appended to reading prompt.")
        else:
            reading_question = question
            if is_verbose:
                print(f"[MSSearch] Free-form mode ACTIVE: Prompt will NOT contain candidate options.")

        if is_verbose:
            print(f"\n[MSSearch] ====================================================================")
            print(f"[MSSearch] Starting Multi-Source Visual Search across {len(candidate_images)} candidate images.")
            print(f"[MSSearch] Base Question: '{question}'")
            print(f"[MSSearch] Options Flag: include_options_in_prompt={should_add_options} (answer_type='{getattr(self.config, 'answer_type', 'free_form')}')")
            print(f"[MSSearch] Direct Answer Enabled: {getattr(self.config, 'enable_direct_answer', True)} (Threshold: {getattr(self.config, 'direct_answer_conf_threshold', 0.60)})")
            print(f"[MSSearch] Fast Path Enabled: {self.config.enable_fast_path} (Threshold: {self.config.fast_path_conf_threshold}) | Partition Mode: {self.config.partition_mode}")
            print(f"[MSSearch] Multi-Answer Mode: {getattr(self.config, 'multi_answer_mode', False)}")
            print(f"[MSSearch] ====================================================================")
        elif not is_quiet:
            print(f"\n[MSSearch] Starting search across {len(candidate_images)} images | Q: '{question[:90]}{'...' if len(question)>90 else ''}'")

        # ---------------------------------------------------------------------
        # Multi-Source Budget Scaling & Early Exit Configuration
        # ---------------------------------------------------------------------
        n_candidates = len(candidate_images)
        if getattr(self.config, "enable_query_adaptive_halting", True) and n_candidates == 1:
            self.config.multi_answer_mode = False
            self.config.enable_early_consensus_exit = True
            sub_config.multi_answer_mode = False
            sub_config.enable_early_consensus_exit = True
            if is_verbose:
                print(f"[MSSearch] Query-Adaptive Halting: Single candidate image (N=1) -> multi_answer_mode=False, early consensus exit enabled.")

        if getattr(self.config, "scale_budgets_by_num_images", True) and n_candidates > 1:
            base_sam3 = getattr(self.config, "max_sam3_eval_patches", 64)
            base_latent = getattr(self.config, "max_latent_evals", 8)
            base_retained = getattr(self.config, "max_retained_patches", 32)
            sub_config.max_sam3_eval_patches = base_sam3 * n_candidates
            sub_config.max_latent_evals = base_latent * n_candidates
            sub_config.max_retained_patches = base_retained * n_candidates
            if is_verbose:
                print(f"[MSSearch] Scaled budgets by N={n_candidates}: max_sam3={sub_config.max_sam3_eval_patches}, max_latent={sub_config.max_latent_evals}, max_retained={sub_config.max_retained_patches}")

        if getattr(self.config, "multi_answer_mode", False):
            self.config.enable_early_consensus_exit = False
            sub_config.enable_early_consensus_exit = False

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
            "target_object": target_obj,
            "stage3_boxes": [],
            "stage4_boxes": [],
            "stage5_boxes": [],
        }

        # Tracking containers for the waterfall cascade
        harvested_answers: List[str] = []
        harvested_evidence_images: List[int] = []
        harvested_extractions: List[Dict[str, Any]] = []
        harvested_patch_boxes: List[List[int]] = []
        routes_executed: List[str] = []

        total_images = len(candidate_images)
        unresolved_images: List[int] = list(range(total_images))
        is_multi = getattr(self.config, "multi_answer_mode", False)

        # ---------------------------------------------------------------------
        # Phase 1: Direct Global Answer Gate (Logit Sufficiency Probing)
        # ---------------------------------------------------------------------
        if getattr(self.config, "enable_direct_answer", True):
            t0 = time.perf_counter()
            direct_result: DirectAnswerResult = run_direct_global_assessment(
                candidate_images=candidate_images,
                question=reading_question,
                zoom_model=self.zoom_model,
                config=sub_config
            )
            lat_p1 = round(time.perf_counter() - t0, 3)
            telemetry["timings_sec"]["phase1_direct_answer"] = lat_p1
            telemetry["stages_executed"].append("phase1_direct_answer")

            if direct_result.triggered:
                routes_executed.append("direct_global_answer")
                telemetry["direct_answer_confidence"] = direct_result.confidence
                ev_imgs = direct_result.evidence_images if direct_result.evidence_images else ([direct_result.evidence_img_idx] if direct_result.evidence_img_idx is not None else [])
                ans_list = direct_result.answers if direct_result.answers else ([direct_result.answer] if direct_result.answer else [])

                for a in ans_list:
                    if a and a not in harvested_answers:
                        harvested_answers.append(a)
                for idx in ev_imgs:
                    if idx not in harvested_evidence_images:
                        harvested_evidence_images.append(idx)
                    harvested_patch_boxes.append([0, 0, int(candidate_images[idx].size[0]), int(candidate_images[idx].size[1])])
                    harvested_extractions.append({
                        "img_idx": idx,
                        "box": [0, 0, int(candidate_images[idx].size[0]), int(candidate_images[idx].size[1])],
                        "expanded_box": [0, 0, int(candidate_images[idx].size[0]), int(candidate_images[idx].size[1])],
                        "raw_answer": direct_result.raw_response,
                        "final_answer": direct_result.answer,
                        "source_branch": "Direct Global Answer"
                    })

                unresolved_images = [i for i in unresolved_images if i not in ev_imgs]

                if not is_quiet:
                    print(f"[Phase 1: Direct Global Answer] Decision: ANSWERED directly (Confidence: {direct_result.confidence:.2f} >= {self.config.direct_answer_conf_threshold:.2f}) -> Answer: '{direct_result.answer}' | Evidence Images: {ev_imgs} (Latency: {lat_p1}s)")

                # Early return if single image or if all candidate images resolved
                if (total_images == 1 and not is_multi) or len(unresolved_images) == 0:
                    telemetry["route"] = "direct_global_answer"
                    telemetry["routes_executed"] = routes_executed
                    telemetry["evidence_images"] = harvested_evidence_images
                    telemetry["stage6_patch_boxes"] = harvested_patch_boxes
                    pre_agg_boxes = [
                        {"img_idx": idx, "box": [0, 0, int(candidate_images[idx].size[0]), int(candidate_images[idx].size[1])], "score": 1.0, "source_branch": "DirectGlobalAnswer", "entropy": 0.0}
                        for idx in harvested_evidence_images
                    ]
                    telemetry["final_crop_boxes_before_aggregate"] = pre_agg_boxes
                    telemetry["num_unified_crops"] = len(harvested_extractions)
                    telemetry["stage5_boxes"] = [{"img_idx": e["img_idx"], "box": list(map(int, e["box"]))} for e in harvested_extractions]
                    telemetry["stage6_crops"] = harvested_extractions
                    telemetry["is_cross_object"] = False
                    telemetry["is_relational"] = False
                    telemetry["stage5_mode"] = "direct_answer"
                    telemetry["total_latency_sec"] = round(time.perf_counter() - t_start, 3)

                    from .stages.stage6_foveated_reader import split_attribute_answer
                    pred_list = []
                    for a in harvested_answers:
                        for t in split_attribute_answer(a):
                            if t.lower() not in [x.lower() for x in pred_list]:
                                pred_list.append(t)
                    is_attr_mode = (getattr(self.config, "answer_type", "free_form") == "attribute_list")
                    primary_ans = ", ".join(pred_list) if (is_attr_mode and pred_list) else direct_result.answer
                    primary_evidence_img_idx = direct_result.evidence_img_idx if direct_result.evidence_img_idx is not None else (harvested_evidence_images[0] if harvested_evidence_images else None)
                    telemetry["primary_evidence_image"] = primary_evidence_img_idx
                    if not is_quiet:
                        print(f"[MSSearch Final Decision] Primary Answer: '{primary_ans}' | Primary Image: {primary_evidence_img_idx} | Evidence Images: {harvested_evidence_images} | Route: direct_global_answer | Latency: {telemetry['total_latency_sec']}s")
                    return {
                        "answers": harvested_answers,
                        "prediction_list": pred_list if pred_list else harvested_answers,
                        "primary_answer": primary_ans,
                        "primary_evidence_image": primary_evidence_img_idx,
                        "raw_vlm_response": direct_result.raw_response,
                        "evidence_images": harvested_evidence_images,
                        "stage6_patch_boxes": harvested_patch_boxes,
                        "final_crop_boxes_before_aggregate": pre_agg_boxes,
                        "is_cross_object": False,
                        "is_relational": False,
                        "stage5_mode": "direct_answer",
                        "extractions": harvested_extractions,
                        "telemetry": telemetry
                    }
                elif is_verbose:
                    print(f"[Phase 1][Waterfall Cascade] Direct answer confirmed on images {ev_imgs}. Unresolved images {unresolved_images} cascading to Fast Path...")
            else:
                if not is_quiet:
                    print(f"[Phase 1: Direct Global Answer] Decision: INSUFFICIENT confidence ({direct_result.confidence:.2f} < {self.config.direct_answer_conf_threshold:.2f}) -> Cascading to Stage 1 (Latency: {lat_p1}s)")

        # ---------------------------------------------------------------------
        # Stage 1: Question Parsing & Active Object Set (A_q)
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        first_img = candidate_images[unresolved_images[0]] if (candidate_images and unresolved_images) else (candidate_images[0] if candidate_images else None)
        active_objects = parse_question_to_active_objects(
            question=question,
            zoom_model=self.zoom_model,
            nlp_model=self.nlp_model,
            ic_examples=ic_examples,
            verbose=sub_config.verbose,
            image_pil=first_img,
            config=sub_config
        )
        lat_s1 = round(time.perf_counter() - t0, 3)
        telemetry["timings_sec"]["stage1_parser"] = lat_s1
        telemetry["active_objects"] = list(active_objects)
        telemetry["parsed_question"] = active_objects.to_dict() if hasattr(active_objects, "to_dict") else {}
        telemetry["stages_executed"].append("stage1_parser")
        if not is_quiet:
            print(f"[Stage 1: Question Parsing] Decision: Extracted Active Objects {list(active_objects)} | Type: '{getattr(active_objects, 'question_type', 'standard')}' (Latency: {lat_s1}s)")

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
            config=sub_config,
            active_img_indices=unresolved_images if is_multi else None
        )
        lat_s2 = round(time.perf_counter() - t0, 3)
        telemetry["timings_sec"]["stage2_fast_path"] = lat_s2
        telemetry["stages_executed"].append("stage2_fast_path")

        if fast_path_result.triggered:
            routes_executed.append("tool_based_fast_path")
            telemetry["fast_path_confidence"] = fast_path_result.confidence
            ev_imgs = fast_path_result.evidence_images if fast_path_result.evidence_images else ([fast_path_result.evidence_img_idx] if fast_path_result.evidence_img_idx is not None else [])
            ans_list = fast_path_result.answers if fast_path_result.answers else ([fast_path_result.answer] if fast_path_result.answer else [])

            for a in ans_list:
                if a and a not in harvested_answers:
                    harvested_answers.append(a)
            for idx in ev_imgs:
                if idx not in harvested_evidence_images:
                    harvested_evidence_images.append(idx)

            if fast_path_result.detected_boxes:
                for b in fast_path_result.detected_boxes:
                    harvested_patch_boxes.append(list(map(int, b)))

            for idx in ev_imgs:
                b = list(fast_path_result.detected_boxes[0]) if fast_path_result.detected_boxes else [0, 0, int(candidate_images[idx].size[0]), int(candidate_images[idx].size[1])]
                harvested_extractions.append({
                    "img_idx": idx,
                    "box": b,
                    "expanded_box": b,
                    "raw_answer": fast_path_result.raw_response or fast_path_result.answer,
                    "final_answer": fast_path_result.answer,
                    "source_branch": "Fast Path"
                })

            unresolved_images = [i for i in unresolved_images if i not in ev_imgs]

            if not is_quiet:
                print(f"[Stage 2: Tool Fast Path] Decision: TRIGGERED (Confidence: {fast_path_result.confidence:.2f} >= {self.config.fast_path_conf_threshold:.2f}) -> Answer: '{fast_path_result.answer}' | Evidence Images: {ev_imgs} (Latency: {lat_s2}s)")

            # Early return if single image or if all candidate images resolved
            if (total_images == 1 and not is_multi) or len(unresolved_images) == 0:
                telemetry["route"] = "tool_based_fast_path" if routes_executed == ["tool_based_fast_path"] else "hybrid_direct_and_fast_path"
                telemetry["routes_executed"] = routes_executed
                telemetry["evidence_images"] = harvested_evidence_images
                telemetry["stage6_patch_boxes"] = harvested_patch_boxes
                pre_agg_boxes = [
                    {
                        "img_idx": idx,
                        "box": list(map(int, fast_path_result.detected_boxes[0])) if fast_path_result.detected_boxes else [0, 0, int(candidate_images[idx].size[0]), int(candidate_images[idx].size[1])],
                        "score": float(fast_path_result.confidence),
                        "source_branch": "FastPath",
                        "entropy": 0.0
                    }
                    for idx in harvested_evidence_images
                ]
                telemetry["final_crop_boxes_before_aggregate"] = pre_agg_boxes
                telemetry["num_unified_crops"] = len(harvested_extractions)
                telemetry["stage5_boxes"] = [{"img_idx": e["img_idx"], "box": list(map(int, e["box"]))} for e in harvested_extractions]
                telemetry["stage6_crops"] = harvested_extractions
                telemetry["is_cross_object"] = False
                telemetry["is_relational"] = False
                telemetry["stage5_mode"] = "fast_path"
                telemetry["total_latency_sec"] = round(time.perf_counter() - t_start, 3)

                from .stages.stage6_foveated_reader import split_attribute_answer
                pred_list = []
                for a in harvested_answers:
                    for t in split_attribute_answer(a):
                        if t.lower() not in [x.lower() for x in pred_list]:
                            pred_list.append(t)
                is_attr_mode = (getattr(self.config, "answer_type", "free_form") == "attribute_list")
                primary_ans = ", ".join(pred_list) if (is_attr_mode and pred_list) else (harvested_answers[0] if harvested_answers else "No answer found")
                primary_evidence_img_idx = fast_path_result.evidence_img_idx if fast_path_result.evidence_img_idx is not None else (harvested_evidence_images[0] if harvested_evidence_images else None)
                telemetry["primary_evidence_image"] = primary_evidence_img_idx
                if not is_quiet:
                    print(f"[MSSearch Final Decision] Primary Answer: '{primary_ans}' | Primary Image: {primary_evidence_img_idx} | Evidence Images: {harvested_evidence_images} | Route: {telemetry['route']} | Latency: {telemetry['total_latency_sec']}s")
                return {
                    "answers": harvested_answers,
                    "prediction_list": pred_list if pred_list else harvested_answers,
                    "primary_answer": primary_ans,
                    "primary_evidence_image": primary_evidence_img_idx,
                    "raw_vlm_response": fast_path_result.raw_response or fast_path_result.answer,
                    "evidence_images": harvested_evidence_images,
                    "stage6_patch_boxes": harvested_patch_boxes,
                    "final_crop_boxes_before_aggregate": pre_agg_boxes,
                    "is_cross_object": False,
                    "is_relational": False,
                    "stage5_mode": "fast_path",
                    "extractions": harvested_extractions,
                    "telemetry": telemetry
                }
            elif is_verbose:
                print(f"[Stage 2][Waterfall Cascade] Fast path confirmed on images {ev_imgs}. Unresolved images {unresolved_images} cascading to Visual Search branch...")
        else:
            if not is_quiet:
                if getattr(fast_path_result, "bypass_reason", None) == "targets_not_detected":
                    print(f"[Stage 2: Tool Fast Path] Decision: NOT TRIGGERED (Targets not found on canvas) -> Cascading to Stage 3 (Latency: {lat_s2}s)")
                else:
                    print(f"[Stage 2: Tool Fast Path] Decision: NOT TRIGGERED (Confidence: {fast_path_result.confidence:.2f} < {self.config.fast_path_conf_threshold:.2f}) -> Cascading to Stage 3 (Latency: {lat_s2}s)")

        # ---------------------------------------------------------------------
        # Stage 3: Entropy-Gated Recursive Patch Partitioning & SAM3-SGAP
        # ---------------------------------------------------------------------
        routes_executed.append("visual_search_branch")
        telemetry["route"] = "visual_search_branch" if len(routes_executed) == 1 else "waterfall_cascade"
        telemetry["routes_executed"] = routes_executed
        t0 = time.perf_counter()
        # Reuses feature map extracted during Stage 2 canvas check
        feature_cache = fast_path_result.feature_cache if self.config.reuse_stage2_features else {}
        bounded_patches = partition_candidate_images(
            candidate_images=candidate_images,
            stage2_feature_cache=feature_cache,
            config=sub_config,
            active_objects=active_objects,
            siglip_model=self.siglip_model,
            target_img_indices=unresolved_images if is_multi else None
        )
        lat_s3 = round(time.perf_counter() - t0, 3)
        telemetry["timings_sec"]["stage3_partitioning"] = lat_s3
        telemetry["num_bounded_patches"] = len(bounded_patches)
        telemetry["stage3_boxes"] = [{"img_idx": p.img_idx, "box": list(map(int, p.box))} for p in bounded_patches]
        telemetry["stages_executed"].append("stage3_partitioning")
        if not is_quiet:
            print(f"[Stage 3: Partitioning] Decision: Partitioned {len(candidate_images)} images into {len(bounded_patches)} bounded patches (Mode: {self.config.partition_mode}) (Latency: {lat_s3}s)")

        # ---------------------------------------------------------------------
        # Stage 4: SAM3 Feature Grouping & Active Object Gating
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        remaining_patches, stage4_groups = run_feature_grouping_and_gating(
            bounded_patches=bounded_patches,
            candidate_images=candidate_images,
            active_objects=active_objects,
            sam_model=self.sam_model,
            config=sub_config,
            detected_boxes=fast_path_result.detected_boxes,
            zoom_model=self.zoom_model,
            return_debug=True,
            question=question
        )
        lat_s4 = round(time.perf_counter() - t0, 3)
        telemetry["timings_sec"]["stage4_grouping_gating"] = lat_s4
        telemetry["num_remaining_patches"] = len(remaining_patches)
        telemetry["stage4_groups"] = stage4_groups
        telemetry["stage4_kept_groups"] = sum(1 for g in stage4_groups if g.get("status") == "KEPT")
        telemetry["stage4_pruned_groups"] = sum(1 for g in stage4_groups if g.get("status") == "PRUNED")
        telemetry["stage4_boxes"] = [{"img_idx": p.img_idx, "box": list(map(int, p.box))} for p in remaining_patches]
        telemetry["stages_executed"].append("stage4_grouping_gating")
        if not is_quiet:
            print(f"[Stage 4: Grouping & Gating] Decision: Evaluated {len(stage4_groups)} semantic groups -> Kept {telemetry['stage4_kept_groups']} groups ({len(remaining_patches)} patches), Pruned {telemetry['stage4_pruned_groups']} groups (Latency: {lat_s4}s)")

        # ---------------------------------------------------------------------
        # Recursive Latent Re-Partitioning Feedback Loop (Stage 4 -> Stage 3)
        # ---------------------------------------------------------------------
        if getattr(sub_config, "enable_latent_repartition", True):
            repart_thresh = getattr(sub_config, "latent_repartition_threshold", 0.70)
            max_repart_per_img = getattr(sub_config, "max_latent_repartition_patches_per_image", 2)

            high_latent_groups = [
                g for g in stage4_groups
                if g.get("status") == "KEPT" and (g.get("vlm_latent_score") or -1.0) >= repart_thresh
            ]

            if high_latent_groups:
                by_img_hl: Dict[int, List[Dict[str, Any]]] = {}
                for g in high_latent_groups:
                    by_img_hl.setdefault(g["img_idx"], []).append(g)

                repartitioned_patches: List[BoundedPatch] = []
                replaced_parent_boxes = set()

                for img_idx, g_list in by_img_hl.items():
                    sorted_g = sorted(g_list, key=lambda g: g.get("vlm_latent_score", 0.0), reverse=True)[:max_repart_per_img]
                    for g in sorted_g:
                        parent_box = tuple(int(round(float(c))) for c in g["rep_box"])
                        pw = parent_box[2] - parent_box[0]
                        ph = parent_box[3] - parent_box[1]
                        if pw < 128 or ph < 128:
                            continue

                        parent_crop = candidate_images[img_idx].crop(parent_box)
                        local_child_patches = partition_candidate_images(
                            candidate_images=[parent_crop],
                            feature_cache={},
                            config=sub_config,
                            active_objects=active_objects,
                            siglip_model=self.siglip_model
                        )

                        if local_child_patches:
                            replaced_parent_boxes.add((img_idx, parent_box))
                            child_crops = []
                            child_patch_objs = []
                            tau_det = getattr(sub_config, "detection_score_threshold", 0.30)

                            for cp in local_child_patches:
                                lx1, ly1, lx2, ly2 = cp.box
                                child_global_box = (
                                    parent_box[0] + lx1,
                                    parent_box[1] + ly1,
                                    parent_box[0] + lx2,
                                    parent_box[1] + ly2
                                )
                                c_crop = candidate_images[img_idx].crop(child_global_box)
                                cw, ch = c_crop.size
                                if cw >= 8 and ch >= 8:
                                    child_crops.append(c_crop)
                                    child_patch_objs.append((cp, child_global_box))

                            # Evaluate child crops with SAM3 open-vocabulary detection
                            surviving_children: List[BoundedPatch] = []
                            if self.sam_model is not None and active_objects and child_crops:
                                sam3_bs = getattr(sub_config, "sam3_batch_size", 4)
                                if hasattr(self.sam_model, "batch_inference_multi_images") and sam3_bs > 1:
                                    child_outputs = self.sam_model.batch_inference_multi_images(
                                        child_crops, list(active_objects), batch_size=sam3_bs
                                    )
                                else:
                                    child_outputs = []
                                    for c_img in child_crops:
                                        try:
                                            import torch
                                            if hasattr(torch, "inference_mode"):
                                                with torch.inference_mode():
                                                    child_outputs.append(self.sam_model.batch_inference(c_img, list(active_objects)))
                                            else:
                                                child_outputs.append(self.sam_model.batch_inference(c_img, list(active_objects)))
                                        except Exception:
                                            child_outputs.append((None, {}, []))

                                for (cp, child_global_box), (_, processed_results, text_id) in zip(child_patch_objs, child_outputs):
                                    child_dets = []
                                    for tid in text_id:
                                        obj_idx = (int(tid) - 1) if str(tid).isdigit() else 0
                                        label_str = active_objects[obj_idx] if 0 <= obj_idx < len(active_objects) else str(tid)
                                        res = processed_results.get(tid, {}) if isinstance(processed_results, dict) else {}
                                        boxes = res.get('boxes', [])
                                        scores = res.get('scores', [])
                                        for b, s in zip(boxes, scores):
                                            s_float = float(s) if not isinstance(s, (list, tuple)) else float(s[0])
                                            b_clean = [int(round(float(c))) for c in (b.tolist() if hasattr(b, 'tolist') else b)]
                                            if s_float >= tau_det:
                                                child_dets.append({
                                                    "label": label_str,
                                                    "score": round(s_float, 3),
                                                    "local_box": b_clean,
                                                    "global_box": [b_clean[0] + child_global_box[0], b_clean[1] + child_global_box[1],
                                                                   b_clean[2] + child_global_box[0], b_clean[3] + child_global_box[1]]
                                                })
                                    if child_dets:
                                        best_det_score = max(d["score"] for d in child_dets)
                                        child_bp = BoundedPatch(
                                            img_idx=img_idx,
                                            box=child_global_box,
                                            feature_vector=cp.feature_vector,
                                            entropy=cp.entropy,
                                            confidence_score=float(best_det_score), # Unboosted continuous score in [0.0, 1.0]
                                            siglip_score=cp.siglip_score,
                                            parent_id=parent_box,
                                            level=getattr(cp, "level", 1) + 1
                                        )
                                        setattr(child_bp, "is_fine_grain", True)
                                        setattr(child_bp, "is_recursive_child", True)
                                        setattr(child_bp, "child_detections", child_dets)
                                        surviving_children.append(child_bp)
                            elif child_patch_objs:
                                # Fallback when SAM3 is not loaded (e.g. testing)
                                for cp, child_global_box in child_patch_objs:
                                    child_bp = BoundedPatch(
                                        img_idx=img_idx,
                                        box=child_global_box,
                                        feature_vector=cp.feature_vector,
                                        entropy=cp.entropy,
                                        confidence_score=float(min(1.0, g.get("vlm_latent_score") or 0.5)),
                                        siglip_score=cp.siglip_score,
                                        parent_id=parent_box,
                                        level=getattr(cp, "level", 1) + 1
                                    )
                                    setattr(child_bp, "is_fine_grain", True)
                                    setattr(child_bp, "is_recursive_child", True)
                                    surviving_children.append(child_bp)

                            if surviving_children:
                                repartitioned_patches.extend(surviving_children)
                            elif is_verbose:
                                print(f"[Stage 4 -> Stage 3 Feedback] Re-partitioned parent {parent_box}: 0 detections above tau_det={tau_det}. Parent and children permanently pruned.")

                if repartitioned_patches:
                    remaining_patches = [
                        p for p in remaining_patches
                        if (p.img_idx, tuple(int(round(float(c))) for c in p.box)) not in replaced_parent_boxes
                    ] + repartitioned_patches
                    if not is_quiet:
                        print(f"[Stage 4 -> Stage 3 Feedback] Re-partitioned {len(replaced_parent_boxes)} high-latent patches into {len(repartitioned_patches)} verified fine sub-patches at 100% native resolution.")
                elif replaced_parent_boxes:
                    remaining_patches = [
                        p for p in remaining_patches
                        if (p.img_idx, tuple(int(round(float(c))) for c in p.box)) not in replaced_parent_boxes
                    ]
                    if not is_quiet:
                        print(f"[Stage 4 -> Stage 3 Feedback] Pruned {len(replaced_parent_boxes)} high-latent parents with 0 confirmed child detections.")

        # ---------------------------------------------------------------------
        # Stage 5: Resolution-Aware Screening & Spatial Optimization
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        question_type = getattr(active_objects, "question_type", None)
        is_relational = getattr(active_objects, "is_relational", None)
        unified_pool = run_resolution_screening(
            remaining_patches=remaining_patches,
            candidate_images=candidate_images,
            active_objects=active_objects,
            siglip_model=self.siglip_model,
            config=sub_config,
            question=question,
            is_relational=is_relational,
            zoom_model=self.zoom_model
        )
        lat_s5 = round(time.perf_counter() - t0, 3)
        telemetry["timings_sec"]["stage5_screening"] = lat_s5
        telemetry["num_unified_crops"] = len(unified_pool)
        telemetry["branch_a_count"] = sum(1 for c in unified_pool if "Branch A" in c.source_branch)
        telemetry["branch_b_count"] = sum(1 for c in unified_pool if "Branch B" in c.source_branch)
        is_cross_obj = bool(getattr(unified_pool, "is_cross_object", is_relational if is_relational is not None else False))
        telemetry["is_relational"] = is_cross_obj
        telemetry["is_cross_object"] = is_cross_obj
        has_compressed_canvas = any(getattr(c, "is_compressed_canvas", False) for c in unified_pool)
        has_continuous_enclosure = any("ContinuousEnclosure" in getattr(c, "source_branch", "") for c in unified_pool)
        if has_continuous_enclosure:
            stage5_mode_val = "continuous_enclosure"
        elif has_compressed_canvas:
            stage5_mode_val = "layout"
        else:
            stage5_mode_val = "crops_only"
        telemetry["stage5_mode"] = stage5_mode_val
        pre_agg_boxes = getattr(unified_pool, "pre_aggregation_boxes", [])
        telemetry["final_crop_boxes_before_aggregate"] = pre_agg_boxes
        telemetry["stage5_boxes"] = [{"img_idx": c.img_idx, "box": list(map(int, c.box))} for c in unified_pool]
        telemetry["stages_executed"].append("stage5_screening")
        if not is_quiet:
            if has_continuous_enclosure:
                mode_desc = "Continuous Enclosure (relational=True)"
            elif has_compressed_canvas:
                mode_desc = "Layout (relational=True)"
            else:
                mode_desc = "Crops Only (relational=False)"
            print(f"[Stage 5: Screening & Aggregation] Decision: Mode={mode_desc} | Synthesized {len(unified_pool)} candidate crops from {len(pre_agg_boxes)} pre-aggregation bboxes (Latency: {lat_s5}s)")

        # ---------------------------------------------------------------------
        # Stage 6: Foveated VLM Reading with 1.25x Padding & Deduplication
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        final_result: FinalSearchResult = run_foveated_vlm_reading(
            unified_pool=unified_pool,
            candidate_images=candidate_images,
            question=reading_question,
            zoom_model=self.zoom_model,
            config=sub_config,
            question_type=question_type
        )
        lat_s6 = round(time.perf_counter() - t0, 3)
        telemetry["timings_sec"]["stage6_reading"] = lat_s6
        telemetry["stage6_crops"] = getattr(final_result, "crop_records", [])
        telemetry["total_latency_sec"] = round(time.perf_counter() - t_start, 3)
        telemetry["stages_executed"].append("stage6_foveated_reading")

        from .stages.stage6_foveated_reader import (
            is_negative_or_unanswerable,
            split_attribute_answer,
            is_singular_query,
            resolve_by_majority_voting
        )

        # Harvest Stage 6 informative answers
        s6_informative = [a for a in final_result.answers if not is_negative_or_unanswerable(a)]
        s6_answers = s6_informative if s6_informative else final_result.answers

        for a in s6_answers:
            if a and a not in harvested_answers:
                harvested_answers.append(a)

        for idx in final_result.evidence_image_indices:
            if idx not in harvested_evidence_images:
                harvested_evidence_images.append(idx)

        for c in unified_pool:
            harvested_patch_boxes.append(list(map(int, c.box)))

        for e in final_result.extractions:
            harvested_extractions.append({
                "img_idx": e.img_idx,
                "box": list(e.box),
                "expanded_box": list(e.expanded_box),
                "raw_answer": e.raw_answer,
                "final_answer": e.final_answer,
                "source_branch": e.source_branch,
                "crop_score": getattr(e, "crop_score", 0.0)
            })

        # Final answers consolidation
        informative_answers = [a for a in harvested_answers if not is_negative_or_unanswerable(a)]
        final_answers_out = informative_answers if informative_answers else harvested_answers

        is_singular = is_singular_query(question, question_type)
        is_attr_mode = (getattr(self.config, "answer_type", "free_form") == "attribute_list")

        primary_answer = None
        primary_evidence_img_idx = None

        if is_singular and final_result.extractions:
            voted_ans, voted_img_idx, _ = resolve_by_majority_voting(
                final_result.extractions,
                similarity_thresh=getattr(self.config, "answer_dedup_similarity", 0.75),
                verbose=self.config.verbose
            )
            if voted_ans:
                primary_answer = voted_ans
                primary_evidence_img_idx = voted_img_idx

        if primary_answer is None:
            pred_list = []
            for a in final_answers_out:
                tokens = split_attribute_answer(a)
                for t in tokens:
                    if t.lower() not in [x.lower() for x in pred_list]:
                        pred_list.append(t)

            if is_attr_mode and pred_list and not is_singular:
                primary_answer = ", ".join(pred_list)
            elif informative_answers:
                primary_answer = "; ".join(informative_answers) if is_multi else informative_answers[0]
            elif harvested_answers:
                primary_answer = "; ".join(harvested_answers) if is_multi else harvested_answers[0]
            else:
                primary_answer = "No answer found"

        if primary_evidence_img_idx is None:
            for e in harvested_extractions:
                cand_clean = e.get("final_answer") or e.get("raw_answer")
                if cand_clean and (cand_clean == primary_answer or cand_clean in informative_answers):
                    primary_evidence_img_idx = e.get("img_idx")
                    break
        if primary_evidence_img_idx is None and harvested_evidence_images:
            primary_evidence_img_idx = harvested_evidence_images[0]

        telemetry["primary_evidence_image"] = primary_evidence_img_idx
        telemetry["evidence_images"] = harvested_evidence_images
        telemetry["stage6_patch_boxes"] = harvested_patch_boxes

        raw_vlm_resp = None
        for e in harvested_extractions:
            cand_clean = e.get("final_answer") or e.get("raw_answer")
            if cand_clean in informative_answers or cand_clean == primary_answer:
                raw_vlm_resp = e.get("raw_answer")
                break
        if raw_vlm_resp is None and harvested_extractions:
            raw_vlm_resp = harvested_extractions[0].get("raw_answer")

        if not is_quiet:
            print(f"[Stage 6: Foveated Reading] Decision: Read {len(unified_pool)} candidate crops -> Harvested {len(final_answers_out)} answer(s): {final_answers_out} (Latency: {lat_s6}s)")
            print(f"[MSSearch Final Decision] Primary Answer: '{primary_answer}' | Primary Image: {primary_evidence_img_idx} | Evidence Images: {harvested_evidence_images} | Route: {telemetry.get('route')} | Latency: {telemetry.get('total_latency_sec')}s\n")

        return {
            "answers": final_answers_out,
            "prediction_list": pred_list if pred_list else final_answers_out,
            "primary_answer": primary_answer,
            "primary_evidence_image": primary_evidence_img_idx,
            "raw_vlm_response": raw_vlm_resp,
            "evidence_images": harvested_evidence_images,
            "stage6_patch_boxes": harvested_patch_boxes,
            "final_crop_boxes_before_aggregate": pre_agg_boxes,
            "is_cross_object": is_cross_obj,
            "is_relational": is_cross_obj,
            "stage5_mode": stage5_mode_val,
            "extractions": harvested_extractions,
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
        if anno_type in ["option_single", "option_list", "multiple_choice", "mc", "Multiple Choice"]:
            engine.config.include_options_in_prompt = True
            engine.config.answer_type = anno_type
        elif anno_type == "free_form":
            engine.config.include_options_in_prompt = False
            engine.config.answer_type = "free_form"
    elif options and engine.config is not None and getattr(engine.config, "answer_type", "free_form") != "free_form":
        engine.config.include_options_in_prompt = True
        engine.config.answer_type = "option_list"

    if 'query' in annotation and engine.config is not None:
        engine.config.query_meta = annotation['query']

    return engine.search(
        candidate_images=candidate_images,
        question=question,
        options=options,
        cycle_idx=cycle_idx,
        ic_examples=ic_examples
    )
