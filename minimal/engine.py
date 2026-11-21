"""
Master Search Engine for STASH-Search: Source-Attributed Spatial-Heuristic Visual Search.
Orchestrates Stage 1 (Target Extraction), Stage 2 (Quadtree Partitioning),
Stage 3 (Visual Screening & Inspection), and Stage 4 (Source-Attributed Consensus).
"""

import time
from typing import List, Dict, Any, Optional
from PIL import Image

from .config import STASHConfig
from .models.base import BaseMLLM, BaseDualEncoder
from .stages.target_extraction import extract_visual_targets
from .stages.quadtree_partitioning import partition_gallery
from .stages.visual_screening import screen_and_inspect_crops
from .stages.consensus import resolve_source_attributed_consensus


class STASHSearchEngine:
    """
    Unified end-to-end multi-image visual search engine.
    """

    def __init__(
        self,
        mllm: BaseMLLM,
        dual_encoder: BaseDualEncoder,
        config: Optional[STASHConfig] = None,
        nlp_model: Optional[Any] = None
    ):
        self.mllm = mllm
        self.dual_encoder = dual_encoder
        self.config = config if config is not None else STASHConfig()
        self.nlp_model = nlp_model

    @classmethod
    def from_pretrained(
        cls,
        mllm_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        dual_encoder_name: str = "google/siglip2-so400m-patch16-512",
        config: Optional[STASHConfig] = None,
        device: str = "cuda:0",
        nlp_model_path: Optional[str] = "en_core_web_sm"
    ) -> "STASHSearchEngine":
        """Factory method to load production HuggingFace backbones."""
        from .models.dual_encoder import SigLIPDualEncoder
        from .models.factory import load_mllm

        cfg = config if config is not None else STASHConfig()
        cfg.mllm_model_name = mllm_name_or_path
        cfg.dual_encoder_name = dual_encoder_name
        cfg.device = device

        print(f"[STASHSearchEngine] Loading Dual-Encoder: '{dual_encoder_name}'...")
        dual_encoder = SigLIPDualEncoder(model_name=dual_encoder_name, device=device)

        print(f"[STASHSearchEngine] Loading MLLM: '{mllm_name_or_path}'...")
        mllm = load_mllm(model_name_or_path=mllm_name_or_path, device=device)

        nlp_parser = None
        if nlp_model_path:
            try:
                import spacy
                nlp_parser = spacy.load(nlp_model_path)
            except Exception as e:
                if cfg.verbose:
                    print(f"[STASHSearchEngine] spaCy loading notice ({e}); using built-in closed-class pronoun filtering.")

        return cls(mllm=mllm, dual_encoder=dual_encoder, config=cfg, nlp_model=nlp_parser)

    def search(
        self,
        candidate_images: List[Image.Image],
        question: str,
        options_str: Optional[str] = None,
        target_hint: Optional[str] = None,
        inspection_prompt: Optional[str] = None,
        sample_id: Optional[str] = None,
        pool_mode: str = "N_5",
        gt_answer: Optional[str] = None,
        gt_image_idx: Optional[int] = None,
        save_vis: Optional[bool] = None,
        vis_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-image visual search across candidate images.

        Args:
            candidate_images: List of PIL images representing the gallery (size N).
            question: User natural language query.
            options_str: Optional formatted multiple-choice candidate options (e.g. for Vstar or HR-Bench).
            target_hint: Optional explicit target entity/phrase to guide quadtree search.
            inspection_prompt: Optional full prompt for Stage 3 crop inspection (e.g. for TreeBench or MME-RealWorld).
            sample_id: Optional unique sample identifier for visual artifact logging.
            pool_mode: Gallery pool size mode ('N_1', 'N_5', 'N_10', 'N_20').
            gt_answer: Optional ground truth answer string for visualization comparison.
            gt_image_idx: Optional ground truth evidence image index.
            save_vis: Explicit boolean override to enable/disable saving visualization artifacts.
            vis_dir: Root directory path for saving visualization artifacts.

        Returns:
            Dictionary containing predicted answer, supporting image index, extractions, and timings.
        """
        t_start = time.perf_counter()
        timings: Dict[str, float] = {}

        # -------------------------------------------------------------
        # Stage 1: Target Extraction
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        query_for_target = target_hint if (target_hint and target_hint.strip()) else question
        active_objects = extract_visual_targets(
            question=query_for_target,
            mllm=self.mllm,
            nlp_model=self.nlp_model,
            verbose=self.config.verbose
        )
        timings["stage1_target_extraction"] = round(time.perf_counter() - t0, 3)

        # -------------------------------------------------------------
        # Stage 2: Query-Guided Adaptive Partitioning
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        candidate_patches, entropy_pruned_leaves = partition_gallery(
            gallery_images=candidate_images,
            active_objects=active_objects,
            dual_encoder=self.dual_encoder,
            config=self.config,
            return_stats=True
        )
        timings["stage2_quadtree_partitioning"] = round(time.perf_counter() - t0, 3)

        # -------------------------------------------------------------
        # Stage 3: Gallery-Level Visual Screening & Inspection
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if inspection_prompt and inspection_prompt.strip():
            inspection_question = inspection_prompt
        elif options_str and options_str.strip():
            inspection_question = f"{question}\n{options_str.strip()}\nAnswer the option letter directly."
        else:
            inspection_question = question

        extractions, dedup_stats, crops_to_read = screen_and_inspect_crops(
            candidate_patches=candidate_patches,
            gallery_images=candidate_images,
            question=inspection_question,
            mllm=self.mllm,
            config=self.config,
            return_stats=True,
            return_crops=True
        )
        timings["stage3_visual_screening"] = round(time.perf_counter() - t0, 3)

        # -------------------------------------------------------------
        # Stage 4: Source-Attributed Consensus (SAC)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        primary_answer, primary_evidence_img, consensus_details = resolve_source_attributed_consensus(
            extractions=extractions,
            config=self.config,
            options_str=options_str,
            return_details=True
        )
        timings["stage4_consensus"] = round(time.perf_counter() - t0, 3)
        timings["total_latency_sec"] = round(time.perf_counter() - t_start, 3)

        # -------------------------------------------------------------
        # Optional Visualization & Qualitative Telemetry
        # -------------------------------------------------------------
        vis_trace = None
        should_vis = save_vis if save_vis is not None else getattr(self.config, "save_vis", False)
        if should_vis and sample_id:
            try:
                from .visualization import save_sample_visualization
                target_vis_dir = vis_dir or getattr(self.config, "vis_dir", "vis")
                vis_trace = save_sample_visualization(
                    vis_root_dir=target_vis_dir,
                    benchmark_name="stash",
                    pool_mode=pool_mode,
                    sample_id=sample_id,
                    gallery_images=candidate_images,
                    question=question,
                    active_objects=active_objects,
                    candidate_patches=candidate_patches,
                    extractions=extractions,
                    crops_to_read=crops_to_read,
                    primary_answer=primary_answer,
                    primary_evidence_img=primary_evidence_img,
                    consensus_details=consensus_details,
                    timings=timings,
                    gt_answer=gt_answer,
                    gt_image_idx=gt_image_idx,
                    config=self.config
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"[STASHSearchEngine Visualization Warning] Failed to generate visual artifacts: {e}")

        if self.config.verbose:
            print(f"[STASHSearchEngine Result] Final Answer: '{primary_answer}' | "
                  f"Evidence Image: {primary_evidence_img} | Latency: {timings['total_latency_sec']}s | "
                  f"Dedup Affected Crops (a): {dedup_stats['total_affected']} (IoS={dedup_stats['ios_merged']}, NMS={dedup_stats['nms_suppressed']}) | "
                  f"Entropy Pruned Leaves (b): {entropy_pruned_leaves}")

        return {
            "primary_answer": primary_answer,
            "primary_evidence_image": primary_evidence_img,
            "active_objects": active_objects,
            "num_candidate_patches": len(candidate_patches),
            "num_inspected_crops": len(extractions),
            "entropy_pruned_leaves": entropy_pruned_leaves,
            "dedup_affected_crops": dedup_stats,
            "extractions": extractions,
            "consensus_details": consensus_details,
            "vis_trace": vis_trace,
            "telemetry": {
                "timings_sec": timings,
                "entropy_pruned_leaves": entropy_pruned_leaves,
                "dedup_affected_crops": dedup_stats,
                "consensus": consensus_details,
                "vis_trace": vis_trace,
            }
        }
