"""
Engine loader and pipeline executor for MSSearch ablation studies.
Seamlessly injects alternative detection backbones (Grounding DINO, Florence-2, DINOv3),
screening scorers (SigLIP2, CLIP-ViT-B/32, CLIP-ViT-L/14, EVA-02-L/14), and
component-increment ablation pipelines (Qwen baseline, Fast Path, Rigid quadtree, SGAP, ASP, Aggregation).
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import torch
from PIL import Image

try:
    from .models import load_detector, load_visual_scorer
    from .eval_runner_utils import run_qwen_direct_multi_image
except (ImportError, ValueError):
    from experiments.our.ablations.models import load_detector, load_visual_scorer
    from experiments.our.ablations.eval_runner_utils import run_qwen_direct_multi_image


def load_ablation_engines(
    model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
    detector_type: str = "sam3",
    detector_path: Optional[str] = None,
    scorer_type: str = "siglip",
    scorer_name: Optional[str] = None,
    nlp_model: str = "en_core_web_sm",
    config_path: Optional[str] = None,
    gate_threshold: Optional[float] = None,
    component: Optional[str] = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """
    Initializes VLM, perception detector, visual screening scorer,
    and NLP pipeline for ablation experiments, configuring component increments.
    """
    from mssearch.config import MSSearchConfig
    from mssearch.models import ModelQwenVL
    try:
        import spacy
    except ImportError:
        spacy = None

    # 1. Load VLM
    m_str = str(model_path).lower()
    print(f"[Ablation Setup] Loading primary VLM: '{model_path}'...")
    if "internvl" in m_str:
        from mssearch.models import ModelInternvl
        zoom_model = ModelInternvl(model_path=model_path, device=device)
    elif "llava" in m_str:
        from mssearch.models import ModelGlobalLocal
        zoom_model = ModelGlobalLocal(model_path=model_path, device=device)
    else:
        zoom_model = ModelQwenVL(model_path=model_path, device=device)

    # 2. Load Detector (injected into sam_model slot)
    detector = None
    if component != "qwen":
        print(f"[Ablation Setup] Loading detector: '{detector_type}'...")
        detector = load_detector(
            detector_type=detector_type,
            model_path_or_name=detector_path,
            device=device,
        )

    # 3. Load Scorer (injected into siglip_model slot)
    scorer = None
    if component != "qwen":
        scorer = load_visual_scorer(
            scorer_type=scorer_type,
            model_name_or_path=scorer_name,
            device=device,
        )

    # 4. Load spaCy NLP Pipeline
    nlp_parser = None
    if component not in ("qwen",):
        print(f"[Ablation Setup] Loading spaCy NLP pipeline from '{nlp_model}'...")
        if spacy is None:
            raise RuntimeError("[Ablation Setup Error] spaCy package is not installed.")
        try:
            nlp_parser = spacy.load(nlp_model)
        except Exception as e:
            raise RuntimeError(
                f"[Ablation Setup Error] Failed to load spaCy model from NLP_MODEL='{nlp_model}': {e}. "
                "NLP_MODEL must point to a valid spaCy model folder. Continuing without spaCy is strictly disabled."
            )

    # 5. Build Configuration
    cfg_file = config_path or os.path.join(os.path.dirname(__file__), "..", "config_default.json")
    cfg_dict = {}
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
    if gate_threshold is not None:
        cfg_dict["gate_threshold"] = float(gate_threshold)

    # Apply Default Configuration Guardrails (Fast Path & Sub-canvas Aggregation Disabled)
    cfg_dict["enable_fast_path"] = False
    cfg_dict["enable_direct_answer"] = False
    cfg_dict["enable_relational_parsing"] = False
    cfg_dict["enable_continuous_enclosure_subcanvas"] = False
    cfg_dict["enable_spatial_compressed_canvas"] = False
    cfg_dict["spatial_canvas_relational_only"] = False
    # Apply Paired Component Increments (Rigid vs SGAP)
    if component in ("qwen_rigid_topk",):
        cfg_dict["partition_mode"] = "quadrant_only"
        cfg_dict["enable_feature_grouping"] = False
        cfg_dict["max_retained_patches"] = 20
    elif component in ("qwen_sgap_topk",):
        cfg_dict["partition_mode"] = "sgap_only"
        cfg_dict["enable_feature_grouping"] = False
        cfg_dict["max_retained_patches"] = 20
    elif component in ("qwen_rigid_fg_det_abs", "qwen_rigid_fg_det_abs_topk"):
        cfg_dict["partition_mode"] = "quadrant_only"
        cfg_dict["enable_feature_grouping"] = True
        cfg_dict["enable_existence_scanning"] = False
        cfg_dict["enable_latent_repartition"] = False
        cfg_dict["retain_absent_groups_naive"] = True
    elif component in ("qwen_sgap_fg_det_abs", "qwen_sgap_fg_det_abs_topk"):
        cfg_dict["partition_mode"] = "sgap_only"
        cfg_dict["enable_feature_grouping"] = True
        cfg_dict["enable_existence_scanning"] = False
        cfg_dict["enable_latent_repartition"] = False
        cfg_dict["retain_absent_groups_naive"] = True
    elif component in ("qwen_rigid_fg_exist_abs", "qwen_rigid_fg_exist_abs_topk"):
        cfg_dict["partition_mode"] = "quadrant_only"
        cfg_dict["enable_feature_grouping"] = True
        cfg_dict["enable_existence_scanning"] = True
        cfg_dict["enable_latent_repartition"] = False
        cfg_dict["retain_absent_groups_naive"] = True
    elif component in ("qwen_sgap_fg_exist_abs", "qwen_sgap_fg_exist_abs_topk"):
        cfg_dict["partition_mode"] = "sgap_only"
        cfg_dict["enable_feature_grouping"] = True
        cfg_dict["enable_existence_scanning"] = True
        cfg_dict["enable_latent_repartition"] = False
        cfg_dict["retain_absent_groups_naive"] = True
    elif component in ("qwen_rigid_fg_det_latent", "qwen_rigid_fg_det_latent_repart"):
        cfg_dict["partition_mode"] = "quadrant_only"
        cfg_dict["enable_feature_grouping"] = True
        cfg_dict["enable_existence_scanning"] = False
        cfg_dict["enable_latent_repartition"] = True
        cfg_dict["retain_absent_groups_naive"] = False
    elif component in ("qwen_sgap_fg_det_latent", "qwen_sgap_fg_det_latent_repart"):
        cfg_dict["partition_mode"] = "sgap_only"
        cfg_dict["enable_feature_grouping"] = True
        cfg_dict["enable_existence_scanning"] = False
        cfg_dict["enable_latent_repartition"] = True
        cfg_dict["retain_absent_groups_naive"] = False
    elif component == "qwen_rigid_full":
        cfg_dict["partition_mode"] = "quadrant_only"
        cfg_dict["enable_feature_grouping"] = True
        cfg_dict["enable_existence_scanning"] = True
        cfg_dict["enable_latent_repartition"] = True
        cfg_dict["retain_absent_groups_naive"] = False
    elif component == "qwen_sgap_full":
        cfg_dict["partition_mode"] = "sgap_only"
        cfg_dict["enable_feature_grouping"] = True
        cfg_dict["enable_existence_scanning"] = True
        cfg_dict["enable_latent_repartition"] = True
        cfg_dict["retain_absent_groups_naive"] = False
    config = MSSearchConfig.from_dict(cfg_dict)
    return {
        "zoom_model": zoom_model,
        "sam_model": detector,
        "siglip_model": scorer,
        "nlp_model": nlp_parser,
        "config": config,
        "detector_type": detector_type,
        "scorer_type": scorer_type,
        "component": component,
    }


def run_ablation_pipeline(
    engines: Dict[str, Any],
    candidate_images: List[Image.Image],
    sample_data: Dict[str, Any],
    sample_idx: int,
) -> Dict[str, Any]:
    """Executes MSSearch pipeline or component ablation route with injected backbones."""
    component = engines.get("component")
    sample_id = sample_data.get("sample_id", f"sample_{sample_idx}")
    question = sample_data.get("question") or sample_data.get("query", {}).get("question", "")
    t_start = time.perf_counter()

    # Component Route 1: Direct Qwen Baseline
    if component == "qwen":
        pred_ans, pred_0based, _ = run_qwen_direct_multi_image(engines["zoom_model"], candidate_images, question)
        prim_img = pred_0based[0] if pred_0based else None
        boxes = [{"img_idx": i, "box": [0, 0, candidate_images[i].width, candidate_images[i].height]} for i in pred_0based if 0 <= i < len(candidate_images)]
        c_recs = [{
            "crop_idx": 0, "img_idx": prim_img, "box": boxes[0]["box"] if boxes else [],
            "raw_answer": pred_ans, "final_answer": pred_ans, "is_evidence": bool(pred_0based), "source_branch": "direct_baseline"
        }] if prim_img is not None else []
        v_scores = [{"answer": pred_ans, "count": 1, "weighted_score": 1.0, "img_idx": prim_img}] if (pred_ans and prim_img is not None) else []
        return {
            "primary_answer": pred_ans, "primary_evidence_image": prim_img, "evidence_images": pred_0based,
            "raw_vlm_response": pred_ans, "extractions": [],
            "telemetry": {
                "route": "direct_baseline", "routes_executed": ["direct_baseline"], "stages_executed": ["direct_baseline"],
                "stage5_mode": "direct_baseline", "evidence_images": pred_0based, "primary_evidence_image": prim_img,
                "stage5_boxes": boxes, "stage6_patch_boxes": [b["box"] for b in boxes], "stage6_crops": c_recs,
                "voting_scores": v_scores, "primary_answer_score": 1.0 if v_scores else None,
                "num_unified_crops": len(boxes), "timings_sec": {"direct_baseline": round(time.perf_counter() - t_start, 3)},
            }
        }

    # Standard MSSearch pipeline (including steps 3, 4, 5, 6 with config overrides)
    from mssearch.MSSearch import get_mssearch_response
    query_info = sample_data.get("query") or {
        "question": question, "question_type": sample_data.get("question_type", "Attribute"),
        "target_object": sample_data.get("target_object", ""),
    }
    annotation = {
        "question": question, "sample_id": sample_id, "answer_type": "free_form",
        "query": query_info, "target_bbox": sample_data.get("target_bbox"),
    }
    return get_mssearch_response(
        sam_model=engines["sam_model"], zoom_model=engines["zoom_model"],
        siglip_model=engines["siglip_model"], nlp_model=engines["nlp_model"],
        annotation=annotation, candidate_images=candidate_images, config=engines["config"],
    )
