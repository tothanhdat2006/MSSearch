"""
Engine loader and pipeline executor for MSSearch ablation studies.
Seamlessly injects alternative detection backbones (Grounding DINO, DINO-X)
and screening scorers (CLIP-ViT-B/32, CLIP-ViT-L/14, EVA-02-L/14).
"""

import os
import json
import torch
from typing import Dict, Any, List, Optional
from PIL import Image

try:
    from .models import load_detector, load_visual_scorer
except (ImportError, ValueError):
    from experiments.our.ablations.models import load_detector, load_visual_scorer


def load_ablation_engines(
    model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    detector_type: str = "sam3",
    detector_path: Optional[str] = None,
    scorer_type: str = "siglip",
    scorer_name: Optional[str] = None,
    nlp_model: str = "en_core_web_sm",
    config_path: Optional[str] = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """
    Initializes VLM, perception detector (SAM3 / Grounding DINO / DINO-X),
    visual screening scorer (SigLIP / CLIP-B/32 / CLIP-L/14 / EVA-02-L/14),
    and NLP pipeline for ablation experiments.
    """
    from mssearch.config import MSSearchConfig
    from mssearch.models import ModelQwenVL, ModelInternvl
    import spacy

    # 1. Load VLM
    m_str = str(model_path).lower()
    print(f"[Ablation Setup] Loading primary VLM: '{model_path}'...")
    if "internvl" in m_str:
        zoom_model = ModelInternvl(model_path=model_path, device=device)
    elif "llava" in m_str:
        from mssearch.models import ModelGlobalLocal
        zoom_model = ModelGlobalLocal(model_path=model_path, device=device)
    else:
        zoom_model = ModelQwenVL(model_path=model_path, device=device)

    # 2. Load Ablation Detector (Injected into sam_model slot)
    print(f"[Ablation Setup] Loading detector: '{detector_type}'...")
    detector = load_detector(
        detector_type=detector_type,
        model_path_or_name=detector_path,
        device=device,
    )

    # 3. Load Ablation Visual Scorer (Injected into siglip_model slot)
    print(f"[Ablation Setup] Loading visual scorer: '{scorer_type}'...")
    scorer = load_visual_scorer(
        scorer_type=scorer_type,
        model_name_or_path=scorer_name,
        device=device,
    )

    # 4. Load spaCy NLP Pipeline
    print(f"[Ablation Setup] Loading spaCy NLP pipeline...")
    try:
        nlp_parser = spacy.load(nlp_model)
    except Exception:
        try:
            nlp_parser = spacy.load("en_core_web_sm")
        except Exception:
            nlp_parser = None

    # 5. Build Configuration
    cfg_file = config_path or os.path.join(os.path.dirname(__file__), "..", "config_default.json")
    cfg_dict = {}
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
    config = MSSearchConfig.from_dict(cfg_dict)

    return {
        "zoom_model": zoom_model,
        "sam_model": detector,      # Core pipeline consumes this under sam_model
        "siglip_model": scorer,     # Core pipeline consumes this under siglip_model
        "nlp_model": nlp_parser,
        "config": config,
        "detector_type": detector_type,
        "scorer_type": scorer_type,
    }


def run_ablation_pipeline(
    engines: Dict[str, Any],
    candidate_images: List[Image.Image],
    sample_data: Dict[str, Any],
    sample_idx: int,
) -> Dict[str, Any]:
    """
    Executes MSSearch pipeline with injected ablation backbones.
    """
    from mssearch.MSSearch import get_mssearch_response

    sample_id = sample_data.get("sample_id", f"sample_{sample_idx}")
    question = sample_data.get("question") or sample_data.get("query", {}).get("question", "")
    query_info = sample_data.get("query") or {
        "question": question,
        "question_type": sample_data.get("question_type", "Attribute"),
        "target_object": sample_data.get("target_object", ""),
    }

    annotation = {
        "question": question,
        "sample_id": sample_id,
        "answer_type": sample_data.get("answer_type", "attribute_list"),
        "query": query_info,
        "target_bbox": sample_data.get("target_bbox"),
    }

    resp = get_mssearch_response(
        sam_model=engines["sam_model"],
        zoom_model=engines["zoom_model"],
        siglip_model=engines["siglip_model"],
        nlp_model=engines["nlp_model"],
        annotation=annotation,
        candidate_images=candidate_images,
        config=engines["config"],
    )
    return resp
