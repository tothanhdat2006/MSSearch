import os
import json
import torch
from typing import Dict, Any, List
from PIL import Image

def load_our_engines(
    model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    sam3_path: str = "/data/shared/cvpr/dat/models/sam3/sam3.pt",
    siglip_name: str = "/data/shared/cvpr/dat/models/siglip2-so400m-patch16-512",
    nlp_model: str = "/data/shared/cvpr/dat/models/spacy/en_core_web_sm-3.8.0",
    config_path: str = None,
    gate_threshold: float = None,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """Initializes VLM, SAM3, SigLIP, and spaCy engines for MSSearch V2."""
    from mssearch.config import MSSearchConfig
    from mssearch.models import ModelQwenVL, ModelInternvl, sam3_inference, SigLIPScorer
    import spacy

    # 1. Load VLM
    m_str = str(model_path).lower()
    print(f"[Engine Setup] Loading primary VLM: '{model_path}'...")
    if "internvl" in m_str:
        zoom_model = ModelInternvl(model_path=model_path, device=device)
    elif "llava" in m_str:
        from mssearch.models import ModelGlobalLocal
        zoom_model = ModelGlobalLocal(model_path=model_path, device=device)
    else:
        zoom_model = ModelQwenVL(model_path=model_path, device=device)

    # 2. Load SAM3
    print(f"[Engine Setup] Loading SAM3 from '{sam3_path}'...")
    sam_model = sam3_inference(model_path=sam3_path, device=None)

    # 3. Load SigLIP
    print(f"[Engine Setup] Loading SigLIP from '{siglip_name}'...")
    siglip_model = SigLIPScorer(
        model_name=siglip_name,
        device=device,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # 4. Load spaCy
    print(f"[Engine Setup] Loading spaCy NLP pipeline...")
    try:
        nlp_parser = spacy.load(nlp_model)
    except Exception:
        try:
            nlp_parser = spacy.load("en_core_web_sm")
        except Exception:
            nlp_parser = None

    # 5. Build MSSearchConfig
    cfg_file = config_path or os.path.join(os.path.dirname(__file__), "config_default.json")
    cfg_dict = {}
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
    if gate_threshold is not None:
        cfg_dict["gate_threshold"] = float(gate_threshold)
    config = MSSearchConfig.from_dict(cfg_dict)

    return {
        "zoom_model": zoom_model,
        "sam_model": sam_model,
        "siglip_model": siglip_model,
        "nlp_model": nlp_parser,
        "config": config,
    }

def run_our_pipeline(
    engines: Dict[str, Any],
    candidate_images: List[Image.Image],
    sample_data: Dict[str, Any],
    sample_idx: int
) -> Dict[str, Any]:
    """Executes MSSearch V2 end-to-end multi-image search pipeline."""
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
        config=engines["config"]
    )
    return resp
