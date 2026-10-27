import os
import sys
import gc
import json
import time
import copy
import argparse
import re
import contextlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
try:
    import torch
except ImportError:
    torch = None

# Ensure MSSearch root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MSSEARCH_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if MSSEARCH_ROOT not in sys.path:
    sys.path.insert(0, MSSEARCH_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="MSSearch 200-Sample Multi-Source Evaluation Suite")

    # Mode & Benchmark flags
    parser.add_argument("--mode", type=str, choices=["e2e", "direct", "baseline"], default="e2e",
                        help="Execution mode: 'e2e' (full visual search pipeline) or 'direct'/'baseline' (vanilla VLM baseline).")
    parser.add_argument("--direct_target_only", action="store_true", default=False,
                        help="In direct baseline mode, bypass distractors and evaluate only target gold images (N=K).")
    parser.add_argument("--pool_mode", type=str, choices=["N_5", "N_10", "N_20", "ALL"], default="N_5",
                        help="Candidate pool size to evaluate: N_5 (default), N_10, N_20, or ALL.")

    # Paths
    parser.add_argument("--dataset_path", type=str,
                        default=os.environ.get("DATASET_PATH", "/home/ubuntu/workspace/phong/multi_image_visual_search/data/output/200_samples.json"),
                        help="Path to 200_samples.json dataset file.")
    parser.add_argument("--image_dir", type=str,
                        default=os.environ.get("IMAGE_DIR", "/home/ubuntu/workspace/phong/multi_image_visual_search/data/pilot_5000_images"),
                        help="Root directory containing raw images named <image_id>.jpg on the server.")
    parser.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", ""),
                        help="Directory to save JSONL logs and summary outputs.")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Explicit filename for results JSONL (defaults to auto-generated name).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom MSSearch configuration JSON file (defaults to config_default.json).")

    # Foundation Model Weights
    parser.add_argument("--model_path", "--qwen_path", dest="model_path", type=str,
                        default=os.environ.get("MODEL_PATH", os.environ.get("QWEN_PATH", "/home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct")),
                        help="Local checkpoint path for primary VLM (Qwen2.5-VL).")
    parser.add_argument("--sam3_path", type=str,
                        default=os.environ.get("SAM3_PATH", "/home/ubuntu/workspace/hr/models/sam3/sam3/sam3.pt"),
                        help="Local checkpoint path for SAM3.")
    parser.add_argument("--siglip_name", "--siglip_path", dest="siglip_name", type=str,
                        default=os.environ.get("SIGLIP_NAME", "/home/ubuntu/workspace/hr/models/siglip-base-patch16-384"),
                        help="Local checkpoint path or HuggingFace identifier for SigLIP.")
    parser.add_argument("--nlp_model", type=str,
                        default=os.environ.get("NLP_MODEL", "/home/ubuntu/workspace/hr/models/spacy/en_core_web_sm-3.8.0"),
                        help="Local spaCy pipeline path.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device assignment (defaults to 'auto' for automatic allocation).")

    # Queue slicing and debugging
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Single sample index to evaluate (0-199).")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index for dataset slicing.")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index for dataset slicing.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process from the queue.")

    # Resumption
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from existing output JSONL file, skipping completed tasks.")
    parser.add_argument("--resume_from", type=str, default=os.environ.get("RESUME_FROM", None),
                        help="Specific JSONL path to resume from.")

    # Budget & Resolution
    parser.add_argument("--attn_implementation", type=str,
                        default=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
                        help="Attention backend (sdpa, flash_attention_2, eager).")
    parser.add_argument("--min_pixels", type=int, default=3136,
                        help="Minimum visual patch resolution in pixels (56x56 floor).")
    parser.add_argument("--max_pixels", type=int, default=12845056,
                        help="Maximum visual canvas resolution in pixels (auto-capped to 4194304 for 32B).")
    parser.add_argument("--log_level", type=str, choices=["summary", "verbose", "quiet"], default="summary",
                        help="Console logging verbosity.")

    return parser.parse_args()


def resolve_image(image_id: str, image_dir: str) -> Image.Image:
    """Strict GPU server image resolver: expects <image_id>.jpg inside image_dir."""
    target_filename = f"{image_id}.jpg" if not str(image_id).lower().endswith((".jpg", ".jpeg", ".png", ".webp")) else str(image_id)
    target_path = os.path.join(image_dir, target_filename)
    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"[Dataset Error] Image '{image_id}' not found at required server path: '{target_path}'")
    return Image.open(target_path).convert("RGB")


def evaluate_predictions(
    pred_text: str,
    raw_pred_list: Optional[List[str]],
    gt_list: List[str]
) -> Dict[str, Any]:
    """
    Computes token-level set Precision, Recall, F1, and Exact Match between
    predicted attribute list and ground-truth attribute annotations.
    """
    gt_set = set(g.lower().strip() for g in gt_list if g and str(g).strip())

    extracted_preds: List[str] = []
    if raw_pred_list and isinstance(raw_pred_list, list):
        for item in raw_pred_list:
            if isinstance(item, str):
                parts = re.split(r"[,;\n/]+", item)
                for p in parts:
                    clean = re.sub(r"^[ \t\-\*\d\.\)]+", "", p).strip().lower()
                    clean = clean.strip(".,;!?\"'()[]{}")
                    if clean:
                        extracted_preds.append(clean)

    # Fallback to parse free-form prediction text if no structured tokens
    if not extracted_preds and pred_text:
        tag_match = re.search(r"<answer>(.*?)</answer>", pred_text, re.DOTALL | re.IGNORECASE)
        text_to_parse = tag_match.group(1) if tag_match else pred_text
        parts = re.split(r"[,;\n/]+", text_to_parse)
        for p in parts:
            clean = re.sub(r"^[ \t\-\*\d\.\)]+", "", p).strip().lower()
            clean = clean.strip(".,;!?\"'()[]{}")
            if clean and not clean.startswith("the ") and len(clean) > 1:
                extracted_preds.append(clean)
            elif clean:
                extracted_preds.append(clean)

    pred_set = set(extracted_preds)

    # Calculate overlaps with word-boundary awareness
    matched_gt = set()
    for gt in gt_set:
        for pred in pred_set:
            if gt == pred or (len(gt) > 2 and (gt in pred or pred in gt)):
                matched_gt.add(gt)
                break
        if gt not in matched_gt and pred_text:
            if re.search(r"\b" + re.escape(gt) + r"\b", pred_text.lower()):
                matched_gt.add(gt)

    matched_pred = set()
    for pred in pred_set:
        for gt in gt_set:
            if gt == pred or (len(gt) > 2 and (gt in pred or pred in gt)):
                matched_pred.add(pred)
                break

    recall = len(matched_gt) / len(gt_set) if gt_set else 0.0
    precision = len(matched_pred) / len(pred_set) if pred_set else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    exact_match = (matched_gt == gt_set) and (len(matched_pred) == len(pred_set)) and len(gt_set) > 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
        "gt_items": sorted(list(gt_set)),
        "pred_items": sorted(list(pred_set)),
        "matched_gt": sorted(list(matched_gt)),
        "matched_pred": sorted(list(matched_pred)),
    }


def _create_composite_canvas(images: List[Image.Image], max_side: int = 1024) -> Image.Image:
    """Stitches candidate images horizontally into a composite canvas for direct VLM baseline."""
    scaled = []
    for img in images:
        if max(img.width, img.height) > max_side:
            scale = max_side / float(max(img.width, img.height))
            scaled.append(img.resize((int(round(img.width * scale)), int(round(img.height * scale))), Image.Resampling.BILINEAR))
        else:
            scaled.append(img)
    total_w = sum(im.width for im in scaled)
    max_h = max(im.height for im in scaled)
    canvas = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    curr_x = 0
    for im in scaled:
        canvas.paste(im, (curr_x, 0))
        curr_x += im.width
    return canvas


def run_direct_vlm_inference(
    qwen_engine: Any,
    images: List[Image.Image],
    prompt: str,
    max_new_tokens: int = 128,
) -> str:
    """Direct Vanilla VLM baseline inference across multi-image candidates."""
    safe_images = []
    for img in images:
        if max(img.width, img.height) > 1024:
            scale = 1024.0 / max(img.width, img.height)
            safe_images.append(
                img.resize(
                    (int(round(img.width * scale)), int(round(img.height * scale))),
                    Image.Resampling.BILINEAR,
                )
            )
        else:
            safe_images.append(img)

    # InternVL Chat Route
    if hasattr(qwen_engine, "model") and hasattr(qwen_engine, "tokenizer") and hasattr(getattr(qwen_engine, "model", None), "chat"):
        try:
            from mssearch.models.modeling_internvl import load_image, IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN
            full_pixel_values = []
            num_patches_list = []
            for img in safe_images:
                pv = load_image(img, input_size=getattr(qwen_engine, "input_size", (448, 448))[0], max_num=getattr(qwen_engine, "anyres_num", 12))
                full_pixel_values.append(pv.to(qwen_engine.dtype))
                num_patches_list.append(pv.shape[0])
            full_pixel_values = torch.cat(full_pixel_values, dim=0).to(qwen_engine.device)

            qs_parts = [f"Candidate Image {idx + 1}:\n<image>\n" for idx in range(len(safe_images))]
            qs_parts.append(f"\n{prompt}")
            chat_prompt = "".join(qs_parts)

            image_idx = 0
            for num_patches in num_patches_list:
                while "<image>" in chat_prompt:
                    num_patches = num_patches_list[image_idx]
                    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * qwen_engine.model.num_image_token * num_patches + IMG_END_TOKEN
                    chat_prompt = chat_prompt.replace("<image>", image_tokens, 1)
                    image_idx += 1

            gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
            response = qwen_engine.model.chat(qwen_engine.tokenizer, full_pixel_values, chat_prompt, gen_cfg)
            return response.strip()
        except Exception:
            pass

    # ModelGlobalLocal Route (LLaVA models using image_processor + tokenizer)
    if hasattr(qwen_engine, "image_processor") and hasattr(qwen_engine, "tokenizer") and not hasattr(qwen_engine, "processor"):
        if len(safe_images) > 1:
            composite = _create_composite_canvas(safe_images)
            return qwen_engine.generate_response(image_pil=composite, question=prompt, max_new_tokens=max_new_tokens)
        else:
            return qwen_engine.generate_response(image_pil=safe_images[0], question=prompt, max_new_tokens=max_new_tokens)

    # Single-Image Route
    if len(safe_images) == 1:
        if hasattr(qwen_engine, "generate_response"):
            return qwen_engine.generate_response(image_pil=safe_images[0], question=prompt, max_new_tokens=max_new_tokens)
        elif hasattr(qwen_engine, "free_form_using_nodes"):
            return qwen_engine.free_form_using_nodes(image_pil=safe_images[0], question=prompt, searched_nodes=[], max_new_tokens=max_new_tokens)

    # Multi-Image Route via HuggingFace Processor (Qwen2.5-VL / LLaVA-OneVision)
    if (
        hasattr(qwen_engine, "processor")
        and hasattr(qwen_engine, "model")
        and hasattr(qwen_engine, "get_prompt_from_qs")
    ):
        qs_parts = [f"Candidate Image {idx + 1}:\n<image>\n" for idx in range(len(safe_images))]
        qs_parts.append(f"\n{prompt}")
        qs_full = "".join(qs_parts)
        prompt_text = qwen_engine.get_prompt_from_qs(qs_full)

        for images_arg in (safe_images, [safe_images]):
            try:
                model_inputs = qwen_engine.processor(
                    text=[prompt_text],
                    images=images_arg,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                ).to(qwen_engine.device)

                if "pixel_values" in model_inputs and hasattr(model_inputs["pixel_values"], "dtype"):
                    m_dtype = getattr(qwen_engine.model, "dtype", getattr(qwen_engine, "dtype", None))
                    if m_dtype in (torch.float16, torch.bfloat16):
                        model_inputs["pixel_values"] = model_inputs["pixel_values"].to(m_dtype)

                if model_inputs.input_ids.shape[0] == 1 and "attention_mask" in model_inputs:
                    if not (model_inputs.attention_mask == 0).any():
                        model_inputs.pop("attention_mask", None)

                infer_ctx = torch.inference_mode() if torch is not None else contextlib.nullcontext()
                with infer_ctx:
                    generated_ids = qwen_engine.model.generate(
                        **model_inputs,
                        use_cache=True,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=1.0,
                        do_sample=False,
                    )
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
                decoded = qwen_engine.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                return decoded[0].strip()
            except Exception:
                pass

        # Composite canvas fallback
        try:
            composite_canvas = _create_composite_canvas(safe_images)
            single_prompt = f"<image>\n{prompt}"
            single_prompt_text = qwen_engine.get_prompt_from_qs(single_prompt)
            model_inputs = qwen_engine.processor(
                text=[single_prompt_text],
                images=composite_canvas,
                return_tensors="pt",
                padding=True,
                padding_side="left",
            ).to(qwen_engine.device)

            if "pixel_values" in model_inputs and hasattr(model_inputs["pixel_values"], "dtype"):
                m_dtype = getattr(qwen_engine.model, "dtype", getattr(qwen_engine, "dtype", None))
                if m_dtype in (torch.float16, torch.bfloat16):
                    model_inputs["pixel_values"] = model_inputs["pixel_values"].to(m_dtype)

            if model_inputs.input_ids.shape[0] == 1 and "attention_mask" in model_inputs:
                if not (model_inputs.attention_mask == 0).any():
                    model_inputs.pop("attention_mask", None)

            infer_ctx = torch.inference_mode() if torch is not None else contextlib.nullcontext()
            with infer_ctx:
                generated_ids = qwen_engine.model.generate(
                    **model_inputs,
                    use_cache=True,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.0,
                    do_sample=False,
                )
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
            decoded = qwen_engine.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return decoded[0].strip()
        except Exception as e_comp:
            return f"[ERROR: {str(e_comp)}]"

    return "[ERROR: Model engine does not support direct visual inference]"


def main():
    args = parse_args()

    # Map baseline mode name
    execution_mode = "direct" if args.mode in ("direct", "baseline") else "e2e"

    # Resolve dataset path
    resolved_dataset_path = os.path.abspath(args.dataset_path)
    if not os.path.exists(resolved_dataset_path):
        local_fallback = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "datasets", "200_samples.json"))
        if os.path.exists(local_fallback):
            resolved_dataset_path = local_fallback
        else:
            raise FileNotFoundError(f"[Dataset Error] Dataset file does not exist: '{args.dataset_path}'")

    # Resolve output directory
    model_name = os.path.basename(os.path.normpath(args.model_path)) if args.model_path else "model"
    if not args.output_dir:
        tag = f"baseline_{model_name}" if execution_mode == "direct" else f"mssearch_{model_name}"
        if args.direct_target_only and execution_mode == "direct":
            tag += "_target_only"
        output_dir = os.path.join(SCRIPT_DIR, f"results_{tag}")
    else:
        output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.results_file:
        results_jsonl = os.path.join(output_dir, args.results_file) if not os.path.isabs(args.results_file) else args.results_file
    elif args.resume_from and os.path.exists(args.resume_from):
        results_jsonl = os.path.abspath(args.resume_from)
    else:
        mode_prefix = "baseline" if execution_mode == "direct" else "mssearch"
        results_jsonl = os.path.join(output_dir, f"results_{mode_prefix}_200_{args.pool_mode.lower()}_{timestamp}.jsonl")

    # Ensure torch is installed when running evaluation
    if torch is None:
        raise RuntimeError("PyTorch is required for model inference. Please activate your environment (e.g. hrsearch).")

    # Load dataset
    with open(resolved_dataset_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    print("=" * 80)
    print("MSSEARCH 200-SAMPLE MULTI-SOURCE EVALUATION SUITE")
    print("=" * 80)
    print(f"Mode              : {execution_mode.upper()} {'(Target Only N=K)' if args.direct_target_only else ''}")
    print(f"Pool Mode         : {args.pool_mode}")
    print(f"Dataset Path      : {resolved_dataset_path} (Total Samples: {len(all_samples)})")
    print(f"Image Directory   : {args.image_dir}")
    print(f"Output Directory  : {output_dir}")
    print(f"Results File      : {results_jsonl}")
    print(f"Model Path        : {args.model_path}")
    print(f"Device Allocation : {args.device}")
    print("=" * 80)

    # Slicing
    sample_indices = list(range(len(all_samples)))
    if args.sample_idx is not None:
        sample_indices = [args.sample_idx]
    else:
        start = args.start_idx if args.start_idx is not None else 0
        end = args.end_idx if args.end_idx is not None else len(all_samples)
        sample_indices = sample_indices[start:end]
        if args.max_samples is not None:
            sample_indices = sample_indices[:args.max_samples]

    # Formulate evaluation tasks
    eval_tasks: List[Tuple[int, Dict[str, Any], str, Dict[str, Any]]] = []
    pool_keys = ["N_5", "N_10", "N_20"] if args.pool_mode == "ALL" else [args.pool_mode]

    for s_idx in sample_indices:
        s = all_samples[s_idx]
        pools = s.get("pools", {})
        for p_name in pool_keys:
            if p_name in pools:
                eval_tasks.append((s_idx, s, p_name, pools[p_name]))

    # Auto-resume handling
    completed_task_keys = set()
    if args.resume or (args.resume_from and os.path.exists(results_jsonl)):
        if os.path.exists(results_jsonl):
            with open(results_jsonl, "r", encoding="utf-8") as rf:
                for line in rf:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        completed_task_keys.add(f"{rec.get('sample_id')}_{rec.get('pool_name')}")
                    except Exception:
                        pass
        print(f"[Auto-Resume] Loaded {len(completed_task_keys)} completed tasks from {results_jsonl}")

    remaining_tasks = [t for t in eval_tasks if f"{t[1].get('sample_id')}_{t[2]}" not in completed_task_keys]
    print(f"[Task Queue] Total Tasks: {len(eval_tasks)} | Resumed: {len(completed_task_keys)} | To Execute: {len(remaining_tasks)}")
    if not remaining_tasks:
        print("[Notice] All tasks already completed. Proceeding to summary calculation.")

    # Model Initialization
    from mssearch.config import MSSearchConfig
    from mssearch.models.modeling_qwenvl import ModelQwenVL

    # Auto-cap max_pixels for 32B checkpoints
    max_pixels = args.max_pixels
    if "32b" in str(args.model_path).lower() and max_pixels > 4194304:
        print(f"[OOM Guard] Detected 32B model path. Capping max_pixels to 4194304 (was {max_pixels}).")
        max_pixels = 4194304

    # Load VLM Engine
    qwen_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path_str = str(args.model_path).lower()
    cfg_data = {}
    cfg_path = os.path.join(args.model_path, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_data = json.load(f)
        except Exception:
            pass
    archs = [a.lower() for a in cfg_data.get("architectures", [])]
    model_type = cfg_data.get("model_type", "").lower()

    if "internvl" in model_path_str or any("internvl" in a for a in archs) or "internvl" in model_type:
        try:
            from mssearch.models.modeling_internvl import ModelInternvl
            print(f"[Loading] InternVL Engine from '{args.model_path}' on {qwen_device}...")
            qwen_engine = ModelInternvl(
                model_path=args.model_path,
                device=qwen_device,
                torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16,
                patch_scale=1.2,
            )
        except Exception as e_ivl:
            print(f"[Notice] ModelInternvl failed to initialize ({e_ivl}), falling back to ModelQwenVL.")
            qwen_engine = ModelQwenVL(
                model_path=args.model_path,
                device=qwen_device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                min_pixels=args.min_pixels,
                max_pixels=max_pixels,
            )
    elif "llava" in model_path_str or any("llava" in a for a in archs) or "llava" in model_type:
        from mssearch.models.modeling_llava import ModelGlobalLocal
        print(f"[Loading] LLaVA GlobalLocal Engine from '{args.model_path}' on {qwen_device}...")
        is_qwen = "qwen" in model_path_str or "ov" in model_path_str or "onevision" in model_path_str or any("qwen" in a for a in archs)
        qwen_engine = ModelGlobalLocal(
            model_path=args.model_path,
            device=qwen_device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            conv_type="qwen_1_5" if is_qwen else "llava_v1",
            patch_scale=1.2,
            bias_value=0.6,
            siglip_model_path=args.siglip_name,
            attn_implementation=getattr(args, "attn_implementation", "sdpa"),
        )
    else:
        print(f"[Loading] QwenVL Engine from '{args.model_path}' on {qwen_device}...")
        qwen_engine = ModelQwenVL(
            model_path=args.model_path,
            device=qwen_device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            min_pixels=args.min_pixels,
            max_pixels=max_pixels,
        )

    sam3_engine = None
    siglip_engine = None
    nlp_parser = None
    pipeline_config = None

    if execution_mode == "e2e":
        from mssearch.MSSearch import get_mssearch_response
        from mssearch.models.modeling_sam3 import sam3_inference
        from mssearch.models.modeling_siglip import SigLIPScorer
        import spacy

        print("[Engine Setup] Initializing SAM3 open-vocabulary segmentor...")
        sam3_engine = sam3_inference(model_path=args.sam3_path, device=None)

        print("[Engine Setup] Initializing SigLIP zero-shot vision-language scorer...")
        siglip_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        siglip_engine = SigLIPScorer(
            model_name=args.siglip_name,
            device=siglip_device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        print("[Engine Setup] Initializing spaCy linguistic parser...")
        try:
            nlp_parser = spacy.load(args.nlp_model)
        except Exception:
            try:
                nlp_parser = spacy.load("en_core_web_sm")
            except Exception:
                nlp_parser = None

        # Build MSSearch configuration
        cfg_file = args.config or os.path.join(SCRIPT_DIR, "config_default.json")
        cfg_kwargs = {}
        if os.path.exists(cfg_file):
            with open(cfg_file, "r", encoding="utf-8") as cf:
                cfg_kwargs = json.load(cf)

        cfg_kwargs["min_pixels"] = args.min_pixels
        cfg_kwargs["max_pixels"] = max_pixels
        cfg_kwargs["multi_answer_mode"] = True
        cfg_kwargs["answer_type"] = "attribute_list"
        cfg_kwargs["scale_budgets_by_num_images"] = True
        cfg_kwargs["enable_early_consensus_exit"] = False
        cfg_kwargs["log_level"] = args.log_level

        pipeline_config = MSSearchConfig(**cfg_kwargs)

    # Execution Loop
    print("\n" + "=" * 80)
    print(f"STARTING EVALUATION RUN: {len(remaining_tasks)} TASKS REMAINING")
    print("=" * 80)

    with open(results_jsonl, "a", encoding="utf-8") as out_f:
        for run_idx, (sample_idx, sample, pool_name, pool_data) in enumerate(remaining_tasks, start=1):
            sample_id = sample.get("sample_id", f"sample_{sample_idx:03d}")
            task_key = f"{sample_id}_{pool_name}"
            query_info = sample.get("query", {})
            question = query_info.get("question", "")
            target_object = query_info.get("target_object", "")
            target_attr_type = query_info.get("target_attribute_type", "")
            k_gt_images = sample.get("K_gt_images", len(sample.get("ground_truth", [])))

            answer_info = sample.get("answer", {})
            gt_list = answer_info.get("list", [])
            gt_split = answer_info.get("split_comma", "")
            gt_freeform = answer_info.get("freeform", "")

            # Filter candidate pool images
            raw_pool_images = pool_data.get("images", [])
            if args.direct_target_only and execution_mode == "direct":
                selected_pool_images = [img for img in raw_pool_images if img.get("role") == "gold"]
            else:
                selected_pool_images = raw_pool_images

            # Load candidate PIL images strictly from server image_dir
            candidate_images: List[Image.Image] = []
            candidate_meta: List[Dict[str, Any]] = []

            for p_img in selected_pool_images:
                iid = p_img.get("image_id")
                role = p_img.get("role", "unknown")
                img_pil = resolve_image(image_id=iid, image_dir=args.image_dir)
                candidate_images.append(img_pil)
                candidate_meta.append({
                    "image_id": iid,
                    "role": role,
                    "size": [img_pil.width, img_pil.height],
                    "pool_index": p_img.get("pool_index", len(candidate_meta)),
                })

            n_candidates = len(candidate_images)
            t0 = time.perf_counter()

            if execution_mode == "direct":
                prompt = (
                    f"Question: {question}\n"
                    f"Answer the question based on the provided images. "
                    f"State only the extracted attributes as a concise comma-separated list."
                )
                try:
                    pred_answer = run_direct_vlm_inference(qwen_engine, candidate_images, prompt)
                except Exception as e_vlm:
                    pred_answer = f"[ERROR: {str(e_vlm)}]"
                latency = round(time.perf_counter() - t0, 3)

                resp_obj: Dict[str, Any] = {
                    "primary_answer": pred_answer,
                    "answers": [pred_answer],
                    "prediction_list": [pred_answer],
                    "telemetry": {"route": "direct_vlm_baseline", "timings_sec": {"vlm_inference": latency}},
                }
            else:
                # E2E Visual Search Pipeline
                task_config = copy.deepcopy(pipeline_config)
                try:
                    resp_obj = get_mssearch_response(
                        sam_model=sam3_engine,
                        zoom_model=qwen_engine,
                        siglip_model=siglip_engine,
                        nlp_model=nlp_parser,
                        annotation={
                            "question": question,
                            "sample_id": sample_id,
                            "answer_type": "attribute_list",
                            "query": query_info,
                        },
                        candidate_images=candidate_images,
                        config=task_config,
                    )
                except Exception as e_pipe:
                    print(f"  [ERROR] Sample {sample_id} pipeline exception: {e_pipe}")
                    resp_obj = {
                        "primary_answer": f"[ERROR: {str(e_pipe)}]",
                        "answers": [],
                        "prediction_list": [],
                        "telemetry": {"route": "error_exception", "timings_sec": {}},
                    }
                latency = round(time.perf_counter() - t0, 3)
                pred_answer = resp_obj.get("primary_answer", "")

            # Compute Precision, Recall, F1
            pred_tokens_candidate = resp_obj.get("prediction_list", []) or resp_obj.get("answers", [])
            metrics = evaluate_predictions(pred_answer, pred_tokens_candidate, gt_list)

            telemetry = resp_obj.get("telemetry", {})
            route = telemetry.get("route", "UNKNOWN")
            routes_executed = telemetry.get("routes_executed", [route])
            stages_executed = telemetry.get("stages_executed", [])
            discovered_evidence = resp_obj.get("evidence_images", [])

            record = {
                "sample_id": sample_id,
                "sample_index": sample_idx,
                "pool_name": pool_name,
                "pool_N": n_candidates,
                "K_gt_images": k_gt_images,
                "query": query_info,
                "question": question,
                "target_object": target_object,
                "target_attribute_type": target_attr_type,
                "ground_truth_list": gt_list,
                "ground_truth_split": gt_split,
                "ground_truth_freeform": gt_freeform,
                "candidate_images": candidate_meta,
                "prediction": pred_answer,
                "predicted_tokens": metrics["pred_items"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "exact_match": metrics["exact_match"],
                "matched_gt": metrics["matched_gt"],
                "matched_pred": metrics["matched_pred"],
                "discovered_evidence_images": discovered_evidence,
                "route": route,
                "routes_executed": routes_executed,
                "stages_executed": stages_executed,
                "latency_sec": latency,
                "timings_sec": telemetry.get("timings_sec", {}),
                "mode": execution_mode,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            completed_task_keys.add(task_key)

            # Console logging
            routes_display = "+".join(routes_executed) if routes_executed else route
            print(
                f"[{run_idx}/{len(remaining_tasks)}] #{sample_idx} ({sample_id}) [{pool_name}|N={n_candidates}|K={k_gt_images}] "
                f"P:{metrics['precision']*100:5.1f}% | R:{metrics['recall']*100:5.1f}% | F1:{metrics['f1']*100:5.1f}% "
                f"| Route: {routes_display:<20} | Time: {latency:6.2f}s | Pred: '{pred_answer}'"
            )

            # Cache clearing
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Generate Aggregate Summary & Stratification
    print("\n" + "=" * 80)
    print("AGGREGATE BENCHMARK RESULTS")
    print("=" * 80)

    all_records: List[Dict[str, Any]] = []
    if os.path.exists(results_jsonl):
        with open(results_jsonl, "r", encoding="utf-8") as rf:
            for line in rf:
                if line.strip():
                    try:
                        all_records.append(json.loads(line))
                    except Exception:
                        pass

    if not all_records:
        print("[Notice] No evaluation records found.")
        return

    def aggregate_metrics(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        count = len(recs)
        if count == 0:
            return {"count": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "avg_latency": 0.0}
        p = sum(r.get("precision", 0.0) for r in recs) / count
        r = sum(r.get("recall", 0.0) for r in recs) / count
        f1 = sum(r.get("f1", 0.0) for r in recs) / count
        em = sum(1.0 if r.get("exact_match", False) else 0.0 for r in recs) / count
        lat = sum(r.get("latency_sec", 0.0) for r in recs) / count
        return {
            "count": count,
            "precision": round(p * 100, 2),
            "recall": round(r * 100, 2),
            "f1": round(f1 * 100, 2),
            "exact_match": round(em * 100, 2),
            "avg_latency_sec": round(lat, 2),
        }

    overall_summary = aggregate_metrics(all_records)

    # Stratify by Pool Size N
    by_pool_n: Dict[str, Any] = {}
    for p_name in sorted(list(set(r.get("pool_name", "UNKNOWN") for r in all_records))):
        sub = [r for r in all_records if r.get("pool_name") == p_name]
        by_pool_n[p_name] = aggregate_metrics(sub)

    # Stratify by Ground-Truth Cardinality K
    by_cardinality_k: Dict[str, Any] = {}
    for k_val in sorted(list(set(r.get("K_gt_images", 1) for r in all_records))):
        sub = [r for r in all_records if r.get("K_gt_images") == k_val]
        by_cardinality_k[f"K_{k_val}"] = aggregate_metrics(sub)

    # Stratify by (N, K) Grid
    by_n_k_grid: Dict[str, Any] = {}
    for p_name in sorted(list(by_pool_n.keys())):
        for k_key in sorted(list(by_cardinality_k.keys())):
            k_val = int(k_key.replace("K_", ""))
            sub = [r for r in all_records if r.get("pool_name") == p_name and r.get("K_gt_images") == k_val]
            if sub:
                by_n_k_grid[f"{p_name}_{k_key}"] = aggregate_metrics(sub)

    # Stratify by Route
    by_route: Dict[str, Any] = {}
    for rt in sorted(list(set(r.get("route", "UNKNOWN") for r in all_records))):
        sub = [r for r in all_records if r.get("route") == rt]
        by_route[rt] = aggregate_metrics(sub)

    full_report = {
        "benchmark": "200_samples_multisource",
        "mode": execution_mode,
        "direct_target_only": args.direct_target_only,
        "model_path": args.model_path,
        "pool_mode": args.pool_mode,
        "total_records": len(all_records),
        "overall": overall_summary,
        "by_pool_N": by_pool_n,
        "by_cardinality_K": by_cardinality_k,
        "by_n_k_grid": by_n_k_grid,
        "by_route": by_route,
        "timestamp": timestamp,
    }

    # Save summary JSON
    summary_path = results_jsonl.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(full_report, sf, indent=2, ensure_ascii=False)

    print(f"Total Evaluations : {overall_summary['count']}")
    print(f"Precision         : {overall_summary['precision']}%")
    print(f"Recall            : {overall_summary['recall']}%")
    print(f"F1 Score          : {overall_summary['f1']}%")
    print(f"Exact Match       : {overall_summary['exact_match']}%")
    print(f"Avg Latency       : {overall_summary['avg_latency_sec']}s")
    print("-" * 80)
    print("Performance by Pool Size N:")
    for pname, pmetrics in by_pool_n.items():
        print(f"  {pname:<6} (N={pmetrics['count']:3d}) -> P: {pmetrics['precision']:5.1f}% | R: {pmetrics['recall']:5.1f}% | F1: {pmetrics['f1']:5.1f}% | EM: {pmetrics['exact_match']:5.1f}% | Lat: {pmetrics['avg_latency_sec']:5.1f}s")
    print("-" * 80)
    print("Performance by Ground-Truth Cardinality K:")
    for kname, kmetrics in by_cardinality_k.items():
        print(f"  {kname:<6} (N={kmetrics['count']:3d}) -> P: {kmetrics['precision']:5.1f}% | R: {kmetrics['recall']:5.1f}% | F1: {kmetrics['f1']:5.1f}% | EM: {kmetrics['exact_match']:5.1f}% | Lat: {kmetrics['avg_latency_sec']:5.1f}s")
    print("=" * 80)
    print(f"Summary JSON saved: {summary_path}")


if __name__ == "__main__":
    main()
