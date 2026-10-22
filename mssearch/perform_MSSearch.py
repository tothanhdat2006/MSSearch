"""
Batch Execution Runner for Dual-Path Multi-Source Visual Search (MSSearch).
Refactored directly from perform_CVSearch.py with multi-image ingestion and zero-heuristic path resolution.
"""

import sys
import os
import argparse
import json
import warnings
import gc
import torch
import spacy
import numpy as np
from PIL import Image
from tqdm import tqdm

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from mssearch.config import MSSearchConfig
from mssearch.models.modeling_qwenvl import ModelQwenVL
from mssearch.models.modeling_sam3 import sam3_inference
from mssearch.models.modeling_siglip import SigLIPScorer
from mssearch.MSSearch import get_mssearch_response

warnings.filterwarnings("ignore")


def get_chunk(lst, n, k):
    """Splits list into n interleaved chunks and returns chunk k."""
    subarrays = [[] for _ in range(n)]
    for i in range(n):
        subarrays[i] = lst[i::n]
    return subarrays[k]


def resolve_candidate_image_path(base_dir: str, img_rel_path: str) -> str:
    """
    Direct, explicit path resolution without ambiguous heuristics.
    """
    if os.path.isabs(img_rel_path):
        return img_rel_path
    return os.path.join(base_dir, img_rel_path)


def main():
    parser = argparse.ArgumentParser(description="MSSearch Batch Runner on 2x T4 Kaggle")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--annotation-path", type=str, required=True)
    parser.add_argument("--base-data-dir", type=str, default="")
    parser.add_argument("--benchmark", type=str, default="hr-bench_multisource")
    parser.add_argument("--sam-model-path", type=str, default="/kaggle/input/datasets/thnhtt/vs-sam3/sam3/sam3/sam3.pt")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--siglip-model-name", type=str, default="google/siglip-base-patch16-384")
    parser.add_argument("--nlp-model-path", type=str, default="en_core_web_sm")

    # Ablation & Diagnostic Flags
    parser.add_argument("--disable-direct-answer", action="store_true", help="Disable Phase 1 Direct Answer probing")
    parser.add_argument("--direct-answer-threshold", type=float, default=0.60, help="Confidence threshold for Phase 1 Direct Answer")
    parser.add_argument("--disable-fast-path", action="store_true", help="Disable Stage 2 Tool-Based Fast Path")
    parser.add_argument("--fast-path-threshold", type=float, default=0.70, help="Confidence threshold for Stage 2 Fast Path")
    parser.add_argument("--early-consensus-threshold", type=float, default=0.75, help="Similarity threshold for Stage 6 early consensus exit")
    parser.add_argument("--disable-visual-prompt-bbox", action="store_true")
    parser.add_argument("--enable-textual-prompt-coords", action="store_true")
    parser.add_argument("--entropy-threshold", type=float, default=0.35)
    parser.add_argument("--partition-mode", type=str, default="hybrid", choices=["hybrid", "quadrant_only", "sgap_only"])
    parser.add_argument("--partition-stopping-mode", type=str, default="saliency_plateau", choices=["saliency_plateau", "fixed_size"], help="Stopping mode for Stage 3 partitioning: dynamic saliency plateau vs legacy fixed max_patch_size")
    parser.add_argument("--max-patch-size", type=int, default=512, help="Patch size constraint when partition-stopping-mode is fixed_size")
    parser.add_argument("--min-plateau-size", type=int, default=512, help="Scale floor for saliency plateau stopping")
    parser.add_argument("--max-plateau-size", type=int, default=1536, help="Scale ceiling for saliency plateau stopping (GPU safety)")
    parser.add_argument("--saliency-plateau-gain-threshold", type=float, default=0.04, help="Minimum saliency gain required to descend deeper")
    parser.add_argument("--bbox-expansion-factor", type=float, default=1.25)
    parser.add_argument("--answer-type", "--answer_type", dest="answer_type", type=str, default="free_form",
                        choices=["free_form", "option_single", "multiple_choice", "mc"],
                        help="Prompt answer format: 'free_form' (default open-ended prompt without options) or 'option_single' / 'multiple_choice' (appends candidate options)")
    parser.add_argument("--include-options-in-prompt", "--add-options-to-prompt", dest="include_options_in_prompt", action="store_true",
                        help="Explicit flag to include options in the reading prompt (equivalent to --answer-type option_single)")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose debug logging on fallback/non-fallback decisions")
    parser.add_argument("--quiet", action="store_true", help="Silence verbose debug logging")

    args = parser.parse_args()
    is_verbose = args.verbose and not args.quiet

    # Configure output file path
    if args.answers_file is None:
        answers_dir = f"/kaggle/working/answers/{args.benchmark}"
        os.makedirs(answers_dir, exist_ok=True)
        args.answers_file = os.path.join(answers_dir, f"mssearch_chunk{args.chunk_idx}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(args.answers_file)), exist_ok=True)
    print(f"[MSSearch] Destination file: {args.answers_file}")

    # Build typed configuration dataclass
    opt_in_prompt = args.include_options_in_prompt or (args.answer_type in ["option_single", "option_list", "multiple_choice", "mc"])
    chosen_answer_type = "option_single" if args.include_options_in_prompt and args.answer_type == "free_form" else args.answer_type

    config = MSSearchConfig(
        enable_direct_answer=not args.disable_direct_answer,
        direct_answer_conf_threshold=args.direct_answer_threshold,
        enable_fast_path=not args.disable_fast_path,
        fast_path_conf_threshold=args.fast_path_threshold,
        use_mllm_object_parser=True,
        early_consensus_threshold=args.early_consensus_threshold,
        enable_visual_prompt_bbox=not args.disable_visual_prompt_bbox,
        enable_textual_prompt_coords=args.enable_textual_prompt_coords,
        entropy_threshold=args.entropy_threshold,
        partition_mode=args.partition_mode,
        partition_stopping_mode=args.partition_stopping_mode,
        max_patch_size=args.max_patch_size,
        min_plateau_size=args.min_plateau_size,
        max_plateau_size=args.max_plateau_size,
        saliency_plateau_gain_threshold=args.saliency_plateau_gain_threshold,
        bbox_expansion_factor=args.bbox_expansion_factor,
        include_options_in_prompt=opt_in_prompt,
        answer_type=chosen_answer_type,
        siglip_model_name=args.siglip_model_name,
        verbose=is_verbose
    )

    # 1. Load Qwen2.5-VL in float16 (Asymmetric Sharding across 2x T4)
    print("[MSSearch] Loading Qwen2.5-VL on dual-T4 with SDPA backend...")
    search_model = ModelQwenVL(
        model_path=args.model_path,
        torch_dtype=torch.float16,
        patch_scale=1.2
    )

    # 2. Load SAM3 on cuda:1
    sam3_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
    print(f"[MSSearch] Loading SAM3 on {sam3_device}...")
    sam3 = sam3_inference(model_path=args.sam_model_path, device=sam3_device)

    # 3. Load SigLIP-base on cuda:0 (Shared memory space with Qwen2.5-VL Vision Tower)
    print(f"[MSSearch] Loading SigLIP Scorer on cuda:0...")
    siglip = SigLIPScorer(model_name=args.siglip_model_name, device="cuda:0")

    # 4. Load SpaCy NLP Parser
    try:
        nlp = spacy.load(args.nlp_model_path)
    except Exception:
        nlp = spacy.load("en_core_web_sm")

    # 5. Load Annotations
    print(f"[MSSearch] Loading annotations from: {args.annotation_path}")
    with open(args.annotation_path, "r", encoding="utf-8") as f:
        if args.annotation_path.endswith(".jsonl"):
            annotations = [json.loads(line) for line in f if line.strip()]
        else:
            annotations = json.load(f)

    chunk_data = get_chunk(annotations, args.num_chunks, args.chunk_idx)
    print(f"[MSSearch] Total samples: {len(annotations)} | Chunk {args.chunk_idx}/{args.num_chunks}: {len(chunk_data)}")

    # In-context examples if available
    ic_path = os.path.join(root_dir, f"mssearch/ic_examples/{args.benchmark}.json")
    ic_examples = json.load(open(ic_path, "r")) if os.path.exists(ic_path) else None

    # Base dataset directory resolution
    base_data_dir = args.base_data_dir if args.base_data_dir else os.path.dirname(os.path.abspath(args.annotation_path))

    with open(args.answers_file, "w", encoding="utf-8") as f_out:
        for idx, anno in enumerate(tqdm(chunk_data, desc=f"MSSearch Chunk {args.chunk_idx}")):
            # Support both 'candidate_images' and legacy 'input_image' fallback
            candidate_paths = anno.get('candidate_images', [])
            if not candidate_paths and 'input_image' in anno:
                candidate_paths = [anno['input_image']]
                if is_verbose:
                    print(f"\n[MSSearch][Image Fallback] Sample {anno.get('id', idx)}: 'candidate_images' absent; falling back to single 'input_image'.")
            elif is_verbose:
                print(f"\n[MSSearch][Non-Fallback] Sample {anno.get('id', idx)}: Found {len(candidate_paths)} candidate images in annotation.")

            # Explicit hardcoded path load
            candidate_images: List[Image.Image] = []
            for rel_p in candidate_paths:
                full_p = resolve_candidate_image_path(base_data_dir, rel_p)
                if os.path.exists(full_p):
                    candidate_images.append(Image.open(full_p).convert("RGB"))
                elif is_verbose:
                    print(f"[MSSearch][Path Warning] Image does not exist on disk: '{full_p}'")

            if not candidate_images:
                print(f"[MSSearch] Warning: No candidate images found on disk for sample {anno.get('id', idx)}!")
                continue

            if chosen_answer_type != "free_form":
                anno['answer_type'] = chosen_answer_type
            elif 'answer_type' not in anno:
                anno['answer_type'] = 'free_form'

            resp = get_mssearch_response(
                sam_model=sam3,
                zoom_model=search_model,
                siglip_model=siglip,
                nlp_model=nlp,
                annotation=anno,
                candidate_images=candidate_images,
                config=config,
                ic_examples=ic_examples
            )


            # Record enriched outputs into annotation
            anno['mssearch_output'] = resp
            anno['output'] = resp.get('primary_answer', '')
            anno['mssearch_prediction'] = resp.get('primary_answer', '')
            anno['raw_vlm_response'] = resp.get('raw_vlm_response', '')
            anno['discovered_evidence_images'] = resp.get('evidence_images', [])
            anno['all_answers'] = resp.get('answers', [])

            f_out.write(json.dumps(anno) + "\n")
            f_out.flush()

            # Dynamic Memory Shield: Periodic CUDA flush
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n[SUCCESS] Completed Chunk {args.chunk_idx}! Saved to: {args.answers_file}")


if __name__ == "__main__":
    main()
