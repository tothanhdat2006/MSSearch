"""
Command-Line Interface (CLI) Runner for STASH-Search.
Supports single-query inference and dataset batch evaluation.
"""

import os
import sys
import argparse
import json
import time
from typing import List, Dict, Any
from PIL import Image, ImageOps

# Ensure repository root is in python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from minimal.config import STASHConfig
from minimal.engine import STASHSearchEngine


def parse_args():
    parser = argparse.ArgumentParser(description="STASH-Search Minimal Runner")
    # Model parameters
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Path or HuggingFace ID of primary MLLM backbone.")
    parser.add_argument("--dual_encoder_name", type=str, default="google/siglip2-so400m-patch16-512",
                        help="Path or HuggingFace ID of SigLIP dual-encoder.")
    parser.add_argument("--nlp_model", type=str, default="en_core_web_sm",
                        help="spaCy model name or path (optional).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Inference device (e.g. 'cuda:0', 'cpu').")

    # Hyperparameters
    parser.add_argument("--k", type=int, default=20,
                        help="Base retention budget per image (default: 20).")
    parser.add_argument("--saliency_gain_thresh", type=float, default=0.04,
                        help="Saliency plateau gain threshold (default: 0.04).")
    parser.add_argument("--consensus_sim", type=float, default=0.85,
                        help="Answer deduplication string similarity (default: 0.85).")

    # Mode 1: Single query execution
    parser.add_argument("--question", type=str, default=None,
                        help="User question for single-query search.")
    parser.add_argument("--images", nargs="+", default=None,
                        help="Paths to gallery images for single-query search.")

    # Mode 2: Batch benchmark execution
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to benchmark JSON or JSONL file.")
    parser.add_argument("--image_dir", type=str, default="",
                        help="Base directory containing gallery images.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Destination JSONL path for saving predictions.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of evaluated samples.")
    parser.add_argument("--quiet", action="store_true",
                        help="Silence stage-level verbose debug logs.")

    return parser.parse_args()


def run_single(engine: STASHSearchEngine, image_paths: List[str], question: str):
    """Executes single question across gallery images."""
    print(f"\n--- Running STASH-Search on {len(image_paths)} images ---")
    print(f"Question: \"{question}\"")

    images = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in image_paths]
    result = engine.search(candidate_images=images, question=question)

    print("\n=== Search Results ===")
    print(f"Predicted Answer   : {result['primary_answer']}")
    print(f"Supporting Image   : {result['primary_evidence_image']} ({image_paths[result['primary_evidence_image']] if result['primary_evidence_image'] is not None else 'None'})")
    print(f"Active Targets     : {result['active_objects']}")
    print(f"Candidate Patches  : {result['num_candidate_patches']}")
    print(f"Inspected Crops    : {result['num_inspected_crops']}")
    print(f"Total Latency      : {result['telemetry']['timings_sec']['total_latency_sec']}s")


def run_batch(engine: STASHSearchEngine, dataset_path: str, image_dir: str, output_file: str, max_samples: int = None):
    """Executes batch evaluation on dataset file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        if dataset_path.endswith(".jsonl"):
            samples = [json.loads(line) for line in f if line.strip()]
        else:
            samples = json.load(f)

    if max_samples:
        samples = samples[:max_samples]

    print(f"\n[Batch Evaluation] Evaluating {len(samples)} samples -> '{output_file}'...")

    with open(output_file, "a", encoding="utf-8") as out_f:
        for idx, sample in enumerate(samples):
            q = sample.get("question", "")
            img_list = sample.get("images", [])

            # Resolve paths
            pil_images = []
            for item in img_list:
                rel_path = item if isinstance(item, str) else item.get("path", "")
                full_path = os.path.join(image_dir, rel_path) if not os.path.isabs(rel_path) else rel_path
                pil_images.append(ImageOps.exif_transpose(Image.open(full_path)).convert("RGB"))

            t0 = time.perf_counter()
            res = engine.search(candidate_images=pil_images, question=q)
            lat = round(time.perf_counter() - t0, 3)

            record = {
                "sample_idx": idx,
                "question": q,
                "predicted_answer": res["primary_answer"],
                "predicted_image_index": res["primary_evidence_image"],
                "active_objects": res["active_objects"],
                "latency_sec": lat,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            print(f"Sample [{idx + 1}/{len(samples)}] Ans: '{res['primary_answer']}' | Img: {res['primary_evidence_image']} ({lat}s)")


def main():
    args = parse_args()

    config = STASHConfig(
        k=args.k,
        saliency_plateau_gain_threshold=args.saliency_gain_thresh,
        answer_dedup_similarity=args.consensus_sim,
        verbose=not args.quiet
    )

    engine = STASHSearchEngine.from_pretrained(
        mllm_name_or_path=args.model_path,
        dual_encoder_name=args.dual_encoder_name,
        config=config,
        device=args.device,
        nlp_model_path=args.nlp_model
    )

    if args.question and args.images:
        run_single(engine, args.images, args.question)
    elif args.dataset_path and args.output_file:
        run_batch(engine, args.dataset_path, args.image_dir, args.output_file, args.max_samples)
    else:
        print("[STASH-Search] Please specify either (--question and --images) or (--dataset_path and --output_file).")
        print("Run with --help for full usage details.")


if __name__ == "__main__":
    main()
