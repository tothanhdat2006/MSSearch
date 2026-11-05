"""
Evaluation runner for MSSearch ablation studies.
Ablates Detection Backbone (SAM3 vs Grounding DINO vs DINO-X)
and Screening Scorer (SigLIP vs CLIP-ViT-B/32 vs CLIP-ViT-L/14 vs EVA-02-L/14).
"""

import os
import sys
import gc
import glob
import json
import time
import argparse
from datetime import datetime
import torch

# Ensure parent and repo root are in sys.path
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_OUR_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
_MSSEARCH_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "../../.."))
for p in [_MSSEARCH_ROOT, _OUR_DIR, _CUR_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_dataset import resolve_image, load_and_slice_dataset
from eval_metrics import compute_stage_ret_rej_tokens
from eval_models_ablation import load_ablation_engines, run_ablation_pipeline


def load_env_paths():
    """Loads ablation and base environment paths if available."""
    for f_name in ["env_paths_ablation.sh", "env_paths.sh"]:
        p = os.path.join(_CUR_DIR, f_name)
        if not os.path.isfile(p):
            p = os.path.join(_OUR_DIR, f_name)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("export "):
                        line = line[7:].strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if "${MODEL_DIR}" in v and "MODEL_DIR" in os.environ:
                            v = v.replace("${MODEL_DIR}", os.environ["MODEL_DIR"])
                        if k not in os.environ and v:
                            os.environ[k] = v


def parse_args():
    load_env_paths()
    parser = argparse.ArgumentParser(description="MSSearch Ablation Study Runner.")

    # Ablation Dimension 1: Detector Backbone
    parser.add_argument("--detector", type=str, default=os.environ.get("DETECTOR_TYPE", "sam3"),
                        choices=["sam3", "grounding_dino", "dinox"],
                        help="Detection backbone: 'sam3', 'grounding_dino', or 'dinox'.")
    parser.add_argument("--detector_path", type=str, default=os.environ.get("DETECTOR_PATH", None),
                        help="Path or HF ID for detection checkpoint.")

    # Ablation Dimension 2: Visual Screening Scorer
    parser.add_argument("--scorer", type=str, default=os.environ.get("SCORER_TYPE", "siglip"),
                        choices=["siglip", "clip_b32", "clip_l14", "eva02_l14"],
                        help="Visual screening scorer: 'siglip', 'clip_b32', 'clip_l14', or 'eva02_l14'.")
    parser.add_argument("--scorer_path", type=str, default=os.environ.get("SCORER_PATH", None),
                        help="Path or HF ID for screening checkpoint.")

    # Base Pipeline Settings
    parser.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"),
                        help="Primary VLM path.")
    parser.add_argument("--nlp_model", type=str, default=os.environ.get("NLP_MODEL", "en_core_web_sm"),
                        help="spaCy model path or name.")
    parser.add_argument("--dataset_path", type=str, default=os.environ.get("DATASET_PATH", "/data/STASHBench/annotations_part1234_merged.json"),
                        help="Dataset JSON path.")
    parser.add_argument("--image_dir", type=str, default=os.environ.get("IMAGE_DIR", "/data/STASHBench/pool"),
                        help="Pool image directory.")
    parser.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", "./results_ablation"),
                        help="Output results directory.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional JSON config override.")
    parser.add_argument("--pool_mode", type=str, choices=["N_1", "N_5", "N_10", "N_20", "crop", "N_crop", "ALL"], default="N_20",
                        help="Candidate pool size.")

    # Subsetting & Execution Controls
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    m_name = os.path.basename(os.path.normpath(args.model_path))

    prefix = f"results_ablation_{args.detector}_{args.scorer}_{m_name}_{args.pool_mode}"
    cands = sorted(glob.glob(os.path.join(args.output_dir, f"{prefix}_*.jsonl")))

    if args.resume_from:
        results_file = args.resume_from
    elif args.resume and cands:
        results_file = cands[-1]
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.output_dir, f"{prefix}_{stamp}.jsonl")

    res_dir = os.path.dirname(os.path.abspath(results_file))
    cfg_base = os.path.basename(results_file).replace("results_ablation_", "config_ablation_").replace(".jsonl", ".json")
    config_file = os.path.join(res_dir, cfg_base)

    # Load dataset
    all_samples, eval_tasks = load_and_slice_dataset(
        args.dataset_path, args.pool_mode, args.sample_idx, args.start_idx, args.end_idx, args.max_samples
    )

    # Resume handling
    completed_keys = set()
    if args.resume or args.resume_from:
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            completed_keys.add(f"{rec.get('sample_id')}_{rec.get('pool_name')}")
                        except Exception:
                            pass
            print(f"[Auto-Resume] Found {len(completed_keys)} completed tasks in '{results_file}'.")

    remaining_tasks = [t for t in eval_tasks if f"{t[1].get('sample_id')}_{t[2]}" not in completed_keys]
    print(f"[Ablation Queue] Detector: {args.detector} | Scorer: {args.scorer} | Tasks: {len(remaining_tasks)}/{len(eval_tasks)}")

    # Initialize engines
    engines = load_ablation_engines(
        model_path=args.model_path,
        detector_type=args.detector,
        detector_path=args.detector_path,
        scorer_type=args.scorer,
        scorer_name=args.scorer_path,
        nlp_model=args.nlp_model,
        config_path=args.config,
        device=args.device,
    )
    engines["config"].save_json(config_file)
    print(f"[Config Saved] Ablation config snapshot saved to '{config_file}'.")

    # Evaluation loop
    with open(results_file, "a", encoding="utf-8") as out_f:
        for idx, (s_idx, sample, pool_name, pool_info) in enumerate(remaining_tasks):
            t_start = time.time()
            s_id = sample.get("sample_id", f"sample_{s_idx}")
            pool_paths = pool_info.get("images", [])

            loaded_images = []
            for p in pool_paths:
                img = resolve_image(p, args.image_dir)
                if img is not None:
                    loaded_images.append(img)

            if not loaded_images:
                print(f"[Skip] Sample {s_id} ({pool_name}): no valid images resolved.")
                continue

            try:
                resp = run_ablation_pipeline(engines, loaded_images, sample, s_idx)
            except Exception as e:
                print(f"[Error] Pipeline execution failed on {s_id}: {e}")
                resp = {"response": "", "error": str(e), "telemetry": {}}

            t_elapsed = time.time() - t_start

            record = {
                "sample_id": s_id,
                "dataset_sample_idx": s_idx,
                "pool_name": pool_name,
                "pool_size": len(loaded_images),
                "model_name": m_name,
                "detector_type": args.detector,
                "scorer_type": args.scorer,
                "prediction": resp.get("response", ""),
                "ground_truth": sample.get("answer", ""),
                "question": sample.get("question", ""),
                "time_taken": round(t_elapsed, 3),
                "telemetry": resp.get("telemetry", {}),
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            for img in loaded_images:
                try:
                    img.close()
                except Exception:
                    pass
            loaded_images.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[{idx+1}/{len(remaining_tasks)}] {s_id} ({pool_name}) -> {record['prediction'][:40]} ({t_elapsed:.2f}s)")

    print(f"[Ablation Complete] All outputs saved to '{results_file}'.")


if __name__ == "__main__":
    main()
