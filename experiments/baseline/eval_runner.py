import os
import sys
import gc
import glob
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any
import torch

try:
    from eval_dataset import resolve_image, load_and_slice_dataset
    from eval_models import load_vlm_engine, run_vlm_baseline_inference, parse_baseline_json_response
except ModuleNotFoundError:
    from experiments.baseline.eval_dataset import resolve_image, load_and_slice_dataset
    from experiments.baseline.eval_models import load_vlm_engine, run_vlm_baseline_inference, parse_baseline_json_response

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline VLM Benchmark Runner.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Path or HuggingFace ID of VLM checkpoint.")
    parser.add_argument("--dataset_path", type=str, default="/data/STASHBench/annotations_part1234_merged.json",
                        help="Path to annotations_part1234_merged.json or 200_samples.json.")
    parser.add_argument("--image_dir", type=str, default="/data/STASHBench/pool",
                        help="Directory containing candidate images or pool.")
    parser.add_argument("--output_dir", type=str, default="./results_baseline",
                        help="Output directory for results.")
    parser.add_argument("--pool_mode", type=str, choices=["N_5", "N_10", "N_20", "crop", "N_crop", "ALL"], default="N_20",
                        help="Candidate pool size: N_5, N_10, N_20, crop, or ALL.")
    parser.add_argument("--eval_strategy", type=str, choices=["multi_image", "one_by_one"], default="multi_image",
                        help="Evaluation strategy: multi_image (native or composite canvas) or one_by_one.")
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    m_name = os.path.basename(os.path.normpath(args.model_path))
    cands = sorted(glob.glob(os.path.join(args.output_dir, f"results_{m_name}_{args.pool_mode}_*.jsonl")))
    if args.resume_from:
        results_file = args.resume_from
    elif args.resume and cands:
        results_file = cands[-1]
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.output_dir, f"results_{m_name}_{args.pool_mode}_{stamp}.jsonl")

    # Load dataset & slice
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
    print(f"[Task Queue] Total: {len(eval_tasks)} | Done: {len(completed_keys)} | Remaining: {len(remaining_tasks)}")

    # Load model engine
    engine = load_vlm_engine(args.model_path, device=args.device)

    # Execution loop
    with open(results_file, "a", encoding="utf-8") as out_f:
        for idx, (s_idx, sample, pool_name, pool_info) in enumerate(remaining_tasks):
            sample_id = sample.get("sample_id", f"sample_{s_idx}")
            question = sample.get("question") or sample.get("query", {}).get("question", "")

            ans_field = sample.get("answer")
            if isinstance(ans_field, dict):
                gt_list = ans_field.get("list", [])
            elif isinstance(ans_field, list):
                gt_list = ans_field
            elif isinstance(ans_field, str):
                gt_list = [ans_field]
            else:
                gt_list = []
            if "references" in sample and isinstance(sample["references"], list):
                for ref in sample["references"]:
                    if ref and str(ref).strip() and str(ref).strip() not in gt_list:
                        gt_list.append(str(ref).strip())

            candidate_meta = pool_info.get("images", [])

            # Load candidate images
            images = [resolve_image(img_meta, args.image_dir) for img_meta in candidate_meta]

            t0 = time.perf_counter()
            pred_text, raw_list = run_vlm_baseline_inference(
                engine, images, question, eval_strategy=args.eval_strategy, max_new_tokens=args.max_new_tokens
            )
            latency = round(time.perf_counter() - t0, 3)

            # Parse JSON for image_ids and free-form answer
            raw_ids, actual_img_ids, img_indices, pred_answer = parse_baseline_json_response(pred_text, candidate_meta)
            gt_img = pool_info.get("gt_image") or sample.get("gt_image")
            gt_img_id = gt_img.get("image_id") if isinstance(gt_img, dict) else (gt_img or "")
            is_img_correct = (str(gt_img_id) in [str(x) for x in actual_img_ids]) if (actual_img_ids and gt_img_id) else None

            record = {
                "sample_id": sample_id,
                "sample_index": s_idx,
                "pool_name": pool_name,
                "pool_N": len(candidate_meta),
                "question": question,
                "ground_truth_list": gt_list,
                "gt_image_id": gt_img_id,
                "pred_image_ids": actual_img_ids,
                "pred_image_indices": img_indices,
                "is_image_correct": is_img_correct,
                "pred_answer": pred_answer,
                "prediction": pred_answer,
                "raw_response": pred_text,
                "latency_sec": latency,
                "eval_strategy": args.eval_strategy,
            }

            # Atomic disk flush
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            status = "OK" if is_img_correct else "MISS"
            print(f"[{idx+1}/{len(remaining_tasks)}] #{s_idx} ({sample_id}) [{pool_name}|N={len(candidate_meta)}] "
                  f"Img:{actual_img_ids}({status}) | Time:{latency:5.2f}s | Ans: '{pred_answer[:40]}'")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n[Completed] Evaluation finished. Log saved to '{results_file}'.")

if __name__ == "__main__":
    main()
