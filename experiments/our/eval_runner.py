import os
import sys
import gc
import json
import time
import argparse
import torch

from eval_dataset import resolve_image, load_and_slice_dataset
from eval_metrics import evaluate_predictions
from eval_models import load_our_engines, run_our_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="MSSearch V2 Evaluation Runner.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Path or HuggingFace ID of VLM checkpoint.")
    parser.add_argument("--sam3_path", type=str, default="/data/shared/cvpr/dat/models/sam3/sam3.pt",
                        help="Path to sam3.pt.")
    parser.add_argument("--siglip_name", type=str, default="/data/shared/cvpr/dat/models/siglip-base-patch16-384",
                        help="Path or HF ID for SigLIP.")
    parser.add_argument("--nlp_model", type=str, default="/data/shared/cvpr/dat/models/spacy/en_core_web_sm-3.8.0",
                        help="Path or name of spaCy model.")
    parser.add_argument("--dataset_path", type=str, default="/data/STASHBench/annotations_part1234_merged.json",
                        help="Path to annotations_part1234_merged.json or 200_samples.json.")
    parser.add_argument("--image_dir", type=str, default="/data/STASHBench/pool",
                        help="Directory containing candidate images or pool.")
    parser.add_argument("--output_dir", type=str, default="./results_our",
                        help="Directory to save JSONL logs.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom config JSON file.")
    parser.add_argument("--pool_mode", type=str, choices=["N_5", "N_10", "N_20", "ALL"], default="N_20",
                        help="Candidate pool size. Default: N_20.")
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
    results_file = args.resume_from or os.path.join(args.output_dir, f"results_our_{m_name}_{args.pool_mode}.jsonl")

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

    # Initialize engines
    engines = load_our_engines(
        model_path=args.model_path,
        sam3_path=args.sam3_path,
        siglip_name=args.siglip_name,
        nlp_model=args.nlp_model,
        config_path=args.config,
        device=args.device
    )

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

            images = [resolve_image(m, args.image_dir) for m in candidate_meta]

            t0 = time.perf_counter()
            resp = run_our_pipeline(engines, images, sample, s_idx)
            latency = round(time.perf_counter() - t0, 3)

            pred_text = resp.get("primary_answer", "")
            pred_list = resp.get("prediction_list", []) or resp.get("answers", [])
            metrics = evaluate_predictions(pred_text, pred_list, gt_list)

            telemetry = resp.get("telemetry", {})
            record = {
                "sample_id": sample_id,
                "sample_index": s_idx,
                "pool_name": pool_name,
                "pool_N": len(candidate_meta),
                "question": question,
                "ground_truth_list": gt_list,
                "prediction": pred_text,
                "predicted_tokens": metrics["pred_items"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "exact_match": metrics["exact_match"],
                "discovered_evidence_images": resp.get("evidence_images", []),
                "route": telemetry.get("route", "UNKNOWN"),
                "routes_executed": telemetry.get("routes_executed", []),
                "latency_sec": latency,
                "timings_sec": telemetry.get("timings_sec", {}),
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            route_str = "+".join(telemetry.get("routes_executed", [])) or telemetry.get("route", "UNKNOWN")
            print(f"[{idx+1}/{len(remaining_tasks)}] #{s_idx} ({sample_id}) [{pool_name}|N={len(candidate_meta)}] "
                  f"P:{metrics['precision']*100:5.1f}% | R:{metrics['recall']*100:5.1f}% | F1:{metrics['f1']*100:5.1f}% "
                  f"| Route:{route_str:<18} | Time:{latency:5.2f}s | Pred:'{pred_text[:50]}'")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n[Completed] MSSearch evaluation finished. Log saved to '{results_file}'.")

if __name__ == "__main__":
    main()
