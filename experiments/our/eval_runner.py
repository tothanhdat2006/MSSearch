import os
import sys
import gc
import glob
import json
import time
import argparse
from datetime import datetime
import torch

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "../.."))
for _p in [_REPO_ROOT, _CUR_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from eval_dataset import resolve_image, load_and_slice_dataset
    from eval_metrics import compute_stage_ret_rej_tokens
    from eval_models import load_our_engines, run_our_pipeline
except ModuleNotFoundError:
    from experiments.our.eval_dataset import resolve_image, load_and_slice_dataset
    from experiments.our.eval_metrics import compute_stage_ret_rej_tokens
    from experiments.our.eval_models import load_our_engines, run_our_pipeline

def load_env_paths():
    for f_name in ["env_paths.sh", "paths.env"]:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f_name)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("export "):
                        line = line[7:].strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if "${SAM3_PATH}" in v and "SAM3_PATH" in os.environ:
                            v = v.replace("${SAM3_PATH}", os.environ["SAM3_PATH"])
                        if "${MODEL_DIR}" in v and "MODEL_DIR" in os.environ:
                            v = v.replace("${MODEL_DIR}", os.environ["MODEL_DIR"])
                        if k not in os.environ and v:
                            os.environ[k] = v
            break

def parse_args():
    load_env_paths()
    parser = argparse.ArgumentParser(description="MSSearch V2 Evaluation Runner.")
    parser.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"),
                        help="Path or HuggingFace ID of VLM checkpoint.")
    parser.add_argument("--sam3_path", type=str, default=os.environ.get("SAM3_PATH", "/data/shared/cvpr/dat/models/sam3/sam3.pt"),
                        help="Path to sam3.pt.")
    parser.add_argument("--siglip_name", type=str, default=os.environ.get("SIGLIP_NAME", "/data/shared/cvpr/dat/models/siglip2-so400m-patch16-512"),
                        help="Path or HF ID for SigLIP.")
    parser.add_argument("--nlp_model", type=str, default=os.environ.get("NLP_MODEL", "/data/shared/cvpr/dat/models/spacy/en_core_web_sm-3.8.0"),
                        help="Path or name of spaCy model.")
    parser.add_argument("--dataset_path", type=str, default=os.environ.get("DATASET_PATH", "/data/STASHBench/annotations_part1234_merged.json"),
                        help="Path to annotations_part1234_merged.json or 200_samples.json.")
    parser.add_argument("--image_dir", type=str, default=os.environ.get("IMAGE_DIR", "/data/STASHBench/pool"),
                        help="Directory containing candidate images or pool.")
    parser.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", "./results_our"),
                        help="Directory to save JSONL logs.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom config JSON file.")
    parser.add_argument("--pool_mode", type=str, choices=["N_1", "N_5", "N_10", "N_20", "crop", "N_crop", "ALL"], default="N_20",
                        help="Candidate pool size: N_1, N_5, N_10, N_20, crop, or ALL.")
    parser.add_argument("--sample_idx", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gate_threshold", type=float, default=None,
                        help="VLM gating probability threshold in [0.0, 1.0] (e.g. 0.3, 0.5, 0.7).")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    m_name = os.path.basename(os.path.normpath(args.model_path))
    cands = sorted(glob.glob(os.path.join(args.output_dir, f"results_our_{m_name}_{args.pool_mode}_*.jsonl")))
    if args.resume_from:
        results_file = args.resume_from
    elif args.resume and cands:
        results_file = cands[-1]
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.output_dir, f"results_our_{m_name}_{args.pool_mode}_{stamp}.jsonl")
    res_dir = os.path.dirname(os.path.abspath(results_file))
    cfg_base = os.path.basename(results_file).replace("results_our_", "config_our_").replace(".jsonl", ".json")
    config_file = os.path.join(res_dir, cfg_base)

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
        gate_threshold=args.gate_threshold,
        device=args.device
    )
    engines["config"].save_json(config_file)
    print(f"[Config Saved] Config snapshot saved to '{config_file}'.")

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

            candidate_meta = pool_info.get("images", [])

            images = [resolve_image(m, args.image_dir) for m in candidate_meta]

            gpus = range(torch.cuda.device_count() if torch.cuda.is_initialized() else 0)
            for d in gpus:
                torch.cuda.reset_peak_memory_stats(d)
            ooms_before = sum(torch.cuda.memory_stats(d).get("num_ooms", 0) for d in gpus)

            t0 = time.perf_counter()
            resp = run_our_pipeline(engines, images, sample, s_idx)
            latency = round(time.perf_counter() - t0, 3)

            # stages catch exceptions and keep going, so OOMs are read from the allocator
            cuda_stats = {}
            if gpus:
                cuda_stats = {
                    "cuda_ooms": sum(torch.cuda.memory_stats(d).get("num_ooms", 0) for d in gpus) - ooms_before,
                    "peak_mem_gib": round(max(torch.cuda.max_memory_allocated(d) for d in gpus) / 2**30, 2),
                }
                if cuda_stats["cuda_ooms"]:
                    print(f"[OOM Count] sample {sample_id}: {cuda_stats['cuda_ooms']} OOM(s) caught inside the pipeline")

            pred_text = resp.get("primary_answer", "")
            telemetry = resp.get("telemetry", {})

            gt_img = pool_info.get("gt_image") or sample.get("gt_image")
            gt_img_id = gt_img.get("image_id") if isinstance(gt_img, dict) else (gt_img or "")
            gt_idx = pool_info.get("gt_index")
            gt_img_idx = (gt_idx - 1) if gt_idx is not None else next((i for i, m in enumerate(candidate_meta) if str(m.get("image_id")) == str(gt_img_id)), None)

            evidence_indices = telemetry.get("evidence_images", [])
            evidence_img_ids = [
                candidate_meta[i].get("image_id") if isinstance(candidate_meta[i], dict) else str(candidate_meta[i])
                for i in evidence_indices if 0 <= i < len(candidate_meta)
            ]
            is_img_correct = (str(gt_img_id) in [str(x) for x in evidence_img_ids]) if (evidence_img_ids and gt_img_id) else (False if gt_img_id else None)

            prim_img_idx = telemetry.get("primary_evidence_image")
            if prim_img_idx is None and evidence_indices:
                prim_img_idx = evidence_indices[0]
            prim_img_id = (
                candidate_meta[prim_img_idx].get("image_id") if isinstance(candidate_meta[prim_img_idx], dict) else str(candidate_meta[prim_img_idx])
            ) if (prim_img_idx is not None and 0 <= prim_img_idx < len(candidate_meta)) else None
            is_prim_img_correct = (str(gt_img_id) == str(prim_img_id)) if (gt_img_id and prim_img_id) else (False if gt_img_id else None)

            stage_stats = compute_stage_ret_rej_tokens(telemetry, gt_img_idx, sample.get("target_bbox"), images)

            record = {
                "sample_id": sample_id, "sample_index": s_idx, "pool_name": pool_name, "pool_N": len(candidate_meta),
                "gate_threshold": getattr(engines["config"], "gate_threshold", args.gate_threshold or 0.5),
                "question": question, "ground_truth_list": gt_list, "gt_image_id": gt_img_id, "gt_image_idx": gt_img_idx,
                "primary_pred_image_id": prim_img_id, "primary_pred_image_idx": prim_img_idx, "is_primary_image_correct": is_prim_img_correct,
                "pred_image_ids": evidence_img_ids, "pred_image_indices": evidence_indices, "is_image_correct": is_img_correct,
                "prediction": pred_text, "route": telemetry.get("route", "UNKNOWN"),
                "stage5_mode": telemetry.get("stage5_mode", ""), "stages_executed": telemetry.get("stages_executed", []),
                "num_unified_crops": telemetry.get("num_unified_crops", 0), "timings_sec": telemetry.get("timings_sec", {}),
                "stage_retention": stage_stats["stage_retention"], "stage_rejection": stage_stats["stage_rejection"],
                "visual_tokens": stage_stats["visual_tokens"], "stage6_crop_sizes": stage_stats["stage6_crop_sizes"],
                "has_gt_bbox": stage_stats["has_gt_bbox"], "max_iou_stage4": stage_stats["max_iou_stage4"],
                "gt_survived_stage4": stage_stats["gt_survived_stage4"], "distractor_imgs_pruned": stage_stats["distractor_imgs_pruned"],
                "stage4_image_scores": stage_stats["stage4_image_scores"], "gt_rank_stage4": stage_stats["gt_rank_stage4"],
                "stage4_r1": stage_stats["stage4_r1"], "stage4_mrr": stage_stats["stage4_mrr"],
                "s6_visual_tokens": stage_stats["s6_visual_tokens"],
                "latency_sec": latency,
                **cuda_stats,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            route_str = "+".join(telemetry.get("routes_executed", [])) or telemetry.get("route", "UNKNOWN")
            print(f"[{idx+1}/{len(remaining_tasks)}] #{s_idx} ({sample_id}) [{pool_name}|N={len(candidate_meta)}] "
                  f"ImgOk:{is_img_correct} | Route:{route_str:<18} | Time:{latency:5.2f}s | Pred:'{pred_text[:50]}'")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n[Completed] MSSearch evaluation finished. Log saved to '{results_file}'.")

if __name__ == "__main__":
    main()
