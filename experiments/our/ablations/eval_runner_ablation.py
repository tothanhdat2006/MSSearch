"""
Evaluation runner for MSSearch ablation studies.
Ablates Detection Backbone (SAM3 vs Grounding DINO vs Florence-2 vs DINOv3),
Screening Scorer (SigLIP vs SigLIP2 vs CLIP-B/32 vs CLIP-L/14 vs EVA-02-L/14),
Gating Threshold (0.3 vs 0.5 vs 0.7), and Component Increments (Qwen -> Fast Path -> Rigid -> SGAP -> ASP -> Aggregation).
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

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_OUR_DIR = os.path.abspath(os.path.join(_CUR_DIR, ".."))
_MSSEARCH_ROOT = os.path.abspath(os.path.join(_CUR_DIR, "../../.."))
for p in [_MSSEARCH_ROOT, _OUR_DIR, _CUR_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_dataset import resolve_image, load_and_slice_dataset
from eval_metrics import compute_stage_ret_rej_tokens, format_stage6_answers, format_voting_scores
from eval_models_ablation import load_ablation_engines, run_ablation_pipeline
from eval_runner_utils import load_completed_keys, resolve_ground_truth


def load_env_paths():
    """Loads ablation and base environment paths if available."""
    import re
    for f_name in ["env_paths.sh", "env_paths_ablation.sh"]:
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
                        m = re.match(r"^\$\{([A-Za-z0-9_]+):-(.*)\}$", v)
                        if m:
                            var_name, default_val = m.groups()
                            v = os.environ.get(var_name, default_val)
                        if "${SAM3_PATH}" in v and "SAM3_PATH" in os.environ:
                            v = v.replace("${SAM3_PATH}", os.environ["SAM3_PATH"])
                        if "${MODEL_DIR}" in v and "MODEL_DIR" in os.environ:
                            v = v.replace("${MODEL_DIR}", os.environ["MODEL_DIR"])
                        if k not in os.environ and v:
                            os.environ[k] = v


def parse_args():
    load_env_paths()
    parser = argparse.ArgumentParser(description="MSSearch Ablation Study Runner.")

    # Component Increment Ablation
    parser.add_argument("--component", type=str, default=None,
                        choices=["qwen", "qwen_fast_path", "qwen_fp_rigid_cg_single",
                                 "qwen_fp_sgap_cg_single", "qwen_fp_asp_cg_single", "qwen_fp_asp_cg_agg"],
                        help="Component increment ablation variant.")

    # Detector & Scorer Backbones
    parser.add_argument("--detector", type=str, default=os.environ.get("DETECTOR_TYPE", "sam3"),
                        choices=["sam3", "grounding_dino", "florence2", "dinov3"])
    parser.add_argument("--detector_path", type=str, default=os.environ.get("DETECTOR_PATH", os.environ.get("SAM3_PATH", None)))
    parser.add_argument("--scorer", type=str, default=os.environ.get("SCORER_TYPE", "siglip"),
                        choices=["siglip", "siglip2_384", "siglip2_512", "clip_b32", "clip_l14", "eva02_l14"])
    parser.add_argument("--scorer_path", type=str, default=os.environ.get("SCORER_PATH", None))
    parser.add_argument("--gate_threshold", type=float, default=None)

    # Base Pipeline Settings
    parser.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--nlp_model", type=str, default=os.environ.get("NLP_MODEL", "/data/shared/cvpr/dat/models/spacy/en_core_web_sm-3.8.0"))
    parser.add_argument("--dataset_path", type=str, default=os.environ.get("DATASET_PATH", "/data/STASHBench/annotations_part1234_merged.json"))
    parser.add_argument("--image_dir", type=str, default=os.environ.get("IMAGE_DIR", "/data/STASHBench/pool"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", "./results_ablation"))
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pool_mode", type=str, choices=["N_1", "N_5", "N_10", "N_20", "crop", "N_crop", "ALL"], default="N_20")

    # Execution Controls
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

    if args.component:
        prefix = f"results_ablation_component_{args.component}_{m_name}_{args.pool_mode}"
    else:
        gate_tag = f"_gate{args.gate_threshold}" if args.gate_threshold is not None else ""
        prefix = f"results_ablation_{args.detector}_{args.scorer}{gate_tag}_{m_name}_{args.pool_mode}"

    cands = sorted(glob.glob(os.path.join(args.output_dir, f"{prefix}_*.jsonl")))
    results_file = args.resume_from or (cands[-1] if (args.resume and cands) else os.path.join(args.output_dir, f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"))
    config_file = os.path.join(os.path.dirname(os.path.abspath(results_file)), os.path.basename(results_file).replace("results_ablation_", "config_ablation_").replace(".jsonl", ".json"))

    all_samples, eval_tasks = load_and_slice_dataset(
        args.dataset_path, args.pool_mode, args.sample_idx, args.start_idx, args.end_idx, args.max_samples
    )

    completed_keys = load_completed_keys(results_file) if (args.resume or args.resume_from) else set()
    if completed_keys:
        print(f"[Auto-Resume] Found {len(completed_keys)} completed tasks in '{results_file}'.")

    remaining_tasks = [t for t in eval_tasks if f"{t[1].get('sample_id')}_{t[2]}" not in completed_keys]
    print(f"[Ablation Queue] Component: {args.component or 'N/A'} | Detector: {args.detector} | Scorer: {args.scorer} | Gate: {args.gate_threshold or 0.5} | Tasks: {len(remaining_tasks)}/{len(eval_tasks)}")

    engines = load_ablation_engines(
        model_path=args.model_path, detector_type=args.detector, detector_path=args.detector_path,
        scorer_type=args.scorer, scorer_name=args.scorer_path, nlp_model=args.nlp_model,
        config_path=args.config, gate_threshold=args.gate_threshold, component=args.component, device=args.device,
    )
    engines["config"].save_json(config_file)
    print(f"[Config Saved] Ablation config snapshot saved to '{config_file}'.")

    with open(results_file, "a", encoding="utf-8") as out_f:
        for idx, (s_idx, sample, pool_name, pool_info) in enumerate(remaining_tasks):
            sample_id = sample.get("sample_id", f"sample_{s_idx}")
            question = sample.get("question") or sample.get("query", {}).get("question", "")
            gt_list = resolve_ground_truth(sample)

            candidate_meta = pool_info.get("images", [])
            images = [resolve_image(m, args.image_dir) for m in candidate_meta]

            t0 = time.perf_counter()
            resp = run_ablation_pipeline(engines, images, sample, s_idx)
            latency = round(time.perf_counter() - t0, 3)

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
                "model_name": m_name, "component": args.component, "detector_type": args.detector, "scorer_type": args.scorer,
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
                "latency_sec": latency, "primary_answer_score": telemetry.get("primary_answer_score"),
                "stage6_answers": format_stage6_answers(telemetry, candidate_meta),
                "voting_scores": format_voting_scores(telemetry, candidate_meta),
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush(); os.fsync(out_f.fileno())

            for img in images:
                if img is not None:
                    try: img.close()
                    except Exception: pass
            images.clear(); gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            route_str = "+".join(telemetry.get("routes_executed", [])) or telemetry.get("route", "UNKNOWN")
            print(f"[{idx+1}/{len(remaining_tasks)}] #{s_idx} ({sample_id}) [{pool_name}|N={len(candidate_meta)}] "
                  f"ImgOk:{is_img_correct} | Route:{route_str:<18} | Time:{latency:5.2f}s | Pred:'{pred_text[:40]}'")

    print(f"[Ablation Complete] All outputs saved to '{results_file}'.")


if __name__ == "__main__":
    main()
