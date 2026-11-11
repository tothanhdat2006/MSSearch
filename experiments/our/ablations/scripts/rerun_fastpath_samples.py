#!/usr/bin/env python3
"""
Standalone Local Replay Tool for Fast-Path Affected Samples.
Identifies samples where fast path was triggered ('waterfall_cascade' or 'tool_based_fast_path'),
re-evaluates only those sample indices with fast path strictly disabled (enable_fast_path=False),
and splices the pure visual search records into an updated output JSONL without touching main scripts.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Set

# Ensure MSSearch and ablation modules are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATION_DIR = os.path.dirname(SCRIPT_DIR)
OUR_DIR = os.path.dirname(ABLATION_DIR)
MSSEARCH_ROOT = os.path.dirname(os.path.dirname(OUR_DIR))

for p in [MSSEARCH_ROOT, OUR_DIR, ABLATION_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.our.eval_dataset import load_and_slice_dataset, resolve_image
from experiments.our.eval_metrics import compute_stage_ret_rej_tokens, format_stage6_answers, format_voting_scores
from experiments.our.ablations.eval_models_ablation import load_ablation_engines, run_ablation_pipeline
from experiments.our.ablations.eval_runner_utils import resolve_ground_truth


def parse_args():
    p = argparse.ArgumentParser(description="Standalone Local Fast-Path Re-run Tool.")
    p.add_argument("--input_jsonl", type=str, required=True, help="Path to ablation JSONL to repair.")
    p.add_argument("--output_jsonl", type=str, default=None, help="Path to output repaired JSONL.")
    p.add_argument("--component", type=str, default=None, help="Target component to execute.")
    p.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"))
    p.add_argument("--detector", type=str, default=os.environ.get("DETECTOR_TYPE", "sam3"), choices=["sam3", "grounding_dino", "florence2", "dinov3"])
    p.add_argument("--detector_path", type=str, default=os.environ.get("DETECTOR_PATH", os.environ.get("SAM3_PATH", None)))
    p.add_argument("--scorer", type=str, default=os.environ.get("SCORER_TYPE", "siglip"), choices=["siglip", "siglip2_384", "siglip2_512", "clip_b32", "clip_l14", "eva02_l14"])
    p.add_argument("--scorer_path", type=str, default=os.environ.get("SCORER_PATH", None))
    p.add_argument("--nlp_model", type=str, default=os.environ.get("NLP_MODEL", "en_core_web_sm"))
    p.add_argument("--dataset_path", type=str, default=os.environ.get("DATASET_PATH", "/data/STASHBench/annotations_part1234_merged.json"))
    p.add_argument("--image_dir", type=str, default=os.environ.get("IMAGE_DIR", None))
    p.add_argument("--pool_mode", type=str, default="N_5")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


def is_fast_path_affected(record: Dict[str, Any]) -> bool:
    """Returns True if the record was affected by early fast path triggering."""
    route = record.get("route", "")
    stage5_mode = record.get("stage5_mode", "")
    stages = record.get("stages_executed", [])
    return (
        route in ("waterfall_cascade", "tool_based_fast_path", "hybrid_direct_and_fast_path")
        or stage5_mode == "fast_path"
        or ("stage2_fast_path" in stages and "stage3_partitioning" not in stages)
    )


def map_legacy_component_to_repaired(comp: str) -> str:
    """Maps legacy fast-path component keys to their pure visual search counterparts."""
    mapping = {
        "qwen_fp_rigid_cg_single": "qwen_rigid_full",
        "qwen_fp_sgap_cg_single": "qwen_sgap_full",
        "qwen_fp_rigid_cg_agg": "qwen_rigid_full",
        "qwen_fp_sgap_cg_agg": "qwen_sgap_full",
        "qwen_fp_asp_cg_single": "qwen_sgap_full",
        "qwen_fp_asp_cg_agg": "qwen_sgap_full",
    }
    return mapping.get(comp, comp)


def main():
    args = parse_args()
    if not os.path.exists(args.input_jsonl):
        raise FileNotFoundError(f"[Re-run Tool Error] Input file not found: {args.input_jsonl}")

    out_file = args.output_jsonl
    if not out_file:
        stem, ext = os.path.splitext(args.input_jsonl)
        out_file = f"{stem}_repaired{ext}"

    # 1. Load existing records
    records: List[Dict[str, Any]] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[Re-run Tool] Loaded {len(records)} records from '{args.input_jsonl}'.")
    if not records:
        print("[Re-run Tool] No records found in file. Exiting.")
        return

    # 2. Identify fast-path affected samples
    affected_indices: List[int] = []
    affected_sample_ids: Set[str] = set()
    for idx, r in enumerate(records):
        if is_fast_path_affected(r):
            affected_indices.append(idx)
            affected_sample_ids.add(str(r.get("sample_id", r.get("sample_index"))))

    print(f"[Re-run Tool] Found {len(affected_indices)} / {len(records)} samples affected by fast path ({len(affected_indices)/len(records)*100:.1f}%).")
    if not affected_indices:
        print("[Re-run Tool] Zero samples were affected by fast path. Nothing to re-run!")
        return

    # 3. Determine target component
    inferred_comp = records[0].get("component", "qwen_sgap_full")
    target_comp = args.component or map_legacy_component_to_repaired(inferred_comp)
    inferred_pool = records[0].get("pool_name", args.pool_mode)
    effective_pool = args.pool_mode if args.pool_mode != "N_5" or "pool_name" not in records[0] else inferred_pool
    print(f"[Re-run Tool] Original component: '{inferred_comp}' -> Target component: '{target_comp}' | Pool: '{effective_pool}'.")

    # 4. Load dataset tasks for fast lookup
    print(f"[Re-run Tool] Loading dataset from '{args.dataset_path}'...")
    all_samples, eval_tasks = load_and_slice_dataset(
        args.dataset_path, effective_pool, None, None, None, args.max_samples
    )
    task_map: Dict[str, Any] = {}
    for t in eval_tasks:
        s_idx, sample, p_name, p_info = t
        s_id = str(sample.get("sample_id", f"sample_{s_idx}"))
        task_map[s_id] = t
        task_map[str(s_idx)] = t

    # 5. Load ablation engines with fast path strictly disabled
    print(f"[Re-run Tool] Loading ablation engines on {args.device}...")
    engines = load_ablation_engines(
        model_path=args.model_path,
        detector_type=args.detector,
        detector_path=args.detector_path,
        scorer_type=args.scorer,
        scorer_name=args.scorer_path,
        nlp_model=args.nlp_model,
        component=target_comp,
        device=args.device,
    )
    # Enforce pure visual search guardrails
    engines["config"].enable_fast_path = False
    engines["config"].enable_direct_answer = False
    engines["config"].enable_relational_parsing = False
    engines["config"].enable_continuous_enclosure_subcanvas = False
    engines["config"].enable_spatial_compressed_canvas = False
    engines["config"].spatial_canvas_relational_only = False

    # 6. Re-evaluate affected samples
    m_name = os.path.basename(os.path.normpath(args.model_path))
    repaired_records = list(records)
    old_correct_count = sum(1 for r in records if r.get("is_image_correct") is True)
    
    print(f"[Re-run Tool] Re-evaluating {len(affected_indices)} samples sequentially...")
    t_start_all = time.perf_counter()

    for count, rec_idx in enumerate(affected_indices):
        old_rec = records[rec_idx]
        sample_id = str(old_rec.get("sample_id", old_rec.get("sample_index")))
        s_idx_val = old_rec.get("sample_index")

        task = task_map.get(sample_id) or task_map.get(str(s_idx_val))
        if not task:
            print(f"[Warning] Task for sample ID '{sample_id}' not found in dataset! Skipping.")
            continue

        s_idx, sample, pool_name, pool_info = task
        candidate_meta = pool_info.get("images", [])
        images = [resolve_image(c, args.image_dir) for c in candidate_meta]

        t0 = time.perf_counter()
        resp = run_ablation_pipeline(engines, images, sample, s_idx)
        lat = round(time.perf_counter() - t0, 3)

        gt_list = resolve_ground_truth(sample)
        gt_img = pool_info.get("gt_image") or sample.get("gt_image")
        gt_img_id = gt_img.get("image_id") if isinstance(gt_img, dict) else (gt_img or "")
        gt_idx = pool_info.get("gt_index")
        gt_img_idx = (gt_idx - 1) if gt_idx is not None else next((i for i, m in enumerate(candidate_meta) if str(m.get("image_id")) == str(gt_img_id)), None)

        pred_text = str(resp.get("primary_answer", ""))
        telemetry = resp.get("telemetry", {})
        evidence_indices = telemetry.get("evidence_images", [])
        evidence_img_ids = [
            candidate_meta[i].get("image_id") if isinstance(candidate_meta[i], dict) else str(candidate_meta[i])
            for i in evidence_indices if 0 <= i < len(candidate_meta)
        ]
        is_img_ok = (str(gt_img_id) in [str(x) for x in evidence_img_ids]) if (evidence_img_ids and gt_img_id) else (False if gt_img_id else None)

        prim_img_idx = telemetry.get("primary_evidence_image")
        if prim_img_idx is None and evidence_indices:
            prim_img_idx = evidence_indices[0]
        prim_img_id = (
            candidate_meta[prim_img_idx].get("image_id") if isinstance(candidate_meta[prim_img_idx], dict) else str(candidate_meta[prim_img_idx])
        ) if (prim_img_idx is not None and 0 <= prim_img_idx < len(candidate_meta)) else None
        is_prim_img_ok = (str(gt_img_id) == str(prim_img_id)) if (gt_img_id and prim_img_id) else (False if gt_img_id else None)

        stage_stats = compute_stage_ret_rej_tokens(telemetry, gt_img_idx, sample.get("target_bbox"), images)

        new_record = {
            "sample_id": sample.get("sample_id", f"sample_{s_idx}"), "sample_index": s_idx,
            "pool_name": pool_name, "pool_N": len(candidate_meta), "model_name": m_name, "component": target_comp,
            "detector_type": args.detector, "scorer_type": args.scorer,
            "gate_threshold": getattr(engines["config"], "gate_threshold", 0.5),
            "question": sample.get("question") or sample.get("query", {}).get("question", ""),
            "ground_truth_list": gt_list, "gt_image_id": gt_img_id, "gt_image_idx": gt_img_idx,
            "primary_pred_image_id": prim_img_id, "primary_pred_image_idx": prim_img_idx,
            "is_primary_image_correct": is_prim_img_ok, "pred_image_ids": evidence_img_ids,
            "pred_image_indices": evidence_indices, "is_image_correct": is_img_ok, "prediction": pred_text,
            "route": telemetry.get("route", "visual_search_branch"),
            "stage5_mode": telemetry.get("stage5_mode", "crops_only"),
            "stages_executed": telemetry.get("stages_executed", []),
            "num_unified_crops": telemetry.get("num_unified_crops", 0),
            "timings_sec": telemetry.get("timings_sec", {}),
            "stage_retention": stage_stats["stage_retention"], "stage_rejection": stage_stats["stage_rejection"],
            "visual_tokens": stage_stats["visual_tokens"], "stage6_crop_sizes": stage_stats["stage6_crop_sizes"],
            "has_gt_bbox": stage_stats["has_gt_bbox"], "max_iou_stage4": stage_stats["max_iou_stage4"],
            "gt_survived_stage4": stage_stats["gt_survived_stage4"],
            "distractor_imgs_pruned": stage_stats["distractor_imgs_pruned"],
            "stage4_image_scores": stage_stats["stage4_image_scores"], "gt_rank_stage4": stage_stats["gt_rank_stage4"],
            "stage4_r1": stage_stats["stage4_r1"], "stage4_mrr": stage_stats["stage4_mrr"],
            "s6_visual_tokens": stage_stats["s6_visual_tokens"], "latency_sec": lat,
            "primary_answer_score": telemetry.get("primary_answer_score"),
            "stage6_answers": format_stage6_answers(telemetry, candidate_meta),
            "voting_scores": format_voting_scores(telemetry, candidate_meta),
        }

        repaired_records[rec_idx] = new_record
        for img in images:
            if img is not None:
                try: img.close()
                except Exception: pass
        images.clear()

        print(f"[{count+1}/{len(affected_indices)}] Repaired #{s_idx} ({sample_id}): "
              f"ImgOk: {old_rec.get('is_image_correct')} -> {is_img_ok} | Latency: {lat:5.2f}s | Route: {new_record['route']}")

    # 7. Write repaired JSONL
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for r in repaired_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    new_correct_count = sum(1 for r in repaired_records if r.get("is_image_correct") is True)
    tot = len(repaired_records)
    print("\n" + "=" * 60)
    print(f"[Re-run Complete] Successfully repaired {len(affected_indices)} fast-path samples.")
    print(f"Output File: '{out_file}'")
    print(f"Image Accuracy: {old_correct_count}/{tot} ({old_correct_count/tot*100:.2f}%) -> {new_correct_count}/{tot} ({new_correct_count/tot*100:.2f}%)")
    print(f"Total Replay Time: {time.perf_counter() - t_start_all:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
