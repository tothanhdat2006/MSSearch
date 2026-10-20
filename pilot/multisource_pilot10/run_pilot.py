import os
import sys
import gc
import json
import time
import copy
import argparse
import re
from typing import List, Dict, Any, Optional
from PIL import Image
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run MSSearch on local pilot dataset")
    parser.add_argument("--dataset_path", type=str, default=os.environ.get("DATASET_PATH", "./datasets/pilot_10_samples.json"))
    parser.add_argument("--image_dir", type=str, default=os.environ.get("IMAGE_DIR", "./datasets/pilot_1000_images"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", "./pilot/multisource_pilot10/results"))
    parser.add_argument("--sam3_path", type=str, default=os.environ.get("SAM3_PATH", "/home/ubuntu/workspace/hr/models/sam3/sam3/sam3.pt"))
    parser.add_argument("--qwen_path", type=str, default=os.environ.get("QWEN_PATH", "/home/ubuntu/workspace/hr/models/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--siglip_name", type=str, default=os.environ.get("SIGLIP_NAME", "google/siglip-base-patch16-384"))
    parser.add_argument("--nlp_model", type=str, default=os.environ.get("NLP_MODEL", "/home/ubuntu/workspace/hr/models/spacy/en_core_web_sm-3.8.0"))
    parser.add_argument("--pool_mode", type=str, choices=["N_5", "ALL"], default=os.environ.get("POOL_MODE", "N_5"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_image(image_id: str, image_dir: str, rel_path: Optional[str] = None, **kwargs) -> Image.Image:
    target_filename = f"{image_id}.jpg" if not str(image_id).lower().endswith((".jpg", ".jpeg", ".png", ".webp")) else str(image_id)
    candidates = [
        os.path.join(image_dir, target_filename),
        os.path.join(image_dir, str(image_id)),
    ]
    if rel_path:
        candidates.insert(0, rel_path)
        candidates.append(os.path.join(image_dir, os.path.basename(rel_path)))
        candidates.append(os.path.join(os.path.dirname(image_dir), rel_path))

    for sp in candidates:
        if os.path.isfile(sp):
            return Image.open(sp).convert("RGB")

    raise FileNotFoundError(
        f"Image '{image_id}' not found in local candidate paths: {candidates}"
    )


resolve_or_download_image = resolve_image


def evaluate_freeform_prediction(pred_text: str, gt_list: List[str]) -> Dict[str, Any]:
    pred_clean = str(pred_text).lower().replace("\n", " ").strip()
    gt_set = set(item.lower().strip() for item in gt_list if item and str(item).strip())

    matched_gt = set()
    for gt_item in gt_set:
        pattern = r"\b" + re.escape(gt_item) + r"\b"
        if re.search(pattern, pred_clean):
            matched_gt.add(gt_item)

    recall = len(matched_gt) / len(gt_set) if gt_set else 0.0
    exact_match = (matched_gt == gt_set) and len(gt_set) > 0

    return {
        "gt_items": ", ".join(sorted(list(gt_set))),
        "matched_items": ", ".join(sorted(list(matched_gt))),
        "recall": recall,
        "exact_match": exact_match,
        "num_gt": len(gt_set),
        "num_matched": len(matched_gt),
    }


def main():
    args = parse_args()

    # Add MSSearch repo root to sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mssearch_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    if mssearch_root not in sys.path:
        sys.path.insert(0, mssearch_root)
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from mssearch.config import MSSearchConfig
    from mssearch.MSSearch import get_mssearch_response
    from mssearch.models.modeling_qwenvl import ModelQwenVL
    from mssearch.models.modeling_sam3 import sam3_inference
    from mssearch.models.modeling_siglip import SigLIPScorer

    import spacy
    try:
        nlp_parser = spacy.load(args.nlp_model)
    except Exception:
        nlp_parser = None

    os.makedirs(args.output_dir, exist_ok=True)
    predictions_jsonl = os.path.join(args.output_dir, f"mssearch_pilot10_predictions_{args.pool_mode.lower()}.jsonl")

    # Hardware devices
    num_gpus = torch.cuda.device_count()
    sam3_device = "cuda:1" if num_gpus > 1 else ("cuda:0" if num_gpus > 0 else "cpu")
    siglip_device = "cuda:0" if num_gpus > 0 else "cpu"
    qwen_device = "cuda:0" if num_gpus > 0 else "cpu"

    print("=" * 75)
    print("INITIALIZING FOUNDATION MODELS")
    print(f"GPUs detected : {num_gpus}")
    print(f"SAM3 Device   : {sam3_device}")
    print(f"SigLIP Device : {siglip_device}")
    print(f"Qwen Device   : {qwen_device}")
    print("=" * 75)

    sam3_engine = sam3_inference(model_path=args.sam3_path, device=sam3_device)
    siglip_engine = SigLIPScorer(model_name=args.siglip_name, device=siglip_device, torch_dtype=torch.float16 if "cuda" in siglip_device else torch.float32)
    qwen_engine = ModelQwenVL(model_path=args.qwen_path, device=qwen_device, torch_dtype=torch.float16 if "cuda" in qwen_device else torch.float32, patch_scale=1.2)

    master_config = MSSearchConfig(
        multi_answer_mode=True,
        scale_budgets_by_num_images=True,
        answer_dedup_similarity=0.85,
        enable_direct_answer=True,
        direct_answer_conf_threshold=0.60,
        enable_fast_path=True,
        fast_path_conf_threshold=0.70,
        enable_anchor_first_ordering=True,
        enable_image_grounded_anchors=True,
        image_grounded_anchor_resolution=1024,
        enable_query_adaptive_dual_scale=True,
        entropy_threshold=0.35,
        min_patch_entropy=0.35,
        partition_mode="hybrid",
        partition_stopping_mode="saliency_plateau",
        max_patch_size=512,
        min_plateau_size=512,
        max_plateau_size=1536,
        saliency_plateau_gain_threshold=0.04,
        saliency_target_threshold=0.35,
        max_patches_per_image=100,
        stage3_budget_mode="siglip_similarity",
        reuse_stage2_features=True,
        enable_feature_grouping=True,
        grouping_sim_threshold=0.80,
        reuse_stage3_features=True,
        sam3_group_det_threshold=0.35,
        enable_latent_gating=True,
        latent_confidence_threshold=0.0,
        latent_prompt_mode="question_aware",
        max_latent_evals=16,
        enable_existence_gating=True,
        existence_confidence_threshold=0.0,
        bypass_vlm_for_confirmed_groups=True,
        enable_prioritized_sam3_scanning=True,
        max_sam3_eval_patches=16,
        enable_group_retention_caps=True,
        enable_retention_hard_cap=True,
        max_retained_patches=16,
        retention_ranking_metric="confidence_hybrid",
        bbox_expansion_factor=1.25,
        enable_visual_prompt_bbox=True,
        max_vlm_tokens=128,
        enable_early_consensus_exit=False,
        prompt_directed_final_answer=True,
        include_options_in_prompt=False,
        answer_type="free_form",
        verbose=False
    )

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        pilot_samples = json.load(f)

    if args.max_samples:
        pilot_samples = pilot_samples[:args.max_samples]

    completed_keys = set()
    if os.path.exists(predictions_jsonl):
        with open(predictions_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    completed_keys.add(f"{rec.get('sample_id')}_{rec.get('pool_name')}")
                except Exception:
                    pass

    eval_tasks = []
    for s in pilot_samples:
        pools = s.get("pools", {})
        if args.pool_mode == "N_5":
            if "N_5" in pools:
                eval_tasks.append((s, "N_5", pools["N_5"]))
        else:
            for pname in ["N_5", "N_10", "N_20"]:
                if pname in pools:
                    eval_tasks.append((s, pname, pools[pname]))

    gt_lookup = {}
    for s in pilot_samples:
        for g_img in s.get("ground_truth", []):
            gt_lookup[g_img["image_id"]] = g_img

    print(f"[Execution Plan] Total Tasks: {len(eval_tasks)} | Resumed: {len(completed_keys)}")

    with open(predictions_jsonl, "a", encoding="utf-8") as out_f:
        for task_idx, (sample, pool_name, pool_data) in enumerate(eval_tasks):
            sample_id = sample.get("sample_id", f"sample_{task_idx}")
            task_key = f"{sample_id}_{pool_name}"
            if task_key in completed_keys:
                continue

            query_info = sample.get("query", {})
            question = query_info.get("question", "")
            answer_info = sample.get("answer", {})
            gt_list = answer_info.get("list", [])
            gt_split = answer_info.get("split_comma", "")
            gt_freeform = answer_info.get("freeform", "")

            pool_images_meta = pool_data.get("images", [])
            candidate_images = []
            candidate_meta = []

            for p_img in pool_images_meta:
                iid = p_img.get("image_id")
                role = p_img.get("role", "unknown")
                gt_meta = gt_lookup.get(iid, {})
                local_p = gt_meta.get("local_path")

                img_pil = resolve_image(image_id=iid, image_dir=args.image_dir, rel_path=local_p)
                candidate_images.append(img_pil)
                candidate_meta.append({"image_id": iid, "role": role, "size": list(img_pil.size)})

            task_config = copy.deepcopy(master_config)
            n_cand = len(candidate_images)
            print(f"\n[{task_idx + 1}/{len(eval_tasks)}] Running {task_key} (N={n_cand})...")

            t0 = time.perf_counter()
            try:
                response = get_mssearch_response(
                    sam_model=sam3_engine,
                    zoom_model=qwen_engine,
                    siglip_model=siglip_engine,
                    nlp_model=nlp_parser,
                    annotation={
                        "question": question,
                        "sample_id": sample_id,
                        "answer_type": "free_form",
                        "query": query_info
                    },
                    candidate_images=candidate_images,
                    config=task_config
                )
            except Exception as e:
                print(f"  [ERROR] {task_key} failed: {e}")
                response = {
                    "primary_answer": f"[ERROR: {str(e)}]",
                    "answers": [],
                    "telemetry": {"route": "error_exception", "timings_sec": {}}
                }

            latency = round(time.perf_counter() - t0, 3)
            pred_answer = response.get("primary_answer", "")
            telemetry = response.get("telemetry", {})
            route = telemetry.get("route", "unknown")
            routes_executed = telemetry.get("routes_executed", [route])
            stages_executed = telemetry.get("stages_executed", [])
            discovered_evidence = response.get("evidence_images", [])

            record = {
                "sample_id": sample_id,
                "id": sample_id,
                "pool_name": pool_name,
                "pool_N": pool_data.get("N", len(candidate_images)),
                "query": query_info,
                "question": question,
                "ground_truth_list": gt_list,
                "ground_truth_split": gt_split,
                "ground_truth_freeform": gt_freeform,
                "candidate_images": candidate_meta,
                "mssearch_prediction": pred_answer,
                "raw_vlm_response": response.get("raw_vlm_response", ""),
                "extracted_answers": response.get("answers", []),
                "prediction_list": response.get("prediction_list", []),
                "discovered_evidence_images": discovered_evidence,
                "evidence_images": discovered_evidence,
                "route": route,
                "routes_executed": routes_executed,
                "stages_executed": stages_executed,
                "stage6_patch_boxes": response.get("stage6_patch_boxes", []),
                "extractions": response.get("extractions", []),
                "latency_sec": latency,
                "timings_sec": telemetry.get("timings_sec", {}),
                "mssearch_output": {
                    "primary_answer": pred_answer,
                    "answers": response.get("answers", []),
                    "prediction_list": response.get("prediction_list", []),
                    "route": route,
                    "routes_executed": routes_executed,
                    "telemetry": {
                        "timings_sec": telemetry.get("timings_sec", {}),
                        "stages_executed": stages_executed,
                        "total_latency_sec": latency
                    }
                }
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            completed_keys.add(task_key)

            metrics = evaluate_freeform_prediction(pred_answer, gt_list)
            routes_str = "+".join(routes_executed) if routes_executed else route
            print(f"  -> Pred: '{pred_answer}' | GT: {metrics['gt_items']} | Recall: {metrics['recall'] * 100:.1f}% | Route: {routes_str} | Latency: {latency:.2f}s")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY")
    print("=" * 75)
    recalls = []
    exact_matches = []
    route_counts = {}
    with open(predictions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            m = evaluate_freeform_prediction(rec.get("mssearch_prediction", ""), rec.get("ground_truth_list", []))
            recalls.append(m["recall"])
            exact_matches.append(1.0 if m["exact_match"] else 0.0)
            r = rec.get("route", "unknown")
            route_counts[r] = route_counts.get(r, 0) + 1

    if recalls:
        mean_recall = sum(recalls) / len(recalls) * 100.0
        exact_acc = sum(exact_matches) / len(exact_matches) * 100.0
        print(f"Tasks Evaluated : {len(recalls)}")
        print(f"Mean Set Recall : {mean_recall:.2f}%")
        print(f"Exact Match Acc : {exact_acc:.2f}%")
        print("Route Breakdown :")
        for r_name, r_count in sorted(route_counts.items()):
            print(f"  - {r_name:25s}: {r_count} tasks ({r_count / len(recalls) * 100:.1f}%)")
        print(f"Saved to        : {predictions_jsonl}")
    print("=" * 75)


if __name__ == "__main__":
    main()
