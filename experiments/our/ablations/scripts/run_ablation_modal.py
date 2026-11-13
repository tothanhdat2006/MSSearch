import os, sys, glob, json, time, modal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATION_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUR_DIR = os.path.abspath(os.path.join(ABLATION_DIR, ".."))
MSSEARCH_ROOT = os.path.abspath(os.path.join(OUR_DIR, "../.."))

for p in [MSSEARCH_ROOT, OUR_DIR, ABLATION_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

app = modal.App("mssearch-ablation-eval")

req_file = next((p for p in [
    os.path.join(OUR_DIR, "requirements.txt"),
    os.path.join(MSSEARCH_ROOT, "requirements.txt"),
    "requirements.txt"
] if os.path.isfile(p)), "requirements.txt")

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6")
    .pip_install("torch", "torchvision", extra_index_url="https://download.pytorch.org/whl/cu126")
    .pip_install_from_requirements(req_file)
    .pip_install("timm", "open_clip_torch", "einops")
    .run_commands("python -m spacy download en_core_web_sm")
    .add_local_python_source("mssearch", "experiments", "sam3", "llava")
)

data_volume = modal.Volume.from_name("mssearch-data", create_if_missing=True)
models_volume = modal.Volume.from_name("mssearch-models", create_if_missing=True)
results_volume = modal.Volume.from_name("mssearch-results", create_if_missing=True)


@app.function(
    image=eval_image,
    gpu="L40S",
    volumes={"/data": data_volume, "/models": models_volume, "/results": results_volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def evaluate_ablation_modal(
    detector: str = "sam3",
    detector_path: str = None,
    scorer: str = "siglip",
    scorer_path: str = None,
    gate_threshold: float = None,
    component: str = None,
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    pool_mode: str = "N_10",
    dataset_path: str = "/data/STASHBench/annotations_part1234_merged.json",
    image_dir: str = "/data/STASHBench/pool",
    sample_idx: int = None,
    start_idx: int = None,
    end_idx: int = None,
    max_samples: int = None,
    resume: bool = False
):
    import gc, torch
    from datetime import datetime
    for p in ["/root", "/root/experiments/our", "/root/experiments/our/ablations", os.path.dirname(__file__)]:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        from experiments.our.eval_dataset import load_and_slice_dataset, resolve_image
        from experiments.our.eval_metrics import compute_stage_ret_rej_tokens, format_stage6_answers, format_voting_scores
        from experiments.our.ablations.eval_models_ablation import load_ablation_engines, run_ablation_pipeline
    except ModuleNotFoundError:
        from eval_dataset import load_and_slice_dataset, resolve_image
        from eval_metrics import compute_stage_ret_rej_tokens, format_stage6_answers, format_voting_scores
        from eval_models_ablation import load_ablation_engines, run_ablation_pipeline

    dataset_path = next((p for p in [dataset_path, "/data/annotations_part1234_merged.json", "/data/STASHBench/annotations_part1234_merged.json", "/data/200_samples.json"] if os.path.exists(p)), dataset_path)
    image_dir = next((p for p in [image_dir, "/data/pool", "/data/STASHBench/pool", "/data/images", "/data"] if os.path.exists(p)), image_dir)
    output_dir = "/results/ablation"
    os.makedirs(output_dir, exist_ok=True)

    m_clean = os.path.basename(os.path.normpath(model_name))
    if component:
        prefix = f"results_ablation_component_{component}_{m_clean}_{pool_mode}"
    else:
        gate_tag = f"_gate{gate_threshold}" if gate_threshold is not None else ""
        prefix = f"results_ablation_{detector}_{scorer}{gate_tag}_{m_clean}_{pool_mode}"
    cands = sorted(glob.glob(os.path.join(output_dir, f"{prefix}_*.jsonl")))
    results_file = cands[-1] if resume and cands else os.path.join(output_dir, f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    config_file = os.path.join(os.path.dirname(os.path.abspath(results_file)), os.path.basename(results_file).replace("results_ablation_", "config_ablation_").replace(".jsonl", ".json"))

    _, eval_tasks = load_and_slice_dataset(dataset_path, pool_mode, sample_idx, start_idx, end_idx, max_samples)
    completed_keys = set()
    if resume and os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try: completed_keys.add(f"{json.loads(line).get('sample_id')}_{json.loads(line).get('pool_name')}")
                    except Exception: pass
        print(f"[Auto-Resume] Resumed {len(completed_keys)} completed tasks from {results_file}")

    remaining_tasks = [t for t in eval_tasks if f"{t[1].get('sample_id')}_{t[2]}" not in completed_keys]
    print(f"[Modal Ablation Worker] Detector: {detector} | Scorer: {scorer} | Gate: {gate_threshold or 0.5} | Tasks: {len(remaining_tasks)}/{len(eval_tasks)}")

    local_m = os.path.join("/models", m_clean)
    resolved_model = local_m if os.path.isdir(local_m) else model_name
    if "qwen3" in model_name.lower() and not os.path.isdir(resolved_model):
        for candidate in ["/models/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]:
            if os.path.isdir(candidate) or candidate.startswith("Qwen/"):
                print(f"[Modal Ablation] Resolving paper alias '{model_name}' -> '{candidate}'")
                resolved_model = candidate
                break

    det_map = {
        "grounding_dino": ("/models/grounding-dino-base", "IDEA-Research/grounding-dino-base"),
        "florence2": ("/models/Florence-2-large", "florence-community/Florence-2-large"),
        "dinov3": ("/models/dinov3-vitl16-pretrain-lvd1689m", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
    }
    sc_map = {
        "siglip": ("/models/siglip2-so400m-patch16-512", "google/siglip2-so400m-patch16-512"),
        "siglip2_384": ("/models/siglip2-so400m-patch16-384", "google/siglip2-so400m-patch16-384"),
        "siglip2_512": ("/models/siglip2-so400m-patch16-512", "google/siglip2-so400m-patch16-512"),
        "clip_b32": ("/models/clip-vit-base-patch32", "openai/clip-vit-base-patch32"),
        "clip_l14": ("/models/clip-vit-large-patch14", "openai/clip-vit-large-patch14"),
        "eva02_l14": ("/models/eva02_large_patch14_clip_336", "timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k"),
    }

    resolved_det = detector_path
    if detector == "sam3" and not resolved_det:
        s_path = "/models/sam3/sam3.pt"
        if not os.path.isfile(s_path):
            for alt in ["/models/sam3/sam3/sam3.pt", "/models/sam3.pt", "/root/models/sam3/sam3.pt"]:
                if os.path.isfile(alt): s_path = alt; break
        resolved_det = s_path
    elif detector in det_map and not resolved_det:
        loc, hf = det_map[detector]
        resolved_det = loc if os.path.isdir(loc) else hf

    resolved_sc = scorer_path
    if scorer in sc_map and not resolved_sc:
        loc, hf = sc_map[scorer]
        resolved_sc = loc if os.path.isdir(loc) else hf

    engines = load_ablation_engines(
        model_path=resolved_model, detector_type=detector, detector_path=resolved_det,
        scorer_type=scorer, scorer_name=resolved_sc, nlp_model="en_core_web_sm",
        gate_threshold=gate_threshold, component=component, device="cuda:0"
    )
    engines["config"].save_json(config_file)
    results_volume.commit()

    with open(results_file, "a", encoding="utf-8") as out_f:
        for idx, (s_idx, sample, pool_name, pool_info) in enumerate(remaining_tasks):
            sample_id = sample.get("sample_id", f"sample_{s_idx}")
            question = sample.get("question") or sample.get("query", {}).get("question", "")
            ans = sample.get("answer")
            gt_list = ans.get("list", []) if isinstance(ans, dict) else (ans if isinstance(ans, list) else ([ans] if ans else []))

            candidate_meta = pool_info.get("images", [])
            images = [resolve_image(m, image_dir) for m in candidate_meta]

            t0 = time.perf_counter()
            resp = run_ablation_pipeline(engines, images, sample, s_idx)
            latency = round(time.perf_counter() - t0, 3)

            pred_text = resp.get("primary_answer", "")
            telemetry = resp.get("telemetry", {})
            gt_img = pool_info.get("gt_image") or sample.get("gt_image")
            gt_img_id = gt_img.get("image_id") if isinstance(gt_img, dict) else (gt_img or "")
            gt_idx = pool_info.get("gt_index")
            gt_img_idx = (gt_idx - 1) if gt_idx is not None else next((i for i, m in enumerate(candidate_meta) if str(m.get("image_id")) == str(gt_img_id)), None)

            ev_indices = telemetry.get("evidence_images", [])
            ev_img_ids = [candidate_meta[i].get("image_id") if isinstance(candidate_meta[i], dict) else str(candidate_meta[i]) for i in ev_indices if 0 <= i < len(candidate_meta)]
            is_img_ok = (str(gt_img_id) in [str(x) for x in ev_img_ids]) if (ev_img_ids and gt_img_id) else (False if gt_img_id else None)

            prim_idx = telemetry.get("primary_evidence_image") if telemetry.get("primary_evidence_image") is not None else (ev_indices[0] if ev_indices else None)
            prim_id = (candidate_meta[prim_idx].get("image_id") if isinstance(candidate_meta[prim_idx], dict) else str(candidate_meta[prim_idx])) if (prim_idx is not None and 0 <= prim_idx < len(candidate_meta)) else None
            is_prim_ok = (str(gt_img_id) == str(prim_id)) if (gt_img_id and prim_id) else (False if gt_img_id else None)
            stage_stats = compute_stage_ret_rej_tokens(telemetry, gt_img_idx, sample.get("target_bbox"), images)

            record = {
                "sample_id": sample_id, "sample_index": s_idx, "pool_name": pool_name, "pool_N": len(candidate_meta),
                "model_name": m_clean, "component": component, "detector_type": detector, "scorer_type": scorer,
                "gate_threshold": getattr(engines["config"], "gate_threshold", gate_threshold or 0.5),
                "question": question, "ground_truth_list": gt_list, "gt_image_id": gt_img_id, "gt_image_idx": gt_img_idx,
                "primary_pred_image_id": prim_id, "primary_pred_image_idx": prim_idx, "is_primary_image_correct": is_prim_ok,
                "pred_image_ids": ev_img_ids, "pred_image_indices": ev_indices, "is_image_correct": is_img_ok,
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
            results_volume.commit()

            for img in images:
                if img is not None:
                    try: img.close()
                    except Exception: pass
            images.clear()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            route_str = "+".join(telemetry.get("routes_executed", [])) or telemetry.get("route", "UNKNOWN")
            print(f"[Modal Ablation] [{idx+1}/{len(remaining_tasks)}] #{s_idx} ({sample_id}) [{pool_name}|N={len(candidate_meta)}] "
                  f"ImgOk:{is_img_ok} | Route:{route_str:<18} | Time:{latency:5.2f}s | Pred:'{pred_text[:40]}'")


@app.local_entrypoint()
def main(
    detector: str = "sam3",
    detector_path: str = None,
    scorer: str = "siglip",
    scorer_path: str = None,
    gate_threshold: float = None,
    component: str = None,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    pool_mode: str = "N_10",
    dataset_path: str = "/data/STASHBench/annotations_part1234_merged.json",
    image_dir: str = "/data/STASHBench/pool",
    sample_idx: int = None,
    start_idx: int = None,
    end_idx: int = None,
    max_samples: int = None,
    resume: bool = False,
    gpu: str = "L40S",
):
    print(f"Launching Modal Ablation (Component: {component or 'N/A'} | Detector: {detector} | Scorer: {scorer} | Model: {model} | Pool: {pool_mode} on {gpu})", flush=True)
    fn = evaluate_ablation_modal.with_options(gpu=gpu)
    fn.remote(
        detector=detector, detector_path=detector_path, scorer=scorer, scorer_path=scorer_path,
        gate_threshold=gate_threshold, component=component, model_name=model, pool_mode=pool_mode,
        dataset_path=dataset_path, image_dir=image_dir, sample_idx=sample_idx,
        start_idx=start_idx, end_idx=end_idx, max_samples=max_samples, resume=resume,
    )
