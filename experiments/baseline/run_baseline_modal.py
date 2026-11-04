import os
import sys
import json
import time
import modal

app = modal.App("mssearch-baseline-eval")

_req_cands = [
    os.path.join(os.path.dirname(__file__), "requirements.txt"),
    os.path.join(os.path.dirname(__file__), "../../requirements.txt"),
    "requirements.txt",
]
req_file = next((p for p in _req_cands if os.path.isfile(p)), "requirements.txt")

# Modal Image Definition
eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu126"
    )
    .pip_install_from_requirements(req_file)
    .add_local_python_source("mssearch", "experiments")
)

data_volume = modal.Volume.from_name("mssearch-data", create_if_missing=True)
models_volume = modal.Volume.from_name("mssearch-models", create_if_missing=True)
results_volume = modal.Volume.from_name("mssearch-results", create_if_missing=True)

def select_gpu(model_name: str) -> str:
    """Assigns A10G (24GB) for <=8B and L40S (48GB) for 13B-38B variants."""
    m_lower = model_name.lower()
    if any(tag in m_lower for tag in ["13b", "32b", "34b", "38b", "72b"]):
        return "L40S"
    return "A10G"

@app.function(
    image=eval_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/models": models_volume, "/results": results_volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def evaluate_baseline_modal(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    pool_mode: str = "N_20",
    eval_strategy: str = "multi_image",
    dataset_path: str = "/data/STASHBench/annotations_part1234_merged.json",
    image_dir: str = "/data/STASHBench/pool",
    sample_idx: int = None,
    start_idx: int = None,
    end_idx: int = None,
    max_samples: int = None,
    resume: bool = False
):
    import sys, os, glob, torch
    from datetime import datetime
    for p in ["/root/experiments/baseline", os.path.abspath(os.path.join(os.path.dirname(__file__), "experiments/baseline")), os.path.dirname(__file__)]:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        from experiments.baseline.eval_dataset import load_and_slice_dataset, resolve_image
        from experiments.baseline.eval_models import load_vlm_engine, run_vlm_baseline_inference, parse_baseline_json_response
    except ModuleNotFoundError:
        from eval_dataset import load_and_slice_dataset, resolve_image
        from eval_models import load_vlm_engine, run_vlm_baseline_inference, parse_baseline_json_response

    if not os.path.exists(dataset_path):
        for alt in ["/data/annotations_part1234_merged.json", "/data/STASHBench/annotations_part1234_merged.json", "/data/200_samples.json"]:
            if os.path.exists(alt):
                dataset_path = alt
                break
    if not os.path.exists(image_dir):
        for alt in ["/data/pool", "/data/STASHBench/pool", "/data/images", "/data"]:
            if os.path.exists(alt):
                image_dir = alt
                break

    output_dir = "/results/baseline"
    os.makedirs(output_dir, exist_ok=True)

    m_clean = os.path.basename(os.path.normpath(model_name))
    cands = sorted(glob.glob(os.path.join(output_dir, f"results_{m_clean}_{pool_mode}_*.jsonl")))
    if resume and cands:
        results_file = cands[-1]
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"results_{m_clean}_{pool_mode}_{stamp}.jsonl")
    config_file = results_file.replace("results_", "config_").replace(".jsonl", ".json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name, "pool_mode": pool_mode, "eval_strategy": eval_strategy,
            "max_new_tokens": max_new_tokens, "max_samples": max_samples
        }, f, indent=2)
    results_volume.commit()
    print(f"[Config Saved] Config snapshot saved to '{config_file}'.")

    # Load & slice dataset
    all_samples, eval_tasks = load_and_slice_dataset(
        dataset_path, pool_mode, sample_idx, start_idx, end_idx, max_samples
    )

    completed_keys = set()
    if resume and os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed_keys.add(f"{rec.get('sample_id')}_{rec.get('pool_name')}")
                    except Exception:
                        pass
        print(f"[Auto-Resume] Resumed {len(completed_keys)} completed tasks from {results_file}")

    remaining_tasks = [t for t in eval_tasks if f"{t[1].get('sample_id')}_{t[2]}" not in completed_keys]
    print(f"[Modal Worker] Remaining Tasks to Execute: {len(remaining_tasks)} / {len(eval_tasks)}")

    local_model_path = os.path.join("/models", m_clean)
    resolved_model = local_model_path if os.path.isdir(local_model_path) else model_name
    engine = load_vlm_engine(resolved_model, device="cuda:0")

    with open(results_file, "a", encoding="utf-8") as out_f:
        for idx, (s_idx, sample, p_name, pool_info) in enumerate(remaining_tasks):
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
            images = [resolve_image(m, image_dir) for m in candidate_meta]
            t0 = time.perf_counter()
            pred_text, raw_list = run_vlm_baseline_inference(
                engine, images, question, eval_strategy=eval_strategy
            )
            lat = round(time.perf_counter() - t0, 3)

            raw_ids, actual_img_ids, img_indices, pred_answer = parse_baseline_json_response(pred_text, candidate_meta)
            gt_img = pool_info.get("gt_image") or sample.get("gt_image")
            gt_img_id = gt_img.get("image_id") if isinstance(gt_img, dict) else (gt_img or "")
            is_img_correct = (str(gt_img_id) in [str(x) for x in actual_img_ids]) if (actual_img_ids and gt_img_id) else None

            record = {
                "sample_id": sample_id,
                "sample_index": s_idx,
                "pool_name": p_name,
                "pool_N": len(candidate_meta),
                "question": question,
                "ground_truth_list": gt_list,
                "gt_image_id": gt_img_id,
                "pred_image_ids": actual_img_ids,
                "pred_image_indices": img_indices,
                "is_image_correct": is_img_correct,
                "pred_answer": pred_answer,
                "prediction": pred_answer,
                "raw_prediction": pred_text,
                "latency_sec": lat,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            results_volume.commit()  # Persistent atomic commit per sample

            status = "OK" if is_img_correct else "MISS"
            print(f"[Modal] [{idx+1}/{len(remaining_tasks)}] #{s_idx} ({sample_id}) Img:{actual_img_ids}({status}) Ans:'{pred_answer[:30]}' Time:{lat:.2f}s")

@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    pool_mode: str = "N_20",
    eval_strategy: str = "multi_image",
    dataset_path: str = "/data/STASHBench/annotations_part1234_merged.json",
    image_dir: str = "/data/STASHBench/pool",
    sample_idx: int = None,
    start_idx: int = None,
    end_idx: int = None,
    max_samples: int = None,
    resume: bool = False,
    detach_run: bool = False
):
    gpu_type = select_gpu(model)
    print(f"Launching Modal Baseline Evaluation for model: {model} on {gpu_type} (Pool: {pool_mode})")
    fn = evaluate_baseline_modal.with_options(gpu=gpu_type)
    kwargs = dict(
        model_name=model,
        pool_mode=pool_mode,
        eval_strategy=eval_strategy,
        dataset_path=dataset_path,
        image_dir=image_dir,
        sample_idx=sample_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        max_samples=max_samples,
        resume=resume
    )
    if detach_run:
        call = fn.spawn(**kwargs)
        print(f"[Modal] Job spawned asynchronously in background! Call ID: {call.object_id}")
        print("[Modal] Local entrypoint exiting cleanly. Safe to close terminal.")
        print("[Modal] Track progress with: modal app list")
    else:
        fn.remote(**kwargs)
