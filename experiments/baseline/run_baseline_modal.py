import os
import sys
import json
import time
import modal

app = modal.App("mssearch-baseline-eval")

# Modal Image Definition
eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "torch>=2.2.0",
        "torchvision",
        "transformers>=4.45.0",
        "accelerate>=0.28.0",
        "pillow>=10.0.0",
        "sentencepiece",
        "huggingface_hub"
    )
    .add_local_python_source("mssearch")
)

data_volume = modal.Volume.from_name("mssearch-data", create_if_missing=True)
models_volume = modal.Volume.from_name("mssearch-models", create_if_missing=True)
results_volume = modal.Volume.from_name("mssearch-results", create_if_missing=True)

def select_gpu(model_name: str):
    """Assigns A10G (24GB) for <=8B and L40S (48GB) for 13B-38B variants."""
    m_lower = model_name.lower()
    if any(tag in m_lower for tag in ["13b", "32b", "34b", "38b", "72b"]):
        return modal.gpu.L40S()
    return modal.gpu.A10G()

@app.function(
    image=eval_image,
    gpu=modal.gpu.A10G(),  # Default dynamically overridden per invocation or fixed
    volumes={"/data": data_volume, "/models": models_volume, "/results": results_volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)]
)
def evaluate_baseline_modal(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    pool_mode: str = "N_20",
    eval_strategy: str = "multi_image",
    sample_idx: int = None,
    start_idx: int = None,
    end_idx: int = None,
    max_samples: int = None,
    resume: bool = True
):
    import torch
    from experiments.baseline.eval_dataset import load_and_slice_dataset, resolve_image
    from experiments.baseline.eval_metrics import evaluate_predictions
    from experiments.baseline.eval_models import load_vlm_engine, run_vlm_baseline_inference

    dataset_path = "/data/200_samples.json"
    image_dir = "/data/images"
    output_dir = "/results/baseline"
    os.makedirs(output_dir, exist_ok=True)

    m_clean = os.path.basename(os.path.normpath(model_name))
    results_file = os.path.join(output_dir, f"results_{m_clean}_{pool_mode}.jsonl")

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
            question = sample.get("query", {}).get("question", "")
            gt_list = sample.get("answer", {}).get("list", [])
            candidate_meta = pool_info.get("images", [])

            images = [resolve_image(m["image_id"], image_dir) for m in candidate_meta]
            t0 = time.perf_counter()
            pred_text, raw_list = run_vlm_baseline_inference(
                engine, images, question, eval_strategy=eval_strategy
            )
            lat = round(time.perf_counter() - t0, 3)
            metrics = evaluate_predictions(pred_text, raw_list, gt_list)

            record = {
                "sample_id": sample_id,
                "sample_index": s_idx,
                "pool_name": p_name,
                "pool_N": len(candidate_meta),
                "question": question,
                "ground_truth_list": gt_list,
                "prediction": pred_text,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "exact_match": metrics["exact_match"],
                "latency_sec": lat,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            results_volume.commit()  # Persistent atomic commit per sample

            print(f"[Modal] [{idx+1}/{len(remaining_tasks)}] #{s_idx} ({sample_id}) F1:{metrics['f1']*100:.1f}% Time:{lat:.2f}s")

@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    pool_mode: str = "N_20",
    eval_strategy: str = "multi_image",
    sample_idx: int = None,
    start_idx: int = None,
    end_idx: int = None,
    max_samples: int = None,
    resume: bool = True
):
    print(f"Launching Modal Baseline Evaluation for model: {model} (Pool: {pool_mode})")
    evaluate_baseline_modal.remote(
        model_name=model,
        pool_mode=pool_mode,
        eval_strategy=eval_strategy,
        sample_idx=sample_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        max_samples=max_samples,
        resume=resume
    )
