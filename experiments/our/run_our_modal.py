import os
import sys
import json
import time
import modal

app = modal.App("mssearch-our-eval")

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
        "spacy>=3.7.0",
        "scikit-learn",
        "huggingface_hub"
    )
    .run_commands("python -m spacy download en_core_web_sm")
    .add_local_python_source("mssearch", "experiments")
)

data_volume = modal.Volume.from_name("mssearch-data", create_if_missing=True)
models_volume = modal.Volume.from_name("mssearch-models", create_if_missing=True)
results_volume = modal.Volume.from_name("mssearch-results", create_if_missing=True)

def select_gpu(model_name: str) -> str:
    """Assigns A10G (24GB) for <=8B and L40S (48GB) for 32B variants."""
    m_lower = model_name.lower()
    if any(tag in m_lower for tag in ["32b", "34b", "38b", "72b"]):
        return "L40S"
    return "A10G"

@app.function(
    image=eval_image,
    gpu="A10G",
    volumes={"/data": data_volume, "/models": models_volume, "/results": results_volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)]
)
def evaluate_our_modal(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    pool_mode: str = "N_20",
    dataset_path: str = "/data/STASHBench/annotations_part1234_merged.json",
    image_dir: str = "/data/STASHBench/pool",
    sample_idx: int = None,
    start_idx: int = None,
    end_idx: int = None,
    max_samples: int = None,
    resume: bool = True
):
    import torch
    try:
        from experiments.our.eval_dataset import load_and_slice_dataset, resolve_image
        from experiments.our.eval_metrics import evaluate_predictions
        from experiments.our.eval_models import load_our_engines, run_our_pipeline
    except ModuleNotFoundError:
        from eval_dataset import load_and_slice_dataset, resolve_image
        from eval_metrics import evaluate_predictions
        from eval_models import load_our_engines, run_our_pipeline

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

    output_dir = "/results/our"
    os.makedirs(output_dir, exist_ok=True)

    m_clean = os.path.basename(os.path.normpath(model_name))
    results_file = os.path.join(output_dir, f"results_our_{m_clean}_{pool_mode}.jsonl")

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

    sam3_path = "/models/sam3/sam3.pt"
    siglip_name = "/models/siglip-base-patch16-384"
    nlp_model = "en_core_web_sm"

    engines = load_our_engines(
        model_path=resolved_model,
        sam3_path=sam3_path,
        siglip_name=siglip_name,
        nlp_model=nlp_model,
        device="cuda:0"
    )

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
            resp = run_our_pipeline(engines, images, sample, s_idx)
            lat = round(time.perf_counter() - t0, 3)

            pred_text = resp.get("primary_answer", "")
            pred_list = resp.get("prediction_list", []) or resp.get("answers", [])
            metrics = evaluate_predictions(pred_text, pred_list, gt_list)
            telemetry = resp.get("telemetry", {})

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
                "route": telemetry.get("route", "UNKNOWN"),
                "latency_sec": lat,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            results_volume.commit()  # Persistent atomic commit per sample

            print(f"[Modal MSSearch] [{idx+1}/{len(remaining_tasks)}] #{s_idx} ({sample_id}) F1:{metrics['f1']*100:.1f}% Time:{lat:.2f}s")

@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    pool_mode: str = "N_20",
    dataset_path: str = "/data/STASHBench/annotations_part1234_merged.json",
    image_dir: str = "/data/STASHBench/pool",
    sample_idx: int = None,
    start_idx: int = None,
    end_idx: int = None,
    max_samples: int = None,
    resume: bool = True
):
    gpu_type = select_gpu(model)
    print(f"Launching Modal MSSearch Evaluation for model: {model} on {gpu_type} (Pool: {pool_mode})")
    evaluate_our_modal.with_options(gpu=gpu_type).remote(
        model_name=model,
        pool_mode=pool_mode,
        dataset_path=dataset_path,
        image_dir=image_dir,
        sample_idx=sample_idx,
        start_idx=start_idx,
        end_idx=end_idx,
        max_samples=max_samples,
        resume=resume
    )
