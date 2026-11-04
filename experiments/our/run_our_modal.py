import os
import sys
import json
import time
import modal

app = modal.App("mssearch-our-eval")

_req_cands = [
    os.path.join(os.path.dirname(__file__), "requirements.txt"),
    os.path.join(os.path.dirname(__file__), "../../requirements.txt"),
    "requirements.txt",
]
req_file = next((p for p in _req_cands if os.path.isfile(p)), "requirements.txt")

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu126"
    )
    .pip_install_from_requirements(req_file)
    .run_commands("python -m spacy download en_core_web_sm")
    .add_local_python_source("mssearch", "experiments", "sam3")
)

data_volume = modal.Volume.from_name("mssearch-data", create_if_missing=True)
models_volume = modal.Volume.from_name("mssearch-models", create_if_missing=True)
results_volume = modal.Volume.from_name("mssearch-results", create_if_missing=True)

def select_gpu(model_name: str) -> str:
    """Always assigns L40S (48GB) for our multi-model pipeline (VLM + SAM3 + SigLIP)."""
    return "L40S"

@app.function(
    image=eval_image,
    gpu="L40S",
    volumes={"/data": data_volume, "/models": models_volume, "/results": results_volume},
    timeout=86400,
    secrets=[modal.Secret.from_name("huggingface-secret")]
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
    import sys, os, torch
    for p in ["/root", "/root/experiments/our", os.path.dirname(__file__)]:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        from experiments.our.eval_dataset import load_and_slice_dataset, resolve_image
        from experiments.our.eval_metrics import evaluate_predictions
        from experiments.our.eval_models import load_our_engines, run_our_pipeline
    except ModuleNotFoundError:
        from eval_dataset import load_and_slice_dataset, resolve_image
        from eval_metrics import evaluate_predictions
        from eval_models import load_our_engines, run_our_pipeline

    dataset_path = next((p for p in [dataset_path, "/data/annotations_part1234_merged.json", "/data/STASHBench/annotations_part1234_merged.json", "/data/200_samples.json"] if os.path.exists(p)), dataset_path)
    image_dir = next((p for p in [image_dir, "/data/pool", "/data/STASHBench/pool", "/data/images", "/data"] if os.path.exists(p)), image_dir)

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
    if not os.path.isfile(sam3_path):
        for alt_sam in ["/models/sam3/sam3/sam3.pt", "/models/sam3.pt", "/root/models/sam3/sam3.pt"]:
            if os.path.isfile(alt_sam):
                sam3_path = alt_sam
                break
        if not os.path.isfile(sam3_path):
            print(f"[Modal Worker] sam3.pt not found at {sam3_path}, fetching from ModelScope...")
            os.makedirs("/models/sam3", exist_ok=True)
            tar_tmp = "/models/sam3.tar.gz"
            import urllib.request, tarfile, shutil

            def _hook(count, block_size, total_size):
                if total_size > 0 and count % 5000 == 0:
                    pct = min(100, count * block_size * 100 // total_size)
                    mb = count * block_size // (1024 * 1024)
                    tot_mb = total_size // (1024 * 1024)
                    print(f"[Modal Worker] Fetching sam3.tar.gz: {pct}% ({mb}MB / {tot_mb}MB)")

            urllib.request.urlretrieve("https://www.modelscope.cn/models/llp1995/sam3/resolve/master/sam3.tar.gz", tar_tmp, reporthook=_hook)
            print("[Modal Worker] Extracting sam3.tar.gz...")
            with tarfile.open(tar_tmp, "r:*") as tar:
                tar.extractall("/models/sam3")
            if os.path.exists(tar_tmp):
                os.remove(tar_tmp)
            nested = "/models/sam3/sam3/sam3.pt"
            if os.path.isfile(nested) and not os.path.isfile(sam3_path):
                shutil.copyfile(nested, sam3_path)
            models_volume.commit()

    siglip_name = "/models/siglip-base-patch16-384"
    if not os.path.isdir(siglip_name):
        siglip_name = "google/siglip-base-patch16-384"
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
    resume: bool = True,
    detach_run: bool = False
):
    gpu_type = select_gpu(model)
    print(f"Launching Modal MSSearch Evaluation for model: {model} on {gpu_type} (Pool: {pool_mode})")
    fn = evaluate_our_modal.with_options(gpu=gpu_type)
    kwargs = dict(
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
    if detach_run:
        call = fn.spawn(**kwargs)
        print(f"[Modal] Job spawned asynchronously in background! Call ID: {call.object_id}")
        print("[Modal] Local entrypoint exiting cleanly. Safe to close terminal.")
        print("[Modal] Track progress with: modal app list")
    else:
        fn.remote(**kwargs)
