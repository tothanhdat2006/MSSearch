"""
High-resolution benchmark baseline inference: one model, one benchmark, raw predictions to JSONL (resumable).
Supports V*Bench, HR-Bench (4K/8K), TreeBench, and MME-RealWorld-Lite.
Direct end-to-end MLLM feed-forward baseline without search/quadtree partitioning.
"""

import argparse
import json
import os
import sys
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

from PIL import Image, ImageOps

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor
except ImportError:
    torch = None
    DataLoader = None
    Dataset = object
    AutoModel = None
    AutoModelForImageTextToText = None
    AutoProcessor = None

# Ensure repository root is in python path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from minimal.datasets import load_benchmark_dataset, BenchmarkSample
from minimal.eval_runner import extract_predicted_option, load_env_defaults

Image.MAX_IMAGE_PIXELS = None

MODELS = {
    # "llava16_7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
    "llavaov_7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    "internvl3_8b": "OpenGVLab/InternVL3-8B-hf",
    # "internvl3_38b": "OpenGVLab/InternVL3-38B-hf",
    "qwen3vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3vl_32b": "Qwen/Qwen3-VL-32B-Instruct",
    # "molmo2_8b": "allenai/Molmo2-8B",
    # "eagle25_8b": "nvidia/Eagle2.5-8B",
}
# REMOTE_CODE = {"molmo2", "eagle25"}
BENCHMARKS = ["vstar", "hrbench4k", "hrbench8k", "treebench", "mme-realworld-lite"]
INTERNVL_TOTAL_TILES = 64


def family(model_key: str) -> str:
    """Extracts model family string for architecture-specific handling."""
    k = model_key.lower().replace("-", "").replace("_", "")
    if "llavaov" in k or "onevision" in k:
        return "llavaov"
    if "llava" in k:
        return "llava16"
    if "internvl" in k:
        return "internvl3"
    if "qwen" in k:
        return "qwen3vl"
    if "molmo" in k:
        return "molmo2"
    if "eagle" in k:
        return "eagle25"
    return model_key.split("_")[0]


def image_placeholder(processor: Any) -> str:
    """Returns model-specific image token placeholder."""
    return getattr(processor, "image_token", None) or getattr(processor, "image_placeholder_token", "<image>")


def build_prompt_text(sample: BenchmarkSample) -> str:
    """
    Constructs unified visual question answering prompt text.
    Uses benchmark-specific official inspection prompt when provided (TreeBench, MME-RealWorld-Lite),
    or standard multiple-choice instruction for V* and HR-Bench.
    """
    if getattr(sample, "inspection_prompt", None):
        body = sample.inspection_prompt
    elif getattr(sample, "options_str", None):
        body = (
            f"{sample.question}\n{sample.options_str.strip()}\n"
            "Select the best answer to the above multiple-choice question based on the image. "
            "Respond with only the letter of the correct option.\nThe best answer is:"
        )
    else:
        body = sample.question
    return f"<image>\n{body}"


def render_chat(model_key: str, processor: Any, text: str) -> str:
    """Applies model-specific chat template with image placeholder."""
    fam = family(model_key)
    if fam in ("llava16", "llavaov", "internvl3", "molmo2"):
        # llava-hf templates move images to the front, InternVL adds a newline after each image and Molmo2 puts
        # them before the user turn, so the placeholder goes inline to keep the prompt text exactly as specified.
        placeholder = image_placeholder(processor)
        content = [{"type": "text", "text": text.replace("<image>", placeholder)}]
    else:
        content = []
        parts = text.split("<image>")
        for i, part in enumerate(parts):
            if part:
                content.append({"type": "text", "text": part})
            if i < len(parts) - 1:
                content.append({"type": "image"})
    messages = [{"role": "user", "content": content}]
    try:
        return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        return text


def load_image(path: str) -> Image.Image:
    """Loads PIL Image with EXIF orientation correction in RGB mode."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def force_multi_image_preprocessing(image_processor: Any):
    """
    LLaVA-OV uses AnyRes only for single-image samples; otherwise each image becomes one 384px view
    (729 tokens + newline). Processor and model both decide from batch_num_images, so report at least 2.
    """
    parent_preprocess = super(type(image_processor), image_processor).preprocess

    def preprocess(images, **kwargs):
        return parent_preprocess(images, [max(len(x), 2) for x in images], **kwargs)

    image_processor.preprocess = preprocess


class BenchmarkDataset(Dataset):
    """Dataset wrapper for high-resolution benchmark baseline inference."""

    def __init__(self, samples: List[BenchmarkSample], model_key: str, processor: Any, budget: Dict[str, Any]):
        self.samples = samples
        self.model_key = model_key
        self.processor = processor
        self.budget = budget
        token = "<im_patch>" if family(model_key) == "molmo2" else getattr(processor, "image_token", None)
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(token) if token and hasattr(processor, "tokenizer") else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if torch is not None:
            torch.set_num_threads(1)
        sample = self.samples[idx]
        images = [load_image(p) for p in sample.candidate_image_paths]

        text = build_prompt_text(sample)
        chat = render_chat(self.model_key, self.processor, text)
        kwargs = {}
        fam = family(self.model_key)

        if fam == "internvl3":
            if self.budget.get("internvl_max_patches"):
                kwargs["max_patches"] = self.budget["internvl_max_patches"]
            else:
                default = getattr(getattr(self.processor, "image_processor", None), "max_patches", 12)
                kwargs["max_patches"] = max(1, min(default, INTERNVL_TOTAL_TILES // max(1, len(images))))
        if fam == "qwen3vl" and self.budget.get("qwen_image_max_pixels"):
            kwargs["min_pixels"] = getattr(getattr(self.processor, "image_processor", None), "size", {}).get("shortest_edge", 3136)
            kwargs["max_pixels"] = self.budget["qwen_image_max_pixels"]
        if fam == "molmo2" and self.budget.get("molmo_max_crops"):
            kwargs["max_crops"] = self.budget["molmo_max_crops"]
        if fam == "eagle25" and self.budget.get("eagle_max_tiles"):
            kwargs["max_dynamic_tiles"] = self.budget["eagle_max_tiles"]

        image_arg = [images] if self.model_key == "llavaov_7b" else images
        call_kwargs = {"images_kwargs": kwargs} if fam == "eagle25" else kwargs
        inputs = self.processor(text=[chat], images=image_arg, return_tensors="pt", **call_kwargs)
        settings = dict(kwargs)
        if self.model_key == "llavaov_7b" and self.budget.get("ov_multi_image_mode"):
            settings["ov_multi_image_mode"] = True

        visual_tokens = int((inputs["input_ids"] == self.image_token_id).sum()) if self.image_token_id else 0

        meta = {
            "sample_id": sample.sample_id,
            "benchmark": sample.metadata.get("benchmark", "benchmark"),
            "category": sample.category,
            "question": sample.question,
            "options_str": sample.options_str,
            "gt_answer": sample.gt_answer,
            "cycle_idx": sample.metadata.get("cycle_idx"),
            "prompt": chat,
            "processor_kwargs": settings,
            "visual_tokens": visual_tokens,
        }
        return inputs, meta


# Alias for backward compatibility with STASH reference script
StashDataset = BenchmarkDataset


def resolve_benchmark_path(benchmark: str, data_dir: Optional[Path] = None) -> str:
    """Resolves dataset directory or annotation path using argument or env_paths.sh defaults."""
    if data_dir is not None:
        return str(data_dir)

    b = benchmark.lower().strip()
    if b == "vstar":
        return os.environ.get("VSTAR_ANNOTATION") or os.environ.get("VSTAR_DIR", "/data/Vstar")
    elif b == "hrbench4k":
        return os.environ.get("HRBENCH_4K_DIR") or os.environ.get("HRBENCH_4K_ANNOTATION", "/data/HRBench4K")
    elif b == "hrbench8k":
        return os.environ.get("HRBENCH_8K_DIR") or os.environ.get("HRBENCH_8K_ANNOTATION", "/data/HRBench8K")
    elif b == "treebench":
        return os.environ.get("TREEBENCH_DIR") or os.environ.get("TREEBENCH_ANNOTATION", "/data/TreeBench")
    elif b == "mme-realworld-lite":
        return os.environ.get("MME_REALWORLD_DIR") or os.environ.get("MME_REALWORLD_ANNOTATION", "/data/mme-realworld-lite")
    raise ValueError(f"Unknown benchmark '{benchmark}'. Supported: {BENCHMARKS}")


def main():
    load_env_defaults()
    parser = argparse.ArgumentParser(description="High-Resolution Benchmark Baseline MLLM Inference")

    # Model and Benchmark Selection
    parser.add_argument("--model", type=str, required=True,
                        help="Model key from MODELS or path/HuggingFace ID.")
    parser.add_argument("--benchmark", "--setting", dest="benchmark", required=True,
                        choices=BENCHMARKS,
                        help="Benchmark to evaluate: 'vstar', 'hrbench4k', 'hrbench8k', 'treebench', 'mme-realworld-lite'.")

    # Dataset & Output Paths
    parser.add_argument("--data-dir", "--dataset_dir", dest="data_dir", type=Path, default=None,
                        help="Root directory or annotation path for the benchmark (default: resolved from env_paths.sh).")
    parser.add_argument("--out-dir", type=Path, default=Path("predictions"),
                        help="Output directory to save predictions JSONL.")

    # Execution Bounds & Hyperparameters
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to run.")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum generation tokens (default: 512).")
    parser.add_argument("--qwen-image-max-pixels", type=int, default=None,
                        help="Fixed per-image max pixels for Qwen3-VL (default: native).")
    parser.add_argument("--internvl-max-patches", type=int, default=None,
                        help="Fixed per-image tile cap for InternVL3 (default: 64 tiles split across images).")
    parser.add_argument("--molmo-max-crops", type=int, default=None,
                        help="Fixed per-image crop cap for Molmo2 (default: processor default).")
    parser.add_argument("--eagle-max-tiles", type=int, default=None,
                        help="Fixed per-image tile cap for Eagle2.5 (default: processor default).")
    parser.add_argument("--ov-multi-image-mode", action="store_true",
                        help="LLaVA-OV: use the multi-image preprocessing (one 384px view) even for a single image.")
    parser.add_argument("--cycle_idx", type=int, default=None,
                        help="HR-Bench option rotation index (0-3). If None, runs all 4 rotations.")

    # Resumption & Device Slicing
    parser.add_argument("--sample-ids-file", type=Path, default=None,
                        help="JSON list of sample_ids to run.")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader worker count.")
    parser.add_argument("--device", default="cuda",
                        help="Inference device (e.g. 'cuda', 'cuda:0', 'cpu').")
    parser.add_argument("--max-gpu-memory", type=int, default=None,
                        help="GiB of weights to keep on GPU 0; the rest is offloaded to CPU (bf16, slower).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preprocess only, no model loading or generation.")

    args = parser.parse_args()

    if torch is None or AutoProcessor is None:
        raise ImportError("PyTorch and HuggingFace Transformers are required to run baseline inference.")

    benchmark = args.benchmark.lower().strip()
    repo = MODELS.get(args.model, args.model)
    remote = family(args.model) in REMOTE_CODE
    processor = AutoProcessor.from_pretrained(repo, trust_remote_code=remote)

    # 1. Resolve dataset path and load benchmark instances
    dataset_path = resolve_benchmark_path(benchmark, args.data_dir)
    print(f"\n==========================================================================")
    print(f"BASELINE INFERENCE | Benchmark: {benchmark.upper()} | Model: {args.model}")
    print(f"Repository : {repo}")
    print(f"Dataset    : {dataset_path}")
    print(f"Out Dir    : {args.out_dir}")
    print(f"==========================================================================")

    samples = load_benchmark_dataset(
        benchmark=benchmark,
        dataset_dir_or_path=dataset_path,
        cycle_idx=args.cycle_idx
    )
    print(f"[DataLoader] Loaded {len(samples)} benchmark instances.")

    if args.ov_multi_image_mode and hasattr(processor, "image_processor"):
        force_multi_image_preprocessing(processor.image_processor)

    budget = {
        "qwen_image_max_pixels": args.qwen_image_max_pixels,
        "internvl_max_patches": args.internvl_max_patches,
        "ov_multi_image_mode": args.ov_multi_image_mode,
        "molmo_max_crops": args.molmo_max_crops,
        "eagle_max_tiles": args.eagle_max_tiles,
    }

    # 2. Setup output path and handle resumption
    model_dir_name = os.path.basename(os.path.normpath(args.model))
    out_path = args.out_dir / model_dir_name / f"{benchmark}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()

    if out_path.exists() and not args.dry_run:
        with open(out_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        ok = [r for r in rows if r.get("prediction") is not None]
        if len(ok) < len(rows):  # drop failed rows (e.g. cuda_oom) so those samples are retried
            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in ok)
        done = {str(r["sample_id"]) for r in ok}

    keep = set(json.loads(args.sample_ids_file.read_text())) if args.sample_ids_file else None
    items = [s for s in samples if str(s.sample_id) not in done and (keep is None or str(s.sample_id) in keep)]

    if args.limit is not None:
        items = items[: args.limit]

    print(f"[Queue] {args.model} {benchmark}: {len(done)} already done, {len(items)} to run -> '{out_path}'\n", flush=True)
    if not items and not args.dry_run:
        return

    # 3. Create Dataset and DataLoader
    dataset = BenchmarkDataset(items, args.model, processor, budget)
    loader = DataLoader(dataset, batch_size=None, num_workers=args.workers, prefetch_factor=2 if args.workers else None)

    if args.dry_run:
        for inputs, meta in loader:
            print(f"Sample: {meta['sample_id']} | in_tokens: {inputs['input_ids'].shape[1]} | visual: {meta['visual_tokens']} | {meta['processor_kwargs']}")
        if items:
            print(f"\n[Dry Run Sample Prompt]:\n{meta['prompt']}")
        return

    # 4. Load Model
    device_map, max_memory = args.device, None
    if args.max_gpu_memory:
        device_map, max_memory = "auto", {0: f"{args.max_gpu_memory}GiB", "cpu": "600GiB"}
        args.device = "cuda:0"

    model_cls = AutoModel if family(args.model) == "eagle25" else AutoModelForImageTextToText
    attn = "flash_attention_2" if family(args.model) == "eagle25" else "sdpa"  # Eagle2.5 requires FA2
    model = model_cls.from_pretrained(
        repo,
        dtype=torch.bfloat16,
        device_map=device_map,
        max_memory=max_memory,
        attn_implementation=attn,
        trust_remote_code=remote
    ).eval()
    on_gpu = args.device.startswith("cuda")

    # 5. Execute Inference Loop
    with open(out_path, "a", encoding="utf-8") as f:
        for inputs, meta in loader:
            inputs = inputs.to(args.device)
            n_in = inputs["input_ids"].shape[1]
            if on_gpu:
                torch.cuda.reset_peak_memory_stats(args.device)
            start = time.time()
            try:
                with torch.inference_mode():
                    out = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
            except torch.OutOfMemoryError:
                record = {"model": repo, **meta, "prediction": None, "error": "cuda_oom", "input_tokens": n_in}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{meta['sample_id']}] in={n_in} OOM", flush=True)
                del inputs
                torch.cuda.empty_cache()
                continue

            elapsed = time.time() - start
            # Eagle2.5 generates from inputs_embeds, so its output holds only the new tokens
            new_tokens = out[0] if family(args.model) == "eagle25" else out[0, n_in:]
            pred_text = processor.decode(new_tokens, skip_special_tokens=True).strip()
            pred_letter = extract_predicted_option(pred_text, meta.get("options_str"))

            m_gt = re.search(r"([A-Ka-k])", str(meta.get("gt_answer", "")))
            gt_letter = m_gt.group(1).upper() if m_gt else str(meta.get("gt_answer", "")).strip().upper()
            is_correct = (pred_letter == gt_letter) if pred_letter else False

            record = {
                "model": repo,
                **meta,
                "prediction": pred_text,
                "predicted_letter": pred_letter,
                "gt_letter": gt_letter,
                "is_correct": is_correct,
                "input_tokens": n_in,
                "output_tokens": int(new_tokens.shape[0]),
                "gen_seconds": round(elapsed, 2),
            }
            if on_gpu:
                record["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(args.device) / 2**30, 1)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            status_tag = "PASS" if is_correct else "FAIL"
            print(f"[{meta['sample_id']}] [{status_tag}] Pred: '{pred_letter}' ('{pred_text}') (GT: '{gt_letter}') | "
                  f"in={n_in} out={record['output_tokens']} {elapsed:.1f}s", flush=True)

            del inputs, out
            torch.cuda.empty_cache()

    # 6. Final Summary Metrics
    if not args.dry_run and os.path.isfile(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        total = len(records)
        correct = sum(1 for r in records if r.get("is_correct"))
        acc = round(correct / max(1, total) * 100, 2)

        # Rotation accuracy breakdown (for HR-Bench)
        rot_stats: Dict[str, Dict[str, int]] = {}
        for r in records:
            rot = r.get("cycle_idx")
            if rot is not None:
                rot_k = f"rot{rot}"
                if rot_k not in rot_stats:
                    rot_stats[rot_k] = {"total": 0, "correct": 0}
                rot_stats[rot_k]["total"] += 1
                if r.get("is_correct"):
                    rot_stats[rot_k]["correct"] += 1

        avg_rot_acc = None
        if rot_stats:
            rot_accs = [
                d["correct"] / max(1, d["total"]) * 100
                for d in rot_stats.values()
            ]
            avg_rot_acc = round(sum(rot_accs) / len(rot_accs), 2)

        metrics_file = out_path.with_suffix(".metrics.json")
        summary = {
            "model": repo,
            "benchmark": benchmark,
            "total_samples": total,
            "correct": correct,
            "accuracy": acc,
            "accuracy_rotation": avg_rot_acc,
            "by_rotation": rot_stats,
        }
        with open(metrics_file, "w", encoding="utf-8") as mf:
            json.dump(summary, mf, indent=2)

        print("\n" + "=" * 70)
        print(f"BASELINE INFERENCE SUMMARY: {benchmark.upper()} | MODEL: {args.model}")
        print("=" * 70)
        print(f"Total Evaluated Samples : {total}")
        print(f"Option Accuracy         : {acc:.2f}% ({correct}/{total})")
        if avg_rot_acc is not None:
            print(f"Accuracy (Rotation)     : {avg_rot_acc:.2f}%")
        print("=" * 70)
        print(f"[Done] Metrics saved to '{metrics_file}'.")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
