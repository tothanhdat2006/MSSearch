import os
import sys
import argparse
import subprocess

BASELINE_MODELS = [
    "Qwen3-VL-8B", "LLaVA-v1.6-7B", "LLaVA-v1.6-13B", "LLaVA-v1.6-34B",
    "LLaVA-HR-X-7B", "LLaVA-HR-X-13B", "Qwen2.5-VL-32B", "InternVL3-38B",
    "Yi-VL-34B", "Qwen2.5-VL-7B", "LLaVA-OV-7B", "InternVL2.5-8B",
    "Qwen2.5-VL-3B", "Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-32B"
]

OUR_MODELS = [
    "Qwen2.5-VL-7B", "LLaVA-OV-7B", "InternVL2.5-8B", "Qwen2.5-VL-3B",
    "Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B", "Qwen3-VL-32B"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Downloader for MSSearch Benchmarks.")
    parser.add_argument("--data_dir", type=str, default="/data/shared/cvpr/dat/simple/data",
                        help="Target dataset directory.")
    parser.add_argument("--base_model_dir", type=str, default="/data/shared/cvpr/dat/models",
                        help="Target base models directory.")
    parser.add_argument("--download_data", action="store_true", default=True,
                        help="Download 200_samples.json and image archive.")
    parser.add_argument("--download_auxiliary", action="store_true", default=True,
                        help="Download SAM3, SigLIP, and spaCy.")
    parser.add_argument("--models", nargs="*", default=[],
                        help="Explicit list of model names to download.")
    parser.add_argument("--all_baseline", action="store_true", default=False,
                        help="Download all 17 baseline models.")
    parser.add_argument("--all_our", action="store_true", default=False,
                        help="Download all 8 MSSearch models.")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", None),
                        help="HuggingFace access token.")
    return parser.parse_args()

def run_cmd(cmd: list):
    print(f"\n[Executing] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Dataset
    if args.download_data:
        print("\n=== STEP 1: DOWNLOADING DATASET ===")
        run_cmd([sys.executable, os.path.join(script_dir, "download_data.py"), "--data_dir", args.data_dir])

    # 2. Auxiliary Models
    if args.download_auxiliary:
        print("\n=== STEP 2: DOWNLOADING AUXILIARY MODELS ===")
        aux_cmd = [sys.executable, os.path.join(script_dir, "download_auxiliary.py"), "--base_model_dir", args.base_model_dir]
        if args.hf_token:
            aux_cmd.extend(["--hf_token", args.hf_token])
        run_cmd(aux_cmd)

    # 3. Vision-Language Models
    targets = set(args.models)
    if args.all_baseline:
        targets.update(BASELINE_MODELS)
    if args.all_our:
        targets.update(OUR_MODELS)

    if targets:
        print(f"\n=== STEP 3: DOWNLOADING VISION-LANGUAGE MODELS ({len(targets)}) ===")
        for m in sorted(targets):
            m_cmd = [
                sys.executable, os.path.join(script_dir, "download_model.py"),
                "--model_name", m,
                "--base_model_dir", args.base_model_dir
            ]
            if args.hf_token:
                m_cmd.extend(["--hf_token", args.hf_token])
            run_cmd(m_cmd)

    print("\n[Complete] All download tasks finished.")

if __name__ == "__main__":
    main()
