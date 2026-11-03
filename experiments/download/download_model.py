import os
import sys
import argparse
from typing import Dict

MODEL_MAP: Dict[str, str] = {
    # Qwen3-VL
    "Qwen3-VL-2B": "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B": "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3-VL-32B": "Qwen/Qwen3-VL-32B-Instruct",
    # Qwen2.5-VL
    "Qwen2.5-VL-3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B": "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen2.5-VL-72B": "Qwen/Qwen2.5-VL-72B-Instruct",
    # LLaVA-v1.6 (LLaVA-NeXT)
    "LLaVA-v1.6-7B": "liuhaotian/llava-v1.6-vicuna-7b",
    "LLaVA-v1.6-13B": "liuhaotian/llava-v1.6-vicuna-13b",
    "LLaVA-v1.6-34B": "liuhaotian/llava-v1.6-34b",
    # LLaVA-HR-X
    "LLaVA-HR-X-7B": "favor123/llava-hr-7b-sft-1024",
    "LLaVA-HR-X-13B": "favor123/llava-hr-13b-x-sft-1024",
    # LLaVA-OV
    "LLaVA-OV-7B": "lmms-lab/llava-onevision-qwen2-7b-ov",
    "llava-onevision-qwen2-7b-ov": "lmms-lab/llava-onevision-qwen2-7b-ov",
    # InternVL
    "InternVL2.5-8B": "OpenGVLab/InternVL2.5-8B",
    "InternVL3-38B": "OpenGVLab/InternVL3-38B",
    # Yi-VL
    "Yi-VL-34B": "01-ai/Yi-VL-34B",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Download model weights from Hugging Face.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Exact model name (e.g. Qwen2.5-VL-7B, Yi-VL-34B, LLaVA-HR-X-7B) or HF repo ID.")
    parser.add_argument("--base_model_dir", type=str, default="/data/shared/cvpr/dat/models",
                        help="Root directory where model folder will be saved.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Explicit destination directory. Defaults to <base_model_dir>/<model_name>.")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", None),
                        help="HuggingFace access token.")
    return parser.parse_args()

def main():
    args = parse_args()
    from huggingface_hub import snapshot_download

    model_key = args.model_name.strip()
    hf_repo = MODEL_MAP.get(model_key, model_key)
    dest_dir = os.path.abspath(args.output_dir or os.path.join(args.base_model_dir, model_key))

    print("==========================================================================")
    print(f"DOWNLOADING MODEL: {model_key}")
    print(f"HuggingFace Repo : {hf_repo}")
    print(f"Target Directory : {dest_dir}")
    print("==========================================================================")

    os.makedirs(dest_dir, exist_ok=True)
    snapshot_download(
        repo_id=hf_repo,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
        token=args.hf_token,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )
    print(f"[Success] Model '{model_key}' downloaded successfully to '{dest_dir}'.")

if __name__ == "__main__":
    main()
