import os
import sys
import tarfile
import argparse
import urllib.request
import shutil
from typing import Optional

MODELSCOPE_SAM3_URL = "https://www.modelscope.cn/models/llp1995/sam3/resolve/master/sam3.tar.gz"
MODELSCOPE_SPACY_URL = "https://www.modelscope.cn/models/llp1995/sam3/resolve/master/en_core_web_sm-3.8.0.tar.gz"
SIGLIP_HF_REPO = "google/siglip-base-patch16-384"

def parse_args():
    parser = argparse.ArgumentParser(description="Download auxiliary weights (SAM3, SigLIP, spaCy).")
    parser.add_argument("--base_model_dir", type=str, default="/data/shared/cvpr/dat/models",
                        help="Root directory for storing auxiliary models.")
    parser.add_argument("--sam3_url", type=str, default=MODELSCOPE_SAM3_URL,
                        help="Direct download link for sam3.tar.gz.")
    parser.add_argument("--spacy_url", type=str, default=MODELSCOPE_SPACY_URL,
                        help="Direct download link for en_core_web_sm-3.8.0.tar.gz.")
    parser.add_argument("--siglip_repo", type=str, default=SIGLIP_HF_REPO,
                        help="HuggingFace repo ID for SigLIP.")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", None),
                        help="Optional HuggingFace access token.")
    return parser.parse_args()

def download_with_progress(url: str, dest_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    print(f"[Download] Fetching: {url} -> {dest_path}")
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            sys.stdout.write(f"\r[Progress] {percent}% ({count * block_size // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    sys.stdout.write("\n")

def extract_tar(tar_path: str, extract_dir: str):
    print(f"[Extract] Extracting {tar_path} -> {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=extract_dir)

def main():
    args = parse_args()
    base_dir = os.path.abspath(args.base_model_dir)
    os.makedirs(base_dir, exist_ok=True)

    # 1. SAM3
    sam3_target_dir = os.path.join(base_dir, "sam3")
    sam3_pt = os.path.join(sam3_target_dir, "sam3.pt")
    if not os.path.exists(sam3_pt):
        tar_dest = os.path.join(base_dir, "sam3.tar.gz")
        download_with_progress(args.sam3_url, tar_dest)
        extract_tar(tar_dest, sam3_target_dir)
        if os.path.exists(tar_dest):
            os.remove(tar_dest)
        # Check if extracted into a nested directory
        nested_pt = os.path.join(sam3_target_dir, "sam3", "sam3.pt")
        if os.path.exists(nested_pt) and not os.path.exists(sam3_pt):
            shutil.copyfile(nested_pt, sam3_pt)
    print(f"[Verified] SAM3 checkpoint ready at '{sam3_pt}'.")

    # 2. spaCy
    spacy_target_dir = os.path.join(base_dir, "spacy")
    spacy_model_path = os.path.join(spacy_target_dir, "en_core_web_sm-3.8.0")
    if not os.path.exists(spacy_model_path):
        spacy_tar = os.path.join(base_dir, "en_core_web_sm-3.8.0.tar.gz")
        download_with_progress(args.spacy_url, spacy_tar)
        extract_tar(spacy_tar, spacy_target_dir)
        if os.path.exists(spacy_tar):
            os.remove(spacy_tar)
    print(f"[Verified] spaCy pipeline ready at '{spacy_model_path}'.")

    # 3. SigLIP
    siglip_target_dir = os.path.join(base_dir, "siglip-base-patch16-384")
    if not os.path.exists(siglip_target_dir) or not os.path.exists(os.path.join(siglip_target_dir, "config.json")):
        print(f"[Download] Downloading SigLIP from Hugging Face '{args.siglip_repo}'...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=args.siglip_repo,
            local_dir=siglip_target_dir,
            local_dir_use_symlinks=False,
            token=args.hf_token
        )
    print(f"[Verified] SigLIP weights ready at '{siglip_target_dir}'.")

if __name__ == "__main__":
    main()
