import os
import sys
import json
import argparse
import urllib.request
import zipfile
import shutil
from typing import Optional

DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "/data/shared/cvpr/dat/simple/data/STASHBench")

def parse_args():
    parser = argparse.ArgumentParser(description="Download and validate evaluation dataset.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Target directory to place annotations_part1234_merged.json and pool/")
    parser.add_argument("--dataset_url", type=str, default=None,
                        help="Remote URL or cloud mirror for dataset annotations")
    parser.add_argument("--images_url", type=str, default=None,
                        help="Remote URL or zip archive containing pool images")
    return parser.parse_args()

def download_file(url: str, dest_path: str):
    """Download a remote file with progress reporting."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    print(f"[Download] Fetching: {url} -> {dest_path}")
    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, count * block_size * 100 // total_size)
            sys.stdout.write(f"\r[Progress] {percent}% ({count * block_size // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    sys.stdout.write("\n")

def main():
    args = parse_args()
    target_data_dir = os.path.abspath(args.data_dir)
    os.makedirs(target_data_dir, exist_ok=True)
    samples_file = os.path.join(target_data_dir, "annotations_part1234_merged.json")
    images_dir = os.path.join(target_data_dir, "pool")

    # 1. Download or verify annotations file
    if not os.path.exists(samples_file):
        legacy_file = os.path.join(target_data_dir, "200_samples.json")
        if os.path.exists(legacy_file):
            samples_file = legacy_file
            images_dir = os.path.join(target_data_dir, "images") if not os.path.exists(images_dir) else images_dir
        elif args.dataset_url:
            download_file(args.dataset_url, samples_file)
        else:
            # Check for local STASHBench or workspace dataset
            local_stash = "D:/MachineLearning/Datasets/STASHBench/annotations_part1234_merged.json"
            local_fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../datasets/200_samples.json"))
            if os.path.exists(local_stash):
                print(f"[Copy] Found STASHBench at '{local_stash}'. Copying to '{samples_file}'...")
                shutil.copyfile(local_stash, samples_file)
            elif os.path.exists(local_fallback):
                print(f"[Copy] Found workspace dataset at '{local_fallback}'. Copying to '{samples_file}'...")
                shutil.copyfile(local_fallback, samples_file)
            else:
                raise FileNotFoundError(f"[ERROR] Dataset annotations missing at '{samples_file}' and no --dataset_url provided.")
    else:
        print(f"[Verified] Dataset file exists at '{samples_file}'.")

    # Validate dataset structure
    with open(samples_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples_count = len(data.get("samples", [])) if isinstance(data, dict) else len(data)
    print(f"[Validated] Loaded {samples_count} evaluation samples from '{samples_file}'.")

    # 2. Images folder validation
    os.makedirs(images_dir, exist_ok=True)
    if args.images_url:
        zip_tmp = os.path.join(target_data_dir, "images_archive.zip")
        download_file(args.images_url, zip_tmp)
        print(f"[Extract] Extracting images archive to '{images_dir}'...")
        with zipfile.ZipFile(zip_tmp, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
        os.remove(zip_tmp)

    img_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    print(f"[Status] Target image directory '{images_dir}' contains {img_count} image files.")

if __name__ == "__main__":
    main()
