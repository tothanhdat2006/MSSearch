"""
Dataset Downloader for STASH-Search High-Resolution Benchmarks.
Downloads and extracts V*Bench, HR-Bench (4K/8K), TreeBench, and MME-RealWorld-Lite
from ModelScope repository (llp1995/hr_data).
"""

import os
import sys
import time
import tarfile
import argparse
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional


DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mme_realworld_lite": {
        "filename": "mme-realworld-lite.tar.gz",
        "view_url": "https://www.modelscope.cn/datasets/llp1995/hr_data/file/view/master/mme-realworld-lite.tar.gz?id=182470",
        "direct_url": "https://modelscope.cn/datasets/llp1995/hr_data/resolve/master/mme-realworld-lite.tar.gz",
        "fallback_url": "https://www.modelscope.cn/api/v1/datasets/llp1995/hr_data/repo?Revision=master&FilePath=mme-realworld-lite.tar.gz",
        "extract_dir": "mme-realworld-lite",
        "description": "MME-RealWorld-Lite (8.78 GB)",
        "aliases": ["mme", "mme_realworld", "mmerealworldlite"],
    },
    "treebench": {
        "filename": "treebench.tar.gz",
        "view_url": "https://www.modelscope.cn/datasets/llp1995/hr_data/file/view/master/treebench.tar.gz?id=182470",
        "direct_url": "https://modelscope.cn/datasets/llp1995/hr_data/resolve/master/treebench.tar.gz",
        "fallback_url": "https://www.modelscope.cn/api/v1/datasets/llp1995/hr_data/repo?Revision=master&FilePath=treebench.tar.gz",
        "extract_dir": "treebench",
        "description": "TreeBench (215.5 MB)",
        "aliases": ["tree", "treenbench"],
    },
    "vstar": {
        "filename": "vstar.tar.gz",
        "view_url": "https://www.modelscope.cn/datasets/llp1995/hr_data/file/view/master/vstar.tar.gz?id=182470",
        "direct_url": "https://modelscope.cn/datasets/llp1995/hr_data/resolve/master/vstar.tar.gz",
        "fallback_url": "https://www.modelscope.cn/api/v1/datasets/llp1995/hr_data/repo?Revision=master&FilePath=vstar.tar.gz",
        "extract_dir": "vstar",
        "description": "V*Bench (280.9 MB)",
        "aliases": ["vstarbench", "v_star"],
    },
    "hrbench8k": {
        "filename": "hr-bench_8k.tar.gz",
        "view_url": "https://www.modelscope.cn/datasets/llp1995/hr_data/file/view/master/hr-bench_8k.tar.gz?id=182470",
        "direct_url": "https://modelscope.cn/datasets/llp1995/hr_data/resolve/master/hr-bench_8k.tar.gz",
        "fallback_url": "https://www.modelscope.cn/api/v1/datasets/llp1995/hr_data/repo?Revision=master&FilePath=hr-bench_8k.tar.gz",
        "extract_dir": "hr-bench_8k",
        "description": "HR-Bench 8K (843.3 MB)",
        "aliases": ["hrbench_8k", "hr8k", "hr-bench_8k"],
    },
    "hrbench4k": {
        "filename": "hr-bench_4k.tar.gz",
        "view_url": "https://www.modelscope.cn/datasets/llp1995/hr_data/file/view/master/hr-bench_4k.tar.gz?id=182470",
        "direct_url": "https://modelscope.cn/datasets/llp1995/hr_data/resolve/master/hr-bench_4k.tar.gz",
        "fallback_url": "https://www.modelscope.cn/api/v1/datasets/llp1995/hr_data/repo?Revision=master&FilePath=hr-bench_4k.tar.gz",
        "extract_dir": "hr-bench_4k",
        "description": "HR-Bench 4K (353.2 MB)",
        "aliases": ["hrbench_4k", "hr4k", "hr-bench_4k"],
    },
}


def canonicalize_dataset_key(key: str) -> Optional[str]:
    """Resolves dataset name or alias to canonical registry key."""
    k_clean = key.lower().replace("-", "_").strip()
    if k_clean in DATASET_REGISTRY:
        return k_clean
    for canon_k, info in DATASET_REGISTRY.items():
        if k_clean == canon_k:
            return canon_k
        for alias in info.get("aliases", []):
            if k_clean == alias.lower().replace("-", "_").strip():
                return canon_k
    return None


def resolve_direct_download_url(info: Dict[str, Any]) -> str:
    """Returns direct download URL with fallback."""
    return info.get("direct_url") or info.get("fallback_url") or info.get("view_url")


def format_bytes(n_bytes: int) -> str:
    """Formats bytes into human-readable string (KB, MB, GB)."""
    if n_bytes >= 1024 ** 3:
        return f"{n_bytes / (1024 ** 3):.2f} GB"
    if n_bytes >= 1024 ** 2:
        return f"{n_bytes / (1024 ** 2):.2f} MB"
    if n_bytes >= 1024:
        return f"{n_bytes / 1024:.2f} KB"
    return f"{n_bytes} B"


def download_file(
    url: str,
    dest_path: str,
    test_seconds: Optional[float] = None,
    chunk_size: int = 1024 * 512,
    resume: bool = True
) -> bool:
    """
    Downloads a remote file with streaming progress, resume support, and optional time limit.

    Args:
        url: Remote direct download URL.
        dest_path: Local target file path.
        test_seconds: Optional maximum download duration in seconds (for connectivity testing).
        chunk_size: Stream chunk buffer size (default 512KB).
        resume: Whether to resume partial downloads via Range header.

    Returns:
        True if download completed or connectivity test passed, False otherwise.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    temp_path = dest_path + ".tmp"

    existing_bytes = 0
    if resume and os.path.isfile(temp_path) and test_seconds is None:
        existing_bytes = os.path.getsize(temp_path)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) STASHSearch/1.0",
        "Accept": "*/*",
    }
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = getattr(resp, "status", 200)
            total_size_header = resp.headers.get("Content-Length")
            content_range = resp.headers.get("Content-Range")

            if content_range and "/" in content_range:
                try:
                    total_size = int(content_range.split("/")[-1])
                except (ValueError, IndexError):
                    total_size = None
            elif total_size_header:
                try:
                    total_size = existing_bytes + int(total_size_header)
                except ValueError:
                    total_size = None
            else:
                total_size = None

            mode = "ab" if (existing_bytes > 0 and status == 206) else "wb"
            if mode == "wb":
                existing_bytes = 0

            downloaded_bytes = existing_bytes
            t_start = time.perf_counter()
            t_last_log = t_start

            print(f"  [Connect] HTTP {status} | Target: '{os.path.basename(dest_path)}' "
                  f"| Total: {format_bytes(total_size) if total_size else 'Unknown'}")

            with open(temp_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded_bytes += len(chunk)

                    now = time.perf_counter()
                    elapsed = now - t_start

                    # Check test_seconds limit
                    if test_seconds is not None and elapsed >= test_seconds:
                        speed = (downloaded_bytes - existing_bytes) / max(0.001, elapsed) / (1024 * 1024)
                        print(f"  [Test Mode] Limit of {test_seconds}s reached: "
                              f"downloaded {format_bytes(downloaded_bytes)} at {speed:.2f} MB/s. "
                              f"Connectivity verified successfully!")
                        # Remove temporary test file so we don't leave partial archives on disk
                        try:
                            f.close()
                            if os.path.isfile(temp_path):
                                os.remove(temp_path)
                        except Exception:
                            pass
                        return True

                    # Periodic progress logging (every 1.5 seconds)
                    if now - t_last_log >= 1.5:
                        speed = (downloaded_bytes - existing_bytes) / max(0.001, elapsed) / (1024 * 1024)
                        if total_size and total_size > 0:
                            pct = (downloaded_bytes / total_size) * 100.0
                            eta_sec = (total_size - downloaded_bytes) / max(1.0, (speed * 1024 * 1024))
                            print(f"  Progress: {pct:5.1f}% | {format_bytes(downloaded_bytes)} / {format_bytes(total_size)} "
                                  f"| Speed: {speed:5.2f} MB/s | ETA: {eta_sec:.0f}s", end="\r", flush=True)
                        else:
                            print(f"  Progress: {format_bytes(downloaded_bytes)} downloaded "
                                  f"| Speed: {speed:5.2f} MB/s", end="\r", flush=True)
                        t_last_log = now

            print()  # newline after progress

    except urllib.error.HTTPError as e:
        print(f"  [Error] HTTPError {e.code}: {e.reason} for URL: {url}")
        return False
    except urllib.error.URLError as e:
        print(f"  [Error] URLError: {e.reason} for URL: {url}")
        return False
    except Exception as e:
        print(f"  [Error] Download exception: {e}")
        return False

    # Download completed, atomic rename
    if os.path.isfile(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)
    print(f"  [Complete] Successfully saved to '{dest_path}' ({format_bytes(os.path.getsize(dest_path))}).")
    return True


def extract_tar_gz(archive_path: str, extract_to: str) -> bool:
    """Extracts a .tar.gz archive into destination directory."""
    print(f"  [Extract] Unpacking '{os.path.basename(archive_path)}' -> '{extract_to}'...")
    os.makedirs(extract_to, exist_ok=True)
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_to)
        print(f"  [Extract] Finished unpacking into '{extract_to}'.")
        return True
    except Exception as e:
        print(f"  [Error] Tar extraction failed: {e}")
        return False


def run_dataset_download(
    dataset_keys: List[str],
    output_dir: str,
    keep_archive: bool = True,
    no_extract: bool = False,
    test_seconds: Optional[float] = None
) -> Dict[str, bool]:
    """
    Downloads and extracts specified datasets.

    Args:
        dataset_keys: List of registry keys or aliases to download.
        output_dir: Base directory to store and extract datasets.
        keep_archive: Whether to retain .tar.gz archive after unpacking.
        no_extract: If True, skips tar extraction.
        test_seconds: If set, aborts each download after test_seconds (for connectivity testing).

    Returns:
        Dictionary mapping dataset key to success boolean.
    """
    results: Dict[str, bool] = {}
    os.makedirs(output_dir, exist_ok=True)

    for key in dataset_keys:
        canon = canonicalize_dataset_key(key)
        if not canon:
            print(f"[Warning] Unknown dataset key: '{key}'. Skipping.")
            results[key] = False
            continue

        info = DATASET_REGISTRY[canon]
        print(f"\n==========================================================================")
        print(f"Dataset : {canon.upper()} ({info['description']})")
        print(f"File    : {info['filename']}")
        print(f"URL     : {info['direct_url']}")
        print(f"Dest Dir: {output_dir}")
        print(f"==========================================================================")

        dest_archive = os.path.join(output_dir, info["filename"])
        direct_url = resolve_direct_download_url(info)

        # Check if already extracted
        target_subfolder = os.path.join(output_dir, info["extract_dir"])
        if os.path.isdir(target_subfolder) and test_seconds is None:
            print(f"  [Skip] Destination directory '{target_subfolder}' already exists.")
            results[canon] = True
            continue

        # Check if archive already downloaded
        if os.path.isfile(dest_archive) and test_seconds is None:
            print(f"  [Found] Existing archive '{dest_archive}'. Proceeding to extract.")
            ok = True
        else:
            ok = download_file(
                url=direct_url,
                dest_path=dest_archive,
                test_seconds=test_seconds
            )

        if not ok:
            results[canon] = False
            continue

        # In test mode, do not unpack
        if test_seconds is not None:
            results[canon] = True
            continue

        # Extract archive
        if not no_extract and os.path.isfile(dest_archive):
            ext_ok = extract_tar_gz(dest_archive, output_dir)
            if ext_ok and not keep_archive:
                try:
                    os.remove(dest_archive)
                    print(f"  [Cleanup] Removed archive '{dest_archive}'.")
                except Exception:
                    pass
            results[canon] = ext_ok
        else:
            results[canon] = True

    return results


def main():
    parser = argparse.ArgumentParser(description="STASH-Search Dataset Downloader (ModelScope)")
    parser.add_argument(
        "--datasets", nargs="+", default=["all"],
        help="Datasets to download: 'all', 'vstar', 'hrbench4k', 'hrbench8k', 'treebench', 'mme_realworld_lite'."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=os.environ.get("DATASETS_DIR", "D:\\MachineLearning\\Datasets" if os.path.isdir("D:\\MachineLearning\\Datasets") else "./data"),
        help="Directory where datasets will be downloaded and extracted."
    )
    parser.add_argument(
        "--keep_archive", action="store_true", default=True,
        help="Keep downloaded .tar.gz archives after unpacking (default: True)."
    )
    parser.add_argument(
        "--no_extract", action="store_true",
        help="Download archives only without extracting."
    )
    parser.add_argument(
        "--test_seconds", type=float, default=None,
        help="Limit download to N seconds per dataset to test network connectivity without saving large files."
    )

    args = parser.parse_args()

    targets: List[str] = []
    for d in args.datasets:
        if d.lower() == "all":
            targets = list(DATASET_REGISTRY.keys())
            break
        elif d.lower() in ("hrbench", "hr_bench"):
            targets.extend(["hrbench4k", "hrbench8k"])
        else:
            targets.append(d)

    print(f"[Downloader] Selected datasets: {targets}")
    print(f"[Downloader] Output directory : {args.output_dir}")
    if args.test_seconds:
        print(f"[Downloader] Test mode active: capping each download at {args.test_seconds}s.")

    results = run_dataset_download(
        dataset_keys=targets,
        output_dir=args.output_dir,
        keep_archive=args.keep_archive,
        no_extract=args.no_extract,
        test_seconds=args.test_seconds
    )

    print("\n[Summary] Download Results:")
    all_ok = True
    for k, ok in results.items():
        status_str = "SUCCESS" if ok else "FAILED"
        print(f"  - {k:<20}: {status_str}")
        if not ok:
            all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
