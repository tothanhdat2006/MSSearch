import os
import json
from typing import List, Dict, Any, Tuple
from PIL import Image

def resolve_image(image_id: str, image_dir: str) -> Image.Image:
    """Strict server image resolver: expects <image_id>.jpg inside image_dir. Fails immediately if not found."""
    target_filename = f"{image_id}.jpg" if not str(image_id).lower().endswith((".jpg", ".jpeg", ".png", ".webp")) else str(image_id)
    target_path = os.path.join(image_dir, target_filename)
    if not os.path.isfile(target_path):
        raise FileNotFoundError(f"[FATAL ERROR] Image '{image_id}' not found at required path: '{target_path}'. Immediate failure enforced.")
    return Image.open(target_path).convert("RGB")

def load_and_slice_dataset(
    dataset_path: str,
    pool_mode: str = "N_20",
    sample_idx: int = None,
    start_idx: int = None,
    end_idx: int = None,
    max_samples: int = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, Dict[str, Any], str, Dict[str, Any]]]]:
    """Loads 200_samples.json and formulates sliced evaluation tasks."""
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"[FATAL ERROR] Dataset path does not exist: '{dataset_path}'")

    with open(dataset_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    sample_indices = list(range(len(all_samples)))
    if sample_idx is not None:
        sample_indices = [sample_idx]
    else:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else len(all_samples)
        sample_indices = sample_indices[start:end]
        if max_samples is not None:
            sample_indices = sample_indices[:max_samples]

    eval_tasks: List[Tuple[int, Dict[str, Any], str, Dict[str, Any]]] = []
    pool_keys = ["N_5", "N_10", "N_20"] if pool_mode == "ALL" else [pool_mode]

    for s_idx in sample_indices:
        s = all_samples[s_idx]
        pools = s.get("pools", {})
        for p_name in pool_keys:
            if p_name in pools:
                eval_tasks.append((s_idx, s, p_name, pools[p_name]))

    return all_samples, eval_tasks
