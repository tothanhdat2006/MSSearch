import os
import json
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

def resolve_image(image_ref: Any, image_dir: str) -> Image.Image:
    """Resolves image file inside image_dir or image_dir/pool across common image formats."""
    if isinstance(image_ref, Image.Image):
        return ImageOps.exif_transpose(image_ref)

    candidates = []
    crop_bbox = None
    if isinstance(image_ref, dict):
        crop_bbox = image_ref.get("crop_bbox")
        img_id = str(image_ref.get("image_id", ""))
        if image_ref.get("path"):
            candidates.append(os.path.join(image_dir, image_ref["path"]))
            candidates.append(os.path.join(image_dir, os.path.basename(image_ref["path"])))
            candidates.append(os.path.join(image_dir, "pool", os.path.basename(image_ref["path"])))
        if image_ref.get("name"):
            candidates.append(os.path.join(image_dir, image_ref["name"]))
            candidates.append(os.path.join(image_dir, "pool", image_ref["name"]))
    else:
        img_id = str(image_ref)

    clean_stem = os.path.splitext(os.path.basename(img_id))[0] if "." in img_id else img_id
    for ext in ["", ".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"]:
        candidates.append(os.path.join(image_dir, f"{img_id}{ext}"))
        candidates.append(os.path.join(image_dir, f"{clean_stem}{ext}"))
        candidates.append(os.path.join(image_dir, "pool", f"{img_id}{ext}"))
        candidates.append(os.path.join(image_dir, "pool", f"{clean_stem}{ext}"))

    for p in candidates:
        if p and os.path.isfile(p):
            img = Image.open(p)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            if crop_bbox and isinstance(crop_bbox, dict):
                x, y = crop_bbox.get("x", 0), crop_bbox.get("y", 0)
                w, h = crop_bbox.get("width", img.width), crop_bbox.get("height", img.height)
                left, top = max(0, int(round(x))), max(0, int(round(y)))
                right = min(img.width, max(left + 1, int(round(x + w))))
                bottom = min(img.height, max(top + 1, int(round(y + h))))
                img = img.crop((left, top, right, bottom))
            return img

    raise FileNotFoundError(f"[FATAL ERROR] Image '{image_ref}' not found in '{image_dir}'. Immediate failure enforced.")

def load_and_slice_dataset(
    dataset_path: str,
    pool_mode: str = "N_20",
    sample_idx: int = None,
    start_idx: int = None,
    end_idx: int = None,
    max_samples: int = None,
    gallery_order_path: str = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, Dict[str, Any], str, Dict[str, Any]]]]:
    """Loads STASHBench or legacy dataset and formulates sliced evaluation tasks."""
    if os.path.isdir(dataset_path):
        for name in ["annotations_part1234_merged.json", "STASHBench/annotations_part1234_merged.json", "annotations.json", "200_samples.json"]:
            cand = os.path.join(dataset_path, name)
            if os.path.isfile(cand):
                dataset_path = cand
                break

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"[FATAL ERROR] Dataset path does not exist: '{dataset_path}'")

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    expected_gallery = gallery_order_path or os.path.join(os.path.dirname(os.path.abspath(dataset_path)), "gallery_order_seed42.json")
    if not os.path.isfile(expected_gallery):
        raise FileNotFoundError(
            f"[FATAL ERROR] 'gallery_order_seed42.json' must be in the same folder as the annotations file: '{expected_gallery}'. "
            "Without it, the candidates are silently ordered by image id."
        )
    with open(expected_gallery, "r", encoding="utf-8") as gf:
        gallery_data = json.load(gf).get("samples", {})

    all_samples = raw_data.get("samples", []) if isinstance(raw_data, dict) else raw_data

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
    canonical_pool = "N_crop" if pool_mode in ["crop", "N_crop", "N=crop"] else pool_mode
    pool_keys = ["N_5", "N_10", "N_20"] if canonical_pool == "ALL" else [canonical_pool]

    for s_idx in sample_indices:
        s = all_samples[s_idx]
        gt_img = s.get("gt_image")
        sid_keys = [str(s.get("sample_id")), f"{s_idx:03d}", str(s_idx)]
        g_entry = next((gallery_data[k] for k in sid_keys if k in gallery_data), None)

        for p_name in pool_keys:
            if p_name in ["N_1", "N1"]:
                pool_info = {"images": [gt_img] if gt_img else [], "N": 1, "gt_image": gt_img, "gt_index": 1}
                eval_tasks.append((s_idx, s, "N_1", pool_info))
            elif p_name in ["crop", "N_crop", "N=crop"]:
                bbox = None
                if g_entry and "crop" in g_entry and "box_xyxy" in g_entry["crop"]:
                    b = g_entry["crop"]["box_xyxy"]
                    bbox = {"x": b[0], "y": b[1], "width": b[2] - b[0], "height": b[3] - b[1]}
                else:
                    bbox = s.get("target_bbox")
                crop_meta = dict(gt_img) if isinstance(gt_img, dict) else {"image_id": str(gt_img)}
                crop_meta["crop_bbox"] = bbox
                crop_meta["is_crop"] = True
                pool_info = {"images": [crop_meta], "N": 1, "gt_image": gt_img, "is_crop": True, "gt_index": 1}
                eval_tasks.append((s_idx, s, "N_crop", pool_info))
            elif "other_images" in s:
                k = p_name if p_name in s["other_images"] else p_name.replace("_", "")
                if k in s["other_images"]:
                    distractors = s["other_images"][k]
                    cand_pool = [gt_img] + distractors if gt_img else distractors
                    gt_idx = None
                    if g_entry and k in g_entry and "image_ids" in g_entry[k]:
                        meta_map = {str(m.get("image_id")): m for m in cand_pool if isinstance(m, dict)}
                        cand_imgs = [meta_map.get(str(iid), {"image_id": str(iid)}) for iid in g_entry[k]["image_ids"]]
                        gt_idx = g_entry[k].get("gt_index")
                    else:
                        raise RuntimeError(
                            f"[FATAL ERROR] Sample '{s.get('sample_id', s_idx)}' pool '{k}' is missing from "
                            f"{expected_gallery}. Without it, candidates are silently ordered by image id."
                        )
                    pool_info = {"images": cand_imgs, "N": len(cand_imgs), "gt_image": gt_img, "gt_index": gt_idx}
                    eval_tasks.append((s_idx, s, p_name, pool_info))
            elif "pools" in s:
                pools = s.get("pools", {})
                k = p_name if p_name in pools else p_name.replace("_", "")
                if k in pools:
                    eval_tasks.append((s_idx, s, p_name, pools[k]))

    return all_samples, eval_tasks
