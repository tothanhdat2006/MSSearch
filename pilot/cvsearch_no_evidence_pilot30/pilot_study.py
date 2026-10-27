#!/usr/bin/env python3
"""
Pilot Study (N=30): Evaluating Whether CVSearch Realizes No Evidence Exists in Images.

Protocol:
- Given an image I_i and an alien question Q_j sampled randomly from a distinct sample j != i:
  Invariant: i != j, Image(i) != Image(j), Question(i) != Question(j).
- Evaluates:
  1. Root Confidence C_root: Does the VLM bypass search via language prior (C_root > 0.6)?
  2. SAM3 Object Detection: Does SAM3 hallucinate false bounding boxes on the unrelated canvas?
  3. Fine Search Pruning: Does hierarchical fine search recognize target absence or over-segment clutter?
  4. Decision Realization: Boolean check if the search pipeline correctly realized absence.
"""

import os
import sys
import gc
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

# Ensure stdout flushing
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None


# ==============================================================================
# CVSearch Repository Discovery
# ==============================================================================
def resolve_cvsearch_repo(user_path: Optional[str] = None) -> str:
    """
    Dynamically discovers CVSearch repository root.
    Priority:
      1. Explicit user_path via CLI / ENV.
      2. Sibling directory at the same level as this folder: ../ICML26-CVSearch
      3. Reference workspace path: ../../../../freeform/ICML26-CVSearch
      4. Kaggle or current directory fallbacks.
    """
    if user_path and os.path.isdir(user_path):
        return os.path.abspath(user_path)

    env_path = os.environ.get("CVSEARCH_REPO_PATH", "")
    if env_path and os.path.isdir(env_path):
        return os.path.abspath(env_path)

    script_dir = Path(__file__).resolve().parent
    candidates = [
        # Sibling directory (same level as cvsearch_no_evidence_pilot30/):
        script_dir.parent / "ICML26-CVSearch",
        # Sibling inside MSSearch root:
        script_dir.parent.parent / "ICML26-CVSearch",
        # Reference path from user instructions:
        (script_dir / "../../../../freeform/ICML26-CVSearch").resolve(),
        (script_dir / "../../freeform/ICML26-CVSearch").resolve(),
        # Kaggle working directory:
        Path("/kaggle/working/ICML26-CVSearch"),
        Path("./ICML26-CVSearch").resolve(),
    ]

    for cand in candidates:
        if cand.is_dir():
            if (cand / "cvsearch" / "CVSearch.py").is_file() or (cand / "CVSearch.py").is_file():
                return str(cand)

    # If not found, return candidate at same level as default expected path
    return str(script_dir.parent / "ICML26-CVSearch")


def register_cvsearch_to_syspath(repo_root: str):
    """Adds repository root and cvsearch package directory to sys.path."""
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(
            f"CVSearch repository root '{repo_root}' not found. "
            "Please check path or clone repo into the same directory level."
        )
    cvsearch_pkg = os.path.join(repo_root, "cvsearch")
    for p in [repo_root, cvsearch_pkg]:
        if p not in sys.path:
            sys.path.insert(0, p)


# ==============================================================================
# Strict Derangement & Negative Control Generator
# ==============================================================================
def generate_strict_derangement(
    dataset: List[Dict[str, Any]],
    n_samples: int = 30,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Selects n_samples unique items and computes a strict derangement permutation pi:
      - i != pi(i) (strictly no self-matches)
      - image(i) != image(pi(i)) (strictly distinct images)
      - question(i) != question(pi(i)) (strictly distinct questions)
    """
    rng = random.Random(seed)
    
    # Extract unique image-question pairs
    seen_images = set()
    seen_questions = set()
    unique_pool = []
    
    for item in dataset:
        img_id = str(item.get("input_image") or item.get("image") or "").strip()
        q_text = str(item.get("question") or "").strip().lower()
        if not img_id or not q_text:
            continue
        if img_id not in seen_images and q_text not in seen_questions:
            seen_images.add(img_id)
            seen_questions.add(q_text)
            unique_pool.append(item)

    if len(unique_pool) < n_samples:
        raise ValueError(
            f"Dataset does not contain enough unique image-question pairs ({len(unique_pool)} < {n_samples})."
        )

    # Fix subset of N samples
    selected = unique_pool[:n_samples]
    n = len(selected)

    def get_img(sample):
        return str(sample.get("input_image") or sample.get("image") or "").strip()

    def get_q(sample):
        return str(sample.get("question") or "").strip().lower()

    # Find derangement
    perm = list(range(n))
    max_trials = 3000
    success = False

    for _ in range(max_trials):
        rng.shuffle(perm)
        collision = False
        for i, p in enumerate(perm):
            if i == p or get_img(selected[i]) == get_img(selected[p]) or get_q(selected[i]) == get_q(selected[p]):
                collision = True
                break
        if not collision:
            success = True
            break

    # Cyclic shift fallback
    if not success:
        for shift in range(1, n):
            candidate = [(i + shift) % n for i in range(n)]
            if all(
                i != p and get_img(selected[i]) != get_img(selected[p]) and get_q(selected[i]) != get_q(selected[p])
                for i, p in enumerate(candidate)
            ):
                perm = candidate
                success = True
                break

    if not success:
        raise RuntimeError("Failed to generate a collision-free derangement for pilot dataset.")

    # Build mismatched samples
    mismatched_dataset = []
    for i, p in enumerate(perm):
        target_sample = selected[i]  # Target image canvas
        donor_sample = selected[p]   # Donor question and answer

        mismatched_item = dict(donor_sample)
        # Point to target physical canvas
        target_img = target_sample.get("input_image") or target_sample.get("image")
        mismatched_item["input_image"] = target_img
        if "image" in mismatched_item:
            mismatched_item["image"] = target_img

        # Audit metadata
        mismatched_item["is_mismatched"] = True
        mismatched_item["pilot_sample_idx"] = i
        mismatched_item["target_image_idx"] = i
        mismatched_item["donor_question_idx"] = p
        mismatched_item["original_true_question"] = target_sample.get("question", "")
        mismatched_item["original_true_answer"] = target_sample.get("answer", "")
        mismatched_item["mismatched_donor_image_path"] = donor_sample.get("input_image") or donor_sample.get("image")

        mismatched_dataset.append(mismatched_item)

    # Invariant assertions
    assert len(mismatched_dataset) == n_samples
    assert all(m["target_image_idx"] != m["donor_question_idx"] for m in mismatched_dataset)
    assert all(m["question"] != m["original_true_question"] for m in mismatched_dataset)
    assert all(m["input_image"] != m["mismatched_donor_image_path"] for m in mismatched_dataset)

    return mismatched_dataset, perm


# ==============================================================================
# Scientific Metric: Realizing No Evidence Exists
# ==============================================================================
def evaluate_evidence_realization(
    root_ans_conf: float,
    sam_success_flags: List[Any],
    fine_search_success: Optional[bool],
    root_threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Evaluates whether CVSearch realized that no evidence exists for the alien question.
    Criteria:
      1. Root Hallucination: C_root > root_threshold -> True (model claims high confidence from prior).
      2. SAM3 Hallucination: sum(sam_success_flags) > 0 -> True (segmented absent target).
      3. Fine Search Hallucination: fine_search_success is True -> True (claimed target found in patch).
      4. Realized Absence: (not root_hallucinated) and (not sam_hallucinated) and (not fine_search_hallucinated).
    """
    root_hallucinated = float(root_ans_conf) > float(root_threshold)
    sam_hallucinated = bool(sum(1 for f in sam_success_flags if f == 1 or f is True) > 0)
    fine_search_hallucinated = bool(fine_search_success is True)

    realized_no_evidence = (not root_hallucinated) and (not sam_hallucinated) and (not fine_search_hallucinated)

    failure_mode = None
    if root_hallucinated:
        failure_mode = "root_prior_hallucination"
    elif sam_hallucinated:
        failure_mode = "sam3_false_positive"
    elif fine_search_hallucinated:
        failure_mode = "fine_search_false_positive"

    return {
        "realized_no_evidence": realized_no_evidence,
        "failure_mode": failure_mode,
        "root_hallucinated": root_hallucinated,
        "sam_hallucinated": sam_hallucinated,
        "fine_search_hallucinated": fine_search_hallucinated,
    }


def compute_summary_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes aggregate pilot study statistics."""
    n = len(records)
    if n == 0:
        return {}

    num_realized = sum(1 for r in records if r.get("realized_no_evidence"))
    num_root_hallucinated = sum(1 for r in records if r.get("root_hallucinated"))
    num_sam_hallucinated = sum(1 for r in records if r.get("sam_hallucinated"))
    num_fine_hallucinated = sum(1 for r in records if r.get("fine_search_hallucinated"))
    num_donor_correct = sum(1 for r in records if r.get("is_correct_donor"))

    avg_root_conf = sum(r.get("root_ans_conf", 0.0) for r in records) / n
    total_pops = sum(sum(r.get("num_pops", [])) for r in records)

    return {
        "n_samples": n,
        "realization_rate": round(num_realized / n, 4),
        "root_hallucination_rate": round(num_root_hallucinated / n, 4),
        "sam_hallucination_rate": round(num_sam_hallucinated / n, 4),
        "fine_search_hallucination_rate": round(num_fine_hallucinated / n, 4),
        "donor_accuracy": round(num_donor_correct / n, 4),
        "chance_level_accuracy": 0.25,
        "mean_root_answering_conf": round(avg_root_conf, 4),
        "total_tree_pops": total_pops,
        "average_tree_pops_per_sample": round(total_pops / n, 2),
    }


# ==============================================================================
# Image Path Resolution
# ==============================================================================
def resolve_image_path(base_dir: str, rel_path: str) -> str:
    """Resolves existing image file, stripping any duplicate path segments."""
    candidates = [
        os.path.join(base_dir, rel_path),
        os.path.join(base_dir, "image", os.path.basename(rel_path)),
        os.path.join(base_dir, os.path.basename(rel_path)),
        rel_path,
    ]
    # Clean possible duplicate '/image/image/'
    clean_base = base_dir.rstrip("/\\")
    if clean_base.endswith("image") and (rel_path.startswith("image/") or rel_path.startswith("image\\")):
        parent = os.path.dirname(clean_base)
        candidates.insert(0, os.path.join(parent, rel_path))

    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


# ==============================================================================
# Main Pilot Study Runner
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Pilot Study: Does CVSearch Realize No Evidence Exists? (N=30)")
    parser.add_argument("--cvsearch_repo", type=str, default=None, help="Path to ICML26-CVSearch repository root")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to annotations JSON (e.g. annotation_hr-bench_4k.json)")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing canvas images")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save pilot results")
    parser.add_argument("--n_samples", type=int, default=30, help="Number of pilot samples (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for derangement generator")
    parser.add_argument("--qwen_path", type=str, default=os.environ.get("QWEN_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"))
    parser.add_argument("--sam_path", type=str, default=os.environ.get("SAM3_PATH", None))
    parser.add_argument("--nlp_model", type=str, default=os.environ.get("NLP_MODEL", "en_core_web_sm"))
    parser.add_argument("--max_pixels", type=int, default=4194304)
    parser.add_argument("--direct_answer", action="store_true", help="Run direct answering baseline without search")
    parser.add_argument("--mock", action="store_true", help="Run mock evaluation for pipeline testing without GPU/weights")
    return parser.parse_args()


def run_pilot():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    # 1. Resolve CVSearch Repo Root
    cvsearch_root = resolve_cvsearch_repo(args.cvsearch_repo)
    print("=" * 75)
    print("CVSearch No-Evidence Pilot Study (N=30)")
    print("=" * 75)
    print(f"CVSearch Repo Root : {cvsearch_root}")
    register_cvsearch_to_syspath(cvsearch_root)

    # 2. Resolve Dataset Annotations
    dataset_candidates = [
        args.dataset_path,
        os.path.join(cvsearch_root, "cvsearch", "CVsearch", "annotation_hr-bench_4k.json"),
        os.path.join(cvsearch_root, "annotation_hr-bench_4k.json"),
        str(script_dir / "../../../datasets/pilot_10_samples.json"),
        str(script_dir / "../../../../freeform/ICML26-CVSearch/cvsearch/CVsearch/annotation_hr-bench_4k.json"),
    ]
    resolved_dataset_path = None
    for cand in dataset_candidates:
        if cand and os.path.isfile(cand):
            resolved_dataset_path = os.path.abspath(cand)
            break

    if not resolved_dataset_path:
        raise FileNotFoundError(
            f"Could not locate an annotations JSON dataset. Checked candidates: {dataset_candidates}"
        )
    print(f"Dataset Annotations : {resolved_dataset_path}")

    # 3. Resolve Image Directory
    img_candidates = [
        args.image_dir,
        os.path.dirname(resolved_dataset_path),
        os.path.join(os.path.dirname(resolved_dataset_path), "image"),
        str(script_dir / "../../../datasets/images"),
        "/kaggle/input/datasets/thnhtt/vs-dataset/hr-bench_8k/image",
        "/kaggle/input/hr-bench-8k/image",
    ]
    resolved_image_dir = None
    for cand in img_candidates:
        if cand and os.path.isdir(cand):
            resolved_image_dir = os.path.abspath(cand)
            break
    if not resolved_image_dir:
        resolved_image_dir = os.path.dirname(resolved_dataset_path)
    print(f"Image Directory     : {resolved_image_dir}")

    # 4. Resolve Output Directory
    out_dir = args.output_dir or str(script_dir / "results")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output Directory    : {out_dir}")
    print(f"Target Samples (N)  : {args.n_samples} | Seed: {args.seed}")
    print(f"Execution Mode      : {'Direct Baseline' if args.direct_answer else 'CVSearch Hierarchical'}")
    print("=" * 75)

    # 5. Load Dataset and Generate Strict Derangement
    with open(resolved_dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} total samples from dataset.")

    mismatched_samples, derange_perm = generate_strict_derangement(
        raw_data, n_samples=args.n_samples, seed=args.seed
    )
    print(f"Generated {len(mismatched_samples)} strictly mismatched image-question pairs.")

    # Save mismatched annotation audit
    audit_file = os.path.join(out_dir, "mismatched_pilot30_annotations.json")
    with open(audit_file, "w", encoding="utf-8") as f:
        json.dump(mismatched_samples, f, indent=2)
    print(f"Saved audit mismatched annotations to: {audit_file}")

    # 6. Model Initialization or Mock Simulation
    if not args.mock:
        import torch
        from cvsearch.models.modeling_qwenvl import ModelQwenVL
        from cvsearch.models.modeling_sam3 import sam3_inference
        from cvsearch.CVSearch import get_cvsearch_response, get_direct_response
        import spacy

        print("\n[INIT] Initializing Qwen2.5-VL...")
        zoom_model = ModelQwenVL(
            model_path=args.qwen_path,
            torch_dtype=torch.float16,
            patch_scale=1.2,
            max_pixels=args.max_pixels
        )

        print("[INIT] Initializing SAM3 Inference Model...")
        sam3_device = "cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0"
        sam_path = args.sam_path or os.environ.get("SAM3_PATH", "/kaggle/input/datasets/thnhtt/vs-sam3/sam3/sam3/sam3.pt")
        sam_model = sam3_inference(model_path=sam_path, device=sam3_device)

        print("[INIT] Loading SpaCy NLP Parser...")
        try:
            nlp_model = spacy.load(args.nlp_model)
        except Exception:
            nlp_model = spacy.load("en_core_web_sm")

        ic_path = os.path.join(cvsearch_root, "cvsearch", "ic_examples", "hr-bench_8k.json")
        ic_examples = json.load(open(ic_path, "r")) if os.path.exists(ic_path) else {"examples": [], "question_template": "{}", "answer_template": "{}"}
    else:
        print("\n[MOCK MODE] Running simulated pilot execution (zero GPU compute)...")
        zoom_model = None
        sam_model = None
        nlp_model = None
        ic_examples = None

    # Search parameters conforming to CVSearch specification
    search_kwargs = {
        "pop_limit": lambda max_depth: max_depth * 3,
        "threshold_descrease": [0.05, 0.1, 0.2],
        "answering_confidence_threshold_lower": 0.0,
        "answering_confidence_threshold_upper": 0.9,
        "fast_threshold": 0.6,
        "decomposed_question_template": "What is the appearance of the {}?",
    }

    # 7. Execute Pilot Loop
    results_jsonl = os.path.join(out_dir, "pilot30_results.jsonl")
    records = []

    print(f"\n[RUNNING] Executing N={len(mismatched_samples)} mismatched evaluations...")
    with open(results_jsonl, "w", encoding="utf-8") as f_out:
        for idx, sample in enumerate(mismatched_samples):
            t0 = time.time()
            img_rel = sample["input_image"]
            resolved_img = resolve_image_path(resolved_image_dir, img_rel)
            sample["input_image"] = resolved_img

            if not args.mock:
                if not args.direct_answer:
                    pred_output = get_cvsearch_response(
                        sam_model=sam_model,
                        zoom_model=zoom_model,
                        nlp_model=nlp_model,
                        annotation=sample,
                        ic_examples=ic_examples,
                        image_folder=None,
                        search_mode=True,
                        enable_parent_verification=False,
                        **search_kwargs
                    )
                else:
                    pred_output = get_direct_response(
                        zoom_model=zoom_model,
                        annotation=sample,
                        image_folder=None
                    )
            else:
                # Deterministic mock simulation for testing
                pred_output = "A"
                sample["root_ans_conf"] = 0.45 if idx % 2 == 0 else 0.75
                sample["sam"] = [False] if idx % 2 == 0 else [True]
                sample["num_pop"] = [3] if idx % 2 == 0 else []

            elapsed = time.time() - t0

            # Restore original relative path in record
            sample["input_image"] = img_rel
            sample["prediction"] = pred_output

            # Extract metrics
            root_conf = float(sample.get("root_ans_conf", 0.0))
            sam_flags = sample.get("sam", [])
            num_pops = sample.get("num_pop", [])
            fine_search_success = sample.get("fine_search_success", False)

            # Determine answer matching
            donor_gt = sample.get("answer")
            if isinstance(donor_gt, list) and len(donor_gt) > 0:
                donor_gt_val = donor_gt[0]
            else:
                donor_gt_val = str(donor_gt)
            is_correct_donor = (str(pred_output).strip() == str(donor_gt_val).strip())

            # Evaluate whether CVSearch realized absence
            realization_diag = evaluate_evidence_realization(
                root_ans_conf=root_conf,
                sam_success_flags=sam_flags,
                fine_search_success=fine_search_success,
                root_threshold=search_kwargs["fast_threshold"]
            )

            record = {
                "sample_idx": idx,
                "target_image": img_rel,
                "donor_question": sample.get("question"),
                "prediction": pred_output,
                "donor_ground_truth": donor_gt_val,
                "is_correct_donor": is_correct_donor,
                "root_ans_conf": root_conf,
                "sam_flags": sam_flags,
                "num_pops": num_pops,
                "elapsed_sec": round(elapsed, 2),
                **realization_diag
            }
            records.append(record)
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()

            status_str = "REALIZED_ABSENCE" if realization_diag["realized_no_evidence"] else f"FAILED ({realization_diag['failure_mode']})"
            print(
                f"[{idx+1:02d}/{args.n_samples}] C_root: {root_conf:+.3f} | "
                f"Pops: {sum(num_pops)} | "
                f"Result: {status_str} | "
                f"Time: {elapsed:.1f}s"
            )

            # Flush memory
            if not args.mock:
                import torch
                gc.collect()
                torch.cuda.empty_cache()

    # 8. Summary Diagnostics & Export
    summary = compute_summary_metrics(records)
    summary_file = os.path.join(out_dir, "summary_metrics.json")
    with open(summary_file, "w", encoding="utf-8") as f_sum:
        json.dump(summary, f_sum, indent=2)

    print("\n" + "=" * 75)
    print("PILOT STUDY SUMMARY DIAGNOSTICS (N=30)")
    print("=" * 75)
    print(f"Total Samples Tested          : {summary.get('n_samples')}")
    print(f"Evidence Realization Rate     : {summary.get('realization_rate') * 100:.1f}%")
    print(f"Root Prior Hallucination Rate : {summary.get('root_hallucination_rate') * 100:.1f}% (C_root > 0.6)")
    print(f"SAM3 False Detection Rate     : {summary.get('sam_hallucination_rate') * 100:.1f}%")
    print(f"Fine Search Hallucination Rate: {summary.get('fine_search_hallucination_rate') * 100:.1f}%")
    print(f"Accuracy on Donor Ground Truth: {summary.get('donor_accuracy') * 100:.1f}% (Chance: 25.0%)")
    print(f"Mean Root Answering Conf      : {summary.get('mean_root_answering_conf'):+.3f}")
    print(f"Average Tree Expansion Pops   : {summary.get('average_tree_pops_per_sample')}")
    print(f"Summary JSON Exported To      : {summary_file}")
    print(f"Detailed JSONL Traces At      : {results_jsonl}")
    print("=" * 75)


if __name__ == "__main__":
    run_pilot()
