#!/usr/bin/env python3
"""
Stage 1 Question Parser Prompt Comparison Tool.
Evaluates and compares MLLM question parsing under:
  (1) Old Relational Prompt (enable_relational_parsing=True)
  (2) New Stripped Prompt (enable_relational_parsing=False)
Measures Set Exact Match, Strict Ordered Match, Target Entity Match, and Latency savings.
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUR_DIR = os.path.dirname(SCRIPT_DIR)
MSSEARCH_ROOT = os.path.dirname(os.path.dirname(OUR_DIR))

for p in [MSSEARCH_ROOT, OUR_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from mssearch.config import MSSearchConfig
from mssearch.stages.stage1_parser import parse_question_to_active_objects


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1 Prompt Comparison Tool.")
    p.add_argument("--question", type=str, default=None, help="Interactive single question mode.")
    p.add_argument("--dataset_path", type=str, default=os.environ.get("DATASET_PATH", "/data/STASHBench/annotations_part1234_merged.json"), help="Dataset JSON path.")
    p.add_argument("--all", action="store_true", help="Evaluate across all samples in the dataset without limits.")
    p.add_argument("--max_samples", type=int, default=None, help="Cap number of samples (ignored if --all is set).")
    p.add_argument("--start_idx", type=int, default=None, help="Dataset start sample index.")
    p.add_argument("--end_idx", type=int, default=None, help="Dataset end sample index.")
    p.add_argument("--sample_idx", type=int, default=None, help="Single sample index to test.")
    p.add_argument("--model_path", type=str, default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"))
    p.add_argument("--nlp_model", type=str, default=os.environ.get("NLP_MODEL", "en_core_web_sm"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_path", type=str, default="results/stage1_prompt_comparison.json")
    p.add_argument("--verbose", action="store_true", help="Print detailed debug output per sample.")
    return p.parse_args()


def load_vlm_and_nlp(model_path: str, nlp_model_name: str, device: str):
    """Initializes VLM backbone and spaCy pipeline."""
    import spacy
    m_lower = model_path.lower()
    if "internvl" in m_lower:
        from mssearch.models import ModelInternvl
        vlm = ModelInternvl(model_path=model_path, device=device)
    else:
        from mssearch.models import ModelQwenVL
        vlm = ModelQwenVL(model_path=model_path, device=device)

    try:
        nlp = spacy.load(nlp_model_name)
    except Exception:
        nlp = spacy.blank("en")
    return vlm, nlp


def load_questions_from_dataset(dataset_path: str, all_flag: bool, max_samples: Optional[int],
                                start_idx: Optional[int], end_idx: Optional[int], sample_idx: Optional[int]) -> List[Dict[str, Any]]:
    """Loads target questions from dataset file applying slicing flags."""
    candidates = [dataset_path, "/data/annotations_part1234_merged.json", "data/200_samples.json"]
    resolved = next((c for c in candidates if os.path.exists(c)), dataset_path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"[Stage 1 Tool] Dataset file not found: '{dataset_path}'")

    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data if isinstance(data, list) else data.get("samples", data.get("data", []))

    if sample_idx is not None and 0 <= sample_idx < len(samples):
        samples = [samples[sample_idx]]
    else:
        if start_idx is not None or end_idx is not None:
            samples = samples[start_idx or 0 : end_idx or len(samples)]
        if not all_flag and max_samples is not None:
            samples = samples[:max_samples]

    items = []
    for idx, s in enumerate(samples):
        q = s.get("question") or s.get("query", {}).get("question", "")
        if q:
            items.append({"sample_index": s.get("sample_index", idx), "sample_id": s.get("sample_id", f"sample_{idx}"), "question": q})
    return items


def run_comparison(vlm, nlp, questions: List[Dict[str, Any]], verbose: bool = False) -> Dict[str, Any]:
    """Runs dual-prompt evaluation and computes exact matching metrics."""
    cfg_old = MSSearchConfig(enable_relational_parsing=True, verbose=verbose)
    cfg_new = MSSearchConfig(enable_relational_parsing=False, verbose=verbose)

    results = []
    mismatches = []
    t_start = time.perf_counter()

    for i, item in enumerate(questions):
        q = item["question"]
        s_idx = item.get("sample_index", i)
        s_id = item.get("sample_id", f"sample_{i}")

        # 1. Old Prompt (Relational ON)
        t0 = time.perf_counter()
        old_res = parse_question_to_active_objects(q, zoom_model=vlm, nlp_model=nlp, config=cfg_old, verbose=verbose)
        lat_old = round(time.perf_counter() - t0, 3)

        # 2. New Prompt (Relational OFF)
        t1 = time.perf_counter()
        new_res = parse_question_to_active_objects(q, zoom_model=vlm, nlp_model=nlp, config=cfg_new, verbose=verbose)
        lat_new = round(time.perf_counter() - t1, 3)

        old_objs = list(old_res.active_objects)
        new_objs = list(new_res.active_objects)
        set_match = set(old_objs) == set(new_objs)
        strict_match = old_objs == new_objs

        old_tgt = str(old_res.target_entity or (old_objs[0] if old_objs else "")).lower().strip()
        new_tgt = str(new_res.target_entity or (new_objs[0] if new_objs else "")).lower().strip()
        tgt_match = old_tgt == new_tgt

        record = {
            "sample_index": s_idx, "sample_id": s_id, "question": q,
            "old_active_objects": old_objs, "new_active_objects": new_objs,
            "old_target_entity": old_res.target_entity, "new_target_entity": new_res.target_entity,
            "old_is_relational": getattr(old_res, "is_relational", False),
            "new_is_relational": getattr(new_res, "is_relational", False),
            "set_match": set_match, "strict_match": strict_match, "target_match": tgt_match,
            "old_latency_sec": lat_old, "new_latency_sec": lat_new,
            "old_raw_response": getattr(old_res, "raw_response", ""),
            "new_raw_response": getattr(new_res, "raw_response", ""),
        }
        results.append(record)

        if not set_match or not tgt_match:
            mismatches.append(record)

        match_symbol = "✓" if set_match else "✗"
        print(f"[{i+1}/{len(questions)}] #{s_idx} ({s_id}) [{match_symbol}] "
              f"SetMatch:{set_match} | Strict:{strict_match} | TgtMatch:{tgt_match} | "
              f"Old:{old_objs} vs New:{new_objs} | Lat:{lat_old:.2f}s->{lat_new:.2f}s")

    tot = len(results)
    set_matches = sum(1 for r in results if r["set_match"])
    strict_matches = sum(1 for r in results if r["strict_match"])
    tgt_matches = sum(1 for r in results if r["target_match"])
    avg_old_lat = sum(r["old_latency_sec"] for r in results) / max(1, tot)
    avg_new_lat = sum(r["new_latency_sec"] for r in results) / max(1, tot)
    lat_speedup = ((avg_old_lat - avg_new_lat) / max(0.001, avg_old_lat)) * 100

    summary = {
        "total_questions": tot,
        "set_exact_match_count": set_matches,
        "set_exact_match_pct": round(set_matches / max(1, tot) * 100, 2),
        "strict_ordered_match_count": strict_matches,
        "strict_ordered_match_pct": round(strict_matches / max(1, tot) * 100, 2),
        "target_entity_match_count": tgt_matches,
        "target_entity_match_pct": round(tgt_matches / max(1, tot) * 100, 2),
        "avg_latency_old_sec": round(avg_old_lat, 3),
        "avg_latency_new_sec": round(avg_new_lat, 3),
        "latency_reduction_pct": round(lat_speedup, 2),
        "total_mismatches": len(mismatches),
        "total_elapsed_sec": round(time.perf_counter() - t_start, 2),
    }

    print("\n" + "=" * 65)
    print("           STAGE 1 PROMPT COMPARISON SCORECARD")
    print("=" * 65)
    print(f"  • Total Questions Evaluated   : {tot}")
    print(f"  • Active Objects Set Match    : {set_matches}/{tot} ({summary['set_exact_match_pct']}%)")
    print(f"  • Active Objects Strict Match : {strict_matches}/{tot} ({summary['strict_ordered_match_pct']}%)")
    print(f"  • Target Entity Exact Match   : {tgt_matches}/{tot} ({summary['target_entity_match_pct']}%)")
    print(f"  • Avg Latency Old vs. New     : {avg_old_lat:.3f}s -> {avg_new_lat:.3f}s ({lat_speedup:+.1f}%)")
    print(f"  • Divergent Samples Count     : {len(mismatches)}")
    print("=" * 65 + "\n")

    return {"summary": summary, "mismatches": mismatches, "all_records": results}


def main():
    args = parse_args()
    print("[Stage 1 Tool] Initializing VLM and NLP models...")
    vlm, nlp = load_vlm_and_nlp(args.model_path, args.nlp_model, args.device)

    if args.question:
        questions = [{"sample_index": 0, "sample_id": "cli_test", "question": args.question.strip()}]
        print(f"[Stage 1 Tool] Interactive Mode: Evaluated 1 custom question.")
    else:
        questions = load_questions_from_dataset(
            args.dataset_path, args.all, args.max_samples,
            args.start_idx, args.end_idx, args.sample_idx
        )
        print(f"[Stage 1 Tool] Loaded {len(questions)} questions from dataset.")

    if not questions:
        print("[Stage 1 Tool] No questions to evaluate. Exiting.")
        return

    report = run_comparison(vlm, nlp, questions, verbose=args.verbose)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[Stage 1 Tool] Comparison report and diffs saved to '{args.output_path}'.")


if __name__ == "__main__":
    main()
