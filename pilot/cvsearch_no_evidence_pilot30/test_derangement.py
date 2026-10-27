"""
Test suite for CVSearch No-Evidence Pilot Study (N=30)
Validates strict derangement, path resolution, and evidence realization metric logic.
Run with: python test_derangement.py
"""

import os
import sys
from pathlib import Path

# Add current directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pilot_study import (
    generate_strict_derangement,
    resolve_cvsearch_repo,
    evaluate_evidence_realization,
    compute_summary_metrics
)


def test_derangement():
    print("[TEST] Running strict derangement validation...")
    mock_dataset = [
        {
            "index": i,
            "input_image": f"image/{i}.jpg",
            "question": f"Question about object in scene {i}?",
            "options": ["A. opt1", "B. opt2", "C. opt3", "D. opt4"],
            "answer": "A"
        }
        for i in range(50)
    ]

    # Test N=30 sampling and derangement
    n_samples = 30
    mismatched, perm = generate_strict_derangement(mock_dataset, n_samples=n_samples, seed=42)

    assert len(mismatched) == n_samples, f"Expected {n_samples} items, got {len(mismatched)}"
    assert len(perm) == n_samples, f"Expected {n_samples} perm entries, got {len(perm)}"

    for idx, item in enumerate(mismatched):
        donor_idx = item["donor_question_idx"]
        assert idx != donor_idx, f"Self-match at index {idx}!"
        assert item["input_image"] != item["mismatched_donor_image_path"], (
            f"Image collision at index {idx}: {item['input_image']}"
        )
        assert item["question"] != item["original_true_question"], (
            f"Question collision at index {idx}: {item['question']}"
        )
        assert item["is_mismatched"] is True
        assert item["question"] == mock_dataset[donor_idx]["question"]
        assert item["input_image"] == mock_dataset[idx]["input_image"]

    print("  -> Passed: 30/30 derangements strictly collision-free.")


def test_path_resolution():
    print("[TEST] Running CVSearch path resolution validation...")
    repo_path = resolve_cvsearch_repo()
    assert os.path.isdir(repo_path), f"Resolved repo path does not exist: {repo_path}"
    # Must contain cvsearch module or CVSearch.py
    has_cvsearch = (
        os.path.isfile(os.path.join(repo_path, "cvsearch", "CVSearch.py")) or
        os.path.isfile(os.path.join(repo_path, "CVSearch.py")) or
        os.path.isdir(os.path.join(repo_path, "cvsearch"))
    )
    assert has_cvsearch, f"Directory does not appear to be CVSearch repo: {repo_path}"
    print(f"  -> Passed: Resolved valid CVSearch repo at: {repo_path}")


def test_evidence_realization():
    print("[TEST] Running evidence realization diagnostic metrics...")
    
    # Case 1: Realized no evidence (low root conf, SAM3 detected nothing, fine search failed)
    r1 = evaluate_evidence_realization(root_ans_conf=0.2, sam_success_flags=[0], fine_search_success=False)
    assert r1["realized_no_evidence"] is True, "Failed Case 1"
    assert r1["failure_mode"] is None

    # Case 2: Root prior hallucination (root conf > 0.6)
    r2 = evaluate_evidence_realization(root_ans_conf=0.85, sam_success_flags=[], fine_search_success=None)
    assert r2["realized_no_evidence"] is False, "Failed Case 2"
    assert r2["failure_mode"] == "root_prior_hallucination"
    assert r2["root_hallucinated"] is True

    # Case 3: SAM3 false positive (sam segmented non-existent target)
    r3 = evaluate_evidence_realization(root_ans_conf=-0.1, sam_success_flags=[1], fine_search_success=None)
    assert r3["realized_no_evidence"] is False, "Failed Case 3"
    assert r3["failure_mode"] == "sam3_false_positive"
    assert r3["sam_hallucinated"] is True

    # Case 4: Fine search false positive (fine search claimed success)
    r4 = evaluate_evidence_realization(root_ans_conf=0.1, sam_success_flags=[0], fine_search_success=True)
    assert r4["realized_no_evidence"] is False, "Failed Case 4"
    assert r4["failure_mode"] == "fine_search_false_positive"
    assert r4["fine_search_hallucinated"] is True

    print("  -> Passed: Evidence realization rules correctly categorized all 4 states.")


def test_metrics_aggregation():
    print("[TEST] Running summary metrics aggregation...")
    mock_records = [
        {"realized_no_evidence": True, "root_hallucinated": False, "sam_hallucinated": False, "fine_search_hallucinated": False, "is_correct_donor": False, "root_ans_conf": 0.1, "num_pops": [4]},
        {"realized_no_evidence": False, "root_hallucinated": True, "sam_hallucinated": False, "fine_search_hallucinated": False, "is_correct_donor": True, "root_ans_conf": 0.8, "num_pops": []},
        {"realized_no_evidence": False, "root_hallucinated": False, "sam_hallucinated": True, "fine_search_hallucinated": False, "is_correct_donor": False, "root_ans_conf": -0.2, "num_pops": [0]},
        {"realized_no_evidence": False, "root_hallucinated": False, "sam_hallucinated": False, "fine_search_hallucinated": True, "is_correct_donor": False, "root_ans_conf": 0.3, "num_pops": [6]},
    ]
    summary = compute_summary_metrics(mock_records)
    assert summary["n_samples"] == 4
    assert summary["realization_rate"] == 0.25
    assert summary["root_hallucination_rate"] == 0.25
    assert summary["sam_hallucination_rate"] == 0.25
    assert summary["fine_search_hallucination_rate"] == 0.25
    assert summary["donor_accuracy"] == 0.25
    print("  -> Passed: Summary metrics correctly aggregated.")


if __name__ == "__main__":
    test_derangement()
    test_path_resolution()
    test_evidence_realization()
    test_metrics_aggregation()
    print("\n[ALL CHECKS PASSED] CVSearch No-Evidence Pilot components verified.")
