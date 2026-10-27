"""
Self-check unit test for MSSearch multisource_200 evaluation suite.
Validates metric computation and dataset schema compatibility without requiring GPU/torch.
"""
import os
import json
from eval_200 import evaluate_predictions

def test_metrics():
    # 1. Exact single attribute
    res = evaluate_predictions(pred_text="brown", raw_pred_list=["brown"], gt_list=["brown"])
    assert res["precision"] == 1.0, f"Expected 1.0, got {res['precision']}"
    assert res["recall"] == 1.0, f"Expected 1.0, got {res['recall']}"
    assert res["f1"] == 1.0, f"Expected 1.0, got {res['f1']}"
    assert res["exact_match"] is True

    # 2. Precision penalty on extra predicted attributes
    res = evaluate_predictions(pred_text="brown, black", raw_pred_list=["brown", "black"], gt_list=["brown"])
    assert res["precision"] == 0.5, f"Expected 0.5, got {res['precision']}"
    assert res["recall"] == 1.0, f"Expected 1.0, got {res['recall']}"
    assert abs(res["f1"] - 0.6667) < 0.001, f"Expected ~0.6667, got {res['f1']}"
    assert res["exact_match"] is False

    # 3. Recall penalty on missing attributes
    res = evaluate_predictions(pred_text="brown", raw_pred_list=["brown"], gt_list=["brown", "white"])
    assert res["precision"] == 1.0, f"Expected 1.0, got {res['precision']}"
    assert res["recall"] == 0.5, f"Expected 0.5, got {res['recall']}"
    assert abs(res["f1"] - 0.6667) < 0.001, f"Expected ~0.6667, got {res['f1']}"
    assert res["exact_match"] is False

    # 4. XML tag parsing in raw direct text
    res = evaluate_predictions(pred_text="<answer>white, black</answer>", raw_pred_list=None, gt_list=["black", "white"])
    assert res["precision"] == 1.0, f"Expected 1.0, got {res['precision']}"
    assert res["recall"] == 1.0, f"Expected 1.0, got {res['recall']}"
    assert res["exact_match"] is True

    # 5. Punctuation and whitespace normalization
    res = evaluate_predictions(pred_text="  Rear view. ; standing! ", raw_pred_list=None, gt_list=["rear view", "standing"])
    assert res["precision"] == 1.0, f"Expected 1.0, got {res['precision']}"
    assert res["recall"] == 1.0, f"Expected 1.0, got {res['recall']}"

    print("[PASS] All metric evaluation assertion tests succeeded.")

def test_dataset_schema():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "datasets", "200_samples.json"))
    if not os.path.exists(dataset_path):
        print(f"[SKIP] Dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    assert len(samples) == 200, f"Expected 200 samples, got {len(samples)}"
    for idx, sample in enumerate(samples):
        assert "sample_id" in sample, f"Missing sample_id at index {idx}"
        assert "query" in sample, f"Missing query at index {idx}"
        assert "answer" in sample, f"Missing answer at index {idx}"
        assert "pools" in sample, f"Missing pools at index {idx}"
        assert "N_5" in sample["pools"], f"Missing N_5 pool at index {idx}"
        assert "N_10" in sample["pools"], f"Missing N_10 pool at index {idx}"
        assert "N_20" in sample["pools"], f"Missing N_20 pool at index {idx}"

    print(f"[PASS] Successfully validated 200 samples schema.")

if __name__ == "__main__":
    test_metrics()
    test_dataset_schema()
