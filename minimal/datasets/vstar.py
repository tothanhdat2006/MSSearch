"""
V*Bench (Vstar) Dataset Loader for STASH-Search Minimal.
Loads single-image high-resolution visual search benchmarks containing
sub-folders 'direct_attributes/' and 'relative_position/' with annotation_vstar.json.
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from .base import BenchmarkSample

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


def parse_options_from_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Parses options from text formatted like:
    '(A) brown (B) red (C) purple (D) black' or
    'A. brown\nB. red\nC. purple\nD. black'.
    
    Returns:
        letters: list of option letters e.g. ['A', 'B', 'C', 'D']
        texts: list of option text strings e.g. ['brown', 'red', 'purple', 'black']
    """
    if not text or not isinstance(text, str):
        return [], []

    clean_text = text.strip()

    # Pattern 1: Parenthesized letters e.g. (A) ... (B) ... (C) ... (D) ...
    matches = list(re.finditer(r'\(([A-Ha-h])\)', clean_text))
    if len(matches) >= 2:
        letters = []
        texts = []
        for i, m in enumerate(matches):
            letters.append(m.group(1).upper())
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(clean_text)
            t = clean_text[start:end].strip()
            if i == len(matches) - 1:
                # Strip trailing instructions e.g. "Answer with the option's letter from the given choices directly."
                m_inst = re.search(r'\s+(?:Answer\s+(?:with|the)|Select\s+the|Please\s+answer)\b.*$', t, re.IGNORECASE)
                if m_inst:
                    t = t[:m_inst.start()].strip()
            texts.append(t)
        if len(letters) == len(texts) and all(texts):
            return letters, texts

    # Pattern 2: Dotted or bracketed letters e.g. A. ... B. ... or A) ... B) ...
    matches = list(re.finditer(r'(?:^|[\s\n])([A-Ha-h])[\.\)]\s+', clean_text))
    if len(matches) >= 2:
        letters = []
        texts = []
        for i, m in enumerate(matches):
            letters.append(m.group(1).upper())
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(clean_text)
            t = clean_text[start:end].strip()
            if i == len(matches) - 1:
                m_inst = re.search(r'\s+(?:Answer\s+(?:with|the)|Select\s+the|Please\s+answer)\b.*$', t, re.IGNORECASE)
                if m_inst:
                    t = t[:m_inst.start()].strip()
            texts.append(t)
        if len(letters) == len(texts) and all(texts):
            return letters, texts

    return [], []


def format_options(options_list: List[str]) -> Tuple[str, Dict[str, str]]:
    """Formats raw option strings into letter-labeled candidate options."""
    formatted_lines = []
    letter_to_text: Dict[str, str] = {}
    for i, opt in enumerate(options_list):
        letter = OPTION_LETTERS[i] if i < len(OPTION_LETTERS) else f"Option_{i}"
        clean_opt = str(opt).strip()
        formatted_lines.append(f"{letter}. {clean_opt}")
        letter_to_text[letter] = clean_opt
    return "\n".join(formatted_lines), letter_to_text


def load_vstar(
    dataset_dir: str,
    annotation_path: Optional[str] = None
) -> List[BenchmarkSample]:
    """
    Loads V*Bench evaluation samples from annotation_vstar.json.

    Args:
        dataset_dir: Root directory containing 'direct_attributes/', 'relative_position/',
                     and 'annotation_vstar.json'.
        annotation_path: Optional explicit path to annotation_vstar.json.

    Returns:
        List of BenchmarkSample instances.
    """
    anno_file = annotation_path
    if not anno_file or not os.path.isfile(anno_file):
        candidates = [
            os.path.join(dataset_dir, "annotation_vstar_updated.json"),
            os.path.join(dataset_dir, "annotation_vstar.json"),
            os.path.join(dataset_dir, "annotations.json"),
            os.path.join(dataset_dir, "vstar.json")
        ]
        for c in candidates:
            if os.path.isfile(c):
                anno_file = c
                break

    if not anno_file or not os.path.isfile(anno_file):
        raise FileNotFoundError(f"[V*Bench] Neither annotation_vstar_updated.json nor annotation_vstar.json found in '{dataset_dir}'.")

    with open(anno_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, dict):
        raw_items = raw_data.get("samples") or raw_data.get("data") or list(raw_data.values())
    else:
        raw_items = raw_data

    bench_samples: List[BenchmarkSample] = []

    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue

        sample_id = str(item.get("id") or item.get("sample_idx") or f"vstar_{idx:04d}")
        question = item.get("question", "")
        test_type = item.get("test_type") or item.get("category") or "direct_attributes"
        input_image_rel = item.get("input_image") or item.get("image", "")

        full_img_path = os.path.join(dataset_dir, input_image_rel) if not os.path.isabs(input_image_rel) else input_image_rel

        # In annotation_vstar_updated.json, shuffled options matching the ground truth 'label'
        # are stored in the 'text' field e.g. "(A) brown (B) red (C) purple (D) black".
        # In legacy annotation_vstar.json, option 0 in item["options"] is canonically the GT target ("A").
        text_field = item.get("text", "")
        parsed_letters, parsed_texts = parse_options_from_text(text_field) if text_field else ([], [])

        if not question and parsed_letters and text_field:
            m_first = re.search(r'\([A-Ha-h]\)|(?:^|[\s\n])[A-Ha-h][\.\)]\s+', text_field)
            if m_first and m_first.start() > 0:
                question = text_field[:m_first.start()].strip()

        if parsed_letters and parsed_texts:
            options_str = "\n".join([f"{l}. {t}" for l, t in zip(parsed_letters, parsed_texts)])
            letter_map = dict(zip(parsed_letters, parsed_texts))
            options_list = parsed_texts
        else:
            options_list = item.get("options", [])
            if options_list:
                options_str, letter_map = format_options(options_list)
            else:
                options_str = None
                letter_map = {}

        raw_ans = item.get("label") if item.get("label") is not None else item.get("answer")
        if raw_ans is not None:
            raw_ans_str = str(raw_ans).strip()
            m = re.search(r"\b([A-Ha-h])\b", raw_ans_str) or re.search(r"([A-Ha-h])", raw_ans_str)
            if m:
                gt_letter = m.group(1).upper()
            else:
                try:
                    num_idx = int(raw_ans_str)
                    gt_letter = OPTION_LETTERS[num_idx] if num_idx < len(OPTION_LETTERS) else "A"
                except ValueError:
                    gt_letter = "A"
                    for let, opt_t in letter_map.items():
                        if opt_t.lower() == raw_ans_str.lower():
                            gt_letter = let
                            break
        else:
            gt_letter = "A"

        gt_text = letter_map.get(gt_letter, options_list[0] if options_list else "")

        # Use the prompt inside 'text' for Vstar when present:
        # e.g. "What is the color of the woman's jacket? (A) black (B) red (C) blue (D) orange Answer with the option's letter from the given choices directly."
        if text_field and text_field.strip():
            inspection_prompt = text_field.strip()
        elif options_str:
            inspection_prompt = f"{question}\n{options_str.strip()}\nAnswer the option letter directly."
        else:
            inspection_prompt = question

        bench_samples.append(BenchmarkSample(
            sample_id=sample_id,
            question=question,
            candidate_image_paths=[os.path.abspath(full_img_path)],
            gt_answer=gt_letter,
            gt_image_idx=0,
            options_str=options_str,
            category=test_type,
            metadata={
                "benchmark": "vstar",
                "test_type": test_type,
                "input_image": input_image_rel,
                "gt_text": gt_text,
                "options": options_list
            },
            inspection_prompt=inspection_prompt
        ))

    return bench_samples
