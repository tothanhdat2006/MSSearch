"""
Stage 4: Source-Attributed Consensus (SAC).
Filters uninformative crop responses, aggregates answer agreement via string clustering,
and applies lexicographical ranking to jointly predict the final answer and supporting image.
"""

from difflib import SequenceMatcher
import re
from typing import List, Tuple, Dict, Any, Optional

from ..config import STASHConfig
from .visual_screening import CandidateExtraction

# Standalone refusals and absence indicators
ABSENCE_TERMS = {
    "none", "nothing", "null", "unknown", "na", "n/a", "nil", "empty",
    "no answer", "no answer found", "unanswerable", "cannot answer",
    "not found", "not visible", "no text", "no translation needed",
    "no visible text", "no target", "no object", "none visible",
    "not applicable", "no information", "not enough information",
    "not present", "not clearly visible", "not depicted", "unclear"
}

NEGATIVE_REGEX_PATTERNS = [
    r"does not (?:show|contain|depict|feature|display|include)\b",
    r"doesn't (?:show|contain|depict|feature|display|include)\b",
    r"did not (?:show|contain|depict|feature|display|include)\b",
    r"didn't (?:show|contain|depict|feature|display|include)\b",
    r"there is no\b",
    r"there are no\b",
    r"there's no\b",
    r"no (?:\w+\s+){0,5}(?:is |are )?visible\b",
    r"no visible\b",
    r"not visible\b",
    r"not clearly visible\b",
    r"cannot be seen\b",
    r"can't be seen\b",
    r"cannot see\b",
    r"can't see\b",
    r"unable to see\b",
    r"cannot find\b",
    r"can't find\b",
    r"unable to find\b",
    r"cannot be found\b",
    r"cannot answer\b",
    r"can't answer\b",
    r"unable to answer\b",
    r"cannot determine\b",
    r"can't determine\b",
    r"unable to determine\b",
    r"cannot be determined\b",
    r"no information\b",
    r"not enough information\b",
    r"not present\b",
    r"is absent\b",
    r"are absent\b",
    r"no indication of\b",
    r"no sign of\b",
    r"not depicted\b",
    r"not in the (?:image|frame|crop|picture|photo)\b",
    r"outside the (?:image|frame|crop|picture|photo)\b",
    r"not visible in the (?:image|frame|crop|picture|photo)\b",
    r"image (?:you provided )?does not show\b",
    r"image (?:you provided )?does not contain\b",
    r"image does not show\b",
    r"image does not contain\b",
    r"not visible in this (?:image|frame|crop|picture|photo)\b",
    r"\bnone visible\b",
    r"\bnot found\b",
    r"no (?:\w+\s+){0,5}(?:found|detected|seen|present)\b",
    r"no (?:\w+\s+){0,5}(?:or \w+\s+){0,3}present\b",
    r"\bare the borders around\b",
    r"\bare lines at the\b",
    r"\bare the lines at the\b",
    r"\bindicate areas of concern\b",
]


def is_option_choice(text: str, options_str: Optional[str] = None) -> bool:
    """
    Checks if candidate answer is an explicit multiple-choice selection:
    1. Exact single letter e.g. 'A', 'B.', '(C)', '[D]'
    2. Leading option label e.g. 'A. There is no contact', '(A) text', 'A: text', 'A - text'
    3. Explicit answer phrase anywhere e.g. 'The best answer is: A', 'Final answer: B'
    4. Match against candidate option texts in options_str if available
    """
    if not text or not isinstance(text, str):
        return False

    clean = text.strip()
    clean = re.sub(r"^\*+([A-Ka-k])\*+$", r"\1", clean).strip()
    clean = re.sub(r"^`+([A-Ka-k])`+$", r"\1", clean).strip()

    # 1. Exact single letter
    if re.match(r"^[\s\*\(\[\"']*([A-Ka-k])[\s\*\)\]\.\:\'\"]*$", clean):
        return True

    # 2. Leading option label (e.g. 'A. There is no contact', '(B) Direct collision', 'A: ', 'C - ')
    if re.match(r"^[\s\*\(\[\"']*([A-Ka-k])(?:[\)\]\.\:\-]|(?:\s*\n))\s*", clean):
        return True

    # 3. Explicit answer phrase anywhere in text (e.g. 'The best answer is: A', 'Answer is (B)')
    if re.search(
        r"\b(?:final answer|best answer|correct answer|answer|choice|option)\b"
        r"(?:\s+(?:is|to|be|would be|must be|of|should be|selected is))?"
        r"[\s\:\*\(\[\"\']*([A-Ka-k])\b",
        clean,
        re.IGNORECASE
    ):
        return True

    # 4. Direct match against candidate option text lines in options_str if provided
    if options_str:
        clean_lower = clean.rstrip(".").rstrip(",").lower()
        for line in options_str.splitlines():
            line = line.strip()
            opt_m = re.match(r"^[\(\[]?([A-Ka-k])[\)\]\.\:\s]+(.*)", line)
            if opt_m:
                opt_content = opt_m.group(2).strip().rstrip(".").rstrip(",").lower()
                if opt_content and (clean_lower == opt_content or clean_lower.startswith(opt_content) or opt_content.startswith(clean_lower)):
                    return True

    return False


def is_uninformative_response(
    text: str,
    allow_none: bool = False,
    options_str: Optional[str] = None
) -> bool:
    """Checks whether candidate text indicates an unanswerable crop or missing target."""
    cleaned = text.strip().rstrip(".").rstrip(",")
    if not cleaned:
        return True

    # 1. If text is an explicit option choice (e.g. 'A. There is no contact at all.'), do NOT filter!
    if is_option_choice(cleaned, options_str=options_str):
        return False

    # 2. For benchmarks like MME-RealWorld-Lite, 'none' answers are real valid options
    if allow_none:
        return False

    lower = cleaned.lower()

    if lower in ABSENCE_TERMS:
        return True

    # Check for 'no <object>' (excluding phrases like 'no parking', 'no smoking')
    if re.match(r"^no\s+(?!(?:entry|parking|turn|stopping|standing|trespassing|smoking|littering)\b)[a-z0-9_,\s]+$", lower):
        return True

    for pat in NEGATIVE_REGEX_PATTERNS:
        if re.search(pat, lower):
            return True

    return False


def normalize_answer_string(ans: str) -> str:
    """Normalizes answer text by removing leading articles and extra whitespace."""
    cleaned = re.sub(r"^(?:a|an|the)\s+", "", ans.strip(), flags=re.IGNORECASE)
    return cleaned.strip()


def resolve_source_attributed_consensus(
    extractions: List[CandidateExtraction],
    config: STASHConfig,
    options_str: Optional[str] = None
) -> Tuple[str, Optional[int]]:
    """
    Executes Source-Attributed Consensus (SAC):
    1. Filters uninformative / unanswerable responses.
    2. Clusters answer strings by Gestalt similarity or substring matching.
    3. Ranks clusters lexicographically by:
       - Support count C_j = |C_j|
       - Weighted relevance support W_j = sum_{e in C_j} (1 + s_e)
       - Maximum representative crop area M_j = max_{e in C_j} A_e
    4. Selects representative extraction with highest alignment score s_e.

    Returns:
        (predicted_answer, evidence_image_index)
    """
    allow_none = getattr(config, "allow_none_answer", False)
    valid = [
        e for e in extractions
        if e.final_answer and not is_uninformative_response(
            e.final_answer,
            allow_none=allow_none,
            options_str=options_str
        )
    ]
    if not valid:
        if config.verbose:
            print("[Stage 4: Consensus (SAC)] 0 valid responses surviving filter -> Fallback 'No answer found'.")
        return ("No answer found", None)

    clusters: List[Dict[str, Any]] = []

    for e in valid:
        ans = e.final_answer.strip()
        norm_ans = normalize_answer_string(ans).lower()

        matched_cluster = None
        for c in clusters:
            rep_norm = normalize_answer_string(c["rep"]).lower()
            sim = SequenceMatcher(None, norm_ans, rep_norm).ratio()
            has_substring = (len(norm_ans) >= 3 and norm_ans in rep_norm) or \
                            (len(rep_norm) >= 3 and rep_norm in norm_ans)

            if sim >= config.answer_dedup_similarity or has_substring:
                matched_cluster = c
                break

        weight = 1.0 + max(0.0, e.alignment_score)
        area = e.crop_area

        if matched_cluster is not None:
            matched_cluster["count"] += 1
            matched_cluster["weighted_score"] += weight
            matched_cluster["extractions"].append(e)

            best_curr = matched_cluster["best_extraction"]
            if (e.alignment_score > best_curr.alignment_score) or \
               (e.alignment_score == best_curr.alignment_score and area > matched_cluster["max_area"]):
                matched_cluster["best_extraction"] = e
                matched_cluster["max_area"] = max(matched_cluster["max_area"], area)

            # Update representative string if cleaner/longer within moderate bounds
            if len(ans) > len(matched_cluster["rep"]) and len(ans.split()) <= 4:
                matched_cluster["rep"] = ans
        else:
            clusters.append({
                "rep": ans,
                "count": 1,
                "weighted_score": weight,
                "max_area": area,
                "best_extraction": e,
                "extractions": [e]
            })

    # Lexicographical ranking: (count, weighted_score, max_area)
    clusters.sort(
        key=lambda c: (c["count"], c["weighted_score"], c["max_area"]),
        reverse=True
    )

    winning = clusters[0]
    best_extraction: CandidateExtraction = winning["best_extraction"]

    predicted_answer = winning["rep"]
    evidence_img_idx = best_extraction.img_idx

    if config.verbose:
        print(f"[Stage 4: Consensus (SAC)] Winner: '{predicted_answer}' (Image {evidence_img_idx}) | "
              f"Support: {winning['count']} | Weight: {winning['weighted_score']:.2f} | Area: {winning['max_area']} px^2")

    return (predicted_answer, evidence_img_idx)
