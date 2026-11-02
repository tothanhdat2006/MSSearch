"""
Stage 6 Utilities: Data Structures, Answer Extraction, Negative/Absence Filtering,
Bounding Box Expansion, and Deduplication Consensus.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from difflib import SequenceMatcher
import re


@dataclass
class EvidenceExtraction:
    """Records evidence found in a single candidate crop."""
    img_idx: int
    box: Tuple[int, int, int, int]
    expanded_box: Tuple[int, int, int, int]
    raw_answer: str
    source_branch: str
    final_answer: str = ""


@dataclass
class FinalSearchResult:
    """Final output container for the Dual-Path Visual Search Engine."""
    answers: List[str]
    evidence_image_indices: List[int]
    extractions: List[EvidenceExtraction] = field(default_factory=list)
    crop_records: List[Dict[str, Any]] = field(default_factory=list)


def extract_final_answer(text: str) -> str:
    """
    Extracts the clean, concise final answer from model output text.
    Strips markdown wrappers, parentheses, brackets, and quotes directly.
    """
    if not text:
        return ""
    m = re.search(r'<answer>([\s\S]*?)</answer>', text, flags=re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        return re.sub(r'^[\*\(\[\"\'\s]+|[\*\)\]\"\'\.\s]+$', '', val).strip()
    cleaned = text.strip()
    return re.sub(r'^[\*\(\[\"\'\s]+|[\*\)\]\"\'\.\s]+$', '', cleaned).strip()


def expand_bounding_box(
    box: Any,
    img_w: int,
    img_h: int,
    expansion_factor: float = 1.25
) -> Tuple[int, int, int, int]:
    """
    Independently expands a bounding box by expansion_factor (e.g. 1.25x = 25% margin)
    clamped to image boundary, guaranteed pure Python ints.
    """
    if hasattr(box, "tolist"):
        box = box.tolist()
    x1, y1, x2, y2 = [int(round(float(c))) for c in box]
    w = x2 - x1
    h = y2 - y1

    delta_w = int(round((expansion_factor - 1.0) / 2.0 * w))
    delta_h = int(round((expansion_factor - 1.0) / 2.0 * h))

    ex1 = max(0, x1 - delta_w)
    ey1 = max(0, y1 - delta_h)
    ex2 = min(img_w, x2 + delta_w)
    ey2 = min(img_h, y2 + delta_h)

    return (int(ex1), int(ey1), int(ex2), int(ey2))


def is_negative_or_unanswerable(text: str) -> bool:
    """
    Detects whether VLM output indicates absence / refusal / non-visibility of target object.
    Preserves valid answers like numbers, single characters, and road/store signs.
    """
    cleaned = text.strip().rstrip(".").rstrip(",")
    if not cleaned:
        return True
    cleaned_lower = cleaned.lower()

    # 1. Standalone negative words / short phrases
    standalone_refusals = {
        "none", "nothing", "null", "unknown", "na", "n/a", "nil", "empty",
        "no answer", "no answer found", "unanswerable", "cannot answer",
        "not found", "not visible", "no text", "no translation needed",
        "no visible text", "no target", "no object", "none visible",
        "not applicable", "no information", "not enough information",
        "not present", "not clearly visible", "not depicted", "unclear"
    }
    if cleaned_lower in standalone_refusals:
        return True

    # 2. Short absence phrases: 'no <object>'
    if re.match(r"^no\s+(?!(?:entry|parking|turn|stopping|standing|trespassing|smoking|littering)\b)[a-z0-9_,\s]+$", cleaned_lower):
        return True

    # 3. Contextual absence and refusal regex patterns
    negative_patterns = [
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
        r"not visible in this (?:image|frame|crop|picture|photo)\b",
        r"image (?:you provided )?does not show\b",
        r"image (?:you provided )?does not contain\b",
        r"image does not show\b",
        r"image does not contain\b",
        r"\bnone visible\b",
        r"\bnot found\b",
        r"no (?:\w+\s+){0,5}(?:found|detected|seen|present)\b",
        r"no (?:\w+\s+){0,5}(?:or \w+\s+){0,3}present\b",
        r"\bare the borders around\b",
        r"\bare the lines at the\b",
        r"\bare lines at the\b",
        r"\bindicate areas of concern\b",
    ]

    for pat in negative_patterns:
        if re.search(pat, cleaned_lower):
            return True

    return False


def split_attribute_answer(ans_str: str, *args, **kwargs) -> List[str]:
    """
    Splits compound attribute strings into a deduplicated list of concise attribute tokens.
    Handles delimiters: semicolons, commas, ' and ', ' & ', ' or ', and slashes.
    Strips boilerplate conversational preambles.
    """
    if not ans_str or not ans_str.strip():
        return []

    cleaned = re.sub(
        r"^(?:the\s+[\w\s]+\s+(?:is|are|appear to be|have|has)\s+|it\s+is\s+|they\s+are\s+)",
        "",
        ans_str.strip(),
        flags=re.IGNORECASE
    )

    parts = re.split(r"[,;]|\s+(?:and|&|or)\s+|(?:\s*/\s*)", cleaned)

    tokens: List[str] = []
    seen = set()
    for p in parts:
        tok = p.strip().strip(".").strip("\"'").strip()
        tok_clean = re.sub(r"^(?:a|an|the)\s+", "", tok, flags=re.IGNORECASE).strip()
        if tok_clean and len(tok_clean) >= 2 and not is_negative_or_unanswerable(tok_clean):
            key = tok_clean.lower()
            if key not in seen:
                seen.add(key)
                tokens.append(tok_clean)
    return tokens


def deduplicate_answers(answers: List[str], similarity_threshold: float = 0.85, verbose: bool = False) -> List[str]:
    """
    Deduplicates answer strings via Levenshtein / SequenceMatcher semantic similarity.
    Keeps the most descriptive, higher-entropy answer among near-duplicates,
    and returns unique answers sorted by support count (descending consensus).
    """
    valid_answers = [ans.strip() for ans in answers if ans.strip() and not is_negative_or_unanswerable(ans)]
    if not valid_answers:
        return []

    unique_clusters: List[Dict[str, Any]] = []

    for ans in valid_answers:
        matched = False
        for cluster in unique_clusters:
            rep = cluster["rep"]
            ratio = SequenceMatcher(None, ans.lower(), rep.lower()).ratio()
            if ratio >= similarity_threshold:
                matched = True
                cluster["count"] += 1
                if verbose:
                    print(f"[Stage 6][Deduplication] Near-duplicate detected (sim={ratio:.2f}): '{ans}' vs '{rep}'. Merging.")
                if len(ans) > len(rep):
                    cluster["rep"] = ans
                break

        if not matched:
            unique_clusters.append({"rep": ans, "count": 1})

    unique_clusters.sort(key=lambda c: (c["count"], len(c["rep"])), reverse=True)
    return [c["rep"] for c in unique_clusters]


def is_singular_query(question: str, question_type: Optional[str] = None) -> bool:
    """
    Determines whether a query seeks a single factual value rather than open-set enumeration.
    """
    if not question:
        return True
    q_lower = question.lower()
    plural_cues = [
        "all", "each", "every", "list", "how many", "count",
        "which ones", "which images", "which cars", "which people",
        "any other", "what are the"
    ]
    if any(re.search(rf"\b{re.escape(w)}\b", q_lower) for w in plural_cues):
        return False
    if question_type in ["counting"]:
        return False
    return True
