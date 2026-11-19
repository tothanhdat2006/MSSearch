"""
Stage 1: Target Extraction via MLLM Language Backbone.
Extracts active visual search targets O_Q = {o_1, ..., o_m} using 3-shot in-context prompting.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from ..models.base import BaseMLLM

# Canonical closed-class English pronouns to filter out
ENGLISH_PRONOUNS = {
    "it", "its", "they", "them", "their", "theirs", "this", "that", "these", "those",
    "he", "him", "his", "she", "her", "hers", "we", "us", "our", "ours", "you", "your",
    "yours", "i", "me", "my", "mine", "who", "whom", "whose", "which", "what"
}

# Stop words to exclude during naive fallback parsing
QUESTION_STOP_WORDS = {
    "where", "when", "what", "which", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "located", "found", "seen", "visible", "showing", "depicted", "picture", "image",
    "photo", "count", "number", "many", "much", "total", "tell", "describe", "find"
}


def build_target_extraction_messages(question: str) -> List[Dict[str, Any]]:
    """Constructs the canonical 3-shot chat messages for target extraction."""
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are an expert visual search parser. Given a question, extract search targets and return a JSON object with:\n"
                    "- \"active_objects\": list of key visual search entities or text needed to answer the question\n"
                    "- \"question_type\": one of ['single_object_attribute', 'counting', 'text_reading', 'general']\n"
                    "Return ONLY valid JSON."
                )
            }]
        },
        {"role": "user", "content": [{"type": "text", "text": "Question: \"What is the registration number of the boat?\""}]},
        {"role": "assistant", "content": [{"type": "text", "text": '{"active_objects": ["boat"], "question_type": "single_object_attribute"}'}]},
        {"role": "user", "content": [{"type": "text", "text": "Question: \"What is the color of the mailbox?\""}]},
        {"role": "assistant", "content": [{"type": "text", "text": '{"active_objects": ["mailbox"], "question_type": "single_object_attribute"}'}]},
        {"role": "user", "content": [{"type": "text", "text": "Question: \"How many windows are on the brick building?\""}]},
        {"role": "assistant", "content": [{"type": "text", "text": '{"active_objects": ["brick building", "window"], "question_type": "counting"}'}]},
        {"role": "user", "content": [{"type": "text", "text": f"Question: \"{question}\""}]}
    ]


def parse_mllm_json(raw_text: str) -> List[str]:
    """Extracts raw candidate targets from JSON, bracket list, or bullet output."""
    raw_targets: List[str] = []

    # 1. JSON dict match
    dict_match = re.search(r'\{[\s\S]*?\}', raw_text)
    if dict_match:
        try:
            data = json.loads(dict_match.group(0))
            if isinstance(data, dict):
                objs = data.get("active_objects", [])
                if isinstance(objs, list):
                    raw_targets = [str(o).strip() for o in objs if str(o).strip()]
        except Exception:
            pass

    # 2. JSON list match [ ... ]
    if not raw_targets:
        list_match = re.search(r'\[[\s\S]*?\]', raw_text)
        if list_match:
            try:
                data = json.loads(list_match.group(0))
                if isinstance(data, list):
                    raw_targets = [str(o).strip() for o in data if str(o).strip()]
            except Exception:
                pass

    # 3. Bullet list fallback
    if not raw_targets:
        bullets = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+\.)\s*([^\n]+)', raw_text)
        if bullets:
            raw_targets = [b.strip().strip('"\'') for b in bullets if b.strip()]

    return raw_targets


def normalize_target(target: str) -> str:
    """
    Normalizes candidate target string matching the reference 85d3edb pipeline:
    - Strips leading articles (a, an, the, this, that, these, those).
    - Trims punctuation at edges only (preserves internal hyphens like 'open-front bar').
    - Preserves nouns/plurals ('pants', 'shoes') without aggressive singularization.
    - Handles 'all ' prefix by removing prefix and trailing 's'.
    """
    t = target.strip()
    if not t:
        return ""
    # Strip leading articles
    t = re.sub(r'^(?:the|a|an|this|that|these|those)\s+', '', t, flags=re.IGNORECASE).strip()
    # Strip boundary punctuation/whitespace only
    t = re.sub(r'^[\W_]+|[\W_]+$', '', t)
    # Check "all " prefix
    if t.lower().startswith("all "):
        t = t[4:].strip()
        if t.lower().endswith("s"):
            t = t[:-1].strip()
    return t.strip().lower()


def parse_visual_query_structure(question: str) -> Tuple[str, str, str]:
    """
    Regex fallback for syntactic visual query templates:
    'What patterns are brown chairs?' -> ('pattern', 'brown', 'chair')
    """
    q_clean = question.strip()
    m = re.search(
        r"what\s+(patterns?|materials?|colors?|colours?|shapes?)\s+(?:are|is)\s+(?:the\s+)?(?:([\w\-]+)\s+)?([\w\-\s]+)\??",
        q_clean,
        re.IGNORECASE
    )
    if m:
        attr_raw = m.group(1).lower().rstrip("s")
        if attr_raw == "colour":
            attr_raw = "color"
        cond_raw = (m.group(2) or "").lower().strip()
        obj_raw = (m.group(3) or "").lower().strip()
        obj_clean = re.sub(r"s$", "", obj_raw) if len(obj_raw) > 3 and not obj_raw.endswith("ss") else obj_raw
        return attr_raw, cond_raw, obj_clean
    return "", "", ""


def extract_visual_targets(
    question: str,
    mllm: BaseMLLM,
    nlp_model: Optional[Any] = None,
    max_new_tokens: int = 96,
    verbose: bool = True
) -> List[str]:
    """
    Parses natural language question Q into structured visual targets O_Q.

    Args:
        question: Input user question.
        mllm: Multimodal LLM providing pure-text chat generation.
        nlp_model: Optional loaded spaCy model for linguistic filtering.
        max_new_tokens: Token generation ceiling.
        verbose: Verbose logging flag.

    Returns:
        List of cleaned, deduplicated visual target strings O_Q.
    """
    messages = build_target_extraction_messages(question)
    raw_response = mllm.generate_text(messages, max_new_tokens=max_new_tokens)

    raw_targets = parse_mllm_json(raw_response)

    # Post-process: clean, strip pronouns, and deduplicate
    clean_targets: List[str] = []
    for raw in raw_targets:
        norm = normalize_target(raw)
        if not norm or norm in ENGLISH_PRONOUNS:
            continue

        # spaCy pronoun check if available
        if nlp_model is not None:
            doc = nlp_model(norm)
            if any(tok.pos_ == "PRON" for tok in doc):
                continue

        if norm not in clean_targets:
            clean_targets.append(norm)

    # Safety fallback 1: Question Structure Syntax Regex Parser
    if not clean_targets or clean_targets == ["object"]:
        _, _, str_obj = parse_visual_query_structure(question)
        if str_obj:
            clean_targets = [str_obj]

    # Safety fallback 2: Naive keyword extraction excluding stop words
    if not clean_targets or clean_targets == ["object"]:
        q_norm = normalize_target(question)
        stop_words = ENGLISH_PRONOUNS | QUESTION_STOP_WORDS
        words = [w for w in q_norm.split() if w not in stop_words and len(w) > 2]
        clean_targets = [words[-1]] if words else ["object"]

    if verbose:
        print(f"[Stage 1: Target Extraction] Targets: {clean_targets} (from: '{question}')")

    return clean_targets
