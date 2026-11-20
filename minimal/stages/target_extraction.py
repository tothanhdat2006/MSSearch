"""
Stage 1: Target Extraction via MLLM Language Backbone.
Extracts active visual search targets O_Q = {o_1, ..., o_m} using 3-shot in-context prompting.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
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


def build_few_shot_prompt(question: str) -> str:
    """Builds single text prompt for text-completion VLMs (e.g. LLaVA-OV, InternVL). Ported from utils_stage1.py line 149."""
    return (
        "You are an expert visual search parser. Given a question, return a JSON object with 'active_objects' and 'question_type'.\n\n"
        "Example 1 (Single-Object Attribute):\n"
        "Question: \"What is the registration number of the boat?\"\n"
        "JSON: {\"active_objects\": [\"boat\"], \"question_type\": \"single_object_attribute\"}\n\n"
        "Example 2 (Single-Object Attribute):\n"
        "Question: \"What is the color of the mailbox?\"\n"
        "JSON: {\"active_objects\": [\"mailbox\"], \"question_type\": \"single_object_attribute\"}\n\n"
        "Example 3 (Counting):\n"
        "Question: \"How many windows are on the brick building?\"\n"
        "JSON: {\"active_objects\": [\"brick building\", \"window\"], \"question_type\": \"counting\"}\n\n"
        f"Question: \"{question}\"\nJSON:"
    )


def format_prompt_for_text_completion(prompt_or_messages: Union[str, List[Dict[str, Any]]]) -> str:
    """
    Converts prompt or chat messages into a single text prompt suitable for
    legacy text-completion VLMs (e.g. LLaVA-OV, InternVL).
    Handles string prompts or message lists without raising TypeError on list contents.
    """
    if isinstance(prompt_or_messages, str):
        return prompt_or_messages

    if isinstance(prompt_or_messages, list):
        # Extract question if chat messages were passed from build_target_extraction_messages
        question_found = None
        for m in reversed(prompt_or_messages):
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, list):
                    c_text = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
                else:
                    c_text = str(content)
                match = re.search(r'Question:\s*"([^"]+)"', c_text)
                if match:
                    question_found = match.group(1)
                    break
        if question_found is not None:
            return build_few_shot_prompt(question_found)

        # Safely extract text from content without raising TypeError
        text_parts = []
        for m in prompt_or_messages:
            if isinstance(m, dict):
                c = m.get("content", "")
                if isinstance(c, list):
                    text_parts.append("".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in c))
                elif isinstance(c, str):
                    text_parts.append(c)
            else:
                text_parts.append(str(m))
        return "\n".join(text_parts)

    return str(prompt_or_messages)


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
    Normalizes candidate target string:
    - Trims punctuation at edges only (preserves internal hyphens like 'open-front bar').
    - Preserves nouns/plurals ('pants', 'shoes') without aggressive singularization.
    - Preserves articles (matching normalize_target_text).
    - Handles 'all ' prefix by removing prefix and trailing 's'.
    """
    t = target.strip()
    if not t:
        return ""
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
    few_shot_prompt = build_few_shot_prompt(question)

    raw_response = ""
    # In the original code (stage1_parser.py), LLaVA-OV-7B (ModelGlobalLocal) and InternVL3-8B
    # (ModelInternvl) did not use chat messages: stage1_parser.py sent build_few_shot_prompt
    # (utils_stage1.py, line 149) through generate_response(image_pil=None, question=few_shot_prompt).
    # Qwen-VL (ModelQwenVL) has a native chat template processor and uses generate_text(messages).
    is_response_model = (
        type(mllm).__name__ in ("ModelGlobalLocal", "ModelInternvl")
        or (hasattr(mllm, "generate_response") and not hasattr(mllm, "processor"))
    )

    if is_response_model and hasattr(mllm, "generate_response"):
        try:
            raw_response = mllm.generate_response(image_pil=None, question=few_shot_prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            if verbose:
                print(f"[Stage 1 Error] generate_response failed ({e}), falling back to generate_text")
            if hasattr(mllm, "generate_text"):
                raw_response = mllm.generate_text(few_shot_prompt, max_new_tokens=max_new_tokens)
    elif hasattr(mllm, "generate_text"):
        try:
            raw_response = mllm.generate_text(messages, max_new_tokens=max_new_tokens)
        except Exception:
            raw_response = mllm.generate_text(few_shot_prompt, max_new_tokens=max_new_tokens)
    elif hasattr(mllm, "generate_response"):
        raw_response = mllm.generate_response(image_pil=None, question=few_shot_prompt, max_new_tokens=max_new_tokens)

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


# Backward-compatible alias
extract_targets = extract_visual_targets
