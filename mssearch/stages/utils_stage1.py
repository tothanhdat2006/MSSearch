"""
Stage 1 Utilities: Data Structures, Query Parsing Heuristics, and Option Formatters.
"""

import re
import json
import ast
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class ParsedQuestion(list):
    """
    Structured question parsing result produced by Stage 1 LLM reasoning.
    Subclasses list so that it duck-types List[str] seamlessly for 100% backward compatibility
    with all downstream model and pipeline interfaces (e.g. len(p), p[0], iteration, SAM3 input).
    """
    def __init__(
        self,
        active_objects: List[str],
        question_type: str = "general",
        is_relational: bool = False,
        strategy: str = "",
        raw_response: str = "",
        anchors: Optional[List[str]] = None,
        target_element: str = "",
        target_entity: Optional[str] = None,
        anchor_entity: Optional[str] = None
    ):
        super().__init__(active_objects)
        self.active_objects = list(active_objects)
        self.question_type = str(question_type)
        self.is_relational = bool(is_relational)
        self.strategy = str(strategy)
        self.raw_response = str(raw_response)
        self.target_entity = target_entity or target_element or (active_objects[0] if active_objects else None)
        self.anchor_entity = anchor_entity or (anchors[0] if anchors else None)
        self.anchors = list(anchors) if anchors else ([self.anchor_entity] if self.anchor_entity else [])
        self.target_element = str(self.target_entity or "")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_objects": self.active_objects,
            "question_type": self.question_type,
            "is_relational": self.is_relational,
            "target_entity": self.target_entity,
            "anchor_entity": self.anchor_entity,
            "strategy": self.strategy,
            "raw_response": self.raw_response
        }

    def __repr__(self) -> str:
        return (
            f"ParsedQuestion(active_objects={self.active_objects}, "
            f"question_type='{self.question_type}', is_relational={self.is_relational}, "
            f"target='{self.target_entity}', anchor='{self.anchor_entity}')"
        )


def parse_visual_query_structure(question: str, query_meta: Optional[Dict[str, Any]] = None) -> Tuple[str, str, str]:
    """
    Extracts the structured visual query semantics: (target_attribute_type, condition_val, target_object).
    E.g. 'What patterns are brown chairs?' -> ('pattern', 'brown', 'chair')
         'What materials are red cars?'   -> ('material', 'red', 'car')
         'What colors are round helmets?' -> ('color', 'round', 'helmet')
    """
    if query_meta:
        target_attr = query_meta.get("target_attribute_type", "")
        cond = query_meta.get("condition_attribute", {})
        cond_val = cond.get("value", "") if isinstance(cond, dict) else str(cond)
        target_obj = query_meta.get("target_object", "")
        if target_attr and target_obj:
            return target_attr.lower().strip(), cond_val.lower().strip(), target_obj.lower().strip()

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


def build_mllm_messages(question: str, enable_relational: bool = False) -> List[Dict[str, Any]]:
    """Builds chat completion message list for MLLM semantic question parsing."""
    if not enable_relational:
        return [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": (
                        "You are an expert visual search parser. Given a question, extract search targets and return a JSON object with:\n"
                        "- \"active_objects\": list of key visual search entities or text needed to answer the question\n"
                        "- \"target_entity\": the primary object being inspected, counted, or located (or null)\n"
                        "- \"question_type\": one of ['single_object_attribute', 'counting', 'text_reading', 'general']\n"
                        "Return ONLY valid JSON."
                    )
                }]
            },
            {"role": "user", "content": [{"type": "text", "text": "Question: \"What is the registration number of the boat?\""}]},
            {"role": "assistant", "content": [{"type": "text", "text": '{"active_objects": ["boat"], "target_entity": "boat", "question_type": "single_object_attribute"}'}]},
            {"role": "user", "content": [{"type": "text", "text": "Question: \"What is the color of the mailbox?\""}]},
            {"role": "assistant", "content": [{"type": "text", "text": '{"active_objects": ["mailbox"], "target_entity": "mailbox", "question_type": "single_object_attribute"}'}]},
            {"role": "user", "content": [{"type": "text", "text": "Question: \"How many windows are on the brick building?\""}]},
            {"role": "assistant", "content": [{"type": "text", "text": '{"active_objects": ["brick building", "window"], "target_entity": "window", "question_type": "counting"}'}]},
            {"role": "user", "content": [{"type": "text", "text": f"Question: \"{question}\""}]}
        ]

    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": (
                    "You are an expert visual search parser. Given a question, extract search targets and return a JSON object with:\n"
                    "- \"active_objects\": list of key visual search entities or text needed to answer the question\n"
                    "- \"target_entity\": the primary object being inspected, counted, or located (or null)\n"
                    "- \"anchor_entity\": the reference landmark object used for relative bearing/position (or null if none)\n"
                    "- \"question_type\": one of ['cross_object_relation', 'single_object_attribute', 'counting', 'text_reading', 'general']\n"
                    "- \"is_relational\": boolean (true for cross-object questions asking about spatial relationships or relative positioning between multiple objects; false for single-object questions asking about one object's attributes, text, color, or identity)\n"
                    "Return ONLY valid JSON."
                )
            }]
        },
        {"role": "user", "content": [{"type": "text", "text": "Question: \"Is the mailbox on the left or right side of the woman?\""}]},
        {"role": "assistant", "content": [{"type": "text", "text": '{"active_objects": ["mailbox", "woman"], "target_entity": "mailbox", "anchor_entity": "woman", "question_type": "cross_object_relation", "is_relational": true}'}]},
        {"role": "user", "content": [{"type": "text", "text": "Question: \"Where is the small stone cairn located relative to the waterfall?\""}]},
        {"role": "assistant", "content": [{"type": "text", "text": '{"active_objects": ["stone cairn", "waterfall"], "target_entity": "stone cairn", "anchor_entity": "waterfall", "question_type": "cross_object_relation", "is_relational": true}'}]},
        {"role": "user", "content": [{"type": "text", "text": "Question: \"What is the registration number of the boat?\""}]},
        {"role": "assistant", "content": [{"type": "text", "text": '{"active_objects": ["boat"], "target_entity": "boat", "anchor_entity": null, "question_type": "single_object_attribute", "is_relational": false}'}]},
        {"role": "user", "content": [{"type": "text", "text": "Question: \"What is the color of the mailbox?\""}]},
        {"role": "assistant", "content": [{"type": "text", "text": '{"active_objects": ["mailbox"], "target_entity": "mailbox", "anchor_entity": null, "question_type": "single_object_attribute", "is_relational": false}'}]},
        {"role": "user", "content": [{"type": "text", "text": "Question: \"How many windows are on the brick building?\""}]},
        {"role": "assistant", "content": [{"type": "text", "text": '{"active_objects": ["brick building", "window"], "target_entity": "window", "anchor_entity": "brick building", "question_type": "counting", "is_relational": false}'}]},
        {"role": "user", "content": [{"type": "text", "text": f"Question: \"{question}\""}]}
    ]


def build_few_shot_prompt(question: str, enable_relational: bool = False) -> str:
    """Builds single text prompt for text-completion VLMs."""
    if not enable_relational:
        return (
            "You are an expert visual search parser. Given a question, return a JSON object with 'active_objects', 'target_entity', and 'question_type'.\n\n"
            "Example 1 (Single-Object Attribute):\n"
            "Question: \"What is the registration number of the boat?\"\n"
            "JSON: {\"active_objects\": [\"boat\"], \"target_entity\": \"boat\", \"question_type\": \"single_object_attribute\"}\n\n"
            "Example 2 (Single-Object Attribute):\n"
            "Question: \"What is the color of the mailbox?\"\n"
            "JSON: {\"active_objects\": [\"mailbox\"], \"target_entity\": \"mailbox\", \"question_type\": \"single_object_attribute\"}\n\n"
            "Example 3 (Counting):\n"
            "Question: \"How many windows are on the brick building?\"\n"
            "JSON: {\"active_objects\": [\"brick building\", \"window\"], \"target_entity\": \"window\", \"question_type\": \"counting\"}\n\n"
            f"Question: \"{question}\"\nJSON:"
        )

    return (
        "You are an expert visual search parser. Given a question, return a JSON object with 'active_objects', 'target_entity', 'anchor_entity', 'question_type', and 'is_relational'.\n\n"
        "Example 1 (Cross-Object Relational):\n"
        "Question: \"Is the mailbox on the left or right side of the woman?\"\n"
        "JSON: {\"active_objects\": [\"mailbox\", \"woman\"], \"target_entity\": \"mailbox\", \"anchor_entity\": \"woman\", \"question_type\": \"cross_object_relation\", \"is_relational\": true}\n\n"
        "Example 2 (Cross-Object Relational):\n"
        "Question: \"Where is the small stone cairn located relative to the waterfall?\"\n"
        "JSON: {\"active_objects\": [\"stone cairn\", \"waterfall\"], \"target_entity\": \"stone cairn\", \"anchor_entity\": \"waterfall\", \"question_type\": \"cross_object_relation\", \"is_relational\": true}\n\n"
        "Example 3 (Single-Object Attribute):\n"
        "Question: \"What is the registration number of the boat?\"\n"
        "JSON: {\"active_objects\": [\"boat\"], \"target_entity\": \"boat\", \"anchor_entity\": null, \"question_type\": \"single_object_attribute\", \"is_relational\": false}\n\n"
        "Example 4 (Single-Object Attribute):\n"
        "Question: \"What is the color of the mailbox?\"\n"
        "JSON: {\"active_objects\": [\"mailbox\"], \"target_entity\": \"mailbox\", \"anchor_entity\": null, \"question_type\": \"single_object_attribute\", \"is_relational\": false}\n\n"
        f"Question: \"{question}\"\nJSON:"
    )

def parse_mllm_json_response(raw_text: str) -> Tuple[List[str], Optional[str], Optional[str], str, bool, str]:
    """
    Parses LLM response extracting active_objects, target_entity, anchor_entity,
    question_type, is_relational, and strategy name.
    """
    raw_targets: List[str] = []
    parsed_target: Optional[str] = None
    parsed_anchor: Optional[str] = None
    parsed_qtype = "general"
    parsed_relational = False
    strategy = ""

    # 1. Parse JSON dict object
    dict_match = re.search(r'\{[\s\S]*?\}', raw_text)
    if dict_match:
        try:
            data = json.loads(dict_match.group(0))
        except Exception:
            try:
                data = ast.literal_eval(dict_match.group(0))
            except Exception:
                data = None

        if isinstance(data, dict):
            objs = data.get("active_objects", [])
            if isinstance(objs, list) and objs:
                raw_targets = [str(o).strip().lower() for o in objs if str(o).strip()]

            parsed_target = data.get("target_entity")
            if parsed_target:
                parsed_target = str(parsed_target).strip().lower()

            parsed_anchor = data.get("anchor_entity")
            if parsed_anchor:
                parsed_anchor = str(parsed_anchor).strip().lower()

            parsed_qtype = str(data.get("question_type", "general")).strip().lower()
            parsed_relational = bool(data.get("is_relational", False))
            strategy = "Strategy 1: MLLM Unified Semantic Parser (Non-Fallback)"

    # 2. Fallback to bracket list [ ... ]
    if not raw_targets:
        json_match = re.search(r'\[.*?\]', raw_text, re.DOTALL)
        if json_match:
            try:
                parsed_cues = json.loads(json_match.group(0).strip())
            except Exception:
                try:
                    parsed_cues = ast.literal_eval(json_match.group(0).strip())
                except Exception:
                    parsed_cues = []
            if isinstance(parsed_cues, list):
                raw_targets = [str(c).strip().lower() for c in parsed_cues if str(c).strip()]
                if raw_targets:
                    strategy = "Strategy 1: MLLM Entity List Parser (Non-Fallback)"

    # 3. Fallback to bullet lines
    if not raw_targets:
        bullet_lines = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+\.)\s*([^\n]+)', raw_text)
        if bullet_lines:
            raw_targets = [b.strip().strip('"\'').lower() for b in bullet_lines if b.strip()]
            if raw_targets:
                strategy = "Strategy 1: MLLM Bullet Parser (Non-Fallback)"

    return raw_targets, parsed_target, parsed_anchor, parsed_qtype, parsed_relational, strategy


def extract_syntactic_clause_targets(question: str) -> List[str]:
    """Grammar clause-delimited noun extraction fallback (Strategy 3)."""
    delimiters = {
        'what', 'where', 'which', 'who', 'whom', 'whose', 'when', 'how', 'why',
        'above', 'below', 'under', 'next', 'near', 'behind', 'front', 'compared', 'top',
        'in', 'on', 'at', 'by', 'from', 'with', 'to', 'of', 'into', 'onto', 'around', 'between',
        'is', 'are', 'was', 'were', 'standing', 'displayed', 'shown', 'parked', 'located', 'positioned',
        'relative', 'position', 'distance', 'many'
    }
    abstract_heads = {
        'position', 'distance', 'color', 'symbol', 'text', 'word', 'image', 'picture', 'photo', 'kind', 'type'
    }

    q_clean = question.rstrip('?.,! ')
    raw_tokens = re.findall(r'\b[a-zA-Z0-9_\-]+\b', q_clean)

    phrases = []
    current_chunk = []

    for t in raw_tokens:
        tl = t.lower()
        if tl in {'the', 'a', 'an'}:
            if current_chunk:
                phrases.append(' '.join(current_chunk))
                current_chunk = []
            continue
        if tl in delimiters:
            if current_chunk:
                phrases.append(' '.join(current_chunk))
                current_chunk = []
            continue
        current_chunk.append(tl)

    if current_chunk:
        phrases.append(' '.join(current_chunk))

    raw_targets = []
    for p in phrases:
        words = p.strip().split()
        words = [w for w in words if w not in delimiters and w not in {'the', 'a', 'an'}]
        if not words:
            continue
        if len(words) == 1 and words[0] in abstract_heads:
            continue
        cand = ' '.join(words)
        if cand and cand not in raw_targets:
            raw_targets.append(cand)

    return raw_targets


def format_options(options_obj: Any, cycle_idx: int = 0) -> str:
    """
    Standardizes options from dictionary, cycle list, single list, or string format
    into a clean newline-delimited options string block (e.g. A. option1\nB. option2).
    """
    if not options_obj:
        return ""
    if isinstance(options_obj, dict):
        lines = [f"{k}. {v}" for k, v in sorted(options_obj.items())]
        return "\n".join(lines)
    elif isinstance(options_obj, list):
        if len(options_obj) > 0 and isinstance(options_obj[0], str) and ('\n' in options_obj[0] or options_obj[0].strip().startswith("A.") or options_obj[0].strip().startswith("A:")):
            idx = cycle_idx if (0 <= cycle_idx < len(options_obj)) else 0
            return options_obj[idx].strip()
        letters = ["A", "B", "C", "D", "E", "F"]
        lines = []
        for i, opt in enumerate(options_obj):
            prefix = f"{letters[i]}. " if i < len(letters) else f"{i+1}. "
            opt_clean = str(opt).strip()
            if re.match(r"^[\(\[]?[A-Fa-f][\)\.:]\s*", opt_clean):
                lines.append(opt_clean)
            else:
                lines.append(f"{prefix}{opt_clean}")
        return "\n".join(lines)
    elif isinstance(options_obj, str):
        return options_obj.strip()
    return ""


def format_question_with_options(question: str, options: Any = None, cycle_idx: int = 0) -> str:
    """
    Formats question with candidate options for multiple-choice prompting,
    matching CVSearch format_question_new / format_question standard.
    """
    if not options:
        return question

    opt_str = format_options(options, cycle_idx=cycle_idx)
    if not opt_str:
        return question

    return (
        f"{question}\n\n"
        f"Options:\n{opt_str}\n\n"
        f"Select the best answer to the above multiple-choice question based on the image. "
        f"Respond with the option letter and concise answer text."
    )
