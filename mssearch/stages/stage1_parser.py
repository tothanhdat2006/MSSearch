"""
Stage 1: Question Parsing & Active Object Set Extraction.
Extracts all key visual anchor entities from question q into A_q = [e_1, e_2, ...].
"""

import re
import json
import ast
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict, Any
from PIL import Image
from ..config import MSSearchConfig
from ..models.utils import include_pronouns, normalize_target_text, extract_visual_objects


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
        target_element: str = ""
    ):
        super().__init__(active_objects)
        self.active_objects = list(active_objects)
        self.question_type = str(question_type)
        self.is_relational = bool(is_relational)
        self.strategy = str(strategy)
        self.raw_response = str(raw_response)
        self.anchors = list(anchors) if anchors else []
        self.target_element = str(target_element)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_objects": self.active_objects,
            "question_type": self.question_type,
            "is_relational": self.is_relational,
            "strategy": self.strategy,
            "raw_response": self.raw_response
        }

    def __repr__(self) -> str:
        return (
            f"ParsedQuestion(active_objects={self.active_objects}, "
            f"question_type='{self.question_type}', is_relational={self.is_relational})"
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

    # Deterministic regex parser on question
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
        # Singularize common plural objects
        obj_clean = re.sub(r"s$", "", obj_raw) if len(obj_raw) > 3 and not obj_raw.endswith("ss") else obj_raw
        return attr_raw, cond_raw, obj_clean

    return "", "", ""


def parse_question_to_active_objects(
    question: str,
    zoom_model=None,
    nlp_model=None,
    ic_examples: Optional[dict] = None,
    verbose: bool = False,
    return_strategy: bool = False,
    vlm_model=None,
    image_pil: Optional[Union[Image.Image, List[Image.Image]]] = None,
    config: Optional[MSSearchConfig] = None
) -> Union[ParsedQuestion, Tuple[ParsedQuestion, str]]:
    """
    Parses a natural language question into a structured ParsedQuestion object.
    Leverages the LLM backbone of the VLM to perform unified semantic reasoning:
    - Extracts active search objects
    - Classifies question type ('relational_reading', 'spatial_orientation', 'counting', etc.)
    - Determines relational intent (is_relational: True/False)

    Returns ParsedQuestion (subclasses list, duck-typing List[str] for 100% backward compatibility).
    """
    if zoom_model is None and vlm_model is not None:
        zoom_model = vlm_model

    if zoom_model is None:
        raise RuntimeError(
            "[Stage 1 Error] zoom_model (Qwen2.5-VL) is None. Question Parsing requires a loaded VLM model for Strategy 1/2 semantic question parsing; fallback execution is strictly disabled."
        )
    parsed_question_type = "general"
    parsed_is_relational = False
    raw_targets = []
    raw_text = ""
    extraction_strategy = "Strategy 3: Syntactic Clause Extraction (Fallback)"

    if verbose:
        model_name = type(zoom_model).__name__ if zoom_model is not None else "None"
        print(f"[Stage 1 Diagnostics] Model passed: {model_name}")
        if zoom_model is not None:
            has_gt = hasattr(zoom_model, "generate_text")
            has_gr = hasattr(zoom_model, "generate_response")
            print(f"  -> hasattr(generate_text): {has_gt} | hasattr(generate_response): {has_gr}")
            if not has_gt and not has_gr:
                print(f"  -> [Warning] Model '{model_name}' lacks text generation methods. Available: {[m for m in dir(zoom_model) if not m.startswith('_')]}")
        else:
            print("  -> [Notice] No VLM/LLM model provided (zoom_model=None). Skipping Strategy 1 & 2.")

    # Strategy 1: Direct MLLM In-Context Semantic Parser (Preferred - uses LLM backbone)
    if zoom_model is not None and (hasattr(zoom_model, "generate_text") or hasattr(zoom_model, "generate_response")):
        if verbose:
            print("[Stage 1][Strategy 1] Initiating MLLM in-context unified semantic parsing (Active Objects, Question Type, Relational Intent)...")
        try:
            mllm_messages = [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": (
                            "You are an expert visual search parser. Given a question, extract search targets and return a JSON object with:\n"
                            "- \"active_objects\": list of key visual search entities or text needed to answer the question\n"
                            "- \"question_type\": one of ['cross_object_relation', 'single_object_attribute', 'counting', 'text_reading', 'general']\n"
                            "- \"is_relational\": boolean (true for cross-object questions asking about spatial relationships or relative positioning between multiple objects; false for single-object questions asking about one object's attributes, text, color, or identity)\n"
                            "Return ONLY valid JSON."
                        )
                    }]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Question: \"What is the text written above the logo on the jacket's sleeve?\""}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '{"active_objects": ["jacket", "sleeve", "logo", "text"], "question_type": "cross_object_relation", "is_relational": true}'}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Question: \"What is the registration number of the boat?\""}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '{"active_objects": ["boat"], "question_type": "single_object_attribute", "is_relational": false}'}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Question: \"What is the color of the mailbox?\""}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '{"active_objects": ["mailbox"], "question_type": "single_object_attribute", "is_relational": false}'}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Question: \"How many windows are on the brick building?\""}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '{"active_objects": ["brick building", "window"], "question_type": "counting", "is_relational": false}'}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Question: \"{question}\""}]
                }
            ]

            few_shot_prompt = (
                "You are an expert visual search parser. Given a question, return a JSON object with 'active_objects', 'question_type', and 'is_relational'.\n\n"
                "Example 1 (Cross-Object):\n"
                "Question: \"What is the text written above the logo on the jacket's sleeve?\"\n"
                "JSON: {\"active_objects\": [\"jacket\", \"sleeve\", \"logo\", \"text\"], \"question_type\": \"cross_object_relation\", \"is_relational\": true}\n\n"
                "Example 2 (Single-Object):\n"
                "Question: \"What is the registration number of the boat?\"\n"
                "JSON: {\"active_objects\": [\"boat\"], \"question_type\": \"single_object_attribute\", \"is_relational\": false}\n\n"
                "Example 3 (Single-Object):\n"
                "Question: \"What is the color of the mailbox?\"\n"
                "JSON: {\"active_objects\": [\"mailbox\"], \"question_type\": \"single_object_attribute\", \"is_relational\": false}\n\n"
                f"Question: \"{question}\"\nJSON:"
            )

            raw_text = ""
            if hasattr(zoom_model, "generate_text"):
                try:
                    raw_text = zoom_model.generate_text(mllm_messages, max_new_tokens=96)
                except Exception:
                    raw_text = zoom_model.generate_text(few_shot_prompt, max_new_tokens=96)
            elif hasattr(zoom_model, "generate_response"):
                raw_text = zoom_model.generate_response(image_pil=None, question=few_shot_prompt, max_new_tokens=96)

            if verbose:
                print(f"[Stage 1][Strategy 1] Raw MLLM output: {repr(raw_text)}")

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

                    parsed_question_type = str(data.get("question_type", "general")).strip().lower()
                    parsed_is_relational = bool(data.get("is_relational", False))
                    extraction_strategy = "Strategy 1: MLLM Unified Semantic Parser (Non-Fallback)"

            # 2. Fallback to bracket list [ ... ] if dict not matched
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
                            extraction_strategy = "Strategy 1: MLLM Entity List Parser (Non-Fallback)"

            # 3. Fallback to bullet lines
            if not raw_targets:
                bullet_lines = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+\.)\s*([^\n]+)', raw_text)
                if bullet_lines:
                    raw_targets = [b.strip().strip('"\'').lower() for b in bullet_lines if b.strip()]
                    if raw_targets:
                        extraction_strategy = "Strategy 1: MLLM Bullet Parser (Non-Fallback)"

        except Exception as e:
            if verbose:
                import traceback
                print(f"[Stage 1][Fallback Trigger] MLLM text extraction raised {type(e).__name__}: {e}")
                traceback.print_exc()

    # Strategy 2: Use VLM In-Context Cue Generator if available and targets not yet found
    if not raw_targets and zoom_model is not None and ic_examples is not None:
        try:
            generated_cues = zoom_model.generate_visual_cues_using_ic(ic_examples, question)
            if generated_cues:
                raw_targets.extend(generated_cues)
                extraction_strategy = "Strategy 2: VLM In-Context Generation (Non-Fallback)"
                if verbose:
                    print(f"[Stage 1][Non-Fallback] In-context VLM generated cues: {generated_cues}")
        except Exception as e:
            if verbose:
                print(f"[Stage 1][Fallback Trigger] In-context VLM generation failed ({e}); falling back to grammar regex.")

    # Strategy 3: Robust Syntactic Clause-Delimited Noun Extraction (Fallback)
    if not raw_targets:
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

        if raw_targets:
            extraction_strategy = "Strategy 3: Syntactic Clause Extraction (Fallback)"
            if verbose:
                print(f"[Stage 1][Fallback] Syntactic clause extraction found entities: {raw_targets}")

    # Post-process: Remove pronouns and normalize plurals
    clean_targets = []
    for t in raw_targets:
        if not t:
            continue
        if nlp_model is not None and include_pronouns(nlp_model, t):
            if verbose:
                print(f"[Stage 1] Stripped pronoun target: '{t}'")
            continue
        norm_t, _ = normalize_target_text(t)
        norm_t = norm_t.strip().lower()
        if norm_t and norm_t not in clean_targets:
            clean_targets.append(norm_t)

    # Fallback to query structure parser if clean_targets is empty or default
    if not clean_targets or clean_targets == ["object"]:
        _, _, str_obj = parse_visual_query_structure(question)
        if str_obj:
            clean_targets = [str_obj]
            extraction_strategy = "Strategy 1c: Question Structure Syntax Parser"

    if not extraction_strategy.startswith("Strategy 1"):
        parsed_is_relational = False
        q_lower = question.lower()
        if any(w in q_lower for w in ["how many", "count", "number of"]):
            parsed_question_type = "counting"
        elif any(w in q_lower for w in ["what color", "colour", "shape", "appearance"]):
            parsed_question_type = "fine_grained_attribute"
        else:
            parsed_question_type = "general"

    # Strict invariant: single target is definitionally non-relational
    if len(clean_targets) <= 1:
        parsed_is_relational = False

    parsed_question = ParsedQuestion(
        active_objects=clean_targets,
        question_type=parsed_question_type,
        is_relational=parsed_is_relational,
        strategy=extraction_strategy,
        raw_response=raw_text
    )

    if verbose:
        print(f"[Stage 1] Question Parsing Complete via {extraction_strategy}:")
        print(f"  • Question Type  : {parsed_question.question_type}")
        print(f"  • Relational Flag: {parsed_question.is_relational}")
        print(f"  • Active Targets : {parsed_question.active_objects}")

    if return_strategy:
        return parsed_question, extraction_strategy
    return parsed_question


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
    If options is None or empty, returns original question unchanged.
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

