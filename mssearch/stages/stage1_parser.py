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


RELATIONAL_KEYWORDS = {
    "left", "right", "above", "below", "behind", "in front of",
    "between", "next to", "facing", "direction", "relative to",
    "orient", "orientation", "towards", "closer", "further",
    "north", "south", "east", "west", "beside", "adjacent"
}

ABSTRACT_ATTRIBUTE_HEADS = {
    "text", "number", "digit", "word", "letter", "symbol", "logo", "sign",
    "color", "colour", "shape", "appearance", "writing", "label", "written",
    "reading", "font", "character", "displayed", "shown", "printed", "marked",
    "engraved", "carved", "painted", "visible"
}


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
        anchors: Optional[List[str]] = None,
        target_element: str = "",
        question_type: str = "general",
        is_relational: bool = False,
        strategy: str = "",
        raw_response: str = ""
    ):
        super().__init__(active_objects)
        self.active_objects = list(active_objects)
        self.anchors = list(anchors) if anchors else []
        self.target_element = str(target_element)
        self.question_type = str(question_type)
        self.is_relational = bool(is_relational)
        self.strategy = str(strategy)
        self.raw_response = str(raw_response)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_objects": self.active_objects,
            "anchors": self.anchors,
            "target_element": self.target_element,
            "question_type": self.question_type,
            "is_relational": self.is_relational,
            "strategy": self.strategy,
            "raw_response": self.raw_response
        }

    def __repr__(self) -> str:
        return (
            f"ParsedQuestion(active_objects={self.active_objects}, "
            f"anchors={self.anchors}, target_element='{self.target_element}', "
            f"question_type='{self.question_type}', is_relational={self.is_relational})"
        )


def detect_relational_intent(question_or_parsed: Union[str, ParsedQuestion, Any]) -> bool:
    """
    Detects whether a visual question queries spatial relationships or orientation.
    If a ParsedQuestion (or object with .is_relational) is provided, uses its semantic flag.
    Otherwise falls back to keyword regex matching over RELATIONAL_KEYWORDS.
    """
    if hasattr(question_or_parsed, "is_relational"):
        return bool(question_or_parsed.is_relational)
    q_lower = str(question_or_parsed).lower()
    return any(re.search(rf"\b{kw}\b", q_lower) for kw in RELATIONAL_KEYWORDS)


def reorder_anchor_first(targets: List[str]) -> List[str]:
    """
    Prioritizes physical host entities and landmarks ahead of abstract attributes/sub-elements.
    Example: ['text written', 'jacket s sleeve', 'logo'] -> ['jacket s sleeve', 'text written', 'logo']
    """
    delimiters = {'the', 'a', 'an', 'on', 'in', 'of', 'at', 'above', 'below', 'is', 'are', 'to', 'where'}
    anchors = []
    attributes = []
    for t in targets:
        words = [w for w in t.lower().split() if w not in delimiters]
        concrete = [w for w in words if w not in ABSTRACT_ATTRIBUTE_HEADS]
        if not concrete:
            # All substantive words are abstract attributes (e.g. 'text', 'text written', 'number displayed')
            attributes.append(t)
        else:
            anchors.append(t)
    return anchors + attributes if anchors else targets


def parse_question_to_active_objects(
    question: str,
    zoom_model=None,
    nlp_model=None,
    ic_examples: Optional[dict] = None,
    verbose: bool = False,
    return_strategy: bool = False,
    vlm_model=None,
    enable_anchor_first: bool = True,
    image_pil: Optional[Union[Image.Image, List[Image.Image]]] = None,
    config: Optional[MSSearchConfig] = None
) -> Union[ParsedQuestion, Tuple[ParsedQuestion, str]]:
    """
    Parses a natural language question into a structured ParsedQuestion object.
    Leverages the LLM backbone of the VLM to perform unified semantic reasoning:
    - Extracts physical host anchors and spatial landmarks
    - Identifies target attributes / elements
    - Classifies question type ('relational_reading', 'spatial_orientation', 'counting', etc.)
    - Determines relational intent (is_relational: True/False)
    - Enforces Anchor-First ordering: physical host entities precede abstract attributes.

    Returns ParsedQuestion (subclasses list, duck-typing List[str] for 100% backward compatibility).
    """
    if zoom_model is None and vlm_model is not None:
        zoom_model = vlm_model

    raw_targets = []
    parsed_anchors = []
    parsed_target_element = ""
    parsed_question_type = "general"
    parsed_is_relational = False
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
            print("[Stage 1][Strategy 1] Initiating MLLM in-context unified semantic parsing (Anchors, Question Type, Relational Intent)...")
        try:
            mllm_messages = [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": (
                            "You are an expert visual search parser. Given a question, extract search targets and return a JSON object with:\n"
                            "- \"anchors\": physical host entities / spatial landmarks (outermost physical entities first)\n"
                            "- \"target_element\": specific micro-attribute, text, or symbol to read\n"
                            "- \"active_objects\": combined search entities prioritizing physical anchors first\n"
                            "- \"question_type\": one of ['relational_reading', 'spatial_orientation', 'counting', 'fine_grained_attribute', 'general']\n"
                            "- \"is_relational\": boolean (true if question requires relative spatial positioning or dual-scale context)\n"
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
                    "content": [{"type": "text", "text": '{"anchors": ["jacket", "sleeve"], "target_element": "text above logo", "active_objects": ["jacket", "sleeve", "logo", "text"], "question_type": "relational_reading", "is_relational": true}'}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Question: \"How many windows are on the brick building?\""}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '{"anchors": ["brick building"], "target_element": "windows", "active_objects": ["brick building", "window"], "question_type": "counting", "is_relational": false}'}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Question: \"{question}\""}]
                }
            ]

            few_shot_prompt = (
                "You are an expert visual search parser. Given a question, return a JSON object with 'anchors', 'target_element', 'active_objects', 'question_type', and 'is_relational'.\n\n"
                "Example 1:\n"
                "Question: \"What is the text written above the logo on the jacket's sleeve?\"\n"
                "JSON: {\"anchors\": [\"jacket\", \"sleeve\"], \"target_element\": \"text above logo\", \"active_objects\": [\"jacket\", \"sleeve\", \"logo\", \"text\"], \"question_type\": \"relational_reading\", \"is_relational\": true}\n\n"
                "Example 2:\n"
                "Question: \"How many windows are on the brick building?\"\n"
                "JSON: {\"anchors\": [\"brick building\"], \"target_element\": \"windows\", \"active_objects\": [\"brick building\", \"window\"], \"question_type\": \"counting\", \"is_relational\": false}\n\n"
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
                    parsed_anchors = [str(a).strip().lower() for a in data.get("anchors", []) if str(a).strip()]
                    parsed_target_element = str(data.get("target_element", "")).strip().lower()
                    objs = data.get("active_objects", [])
                    if isinstance(objs, list) and objs:
                        raw_targets = [str(o).strip().lower() for o in objs if str(o).strip()]
                    elif parsed_anchors:
                        raw_targets = list(parsed_anchors)
                        if parsed_target_element and parsed_target_element not in raw_targets:
                            raw_targets.append(parsed_target_element)

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

    # Strategy 1b: Image-Grounded Anchor Discovery (Hybrid Low-Res Vision Fallback)
    # Triggered when text-only parsing yields no concrete physical host anchors (e.g. empty or abstract words like 'text'/'word')
    enable_vision_anchors = getattr(config, "enable_image_grounded_anchors", True) if config else True
    has_concrete_anchor = bool(parsed_anchors and any(all(w not in ABSTRACT_ATTRIBUTE_HEADS for w in a.lower().split()) for a in parsed_anchors))

    if enable_vision_anchors and not has_concrete_anchor and image_pil is not None and zoom_model is not None and hasattr(zoom_model, "generate_response"):
        res = getattr(config, "image_grounded_anchor_resolution", 1024) if config else 1024
        target_img = image_pil[0] if isinstance(image_pil, (list, tuple)) else image_pil
        if target_img is not None:
            if max(target_img.width, target_img.height) > res:
                scale = res / float(max(target_img.width, target_img.height))
                lowres_img = target_img.resize(
                    (int(round(target_img.width * scale)), int(round(target_img.height * scale))),
                    Image.Resampling.LANCZOS
                )
            else:
                lowres_img = target_img

            if verbose:
                print(f"[Stage 1][Strategy 1b] No concrete physical anchor in text; invoking Image-Grounded Anchor Discovery ({lowres_img.width}x{lowres_img.height}px, max_res={res})...")

            prompt_vision = (
                f"Question: \"{question}\"\n"
                "Carefully inspect the provided image. Identify the concrete physical object(s) or host entities visible in the scene "
                "that contain, display, or are most relevant to answering the question (e.g. storefront sign, road sign, vehicle, building, clothing, counter, screen).\n"
                "Return ONLY a JSON object with:\n"
                "- \"anchors\": list of physical host entities visible in the scene (outermost physical entities first)\n"
                "- \"target_element\": specific micro-attribute or text to inspect\n"
                "- \"active_objects\": combined search entities prioritizing physical anchors first\n"
                "- \"question_type\": one of ['relational_reading', 'spatial_orientation', 'counting', 'fine_grained_attribute', 'general']\n"
                "- \"is_relational\": boolean (true if question requires relative spatial positioning or dual-scale context)\n"
                "Return ONLY valid JSON."
            )

            try:
                raw_vision_text = zoom_model.generate_response(image_pil=lowres_img, question=prompt_vision, max_new_tokens=96)
                dict_match = re.search(r'\{[\s\S]*?\}', raw_vision_text)
                if dict_match:
                    try:
                        v_data = json.loads(dict_match.group(0))
                    except Exception:
                        try:
                            v_data = ast.literal_eval(dict_match.group(0))
                        except Exception:
                            v_data = None

                    if isinstance(v_data, dict):
                        v_anchors = [str(a).strip().lower() for a in v_data.get("anchors", []) if str(a).strip()]
                        if v_anchors:
                            parsed_anchors = v_anchors
                            parsed_target_element = str(v_data.get("target_element", parsed_target_element)).strip().lower()
                            v_objs = v_data.get("active_objects", [])
                            if isinstance(v_objs, list) and v_objs:
                                raw_targets = [str(o).strip().lower() for o in v_objs if str(o).strip()]
                            else:
                                raw_targets = list(parsed_anchors) + ([parsed_target_element] if parsed_target_element else [])
                            parsed_question_type = str(v_data.get("question_type", parsed_question_type)).strip().lower()
                            parsed_is_relational = bool(v_data.get("is_relational", parsed_is_relational))
                            extraction_strategy = f"Strategy 1b: Image-Grounded Semantic Parser ({res}px Low-Res Vision Fallback)"
                            raw_text = raw_vision_text
                            if verbose:
                                print(f"[Stage 1][Image Grounding] Vision-discovered physical anchors: {parsed_anchors} | active_objects: {raw_targets}")
            except Exception as e:
                if verbose:
                    print(f"[Stage 1][Image Grounding Notice] Vision anchor discovery exception: {e}")

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
                print(f"[Stage 1][Fallback] Syntactic clause extraction found anchor objects: {raw_targets}")

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

    # Strategy 4: Ultimate Fallback if no targets extracted
    if not clean_targets:
        clean_targets = ["object"]
        extraction_strategy = "Strategy 4: Default Fallback ['object']"
        if verbose:
            print("[Stage 1][Ultimate Fallback] No targets extracted from question; defaulting to ['object'].")

    if enable_anchor_first:
        clean_targets = reorder_anchor_first(clean_targets)

    # Derive anchors, question_type, and relational intent if not set by Strategy 1
    if not parsed_anchors:
        parsed_anchors = [t for t in clean_targets if t not in ABSTRACT_ATTRIBUTE_HEADS]
    if not parsed_target_element and len(clean_targets) > len(parsed_anchors):
        parsed_target_element = clean_targets[-1]
    if extraction_strategy != "Strategy 1: MLLM Unified Semantic Parser (Non-Fallback)":
        parsed_is_relational = detect_relational_intent(question)
        q_lower = question.lower()
        if parsed_is_relational:
            parsed_question_type = "relational_reading" if any(w in q_lower for w in ["text", "word", "letter", "symbol", "number", "sign"]) else "spatial_orientation"
        elif any(w in q_lower for w in ["how many", "count", "number of"]):
            parsed_question_type = "counting"
        elif any(w in q_lower for w in ["what color", "colour", "shape", "appearance"]):
            parsed_question_type = "fine_grained_attribute"
        else:
            parsed_question_type = "general"

    parsed_question = ParsedQuestion(
        active_objects=clean_targets,
        anchors=parsed_anchors,
        target_element=parsed_target_element,
        question_type=parsed_question_type,
        is_relational=parsed_is_relational,
        strategy=extraction_strategy,
        raw_response=raw_text
    )

    if verbose:
        print(f"[Stage 1] Unified Question Parsing Complete via {extraction_strategy}:")
        print(f"  • Question Type  : {parsed_question.question_type}")
        print(f"  • Relational Flag: {parsed_question.is_relational}")
        print(f"  • Physical Anchor: {parsed_question.anchors}")
        print(f"  • Target Element : '{parsed_question.target_element}'")
        print(f"  • Active Targets : {parsed_question.active_objects}")

    if return_strategy:
        return parsed_question, extraction_strategy
    return parsed_question


# Backward-compatible alias
parse_question_for_active_objects = parse_question_to_active_objects


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

