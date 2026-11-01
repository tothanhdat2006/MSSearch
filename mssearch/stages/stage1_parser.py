"""
Stage 1: Question Parsing & Active Object Set Extraction.
Extracts all key visual anchor entities from question q into A_q = [e_1, e_2, ...].
"""

from typing import List, Optional, Tuple, Union, Any
from PIL import Image
from ..config import MSSearchConfig
from ..models.utils import include_pronouns, normalize_target_text
from .utils_stage1 import (
    ParsedQuestion,
    parse_visual_query_structure,
    format_options,
    format_question_with_options,
    build_mllm_messages,
    build_few_shot_prompt,
    parse_mllm_json_response,
    extract_syntactic_clause_targets,
)

__all__ = [
    "ParsedQuestion",
    "parse_visual_query_structure",
    "parse_question_to_active_objects",
    "format_options",
    "format_question_with_options",
]


def parse_question_to_active_objects(
    question: str,
    zoom_model=None,
    nlp_model=None,
    ic_examples: Optional[dict] = None,
    verbose: bool = False,
    return_strategy: bool = False,
    vlm_model=None,
    image_pil: Optional[Union[Image.Image, List[Image.Image]]] = None,
    config: Optional[MSSearchConfig] = None,
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
    parsed_target_entity = None
    parsed_anchor_entity = None
    raw_targets = []
    raw_text = ""
    extraction_strategy = "Strategy 3: Syntactic Clause Extraction (Fallback)"

    # Strategy 1: Direct MLLM In-Context Semantic Parser (Preferred - uses LLM backbone)
    if zoom_model is not None and (hasattr(zoom_model, "generate_text") or hasattr(zoom_model, "generate_response")):
        if verbose:
            print("[Stage 1][Strategy 1] Initiating MLLM in-context unified semantic parsing...")
        try:
            mllm_messages = build_mllm_messages(question)
            few_shot_prompt = build_few_shot_prompt(question)

            if hasattr(zoom_model, "generate_text"):
                try:
                    raw_text = zoom_model.generate_text(mllm_messages, max_new_tokens=96)
                except Exception:
                    raw_text = zoom_model.generate_text(few_shot_prompt, max_new_tokens=96)
            elif hasattr(zoom_model, "generate_response"):
                raw_text = zoom_model.generate_response(image_pil=None, question=few_shot_prompt, max_new_tokens=96)

            if verbose:
                print(f"[Stage 1][Strategy 1] Raw MLLM output: {repr(raw_text)}")

            (
                raw_targets,
                parsed_target_entity,
                parsed_anchor_entity,
                parsed_question_type,
                parsed_is_relational,
                strategy_name,
            ) = parse_mllm_json_response(raw_text)
            if strategy_name:
                extraction_strategy = strategy_name

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
        raw_targets = extract_syntactic_clause_targets(question)
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
        parsed_anchor_entity = None

    if not parsed_target_entity and clean_targets:
        parsed_target_entity = clean_targets[0]
    if not parsed_anchor_entity and len(clean_targets) > 1 and parsed_is_relational:
        parsed_anchor_entity = clean_targets[1]

    parsed_question = ParsedQuestion(
        active_objects=clean_targets,
        question_type=parsed_question_type,
        is_relational=parsed_is_relational,
        strategy=extraction_strategy,
        raw_response=raw_text,
        target_entity=parsed_target_entity,
        anchor_entity=parsed_anchor_entity,
    )

    if verbose:
        print(f"[Stage 1] Question Parsing Complete via {extraction_strategy}:")
        print(f"  • Question Type  : {parsed_question.question_type}")
        print(f"  • Relational Flag: {parsed_question.is_relational}")
        print(f"  • Target Entity  : {parsed_question.target_entity}")
        print(f"  • Anchor Entity  : {parsed_question.anchor_entity}")
        print(f"  • Active Targets : {parsed_question.active_objects}")

    if return_strategy:
        return parsed_question, extraction_strategy
    return parsed_question
