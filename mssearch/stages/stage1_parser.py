"""
Stage 1: Question Parsing & Active Object Set Extraction.
Extracts all key visual anchor entities from question q into A_q = [e_1, e_2, ...].
"""

import re
from typing import List, Optional, Tuple, Union
from ..models.utils import include_pronouns, normalize_target_text, extract_visual_objects


RELATIONAL_KEYWORDS = {
    "left", "right", "above", "below", "behind", "in front of",
    "between", "next to", "facing", "direction", "relative to",
    "orient", "orientation", "towards", "closer", "further",
    "north", "south", "east", "west", "beside", "adjacent"
}

ABSTRACT_ATTRIBUTE_HEADS = {
    "text", "number", "digit", "word", "letter", "symbol", "logo", "sign",
    "color", "colour", "shape", "appearance", "writing", "label"
}


def detect_relational_intent(question: str) -> bool:
    """Detects whether a visual question queries spatial relationships or orientation."""
    q_lower = question.lower()
    return any(re.search(rf"\b{kw}\b", q_lower) for kw in RELATIONAL_KEYWORDS)


def reorder_anchor_first(targets: List[str]) -> List[str]:
    """
    Prioritizes physical host entities and landmarks ahead of abstract attributes/sub-elements.
    Example: ['text', 'jacket', 'sleeve'] -> ['jacket', 'sleeve', 'text']
    """
    anchors = []
    attributes = []
    for t in targets:
        words = t.lower().split()
        if len(words) == 1 and words[0] in ABSTRACT_ATTRIBUTE_HEADS:
            attributes.append(t)
        elif len(words) > 1 and words[0] in ABSTRACT_ATTRIBUTE_HEADS and words[-1] in ABSTRACT_ATTRIBUTE_HEADS:
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
    enable_anchor_first: bool = True
) -> Union[List[str], Tuple[List[str], str]]:
    """
    Parses a natural language question into a normalized list of target anchor objects.
    Enforces Anchor-First ordering: physical host entities precede abstract attributes.
    
    Example:
        "What is the text written above the logo on the jacket's sleeve?"
        -> ["jacket", "sleeve", "logo", "text"]
    """
    if zoom_model is None and vlm_model is not None:
        zoom_model = vlm_model

    raw_targets = []
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

    # Strategy 1: Direct MLLM In-Context Text Parser (Preferred - uses LLM backbone)
    if zoom_model is not None and (hasattr(zoom_model, "generate_text") or hasattr(zoom_model, "generate_response")):
        if verbose:
            print("[Stage 1][Strategy 1] Initiating MLLM in-context question parsing (Anchor-First)...")
        try:
            mllm_messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert visual search assistant. Given a visual search question, extract all key visual entities mentioned in the question. IMPORTANT: Always order the physical host objects and landmarks FIRST (e.g. jacket, entrance, building, car, person), and place abstract attributes/sub-elements (e.g. text, number, color, symbol, logo) LAST. Return ONLY a valid JSON list of lowercase strings."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Question: \"What is the relative position of the pink banner compared to the green trash bin?\"\nExtract visual search entities as a JSON list:"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '["pink banner", "green trash bin"]'}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Question: \"Tell me the symbol on the black board above the dog.\"\nExtract visual search entities as a JSON list:"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '["black board", "dog", "symbol"]'}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Question: \"What is the number displayed above the entrance where the woman is standing?\"\nExtract visual search entities as a JSON list:"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '["entrance", "woman", "number"]'}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Question: \"What is the text written above the logo on the jacket's sleeve?\"\nExtract visual search entities as a JSON list:"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": '["jacket", "sleeve", "logo", "text"]'}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Question: \"{question}\"\nExtract visual search entities as a JSON list:"}]
                }
            ]

            few_shot_prompt = (
                "You are an expert visual search assistant. Given a visual question, extract all key visual entities mentioned in the question. Always order physical host objects and landmarks FIRST, and abstract attributes (text, number, color, logo) LAST.\n\n"
                "Examples:\n"
                "Question: \"What is the relative position of the pink banner compared to the green trash bin?\"\n"
                "Entities: [\"pink banner\", \"green trash bin\"]\n\n"
                "Question: \"Tell me the symbol on the black board above the dog.\"\n"
                "Entities: [\"black board\", \"dog\", \"symbol\"]\n\n"
                "Question: \"What is the number displayed above the entrance where the woman is standing?\"\n"
                "Entities: [\"entrance\", \"woman\", \"number\"]\n\n"
                "Question: \"What is the text written above the logo on the jacket's sleeve?\"\n"
                "Entities: [\"jacket\", \"sleeve\", \"logo\", \"text\"]\n\n"
                f"Question: \"{question}\"\n"
                "Entities:"
            )

            raw_text = ""
            if hasattr(zoom_model, "generate_text"):
                try:
                    raw_text = zoom_model.generate_text(mllm_messages, max_new_tokens=64)
                except Exception:
                    # Fallback to string prompt if model does not support list of messages
                    raw_text = zoom_model.generate_text(few_shot_prompt, max_new_tokens=64)
            elif hasattr(zoom_model, "generate_response"):
                raw_text = zoom_model.generate_response(image_pil=None, question=few_shot_prompt)

            if verbose:
                print(f"[Stage 1][Strategy 1] Raw MLLM output: {repr(raw_text)}")

            parsed_cues = []
            import json
            import ast

            # 1. First attempt: JSON or Python AST extraction on [ ... ]
            json_match = re.search(r'\[.*?\]', raw_text, re.DOTALL)
            if json_match:
                block = json_match.group(0).strip()
                try:
                    parsed_cues = json.loads(block)
                except Exception as json_err:
                    try:
                        parsed_cues = ast.literal_eval(block)
                    except Exception as ast_err:
                        if verbose:
                            print(f"[Stage 1][Strategy 1 Debug] Bracket block parsing failed: JSON ({json_err}) | AST ({ast_err})")

            # 2. Second attempt: Markdown bullet points or numbered lists
            if not parsed_cues or not isinstance(parsed_cues, list):
                bullet_lines = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+\.)\s*([^\n]+)', raw_text)
                if bullet_lines:
                    parsed_cues = [b.strip().strip('"\'') for b in bullet_lines if b.strip()]

            # 3. Third attempt: Comma-separated list if quotes present
            if not parsed_cues or not isinstance(parsed_cues, list):
                quoted_tokens = re.findall(r'["\']([^"\']+)["\']', raw_text)
                if quoted_tokens and len(quoted_tokens) <= 5:
                    parsed_cues = quoted_tokens

            # Process candidates
            if isinstance(parsed_cues, list) and len(parsed_cues) > 0:
                valid_items = [str(c).strip().lower() for c in parsed_cues if str(c).strip()]
                if valid_items:
                    raw_targets.extend(valid_items)
                    extraction_strategy = "Strategy 1: MLLM Zero-Shot Text Generation (Non-Fallback)"
                    if verbose:
                        print(f"[Stage 1][Non-Fallback] MLLM successfully extracted anchor objects: {raw_targets}")
                elif verbose:
                    print(f"[Stage 1][Strategy 1 Warning] Parsed items were empty after cleaning: {parsed_cues}")
            elif verbose:
                print(f"[Stage 1][Strategy 1 Warning] Could not extract valid entity list from raw MLLM output: {repr(raw_text)}")

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
    else:
        if enable_anchor_first:
            clean_targets = reorder_anchor_first(clean_targets)
        if verbose:
            print(f"[Stage 1] Successfully parsed question into Active Object Set A_q: {clean_targets} via {extraction_strategy}")

    if return_strategy:
        return clean_targets, extraction_strategy
    return clean_targets


# Backward-compatible alias
parse_question_for_active_objects = parse_question_to_active_objects

