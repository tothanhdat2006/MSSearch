"""
Stage 1: Question Parsing & Active Object Set Extraction.
Extracts all key visual anchor entities from question q into A_q = [e_1, e_2, ...].
"""

import re
from typing import List, Optional, Tuple, Union
from ..models.utils import include_pronouns, normalize_target_text, extract_visual_objects


def parse_question_to_active_objects(
    question: str,
    zoom_model=None,
    nlp_model=None,
    ic_examples: Optional[dict] = None,
    verbose: bool = False,
    return_strategy: bool = False,
    vlm_model=None
) -> Union[List[str], Tuple[List[str], str]]:
    """
    Parses a natural language question into a normalized list of target anchor objects.
    
    Example:
        "What symbol is shown on the blue sign next to the car?"
        -> ["car", "sign"]
    """
    if zoom_model is None and vlm_model is not None:
        zoom_model = vlm_model

    raw_targets = []
    extraction_strategy = "Strategy 3: Regex Noun Extraction (Fallback)"

    # Strategy 1: Direct MLLM Zero-Shot Text Parser (Preferred - uses LLM backbone)
    if zoom_model is not None and (hasattr(zoom_model, "generate_text") or hasattr(zoom_model, "generate_response")):
        try:
            mllm_prompt = (
                "Extract only the concrete physical visual anchor objects mentioned in this visual search question.\n"
                "Do NOT include abstract relation words (such as 'position', 'relative', 'distance', 'location'), question words, or options.\n"
                "Return ONLY a valid JSON list of strings, for example: [\"pink banner\", \"green trash bin\"].\n\n"
                f"Question: \"{question}\"\n"
                "Visual Anchor Objects JSON:"
            )
            raw_text = ""
            if hasattr(zoom_model, "generate_text"):
                raw_text = zoom_model.generate_text(mllm_prompt, max_new_tokens=64)
            elif hasattr(zoom_model, "generate_response"):
                raw_text = zoom_model.generate_response(image_pil=None, question=mllm_prompt)

            import json
            json_match = re.search(r'\[.*?\]', raw_text, re.DOTALL)
            if json_match:
                parsed_cues = json.loads(json_match.group(0))
                if isinstance(parsed_cues, list) and len(parsed_cues) > 0:
                    raw_targets.extend([str(c).strip().lower() for c in parsed_cues if str(c).strip()])
                    extraction_strategy = "Strategy 1: MLLM Zero-Shot Text Generation (Non-Fallback)"
                    if verbose:
                        print(f"[Stage 1][Non-Fallback] MLLM extracted anchor objects: {raw_targets}")
        except Exception as e:
            if verbose:
                print(f"[Stage 1][Fallback Trigger] MLLM text extraction notice ({e}); proceeding to fallback.")

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

    # Strategy 3: Grammar-Aware Multi-Word Regex Fallback (when models absent / simulation)
    if not raw_targets:
        q_lower = question.lower()
        cleaned_q = re.sub(
            r'^(?:what\s+is|where\s+is|which\s+is|how\s+many|can\s+you\s+find|find)\s+(?:the\s+)?(?:relative\s+position\s+of|position\s+of|distance\s+between|color\s+of|number\s+of)?\s*',
            '',
            q_lower
        )
        pattern = r'\b(?:the|a|an)\s+([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+){0,2})'
        raw_matches = re.findall(pattern, cleaned_q)
        
        stopwords = {"image", "picture", "photo", "question", "color", "symbol", "word", "text", "relative", "position", "distance"}
        for m in raw_matches:
            tokens = [w for w in m.split() if w not in stopwords]
            if tokens:
                cand = ' '.join(tokens)
                cand = re.sub(r'\s+(?:compared|to|of|in|on|at|near|next|and)$', '', cand).strip()
                if cand and cand not in raw_targets:
                    raw_targets.append(cand)
                    
        if raw_targets:
            extraction_strategy = "Strategy 3: Grammar-Aware Regex Extraction (Fallback)"
            if verbose:
                print(f"[Stage 1][Fallback] Grammar-aware regex extraction found anchor objects: {raw_targets}")

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
    elif verbose:
        print(f"[Stage 1] Successfully parsed question into Active Object Set A_q: {clean_targets} via {extraction_strategy}")

    if return_strategy:
        return clean_targets, extraction_strategy
    return clean_targets


# Backward-compatible alias
parse_question_for_active_objects = parse_question_to_active_objects

