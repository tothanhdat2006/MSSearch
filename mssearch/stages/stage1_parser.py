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

    # Strategy 1: Use VLM In-Context Cue Generator if available
    if zoom_model is not None and ic_examples is not None:
        try:
            generated_cues = zoom_model.generate_visual_cues_using_ic(ic_examples, question)
            if generated_cues:
                raw_targets.extend(generated_cues)
                extraction_strategy = "Strategy 1: VLM In-Context Generation (Non-Fallback)"
                if verbose:
                    print(f"[Stage 1][Non-Fallback] In-context VLM generated cues: {generated_cues}")
        except Exception as e:
            if verbose:
                print(f"[Stage 1][Fallback Trigger] In-context VLM generation failed ({e}); falling back to SpaCy NLP.")

    # Strategy 2: SpaCy NLP Noun-Chunk Fallback
    if not raw_targets and nlp_model is not None:
        try:
            spacy_targets = extract_visual_objects(nlp_model, question)
            if spacy_targets:
                raw_targets.extend(spacy_targets)
                extraction_strategy = "Strategy 2: SpaCy NLP Fallback"
                if verbose:
                    print(f"[Stage 1][Fallback] SpaCy NLP extracted anchor objects: {spacy_targets}")
        except Exception as e:
            if verbose:
                print(f"[Stage 1][Fallback Trigger] SpaCy NLP extraction failed ({e}); falling back to Regex.")

    # Strategy 3: Heuristic Regex Fallback if both models are absent
    if not raw_targets:
        matches = re.findall(r'\b(?:the|a|an)\s+([a-zA-Z0-9_\-]+)\b', question.lower())
        stopwords = {"image", "picture", "photo", "question", "color", "symbol", "word", "text"}
        raw_targets = [m for m in matches if m not in stopwords]
        if raw_targets:
            extraction_strategy = "Strategy 3: Regex Noun Extraction (Fallback)"
            if verbose:
                print(f"[Stage 1][Fallback] Regex extraction found anchor objects: {raw_targets}")

    # Post-process: Remove pronouns and normalize plurals
    clean_targets = []
    for t in raw_targets:
        if not t:
            continue
        # Filter out pronoun phrases like "his bag" if nlp_model is present
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

