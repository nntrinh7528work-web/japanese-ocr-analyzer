"""Helper module for Japanese NLP using nagisa."""

from __future__ import annotations

import nagisa


def tokenize_japanese(text: str) -> list[str]:
    """Tokenize Japanese text using nagisa.
    
    Args:
        text: The Japanese text to tokenize.
        
    Returns:
        A list of string tokens. Empty list if text is empty.
    """
    if not text or not text.strip():
        return []
    
    extracted = nagisa.tagging(text.strip())
    return extracted.words


def pos_tag_japanese(text: str) -> list[dict[str, str]]:
    """Tokenize and POS tag Japanese text using nagisa.
    
    Args:
        text: The Japanese text to analyze.
        
    Returns:
        A list of dictionaries, each containing 'surface' (the token)
        and 'pos' (part of speech tag). Empty list if text is empty.
    """
    if not text or not text.strip():
        return []
        
    extracted = nagisa.tagging(text.strip())
    result = []
    for word, pos in zip(extracted.words, extracted.postags):
        result.append({"surface": word, "pos": pos})
    return result
