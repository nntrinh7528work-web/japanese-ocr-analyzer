"""Post-OCR Japanese checking layer."""

from __future__ import annotations

import re

from modules.japanese_nlp_helper import tokenize_japanese, pos_tag_japanese


def _has_japanese_chars(text: str) -> bool:
    """Check if text contains Hiragana, Katakana, or Kanji."""
    # Hiragana: 3040-309F, Katakana: 30A0-30FF, Kanji: 4E00-9FAF
    return bool(re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]', text))


def analyze_post_ocr_japanese(text: str) -> dict:
    """Analyze post-OCR text for Japanese NLP features.
    
    Args:
        text: The text extracted from OCR.
        
    Returns:
        A dictionary containing tokens, POS tags, boolean flag for Japanese presence,
        and the total token count.
    """
    if not text or not text.strip():
        return {
            "tokens": [],
            "pos_tags": [],
            "contains_japanese": False,
            "token_count": 0,
        }
        
    tokens = tokenize_japanese(text)
    pos_tags = pos_tag_japanese(text)
    
    return {
        "tokens": tokens,
        "pos_tags": pos_tags,
        "contains_japanese": _has_japanese_chars(text),
        "token_count": len(tokens),
    }


def conditional_japanese_check(text: str, analysis_language: str) -> dict | None:
    """Run Japanese NLP check only if analysis_language is 'japanese'."""
    if analysis_language == "japanese":
        return analyze_post_ocr_japanese(text)
    return None

