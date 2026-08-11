"""Local language detection and mismatch rules for OCR documents."""

from __future__ import annotations

from collections import Counter

from modules.sentence_analyzer import detect_sentence_language


def detect_item_language(text: str, fallback: str = "unknown") -> tuple[str, str, str]:
    """Return a language only when an OCR item has enough readable text."""
    source = str(text or "").strip()
    if not source:
        return "unknown", "low", "no_ocr"
    fallback_language = fallback if fallback in {"japanese", "english"} else "english"
    return detect_sentence_language(source, fallback_language)


def refresh_document_languages(items: list[dict], document_language: str = "unknown") -> tuple[str, str]:
    """Annotate OCR items and select a primary language by readable text size.

    Japanese script wins over isolated Latin product names, while a short
    Japanese quote in an English page does not outweigh the surrounding text.
    """
    weights: Counter[str] = Counter()
    for item in items:
        text = str(item.get("edited_text") or "").strip()
        language, confidence, source = detect_item_language(text, document_language)
        item["detected_language"] = language
        item["language_confidence"] = confidence
        item["language_source"] = source
        if language in {"japanese", "english"}:
            weights[language] += max(1, len(text))

    if document_language in {"japanese", "english"}:
        primary = document_language
        source = "manual"
    elif weights:
        primary = "japanese" if weights["japanese"] >= weights["english"] else "english"
        source = "auto"
    else:
        primary = "unknown"
        source = "auto"

    for item in items:
        detected = item.get("detected_language", "unknown")
        if not str(item.get("edited_text") or "").strip():
            item["mismatch_status"] = "none"
        elif detected in {"unknown", primary} or item.get("language_override"):
            item["mismatch_status"] = "confirmed" if item.get("language_override") else "none"
        else:
            item["mismatch_status"] = "mismatch"
    return primary, source


def analysis_eligible_items(items: list[dict]) -> list[dict]:
    """Exclude a detected foreign page until the learner explicitly accepts it."""
    return [item for item in items if item.get("mismatch_status") != "mismatch"]
