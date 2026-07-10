"""AI analysis pipeline: DeepSeek analysis + Gemini review + fix application."""

from __future__ import annotations

import copy
from typing import Any

from modules.deepseek_analyzer import analyze_with_deepseek
from modules.gemini_reviewer import review_deepseek_analysis

_VALID_CATEGORIES = {"vocabulary", "grammar", "kanji", "ocr_corrections"}


def apply_review_fixes(
    deepseek_analysis: dict,
    review: dict,
) -> tuple[dict, list[str]]:
    """Apply Gemini review fixes to a deep copy of the DeepSeek analysis.

    Args:
        deepseek_analysis: The original DeepSeek analysis dict (not mutated).
        review: The Gemini review dict with ``issues`` and ``missing_items``.

    Returns:
        A tuple of ``(corrected_analysis, warnings)``.
        ``missing_items`` are kept in the review only, not inserted into the analysis.
    """
    corrected = copy.deepcopy(deepseek_analysis)
    warnings: list[str] = []

    issues = review.get("issues", [])
    if not issues:
        return corrected, warnings

    # Group issues by category for index-safe removal.
    removes_by_category: dict[str, list[int]] = {}
    replaces: list[dict[str, Any]] = []

    for issue in issues:
        item_id = issue.get("item_id", "")
        parts = item_id.split(":")
        if len(parts) != 2:
            warnings.append(f"item_id không hợp lệ: '{item_id}'")
            continue

        category, index_str = parts
        if category not in _VALID_CATEGORIES:
            warnings.append(f"category không hợp lệ: '{category}' trong item_id '{item_id}'")
            continue

        try:
            index = int(index_str)
        except ValueError:
            warnings.append(f"index không phải số: '{index_str}' trong item_id '{item_id}'")
            continue

        items_list = corrected.get(category, [])
        if index < 0 or index >= len(items_list):
            warnings.append(f"index {index} ngoài phạm vi cho '{category}' (len={len(items_list)})")
            continue

        action = issue.get("action", "")
        if action == "remove":
            removes_by_category.setdefault(category, []).append(index)
        elif action == "replace":
            field = issue.get("field", "")
            if not field:
                warnings.append(f"issue thiếu field cho replace: item_id '{item_id}'")
                continue
            if field not in items_list[index]:
                warnings.append(f"field '{field}' không tồn tại trong {category}[{index}]")
                continue
            replaces.append({
                "category": category,
                "index": index,
                "field": field,
                "correct_value": issue.get("correct_value"),
            })
        else:
            warnings.append(f"action không hợp lệ: '{action}' cho item_id '{item_id}'")

    # Apply replace actions first.
    for rep in replaces:
        corrected[rep["category"]][rep["index"]][rep["field"]] = rep["correct_value"]

    # Apply remove actions in descending index order per category.
    for category, indices in removes_by_category.items():
        for index in sorted(set(indices), reverse=True):
            del corrected[category][index]

    return corrected, warnings


def run_verified_analysis(
    source_text: str,
    language: str,
) -> dict:
    """Run the full 2-AI pipeline: DeepSeek analysis → Gemini review → apply fixes.

    Args:
        source_text: The OCR text to analyze.
        language: ``"ja"`` or ``"en"``.

    Returns:
        A dict with keys: ``analysis``, ``review``, ``quality_status``, ``warnings``.
    """
    # Step 1: DeepSeek analysis.
    deepseek_result = analyze_with_deepseek(source_text, language)

    # Step 2: Gemini review.
    review = review_deepseek_analysis(source_text, deepseek_result, language)

    # Step 3: Apply fixes based on review verdict.
    if review.get("verdict") == "review_unavailable":
        return {
            "analysis": deepseek_result,
            "review": review,
            "quality_status": "review_unavailable",
            "warnings": [],
        }

    corrected, warnings = apply_review_fixes(deepseek_result, review)

    # Determine quality status.
    has_valid_changes = corrected != deepseek_result
    quality_status = "corrected" if has_valid_changes else "verified"

    return {
        "analysis": corrected,
        "review": review,
        "quality_status": quality_status,
        "warnings": warnings,
    }


def adapt_for_ui(
    deepseek_analysis: dict,
    source_text: str,
    analysis_language: str,
) -> dict:
    """Convert DeepSeek analysis JSON to the format expected by the app UI renderer.

    The existing UI expects keys like ``confirmed_text``, ``summary``,
    ``vocabulary_all``, ``vocabulary_important``, ``grammar_points``,
    ``kanji_analysis``, ``connectors``, ``full_markdown``, ``usage``, etc.

    This adapter bridges the DeepSeek schema to that format without modifying
    the renderer code.
    """
    lang = deepseek_analysis.get("language", "ja")
    result_language = "japanese" if lang == "ja" else "english"

    # --- vocabulary_all: flat table rows ---
    vocab_all = []
    for i, v in enumerate(deepseek_analysis.get("vocabulary", []), 1):
        if result_language == "japanese":
            vocab_all.append({
                "num": str(i),
                "word": v.get("term", ""),
                "reading": v.get("reading_or_ipa", "") or "",
                "type": v.get("part_of_speech", ""),
                "meaning": v.get("meaning_vi", ""),
                "jlpt": v.get("level", "") or "",
                "example": v.get("example_from_text", ""),
            })
        else:
            vocab_all.append({
                "num": str(i),
                "word": v.get("term", ""),
                "base_form": v.get("term", ""),
                "part_of_speech": v.get("part_of_speech", ""),
                "meaning": v.get("meaning_vi", ""),
                "cefr": v.get("level", "") or "",
                "example": v.get("example_from_text", ""),
            })

    # --- vocabulary_important: detailed expanders ---
    vocab_important = []
    for v in deepseek_analysis.get("vocabulary", []):
        vocab_important.append({
            "word": v.get("term", ""),
            "reading": v.get("reading_or_ipa", "") or "",
            "type": v.get("part_of_speech", ""),
            "meaning": v.get("meaning_vi", ""),
            "jlpt": v.get("level", "") or "",
            "cefr": v.get("level", "") or "",
            "example_text": v.get("example_from_text", ""),
            "example_1": v.get("example_1", ""),
            "example_2": v.get("example_2", ""),
            "related": v.get("related_words", ""),
            "mistake": v.get("common_mistake", ""),
            "note": v.get("note", ""),
            "difficulty": v.get("level", "") or "",
        })

    # --- grammar_points ---
    grammar_points = []
    for g in deepseek_analysis.get("grammar", []):
        grammar_points.append({
            "name": g.get("name", ""),
            "structure": g.get("structure", ""),
            "meaning": g.get("meaning_vi", ""),
            "usage": g.get("usage_vi", ""),
            "level": g.get("level", "") or "",
            "example": g.get("example_from_text", ""),
            "example_analysis": g.get("example_analysis", ""),
            "example_1": g.get("example_1", ""),
            "example_2": g.get("example_2", ""),
            "mistake": g.get("common_mistake", ""),
            "note": g.get("note", ""),
        })

    # --- kanji_analysis ---
    kanji_analysis = []
    for k in deepseek_analysis.get("kanji", []):
        kanji_analysis.append({
            "kanji": k.get("kanji", ""),
            "reading": k.get("reading", ""),
            "meaning": k.get("meaning_vi", ""),
            "jlpt": k.get("level", "") or "",
            "vocab": k.get("word_from_text", ""),
        })

    # --- Build full_markdown for export ---
    md_parts = [f"# Phân tích văn bản\n\n## Tóm tắt\n{deepseek_analysis.get('summary_vi', '')}"]

    ocr_corrections = deepseek_analysis.get("ocr_corrections", [])
    if ocr_corrections:
        md_parts.append("\n## Sửa lỗi OCR")
        for c in ocr_corrections:
            md_parts.append(f"- {c.get('original', '')} → {c.get('corrected', '')} ({c.get('reason_vi', '')})")

    if vocab_all:
        md_parts.append("\n## Từ vựng")
        for v in deepseek_analysis.get("vocabulary", []):
            md_parts.append(f"- **{v.get('term', '')}** ({v.get('reading_or_ipa', '') or ''}): {v.get('meaning_vi', '')}")

    if grammar_points:
        md_parts.append("\n## Ngữ pháp")
        for g in deepseek_analysis.get("grammar", []):
            md_parts.append(f"- **{g.get('name', '')}** `{g.get('structure', '')}`: {g.get('meaning_vi', '')}")

    if kanji_analysis:
        md_parts.append("\n## Kanji")
        for k in deepseek_analysis.get("kanji", []):
            md_parts.append(f"- **{k.get('kanji', '')}** ({k.get('reading', '')}): {k.get('meaning_vi', '')}")

    full_markdown = "\n".join(md_parts)

    return {
        "analysis_language": analysis_language,
        "confirmed_text": source_text,
        "summary": deepseek_analysis.get("summary_vi", ""),
        "vocabulary_all": vocab_all,
        "vocabulary_important": vocab_important,
        "grammar_points": grammar_points,
        "kanji_analysis": kanji_analysis,
        "connectors": [],
        "discourse_markers": [],
        "phrasal_collocations": [],
        "sentence_patterns": [],
        "ocr_corrections": ocr_corrections,
        "full_markdown": full_markdown,
        "section_markdown": {},
        "usage": {"input_tokens": 0, "output_tokens": 0, "candidate_tokens": 0, "thinking_tokens": 0},
    }
