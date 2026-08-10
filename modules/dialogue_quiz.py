"""Local, language-aware exercises generated from structured dialogue data."""

from __future__ import annotations

import random
import re
import unicodedata
from typing import Any


def normalize_quiz_answer(value: str, language: str = "") -> str:
    """Accept harmless differences in whitespace, case and punctuation."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    if "anh" in str(language).lower() or "english" in str(language).lower():
        text = text.casefold()
    return re.sub(r"[\s\.,!?！？。、「」'\"-]+", "", text)


def _is_english(result: dict[str, Any]) -> bool:
    return str(result.get("language_code", "")).lower() == "en" or "anh" in str(result.get("language", "")).lower()


def _learning_rows(result: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in result.get("learning_targets", []):
        if not isinstance(row, dict) or not str(row.get("term", "")).strip():
            continue
        rows.append(
            {
                "term": str(row.get("term", "")).strip(),
                "meaning": str(row.get("explanation_vi", "")).strip(),
                "type": str(row.get("type", "vocabulary")),
            }
        )
    if rows:
        return rows
    for kind, values in (("vocabulary", result.get("vocab_used", {})), ("grammar", result.get("grammar_used", {}))):
        for term, meaning in (values or {}).items():
            rows.append({"term": str(term), "meaning": str(meaning), "type": kind})
    return rows


def _distractors(correct: str, terms: list[str], seed: str) -> list[str]:
    candidates = [item for item in dict.fromkeys(terms) if item and item != correct]
    # The fallback is intentionally generic; it is labelled as an option, never as
    # a fabricated definition of the source language target.
    fallback = ["Một ý nghĩa khác trong ngữ cảnh", "Một cách dùng không phù hợp", "Một cấu trúc có chức năng khác"]
    candidates.extend(item for item in fallback if item != correct and item not in candidates)
    rng = random.Random(seed)
    selected = candidates[:]
    rng.shuffle(selected)
    options = [correct] + selected[:3]
    rng.shuffle(options)
    return options


def _fallback_chunks(text: str, english: bool) -> list[str]:
    if english:
        words = re.findall(r"[\w'-]+|[^\w\s]", text, flags=re.UNICODE)
        if len(words) < 4:
            return []
        size = max(1, min(4, len(words) // 3))
        return [" ".join(words[index : index + size]).replace(" ,", ",").replace(" .", ".") for index in range(0, len(words), size)][:6]
    clauses = [part.strip() for part in re.split(r"(?<=[、，])", text) if part.strip()]
    if len(clauses) >= 3:
        return clauses[:6]
    # A conservative fallback based on single-character particles.  Multi-character
    # endings cannot be mixed into a Python look-behind because it must be fixed width.
    parts = [part for part in re.split(r"(?<=[はがをにへとでのも])", text) if part]
    return [part.strip() for part in parts if part.strip()][:6]


def generate_pure_quiz(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Create cloze, meaning, reorder and translation exercises without API calls."""
    dialogue = [row for row in result.get("dialogue", []) if isinstance(row, dict)]
    targets = _learning_rows(result)
    is_english = _is_english(result)
    target_terms = [row["term"] for row in targets]
    meanings = [row["meaning"] for row in targets if row["meaning"]]
    cloze: list[dict[str, Any]] = []
    mcq: list[dict[str, Any]] = []
    reorder: list[dict[str, Any]] = []
    translate: list[dict[str, Any]] = []

    for index, turn in enumerate(dialogue):
        text = str(turn.get("text", ""))
        highlights = list(turn.get("highlights") or [])
        for term in highlights:
            if term and term in text:
                cloze.append(
                    {
                        "id": f"cloze_{index}_{term}", "sentence": text.replace(term, "[ _______ ]", 1),
                        "target_word": term, "text_vi": str(turn.get("text_vi", "")),
                        "options": _distractors(term, target_terms, f"cloze:{index}:{term}"),
                    }
                )
        chunks = [str(item).strip() for item in turn.get("practice_chunks", []) if str(item).strip()]
        if not 3 <= len(chunks) <= 6:
            chunks = _fallback_chunks(text, is_english)
        if 3 <= len(chunks) <= 6:
            shuffled = list(chunks)
            random.Random(f"reorder:{index}:{text}").shuffle(shuffled)
            reorder.append(
                {
                    "id": f"reorder_{index}", "original": text, "text_vi": str(turn.get("text_vi", "")),
                    "shuffled_tokens": shuffled, "language": result.get("language", ""),
                }
            )
        if turn.get("text_vi") and text:
            translate.append(
                {
                    "id": f"trans_{index}", "prompt_vi": str(turn["text_vi"]), "correct_text": text,
                    "text_hira": str(turn.get("text_hira", "")), "language": result.get("language", ""),
                }
            )

    for index, row in enumerate(targets):
        if row["meaning"]:
            question_type = "Từ vựng" if row["type"] == "vocabulary" else "Cấu trúc"
            mcq.append(
                {
                    "id": f"mcq_{index}", "question": f"{question_type} `{row['term']}` có ý nghĩa/cách dùng nào?",
                    "correct_answer": row["meaning"], "options": _distractors(row["meaning"], meanings, f"mcq:{index}"),
                }
            )
    return {"cloze": cloze[:8], "mcq": mcq[:8], "reorder": reorder[:5], "translate": translate[:5]}
