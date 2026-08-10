"""Normalisation and local quality checks for dialogue practice version 2."""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from typing import Any


DIALOGUE_VERSION = "2.0"


def language_code(language: str) -> str:
    value = str(language or "").lower()
    return "en" if "english" in value or "tiếng anh" in value or value == "en" else "ja"


def parse_json_response(text: str) -> dict[str, Any]:
    """Extract an object even when a model unnecessarily wraps it in Markdown."""
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Gemini không trả về JSON hội thoại hợp lệ.")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Phản hồi hội thoại phải là một JSON object.")
    return value


def requested_targets(vocab: list[str], grammar: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for kind, values in (("vocabulary", vocab), ("grammar", grammar)):
        for index, value in enumerate(values, 1):
            text = str(value or "").strip()
            if text:
                rows.append({"id": f"{kind}-{index}", "type": kind, "term": text})
    return rows


def _text(value: Any) -> str:
    return str(value or "").strip()


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def _fallback_roles(language: str) -> list[dict[str, str]]:
    if language == "ja":
        return [
            {"id": "A", "name": "A", "role": "người nói A", "relationship": "theo ngữ cảnh", "register": "lịch sự"},
            {"id": "B", "name": "B", "role": "người nói B", "relationship": "theo ngữ cảnh", "register": "lịch sự"},
        ]
    return [
        {"id": "A", "name": "A", "role": "speaker A", "relationship": "contextual", "register": "neutral"},
        {"id": "B", "name": "B", "role": "speaker B", "relationship": "contextual", "register": "neutral"},
    ]


def _normalise_roles(value: Any, language: str) -> list[dict[str, str]]:
    source = value if isinstance(value, list) else []
    result: list[dict[str, str]] = []
    for index, row in enumerate(source[:2]):
        if not isinstance(row, dict):
            continue
        speaker_id = _text(row.get("id") or row.get("speaker_id") or ("A" if index == 0 else "B")).upper()
        if speaker_id not in {"A", "B"}:
            speaker_id = "A" if index == 0 else "B"
        result.append(
            {
                "id": speaker_id,
                "name": _text(row.get("name")) or speaker_id,
                "role": _text(row.get("role")),
                "relationship": _text(row.get("relationship")),
                "register": _text(row.get("register")),
            }
        )
    return result if {row["id"] for row in result} == {"A", "B"} else _fallback_roles(language)


def _normalise_turns(value: Any, language: str) -> list[dict[str, Any]]:
    source = value if isinstance(value, list) else []
    result: list[dict[str, Any]] = []
    for index, row in enumerate(source):
        if not isinstance(row, dict):
            continue
        speaker = _text(row.get("speaker_id") or row.get("speaker") or ("A" if index % 2 == 0 else "B")).upper()
        if speaker not in {"A", "B"}:
            speaker = "A" if index % 2 == 0 else "B"
        text = _text(row.get("text"))
        if not text:
            continue
        reading = _text(row.get("reading") or row.get("text_hira"))
        result.append(
            {
                "id": _text(row.get("id")) or f"turn-{index + 1:02d}",
                "speaker": speaker,
                "speaker_id": speaker,
                "text": text,
                "text_hira": reading if language == "ja" else "",
                "reading": reading if language == "ja" else "",
                "text_vi": _text(row.get("translation_vi") or row.get("text_vi")),
                "translation_vi": _text(row.get("translation_vi") or row.get("text_vi")),
                "speech_intent": _text(row.get("speech_intent")),
                "register": _text(row.get("register")),
                "target_ids": _clean_list(row.get("target_ids")),
                "naturalness_note": _text(row.get("naturalness_note")),
                "alternative_expression": _text(row.get("alternative_expression")),
                "highlights": _clean_list(row.get("highlights")),
                "practice_chunks": _clean_list(row.get("practice_chunks")),
            }
        )
    return result


def _normalise_key(value: str, language: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").strip()
    if language == "en":
        normalized = re.sub(r"\s+", " ", normalized).casefold()
    return normalized


def _appears_in_text(term: str, source: str, language: str) -> bool:
    term_key = _normalise_key(term, language)
    source_key = _normalise_key(source, language)
    if not term_key:
        return False
    if language == "en":
        return bool(re.search(rf"(?<![\w'-]){re.escape(term_key)}(?![\w'-])", source_key, re.I))
    return term_key in source_key


def _grammar_appears_in_text(term: str, source: str, language: str) -> bool:
    """Recognise conservative Japanese inflections when Gemini omits realised_form."""
    if _appears_in_text(term, source, language):
        return True
    if language != "ja":
        return False
    pattern = _normalise_key(term, language).lstrip("〜～").replace(" ", "")
    # These stems cover common textbook targets such as 〜てもらう -> てもらえます.
    if len(pattern) >= 3 and pattern[-1:] in {"う", "る", "い", "く", "す"}:
        return _appears_in_text(pattern[:-1], source, language)
    return False


def _target_rows(value: Any, targets: list[dict[str, str]], turns: list[dict[str, Any]], language: str) -> list[dict[str, Any]]:
    source = value if isinstance(value, list) else []
    raw_by_id = {str(row.get("id")): row for row in source if isinstance(row, dict)}
    turn_by_id = {row["id"]: row for row in turns}
    result: list[dict[str, Any]] = []
    for target in targets:
        raw = raw_by_id.get(target["id"], {})
        realized = _text(raw.get("realized_form"))
        turn_ids = [turn_id for turn_id in _clean_list(raw.get("turn_ids")) if turn_id in turn_by_id]
        if not turn_ids:
            turn_ids = [row["id"] for row in turns if target["id"] in row.get("target_ids", [])]
        selected_text = " ".join(turn_by_id[turn_id]["text"] for turn_id in turn_ids)
        match_term = realized or target["term"]
        covered = bool(turn_ids) and (
            _appears_in_text(match_term, selected_text, language)
            if realized or target["type"] == "vocabulary"
            else _grammar_appears_in_text(target["term"], selected_text, language)
        )
        # Exact vocabulary is locally checkable. Grammar can be inflected, so the
        # model must return its realised source form and that source form is checked.
        result.append(
            {
                **target,
                "realized_form": realized,
                "turn_ids": turn_ids,
                "explanation_vi": _text(raw.get("explanation_vi")),
                "covered": covered,
                "coverage_warning": (
                    "Dạng dùng cụ thể chưa được model ghi rõ; app đã xác minh theo thân mẫu ngữ pháp."
                    if covered and target["type"] == "grammar" and not realized
                    else "" if covered else "Chưa xác minh được dạng dùng trong câu gốc."
                ),
            }
        )
    return result


def _has_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text))


def _has_english(text: str) -> bool:
    letters = re.findall(r"[A-Za-z]", text)
    return len(letters) >= 2


def validate_dialogue(result: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic quality data; this is the source of truth, not model claims."""
    language = result.get("language_code", "ja")
    turns = result.get("dialogue", [])
    issues: list[str] = []
    warnings: list[str] = []
    if not 8 <= len(turns) <= 12:
        issues.append("Hội thoại cần có từ 8 đến 12 lượt nói.")
    for index, turn in enumerate(turns):
        if not turn.get("text_vi"):
            issues.append(f"Thiếu bản dịch tiếng Việt ở lượt {index + 1}.")
        if language == "ja" and not turn.get("text_hira"):
            issues.append(f"Thiếu cách đọc ở lượt {index + 1}.")
        if language == "ja" and not _has_japanese(turn.get("text", "")):
            issues.append(f"Lượt {index + 1} không có văn bản tiếng Nhật.")
        if language == "en" and not _has_english(turn.get("text", "")):
            issues.append(f"Lượt {index + 1} không có văn bản tiếng Anh.")
        if index and turn.get("speaker") == turns[index - 1].get("speaker"):
            issues.append(f"Hai lượt {index} và {index + 1} do cùng một người nói.")
        if not turn.get("speech_intent"):
            warnings.append(f"Lượt {index + 1} thiếu mục đích giao tiếp.")
    targets = result.get("learning_targets", [])
    uncovered = [row["term"] for row in targets if not row.get("covered")]
    if uncovered:
        issues.append("Chưa xác minh mục tiêu: " + ", ".join(uncovered))
    score = max(0, 100 - len(issues) * 12 - len(warnings) * 2)
    return {
        "quality_status": "complete" if not issues else "needs_repair",
        "quality_score": score,
        "issues": issues,
        "warnings": warnings,
        "missing_fields": issues,
        "uncovered_targets": uncovered,
    }


def normalize_dialogue_payload(
    payload: dict[str, Any],
    *,
    topic: str,
    language: str,
    level: str,
    situation: str,
    politeness_level: str,
    scenario_description: str,
    vocab: list[str],
    grammar: list[str],
) -> dict[str, Any]:
    """Convert a model payload to the stable V2 and legacy-compatible result shape."""
    code = language_code(language)
    targets = requested_targets(vocab, grammar)
    turns = _normalise_turns(payload.get("turns") or payload.get("dialogue"), code)
    learning_targets = _target_rows(payload.get("learning_targets"), targets, turns, code)
    for turn in turns:
        if not turn["highlights"]:
            turn["highlights"] = [
                row["term"] for row in learning_targets if turn["id"] in row["turn_ids"]
            ]
    scenario = payload.get("scenario") if isinstance(payload.get("scenario"), dict) else {}
    result = {
        "dialogue_version": DIALOGUE_VERSION,
        "topic": topic,
        "language": language,
        "language_code": code,
        "level": level,
        "situation": situation,
        "politeness_level": politeness_level,
        "scenario_description": scenario_description,
        "roles": _normalise_roles(payload.get("roles"), code),
        "scenario": {
            "opening": _text(scenario.get("opening")),
            "goal": _text(scenario.get("goal")),
            "problem": _text(scenario.get("problem")),
            "resolution": _text(scenario.get("resolution")),
        },
        "dialogue": turns,
        "learning_targets": learning_targets,
        "vocab_used": {row["term"]: row["explanation_vi"] for row in learning_targets if row["type"] == "vocabulary"},
        "grammar_used": {row["term"]: row["explanation_vi"] for row in learning_targets if row["type"] == "grammar"},
        "coverage_check": {row["term"]: bool(row["covered"]) for row in learning_targets},
        "summary": _text(payload.get("summary")),
        "notes": _text(payload.get("notes")),
        "model_quality": copy.deepcopy(payload.get("quality")) if isinstance(payload.get("quality"), dict) else {},
    }
    result["quality"] = validate_dialogue(result)
    result["fully_covered"] = not result["quality"]["uncovered_targets"]
    return result


def response_schema() -> dict[str, Any]:
    """Small portable schema. Local validation remains stricter than this contract."""
    turn = {
        "type": "object",
        "properties": {
            "id": {"type": "string"}, "speaker_id": {"type": "string"}, "text": {"type": "string"},
            "reading": {"type": "string"}, "translation_vi": {"type": "string"},
            "speech_intent": {"type": "string"}, "register": {"type": "string"},
            "target_ids": {"type": "array", "items": {"type": "string"}},
            "naturalness_note": {"type": "string"}, "alternative_expression": {"type": "string"},
            "practice_chunks": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "speaker_id", "text", "translation_vi", "speech_intent", "target_ids"],
    }
    return {
        "type": "object",
        "properties": {
            "roles": {"type": "array", "items": {"type": "object"}},
            "scenario": {"type": "object"},
            "turns": {"type": "array", "items": turn},
            "learning_targets": {"type": "array", "items": {"type": "object"}},
            "summary": {"type": "string"}, "notes": {"type": "string"}, "quality": {"type": "object"},
        },
        "required": ["roles", "scenario", "turns", "learning_targets", "summary", "notes", "quality"],
    }
