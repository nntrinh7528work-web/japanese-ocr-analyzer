"""Structured Japanese/English conversation-practice generation."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from modules.dialogue_schema import (
    language_code,
    normalize_dialogue_payload,
    parse_json_response,
    requested_targets,
    response_schema,
)


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
TOPIC_PROMPT_PATH = PROMPT_DIR / "topic_suggestion_prompt.txt"
PROMPT_BY_LANGUAGE = {
    "ja": PROMPT_DIR / "dialogue_prompt_ja_v2.txt",
    "en": PROMPT_DIR / "dialogue_prompt_en_v2.txt",
}


def _init_model(model_name: str | None = None):
    from config import GEMINI_API_KEY, GEMINI_MODEL_TEXT
    from modules.gemini_client import create_gemini_model

    if not GEMINI_API_KEY:
        raise ValueError("Thiếu GEMINI_API_KEY.")
    target_model = model_name or GEMINI_MODEL_TEXT
    model = create_gemini_model(target_model, GEMINI_API_KEY)
    model.target_model_name = target_model
    return model


def _usage(response: Any) -> dict[str, int]:
    metadata = getattr(response, "usage_metadata", None)
    candidate_tokens = int(getattr(metadata, "candidates_token_count", 0) or 0)
    thinking_tokens = int(getattr(metadata, "thoughts_token_count", 0) or 0)
    return {
        "input_tokens": int(getattr(metadata, "prompt_token_count", 0) or 0),
        "output_tokens": candidate_tokens + thinking_tokens,
        "candidate_tokens": candidate_tokens,
        "thinking_tokens": thinking_tokens,
    }


def _merge_usage(*items: dict[str, Any] | None) -> dict[str, int]:
    keys = ("input_tokens", "output_tokens", "candidate_tokens", "thinking_tokens")
    return {key: sum(int((item or {}).get(key, 0) or 0) for item in items) for key in keys}


def _section(text: str, name: str) -> str:
    match = re.search(rf"---{name}_START---\s*(.*?)\s*---{name}_END---", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def suggest_topics(language: str, level: str = "trung bình", recent_topics: list[str] | None = None) -> list[dict[str, str]]:
    """Suggest five topics. This deliberately remains lightweight and optional."""
    model = _init_model()
    prompt = TOPIC_PROMPT_PATH.read_text(encoding="utf-8").format(
        language=language, level=level, recent_topics=", ".join(recent_topics or []) or "Không có"
    )
    response = model.generate_content(prompt, generation_config={"temperature": 0.8})
    section = _section(getattr(response, "text", ""), "TOPICS")
    topics: list[dict[str, str]] = []
    for line in section.splitlines():
        match = re.match(r"^\d+\.\s*(.+?)\s*-\s*(.+)$", line.strip())
        if match:
            topics.append({"topic": match.group(1).strip(), "reason": match.group(2).strip()})
    return topics


def _dialogue_input(
    topic: str,
    language: str,
    vocab: list[str],
    grammar: list[str],
    level: str,
    situation: str,
    politeness_level: str,
    scenario_description: str,
    variation_context: str | None = None,
) -> dict[str, Any]:
    code = language_code(language)
    return {
        "task": "Generate a conversation practice lesson.",
        "language": "Japanese" if code == "ja" else "English",
        "locale": "ja-JP" if code == "ja" else "en-US",
        "topic": topic,
        "level": level,
        "situation": situation,
        "politeness_or_register": politeness_level,
        "scenario_description": scenario_description or "Create a realistic everyday development.",
        "learning_targets": requested_targets(vocab, grammar),
        "variation_requirement": variation_context or "",
        "output_contract": {
            "roles": "exactly A and B, each with id/name/role/relationship/register",
            "scenario": "opening, goal, problem, resolution",
            "turns": "8-12 alternating turns with id, speaker_id, text, reading, translation_vi, speech_intent, register, target_ids, naturalness_note, alternative_expression, practice_chunks (3-6 meaningful source-order chunks for reorder practice)",
            "learning_targets": "one item per requested target: id/type/realized_form/turn_ids/explanation_vi",
            "quality": "self-check notes only; do not omit any requested field",
        },
    }


def build_dialogue_prompt(
    topic: str,
    language: str,
    vocab: list[str],
    grammar: list[str],
    level: str,
    situation: str = "Tự nhiên / Thông thường",
    politeness_level: str = "Lịch sự (です/ます)",
    scenario_description: str = "",
    variation_context: str | None = None,
) -> str:
    payload = _dialogue_input(
        topic, language, vocab, grammar, level, situation, politeness_level, scenario_description, variation_context
    )
    template = PROMPT_BY_LANGUAGE[language_code(language)].read_text(encoding="utf-8")
    return template.format(payload_json=json.dumps(payload, ensure_ascii=False, indent=2))


def _generate_structured(model: Any, prompt: str, config: dict[str, Any]) -> Any:
    """Use schema JSON where available, retaining compatibility with older Gemini models."""
    candidates = [config]
    no_schema = {key: value for key, value in config.items() if key != "response_json_schema"}
    candidates.append(no_schema)
    if "thinking_config" in no_schema:
        candidates.append({key: value for key, value in no_schema.items() if key != "thinking_config"})
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return model.generate_content(prompt, generation_config=candidate)
        except Exception as exc:
            last_error = exc
    raise last_error or RuntimeError("Không thể gọi Gemini để tạo hội thoại.")


def _repair_prompt(result: dict[str, Any]) -> str:
    payload = {
        "instruction": "Repair only the invalid or missing fields. Preserve all valid text, speakers, target IDs and target meanings. Return the full JSON object again, with no Markdown.",
        "issues": result.get("quality", {}).get("issues", []),
        "conversation": {
            "language": result.get("language"),
            "roles": result.get("roles"),
            "scenario": result.get("scenario"),
            "turns": result.get("dialogue"),
            "learning_targets": result.get("learning_targets"),
            "summary": result.get("summary"),
            "notes": result.get("notes"),
        },
    }
    return (
        "You are repairing a structured language conversation. Do not alter the learner's requested targets. "
        "For Japanese, retain natural role-appropriate register and fill every reading. For English, retain natural spoken English.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _normalise_response(
    response: Any,
    *,
    topic: str,
    language: str,
    vocab: list[str],
    grammar: list[str],
    level: str,
    situation: str,
    politeness_level: str,
    scenario_description: str,
) -> dict[str, Any]:
    payload = parse_json_response(getattr(response, "text", ""))
    result = normalize_dialogue_payload(
        payload,
        topic=topic,
        language=language,
        level=level,
        situation=situation,
        politeness_level=politeness_level,
        scenario_description=scenario_description,
        vocab=vocab,
        grammar=grammar,
    )
    result["raw_text"] = getattr(response, "text", "")
    return result


def generate_dialogue(
    topic: str,
    language: str,
    vocab: list[str] | None = None,
    grammar: list[str] | None = None,
    level: str = "trung bình",
    situation: str = "Tự nhiên / Thông thường",
    politeness_level: str = "Lịch sự (です/ます)",
    scenario_description: str = "",
    max_retries: int = 1,
    variation_context: str | None = None,
    model_name: str | None = None,
    reasoning_effort: str = "standard",
) -> dict[str, Any]:
    """Generate a valid V2 dialogue and run at most one targeted repair."""
    vocab = [str(item).strip() for item in (vocab or []) if str(item).strip()]
    grammar = [str(item).strip() for item in (grammar or []) if str(item).strip()]
    model = _init_model(model_name)
    config: dict[str, Any] = {
        "temperature": 0.45 if not variation_context else 0.65,
        "max_output_tokens": 12288,
        "response_mime_type": "application/json",
        "response_json_schema": response_schema(),
    }
    if reasoning_effort == "deep":
        config["thinking_config"] = {"thinking_budget": 4096}
    primary = _generate_structured(
        model,
        build_dialogue_prompt(
            topic, language, vocab, grammar, level, situation, politeness_level, scenario_description, variation_context
        ),
        config,
    )
    result = _normalise_response(
        primary, topic=topic, language=language, vocab=vocab, grammar=grammar, level=level,
        situation=situation, politeness_level=politeness_level, scenario_description=scenario_description,
    )
    primary_usage = _usage(primary)
    repair_usage: dict[str, int] = _merge_usage()
    result["quality_repair_error"] = ""
    if result["quality"]["quality_status"] != "complete" and max_retries > 0:
        try:
            repair = _generate_structured(model, _repair_prompt(result), {**config, "temperature": 0.15})
            result = _normalise_response(
                repair, topic=topic, language=language, vocab=vocab, grammar=grammar, level=level,
                situation=situation, politeness_level=politeness_level, scenario_description=scenario_description,
            )
            repair_usage = _usage(repair)
        except Exception as exc:
            result["quality_repair_error"] = str(exc)
    result["model_used"] = getattr(model, "target_model_name", model_name or "")
    result["usage"] = _merge_usage(primary_usage, repair_usage)
    result["usage_detail"] = {"primary": primary_usage, "repair": repair_usage}
    return result


def generate_variation(
    previous_result: dict[str, Any],
    model_name: str | None = None,
    reasoning_effort: str = "standard",
) -> dict[str, Any]:
    """Preserve vocab/grammar types while requesting a substantially new scenario flow."""
    targets = previous_result.get("learning_targets", [])
    if isinstance(targets, list):
        vocab = [str(row.get("term", "")).strip() for row in targets if row.get("type") == "vocabulary"]
        grammar = [str(row.get("term", "")).strip() for row in targets if row.get("type") == "grammar"]
    else:
        vocab = list((previous_result.get("vocab_used") or {}).keys())
        grammar = list((previous_result.get("grammar_used") or {}).keys())
    previous_turns = "\n".join(f"{turn.get('speaker', '')}: {turn.get('text', '')}" for turn in previous_result.get("dialogue", []))
    variation_context = (
        "Create a different practical complication, solution and closing from this prior dialogue. "
        "Do not reuse more than one sentence or the same conversation arc.\n" + previous_turns
    )
    return generate_dialogue(
        topic=previous_result.get("topic", "Hội thoại"),
        language=previous_result.get("language", "Tiếng Nhật"),
        vocab=vocab,
        grammar=grammar,
        level=previous_result.get("level", "trung bình"),
        situation=previous_result.get("situation", "Tự nhiên / Thông thường"),
        politeness_level=previous_result.get("politeness_level", "Lịch sự (です/ます)"),
        scenario_description=previous_result.get("scenario_description", ""),
        variation_context=variation_context,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
    )


def suggest_vocab_grammar(topic: str, language: str, level: str) -> dict[str, list[str]]:
    """Suggest optional targets without silently altering user-entered targets."""
    model = _init_model()
    prompt = f'''Hãy đề xuất 3-5 từ vựng và 2-3 cấu trúc phù hợp để luyện hội thoại chủ đề "{topic}" bằng {language}, cấp độ {level}. Giải thích bằng tiếng Việt.
Trả đúng JSON: {{"vocab":[{{"term":"...","meaning_vi":"..."}}],"grammar":[{{"term":"...","meaning_vi":"..."}}]}}.'''
    response = model.generate_content(prompt, generation_config={"temperature": 0.5, "response_mime_type": "application/json"})
    try:
        payload = parse_json_response(getattr(response, "text", ""))
        def rows(name: str) -> list[str]:
            return [
                f"{str(row.get('term', '')).strip()} : {str(row.get('meaning_vi', '')).strip()}".strip(" :")
                for row in payload.get(name, []) if isinstance(row, dict) and str(row.get("term", "")).strip()
            ]
        return {"vocab": rows("vocab"), "grammar": rows("grammar")}
    except Exception:
        return {"vocab": [], "grammar": []}
