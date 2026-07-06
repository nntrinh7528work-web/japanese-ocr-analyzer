"""Daily conversation practice generator with vocab/grammar constraints."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

DIALOGUE_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "dialogue_prompt.txt"
TOPIC_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "topic_suggestion_prompt.txt"


def _init_model():
    import google.generativeai as genai
    from config import GEMINI_API_KEY, GEMINI_MODEL_TEXT

    if not GEMINI_API_KEY:
        raise ValueError("Thiếu GEMINI_API_KEY.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL_TEXT)


def _section(text: str, name: str) -> str:
    match = re.search(rf"---{name}_START---\s*(.*?)\s*---{name}_END---", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def suggest_topics(language: str, level: str = "trung bình", recent_topics: list[str] | None = None) -> list[dict]:
    """Suggest 5 daily conversation topics."""
    model = _init_model()
    prompt = TOPIC_PROMPT_PATH.read_text(encoding="utf-8").format(
        language=language,
        level=level,
        recent_topics=", ".join(recent_topics or []) or "Không có",
    )
    response = model.generate_content(prompt, generation_config={"temperature": 0.8})
    section = _section(response.text, "TOPICS")
    topics = []
    for line in section.splitlines():
        match = re.match(r"^\d+\.\s*(.+?)\s*-\s*(.+)$", line.strip())
        if match:
            topics.append({"topic": match.group(1).strip(), "reason": match.group(2).strip()})
    return topics


def _parse_dialogue(text: str) -> list[dict[str, str]]:
    section = _section(text, "DIALOGUE")
    lines = [l for l in section.splitlines() if l.strip()]
    turns = []
    i = 0
    while i < len(lines) - 1:
        speaker_match = re.match(r"^([AB]):\s*(.+)$", lines[i].strip())
        vi_match = re.match(r"^([AB])_VI:\s*(.+)$", lines[i + 1].strip())
        if speaker_match and vi_match:
            highlights = re.findall(r"【(.+?)】", speaker_match.group(2))
            clean_text = re.sub(r"【|】", "", speaker_match.group(2))
            turns.append({
                "speaker": speaker_match.group(1),
                "text": clean_text,
                "text_vi": vi_match.group(2),
                "highlights": highlights,
            })
            i += 2
        else:
            i += 1
    return turns


def _parse_check(text: str, section_name: str) -> dict[str, str]:
    section = _section(text, section_name)
    result = {}
    for line in section.splitlines():
        match = re.match(r"^(.+?):\s*(.+)$", line.strip())
        if match:
            result[match.group(1).strip()] = match.group(2).strip()
    return result


def _validate_coverage(dialogue: list[dict], targets: list[str]) -> dict[str, bool]:
    """Check if every target vocab/grammar actually appears in the dialogue."""
    full_text = " ".join(turn["text"] for turn in dialogue)
    return {target: (target in full_text) for target in targets}


def build_dialogue_prompt(
    topic: str, language: str, vocab: list[str], grammar: list[str], level: str
) -> str:
    return DIALOGUE_PROMPT_PATH.read_text(encoding="utf-8").format(
        topic=topic,
        language=language,
        level=level,
        vocab_list="\n".join(f"- {v}" for v in vocab) or "Không có yêu cầu cụ thể",
        grammar_list="\n".join(f"- {g}" for g in grammar) or "Không có yêu cầu cụ thể",
    )


def generate_dialogue(
    topic: str,
    language: str,
    vocab: list[str] | None = None,
    grammar: list[str] | None = None,
    level: str = "trung bình",
    max_retries: int = 2,
) -> dict[str, Any]:
    """Generate a dialogue covering the required vocab/grammar. Retries if coverage incomplete."""
    vocab = vocab or []
    grammar = grammar or []
    model = _init_model()
    targets = vocab + grammar

    last_result = None
    for attempt in range(max_retries + 1):
        prompt = build_dialogue_prompt(topic, language, vocab, grammar, level)
        if attempt > 0 and last_result:
            missing = [t for t, covered in last_result["coverage_check"].items() if not covered]
            prompt += f"\n\nQUAN TRỌNG: Lần trước bạn CHƯA dùng các từ/ngữ pháp sau: {', '.join(missing)}. Bắt buộc phải dùng lần này."

        response = model.generate_content(prompt, generation_config={"temperature": 0.7})
        dialogue = _parse_dialogue(response.text)
        vocab_check = _parse_check(response.text, "VOCAB_CHECK")
        grammar_check = _parse_check(response.text, "GRAMMAR_CHECK")
        notes = _section(response.text, "NOTES")

        coverage = _validate_coverage(dialogue, targets)
        last_result = {
            "topic": topic,
            "language": language,
            "dialogue": dialogue,
            "vocab_used": vocab_check,
            "grammar_used": grammar_check,
            "coverage_check": coverage,
            "notes": notes,
            "fully_covered": all(coverage.values()) if targets else True,
        }

        if last_result["fully_covered"]:
            break

    return last_result
