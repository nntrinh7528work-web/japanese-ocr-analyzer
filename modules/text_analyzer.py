"""Gemini-backed Japanese text analysis and Markdown parsing."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "analysis_prompt.txt"


def build_analysis_prompt(japanese_text: str, ocr_notes: list) -> str:
    """Build the analysis prompt with numbered OCR notes."""
    notes = "\n".join(f"{index}. {note}" for index, note in enumerate(ocr_notes, 1)) or "Không có"
    return (
        PROMPT_PATH.read_text(encoding="utf-8")
        .replace("{japanese_text}", japanese_text)
        .replace("{ocr_notes}", notes)
    )


def _section(text: str, number: int) -> str:
    pattern = rf"(?ms)^##\s+{number}(?:[\.\):\-])?\s+.*?\n(.*?)(?=^##\s+\d+(?:[\.\):\-])?\s+|\Z)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _subsection(text: str, number: str) -> str:
    pattern = rf"(?ms)^###\s+{re.escape(number)}(?:[\.\):\-])?\s+.*?\n(.*?)(?=^###\s+\d+\.\d+(?:[\.\):\-])?\s+|\Z)"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _parse_table(section: str, keys: list[str]) -> list[dict[str, str]]:
    lines = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return []
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        values = [cell.strip() for cell in line.strip("|").split("|")]
        if len(values) < len(keys):
            continue
        rows.append(dict(zip(keys, values[: len(keys)])))
    return rows


def _field(block: str, labels: str) -> str:
    match = re.search(rf"(?mi)^\s*-\s*(?:{labels})\s*:\s*(.+)$", block)
    return match.group(1).strip().strip("`") if match else ""


def _parse_named_blocks(section: str, heading_pattern: str, kind: str) -> list[dict[str, str]]:
    matches = list(re.finditer(heading_pattern, section, re.MULTILINE))
    results = []
    for index, match in enumerate(matches):
        block = section[match.end() : matches[index + 1].start() if index + 1 < len(matches) else None]
        if kind == "grammar":
            results.append(
                {
                    "name": match.group(1).strip("[] "),
                    "rule": _field(block, r"Quy tắc"),
                    "example": _field(block, r"Ví dụ trong bài"),
                    "explanation": _field(block, r"Giải thích(?: ý nghĩa & cách dùng)?"),
                }
            )
        else:
            results.append(
                {
                    "pattern": match.group(1).strip("` "),
                    "example": _field(block, r"Ví dụ trong bài"),
                    "explanation": _field(block, r"Giải thích"),
                }
            )
    return results


def parse_analysis_response(response_text: str) -> dict[str, Any]:
    """Parse the seven-section Markdown analysis into structured output."""
    sections = {number: _section(response_text, number) for number in range(1, 8)}
    summary_match = re.search(
        r"(?is)(?:\*\*)?Tóm tắt(?: nội dung)?\s*:?(?:\*\*)?\s*:?\s*(.*?)(?=\n\s*(?:---|#|\*\*)|\Z)",
        sections[1],
    )
    summary = summary_match.group(1).strip() if summary_match else ""
    if not summary:
        paragraphs = [
            line.strip()
            for line in sections[1].splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "[Sửa:", "-", "|"))
        ]
        summary = paragraphs[-1] if paragraphs else ""
    corrections = re.findall(r"\[Sửa:\s*.*?\]", sections[1], re.DOTALL)
    vocab_all = _parse_table(
        _subsection(sections[2], "2.1"), ["num", "word", "reading", "type", "meaning", "jlpt"]
    )
    vocab_important = _parse_table(
        _subsection(sections[2], "2.2"), ["word", "reading", "type", "meaning", "example", "difficulty"]
    )
    kanji = _parse_table(
        sections[3], ["kanji", "onyomi", "kunyomi", "meaning", "jlpt", "vocab", "example", "role"]
    )
    connectors = _parse_table(
        sections[4], ["phrase", "reading", "type", "meaning", "example", "role", "difficulty"]
    )
    grammar = _parse_named_blocks(sections[5], r"^\s*\*\*(?!Mẫu:)(.+?)\*\*\s*$", "grammar")
    patterns = _parse_named_blocks(sections[6], r"^\s*\*\*Mẫu:\*\*\s*(.+?)\s*$", "pattern")
    return {
        "confirmed_text": sections[1],
        "ocr_corrections": corrections,
        "summary": summary,
        "vocabulary_all": vocab_all,
        "vocabulary_important": vocab_important,
        "kanji_analysis": kanji,
        "connectors": connectors,
        "grammar_points": grammar,
        "sentence_patterns": patterns,
        # The complete response is already a valid export document. Using it also
        # preserves useful analysis when Gemini reaches its token limit before section 7.
        "full_markdown": response_text.strip(),
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _init_model():
    import google.generativeai as genai

    from config import GEMINI_API_KEY, GEMINI_MODEL_TEXT

    if not GEMINI_API_KEY:
        raise ValueError("Thiếu GEMINI_API_KEY. Hãy cấu hình key trong .env hoặc Streamlit secrets.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL_TEXT)


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


def _analyze_chunk(model: Any, text: str, notes: list) -> dict[str, Any]:
    prompt = build_analysis_prompt(text, notes)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 16384},
            )
            if not response.text or not response.text.strip():
                raise ValueError("Gemini không trả về nội dung phân tích.")
            parsed = parse_analysis_response(response.text)
            if not parsed["confirmed_text"]:
                parsed["confirmed_text"] = text
            if not parsed["summary"]:
                parsed["summary"] = "Không thể trích xuất tóm tắt riêng; xem toàn bộ nội dung phân tích bên dưới."
            parsed["usage"] = _usage(response)
            return parsed
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"Phân tích thất bại sau 3 lần thử: {last_error}") from last_error


def _split_text(text: str, max_chars: int = 4000) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    remaining = text
    while remaining:
        split_at = min(max_chars, len(remaining))
        if split_at < len(remaining):
            boundary = max(remaining.rfind("\n", 0, split_at), remaining.rfind("。", 0, split_at))
            if boundary > max_chars // 2:
                split_at = boundary + 1
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return [chunk for chunk in chunks if chunk]


def run_analysis(japanese_text: str, ocr_notes: list) -> dict[str, Any]:
    """Analyze Japanese text, splitting and merging input longer than 4,000 chars."""
    if not japanese_text or not japanese_text.strip():
        raise ValueError("Văn bản tiếng Nhật không được rỗng.")
    model = _init_model()
    results = [_analyze_chunk(model, chunk, ocr_notes) for chunk in _split_text(japanese_text.strip())]
    if len(results) == 1:
        return results[0]

    merged = results[0]
    list_fields = (
        "ocr_corrections",
        "vocabulary_all",
        "vocabulary_important",
        "kanji_analysis",
        "connectors",
        "grammar_points",
        "sentence_patterns",
    )
    for field in list_fields:
        merged[field] = [item for result in results for item in result[field]]
    merged["confirmed_text"] = "\n\n".join(result["confirmed_text"] for result in results)
    merged["summary"] = " ".join(result["summary"] for result in results)
    merged["full_markdown"] = "\n\n".join(result["full_markdown"] for result in results)
    merged["usage"] = {
        "input_tokens": sum(result["usage"]["input_tokens"] for result in results),
        "output_tokens": sum(result["usage"]["output_tokens"] for result in results),
        "candidate_tokens": sum(result["usage"].get("candidate_tokens", 0) for result in results),
        "thinking_tokens": sum(result["usage"].get("thinking_tokens", 0) for result in results),
    }
    return merged
