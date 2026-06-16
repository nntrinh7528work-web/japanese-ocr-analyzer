"""Gemini-backed English text analysis and Markdown parsing."""

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
        .replace("{english_text}", japanese_text)
        .replace("{japanese_text}", japanese_text)
        .replace("{ocr_notes}", notes)
    )


def _section(text: str, number: int) -> str:
    pattern = rf"(?ms)^##\s+{number}(?:[\.\):\-])?\s+.*?\n(.*?)(?=^##\s+\d+(?:[\.\):\-])?\s+|\n?\Z)"
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
    match = re.search(rf"(?mi)^\s*(?:[-*]\s*)?(?:\*\*)?(?:{labels})(?:\*\*)?\s*:\s*(.+)$", block)
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
                    "rule": _field(block, r"Rule|Quy tắc"),
                    "example": _field(block, r"Example from text|Ví dụ trong bài"),
                    "explanation": _field(block, r"Explanation|Giải thích(?: ý nghĩa & cách dùng)?"),
                }
            )
        else:
            results.append(
                {
                    "pattern": match.group(1).strip("` "),
                    "example": _field(block, r"Example from text|Ví dụ trong bài"),
                    "explanation": _field(block, r"Explanation|Giải thích"),
                }
            )
    return results


def _parse_grammar(section: str) -> list[dict[str, str]]:
    grammar = _parse_named_blocks(section, r"^\s*\*\*(?!Mẫu:)(.+?)\*\*\s*$", "grammar")
    if grammar:
        return grammar

    heading_matches = list(re.finditer(r"(?m)^#{3,5}\s+(.+?)\s*$", section))
    results = []
    for index, match in enumerate(heading_matches):
        block = section[match.end() : heading_matches[index + 1].start() if index + 1 < len(heading_matches) else None]
        results.append(
            {
                "name": match.group(1).strip("[] "),
                "rule": _field(block, r"Rule|Quy tắc|Cấu trúc|Mẫu câu"),
                "example": _field(block, r"Example from text|Ví dụ(?: trong bài)?"),
                "explanation": _field(block, r"Explanation|Giải thích(?: ý nghĩa & cách dùng)?|Ý nghĩa|Cách dùng"),
            }
        )
    return [item for item in results if any(item.values())]


def _section_markdown(response_text: str, number: int) -> str:
    section = _section(response_text, number)
    return section.strip()


def parse_analysis_response(response_text: str) -> dict[str, Any]:
    """Parse the seven-section Markdown analysis into structured output."""
    sections = {number: _section(response_text, number) for number in range(1, 8)}
    summary_match = re.search(
        r"(?im)^\s*(?:\*\*)?(?:Summary|Tóm tắt(?: nội dung)?)(?:\*\*)?\s*:\s*(.+)$",
        sections[1],
    )
    summary = summary_match.group(1).strip().strip("* ") if summary_match else ""
    if not summary:
        paragraphs = [
            line.strip()
            for line in sections[1].splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "[Sửa:", "-", "|"))
        ]
        summary = paragraphs[-1] if paragraphs else ""
    corrections = re.findall(r"\[(?:Correction|Sửa):\s*.*?\]", sections[1], re.DOTALL)
    vocab_all = _parse_table(
        _subsection(sections[2], "2.1"), ["num", "word", "base_form", "part_of_speech", "meaning", "cefr"]
    )
    vocab_important = _parse_table(
        _subsection(sections[2], "2.2"), ["word", "base_form", "part_of_speech", "meaning", "example", "difficulty"]
    )
    phrasal_collocations = _parse_table(
        sections[3], ["phrase", "type", "meaning", "example", "note"]
    )
    discourse_markers = _parse_table(
        sections[4], ["phrase", "function", "meaning", "example", "register", "difficulty"]
    )
    grammar = _parse_grammar(sections[5])
    patterns = _parse_named_blocks(sections[6], r"^\s*\*\*(?:Pattern|Mẫu):\*\*\s*(.+?)\s*$", "pattern")
    return {
        "confirmed_text": sections[1],
        "ocr_corrections": corrections,
        "summary": summary,
        "vocabulary_all": vocab_all,
        "vocabulary_important": vocab_important,
        "phrasal_collocations": phrasal_collocations,
        "discourse_markers": discourse_markers,
        "kanji_analysis": phrasal_collocations,
        "connectors": discourse_markers,
        "grammar_points": grammar,
        "sentence_patterns": patterns,
        "section_markdown": {
            "phrasal_collocations": _section_markdown(response_text, 3),
            "discourse_markers": _section_markdown(response_text, 4),
            "kanji": _section_markdown(response_text, 3),
            "connectors": _section_markdown(response_text, 4),
            "grammar": _section_markdown(response_text, 5),
            "patterns": _section_markdown(response_text, 6),
        },
        # The complete response is already a valid export document. Using it also
        # preserves useful analysis when Gemini reaches its token limit before section 7.
        "full_markdown": response_text.strip(),
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def build_missing_sections_prompt(japanese_text: str, missing_sections: list[str]) -> str:
    """Build a compact prompt to recover sections that were empty in the main analysis."""
    requested = ", ".join(missing_sections)
    return f"""You are an English language analyst for Vietnamese learners.
Return ONLY the missing Markdown sections requested here: {requested}.

English text:
{japanese_text}

Use these exact section formats when requested:

## 3. Phrasal Verbs &amp; Collocations
| Phrase | Type | Vietnamese Meaning | Example from Text | Note |
|---|---|---|---|---|

## 4. Linking Words &amp; Discourse Markers
| Word/Phrase | Function | Vietnamese Meaning | Example from Text | Register | Difficulty |
|---|---|---|---|---|---|

## 5. Grammar Points
For each point: **[Grammar Name]**, then `- Rule:`, `- Example from text:`, `- Explanation:`.

## 6. Sentence Patterns &amp; Structures
For each pattern: **Pattern:** `pattern description`, then `- Example from text:` and `- Explanation:`.
"""


def _merge_usage(base: dict[str, int], extra: dict[str, int]) -> dict[str, int]:
    keys = set(base) | set(extra)
    return {key: int(base.get(key, 0) or 0) + int(extra.get(key, 0) or 0) for key in keys}


def _fill_missing_sections(model: Any, parsed: dict[str, Any], text: str) -> dict[str, Any]:
    missing = []
    if not parsed["phrasal_collocations"]:
        missing.append("Phrasal verbs & collocations")
    if not parsed["discourse_markers"]:
        missing.append("Linking words & discourse markers")
    if not parsed["grammar_points"]:
        missing.append("Grammar points")
    if not parsed["sentence_patterns"]:
        missing.append("Sentence patterns")
    if not missing:
        return parsed

    try:
        response = model.generate_content(
            build_missing_sections_prompt(text, missing),
            generation_config={"temperature": 0.1, "max_output_tokens": 8192},
        )
        if not response.text or not response.text.strip():
            return parsed
        supplemental = parse_analysis_response(response.text)
        for field in ("phrasal_collocations", "discourse_markers", "grammar_points", "sentence_patterns"):
            if not parsed[field] and supplemental[field]:
                parsed[field] = supplemental[field]
        parsed["kanji_analysis"] = parsed["phrasal_collocations"]
        parsed["connectors"] = parsed["discourse_markers"]
        for key, value in supplemental.get("section_markdown", {}).items():
            if value and not parsed["section_markdown"].get(key):
                parsed["section_markdown"][key] = value
        if any(
            supplemental[field]
            for field in ("phrasal_collocations", "discourse_markers", "grammar_points", "sentence_patterns")
        ):
            parsed["full_markdown"] = f"{parsed['full_markdown']}\n\n---\n\n# Missing section supplement\n{response.text.strip()}"
        parsed["usage"] = _merge_usage(parsed.get("usage", {}), _usage(response))
    except Exception:
        return parsed
    return parsed


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
            parsed = _fill_missing_sections(model, parsed, text)
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
    """Analyze English text, splitting and merging input longer than 4,000 chars."""
    if not japanese_text or not japanese_text.strip():
        raise ValueError("Văn bản tiếng Anh không được rỗng.")
    model = _init_model()
    results = [_analyze_chunk(model, chunk, ocr_notes) for chunk in _split_text(japanese_text.strip())]
    if len(results) == 1:
        return results[0]

    merged = results[0]
    list_fields = (
        "ocr_corrections",
        "vocabulary_all",
        "vocabulary_important",
        "phrasal_collocations",
        "discourse_markers",
        "kanji_analysis",
        "connectors",
        "grammar_points",
        "sentence_patterns",
    )
    for field in list_fields:
        merged[field] = [item for result in results for item in result[field]]
    merged["kanji_analysis"] = merged["phrasal_collocations"]
    merged["connectors"] = merged["discourse_markers"]
    merged["confirmed_text"] = "\n\n".join(result["confirmed_text"] for result in results)
    merged["summary"] = " ".join(result["summary"] for result in results)
    merged["full_markdown"] = "\n\n".join(result["full_markdown"] for result in results)
    merged["section_markdown"] = {
        key: "\n\n".join(result.get("section_markdown", {}).get(key, "") for result in results).strip()
        for key in ("phrasal_collocations", "discourse_markers", "kanji", "connectors", "grammar", "patterns")
    }
    merged["usage"] = {
        "input_tokens": sum(result["usage"]["input_tokens"] for result in results),
        "output_tokens": sum(result["usage"]["output_tokens"] for result in results),
        "candidate_tokens": sum(result["usage"].get("candidate_tokens", 0) for result in results),
        "thinking_tokens": sum(result["usage"].get("thinking_tokens", 0) for result in results),
    }
    return merged
