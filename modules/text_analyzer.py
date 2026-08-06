"""Gemini-backed Japanese/English text analysis and Markdown parsing."""

from __future__ import annotations

import copy
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from modules.sentence_analyzer import (
    ZERO_USAGE,
    aggregate_sentence_usage,
    aggregate_sentence_runs,
    analyze_sentence_batch,
    attach_sentence_data,
    build_sentence_catalog,
    select_auto_sentences,
)


PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_PATHS = {
    "english": PROMPT_DIR / "analysis_prompt.txt",
    "japanese": PROMPT_DIR / "analysis_prompt_japanese.txt",
}


def _analysis_language(value: str | None) -> str:
    return "japanese" if value == "japanese" else "english"


def build_analysis_prompt(japanese_text: str, ocr_notes: list, analysis_language: str = "english") -> str:
    """Build the analysis prompt with numbered OCR notes."""
    language = _analysis_language(analysis_language)
    notes = "\n".join(f"{index}. {note}" for index, note in enumerate(ocr_notes, 1)) or "Không có"
    return (
        PROMPT_PATHS[language].read_text(encoding="utf-8")
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
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    optional_number_column = (
        len(headers) == len(keys) + 1
        and keys[0] != "num"
        and headers[0].strip().lower().rstrip(".") in {"#", "stt", "no", "number", "số"}
    )
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        values = [cell.strip() for cell in line.strip("|").split("|")]
        if optional_number_column and len(values) >= len(keys) + 1:
            values = values[1:]
        if len(values) < len(keys):
            continue
        rows.append(dict(zip(keys, values[: len(keys)])))
    return rows


def _field(block: str, labels: str) -> str:
    match = re.search(rf"(?mi)^\s*(?:[-*]\s*)?(?:\*\*)?(?:{labels})(?:\*\*)?\s*:\s*(.+)$", block)
    return match.group(1).strip().strip("`") if match else ""


def _split_inline_hiragana(value: str) -> tuple[str, str]:
    """Split examples that include a trailing ``(Hiragana: ...)`` note."""
    match = re.search(r"(?i)(?:\s*[-–—]\s*)?\(?\s*(?:Hiragana|ひらがな)\s*:\s*([^)]+?)\s*\)?\s*$", value)
    if not match:
        return value, ""
    clean_value = value[: match.start()].rstrip(" -–—")
    return clean_value.strip(), match.group(1).strip()


def _example_fields(block: str, example_labels: str, hiragana_labels: str) -> tuple[str, str]:
    example, inline_hiragana = _split_inline_hiragana(_field(block, example_labels))
    explicit_hiragana = _field(block, hiragana_labels)
    return example, explicit_hiragana or inline_hiragana


def _parse_named_blocks(section: str, heading_pattern: str, kind: str) -> list[dict[str, str]]:
    matches = list(re.finditer(heading_pattern, section, re.MULTILINE))
    results = []
    for index, match in enumerate(matches):
        block = section[match.end() : matches[index + 1].start() if index + 1 < len(matches) else None]
        if kind == "grammar":
            example, example_hiragana = _example_fields(block, r"Example from text|Ví dụ trong bài", r"Hiragana ví dụ trong bài")
            example_1, example_1_hiragana = _example_fields(block, r"Example 1|Ví dụ 1", r"Hiragana ví dụ 1")
            example_2, example_2_hiragana = _example_fields(block, r"Example 2|Ví dụ 2", r"Hiragana ví dụ 2")
            results.append(
                {
                    "name": re.sub(r"^\d+[\.\)]\s*", "", match.group(1).strip("[] ")),
                    "structure": _field(block, r"Structure|Công thức|Cấu trúc"),
                    "rule": _field(block, r"Rule|Quy tắc"),
                    "meaning": _field(block, r"Meaning|Ý nghĩa"),
                    "usage": _field(block, r"Cách dùng|Usage"),
                    "formation": _field(block, r"Cấu tạo trong câu|Formation in context"),
                    "nuance": _field(block, r"Sắc thái(?: / văn phong)?|Nuance / Register|Nuance|Register"),
                    "example": example,
                    "example_hiragana": example_hiragana,
                    "example_analysis": _field(block, r"Example analysis|Phân tích ví dụ"),
                    "explanation": _field(block, r"Explanation|Giải thích(?: ý nghĩa & cách dùng)?"),
                    "example_1": example_1,
                    "example_1_hiragana": example_1_hiragana,
                    "example_2": example_2,
                    "example_2_hiragana": example_2_hiragana,
                    "note": _field(block, r"Lưu ý"),
                    "comparison": _field(block, r"Phân biệt|Comparison"),
                    "mistake": _field(block, r"Common mistake"),
                    "level": _field(block, r"Level|Mức độ"),
                }
            )
        elif kind == "vocab_detail":
            example_text, example_text_hiragana = _example_fields(
                block, r"Ví dụ trong bài|Example from text", r"Hiragana ví dụ trong bài"
            )
            example_1, example_1_hiragana = _example_fields(block, r"Ví dụ 1|Example 1", r"Hiragana ví dụ 1")
            example_2, example_2_hiragana = _example_fields(block, r"Ví dụ 2|Example 2", r"Hiragana ví dụ 2")
            results.append(
                {
                    "word": re.sub(r"^\d+[\.\)]\s*", "", match.group(1).strip("[] ")),
                    "type": _field(block, r"Loại từ"),
                    "meaning": _field(block, r"Ý nghĩa"),
                    "vn_meaning": _field(block, r"Vietnamese Meaning"),
                    "definition": _field(block, r"Definition"),
                    "example_text": example_text,
                    "example_text_hiragana": example_text_hiragana,
                    "example_1": example_1,
                    "example_1_hiragana": example_1_hiragana,
                    "example_2": example_2,
                    "example_2_hiragana": example_2_hiragana,
                    "related": _field(block, r"Từ liên quan|Related words"),
                    "note": _field(block, r"Lưu ý"),
                    "mistake": _field(block, r"Common mistake"),
                    "jlpt": _field(block, r"Mức độ"),
                    "cefr": _field(block, r"CEFR Level"),
                }
            )
        else:
            results.append(
                {
                    "pattern": re.sub(r"^\d+[\.\)]\s*", "", match.group(1).strip("` ")),
                    "example": _field(block, r"Example from text|Ví dụ trong bài"),
                    "components": _field(block, r"Sentence components|Thành phần câu"),
                    "function": _field(block, r"Communicative function|Chức năng giao tiếp"),
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
                "name": re.sub(r"^\d+[\.\)]\s*", "", match.group(1).strip("[] ")),
                "structure": _field(block, r"Structure|Công thức|Cấu trúc|Mẫu câu"),
                "rule": _field(block, r"Rule|Quy tắc|Cấu trúc|Mẫu câu"),
                "meaning": _field(block, r"Meaning|Ý nghĩa"),
                "usage": _field(block, r"Cách dùng|Usage"),
                "formation": _field(block, r"Cấu tạo trong câu|Formation in context"),
                "nuance": _field(block, r"Sắc thái(?: / văn phong)?|Nuance / Register|Nuance|Register"),
                "example": _field(block, r"Example from text|Ví dụ(?: trong bài)?"),
                "example_analysis": _field(block, r"Example analysis|Phân tích ví dụ"),
                "explanation": _field(block, r"Explanation|Giải thích(?: ý nghĩa & cách dùng)?|Ý nghĩa|Cách dùng"),
                "example_1": _field(block, r"Example 1|Ví dụ 1"),
                "example_2": _field(block, r"Example 2|Ví dụ 2"),
                "note": _field(block, r"Lưu ý"),
                "comparison": _field(block, r"Phân biệt|Comparison"),
                "mistake": _field(block, r"Common mistake"),
                "level": _field(block, r"Level|Mức độ"),
            }
        )
    return [item for item in results if any(item.values())]


def _section_markdown(response_text: str, number: int) -> str:
    section = _section(response_text, number)
    return section.strip()


def parse_analysis_response(response_text: str, analysis_language: str = "english") -> dict[str, Any]:
    """Parse the seven-section Markdown analysis into structured output."""
    language = _analysis_language(analysis_language)
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
    if language == "japanese":
        vocab_all = _parse_table(
            _subsection(sections[2], "2.1"), ["num", "word", "reading", "type", "meaning", "jlpt"]
        )
        vocab_important_section = _subsection(sections[2], "2.2")
        vocab_important = _parse_named_blocks(
            vocab_important_section,
            r"^\s*\*\*\[(.+?)\]\*\*\s*$",
            "vocab_detail",
        )
        if not vocab_important:
            vocab_important = _parse_table(
                vocab_important_section, ["word", "reading", "type", "meaning", "example", "difficulty"]
            )
        kanji = _parse_table(
            sections[3], ["kanji", "onyomi", "kunyomi", "meaning", "jlpt", "vocab", "example", "role"]
        )
        connectors = _parse_table(
            sections[4],
            [
                "phrase",
                "reading",
                "type",
                "structure",
                "meaning",
                "example",
                "linked_parts",
                "role",
                "difficulty",
            ],
        )
        if not connectors:
            connectors = _parse_table(
                sections[4], ["phrase", "reading", "type", "meaning", "example", "role", "difficulty"]
            )
        phrasal_collocations: list[dict[str, str]] = []
        discourse_markers: list[dict[str, str]] = []
    else:
        vocab_all = _parse_table(
            _subsection(sections[2], "2.1"), ["num", "word", "base_form", "part_of_speech", "meaning", "cefr"]
        )
        vocab_important_section = _subsection(sections[2], "2.2")
        vocab_important = _parse_named_blocks(
            vocab_important_section,
            r"^\s*\*\*\[(.+?)\]\*\*\s*$",
            "vocab_detail",
        )
        if not vocab_important:
            vocab_important = _parse_table(
                vocab_important_section,
                ["word", "base_form", "part_of_speech", "meaning", "example", "difficulty"],
            )
        phrasal_collocations = _parse_table(
            sections[3], ["phrase", "type", "meaning", "example", "note"]
        )
        discourse_markers = _parse_table(
            sections[4],
            [
                "phrase",
                "type",
                "function",
                "meaning",
                "example",
                "linked_parts",
                "register",
                "usage",
                "difficulty",
            ],
        )
        if not discourse_markers:
            discourse_markers = _parse_table(
                sections[4], ["phrase", "function", "meaning", "example", "register", "difficulty"]
            )
        kanji = phrasal_collocations
        connectors = discourse_markers
    grammar = _parse_grammar(sections[5])
    patterns = _parse_named_blocks(sections[6], r"^\s*\*\*(?:Pattern|Mẫu):\*\*\s*(.+?)\s*$", "pattern")
    _renumber_rows(vocab_all)
    return {
        "analysis_language": language,
        "confirmed_text": sections[1],
        "ocr_corrections": corrections,
        "summary": summary,
        "vocabulary_all": vocab_all,
        "vocabulary_important": vocab_important,
        "phrasal_collocations": phrasal_collocations,
        "discourse_markers": discourse_markers,
        "kanji_analysis": kanji,
        "connectors": connectors,
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


def build_missing_sections_prompt(
    japanese_text: str,
    missing_sections: list[str],
    analysis_language: str = "english",
) -> str:
    """Build a compact prompt to recover sections that were empty in the main analysis."""
    language = _analysis_language(analysis_language)
    requested = ", ".join(missing_sections)
    if language == "japanese":
        return f"""Bạn là giáo viên tiếng Nhật cho người Việt.
Phân tích BỔ SUNG chỉ các mục còn thiếu sau: {requested}.

Văn bản:
{japanese_text}

Trả lời đúng Markdown theo các tiêu đề sau nếu được yêu cầu:

## 3. PHÂN TÍCH KANJI
| Kanji | Onyomi | Kunyomi | Nghĩa cơ bản | JLPT | Từ vựng trong bài (≤5 từ) | Câu ví dụ trong bài | Vai trò trong từ |
|---|---|---|---|---|---|---|---|

## 4. TỪ NỐI CÂU & LIÊN TỪ
Bao gồm 接続詞, 接続助詞, trạng từ liên kết/cụm diễn ngôn và từ quy chiếu nối ý.
Xếp theo thứ tự xuất hiện đầu tiên; nêu cấu trúc nối, hai thành phần được nối,
quan hệ logic và sắc thái trong ngữ cảnh.
| Từ/Cụm | Phiên âm | Nhóm | Cấu trúc/Cách nối | Nghĩa tiếng Việt | Ví dụ trong bài | Hai thành phần được nối | Quan hệ logic & sắc thái | JLPT |
|---|---|---|---|---|---|---|---|---|

## 5. PHÂN TÍCH NGỮ PHÁP
Với mỗi điểm:
**[Tên cấu trúc]**
- Công thức:
- Ý nghĩa:
- Cách dùng:
- Cấu tạo trong câu:
- Sắc thái / văn phong:
- Ví dụ trong bài:
- Phân tích ví dụ:
- Ví dụ 1:
- Ví dụ 2:
- Phân biệt:
- Lưu ý:
- Mức độ:

## 6. MẪU CÂU ĐẶC TRƯNG
Với mỗi mẫu: **Mẫu:** `[công thức / pattern]`, rồi `- Ví dụ trong bài:`,
`- Thành phần câu:`, `- Chức năng giao tiếp:` và `- Giải thích:`.

Nếu cần bổ sung mục từ vựng khó ở định dạng 2.2 trong các lần sau, luôn thêm:
`- Hiragana ví dụ trong bài:`, `- Hiragana ví dụ 1:`, `- Hiragana ví dụ 2:`
ngay sau từng câu ví dụ tiếng Nhật.
"""
    return f"""You are an English language analyst for Vietnamese learners.
Return ONLY the missing Markdown sections requested here: {requested}.
All notes, rules, explanations, and recommendations MUST be written in Vietnamese.
Keep only the original English examples/quotes in English.
Keep section headers and table column headers exactly as shown.

English text:
{japanese_text}

Use these exact section formats when requested:

## 3. Phrasal Verbs &amp; Collocations
| Phrase | Type | Vietnamese Meaning | Example from Text | Note |
|---|---|---|---|---|

## 4. Linking Words &amp; Discourse Markers
Cover conjunctions, subordinators, conjunctive adverbs, transition phrases and
referential cohesion. Keep source order and explain the connected parts.
| Word/Phrase | Category | Function | Vietnamese Meaning | Example from Text | Connected Parts | Register & Nuance | Position/Punctuation | Difficulty |
|---|---|---|---|---|---|---|---|---|

## 5. Grammar Points
For each point:
**[Grammar Name]**
- Structure:
- Rule: in Vietnamese
- Meaning: in Vietnamese
- Usage: in Vietnamese
- Nuance / Register: in Vietnamese
- Example from text: exact English quote
- Example analysis: in Vietnamese
- Example 1:
- Example 2:
- Comparison: in Vietnamese
- Common mistake:
- Level:

## 6. Sentence Patterns &amp; Structures
For each pattern: **Pattern:** `pattern description`, then `- Example from text:`
as the exact English quote, `- Sentence components:`, `- Communicative function:`
and `- Explanation:` in Vietnamese.
"""


def _merge_usage(base: dict[str, int], extra: dict[str, int]) -> dict[str, int]:
    keys = set(base) | set(extra)
    return {key: int(base.get(key, 0) or 0) + int(extra.get(key, 0) or 0) for key in keys}


def _fill_missing_sections(
    model: Any,
    parsed: dict[str, Any],
    text: str,
    analysis_language: str = "english",
    reasoning_effort: str = "standard",
) -> dict[str, Any]:
    language = _analysis_language(analysis_language)
    missing = []
    if language == "japanese":
        if not parsed["kanji_analysis"]:
            missing.append("Kanji")
        if not parsed["connectors"]:
            missing.append("Từ nối")
    else:
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

    gen_config: dict[str, Any] = {"temperature": 0.1, "max_output_tokens": 8192}
    if reasoning_effort == "deep":
        gen_config["thinking_config"] = {"thinking_budget": 4096}

    try:
        try:
            response = model.generate_content(
                build_missing_sections_prompt(text, missing, language),
                generation_config=gen_config,
            )
        except Exception:
            if "thinking_config" in gen_config:
                fallback_config = {k: v for k, v in gen_config.items() if k != "thinking_config"}
                response = model.generate_content(
                    build_missing_sections_prompt(text, missing, language),
                    generation_config=fallback_config,
                )
            else:
                raise

        if not response.text or not response.text.strip():
            return parsed
        supplemental = parse_analysis_response(response.text, language)
        fields = (
            ("kanji_analysis", "connectors", "grammar_points", "sentence_patterns")
            if language == "japanese"
            else ("phrasal_collocations", "discourse_markers", "grammar_points", "sentence_patterns")
        )
        for field in fields:
            if not parsed[field] and supplemental[field]:
                parsed[field] = supplemental[field]
        if language == "english":
            parsed["kanji_analysis"] = parsed["phrasal_collocations"]
            parsed["connectors"] = parsed["discourse_markers"]
        for key, value in supplemental.get("section_markdown", {}).items():
            if value and not parsed["section_markdown"].get(key):
                parsed["section_markdown"][key] = value
        if any(supplemental[field] for field in fields):
            title = "Bổ sung mục còn thiếu" if language == "japanese" else "Missing section supplement"
            parsed["full_markdown"] = f"{parsed['full_markdown']}\n\n---\n\n# {title}\n{response.text.strip()}"
        parsed["usage"] = _merge_usage(parsed.get("usage", {}), _usage(response))
    except Exception:
        return parsed
    return parsed


def _init_model(model_name: str | None = None):
    from config import GEMINI_API_KEY, GEMINI_MODEL_TEXT
    from modules.gemini_client import create_gemini_model

    if not GEMINI_API_KEY:
        raise ValueError("Thiếu GEMINI_API_KEY. Hãy cấu hình key trong .env hoặc Streamlit secrets.")
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


def _analyze_chunk(
    model: Any,
    text: str,
    notes: list,
    analysis_language: str = "english",
    reasoning_effort: str = "standard",
) -> dict[str, Any]:
    language = _analysis_language(analysis_language)
    prompt = build_analysis_prompt(text, notes, language)
    last_error: Exception | None = None

    gen_config: dict[str, Any] = {
        "temperature": 0.1,
        "max_output_tokens": 16384,
    }
    if reasoning_effort == "deep":
        gen_config["thinking_config"] = {"thinking_budget": 4096}

    for attempt in range(3):
        try:
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=gen_config,
                )
            except Exception as gen_exc:
                if "thinking_config" in gen_config:
                    fallback_config = {k: v for k, v in gen_config.items() if k != "thinking_config"}
                    response = model.generate_content(
                        prompt,
                        generation_config=fallback_config,
                    )
                else:
                    raise gen_exc

            candidates = getattr(response, "candidates", None)
            finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
            if finish_reason == 2:  # MAX_TOKENS — response bị cắt giữa đường
                raise ValueError("Response bị cắt do vượt giới hạn token, cần thử lại.")
            if not response.text or not response.text.strip():
                raise ValueError("Gemini không trả về nội dung phân tích.")
            parsed = parse_analysis_response(response.text, language)
            if not parsed["confirmed_text"]:
                parsed["confirmed_text"] = text
            if not parsed["summary"]:
                parsed["summary"] = "Không thể trích xuất tóm tắt riêng; xem toàn bộ nội dung phân tích bên dưới."
            parsed["usage"] = _usage(response)
            parsed["model_used"] = getattr(model, "target_model_name", "gemini-3.5-flash")
            parsed = _fill_missing_sections(model, parsed, text, language, reasoning_effort=reasoning_effort)
            return parsed
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"Phân tích thất bại sau 3 lần thử: {last_error}") from last_error


def _split_text(text: str, max_chars: int = 2500) -> list[str]:
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


def _deduplicate_rows(rows: list[dict[str, str]], key_fields: tuple[str, ...]) -> list[dict[str, str]]:
    """Remove duplicate rows based on the first matching key field value."""
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        identifier = ""
        for key in key_fields:
            value = row.get(key, "").strip()
            if value and value not in ("—", "--", "N/A"):
                identifier = value.lower()
                break
        if not identifier or identifier not in seen:
            if identifier:
                seen.add(identifier)
            unique.append(row)
    return unique


def _renumber_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Re-assign sequential STT numbers to rows that have a 'num' field."""
    for index, row in enumerate(rows, 1):
        if "num" in row:
            row["num"] = str(index)
    return rows


def _merge_analysis_results(results: list[dict[str, Any]], analysis_language: str = "english") -> dict[str, Any]:
    """Merge already parsed analysis results without losing per-result detail."""
    if not results:
        raise ValueError("Không có kết quả phân tích để gộp.")
    language = _analysis_language(analysis_language)
    if len(results) == 1:
        return copy.deepcopy(results[0])

    merged = copy.deepcopy(results[0])
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
    # Deduplication key fields for each list type.
    dedup_keys: dict[str, tuple[str, ...]] = {
        "vocabulary_all": ("word",),
        "vocabulary_important": ("word",),
        "phrasal_collocations": ("phrase",),
        "discourse_markers": ("phrase",),
        "kanji_analysis": ("kanji", "phrase"),
        "connectors": ("phrase",),
        "grammar_points": ("name",),
        "sentence_patterns": ("pattern",),
    }
    for field in list_fields:
        combined = [item for result in results for item in result.get(field, [])]
        keys = dedup_keys.get(field)
        if keys:
            combined = _deduplicate_rows(combined, keys)
        merged[field] = combined
    # Renumber vocabulary STT after merge.
    _renumber_rows(merged["vocabulary_all"])
    if language == "english":
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
    merged["model_used"] = results[0].get("model_used", "gemini-3.5-flash") if results else "gemini-3.5-flash"
    return merged


def run_analysis(
    japanese_text: str,
    ocr_notes: list,
    analysis_language: str = "english",
    model_name: str | None = None,
    reasoning_effort: str = "standard",
) -> dict[str, Any]:
    """Analyze Japanese or English text, splitting and merging input longer than 2,500 chars."""
    if not japanese_text or not japanese_text.strip():
        raise ValueError("Văn bản phân tích không được rỗng.")
    language = _analysis_language(analysis_language)
    model = _init_model(model_name) if model_name else _init_model()
    results = [
        _analyze_chunk(model, chunk, ocr_notes, language, reasoning_effort=reasoning_effort)
        for chunk in _split_text(japanese_text.strip())
    ]
    merged = _merge_analysis_results(results, language)

    # Đánh lại số thứ tự liên tục cho vocabulary_all (tránh 1,2,3...1,2,3...)
    for index, row in enumerate(merged["vocabulary_all"], 1):
        row["num"] = str(index)

    return merged


def analyze_single_page(
    model: Any,
    page: dict[str, Any],
    analysis_language: str = "english",
    reasoning_effort: str = "standard",
) -> dict[str, Any]:
    """Analyze a single prepared page dict and return the parsed result."""
    language = _analysis_language(analysis_language)
    page_results = [
        _analyze_chunk(model, chunk, page["notes"], language, reasoning_effort=reasoning_effort)
        for chunk in _split_text(page["text"])
    ]
    page_analysis = _merge_analysis_results(page_results, language)
    page_analysis["page_index"] = page["page_index"]
    page_analysis["page_name"] = page["page_name"]
    page_analysis["source_label"] = f"Trang {page['page_index']}: {page['page_name']}"
    page_analysis["source_text"] = page["text"]
    return page_analysis


def merge_page_analyses(
    page_analyses: list[dict[str, Any]],
    analysis_language: str = "english",
) -> dict[str, Any]:
    """Merge a list of per-page analysis results into a single combined report."""
    language = _analysis_language(analysis_language)

    def _page_sort_key(page: dict[str, Any]) -> tuple[int, str]:
        try:
            page_index = int(page.get("page_index"))
        except (TypeError, ValueError):
            page_index = 10**9
        return page_index, str(page.get("page_name") or "")

    # Completion callbacks arrive in API-finish order. Always rebuild the report
    # in source-page order and repair legacy/model-provided vocabulary numbers.
    ordered_pages = sorted(copy.deepcopy(page_analyses), key=_page_sort_key)

    # Inject source page label so we can show which page a word/grammar came from
    for page in ordered_pages:
        page_label = page.get("source_label", "")
        _renumber_rows(page.get("vocabulary_all", []))
        for field in ("vocabulary_important", "grammar_points", "sentence_patterns"):
            for item in page.get(field, []):
                if "page_label" not in item:
                    item["page_label"] = page_label

    merged = _merge_analysis_results(ordered_pages, language)
    _renumber_rows(merged.get("vocabulary_all", []))
    merged["page_analyses"] = ordered_pages
    merged["confirmed_text"] = "\n\n".join(
        f"## {page['source_label']}\n{page['confirmed_text']}" for page in ordered_pages
    )
    merged["summary"] = "\n\n".join(
        f"**{page['source_label']}:** {page['summary']}" for page in ordered_pages
    )
    merged["full_markdown"] = "\n\n---\n\n".join(
        f"# {page['source_label']}\n\n{page['full_markdown']}" for page in ordered_pages
    )
    merged["sentence_analysis_usage"] = aggregate_sentence_usage(ordered_pages)
    merged["sentence_analysis_runs"] = aggregate_sentence_runs(ordered_pages)
    sentence_models = [page.get("sentence_analysis_model") for page in ordered_pages if page.get("sentence_analysis_model")]
    merged["sentence_analysis_model"] = sentence_models[0] if sentence_models else None
    merged["sentence_analysis_errors"] = [
        {"page_index": page.get("page_index"), "error": page.get("sentence_analysis_error")}
        for page in ordered_pages
        if page.get("sentence_analysis_error")
    ]
    merged["model_used"] = ordered_pages[0].get("model_used", "gemini-3.5-flash") if ordered_pages else "gemini-3.5-flash"
    return merged


def prepare_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalise raw page dicts for analysis."""
    prepared = [
        {
            "page_index": int(page.get("page_index", index)),
            "page_name": str(page.get("page_name") or f"Trang {index}"),
            "text": str(page.get("text") or "").strip(),
            "notes": list(page.get("notes") or []),
        }
        for index, page in enumerate(pages, 1)
        if str(page.get("text") or "").strip()
    ]
    if not prepared:
        raise ValueError("Không có trang nào có văn bản OCR để phân tích.")
    return prepared


def run_page_analyses(
    pages: list[dict[str, Any]],
    analysis_language: str = "english",
    progress_callback: Callable[[int, int, str], None] | None = None,
    page_done_callback: Callable[[dict[str, Any]], None] | None = None,
    max_workers: int = 3,
    model_name: str | None = None,
    reasoning_effort: str = "standard",
    auto_sentence_deep_dive: bool = True,
) -> dict[str, Any]:
    """Analyze each OCR page concurrently, then return a merged report with per-page details.

    Args:
        pages: Raw page dicts with text and notes.
        analysis_language: ``"japanese"`` or ``"english"``.
        progress_callback: ``(done, total, page_name)`` called after each page.
        page_done_callback: ``(page_result)`` called after each page finishes so
            callers can persist partial results for resume on interruption.
        max_workers: Maximum number of concurrent API calls.
        model_name: Optional Gemini text model name to use.
        reasoning_effort: ``"standard"`` or ``"deep"``.
        auto_sentence_deep_dive: Analyze up to 3 complex sentences per page and
            15 across the document in a separate, non-blocking Gemini phase.
    """
    language = _analysis_language(analysis_language)
    prepared_pages = prepare_pages(pages)
    model = _init_model(model_name) if model_name else _init_model()
    total = len(prepared_pages)
    catalogs = build_sentence_catalog(prepared_pages, language)
    selected = select_auto_sentences(catalogs) if auto_sentence_deep_dive else {}

    # Main analysis is persisted first. The sentence phase below only enriches it.
    page_analyses: list[dict[str, Any] | None] = [None] * total
    completed = 0

    def _work(index: int, page: dict) -> tuple[int, dict[str, Any]]:
        base = analyze_single_page(model, page, language, reasoning_effort=reasoning_effort)
        return index, attach_sentence_data(base, catalogs.get(int(page["page_index"]), []))

    if total == 1:
        idx, page_result = _work(0, prepared_pages[0])
        page_analyses[idx] = page_result
        completed = 1
        if page_done_callback:
            page_done_callback(page_result)
        if progress_callback:
            progress_callback(1, 1, page_result["page_name"])
    else:
        workers = min(max_workers, total)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_work, idx, page): idx
                for idx, page in enumerate(prepared_pages)
            }
            for future in as_completed(futures):
                idx, page_result = future.result()
                page_analyses[idx] = page_result
                completed += 1
                if page_done_callback:
                    page_done_callback(page_result)
                if progress_callback:
                    progress_callback(completed, total, page_result["page_name"])

    if selected:
        page_lookup = {int(page["page_index"]): (index, page) for index, page in enumerate(prepared_pages)}

        def _deep_work(page_index: int, sentence_rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]], dict[str, int], str | None]:
            _index, source_page = page_lookup[page_index]
            try:
                rows, usage = analyze_sentence_batch(
                    model,
                    sentence_rows,
                    source_page["text"],
                    language,
                    reasoning_effort=reasoning_effort,
                    origin="auto",
                )
                return page_index, rows, usage, None
            except Exception as exc:
                return page_index, [], dict(ZERO_USAGE), str(exc)

        with ThreadPoolExecutor(max_workers=min(max_workers, len(selected))) as executor:
            futures = [
                executor.submit(_deep_work, page_index, sentence_rows)
                for page_index, sentence_rows in selected.items()
            ]
            for future in as_completed(futures):
                page_index, rows, usage, error = future.result()
                result_index, _source_page = page_lookup[page_index]
                current = page_analyses[result_index]
                if current is None:
                    continue
                enriched = attach_sentence_data(
                    current,
                    current.get("sentence_catalog", []),
                    breakdowns=rows,
                    usage=usage,
                    error=error,
                    model_used=getattr(model, "target_model_name", model_name or "gemini-3.5-flash"),
                )
                page_analyses[result_index] = enriched
                if page_done_callback:
                    page_done_callback(enriched)

    return merge_page_analyses([p for p in page_analyses if p is not None], language)
