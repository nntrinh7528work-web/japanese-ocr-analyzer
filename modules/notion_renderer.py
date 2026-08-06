"""Structured Notion renderers for lesson and study-item pages."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Iterable


NOTION_LAYOUT_VERSION = "3.0"
TABLE_ROWS_PER_BLOCK = 40

_TECHNICAL_KEYS = {
    "analysis_language",
    "full_markdown",
    "model_used",
    "raw_response",
    "section_markdown",
    "sentence_analysis_model",
    "sentence_analysis_runs",
    "sentence_analysis_usage",
    "translation_guidance_model",
    "translation_guidance_runs",
    "translation_guidance_usage",
}

_LABELS = {
    "base_form": "Từ gốc",
    "cefr": "CEFR",
    "clauses": "Mệnh đề",
    "comparison": "So sánh",
    "complexity_score": "Điểm phức tạp",
    "components": "Thành phần",
    "comprehension_questions": "Câu hỏi hiểu bài",
    "difficulty": "Độ khó",
    "example": "Ví dụ",
    "example_1": "Ví dụ 1",
    "example_1_hiragana": "Hiragana ví dụ 1",
    "example_2": "Ví dụ 2",
    "example_2_hiragana": "Hiragana ví dụ 2",
    "example_analysis": "Phân tích ví dụ",
    "example_hiragana": "Hiragana ví dụ",
    "example_reading": "Cách đọc ví dụ",
    "example_translation": "Dịch ví dụ",
    "explanation": "Giải thích",
    "explanation_vi": "Giải thích tiếng Việt",
    "formation": "Cách thành lập",
    "function": "Chức năng",
    "hiragana": "Hiragana",
    "jlpt": "JLPT",
    "key_points": "Điểm mấu chốt",
    "kunyomi": "Kunyomi",
    "linked_parts": "Phần được liên kết",
    "logic": "Luồng logic",
    "meaning": "Nghĩa tiếng Việt",
    "meaning_vi": "Nghĩa tiếng Việt",
    "mistake": "Lỗi thường gặp",
    "natural": "Dịch tự nhiên",
    "note": "Ghi chú",
    "nuance": "Sắc thái",
    "ocr_warning": "Cảnh báo OCR",
    "omitted_elements": "Thành phần lược bỏ",
    "onyomi": "Onyomi",
    "original": "Nguyên văn",
    "part_of_speech": "Từ loại",
    "reading": "Cách đọc / Hiragana",
    "references": "Từ quy chiếu",
    "register": "Văn phong",
    "related": "Từ liên quan",
    "related_analysis": "Phân tích liên quan",
    "role": "Vai trò",
    "rule": "Quy tắc",
    "segments": "Cụm từ",
    "simplified_source": "Câu viết lại đơn giản",
    "simplified_vi": "Dịch câu đơn giản",
    "structure": "Cấu trúc",
    "structure_summary": "Tóm tắt cấu trúc",
    "translation": "Bản dịch",
    "translation_steps": "Thứ tự dịch đề xuất",
    "translations": "Các bản dịch",
    "type": "Loại",
    "usage": "Cách dùng",
    "vocab": "Từ vựng liên quan",
}


def _has_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def _is_technical_key(key: str) -> bool:
    return key in _TECHNICAL_KEYS or key.endswith(("_usage", "_runs", "_model"))


def _label(key: str) -> str:
    return _LABELS.get(key, key.replace("_", " ").strip().capitalize())


def _escape(value: Any) -> str:
    text = str(value or "").strip()
    text = text.replace("\\", "\\\\")
    for char in ("*", "~", "`", "$", "[", "]", "<", ">", "{", "}", "|", "^"):
        text = text.replace(char, "\\" + char)
    return text.replace("\r\n", "<br>").replace("\n", "<br>").replace("\r", "<br>")


def _plain(value: Any) -> str:
    if isinstance(value, bool):
        return "Có" if value else "Không"
    if isinstance(value, (str, int, float)):
        return _escape(value)
    if isinstance(value, list):
        simple = all(not isinstance(item, (dict, list)) for item in value)
        if simple:
            return "; ".join(_escape(item) for item in value if _has_value(item))
    return _escape(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _indent(content: str) -> str:
    return "\n".join("\t" + line for line in content.splitlines())


def _toggle(title: str, content: str, *, color: str = "gray_bg") -> str:
    body = content.strip() or "Không có dữ liệu."
    return (
        f'<details color="{color}">\n'
        f"<summary>{_escape(title)}</summary>\n"
        f"{_indent(body)}\n"
        "</details>"
    )


def _callout(content: str, *, icon: str = "ℹ️", color: str = "blue_bg") -> str:
    return f'<callout icon="{icon}" color="{color}">\n{_indent(content.strip())}\n</callout>'


def _text_chunks(value: Any, max_chars: int = 40_000) -> list[str]:
    text = str(value or "")
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current = ""
    for paragraph in re.split(r"(?<=\n)", text):
        while len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        if current and len(current) + len(paragraph) > max_chars:
            chunks.append(current)
            current = ""
        current += paragraph
    if current:
        chunks.append(current)
    return chunks


def _leaf_paths(value: Any, path: str) -> set[str]:
    if isinstance(value, dict):
        result: set[str] = set()
        for key, item in value.items():
            if _is_technical_key(str(key)) or not _has_value(item):
                continue
            result.update(_leaf_paths(item, f"{path}.{key}" if path else str(key)))
        return result
    if isinstance(value, list):
        result = set()
        for index, item in enumerate(value):
            if _has_value(item):
                result.update(_leaf_paths(item, f"{path}[{index}]"))
        return result
    return {path} if _has_value(value) else set()


@dataclass
class _Coverage:
    expected: set[str]
    rendered: set[str] = field(default_factory=set)

    def mark(self, value: Any, path: str) -> None:
        self.rendered.update(_leaf_paths(value, path))

    @property
    def missing(self) -> list[str]:
        return sorted(self.expected - self.rendered)


def _field_lines(
    row: dict[str, Any],
    path: str,
    coverage: _Coverage,
    *,
    exclude: Iterable[str] = (),
) -> list[str]:
    excluded = set(exclude)
    lines: list[str] = []
    for key, value in row.items():
        if key in excluded or _is_technical_key(key) or not _has_value(value):
            continue
        coverage.mark(value, f"{path}.{key}")
        if isinstance(value, dict):
            lines.append(f"**{_escape(_label(key))}:**")
            for child_key, child_value in value.items():
                if _has_value(child_value):
                    lines.append(f"- **{_escape(_label(str(child_key)))}:** {_plain(child_value)}")
        elif isinstance(value, list) and any(isinstance(item, dict) for item in value):
            lines.append(f"**{_escape(_label(key))}:**")
            for index, item in enumerate(value, 1):
                lines.append(f"{index}. {_plain(item)}")
        else:
            lines.append(f"- **{_escape(_label(key))}:** {_plain(value)}")
    return lines


def _table(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str]],
    base_path: str,
    coverage: _Coverage,
) -> list[str]:
    tables: list[str] = []
    for start in range(0, len(rows), TABLE_ROWS_PER_BLOCK):
        batch = rows[start:start + TABLE_ROWS_PER_BLOCK]
        lines = ['<table fit-page-width="true" header-row="true">', "\t<tr>"]
        lines.extend(f"\t\t<td>{_escape(label)}</td>" for _, label in columns)
        lines.append("\t</tr>")
        for offset, row in enumerate(batch, start):
            lines.append("\t<tr>")
            for key, _ in columns:
                value = row.get(key)
                if _has_value(value):
                    coverage.mark(value, f"{base_path}[{offset}].{key}")
                lines.append(f"\t\t<td>{_plain(value)}</td>")
            lines.append("\t</tr>")
        lines.append("</table>")
        tables.append("\n".join(lines))
    return tables


def _render_collection(
    title: str,
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str]],
    base_path: str,
    coverage: _Coverage,
    *,
    title_keys: tuple[str, ...],
) -> list[str]:
    if not rows:
        return []
    sections = [f"### {title} ({len(rows)})"]
    sections.extend(_table(rows, columns, base_path, coverage))
    column_keys = {key for key, _ in columns}
    details: list[str] = []
    for index, row in enumerate(rows):
        path = f"{base_path}[{index}]"
        extra = _field_lines(row, path, coverage, exclude=column_keys)
        if not extra:
            continue
        item_title = next((str(row.get(key) or "") for key in title_keys if row.get(key)), f"Mục {index + 1}")
        details.append(_toggle(f"Chi tiết: {item_title}", "\n".join(extra)))
    if details:
        sections.extend(details)
    return sections


def _sentence_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    try:
        ordinal = int(row.get("ordinal") or 0)
    except (TypeError, ValueError):
        ordinal = 0
    return ordinal, str(row.get("sentence_id") or "")


def _key_point_text(point: Any) -> str:
    if not isinstance(point, dict):
        return _plain(point)
    source = point.get("source") or point.get("label") or point.get("term")
    explanation = point.get("explanation_vi") or point.get("explanation") or point.get("meaning")
    if source and explanation:
        return f"**{_plain(source)}:** {_plain(explanation)}"
    return _plain(point)


def _render_sentence_cards(page: dict[str, Any], page_path: str, coverage: _Coverage) -> list[str]:
    catalog = {str(row.get("sentence_id") or ""): row for row in page.get("sentence_catalog") or []}
    guidance = {str(row.get("sentence_id") or ""): row for row in page.get("translation_guidance") or []}
    breakdowns = {str(row.get("sentence_id") or ""): row for row in page.get("sentence_breakdowns") or []}
    sentence_ids = sorted(set(catalog) | set(guidance) | set(breakdowns), key=lambda sid: _sentence_sort_key(catalog.get(sid) or guidance.get(sid) or breakdowns.get(sid) or {}))
    sections: list[str] = []
    for fallback_index, sentence_id in enumerate(sentence_ids, 1):
        cat = catalog.get(sentence_id) or {}
        guide = guidance.get(sentence_id) or {}
        deep = breakdowns.get(sentence_id) or {}
        ordinal = cat.get("ordinal") or guide.get("ordinal") or deep.get("ordinal") or fallback_index
        original = guide.get("original") or cat.get("original") or deep.get("original") or ""
        reading = guide.get("reading") or deep.get("reading") or ""
        translations = guide.get("translations") or {}
        natural = translations.get("natural") or (deep.get("translations") or {}).get("natural") or deep.get("simplified_vi") or ""
        lines = [f"### Câu {ordinal} · `{_escape(sentence_id or f's{ordinal}')}`"]
        if original:
            lines.append(f"**Nguyên văn:** {_plain(original)}")
        if reading:
            lines.append(f"**Hiragana / Cách đọc:** {_plain(reading)}")
        if natural:
            lines.append(f"**Dịch tự nhiên:** {_plain(natural)}")
        key_points = guide.get("key_points") or []
        if key_points:
            lines.append("**Điểm mấu chốt:**")
            for point in key_points:
                lines.append(f"- {_key_point_text(point)}")
        if deep:
            lines.append("**Trạng thái:** Có giải mã câu dài")

        cat_path = f"{page_path}.sentence_catalog[{list(catalog).index(sentence_id)}]" if sentence_id in catalog else ""
        guide_path = f"{page_path}.translation_guidance[{list(guidance).index(sentence_id)}]" if sentence_id in guidance else ""
        deep_path = f"{page_path}.sentence_breakdowns[{list(breakdowns).index(sentence_id)}]" if sentence_id in breakdowns else ""
        if cat_path:
            coverage.mark(cat, cat_path)
        if guide_path:
            coverage.mark(guide, guide_path)
        if deep_path:
            coverage.mark(deep, deep_path)

        detail_lines: list[str] = []
        if guide:
            detail_lines.extend(_field_lines(
                guide,
                guide_path,
                coverage,
                exclude=("sentence_id", "ordinal", "original", "reading", "key_points"),
            ))
        if cat:
            detail_lines.extend(_field_lines(cat, cat_path, coverage, exclude=("sentence_id", "ordinal", "original")))
        if deep:
            deep_lines = _field_lines(
                deep,
                deep_path,
                coverage,
                exclude=("sentence_id", "ordinal", "original", "reading"),
            )
            detail_lines.append(_toggle("Tám lớp giải mã câu dài", "\n".join(deep_lines), color="purple_bg"))
        if detail_lines:
            lines.append(_toggle("Dịch theo cụm, dịch sát và phân tích chi tiết", "\n".join(detail_lines)))
        sections.append("\n\n".join(lines))
    return sections


def _page_expected(page: dict[str, Any], page_path: str) -> set[str]:
    expected = _leaf_paths(page, page_path)
    # These are aliases or generated presentation fragments, not independent analysis content.
    for key in ("confirmed_text", "page_name", "usage"):
        expected.difference_update(_leaf_paths(page.get(key), f"{page_path}.{key}"))
    return expected


def render_notion_lesson_markdown(
    items: list[dict[str, Any]],
    analysis: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Render every structured user-facing analysis field into Notion Markdown."""
    pages = analysis.get("page_analyses") or [analysis]
    expected: set[str] = set()
    for index, page in enumerate(pages):
        expected.update(_page_expected(page, f"page_analyses[{index}]"))
    coverage = _Coverage(expected)
    sections: list[str] = [
        f"# {_escape(metadata.get('title') or 'Bài phân tích')}",
        _callout(
            "\n".join(
                [
                    f"**Ngôn ngữ:** {'Tiếng Nhật' if metadata.get('language') == 'japanese' else 'Tiếng Anh'}",
                    f"**Nguồn:** {_plain(', '.join(metadata.get('source_names') or []) or 'Không rõ')}",
                    f"**Số trang:** {metadata.get('analyzed_page_count', 0)}/{metadata.get('page_count', 0)}",
                    f"**Trạng thái:** {_plain(metadata.get('sync_status') or 'Hoàn tất')}",
                    f"**Model:** {_plain(metadata.get('model') or 'Không rõ')}",
                    f"**Tổng token:** {int(metadata.get('total_tokens') or 0):,}",
                    f"**Chi phí ước tính:** ¥{float(metadata.get('cost_jpy') or 0):,.4f}",
                ]
            ),
            icon="📘",
        ),
        "<table_of_contents/>",
        "## Đối chiếu OCR và giáo viên hướng dẫn dịch",
    ]
    if metadata.get("sync_status") == "Một phần":
        missing_pages = ", ".join(str(value) for value in metadata.get("missing_page_indices") or [])
        sections.append(_callout(
            "Đồng bộ một phần. "
            f"Trang chưa có kết quả: {missing_pages or 'không xác định'}. "
            "Notion chỉ hiển thị dữ liệu thực sự đã có, không tự tạo nội dung còn thiếu.",
            icon="⚠️",
            color="orange_bg",
        ))

    category_sections: list[str] = ["## Nội dung học theo từng trang"]
    language = str(metadata.get("language") or analysis.get("analysis_language") or "japanese")
    handled_keys = {
        "page_index", "page_name", "source_label", "source_text", "confirmed_text", "summary",
        "sentence_catalog", "translation_guidance", "sentence_breakdowns", "vocabulary_all",
        "vocabulary_important", "kanji_analysis", "phrasal_collocations", "connectors",
        "discourse_markers", "grammar_points", "sentence_patterns", "ocr_corrections",
        "translation_guidance_errors", "sentence_analysis_error", "sentence_analysis_errors",
        "usage",
    }

    for fallback_index, page in enumerate(pages, 1):
        page_path = f"page_analyses[{fallback_index - 1}]"
        page_index = int(page.get("page_index", fallback_index) or fallback_index)
        page_label = page.get("source_label") or page.get("page_name") or f"Trang {page_index}"
        coverage.mark(page.get("page_index"), f"{page_path}.page_index")
        coverage.mark(page.get("source_label"), f"{page_path}.source_label")
        sections.append(f"## Trang {page_index}: {_escape(page_label)}")
        if page.get("summary"):
            coverage.mark(page["summary"], f"{page_path}.summary")
            sections.append(_callout(f"**Tóm tắt:** {_plain(page['summary'])}", icon="📝", color="yellow_bg"))
        source_text = page.get("source_text") or page.get("confirmed_text")
        if source_text:
            coverage.mark(page.get("source_text"), f"{page_path}.source_text")
            source_chunks = _text_chunks(source_text)
            for chunk_index, chunk in enumerate(source_chunks, 1):
                suffix = f" · phần {chunk_index}/{len(source_chunks)}" if len(source_chunks) > 1 else ""
                sections.append(_toggle(
                    "OCR gốc đã được duyệt" + suffix,
                    f"> {_plain(chunk)}",
                    color="blue_bg",
                ))
        sentence_cards = _render_sentence_cards(page, page_path, coverage)
        if sentence_cards:
            sections.extend(sentence_cards)
        else:
            sections.append("> Chưa có hướng dẫn dịch từng câu cho trang này.")

        category_sections.append(f"## Trang {page_index}: {_escape(page_label)}")
        vocabulary = list(page.get("vocabulary_all") or [])
        category_sections.extend(_render_collection(
            "Từ vựng", vocabulary,
            [("num", "STT"), ("word", "Từ"), ("reading", "Cách đọc"), ("type" if language == "japanese" else "part_of_speech", "Từ loại"), ("meaning", "Nghĩa"), ("jlpt" if language == "japanese" else "cefr", "Mức độ")],
            f"{page_path}.vocabulary_all", coverage, title_keys=("word", "phrase"),
        ))
        important = list(page.get("vocabulary_important") or [])
        category_sections.extend(_render_collection(
            "Từ vựng khó", important,
            [("word", "Từ"), ("reading", "Cách đọc"), ("meaning", "Nghĩa"), ("example", "Ví dụ"), ("example_hiragana", "Hiragana ví dụ")],
            f"{page_path}.vocabulary_important", coverage, title_keys=("word", "phrase"),
        ))
        script_key = "kanji_analysis" if language == "japanese" else "phrasal_collocations"
        script_rows = list(page.get(script_key) or [])
        script_columns = (
            [("kanji", "Kanji"), ("onyomi", "Onyomi"), ("kunyomi", "Kunyomi"), ("meaning", "Nghĩa"), ("jlpt", "JLPT")]
            if language == "japanese"
            else [("phrase", "Cụm từ"), ("type", "Loại"), ("meaning", "Nghĩa"), ("example", "Ví dụ"), ("note", "Ghi chú")]
        )
        category_sections.extend(_render_collection(
            "Kanji" if language == "japanese" else "Cụm từ và collocation",
            script_rows, script_columns, f"{page_path}.{script_key}", coverage,
            title_keys=("kanji", "phrase", "word"),
        ))
        marker_key = "connectors" if language == "japanese" else "discourse_markers"
        markers = list(page.get(marker_key) or [])
        category_sections.extend(_render_collection(
            "Từ nối" if language == "japanese" else "Discourse markers", markers,
            [("phrase", "Từ / Cụm"), ("reading", "Cách đọc"), ("type", "Loại"), ("meaning", "Nghĩa"), ("function", "Chức năng"), ("example", "Ví dụ")],
            f"{page_path}.{marker_key}", coverage, title_keys=("phrase", "marker", "word"),
        ))
        grammar = list(page.get("grammar_points") or [])
        category_sections.extend(_render_collection(
            "Ngữ pháp", grammar,
            [("name", "Ngữ pháp"), ("structure", "Cấu trúc"), ("meaning", "Nghĩa"), ("jlpt" if language == "japanese" else "cefr", "Mức độ")],
            f"{page_path}.grammar_points", coverage, title_keys=("name", "pattern"),
        ))
        patterns = list(page.get("sentence_patterns") or [])
        category_sections.extend(_render_collection(
            "Mẫu câu", patterns,
            [("pattern", "Mẫu câu"), ("components", "Thành phần"), ("function", "Chức năng"), ("example", "Ví dụ")],
            f"{page_path}.sentence_patterns", coverage, title_keys=("pattern", "name"),
        ))
        breakdowns = list(page.get("sentence_breakdowns") or [])
        if breakdowns:
            index_rows = []
            for row in breakdowns:
                index_rows.append({
                    "sentence_id": row.get("sentence_id"),
                    "original": row.get("original"),
                    "natural": (row.get("translations") or {}).get("natural") or row.get("simplified_vi"),
                    "complexity_score": row.get("complexity_score"),
                })
            dummy = _Coverage(set())
            category_sections.append(f"### Giải mã câu dài ({len(index_rows)})")
            category_sections.extend(_table(index_rows, [("sentence_id", "ID"), ("original", "Nguyên văn"), ("natural", "Dịch tự nhiên"), ("complexity_score", "Điểm")], "long_sentence_index", dummy))

        warnings: list[str] = []
        for key in ("ocr_corrections", "translation_guidance_errors", "sentence_analysis_errors"):
            value = page.get(key)
            if _has_value(value):
                coverage.mark(value, f"{page_path}.{key}")
                warnings.append(f"- **{_escape(_label(key))}:** {_plain(value)}")
        if page.get("sentence_analysis_error"):
            coverage.mark(page["sentence_analysis_error"], f"{page_path}.sentence_analysis_error")
            warnings.append(f"- **Lỗi giải mã câu dài:** {_plain(page['sentence_analysis_error'])}")
        if warnings:
            category_sections.append(_toggle("Cảnh báo và lỗi không chặn kết quả", "\n".join(warnings), color="red_bg"))

        extra = {
            key: value for key, value in page.items()
            if key not in handled_keys and not _is_technical_key(key) and _has_value(value)
        }
        if extra:
            extra_lines = _field_lines(extra, page_path, coverage)
            category_sections.append(_toggle("Dữ liệu bổ sung", "\n".join(extra_lines), color="gray_bg"))

    sections.extend(category_sections)
    missing = coverage.missing
    if missing:
        sections.append(_toggle(
            "Kiểm tra dữ liệu chưa biểu diễn",
            "\n".join(f"- `{_escape(path)}`" for path in missing),
            color="red_bg",
        ))
    sections.append("## Thông tin kỹ thuật")
    sections.append(
        "\n".join(
            [
                f"- **OCR hash:** `{_escape(metadata.get('source_hash'))}`",
                f"- **Analysis hash:** `{_escape(metadata.get('analysis_hash'))}`",
                f"- **Phiên bản bố cục:** `{NOTION_LAYOUT_VERSION}`",
                f"- **Số trường chưa hiển thị:** {len(missing)}",
                f"- **Mở lại trên ứng dụng:** [Mở bài phân tích]({_escape(metadata.get('app_url'))})" if metadata.get("app_url") else "",
            ]
        ).strip()
    )
    return {
        "markdown": "\n\n".join(section for section in sections if section),
        "sections": [section for section in sections if section],
        "layout_version": NOTION_LAYOUT_VERSION,
        "rendered_field_paths": sorted(coverage.rendered & coverage.expected),
        "unrendered_field_paths": missing,
        "unrendered_field_count": len(missing),
        "coverage_complete": not missing,
        "section_manifest": [
            re.sub(r"^#+\s*", "", section.splitlines()[0]).strip()
            for section in sections if section.startswith("#")
        ],
    }


def render_notion_item_markdown(item: dict[str, Any]) -> str:
    """Render the complete source record for one study item page."""
    try:
        source = json.loads(str(item.get("source_json") or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        source = {"Dữ liệu nguồn": item.get("source_json")}
    coverage = _Coverage(_leaf_paths(source, "source"))
    details = _field_lines(source if isinstance(source, dict) else {"value": source}, "source", coverage)
    header = [
        f"# {_escape(item.get('title') or 'Mục cần học')}",
        _callout(
            "\n".join(
                [
                    f"**Loại:** {_plain(item.get('type'))}",
                    f"**Trang:** {int(item.get('page_index') or 0)}",
                    f"**Thứ tự nguồn:** {int(item.get('source_order') or 0)}",
                    f"**Cách đọc:** {_plain(item.get('reading'))}",
                    f"**Nghĩa tiếng Việt:** {_plain(item.get('meaning_vi'))}",
                ]
            ),
            icon="📚",
            color="green_bg",
        ),
        "## Nội dung đầy đủ",
        "\n".join(details) or "Không có dữ liệu chi tiết.",
        "## Thông tin nguồn",
        f"- **ID câu:** `{_escape(item.get('sentence_id'))}`\n- **JSON checksum:** `{_escape(item.get('source_checksum'))}`",
    ]
    return "\n\n".join(header)
