"""UI helper functions and component formatters for Streamlit views."""

from __future__ import annotations
import streamlit as st

COLUMN_LABELS = {
    "english": {
        "num": "STT",
        "word": "Từ",
        "base_form": "Dạng gốc",
        "part_of_speech": "Loại từ",
        "meaning": "Nghĩa tiếng Việt",
        "cefr": "Cấp độ CEFR",
        "example": "Ví dụ trong bài",
        "difficulty": "Độ khó",
        "phrase": "Từ/Cụm từ",
        "type": "Loại",
        "note": "Ghi chú",
        "function": "Chức năng",
        "register": "Sắc thái",
        "linked_parts": "Hai thành phần được nối",
        "structure": "Cấu trúc / Cách nối",
        "usage": "Vị trí / Dấu câu",
    },
    "japanese": {
        "num": "STT",
        "word": "Từ gốc",
        "reading": "Phiên âm",
        "type": "Loại từ",
        "meaning": "Nghĩa",
        "jlpt": "JLPT",
        "example": "Ví dụ",
        "difficulty": "Độ khó",
        "kanji": "Kanji",
        "onyomi": "Âm On",
        "kunyomi": "Âm Kun",
        "vocab": "Từ vựng trong bài",
        "role": "Vai trò",
        "phrase": "Từ/Cụm",
        "linked_parts": "Hai thành phần được nối",
        "structure": "Cấu trúc / Cách nối",
    },
}


def display_rows(rows: list[dict], language: str) -> list[dict]:
    """Map dictionary keys to localized column headers."""
    labels = COLUMN_LABELS.get(language, {})
    return [{labels.get(key, key): value for key, value in row.items()} for row in rows]


def render_example(label: str, text: str | None, hiragana: str | None = None) -> None:
    """Render an example box with optional Hiragana subtitle."""
    if not text:
        return
    st.markdown(label)
    st.info(text)
    if hiragana:
        st.caption(f"ひらがな: {hiragana}")


def render_important_vocabulary(items: list[dict]) -> None:
    """Render expandable detailed cards for key vocabulary items."""
    if not items:
        st.info("Chưa trích xuất được từ vựng quan trọng.")
        return

    st.subheader("⭐ Từ vựng khó — Giải thích chi tiết")
    for item in items:
        level = item.get("jlpt") or item.get("cefr") or item.get("difficulty") or ""
        label = item.get("word", "")
        if item.get("reading"):
            label = f"{label}・{item['reading']}"
        page_tag = f" — {item['page_label']}" if item.get("page_label") else ""
        with st.expander(f"📖 {label}   {level}{page_tag}".strip()):
            col1, col2 = st.columns(2)
            with col1:
                if item.get("type") or item.get("part_of_speech"):
                    st.markdown(f"**Loại từ:** {item.get('type') or item.get('part_of_speech')}")
                if item.get("meaning"):
                    st.markdown(f"**Ý nghĩa:** {item['meaning']}")
                if item.get("vn_meaning"):
                    st.markdown(f"**Nghĩa tiếng Việt:** {item['vn_meaning']}")
                if item.get("definition"):
                    st.markdown(f"**Definition:** {item['definition']}")
                if item.get("base_form"):
                    st.markdown(f"**Dạng gốc:** {item['base_form']}")
                if item.get("related") and item["related"] not in ("Không có", "None", "N/A"):
                    st.markdown(f"**Từ liên quan / Related:** {item['related']}")
            with col2:
                example_text = item.get("example_text") or item.get("example")
                render_example("**📌 Ví dụ trong bài:**", example_text, item.get("example_text_hiragana"))
                render_example("**✏️ Ví dụ 1:**", item.get("example_1"), item.get("example_1_hiragana"))
                render_example("**✏️ Ví dụ 2:**", item.get("example_2"), item.get("example_2_hiragana"))

            if item.get("note") and item["note"] not in ("Không có", "None", "N/A"):
                st.warning(f"⚠️ **Lưu ý:** {item['note']}")
            if item.get("mistake") and item["mistake"] not in ("Không có", "None", "N/A"):
                st.warning(f"⚠️ **Common mistake:** {item['mistake']}")


def render_grammar_points(items: list[dict]) -> None:
    """Render expandable detailed cards for grammar points."""
    if not items:
        st.info("Chưa trích xuất được mục ngữ pháp riêng. Xem nội dung gốc bên dưới.")
        return

    for point in items:
        level = point.get("level") or ""
        page_tag = f" — {point['page_label']}" if point.get("page_label") else ""
        with st.expander(f"📌 {point.get('name', '')}   {level}{page_tag}".strip()):
            col1, col2 = st.columns(2)
            with col1:
                if point.get("structure"):
                    st.markdown(f"**Công thức / Structure:** `{point['structure']}`")
                if point.get("rule"):
                    st.markdown(f"**Quy tắc:** {point['rule']}")
                if point.get("meaning"):
                    st.markdown(f"**Ý nghĩa:** {point['meaning']}")
                if point.get("usage"):
                    st.markdown(f"**Cách dùng:** {point['usage']}")
                if point.get("formation"):
                    st.markdown(f"**Cấu tạo trong câu:** {point['formation']}")
                if point.get("nuance"):
                    st.markdown(f"**Sắc thái / Văn phong:** {point['nuance']}")
                if point.get("explanation"):
                    st.markdown(f"**Giải thích:** {point['explanation']}")
            with col2:
                render_example("**📌 Ví dụ trong bài:**", point.get("example"), point.get("example_hiragana"))
                if point.get("example_analysis"):
                    st.markdown(f"**Phân tích ví dụ:** {point['example_analysis']}")
                render_example("**✏️ Ví dụ 1:**", point.get("example_1"), point.get("example_1_hiragana"))
                render_example("**✏️ Ví dụ 2:**", point.get("example_2"), point.get("example_2_hiragana"))

            if point.get("note") and point["note"] not in ("Không có", "None", "N/A"):
                st.warning(f"⚠️ **Lưu ý:** {point['note']}")
            if point.get("comparison") and point["comparison"] not in ("Không có", "None", "N/A"):
                st.info(f"**Phân biệt cấu trúc gần nghĩa:** {point['comparison']}")
            if point.get("mistake") and point["mistake"] not in ("Không có", "None", "N/A"):
                st.warning(f"⚠️ **Common mistake:** {point['mistake']}")
