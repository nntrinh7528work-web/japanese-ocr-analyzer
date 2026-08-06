import json
import io
import zipfile
from datetime import datetime

from modules.result_exporter import analysis_json_bytes, default_export_stem, markdown_bytes, safe_export_stem
from modules.doc_exporter import export_to_docx


def test_safe_export_stem_and_default_name():
    assert safe_export_stem(" bài học 1 / 日本語 ") == "b_i_h_c_1"
    assert default_export_stem(
        [{"name": "lesson page.jpg"}],
        datetime(2026, 6, 16, 9, 30, 5),
    ) == "lesson_page_1_items_20260616_093005"


def test_markdown_and_json_exports_are_readable():
    analysis = {
        "summary": "Tóm tắt",
        "full_markdown": "# Báo cáo\n日本語",
        "vocabulary_all": [{"word": "日本"}],
    }
    items = [
        {
            "name": "page-1.jpg",
            "edited_text": "日本語",
            "ocr_result": {"ocr_notes": ["note"], "confidence": "high", "text_direction": "horizontal"},
            "original_image_bytes": b"not exported",
        }
    ]
    data = analysis_json_bytes(
        items,
        analysis,
        {"total_cost_usd": 0.01},
        "paid",
        155,
        {"remaining_jpy": 99_845},
        datetime(2026, 6, 16, 9, 30, 5),
    )
    payload = json.loads(data.decode("utf-8"))

    assert markdown_bytes(analysis).startswith("# Báo cáo".encode("utf-8"))
    assert payload["sources"][0]["ocr_text"] == "日本語"
    assert "original_image_bytes" not in payload["sources"][0]
    assert payload["analysis"]["summary"] == "Tóm tắt"
    assert payload["usd_to_jpy"] == 155
    assert payload["budget"]["remaining_jpy"] == 99_845


def test_markdown_and_json_include_structured_sentence_breakdowns():
    page = {
        "source_label": "Trang 1: p1",
        "full_markdown": "# Chính",
        "sentence_catalog": [{"sentence_id": "p1-s1"}],
        "sentence_breakdowns": [
            {
                "sentence_id": "p1-s1", "ordinal": 1, "original": "長い文です。",
                "reading": "ながいぶんです。", "analysis_origin": "auto",
                "segments": [], "clauses": [], "translations": {"natural": "Đây là câu dài."},
                "omitted_elements": [], "references": [], "logic": [], "questions": [],
            }
        ],
    }
    analysis = {"page_analyses": [page], "full_markdown": "legacy"}
    md = markdown_bytes(analysis).decode("utf-8")
    payload = json.loads(
        analysis_json_bytes([], analysis, {}, "paid", 155).decode("utf-8")
    )

    assert md.count("Giải mã câu dài") == 1
    assert "ながいぶんです" in md
    assert payload["analysis"]["page_analyses"][0]["sentence_catalog"][0]["sentence_id"] == "p1-s1"

    docx = export_to_docx(md, "sentences.docx")
    with zipfile.ZipFile(io.BytesIO(docx)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "Giải mã câu dài" in document_xml
    assert "ながいぶんです" in document_xml


def test_unified_guidance_export_contains_ocr_teacher_and_deep_once():
    page = {
        "source_label": "Trang 1: guide",
        "full_markdown": "# Phân tích chính",
        "sentence_catalog": [{"sentence_id": "p1-s1", "ordinal": 1, "original": "長い文です。"}],
        "translation_guidance": [
            {
                "sentence_id": "p1-s1", "ordinal": 1, "original": "長い文です。",
                "reading": "ながいぶんです。",
                "translations": {"chunked": "Câu / dài", "literal": "Là câu dài.", "natural": "Đây là một câu dài."},
                "key_points": [{"label": "Chủ đề", "source": "文", "explanation_vi": "Danh từ trung tâm."}],
                "translation_steps": [], "related_analysis": [], "ocr_warning": "",
            }
        ],
        "sentence_breakdowns": [
            {
                "sentence_id": "p1-s1", "segments": [], "clauses": [],
                "structure_summary": "Chủ đề + vị ngữ", "omitted_elements": [],
                "references": [], "logic": [], "questions": [],
            }
        ],
    }
    analysis = {"page_analyses": [page]}
    md = markdown_bytes(analysis).decode("utf-8")
    payload = json.loads(analysis_json_bytes([], analysis, {}, "paid", 155).decode("utf-8"))

    assert md.count("Đối chiếu OCR và giáo viên hướng dẫn dịch") == 1
    assert md.count("Giải mã câu dài") == 1
    assert md.count("長い文です。") == 1
    assert payload["analysis"]["page_analyses"][0]["translation_guidance"][0]["translations"]["natural"]

    docx = export_to_docx(md, "teacher-guidance.docx")
    with zipfile.ZipFile(io.BytesIO(docx)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    assert "giáo viên hướng dẫn dịch" in xml
    assert "ながいぶんです" in xml
