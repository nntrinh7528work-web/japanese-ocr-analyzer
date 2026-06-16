import json
from datetime import datetime

from modules.result_exporter import analysis_json_bytes, default_export_stem, markdown_bytes, safe_export_stem


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
        25500,
        datetime(2026, 6, 16, 9, 30, 5),
    )
    payload = json.loads(data.decode("utf-8"))

    assert markdown_bytes(analysis).startswith("# Báo cáo".encode("utf-8"))
    assert payload["sources"][0]["ocr_text"] == "日本語"
    assert "original_image_bytes" not in payload["sources"][0]
    assert payload["analysis"]["summary"] == "Tóm tắt"
