import io

from PIL import Image
from streamlit.testing.v1 import AppTest

from modules.multi_image_workflow import create_image_item


def _image_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (100, 60), color).save(buffer, "PNG")
    return buffer.getvalue()


def test_app_starts_without_upload():
    app = AppTest.from_file("app.py").run(timeout=20)
    assert not app.exception
    # Branded header is rendered via st.markdown, not st.title
    assert any("Japanese / English OCR Analyzer" in m.value for m in app.markdown)
    assert any("một hoặc nhiều ảnh/PDF" in item.value for item in app.info)
    assert len(app.tabs) == 4


def test_app_can_switch_dark_mode_without_error():
    app = AppTest.from_file("app.py").run(timeout=20)

    dark_toggle = next(toggle for toggle in app.toggle if "Dark Mode" in toggle.label)
    dark_toggle.set_value(True).run(timeout=20)

    assert not app.exception
    assert app.session_state["dark_mode"] is True
    assert any("#0B0E17" in item.value for item in app.markdown)


def test_app_renders_two_independent_image_flows():
    app = AppTest.from_file("app.py")
    app.session_state["image_items"] = [
        create_image_item(_image_bytes("white"), "page-1.png"),
        create_image_item(_image_bytes("black"), "page-2.png"),
    ]
    app.session_state["analysis"] = None
    app.session_state["upload_messages"] = []
    app.session_state["upload_errors"] = []
    app.session_state["uploader_version"] = 0
    app.session_state["camera_version"] = 0
    app.run(timeout=20)

    assert not app.exception
    labels = [button.label for button in app.button]
    assert labels.count("🔍 OCR ảnh này") == 2
    assert "🔍 OCR tất cả ảnh chưa xử lý" in labels
    assert "🔁 OCR/OCR lại toàn bộ ảnh" in labels
    assert "Ảnh/trang PDF trong bộ phân tích (2)" in [heading.value for heading in app.subheader]


def test_app_renders_analysis_download_options():
    item = create_image_item(_image_bytes("white"), "page-1.png")
    item["ocr_result"] = {
        "clean_text": "日本語",
        "ocr_notes": [],
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "confidence": "high",
        "text_direction": "horizontal",
        "has_furigana": False,
    }
    item["edited_text"] = "日本語"

    app = AppTest.from_file("app.py")
    app.session_state["image_items"] = [item]
    app.session_state["analysis"] = {
        "confirmed_text": "日本語",
        "summary": "Tóm tắt",
        "analysis_language": "japanese",
        "vocabulary_all": [{"num": "1", "word": "変更", "reading": "へんこう", "meaning": "thay đổi"}],
        "vocabulary_important": [],
        "phrasal_collocations": [],
        "discourse_markers": [],
        "kanji_analysis": [],
        "connectors": [
            {
                "phrase": "ので",
                "reading": "ので",
                "type": "接続助詞",
                "structure": "V thể thường + ので",
                "meaning": "vì",
                "example": "雨なので、予定を変更した。",
                "linked_parts": "nguyên nhân → kết quả",
                "role": "lý do khách quan",
                "difficulty": "N4",
            }
        ],
        "grammar_points": [
            {
                "name": "～ので",
                "formation": "雨だ + ので → 雨なので",
                "nuance": "lý do khách quan, mềm",
                "comparison": "mềm hơn ～から",
            }
        ],
        "sentence_patterns": [
            {
                "pattern": "理由 + ので + 結果",
                "components": "mệnh đề nguyên nhân + mệnh đề kết quả",
                "function": "giải thích lý do",
            }
        ],
        "section_markdown": {},
        "full_markdown": "# Báo cáo\n日本語",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    app.session_state["upload_messages"] = []
    app.session_state["upload_errors"] = []
    app.session_state["uploader_version"] = 0
    app.session_state["camera_version"] = 0
    app.run(timeout=20)

    assert not app.exception
    assert "Tên file lưu:" in [field.label for field in app.text_input]
    assert "💾 Lưu kết quả phân tích" in [expander.label for expander in app.get("expander")]
    assert any("JSON lưu dữ liệu có cấu trúc" in caption.value for caption in app.caption)
    markdown_values = [item.value for item in app.markdown]
    assert any("Cấu tạo trong câu" in value for value in markdown_values)
    assert any("Sắc thái / Văn phong" in value for value in markdown_values)
    assert any("Phân biệt cấu trúc gần nghĩa" in item.value for item in app.info)
    assert any("Chức năng giao tiếp" in value for value in markdown_values)
