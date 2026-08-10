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


def test_app_blocks_analysis_until_every_uploaded_page_has_ocr_text():
    ready = create_image_item(_image_bytes("white"), "page-1.png")
    ready["ocr_result"] = {
        "clean_text": "一ページ目。",
        "ocr_notes": [],
        "usage": {},
        "text_direction": "horizontal",
        "has_furigana": False,
        "confidence": "high",
    }
    ready["edited_text"] = "一ページ目。"
    pending = create_image_item(_image_bytes("black"), "page-2.png")

    app = AppTest.from_file("app.py")
    app.session_state["image_items"] = [ready, pending]
    app.session_state["analysis"] = None
    app.session_state["upload_messages"] = []
    app.session_state["upload_errors"] = []
    app.session_state["uploader_version"] = 0
    app.session_state["camera_version"] = 0
    app.run(timeout=20)

    analyze = next(
        button for button in app.button
        if button.label == "📚 Phân tích toàn bộ từ vựng, Kanji và ngữ pháp"
    )
    assert analyze.disabled is True
    sentence_analyze = next(
        button for button in app.button
        if button.label == "🧩 Dịch và phân tích từng câu/đoạn dài"
    )
    assert sentence_analyze.disabled is True
    assert any("còn trang chưa có OCR" in message.value for message in app.error)


def test_app_offers_two_separate_token_saving_analysis_buttons():
    ready = create_image_item(_image_bytes("white"), "page-1.png")
    ready["ocr_result"] = {
        "clean_text": "雨が降っています。",
        "ocr_notes": [],
        "usage": {},
        "text_direction": "horizontal",
        "has_furigana": False,
        "confidence": "high",
    }
    ready["edited_text"] = ready["ocr_result"]["clean_text"]
    app = AppTest.from_file("app.py")
    app.session_state["image_items"] = [ready]
    app.run(timeout=20)

    labels = [button.label for button in app.button]
    assert "📚 Phân tích toàn bộ từ vựng, Kanji và ngữ pháp" in labels
    assert "🧩 Dịch và phân tích từng câu/đoạn dài" in labels
    assert "Nội dung phân tích" not in [radio.label for radio in app.radio]
    assert any("Hai nút gọi hai pipeline riêng" in caption.value for caption in app.caption)


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


def test_app_renders_sentence_card_hiragana_and_hidden_answers():
    item = create_image_item(_image_bytes("white"), "page-1.png")
    item["ocr_result"] = {
        "clean_text": "雨が降っているので、出かけません。",
        "ocr_notes": [],
        "usage": {},
        "confidence": "high",
        "text_direction": "horizontal",
        "has_furigana": False,
    }
    item["edited_text"] = item["ocr_result"]["clean_text"]
    breakdown = {
        "sentence_id": "p1-s1",
        "ordinal": 1,
        "original": item["edited_text"],
        "reading": "あめがふっているので、でかけません。",
        "analysis_origin": "auto",
        "sentence_breakdown_version": "2.0",
        "detected_language": "japanese",
        "quality_status": "complete",
        "quality_score": 100,
        "missing_fields": [],
        "complexity_score": 7,
        "segments": [{"text": "雨が", "reading": "あめが", "role": "S", "meaning_vi": "trời mưa", "base_form": "雨", "grammar_form": "chủ ngữ", "particle_or_connector": "が"}],
        "clauses": [{"label": "Mệnh đề phụ", "text": "雨が降っているので", "role": "nguyên nhân"}],
        "structure_summary": "Mệnh đề nguyên nhân + mệnh đề chính",
        "sentence_skeleton": {"pattern": "Mệnh đề lý do + mệnh đề chính", "predicate": "出かけません", "subject": "(tôi)"},
        "grammar_links": [{"source": "ので", "form": "từ nối", "function_vi": "nêu lý do", "nuance_vi": "trung tính"}],
        "connectors": [{"source": "ので", "function_vi": "nối nguyên nhân", "relation": "lý do-kết quả"}],
        "translations": {"chunked": "Vì trời mưa / không đi ra ngoài", "literal": "Vì trời đang mưa, không ra ngoài.", "natural": "Vì trời mưa nên tôi không ra ngoài."},
        "omitted_elements": [], "references": [], "logic": [],
        "translation_steps": [{"order": "1", "source_chunk": "雨が降っている", "meaning_vi": "trời đang mưa", "advice_vi": "đọc vế lý do trước"}],
        "ambiguities": [{"source": "主語", "alternatives": "người nói", "explanation_vi": "chủ ngữ lược bỏ", "confidence": "cao"}],
        "simplified_source": "雨です。出かけません。", "simplified_vi": "Trời mưa. Tôi không ra ngoài.",
        "questions": [{"question": "Vì sao người nói không ra ngoài?", "answer": "Vì trời mưa.", "explanation": "ので nêu lý do."}],
    }
    page = {
        "page_index": 1, "page_name": "page-1.png", "source_label": "Trang 1: page-1.png",
        "source_text": item["edited_text"], "confirmed_text": item["edited_text"], "summary": "Trời mưa.",
        "analysis_language": "japanese", "vocabulary_all": [], "vocabulary_important": [],
        "kanji_analysis": [], "connectors": [], "grammar_points": [], "sentence_patterns": [],
        "full_markdown": "# Báo cáo", "usage": {},
        "sentence_catalog": [{"sentence_id": "p1-s1", "ordinal": 1, "original": item["edited_text"], "detected_language": "japanese", "analyzed": True, "eligible": True, "complexity_score": 7}],
        "sentence_breakdowns": [breakdown], "sentence_analysis_usage": {"input_tokens": 2, "output_tokens": 3},
        "translation_guidance": [
            {
                "sentence_id": "p1-s1", "ordinal": 1, "original": item["edited_text"],
                "reading": "あめがふっているので、でかけません。",
                "translations": {
                    "chunked": "Vì trời mưa / không ra ngoài",
                    "literal": "Vì trời đang mưa, không ra ngoài.",
                    "natural": "Vì trời mưa nên tôi không ra ngoài.",
                },
                "key_points": [
                    {"label": "Từ nối nguyên nhân", "source": "ので", "explanation_vi": "Nối lý do với kết quả."}
                ],
                "translation_steps": [], "related_analysis": [], "ocr_warning": "",
            }
        ],
    }
    app = AppTest.from_file("app.py")
    app.session_state["image_items"] = [item]
    app.session_state["analysis"] = {
        **page,
        "page_analyses": [page],
        "usage": {},
        "sentence_analysis_usage": {"input_tokens": 2, "output_tokens": 3},
        "model_used": "gemini-3.5-flash",
    }
    app.session_state["upload_messages"] = []
    app.session_state["upload_errors"] = []
    app.session_state["uploader_version"] = 0
    app.session_state["camera_version"] = 0
    app.run(timeout=20)

    assert not app.exception
    labels = [expander.label for expander in app.get("expander")]
    assert "Chi tiết dịch và phân tích câu 1" in labels
    assert "Phân tích thêm câu khác" in labels
    assert any(toggle.label == "Hiện đáp án" and toggle.value is False for toggle in app.toggle)
    assert any(toggle.label == "Nói chậm nguyên văn" for toggle in app.toggle)
    button_labels = [button.label for button in app.button]
    assert "Nghe nguyên văn" in button_labels
    assert "Nghe bản dịch tiếng Việt" in button_labels
    markdown_values = [item.value for item in app.markdown]
    assert any("Từ nối nguyên nhân" in value for value in markdown_values)
    assert any("Mệnh đề nguyên nhân" in value for value in markdown_values)
    assert any("Khung câu trung tâm" in value for value in markdown_values)
    assert any("Chuỗi ngữ pháp" in value for value in markdown_values)
    assert any("あめがふっているので" in item.value for item in app.caption)


def test_app_renders_completed_notion_sync_status(tmp_path, monkeypatch):
    from app_pages import ocr_page
    from components import sidebar
    from modules import notion_sync, session_store
    from modules.job_workflow import items_source_hash

    monkeypatch.setattr(session_store, "_DB_PATH", str(tmp_path / "sessions.db"))
    connected = lambda: {"configured": True, "label": "Đã kết nối", "workspace_ready": True}
    monkeypatch.setattr(notion_sync, "notion_connection_state", connected)
    monkeypatch.setattr(sidebar, "notion_connection_state", connected)
    monkeypatch.setattr(ocr_page, "notion_connection_state", connected)

    item = create_image_item(_image_bytes("white"), "notion.png")
    item["ocr_result"] = {
        "clean_text": "日本語",
        "ocr_notes": [],
        "usage": {},
        "text_direction": "horizontal",
        "has_furigana": False,
        "confidence": "high",
    }
    item["edited_text"] = "日本語"
    analysis = {
        "analysis_language": "japanese",
        "summary": "Tóm tắt Notion",
        "full_markdown": "# Báo cáo",
        "usage": {},
        "vocabulary_all": [],
        "vocabulary_important": [],
        "kanji_analysis": [],
        "connectors": [],
        "grammar_points": [],
        "sentence_patterns": [],
    }
    session_store.create_session("notion-ui")
    source_hash = items_source_hash([item])
    run = session_store.ensure_notion_sync_run(
        "notion-ui",
        f"analysis:{source_hash}",
        source_hash,
        {"learning_items": []},
    )
    session_store.finish_notion_sync_run(
        run["run_id"],
        "done",
        notion_page_id="page-id",
        notion_page_url="https://www.notion.so/page-id",
    )

    app = AppTest.from_file("app.py")
    app.session_state["session_id"] = "notion-ui"
    app.session_state["session_restored"] = True
    app.session_state["image_items"] = [item]
    app.session_state["analysis"] = analysis
    app.session_state["partial_page_analyses"] = []
    app.session_state["upload_messages"] = []
    app.session_state["upload_errors"] = []
    app.session_state["uploader_version"] = 0
    app.session_state["camera_version"] = 0
    app.session_state["auto_notion_sync"] = False
    app.run(timeout=20)

    assert not app.exception
    assert any("Đã lưu bài" in message.value for message in app.success)
    assert any(toggle.label == "Tự động lưu bài hoàn tất" for toggle in app.toggle)
