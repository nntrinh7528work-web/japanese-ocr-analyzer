import pytest
from modules.dialogue_exporter import export_dialogue_to_text, export_dialogue_to_json, export_dialogue_to_docx

@pytest.fixture
def sample_result():
    return {
        "topic": "Đặt phòng khách sạn",
        "language": "Tiếng Nhật",
        "level": "Trung cấp",
        "situation": "Khách sạn",
        "politeness_level": "Kính ngữ (敬語)",
        "dialogue": [
            {
                "speaker": "A",
                "text": "こんにちは、予約したいのですが。",
                "text_hira": "こんにちは、よやくしたいのですが。",
                "text_vi": "Xin chào, tôi muốn đặt phòng ạ.",
                "highlights": ["予約"]
            },
            {
                "speaker": "B",
                "text": "はい、かしこまりました。",
                "text_hira": "はい、かしこまりました。",
                "text_vi": "Vâng, tôi hiểu rồi ạ.",
                "highlights": []
            }
        ],
        "summary": "Từ vựng: 予約 (Đặt chỗ)",
        "notes": "Ghi chú ngữ cảnh",
    }

def test_export_to_text(sample_result):
    txt = export_dialogue_to_text(sample_result)
    assert "Đặt phòng khách sạn" in txt
    assert "こんにちは、予約したいのですが。" in txt
    assert "Xin chào, tôi muốn đặt phòng ạ." in txt

def test_export_to_json(sample_result):
    json_str = export_dialogue_to_json(sample_result)
    assert '"topic": "Đặt phòng khách sạn"' in json_str

def test_export_to_docx(sample_result):
    docx_bytes = export_dialogue_to_docx(sample_result)
    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 500
