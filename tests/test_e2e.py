import io
from types import SimpleNamespace

from PIL import Image, ImageDraw

from modules import ocr_engine, text_analyzer
from modules.doc_exporter import export_to_docx
from modules.image_processor import process_image
from modules.multi_image_workflow import add_image_items, combined_notes, combined_text


OCR_RESPONSE = """TEXT_DIRECTION: horizontal
TEXT_REGIONS: 1
HAS_FURIGANA: no
CONFIDENCE: high
---OCR_START---
日本は島国です。
---OCR_END---
---NOTES_START---
なし
---NOTES_END---
"""

ANALYSIS_RESPONSE = """# PHÂN TÍCH VĂN BẢN TIẾNG NHẬT
## 1. XÁC NHẬN VĂN BẢN GỐC
日本は島国です。
**Tóm tắt nội dung:** Nhật Bản là đảo quốc.
## 2. TỪ VỰNG JLPT N4-N1
### 2.1 Danh sách toàn bộ từ vựng trong bài
| # | Từ | Đọc | Loại | Nghĩa | JLPT |
|---|---|---|---|---|---|
| 1 | 日本 | にほん | danh từ | Nhật Bản | N5 |
### 2.2 Từ vựng quan trọng
| Từ | Đọc | Loại | Nghĩa | Ví dụ | Khó |
|---|---|---|---|---|---|
| 島国 | しまぐに | danh từ | đảo quốc | 日本は島国です | N3 |
## 3. PHÂN TÍCH KANJI
| Kanji | On | Kun | Nghĩa | JLPT | Từ | Ví dụ | Vai trò |
|---|---|---|---|---|---|---|---|
| 日 | ニチ | ひ | ngày | N5 | 日本 | 日本 | cấu tạo |
## 4. TỪ NỐI CÂU & LIÊN TỪ
Không có.
## 5. PHÂN TÍCH NGỮ PHÁP
**[N + です]**
- Quy tắc: N + です
- Ví dụ trong bài: 日本は島国です
- Giải thích: Khẳng định lịch sự.
## 6. MẪU CÂU ĐẶC TRƯNG
**Mẫu:** `N は N です`
- Ví dụ trong bài: 日本は島国です
- Giải thích: Câu khẳng định.
## 7. TỔNG HỢP ĐẦY ĐỦ (DÀNH CHO WORD EXPORT)
# Báo cáo phân tích
## Tóm tắt
Nhật Bản là đảo quốc.
| Từ | Nghĩa |
|---|---|
| 日本 | Nhật Bản |
"""


def sample_image():
    image = Image.new("RGB", (600, 300), "white")
    ImageDraw.Draw(image).text((40, 120), "Japanese OCR", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    return buffer.getvalue()


def test_full_pipeline(monkeypatch):
    class OcrModel:
        def generate_content(self, _content):
            return SimpleNamespace(text=OCR_RESPONSE, usage_metadata=None)

    class AnalysisModel:
        def generate_content(self, _prompt, generation_config):
            return SimpleNamespace(text=ANALYSIS_RESPONSE, usage_metadata=None)

    monkeypatch.setattr(ocr_engine, "init_gemini", lambda: OcrModel())
    monkeypatch.setattr(text_analyzer, "_init_model", lambda: AnalysisModel())

    image_result = process_image(sample_image())
    ocr_result = ocr_engine.run_ocr(image_result["processed_image_bytes"], image_result["report"])
    analysis = text_analyzer.run_analysis(ocr_result["clean_text"], ocr_result["ocr_notes"])
    docx_bytes = export_to_docx(analysis["full_markdown"])

    assert len(ocr_result["clean_text"]) > 5
    assert analysis["summary"]
    assert analysis["vocabulary_all"]
    assert len(docx_bytes) > 1000


def test_two_images_ocr_and_combined_analysis(monkeypatch):
    class OcrModel:
        def generate_content(self, _content):
            return SimpleNamespace(text=OCR_RESPONSE, usage_metadata=None)

    class AnalysisModel:
        def generate_content(self, prompt, generation_config):
            assert "=== ẢNH 1: page-1.jpg ===" in prompt
            assert "=== ẢNH 2: page-2.jpg ===" in prompt
            return SimpleNamespace(text=ANALYSIS_RESPONSE, usage_metadata=None)

    monkeypatch.setattr(ocr_engine, "init_gemini", lambda: OcrModel())
    monkeypatch.setattr(text_analyzer, "_init_model", lambda: AnalysisModel())

    items, added, errors = add_image_items(
        [],
        [("page-1.jpg", sample_image()), ("page-2.jpg", sample_image() + b"different")],
    )
    assert added == ["page-1.jpg", "page-2.jpg"]
    assert errors == []

    for item in items:
        item["ocr_result"] = ocr_engine.run_ocr(item["processed_image_bytes"], item["report"])
        item["edited_text"] = item["ocr_result"]["clean_text"]

    analysis = text_analyzer.run_analysis(combined_text(items), combined_notes(items))
    assert analysis["summary"]
