from types import SimpleNamespace

import pytest

from modules import text_analyzer


RESPONSE = """# PHÂN TÍCH VĂN BẢN TIẾNG NHẬT
## 1. XÁC NHẬN VĂN BẢN GỐC
日本は島国です。
[Sửa: 本 → 日本 | Lý do: ngữ cảnh]
**Tóm tắt nội dung:** Bài viết giới thiệu Nhật Bản.

## 2. TỪ VỰNG JLPT N4-N1
### 2.1 Danh sách toàn bộ từ vựng trong bài
| # | Từ gốc | Phiên âm | Loại từ | Nghĩa | JLPT |
|---|---|---|---|---|---|
| 1 | 日本 | にほん | Danh từ | Nhật Bản | N5 |
### 2.2 Từ vựng quan trọng
| Từ | Phiên âm | Loại | Nghĩa | Ví dụ | Khó |
|---|---|---|---|---|---|
| 島国 | しまぐに | Danh từ | Đảo quốc | 日本は島国です | N3 |

## 3. PHÂN TÍCH KANJI
| Kanji | On | Kun | Nghĩa | JLPT | Từ | Ví dụ | Vai trò |
|---|---|---|---|---|---|---|---|
| 日 | ニチ | ひ | ngày | N5 | 日本 | 日本 | bổ nghĩa |

## 4. TỪ NỐI CÂU & LIÊN TỪ
| Cụm | Phiên âm | Loại | Nghĩa | Ví dụ | Vai trò | Khó |
|---|---|---|---|---|---|---|
| しかし | しかし | liên từ | tuy nhiên | しかし | tương phản | N4 |

## 5. PHÂN TÍCH NGỮ PHÁP
**[N + です]**
- Quy tắc: N + です
- Ví dụ trong bài: 日本は島国です
- Giải thích ý nghĩa & cách dùng: Câu lịch sự.

## 6. MẪU CÂU ĐẶC TRƯNG
**Mẫu:** `N は N です`
- Ví dụ trong bài: 日本は島国です
- Giải thích: Khẳng định.

## 7. TỔNG HỢP ĐẦY ĐỦ (DÀNH CHO WORD EXPORT)
# Báo cáo
Nội dung hoàn chỉnh.
"""


def test_build_and_parse():
    prompt = text_analyzer.build_analysis_prompt("日本", ["ghi chú một", "ghi chú hai"])
    result = text_analyzer.parse_analysis_response(RESPONSE)

    assert "1. ghi chú một" in prompt
    assert result["summary"] == "Bài viết giới thiệu Nhật Bản."
    assert len(result["vocabulary_all"]) == 1
    assert len(result["kanji_analysis"]) == 1
    assert result["grammar_points"][0]["rule"] == "N + です"
    assert result["sentence_patterns"][0]["pattern"] == "N は N です"
    assert result["full_markdown"].startswith("# PHÂN TÍCH")


def test_run_analysis_and_long_text_merge(monkeypatch):
    class Model:
        def generate_content(self, _prompt, generation_config):
            assert generation_config["temperature"] == 0.1
            return SimpleNamespace(
                text=RESPONSE,
                usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20),
            )

    monkeypatch.setattr(text_analyzer, "_init_model", lambda: Model())
    result = text_analyzer.run_analysis("日" * 4001, [])
    assert len(result["vocabulary_all"]) == 2
    assert result["usage"]["input_tokens"] == 20
    assert result["usage"]["output_tokens"] == 40
    assert result["usage"]["candidate_tokens"] == 40
    assert result["usage"]["thinking_tokens"] == 0


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        text_analyzer.run_analysis(" ", [])


def test_truncated_response_without_section_7_is_still_usable(monkeypatch):
    truncated = """# PHÂN TÍCH
## 1 XÁC NHẬN VĂN BẢN GỐC
日本は島国です。
Tóm tắt: Nhật Bản là đảo quốc.
## 2 TỪ VỰNG
Nội dung bị cắt trước mục 7.
"""

    class Model:
        def generate_content(self, _prompt, generation_config):
            assert generation_config["max_output_tokens"] == 16384
            return SimpleNamespace(text=truncated, usage_metadata=None)

    monkeypatch.setattr(text_analyzer, "_init_model", lambda: Model())
    result = text_analyzer.run_analysis("日本は島国です。", [])

    assert result["summary"] == "Nhật Bản là đảo quốc."
    assert result["confirmed_text"]
    assert result["full_markdown"] == truncated.strip()


def test_analysis_backfills_missing_kanji_and_grammar(monkeypatch):
    main = """# PHÂN TÍCH
## 1 XÁC NHẬN VĂN BẢN GỐC
日本は島国です。
Tóm tắt: Nhật Bản là đảo quốc.
## 2 TỪ VỰNG
### 2.1 Danh sách toàn bộ từ vựng trong bài
| # | Từ | Đọc | Loại | Nghĩa | JLPT |
|---|---|---|---|---|---|
| 1 | 日本 | にほん | danh từ | Nhật Bản | N5 |
"""
    supplemental = """## 3. PHÂN TÍCH KANJI
| Kanji | On | Kun | Nghĩa | JLPT | Từ | Ví dụ | Vai trò |
|---|---|---|---|---|---|---|---|
| 国 | コク | くに | nước | N5 | 島国 | 島国です | gốc nghĩa |
## 5. PHÂN TÍCH NGỮ PHÁP
### N + です
- Quy tắc: N + です
- Ví dụ trong bài: 島国です
- Giải thích: Câu lịch sự.
"""

    class Model:
        calls = 0

        def generate_content(self, _prompt, generation_config):
            self.calls += 1
            return SimpleNamespace(text=main if self.calls == 1 else supplemental, usage_metadata=None)

    model = Model()
    monkeypatch.setattr(text_analyzer, "_init_model", lambda: model)
    result = text_analyzer.run_analysis("日本は島国です。", [])

    assert result["kanji_analysis"][0]["kanji"] == "国"
    assert result["grammar_points"][0]["name"] == "N + です"
    assert "Bổ sung mục còn thiếu" in result["full_markdown"]
