from types import SimpleNamespace

import pytest

from modules import text_analyzer


RESPONSE = """## 1. Text Confirmation &amp; Summary
**Original text:**
The team decided to carry out the plan because the deadline was approaching.

**OCR Corrections:** [Correction: tearn -> team]

**Summary:** The sentence describes a team choosing to execute a plan under time pressure. The tone is neutral and task-focused.

## 2. Vocabulary Analysis
### 2.1 All Vocabulary
| # | Word | Base Form | Part of Speech | Vietnamese Meaning | CEFR Level |
|---|------|-----------|----------------|--------------------|------------|
| 1 | team | team | noun | đội nhóm | A2 |
| 2 | decided | decide | verb | quyết định | B1 |
| 3 | approaching | approach | verb | đang đến gần | B2 |
### 2.2 Key Vocabulary (B2-C2, idioms, academic, domain-specific)
| Word | Base Form | Part of Speech | Vietnamese Meaning | Example from Text | Difficulty |
|------|-----------|----------------|--------------------|-------------------|------------|
| approaching | approach | verb | đang đến gần | "the deadline was approaching" | B2 |

## 3. Phrasal Verbs &amp; Collocations
| Phrase | Type | Vietnamese Meaning | Example from Text | Note |
|--------|------|--------------------|-------------------|------|
| carry out | phrasal verb | thực hiện | "carry out the plan" | transitive |

## 4. Linking Words &amp; Discourse Markers
| Word/Phrase | Function | Vietnamese Meaning | Example from Text | Register | Difficulty |
|-------------|----------|--------------------|-------------------|----------|------------|
| because | reason | bởi vì | "because the deadline was approaching" | neutral | A2 |

## 5. Grammar Points
**[Past simple]**
- Rule: Use past simple for completed decisions or actions in the past.
- Example from text: "The team decided"
- Explanation: It presents the decision as completed.

## 6. Sentence Patterns &amp; Structures
**Pattern:** `because-clause of reason`
- Example from text: "because the deadline was approaching"
- Explanation: The subordinate clause gives the reason for the main action.

## 7. Full Analysis Report (Markdown)
# English Text Analysis Report
## 1. Text Overview
The team decided to carry out the plan because the deadline was approaching.
## 2. Vocabulary
See vocabulary tables above.
## 3. Phrasal Verbs &amp; Collocations
See section 3.
## 4. Linking Words &amp; Discourse Markers
See section 4.
## 5. Grammar Points
See section 5.
## 6. Sentence Patterns
See section 6.
## 7. Overall Assessment
- **CEFR Level Estimate:** B1
- **Text Type:** conversational
- **Dominant Tense:** simple past
- **Writing Style Notes:** neutral and clear
- **Study Recommendations:** Review phrasal verbs and reason clauses.
"""

JAPANESE_RESPONSE = """# PHÂN TÍCH VĂN BẢN TIẾNG NHẬT
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
    prompt = text_analyzer.build_analysis_prompt("The team decided.", ["ghi chú một", "ghi chú hai"])
    result = text_analyzer.parse_analysis_response(RESPONSE)

    assert "English text to analyze:" in prompt
    assert "The team decided." in prompt
    assert "1. ghi chú một" in prompt
    assert result["summary"].startswith("The sentence describes")
    assert result["ocr_corrections"] == ["[Correction: tearn -> team]"]
    assert result["vocabulary_all"][0]["base_form"] == "team"
    assert result["vocabulary_all"][2]["cefr"] == "B2"
    assert result["vocabulary_important"][0]["difficulty"] == "B2"
    assert result["phrasal_collocations"][0]["phrase"] == "carry out"
    assert result["discourse_markers"][0]["function"] == "reason"
    assert result["grammar_points"][0]["rule"].startswith("Use past simple")
    assert result["sentence_patterns"][0]["pattern"] == "because-clause of reason"
    assert result["full_markdown"].startswith("## 1. Text Confirmation")


def test_build_and_parse_japanese_mode():
    prompt = text_analyzer.build_analysis_prompt("日本", ["ghi chú"], "japanese")
    result = text_analyzer.parse_analysis_response(JAPANESE_RESPONSE, "japanese")

    assert "PHÂN TÍCH VĂN BẢN TIẾNG NHẬT" in prompt
    assert "English text to analyze" not in prompt
    assert result["analysis_language"] == "japanese"
    assert result["summary"] == "Bài viết giới thiệu Nhật Bản."
    assert result["vocabulary_all"][0]["reading"] == "にほん"
    assert result["kanji_analysis"][0]["kanji"] == "日"
    assert result["connectors"][0]["phrase"] == "しかし"
    assert result["phrasal_collocations"] == []
    assert result["discourse_markers"] == []


def test_parse_japanese_vocab_detail_blocks():
    mock = """
**[一概に・いちがいに]**
- Loại từ: trạng từ
- Ý nghĩa: nói chung, một cách cào bằng
- Ví dụ trong bài: 一概には言えない
- Ví dụ 1: 人の性格は一概には決められない。
- Ví dụ 2: この問題は一概に悪いとは言えない。
- Từ liên quan: 総じて、概して
- Lưu ý: Thường dùng trong câu phủ định
- Mức độ: N2
"""
    result = text_analyzer._parse_named_blocks(mock, r"^\s*\*\*\[(.+?)\]\*\*\s*$", "vocab_detail")

    assert result[0]["word"] == "一概に・いちがいに"
    assert result[0]["example_1"] == "人の性格は一概には決められない。"
    assert result[0]["jlpt"] == "N2"


def test_parse_english_vocab_detail_blocks():
    mock = """
**[meticulous (adjective)]**
- Vietnamese Meaning: tỉ mỉ, cẩn thận từng chi tiết nhỏ
- Definition: showing great attention to detail; very careful and precise
- Example from text: She was meticulous in her research.
- Example 1: He is meticulous about keeping his desk tidy.
- Example 2: The scientist conducted a meticulous analysis of the data.
- Related words: thorough, precise, painstaking; antonym: careless
- Common mistake: Confusing with careful — meticulous implies extreme detail
- CEFR Level: C1
"""
    result = text_analyzer._parse_named_blocks(mock, r"^\s*\*\*\[(.+?)\]\*\*\s*$", "vocab_detail")

    assert result[0]["word"] == "meticulous (adjective)"
    assert result[0]["vn_meaning"].startswith("tỉ mỉ")
    assert result[0]["example_1"] == "He is meticulous about keeping his desk tidy."
    assert result[0]["cefr"] == "C1"


def test_parse_japanese_detailed_grammar_blocks():
    mock = """
**[〜とは限らない]**
- Công thức: 普通形 + とは限らない
- Ý nghĩa: không hẳn là, không nhất thiết là
- Cách dùng: Dùng để phủ định một nhận định tuyệt đối.
- Ví dụ trong bài: いつも正しいとは限らない
- Phân tích ví dụ: 正しい là nội dung bị phủ định tính tuyệt đối.
- Ví dụ 1: 高い物が必ず良いとは限らない。
- Ví dụ 2: この結果が全体を示すとは限らない。
- Lưu ý: Khác với とは言えない vì nhấn mạnh ngoại lệ.
- Mức độ: N2
"""
    result = text_analyzer._parse_named_blocks(mock, r"^\s*\*\*(?!Mẫu:)(.+?)\*\*\s*$", "grammar")

    assert result[0]["name"] == "〜とは限らない"
    assert result[0]["structure"] == "普通形 + とは限らない"
    assert result[0]["example_analysis"].startswith("正しい")
    assert result[0]["level"] == "N2"


def test_parse_english_detailed_grammar_blocks():
    mock = """
**[Present perfect]**
- Structure: have/has + past participle
- Rule: Dùng để nói về trải nghiệm hoặc kết quả liên quan đến hiện tại.
- Meaning: nhấn mạnh kết quả hiện tại của hành động.
- Example from text: She has completed the report.
- Example analysis: has completed cho thấy báo cáo đã xong và kết quả còn liên quan hiện tại.
- Example 1: I have lost my keys.
- Example 2: Researchers have identified a new pattern.
- Common mistake: Không dùng thời gian quá khứ cụ thể như yesterday với present perfect.
- Level: B1
"""
    result = text_analyzer._parse_named_blocks(mock, r"^\s*\*\*(?!Mẫu:)(.+?)\*\*\s*$", "grammar")

    assert result[0]["name"] == "Present perfect"
    assert result[0]["structure"] == "have/has + past participle"
    assert result[0]["mistake"].startswith("Không dùng")
    assert result[0]["level"] == "B1"


def test_run_analysis_and_long_text_merge(monkeypatch):
    class Model:
        def generate_content(self, _prompt, generation_config):
            assert generation_config["temperature"] == 0.1
            return SimpleNamespace(
                text=RESPONSE,
                usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20),
            )

    monkeypatch.setattr(text_analyzer, "_init_model", lambda: Model())
    result = text_analyzer.run_analysis("A" * 4001, [])
    assert len(result["vocabulary_all"]) == 6
    assert len(result["phrasal_collocations"]) == 2
    assert result["usage"]["input_tokens"] == 20
    assert result["usage"]["output_tokens"] == 40
    assert result["usage"]["candidate_tokens"] == 40
    assert result["usage"]["thinking_tokens"] == 0


def test_run_page_analyses_keeps_per_page_results(monkeypatch):
    class Model:
        prompts = []

        def generate_content(self, prompt, generation_config):
            self.prompts.append(prompt)
            return SimpleNamespace(
                text=RESPONSE,
                usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=5),
            )

    model = Model()
    monkeypatch.setattr(text_analyzer, "_init_model", lambda: model)
    result = text_analyzer.run_page_analyses(
        [
            {"page_index": 1, "page_name": "page-1.png", "text": "First page text.", "notes": ["note 1"]},
            {"page_index": 2, "page_name": "page-2.png", "text": "Second page text.", "notes": ["note 2"]},
        ]
    )

    assert len(model.prompts) == 2
    assert "First page text." in model.prompts[0]
    assert "Second page text." not in model.prompts[0]
    assert "Second page text." in model.prompts[1]
    assert len(result["page_analyses"]) == 2
    assert result["page_analyses"][0]["source_label"] == "Trang 1: page-1.png"
    assert result["usage"]["input_tokens"] == 6
    assert result["usage"]["output_tokens"] == 10


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        text_analyzer.run_analysis(" ", [])


def test_truncated_response_without_section_7_is_still_usable(monkeypatch):
    truncated = """## 1. Text Confirmation &amp; Summary
The deadline was approaching.
Summary: The text describes time pressure.
## 2. Vocabulary Analysis
Content was cut before section 7.
"""

    class Model:
        def generate_content(self, _prompt, generation_config):
            assert generation_config["max_output_tokens"] == 16384
            return SimpleNamespace(text=truncated, usage_metadata=None)

    monkeypatch.setattr(text_analyzer, "_init_model", lambda: Model())
    result = text_analyzer.run_analysis("The deadline was approaching.", [])

    assert result["summary"] == "The text describes time pressure."
    assert result["confirmed_text"]
    assert result["full_markdown"] == truncated.strip()


def test_analysis_backfills_missing_phrasal_and_grammar(monkeypatch):
    main = """## 1. Text Confirmation &amp; Summary
The team decided to carry out the plan.
Summary: The text describes a decision.
## 2. Vocabulary Analysis
### 2.1 All Vocabulary
| # | Word | Base Form | Part of Speech | Vietnamese Meaning | CEFR Level |
|---|---|---|---|---|---|
| 1 | team | team | noun | đội nhóm | A2 |
"""
    supplemental = """## 3. Phrasal Verbs &amp; Collocations
| Phrase | Type | Vietnamese Meaning | Example from Text | Note |
|---|---|---|---|---|
| carry out | phrasal verb | thực hiện | "carry out the plan" | transitive |
## 5. Grammar Points
**[Past simple]**
- Rule: Use past simple for completed past actions.
- Example from text: "decided"
- Explanation: It marks the decision as completed.
"""

    class Model:
        calls = 0

        def generate_content(self, _prompt, generation_config):
            self.calls += 1
            return SimpleNamespace(text=main if self.calls == 1 else supplemental, usage_metadata=None)

    model = Model()
    monkeypatch.setattr(text_analyzer, "_init_model", lambda: model)
    result = text_analyzer.run_analysis("The team decided to carry out the plan.", [])

    assert result["phrasal_collocations"][0]["phrase"] == "carry out"
    assert result["grammar_points"][0]["name"] == "Past simple"
    assert "Missing section supplement" in result["full_markdown"]
