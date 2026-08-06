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


def test_japanese_prompt_requires_detailed_connectors_and_source_order():
    prompt = text_analyzer.build_analysis_prompt("雨なので、家にいた。しかし、退屈だった。", [], "japanese")

    assert "接続助詞" in prompt
    assert "trạng từ liên kết" in prompt.lower()
    assert "thứ tự xuất hiện đầu tiên" in prompt
    assert "Hai thành phần được nối" in prompt
    assert "Quan hệ logic & sắc thái" in prompt


def test_parse_enhanced_japanese_connector_columns():
    response = JAPANESE_RESPONSE.replace(
        "| Cụm | Phiên âm | Loại | Nghĩa | Ví dụ | Vai trò | Khó |\n"
        "|---|---|---|---|---|---|---|\n"
        "| しかし | しかし | liên từ | tuy nhiên | しかし | tương phản | N4 |",
        "| STT | Từ/Cụm | Phiên âm | Nhóm | Cấu trúc/Cách nối | Nghĩa tiếng Việt | Ví dụ trong bài | Hai thành phần được nối | Quan hệ logic & sắc thái | JLPT |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
        "| 1 | ので | ので | 接続助詞 | V/A thể thường + ので | vì | 雨なので、家にいた。 | nguyên nhân → kết quả | nguyên nhân khách quan, mềm hơn から | N4 |",
    )

    result = text_analyzer.parse_analysis_response(response, "japanese")

    connector = result["connectors"][0]
    assert connector["phrase"] == "ので"
    assert connector["type"] == "接続助詞"
    assert connector["structure"] == "V/A thể thường + ので"
    assert connector["linked_parts"] == "nguyên nhân → kết quả"
    assert connector["role"].startswith("nguyên nhân khách quan")


def test_parse_japanese_vocab_detail_blocks():
    mock = """
**[一概に・いちがいに]**
- Loại từ: trạng từ
- Ý nghĩa: nói chung, một cách cào bằng
- Ví dụ trong bài: 一概には言えない
- Hiragana ví dụ trong bài: いちがいにはいえない
- Ví dụ 1: 人の性格は一概には決められない。
- Hiragana ví dụ 1: ひとのせいかくはいちがいにはきめられない。
- Ví dụ 2: この問題は一概に悪いとは言えない。
- Hiragana ví dụ 2: このもんだいはいちがいにわるいとはいえない。
- Từ liên quan: 総じて、概して
- Lưu ý: Thường dùng trong câu phủ định
- Mức độ: N2
"""
    result = text_analyzer._parse_named_blocks(mock, r"^\s*\*\*\[(.+?)\]\*\*\s*$", "vocab_detail")

    assert result[0]["word"] == "一概に・いちがいに"
    assert result[0]["example_text_hiragana"] == "いちがいにはいえない"
    assert result[0]["example_1"] == "人の性格は一概には決められない。"
    assert result[0]["example_1_hiragana"] == "ひとのせいかくはいちがいにはきめられない。"
    assert result[0]["jlpt"] == "N2"


def test_parse_inline_hiragana_examples():
    mock = """
**[検討・けんとう]**
- Loại từ: danh từ / tha động từ
- Ý nghĩa: cân nhắc, xem xét kỹ
- Ví dụ trong bài: この案を検討します - Tôi sẽ xem xét phương án này - (Hiragana: このあんをけんとうします)
- Ví dụ 1: 詳細を検討してください。 - Hãy xem xét chi tiết. - (Hiragana: しょうさいをけんとうしてください。)
- Ví dụ 2: 委員会で政策を検討した。 - Ủy ban đã xem xét chính sách. - (Hiragana: いいんかいでせいさくをけんとうした。)
- Mức độ: N2
"""
    result = text_analyzer._parse_named_blocks(mock, r"^\s*\*\*\[(.+?)\]\*\*\s*$", "vocab_detail")

    assert result[0]["example_text"] == "この案を検討します - Tôi sẽ xem xét phương án này"
    assert result[0]["example_text_hiragana"] == "このあんをけんとうします"
    assert result[0]["example_1_hiragana"] == "しょうさいをけんとうしてください。"
    assert result[0]["example_2_hiragana"] == "いいんかいでせいさくをけんとうした。"


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


def test_parse_japanese_grammar_hiragana_examples():
    mock = """
**[〜とは限らない]**
- Công thức: 普通形 + とは限らない
- Ý nghĩa: không nhất thiết là
- Ví dụ trong bài: いつも正しいとは限らない - Không phải lúc nào cũng đúng.
- Hiragana ví dụ trong bài: いつもただしいとはかぎらない
- Ví dụ 1: 高いものが良いとは限らない。 - Đồ đắt chưa chắc đã tốt.
- Hiragana ví dụ 1: たかいものがよいとはかぎらない。
- Ví dụ 2: 専門家の意見が常に正しいとは限らない。 - Ý kiến chuyên gia không phải luôn đúng.
- Hiragana ví dụ 2: せんもんかのいけんがつねにただしいとはかぎらない。
- Mức độ: N2
"""
    result = text_analyzer._parse_named_blocks(mock, r"^\s*\*\*(?!Mẫu:)(.+?)\*\*\s*$", "grammar")

    assert result[0]["example_hiragana"] == "いつもただしいとはかぎらない"
    assert result[0]["example_1_hiragana"] == "たかいものがよいとはかぎらない。"
    assert result[0]["example_2_hiragana"] == "せんもんかのいけんがつねにただしいとはかぎらない。"


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
    # Text must exceed 2500 chars to trigger chunk splitting.
    result = text_analyzer.run_analysis("A" * 8001, [])
    # Deduplication removes identical vocab/phrasal rows across chunks.
    assert len(result["vocabulary_all"]) == 3
    assert len(result["phrasal_collocations"]) == 1
    # 8001 chars / 2500 max_chars = 4 chunks, each contributing 10 input tokens.
    assert result["usage"]["input_tokens"] == 40
    assert result["usage"]["output_tokens"] == 80
    assert result["usage"]["candidate_tokens"] == 80
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

    progress_calls = []
    page_done_calls = []

    def _on_progress(done, total, name):
        progress_calls.append((done, total, name))

    def _on_page_done(page_result):
        page_done_calls.append(page_result)

    result = text_analyzer.run_page_analyses(
        [
            {"page_index": 1, "page_name": "page-1.png", "text": "First page text.", "notes": ["note 1"]},
            {"page_index": 2, "page_name": "page-2.png", "text": "Second page text.", "notes": ["note 2"]},
        ],
        progress_callback=_on_progress,
        page_done_callback=_on_page_done,
        auto_translation_guidance=False,
    )

    assert len(model.prompts) == 2
    # With concurrent execution prompts may arrive in any order.
    all_prompt_text = "\n".join(model.prompts)
    assert "First page text." in all_prompt_text
    assert "Second page text." in all_prompt_text
    assert len(result["page_analyses"]) == 2
    assert result["page_analyses"][0]["source_label"] == "Trang 1: page-1.png"
    assert result["usage"]["input_tokens"] == 6
    assert result["usage"]["output_tokens"] == 10
    # Progress and page_done callbacks should have been called.
    assert len(progress_calls) == 2
    assert len(page_done_calls) == 2


def test_sentence_deep_dive_failure_does_not_fail_main_page(monkeypatch):
    class Model:
        def generate_content(self, _prompt, generation_config):
            return SimpleNamespace(
                text=RESPONSE,
                usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=5),
            )

    monkeypatch.setattr(text_analyzer, "_init_model", lambda: Model())
    monkeypatch.setattr(
        text_analyzer,
        "analyze_sentence_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("deep failed")),
    )
    callbacks = []
    long_sentence = (
        "Although the proposal appeared reasonable, the committee rejected it because "
        "the evidence that the researchers provided was incomplete and difficult to verify."
    )

    result = text_analyzer.run_page_analyses(
        [{"page_index": 1, "page_name": "p1", "text": long_sentence, "notes": []}],
        page_done_callback=callbacks.append,
        auto_translation_guidance=False,
    )

    assert result["vocabulary_all"]
    assert result["page_analyses"][0]["sentence_analysis_error"] == "deep failed"
    assert result["page_analyses"][0]["sentence_breakdowns"] == []
    assert len(callbacks) == 2


def test_guidance_batches_persist_and_one_failure_does_not_block_later_batch(monkeypatch):
    class Model:
        def generate_content(self, _prompt, generation_config):
            return SimpleNamespace(
                text=RESPONSE,
                usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=5),
            )

    monkeypatch.setattr(text_analyzer, "_init_model", lambda: Model())
    calls = 0

    def _guidance(_model, batch, _text, _language, reasoning_effort="standard"):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("first batch failed")
        return (
            [
                {
                    "sentence_id": row["sentence_id"], "ordinal": row["ordinal"],
                    "original": row["original"], "translations": {"natural": "Dịch"},
                    "key_points": [], "translation_steps": [], "related_analysis": [],
                }
                for row in batch
            ],
            {"input_tokens": 2, "output_tokens": 3},
        )

    monkeypatch.setattr(text_analyzer, "analyze_guidance_batch", _guidance)
    callbacks = []
    source = " ".join(
        f"This is sentence number {index}, and it contains enough words to remain a complete example."
        for index in range(1, 10)
    )
    result = text_analyzer.run_page_analyses(
        [{"page_index": 1, "page_name": "p1", "text": source, "notes": []}],
        page_done_callback=callbacks.append,
        auto_sentence_deep_dive=False,
    )
    page = result["page_analyses"][0]

    assert calls == 2
    assert len(page["translation_guidance_errors"]) == 1
    assert len(page["translation_guidance"]) == 1
    assert page["translation_guidance_usage"]["input_tokens"] == 2
    assert len(callbacks) == 3  # main result, failed batch, successful batch


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


def test_deduplicate_rows():
    rows = [
        {"word": "team", "meaning": "đội nhóm"},
        {"word": "decide", "meaning": "quyết định"},
        {"word": "team", "meaning": "đội nhóm (duplicate)"},
    ]
    result = text_analyzer._deduplicate_rows(rows, ("word",))
    assert len(result) == 2
    assert result[0]["word"] == "team"
    assert result[1]["word"] == "decide"


def test_deduplicate_rows_keeps_placeholder_rows():
    rows = [
        {"word": "—", "meaning": "N/A"},
        {"word": "—", "meaning": "N/A"},
    ]
    result = text_analyzer._deduplicate_rows(rows, ("word",))
    assert len(result) == 2


def test_renumber_rows():
    rows = [{"num": "1", "word": "a"}, {"num": "1", "word": "b"}, {"num": "3", "word": "c"}]
    text_analyzer._renumber_rows(rows)
    assert [r["num"] for r in rows] == ["1", "2", "3"]


def test_parse_always_normalizes_model_vocabulary_numbers():
    response = RESPONSE.replace("| 1 | team", "| 9 | team").replace(
        "| 2 | decided", "| 2 | decided"
    ).replace("| 3 | approaching", "| 1 | approaching")

    parsed = text_analyzer.parse_analysis_response(response)

    assert [row["num"] for row in parsed["vocabulary_all"]] == ["1", "2", "3"]


def test_merge_deduplicates_vocabulary(monkeypatch):
    parsed1 = text_analyzer.parse_analysis_response(RESPONSE)
    parsed1["usage"] = {"input_tokens": 5, "output_tokens": 10, "candidate_tokens": 10, "thinking_tokens": 0}
    parsed2 = text_analyzer.parse_analysis_response(RESPONSE)
    parsed2["usage"] = {"input_tokens": 5, "output_tokens": 10, "candidate_tokens": 10, "thinking_tokens": 0}
    merged = text_analyzer._merge_analysis_results([parsed1, parsed2])
    vocab_words = [r["word"] for r in merged["vocabulary_all"]]
    assert vocab_words == ["team", "decided", "approaching"]
    assert [r["num"] for r in merged["vocabulary_all"]] == ["1", "2", "3"]


def test_analyze_single_page_and_merge(monkeypatch):
    class Model:
        def generate_content(self, _prompt, generation_config):
            return SimpleNamespace(
                text=RESPONSE,
                usage_metadata=SimpleNamespace(prompt_token_count=4, candidates_token_count=6),
            )

    model = Model()
    monkeypatch.setattr(text_analyzer, "_init_model", lambda: model)

    page = {"page_index": 1, "page_name": "test.png", "text": "Sample text.", "notes": []}
    result = text_analyzer.analyze_single_page(model, page)
    assert result["page_index"] == 1
    assert result["source_label"] == "Trang 1: test.png"
    assert result["vocabulary_all"]

    merged = text_analyzer.merge_page_analyses([result])
    assert len(merged["page_analyses"]) == 1
    assert merged["page_analyses"][0]["page_index"] == result["page_index"]
    assert "test.png" in merged["confirmed_text"]


def test_merge_page_analyses_sorts_pages_and_renumbers_combined_vocabulary():
    page_2 = text_analyzer.parse_analysis_response(RESPONSE)
    page_2.update(
        page_index=2,
        page_name="page-2.png",
        source_label="Trang 2: page-2.png",
        vocabulary_all=[{"num": "7", "word": "second", "meaning": "thứ hai"}],
    )
    page_1 = text_analyzer.parse_analysis_response(RESPONSE)
    page_1.update(
        page_index=1,
        page_name="page-1.png",
        source_label="Trang 1: page-1.png",
        vocabulary_all=[{"num": "4", "word": "first", "meaning": "đầu tiên"}],
    )

    merged = text_analyzer.merge_page_analyses([page_2, page_1])

    assert [page["page_index"] for page in merged["page_analyses"]] == [1, 2]
    assert [row["word"] for row in merged["vocabulary_all"]] == ["first", "second"]
    assert [row["num"] for row in merged["vocabulary_all"]] == ["1", "2"]
