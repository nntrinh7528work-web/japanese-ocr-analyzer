import json
from types import SimpleNamespace

from modules.sentence_analyzer import (
    analysis_markdown,
    analyze_sentence_batch,
    build_sentence_prompt,
    deep_analysis_batches,
    merge_manual_breakdown,
    normalize_breakdown,
    score_complexity,
    select_auto_sentences,
    split_sentences,
)


def test_japanese_split_respects_nested_quotes_and_stable_ids():
    text = "先生は「雨でも、試合は続ける。」と言った。私は驚いたが、理由を聞かなかった。"
    rows = split_sentences(text, "japanese", 4)

    assert [row["sentence_id"] for row in rows] == ["p4-s1", "p4-s2"]
    assert rows[0]["original"] == "先生は「雨でも、試合は続ける。」と言った。"
    assert rows[1]["ordinal"] == 2


def test_english_split_keeps_abbreviations_and_decimal_together():
    text = "Dr. Smith used e.g. 2.5 grams in the trial. However, the result changed."
    rows = split_sentences(text, "english", 2)

    assert len(rows) == 2
    assert rows[0]["original"].startswith("Dr. Smith")
    assert "e.g. 2.5" in rows[0]["original"]
    assert rows[1]["sentence_id"] == "p2-s2"


def test_ocr_line_wrap_does_not_split_one_sentence():
    rows = split_sentences(
        "Although the document is long,\nit remains one sentence because the line break came from OCR.",
        "english",
        1,
    )

    assert len(rows) == 1
    assert "long, it remains" in rows[0]["original"]


def test_blank_ocr_lines_and_commas_do_not_end_japanese_sentence():
    rows = split_sentences(
        "この技術は便利ですが、\n\n個人情報の扱いが分からないため、\n慎重に導入すべきです。次の文です。",
        "japanese",
        1,
    )

    assert len(rows) == 2
    assert rows[0]["original"] == (
        "この技術は便利ですが、 個人情報の扱いが分からないため、 慎重に導入すべきです。"
    )


def test_english_period_ends_sentence_even_when_next_sentence_is_lowercase():
    rows = split_sentences(
        "The first clause is long, and the second clause remains connected. next sentence starts lowercase.",
        "english",
        1,
    )

    assert [row["original"] for row in rows] == [
        "The first clause is long, and the second clause remains connected.",
        "next sentence starts lowercase.",
    ]


def test_japanese_ocr_ascii_period_ends_sentence_but_decimal_does_not():
    rows = split_sentences(
        "成功率は3.5パーセントです.次の説明です．最後です。",
        "japanese",
        1,
    )

    assert [row["original"] for row in rows] == [
        "成功率は3.5パーセントです.",
        "次の説明です．",
        "最後です。",
    ]


def test_mixed_page_detects_sentence_language_without_changing_ids():
    rows = split_sentences(
        "これはAPIの説明です。The result, which was tested, changed significantly.",
        "japanese",
        3,
    )

    assert [row["sentence_id"] for row in rows] == ["p3-s1", "p3-s2"]
    assert [row["detected_language"] for row in rows] == ["japanese", "english"]
    assert all(row["language_source"] == "auto" for row in rows)


def test_complexity_does_not_treat_subject_ga_and_simple_and_that_as_clauses():
    japanese_score, japanese_signals = score_complexity("私が本を読む。", "japanese")
    english_score, english_signals = score_complexity("That book and that pen are useful.", "english")

    assert japanese_score < 5
    assert "が nối mệnh đề" not in japanese_signals
    assert english_score < 5
    assert not any("mệnh đề/liên từ" in signal for signal in english_signals)


def test_deep_batches_keep_languages_separate_and_isolate_very_long_sentences():
    rows = [
        {"sentence_id": "p1-s1", "ordinal": 1, "original": "短い日本語の文です。", "detected_language": "japanese", "complexity_score": 7},
        {"sentence_id": "p1-s2", "ordinal": 2, "original": "word " * 55, "detected_language": "english", "complexity_score": 13},
        {"sentence_id": "p1-s3", "ordinal": 3, "original": "Another short but complex English sentence.", "detected_language": "english", "complexity_score": 7},
    ]

    batches = deep_analysis_batches(rows, "japanese")

    assert [(language, [row["sentence_id"] for row in batch]) for language, batch in batches] == [
        ("japanese", ["p1-s1"]), ("english", ["p1-s2"]), ("english", ["p1-s3"])
    ]


def test_prompt_contains_language_specific_v2_requirements():
    japanese = build_sentence_prompt([{"sentence_id": "p1-s1", "ordinal": 1, "original": "雨が降るので、行きません。"}], "雨が降るので、行きません。", "japanese")
    english = build_sentence_prompt([{"sentence_id": "p1-s2", "ordinal": 2, "original": "The book that I bought was expensive."}], "The book that I bought was expensive.", "english")

    assert "hiragana" in japanese.lower()
    assert "trợ từ" in japanese
    assert "S-V-O-C-A" in english
    assert "relative" in english
    assert "sentence_breakdown_version" not in japanese


def test_selection_enforces_page_and_document_caps_with_source_tie_break():
    catalog = {}
    for page_index in range(1, 8):
        catalog[page_index] = [
            {
                "sentence_id": f"p{page_index}-s{ordinal}",
                "ordinal": ordinal,
                "original": "complex",
                "complexity_score": 10,
                "eligible": True,
                "selected_auto": False,
            }
            for ordinal in range(1, 5)
        ]

    selected = select_auto_sentences(catalog)

    assert sum(len(rows) for rows in selected.values()) == 15
    assert all(len(rows) <= 3 for rows in selected.values())
    assert [row["sentence_id"] for row in selected[1]] == ["p1-s1", "p1-s2", "p1-s3"]
    assert set(selected) == {1, 2, 3, 4, 5}


def test_normalization_supplies_all_eight_layers_for_partial_json():
    requested = {
        "sentence_id": "p1-s1",
        "ordinal": 1,
        "original": "雨が降っているので、出かけません。",
        "complexity_score": 8,
    }
    row = normalize_breakdown(
        {"translations": {"natural": "Vì trời mưa nên tôi không ra ngoài."}},
        requested,
        "japanese",
        "auto",
    )

    assert row["sentence_id"] == "p1-s1"
    assert set(row["translations"]) == {"chunked", "literal", "natural"}
    for field in (
        "segments", "clauses", "structure_summary", "omitted_elements", "references",
        "logic", "simplified_source", "simplified_vi", "questions",
    ):
        assert field in row


def test_sentence_batch_preserves_requested_order_and_normalizes_missing_row():
    class Model:
        target_model_name = "gemini-test"
        calls = 0

        def generate_content(self, prompt, generation_config):
            assert generation_config["response_mime_type"] == "application/json"
            self.calls += 1
            if self.calls == 1:
                assert "Toàn bộ giải thích" in prompt
                payload = {
                    "sentences": [
                        {"sentence_id": "p1-s2", "translations": {"natural": "Câu thứ hai."}},
                    ]
                }
            else:
                assert "chỉ bổ sung các trường thiếu" in prompt.lower()
                payload = {
                    "sentences": [
                        {
                            "sentence_id": sentence_id,
                            "segments": [{"text": "sentence", "role": "S", "meaning_vi": "câu"}],
                            "structure_summary": "S + V",
                            "sentence_skeleton": {"pattern": "S + V", "predicate": "is"},
                            "grammar_links": [{"source": "is", "form": "be", "function_vi": "động từ nối"}],
                            "translations": {"literal": "Câu.", "natural": "Đây là câu."},
                            "translation_steps": [{"order": "1", "source_chunk": "sentence", "meaning_vi": "câu", "advice_vi": "dịch"}],
                            "questions": [{"question": "Gì?", "answer": "Câu", "explanation": ""}],
                        }
                        for sentence_id in ("p1-s1", "p1-s2")
                    ]
                }
            return SimpleNamespace(
                text=json.dumps(payload, ensure_ascii=False),
                usage_metadata=SimpleNamespace(
                    prompt_token_count=11,
                    candidates_token_count=7,
                    thoughts_token_count=2,
                ),
            )

    requested = [
        {"sentence_id": "p1-s1", "ordinal": 1, "original": "First long sentence.", "complexity_score": 5},
        {"sentence_id": "p1-s2", "ordinal": 2, "original": "Second long sentence.", "complexity_score": 6},
    ]
    rows, usage = analyze_sentence_batch(Model(), requested, "Context", "english")

    assert [row["sentence_id"] for row in rows] == ["p1-s1", "p1-s2"]
    assert all(row["quality_status"] == "complete" for row in rows)
    assert rows[1]["translations"]["natural"] == "Đây là câu."
    assert rows[0]["analysis_usage_detail"]["primary"]["input_tokens"] == 11
    assert rows[0]["analysis_usage_detail"]["repair"]["input_tokens"] == 11
    assert usage == {"input_tokens": 22, "output_tokens": 18, "candidate_tokens": 14, "thinking_tokens": 4}


def test_failed_repair_keeps_the_primary_partial_breakdown():
    class Model:
        def generate_content(self, prompt, generation_config):
            if "chỉ bổ sung các trường thiếu" in prompt.lower():
                raise RuntimeError("temporary repair outage")
            return SimpleNamespace(
                text=json.dumps({"sentences": [{"sentence_id": "p1-s1", "translations": {"natural": "Bản dịch."}}]}),
                usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=2, thoughts_token_count=0),
            )

    rows, usage = analyze_sentence_batch(
        Model(), [{"sentence_id": "p1-s1", "ordinal": 1, "original": "A difficult sentence."}], "Context", "english"
    )

    assert rows[0]["quality_status"] == "partial"
    assert rows[0]["quality_repair_error"] == "temporary repair outage"
    assert rows[0]["translations"]["natural"] == "Bản dịch."
    assert rows[0]["analysis_usage_detail"]["repair"]["input_tokens"] == 0
    assert usage["input_tokens"] == 3


def test_manual_merge_is_idempotent_and_updates_only_target_page():
    analysis = {
        "page_analyses": [
            {
                "page_index": 1,
                "sentence_catalog": [{"sentence_id": "p1-s1", "analyzed": False}],
                "sentence_breakdowns": [],
                "sentence_analysis_usage": {},
            },
            {"page_index": 2, "sentence_breakdowns": [], "sentence_analysis_usage": {}},
        ]
    }
    envelope = {
        "job_kind": "sentence_deep_dive",
        "page_index": 1,
        "sentence_id": "p1-s1",
        "breakdown": {"sentence_id": "p1-s1", "ordinal": 1, "analysis_origin": "manual"},
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "model_used": "gemini-test",
    }

    merged, changed = merge_manual_breakdown(analysis, envelope, "job-1")
    again, changed_again = merge_manual_breakdown(merged, envelope, "job-1")

    assert changed is True
    assert changed_again is False
    assert again["sentence_analysis_usage"]["input_tokens"] == 10
    assert len(again["page_analyses"][0]["sentence_breakdowns"]) == 1
    assert again["page_analyses"][0]["sentence_analysis_model"] == "gemini-test"
    assert again["sentence_analysis_runs"] == [
        {
            "run_id": "job-1", "origin": "manual", "model_used": "gemini-test",
            "usage": {"input_tokens": 10, "output_tokens": 20, "candidate_tokens": 0, "thinking_tokens": 0},
        }
    ]
    assert again["page_analyses"][1]["sentence_breakdowns"] == []


def test_dynamic_markdown_contains_auto_and_manual_once():
    page = {
        "source_label": "Trang 1: test",
        "full_markdown": "## Phân tích chính",
        "sentence_breakdowns": [
            {
                "sentence_id": "p1-s1",
                "ordinal": 1,
                "original": "A long sentence.",
                "analysis_origin": "auto",
                "translations": {"chunked": "A / long sentence", "literal": "Một câu dài", "natural": "Đây là câu dài."},
                "segments": [], "clauses": [], "omitted_elements": [], "references": [], "logic": [], "questions": [],
            },
            {
                "sentence_id": "p1-s2",
                "ordinal": 2,
                "original": "Another sentence.",
                "analysis_origin": "manual",
                "translations": {}, "segments": [], "clauses": [], "omitted_elements": [], "references": [], "logic": [], "questions": [],
            },
        ],
    }
    markdown = analysis_markdown({"page_analyses": [page]})

    assert markdown.count("## Giải mã câu dài") == 1
    assert "Câu 1 - Tự động" in markdown
    assert "Câu 2 - Phân tích thêm" in markdown
