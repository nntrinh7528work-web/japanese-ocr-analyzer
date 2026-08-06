import json
from types import SimpleNamespace

from modules.sentence_analyzer import (
    analysis_markdown,
    analyze_sentence_batch,
    merge_manual_breakdown,
    normalize_breakdown,
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

        def generate_content(self, prompt, generation_config):
            assert generation_config["response_mime_type"] == "application/json"
            assert "Toàn bộ giải thích" in prompt
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "sentences": [
                            {
                                "sentence_id": "p1-s2",
                                "translations": {"natural": "Câu thứ hai."},
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
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
    assert rows[0]["translations"]["natural"] == ""
    assert rows[1]["translations"]["natural"] == "Câu thứ hai."
    assert usage == {"input_tokens": 11, "output_tokens": 9, "candidate_tokens": 7, "thinking_tokens": 2}


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
