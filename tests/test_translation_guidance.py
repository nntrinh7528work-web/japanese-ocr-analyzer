import json
from types import SimpleNamespace

from modules.translation_guidance import (
    add_related_analysis,
    analyze_guidance_batch,
    apply_guidance_batch,
    guidance_batches,
    guidance_markdown,
    merge_guidance_job,
    normalize_guidance,
    related_analysis_for_sentence,
)


def _sentence(index: int, text: str | None = None) -> dict:
    return {
        "sentence_id": f"p1-s{index}",
        "ordinal": index,
        "original": text or f"Sentence number {index}.",
    }


def test_guidance_batches_respect_count_and_character_limits():
    count_batches = list(guidance_batches([_sentence(index) for index in range(1, 18)]))
    char_batches = list(
        guidance_batches(
            [_sentence(1, "A" * 3000), _sentence(2, "B" * 2500), _sentence(3, "C")]
        )
    )

    assert [len(batch) for batch in count_batches] == [8, 8, 1]
    assert [[row["sentence_id"] for row in batch] for batch in char_batches] == [
        ["p1-s1"], ["p1-s2", "p1-s3"]
    ]


def test_normalization_keeps_exact_ocr_and_limits_key_points():
    requested = _sentence(1, "OCR text stays unchanged.")
    raw = {
        "original": "Gemini tried to replace it.",
        "translations": {"natural": "Bản dịch."},
        "key_points": [{"label": str(index)} for index in range(5)],
        "ocr_warning": "Có thể kiểm tra lại từ OCR.",
    }

    row = normalize_guidance(raw, requested, "english")

    assert row["original"] == "OCR text stays unchanged."
    assert len(row["key_points"]) == 3
    assert row["ocr_warning"] == "Có thể kiểm tra lại từ OCR."
    assert set(row["translations"]) == {"chunked", "literal", "natural"}


def test_gemini_guidance_is_source_ordered_and_vietnamese_prompted():
    class Model:
        def generate_content(self, prompt, generation_config):
            assert "Giữ nguyên tuyệt đối" in prompt
            assert "tiếng Việt" in prompt
            assert generation_config["response_mime_type"] == "application/json"
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "sentences": [
                            {"sentence_id": "p1-s2", "translations": {"natural": "Câu hai."}},
                            {"sentence_id": "p1-s1", "translations": {"natural": "Câu một."}},
                        ]
                    },
                    ensure_ascii=False,
                ),
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=5,
                    thoughts_token_count=2,
                ),
            )

    rows, usage = analyze_guidance_batch(
        Model(), [_sentence(1), _sentence(2)], "Page context", "english"
    )

    assert [row["sentence_id"] for row in rows] == ["p1-s1", "p1-s2"]
    assert [row["translations"]["natural"] for row in rows] == ["Câu một.", "Câu hai."]
    assert usage["output_tokens"] == 7


def test_related_analysis_uses_word_boundaries_and_exact_examples():
    page = {
        "vocabulary_all": [
            {"word": "he", "meaning": "anh ấy"},
            {"word": "team", "meaning": "đội"},
        ],
        "vocabulary_important": [],
        "discourse_markers": [{"phrase": "because", "meaning": "bởi vì"}],
        "phrasal_collocations": [],
        "grammar_points": [
            {"name": "Past simple", "example": "The team decided because time was short.", "rule": "quá khứ"},
            {"name": "Unrelated", "example": "Another exact sentence.", "rule": "khác"},
        ],
        "sentence_patterns": [],
    }
    refs = related_analysis_for_sentence(
        "The team decided because time was short.", page, "english"
    )

    labels = [ref["label"] for ref in refs]
    assert "he" not in labels
    assert labels == ["team", "because", "Past simple"]


def test_apply_batch_and_background_merge_are_idempotent():
    page = {
        "page_index": 1,
        "translation_guidance": [],
        "translation_guidance_runs": [],
        "vocabulary_all": [],
    }
    row = normalize_guidance(
        {"translations": {"natural": "Dịch."}}, _sentence(1), "english"
    )
    applied = apply_guidance_batch(
        page, [row], {"input_tokens": 3, "output_tokens": 4}, "gemini-a", "run-1"
    )
    applied_again = apply_guidance_batch(
        applied, [row], {"input_tokens": 3, "output_tokens": 4}, "gemini-a", "run-1"
    )

    assert len(applied_again["translation_guidance_runs"]) == 1
    assert applied_again["translation_guidance_usage"]["input_tokens"] == 3

    analysis = {"analysis_language": "english", "page_analyses": [page]}
    envelope = {
        "job_kind": "translation_guidance",
        "page_index": 1,
        "analysis_language": "english",
        "model_used": "gemini-b",
        "batch_results": [
            {"batch_index": 1, "rows": [row], "usage": {"input_tokens": 5, "output_tokens": 6}, "error": None}
        ],
    }
    merged, changed = merge_guidance_job(analysis, envelope, "job-guidance")
    merged_again, changed_again = merge_guidance_job(merged, envelope, "job-guidance")

    assert changed is True
    assert changed_again is False
    assert merged_again["translation_guidance_usage"]["output_tokens"] == 6
    assert merged_again["page_analyses"][0]["translation_guidance"][0]["sentence_id"] == "p1-s1"


def test_guidance_markdown_inlines_deep_analysis_once():
    guidance = normalize_guidance(
        {"translations": {"natural": "Bản dịch."}}, _sentence(1), "english"
    )
    page = {
        "sentence_catalog": [_sentence(1)],
        "translation_guidance": [guidance],
        "sentence_breakdowns": [
            {
                "sentence_id": "p1-s1",
                "segments": [{"text": "Sentence", "role": "S", "meaning_vi": "câu"}],
                "clauses": [], "omitted_elements": [], "references": [], "logic": [], "questions": [],
                "structure_summary": "S + V",
            }
        ],
    }

    markdown = guidance_markdown(page)

    assert markdown.count("Đối chiếu OCR và giáo viên hướng dẫn dịch") == 1
    assert markdown.count("Giải mã câu dài") == 1
    assert markdown.count("Sentence number 1.") == 1


def test_guidance_markdown_renders_translation_steps_as_a_numbered_list():
    guidance = normalize_guidance(
        {
            "translations": {"natural": "Bản dịch."},
            "translation_steps": [
                {"order": 1, "source_chunk": "First", "meaning_vi": "Đầu tiên", "advice_vi": "Dịch trước"},
                {"order": 2, "source_chunk": "Second", "meaning_vi": "Tiếp theo", "advice_vi": "Dịch sau"},
            ],
        },
        _sentence(1),
        "english",
    )
    page = {
        "sentence_catalog": [_sentence(1)],
        "translation_guidance": [guidance],
        "sentence_breakdowns": [],
    }

    markdown = guidance_markdown(page)

    assert "1. `First`" in markdown
    assert "2. `Second`" in markdown
    assert "- 1. `First`" not in markdown
