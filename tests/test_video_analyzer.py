from __future__ import annotations

import pytest

from modules.video_analyzer import (
    build_cost_estimate,
    build_segment_batches,
    build_segments,
    build_video_analysis,
    clean_transcript,
    normalize_transcript,
    parse_youtube_url,
    transcript_hash,
    supported_video_model,
    validate_video_upload,
)
from modules.notion_sync import extract_notion_entities


@pytest.mark.parametrize(
    ("url", "video_id"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_parse_youtube_urls(url, video_id):
    assert parse_youtube_url(url)["video_id"] == video_id


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=bad",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123",
    ],
)
def test_rejects_unsafe_or_unsupported_youtube_urls(url):
    with pytest.raises(ValueError):
        parse_youtube_url(url)


def test_upload_validation_checks_extension_mime_and_size():
    assert validate_video_upload("lesson.mp4", b"video", "video/mp4")["suffix"] == "mp4"
    with pytest.raises(ValueError):
        validate_video_upload("lesson.exe", b"video", "video/mp4")
    with pytest.raises(ValueError):
        validate_video_upload("lesson.mp4", b"video", "application/octet-stream")


def test_transcript_normalization_segmentation_and_stable_hash():
    rows = normalize_transcript([
        {"start": 0, "duration": 4, "text": " 今日は  学びます "},
        {"start": 181, "end": 185, "text": "This is the next topic."},
    ])
    assert rows[0]["text"] == "今日は 学びます"
    assert transcript_hash(rows) == transcript_hash(list(rows))
    segments = build_segments(rows)
    assert [row["ordinal"] if "ordinal" in row else index for index, row in enumerate(segments, 1)] == [1, 2]
    assert [row["language"] for row in segments] == ["japanese", "english"]


def test_transcript_cleaner_keeps_raw_available_but_removes_safe_repeats():
    raw = normalize_transcript([
        {"start": 0, "end": 1, "text": "um"},
        {"start": 1, "end": 2, "text": "Hello world"},
        {"start": 2, "end": 3, "text": "Hello world"},
    ])
    cleaned, warnings = clean_transcript(raw)
    assert [row["text"] for row in raw] == ["um", "Hello world", "Hello world"]
    assert [row["text"] for row in cleaned] == ["Hello world"]
    assert len(warnings) == 2


def test_dynamic_batches_are_language_homogeneous_and_size_limited():
    segments = [
        {"segment_id": "a", "language": "japanese", "clean_text": "あ" * 100},
        {"segment_id": "b", "language": "japanese", "clean_text": "い" * 100},
        {"segment_id": "c", "language": "english", "clean_text": "word " * 50},
        {"segment_id": "d", "language": "english", "clean_text": "x" * 7000},
    ]
    batches = build_segment_batches(segments)
    assert [[row["segment_id"] for row in batch] for batch in batches] == [["a", "b"], ["c"], ["d"]]


def test_caption_ingest_is_free_but_analysis_has_paid_equivalent():
    source = {
        "duration_seconds": 600,
        "transcript_provider": "youtube_caption_auto",
        "clean_transcript": [{"text": "hello", "end": 600}],
    }
    estimate = build_cost_estimate(source, [{"segment_id": "one"}], "paid")
    assert estimate["ingest"]["paid_equivalent_usd"] == 0
    assert estimate["analysis_expected"]["paid_equivalent_usd"] > 0


def test_deprecated_video_batch_model_is_migrated_automatically():
    assert supported_video_model("gemini-2.5-flash-lite", "gemini-3.5-flash-lite") == "gemini-3.5-flash-lite"


def test_video_analysis_exposes_pages_for_notion_without_raw_transcript():
    source = {
        "source_id": "source",
        "source_kind": "youtube",
        "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "clean_transcript": [{"start": 0, "end": 3, "text": "Hello"}],
        "raw_transcript": [{"start": 0, "end": 3, "text": "RAW"}],
        "metadata": {"title": "Lesson"},
    }
    segments = [{
        "segment_id": "one", "ordinal": 1, "start_seconds": 0, "end_seconds": 3,
        "title": "Intro", "language": "english", "clean_text": "Hello",
        "analysis": {"summary": "Mở đầu", "vocabulary_all": [{"word": "hello", "meaning": "xin chào"}]},
        "usage": {},
    }]
    analysis = build_video_analysis(source, segments)
    assert analysis["page_analyses"][0]["source_text"] == "Hello"
    assert "raw_transcript" not in analysis
    assert "RAW" not in analysis["full_markdown"]


def test_video_analysis_normalizes_string_rows_from_model_without_crashing():
    source = {"source_id": "source", "clean_transcript": [{"text": "Hello", "end": 2}]}
    segments = [{
        "segment_id": "one", "ordinal": 1, "start_seconds": 0, "end_seconds": 2,
        "title": "Intro", "language": "english", "clean_text": "Hello.",
        "analysis": {
            "summary": "Mở đầu", "dialogue_turns": ["Hello"],
            "vocabulary_all": ["hello"], "key_points": "Lời chào",
        },
        "usage": {},
    }]
    analysis = build_video_analysis(source, segments)
    page = analysis["page_analyses"][0]
    assert page["dialogue_turns"][0]["text"] == "Hello"
    assert page["vocabulary_all"][0]["value"] == "hello"
    assert "Hello" in analysis["full_markdown"]


def test_mixed_video_routes_japanese_and_english_entities_independently():
    analysis = {
        "analysis_language": "mixed",
        "page_analyses": [
            {
                "page_index": 1, "language": "japanese", "source_text": "日本語を学びます。",
                "vocabulary_all": [{"word": "学ぶ", "reading": "まなぶ", "meaning": "học"}],
                "kanji_analysis": [{"kanji": "学", "onyomi": "ガク", "kunyomi": "まなぶ", "meaning": "học", "vocab": "学ぶ"}],
                "connectors": [{"connector": "しかし", "meaning": "tuy nhiên"}],
            },
            {
                "page_index": 2, "language": "english", "source_text": "However, we continue.",
                "vocabulary_all": [{"word": "continue", "meaning": "tiếp tục"}],
                "discourse_markers": [{"marker": "however", "meaning": "tuy nhiên"}],
            },
        ],
    }
    entities = extract_notion_entities(analysis, "lesson")
    assert {row["language"] for row in entities["vocabulary"]} == {"japanese", "english"}
    assert entities["kanji"][0]["title"] == "学"
    assert {row["title"] for row in entities["language_items"]} == {"しかし", "however"}
