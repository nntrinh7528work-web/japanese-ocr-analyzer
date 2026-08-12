from __future__ import annotations

import pytest

from modules.video_analyzer import (
    build_audio_windows,
    build_cost_estimate,
    build_cue_translation_batches,
    build_segment_batches,
    build_segments,
    build_video_analysis,
    clean_transcript,
    cues_to_transcript_rows,
    estimate_audio_transcription_cost,
    merge_transcript_cues,
    normalize_transcript,
    parse_youtube_url,
    transcript_rows_to_cues,
    transcript_hash,
    supported_video_model,
    validate_video_upload,
    validate_video_duration,
)
from modules.notion_sync import build_notion_sync_payload, extract_notion_entities


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


def test_segment_ids_are_stable_but_isolated_between_video_sources():
    rows = normalize_transcript([{"start": 0, "end": 2, "text": "Hello."}])
    first = build_segments(rows, namespace="source-a")
    repeated = build_segments(rows, namespace="source-a")
    second = build_segments(rows, namespace="source-b")
    assert first[0]["segment_id"] == repeated[0]["segment_id"]
    assert first[0]["segment_id"] != second[0]["segment_id"]


def test_audio_windows_overlap_without_exceeding_duration():
    windows = build_audio_windows(601)
    assert windows == [
        {"index": 1, "start": 0.0, "end": 300.0},
        {"index": 2, "start": 298.0, "end": 598.0},
        {"index": 3, "start": 596.0, "end": 601.0},
    ]


def test_video_duration_rejects_long_or_silent_uploads():
    validate_video_duration({"duration_seconds": 1800, "has_audio": True})
    with pytest.raises(ValueError, match="30 phút"):
        validate_video_duration({"duration_seconds": 1801, "has_audio": True})
    with pytest.raises(ValueError, match="âm thanh"):
        validate_video_duration({"duration_seconds": 10, "has_audio": False})


def test_cues_round_trip_language_translation_and_stable_order():
    rows = normalize_transcript([
        {"start": 0, "end": 2, "text": "今日は", "language": "japanese", "translation_vi": "Hôm nay"},
        {"start": 2, "end": 4, "text": "we study", "language": "english", "translation_vi": "chúng ta học"},
    ])
    cues = transcript_rows_to_cues(rows, "source", "gemini_audio")
    assert [cue["language"] for cue in cues] == ["japanese", "english"]
    assert all(cue["status"] == "translated" for cue in cues)
    restored = cues_to_transcript_rows(cues)
    assert [row["translation_vi"] for row in restored] == ["Hôm nay", "chúng ta học"]


def test_overlapping_audio_cues_are_deduplicated_but_keep_translation():
    existing = [{
        "cue_id": "one", "start_seconds": 297, "end_seconds": 299,
        "source_text": "Hello world", "translation_vi": "", "language": "english",
    }]
    incoming = [{
        "cue_id": "two", "start_seconds": 298, "end_seconds": 300,
        "source_text": "Hello world", "translation_vi": "Xin chào", "language": "english",
    }]
    merged = merge_transcript_cues(existing, incoming)
    assert len(merged) == 1
    assert merged[0]["translation_vi"] == "Xin chào"
    assert merged[0]["end_seconds"] == 300


def test_translation_batches_split_on_language_and_limits():
    cues = [
        {"cue_id": "ja", "language": "japanese", "source_text": "あ" * 100},
        {"cue_id": "en", "language": "english", "source_text": "hello"},
    ]
    assert [[row["cue_id"] for row in batch] for batch in build_cue_translation_batches(cues)] == [["ja"], ["en"]]


def test_audio_transcription_estimate_uses_32_tokens_per_second():
    estimate = estimate_audio_transcription_cost(600, "paid")
    assert estimate["input_tokens"] == 19_200
    assert estimate["window_count"] == 3
    assert estimate["expected"]["paid_equivalent_usd"] > 0


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


def test_video_analysis_exports_aligned_cues_and_uses_them_for_guidance():
    source = {
        "source_id": "source", "source_kind": "upload", "transcript_provider": "gemini_audio",
        "video_cues": [{
            "cue_id": "cue-1", "start_seconds": 1, "end_seconds": 3,
            "source_text": "How are you?", "translation_vi": "Bạn khỏe không?", "language": "english",
        }],
        "clean_transcript": [{"start": 1, "end": 3, "text": "How are you?", "translation_vi": "Bạn khỏe không?"}],
    }
    segments = [{
        "segment_id": "one", "ordinal": 1, "start_seconds": 0, "end_seconds": 5,
        "title": "Intro", "language": "english", "clean_text": "How are you?",
        "analysis": {"summary": "Lời hỏi thăm"}, "usage": {},
    }]
    analysis = build_video_analysis(source, segments)
    assert analysis["video_cues"][0]["translation_vi"] == "Bạn khỏe không?"
    assert analysis["page_analyses"][0]["translation_guidance"][0]["translations"]["natural"] == "Bạn khỏe không?"
    assert "| 00:01 | How are you? | Bạn khỏe không? |" in analysis["full_markdown"]


def test_video_notion_payload_uses_video_title_and_timestamp_link():
    source = {
        "source_id": "source", "source_kind": "youtube",
        "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "metadata": {"title": "English listening lesson"},
        "video_cues": [{
            "cue_id": "cue-1", "start_seconds": 12, "end_seconds": 15,
            "source_text": "Listen carefully.", "translation_vi": "Hãy nghe kỹ.",
            "language": "english",
        }],
    }
    segments = [{
        "segment_id": "one", "ordinal": 1, "start_seconds": 10, "end_seconds": 20,
        "title": "Intro", "language": "english", "clean_text": "Listen carefully.",
        "analysis": {"summary": "Luyện nghe."}, "usage": {},
    }]
    analysis = build_video_analysis(source, segments)
    payload = build_notion_sync_payload(
        "session", [{"id": "one", "name": "Đoạn 1", "edited_text": "Listen carefully."}],
        analysis, document_id="video-document",
    )
    assert payload["title"] == "English listening lesson"
    assert payload["language"] == "english"
    assert "Mở video tại câu này" in payload["markdown"]
    assert "t=12s" in payload["markdown"]


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
