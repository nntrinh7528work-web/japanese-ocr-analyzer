"""Tests for DeepSeek analyzer — all API calls are mocked."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import modules.deepseek_analyzer as analyzer

# ----------- Helpers -----------

VALID_JA_ANALYSIS = {
    "language": "ja",
    "summary_vi": "Đây là tóm tắt bằng tiếng Việt.",
    "ocr_corrections": [],
    "vocabulary": [
        {
            "term": "歩いて",
            "reading_or_ipa": "あるいて",
            "part_of_speech": "動詞",
            "meaning_vi": "đi bộ",
            "level": "N4",
            "example_from_text": "駅まで歩いて",
        }
    ],
    "grammar": [
        {
            "name": "て形接続",
            "structure": "V-て + V",
            "meaning_vi": "nối hai hành động",
            "usage_vi": "dùng để nối hai hành động liên tiếp",
            "level": "N4",
            "example_from_text": "歩いて会社へ行きます",
        }
    ],
    "kanji": [
        {
            "kanji": "駅",
            "reading": "えき",
            "meaning_vi": "ga tàu",
            "level": "N4",
            "word_from_text": "駅まで",
        }
    ],
}


def _mock_response(content: str) -> SimpleNamespace:
    """Build a mock chat completion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _make_mock_client(responses: list[str]) -> MagicMock:
    """Return a mock OpenAI client that returns given response strings in order."""
    mock_client = MagicMock()
    side_effects = [_mock_response(r) for r in responses]
    mock_client.chat.completions.create.side_effect = side_effects
    return mock_client


# ---------------------------------------------------------------------------
# A. Valid Japanese input -> dict with correct fields
# ---------------------------------------------------------------------------
def test_valid_japanese_input():
    """Valid Japanese text returns a dict with language='ja' and list fields."""
    mock_client = _make_mock_client([json.dumps(VALID_JA_ANALYSIS, ensure_ascii=False)])

    with patch.object(analyzer, "get_deepseek_client", return_value=mock_client):
        result = analyzer.analyze_with_deepseek("私は毎朝駅まで歩いて会社へ行きます。", "ja")

    assert result["language"] == "ja"
    assert isinstance(result["vocabulary"], list)
    assert isinstance(result["grammar"], list)
    assert isinstance(result["kanji"], list)
    assert isinstance(result["ocr_corrections"], list)


# ---------------------------------------------------------------------------
# B. language="vi" -> ValueError
# ---------------------------------------------------------------------------
def test_invalid_language_raises():
    """Unsupported language raises ValueError."""
    with pytest.raises(ValueError, match="'ja' hoặc 'en'"):
        analyzer.analyze_with_deepseek("some text", "vi")


# ---------------------------------------------------------------------------
# C. Empty source_text -> ValueError
# ---------------------------------------------------------------------------
def test_empty_text_raises():
    """Whitespace-only source text raises ValueError."""
    with pytest.raises(ValueError, match="rỗng"):
        analyzer.analyze_with_deepseek("   ", "ja")


# ---------------------------------------------------------------------------
# D. Both attempts return JSON missing 'grammar' -> RuntimeError
# ---------------------------------------------------------------------------
def test_both_attempts_invalid_raises_runtime_error():
    """Two consecutive invalid JSON responses raise RuntimeError."""
    bad_json = json.dumps({
        "language": "ja",
        "summary_vi": "tóm tắt",
        "ocr_corrections": [],
        "vocabulary": [],
        # "grammar" key deliberately missing
        "kanji": [],
    })
    mock_client = _make_mock_client([bad_json, bad_json])

    with patch.object(analyzer, "get_deepseek_client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="2 lần"):
            analyzer.analyze_with_deepseek("テスト文章", "ja")


# ---------------------------------------------------------------------------
# E. First attempt bad JSON, second attempt valid -> success, called twice
# ---------------------------------------------------------------------------
def test_retry_on_first_failure():
    """First invalid response triggers retry; second valid response succeeds."""
    bad_json = "NOT VALID JSON {{{}"
    good_json = json.dumps(VALID_JA_ANALYSIS, ensure_ascii=False)
    mock_client = _make_mock_client([bad_json, good_json])

    with patch.object(analyzer, "get_deepseek_client", return_value=mock_client):
        result = analyzer.analyze_with_deepseek("テスト", "ja")

    assert result["language"] == "ja"
    assert mock_client.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# F. Assert API call params: response_format, temperature, max_tokens
# ---------------------------------------------------------------------------
def test_api_call_params():
    """API is called with correct response_format, temperature, and max_tokens."""
    mock_client = _make_mock_client([json.dumps(VALID_JA_ANALYSIS, ensure_ascii=False)])

    with patch.object(analyzer, "get_deepseek_client", return_value=mock_client):
        analyzer.analyze_with_deepseek("テスト", "ja")

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}
    assert call_kwargs["temperature"] == 0.1
    assert call_kwargs["max_tokens"] == 30000
