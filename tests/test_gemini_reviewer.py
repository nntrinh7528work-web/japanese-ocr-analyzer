"""Tests for Gemini reviewer — all API calls are mocked."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import modules.gemini_reviewer as reviewer

# ----------- Helpers -----------

_SAMPLE_ANALYSIS = {
    "language": "ja",
    "summary_vi": "tóm tắt",
    "ocr_corrections": [],
    "vocabulary": [{"term": "歩いて", "meaning_vi": "đi bộ"}],
    "grammar": [{"name": "て形"}],
    "kanji": [],
}

_APPROVED_REVIEW = {
    "verdict": "approved",
    "confidence": 0.95,
    "issues": [],
    "missing_items": [],
    "review_note_vi": "Phân tích chính xác, không tìm thấy lỗi.",
}

_NEEDS_FIX_REVIEW = {
    "verdict": "needs_fix",
    "confidence": 0.85,
    "issues": [
        {
            "item_id": "vocabulary:0",
            "field": "meaning_vi",
            "problem_vi": "nghĩa không chính xác",
            "correct_value": "bước đi",
            "action": "replace",
        }
    ],
    "missing_items": [],
    "review_note_vi": "Phát hiện 1 lỗi nghĩa từ vựng.",
}


def _mock_gemini_response(content: str) -> SimpleNamespace:
    """Build a mock Gemini generate_content response."""
    return SimpleNamespace(text=content)


def _make_mock_model(responses: list[str]) -> MagicMock:
    """Return a mock Gemini model that returns given response strings in order."""
    mock_model = MagicMock()
    side_effects = [_mock_gemini_response(r) for r in responses]
    mock_model.generate_content.side_effect = side_effects
    return mock_model


# ---------------------------------------------------------------------------
# A. JSON approved -> parse được
# ---------------------------------------------------------------------------
def test_approved_review():
    """Approved review JSON is parsed correctly."""
    mock_model = _make_mock_model([json.dumps(_APPROVED_REVIEW, ensure_ascii=False)])

    with patch.object(reviewer, "_init_review_model", return_value=mock_model):
        result = reviewer.review_deepseek_analysis("テスト文章", _SAMPLE_ANALYSIS, "ja")

    assert result["verdict"] == "approved"
    assert result["confidence"] == 0.95
    assert result["issues"] == []
    assert "review_note_vi" in result


# ---------------------------------------------------------------------------
# B. JSON needs_fix with issue replace -> parse đúng
# ---------------------------------------------------------------------------
def test_needs_fix_review_with_replace():
    """needs_fix review with a replace issue is parsed correctly."""
    mock_model = _make_mock_model([json.dumps(_NEEDS_FIX_REVIEW, ensure_ascii=False)])

    with patch.object(reviewer, "_init_review_model", return_value=mock_model):
        result = reviewer.review_deepseek_analysis("テスト文章", _SAMPLE_ANALYSIS, "ja")

    assert result["verdict"] == "needs_fix"
    assert len(result["issues"]) == 1
    assert result["issues"][0]["action"] == "replace"
    assert result["issues"][0]["correct_value"] == "bước đi"


# ---------------------------------------------------------------------------
# C. JSON invalid two times -> review_unavailable
# ---------------------------------------------------------------------------
def test_invalid_json_twice_returns_fallback():
    """Two invalid JSON responses return review_unavailable fallback."""
    mock_model = _make_mock_model(["NOT JSON {{", "STILL NOT JSON {{"])

    with patch.object(reviewer, "_init_review_model", return_value=mock_model):
        result = reviewer.review_deepseek_analysis("テスト文章", _SAMPLE_ANALYSIS, "ja")

    assert result["verdict"] == "review_unavailable"
    assert result["issues"] == []


# ---------------------------------------------------------------------------
# D. Assert max_output_tokens=2000, temperature=0.15
# ---------------------------------------------------------------------------
def test_generation_config_params():
    """Gemini is called with correct temperature and max_output_tokens."""
    mock_model = _make_mock_model([json.dumps(_APPROVED_REVIEW, ensure_ascii=False)])

    with patch.object(reviewer, "_init_review_model", return_value=mock_model):
        reviewer.review_deepseek_analysis("テスト文章", _SAMPLE_ANALYSIS, "ja")

    call_kwargs = mock_model.generate_content.call_args.kwargs
    gen_config = call_kwargs["generation_config"]
    assert gen_config["max_output_tokens"] == 8192
    assert gen_config["temperature"] == 0.15


# ---------------------------------------------------------------------------
# E. Assert prompt contains source_text and marker
# ---------------------------------------------------------------------------
def test_prompt_contains_source_and_marker():
    """Prompt sent to Gemini contains source_text and 'KẾT QUẢ CẦN KIỂM TRA'."""
    mock_model = _make_mock_model([json.dumps(_APPROVED_REVIEW, ensure_ascii=False)])

    with patch.object(reviewer, "_init_review_model", return_value=mock_model):
        reviewer.review_deepseek_analysis("テスト文章XYZ", _SAMPLE_ANALYSIS, "ja")

    call_args = mock_model.generate_content.call_args
    prompt_text = call_args.args[0]
    assert "テスト文章XYZ" in prompt_text
    assert "KẾT QUẢ CẦN KIỂM TRA" in prompt_text


# ---------------------------------------------------------------------------
# F. Gemini reviewer does not return/request full analysis
# ---------------------------------------------------------------------------
def test_reviewer_does_not_return_full_analysis():
    """Review output does not contain full analysis keys like vocabulary/grammar."""
    mock_model = _make_mock_model([json.dumps(_APPROVED_REVIEW, ensure_ascii=False)])

    with patch.object(reviewer, "_init_review_model", return_value=mock_model):
        result = reviewer.review_deepseek_analysis("テスト文章", _SAMPLE_ANALYSIS, "ja")

    # Review should NOT have analysis keys.
    assert "vocabulary" not in result
    assert "grammar" not in result
    assert "kanji" not in result
    assert "summary_vi" not in result
