"""Tests for the analysis pipeline — all API calls are mocked."""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

import modules.analysis_pipeline as pipeline

# ----------- Shared test data -----------

_BASE_ANALYSIS = {
    "language": "ja",
    "summary_vi": "tóm tắt",
    "ocr_corrections": [
        {"original": "tset", "corrected": "test", "reason_vi": "OCR lỗi"},
    ],
    "vocabulary": [
        {
            "term": "歩いて",
            "reading_or_ipa": "あるいて",
            "part_of_speech": "動詞",
            "meaning_vi": "đi bộ",
            "level": "N4",
            "example_from_text": "駅まで歩いて",
        },
        {
            "term": "会社",
            "reading_or_ipa": "かいしゃ",
            "part_of_speech": "名詞",
            "meaning_vi": "công ty",
            "level": "N4",
            "example_from_text": "会社へ行きます",
        },
    ],
    "grammar": [
        {
            "name": "て形接続",
            "structure": "V-て + V",
            "meaning_vi": "nối hai hành động",
            "usage_vi": "liên tiếp",
            "level": "N4",
            "example_from_text": "歩いて会社へ行きます",
        },
        {
            "name": "へ方向",
            "structure": "N + へ",
            "meaning_vi": "hướng đi",
            "usage_vi": "chỉ hướng",
            "level": "N5",
            "example_from_text": "会社へ行きます",
        },
    ],
    "kanji": [
        {
            "kanji": "駅",
            "reading": "えき",
            "meaning_vi": "ga tàu",
            "level": "N4",
            "word_from_text": "駅まで",
        },
    ],
    "connectors": [],
    "sentence_patterns": [],
}


# ---------------------------------------------------------------------------
# A. replace vocabulary:0 meaning_vi — input unchanged
# ---------------------------------------------------------------------------
def test_replace_vocabulary_meaning():
    """Replace meaning_vi on vocabulary:0; original dict is unchanged."""
    original = copy.deepcopy(_BASE_ANALYSIS)
    review = {
        "verdict": "needs_fix",
        "confidence": 0.9,
        "issues": [
            {
                "item_id": "vocabulary:0",
                "field": "meaning_vi",
                "problem_vi": "nghĩa sai",
                "correct_value": "bước đi",
                "action": "replace",
            }
        ],
        "missing_items": [],
        "review_note_vi": "1 sửa.",
    }

    corrected, warnings = pipeline.apply_review_fixes(original, review)

    assert corrected["vocabulary"][0]["meaning_vi"] == "bước đi"
    # Original must be untouched.
    assert original["vocabulary"][0]["meaning_vi"] == "đi bộ"
    assert not warnings


# ---------------------------------------------------------------------------
# B. remove grammar:0
# ---------------------------------------------------------------------------
def test_remove_grammar():
    """Remove grammar:0 correctly."""
    original = copy.deepcopy(_BASE_ANALYSIS)
    review = {
        "verdict": "needs_fix",
        "confidence": 0.9,
        "issues": [
            {
                "item_id": "grammar:0",
                "field": "",
                "problem_vi": "không đúng",
                "correct_value": None,
                "action": "remove",
            }
        ],
        "missing_items": [],
        "review_note_vi": "1 xóa.",
    }

    corrected, warnings = pipeline.apply_review_fixes(original, review)

    assert len(corrected["grammar"]) == 1
    assert corrected["grammar"][0]["name"] == "へ方向"
    assert len(original["grammar"]) == 2  # Original unchanged.


# ---------------------------------------------------------------------------
# C. remove vocabulary:0 and vocabulary:1 — no index shift bug
# ---------------------------------------------------------------------------
def test_remove_two_vocabulary_items():
    """Remove two vocabulary items without index shift issues."""
    original = copy.deepcopy(_BASE_ANALYSIS)
    review = {
        "verdict": "needs_fix",
        "confidence": 0.9,
        "issues": [
            {"item_id": "vocabulary:0", "field": "", "problem_vi": "x", "correct_value": None, "action": "remove"},
            {"item_id": "vocabulary:1", "field": "", "problem_vi": "y", "correct_value": None, "action": "remove"},
        ],
        "missing_items": [],
        "review_note_vi": "2 xóa.",
    }

    corrected, warnings = pipeline.apply_review_fixes(original, review)

    assert len(corrected["vocabulary"]) == 0
    assert len(original["vocabulary"]) == 2  # Original unchanged.


# ---------------------------------------------------------------------------
# D. item_id="grammar:99" — no crash, warning generated
# ---------------------------------------------------------------------------
def test_out_of_range_index_generates_warning():
    """Out-of-range index does not crash and produces a warning."""
    original = copy.deepcopy(_BASE_ANALYSIS)
    review = {
        "verdict": "needs_fix",
        "confidence": 0.5,
        "issues": [
            {"item_id": "grammar:99", "field": "name", "problem_vi": "x", "correct_value": "y", "action": "replace"},
        ],
        "missing_items": [],
        "review_note_vi": "test.",
    }

    corrected, warnings = pipeline.apply_review_fixes(original, review)

    assert len(warnings) > 0
    assert "99" in warnings[0]
    # Analysis unchanged.
    assert corrected == original


# ---------------------------------------------------------------------------
# E. review_unavailable -> keeps DeepSeek analysis, status=review_unavailable
# ---------------------------------------------------------------------------
def test_review_unavailable_keeps_original():
    """review_unavailable preserves original DeepSeek analysis."""
    original = copy.deepcopy(_BASE_ANALYSIS)
    unavailable_review = {
        "verdict": "review_unavailable",
        "confidence": 0.0,
        "issues": [],
        "missing_items": [],
        "review_note_vi": "Không thể hoàn tất bước kiểm tra Gemini.",
    }

    with patch.object(pipeline, "analyze_with_deepseek", return_value=original):
        with patch.object(pipeline, "review_deepseek_analysis", return_value=unavailable_review):
            result = pipeline.run_verified_analysis("テスト", "ja")

    assert result["quality_status"] == "review_unavailable"
    assert result["analysis"] == original


# ---------------------------------------------------------------------------
# F. missing_items not inserted into vocabulary/grammar/kanji
# ---------------------------------------------------------------------------
def test_missing_items_not_inserted():
    """missing_items from review are not added to the analysis."""
    original = copy.deepcopy(_BASE_ANALYSIS)
    original_vocab_len = len(original["vocabulary"])
    review = {
        "verdict": "needs_fix",
        "confidence": 0.8,
        "issues": [],
        "missing_items": [
            {"category": "vocabulary", "term_or_name": "走る", "reason_vi": "thiếu từ"},
        ],
        "review_note_vi": "thiếu 1 từ.",
    }

    corrected, warnings = pipeline.apply_review_fixes(original, review)

    assert len(corrected["vocabulary"]) == original_vocab_len
    assert len(corrected["grammar"]) == len(original["grammar"])
    assert len(corrected["kanji"]) == len(original["kanji"])


# ---------------------------------------------------------------------------
# G. run_verified_analysis: verified when no issues, corrected when valid fix
# ---------------------------------------------------------------------------
def test_run_verified_analysis_verified():
    """run_verified_analysis returns 'verified' when Gemini finds no issues."""
    ds_result = copy.deepcopy(_BASE_ANALYSIS)
    approved_review = {
        "verdict": "approved",
        "confidence": 0.95,
        "issues": [],
        "missing_items": [],
        "review_note_vi": "Tốt.",
    }

    with patch.object(pipeline, "analyze_with_deepseek", return_value=ds_result):
        with patch.object(pipeline, "review_deepseek_analysis", return_value=approved_review):
            result = pipeline.run_verified_analysis("テスト", "ja")

    assert result["quality_status"] == "verified"


def test_run_verified_analysis_corrected():
    """run_verified_analysis returns 'corrected' when Gemini has valid fixes."""
    ds_result = copy.deepcopy(_BASE_ANALYSIS)
    fix_review = {
        "verdict": "needs_fix",
        "confidence": 0.9,
        "issues": [
            {
                "item_id": "vocabulary:0",
                "field": "meaning_vi",
                "problem_vi": "sai",
                "correct_value": "bước đi",
                "action": "replace",
            }
        ],
        "missing_items": [],
        "review_note_vi": "1 sửa.",
    }

    with patch.object(pipeline, "analyze_with_deepseek", return_value=ds_result):
        with patch.object(pipeline, "review_deepseek_analysis", return_value=fix_review):
            result = pipeline.run_verified_analysis("テスト", "ja")

    assert result["quality_status"] == "corrected"
    assert result["analysis"]["vocabulary"][0]["meaning_vi"] == "bước đi"


def test_run_page_analyses_pipeline():
    """run_page_analyses_pipeline calls the pipeline for each page and merges the results."""
    pages = [
        {"page_index": 1, "page_name": "Page One", "text": "Văn bản trang một", "notes": []},
        {"page_index": 2, "page_name": "Page Two", "text": "Văn bản trang hai", "notes": []},
    ]

    pipeline_res_1 = {
        "analysis": {
            "language": "ja",
            "summary_vi": "Tóm tắt trang 1",
            "ocr_corrections": [],
            "vocabulary": [{"term": "từ1", "reading_or_ipa": "yomi1", "part_of_speech": "n", "meaning_vi": "nghĩa1", "level": "N3", "example_from_text": "từ1 trong bài"}],
            "grammar": [],
            "kanji": [],
            "connectors": [],
            "sentence_patterns": [],
        },
        "review": {"verdict": "approved", "confidence": 0.9, "issues": [], "missing_items": [], "review_note_vi": "ok"},
        "quality_status": "verified",
        "warnings": [],
    }
    
    pipeline_res_2 = {
        "analysis": {
            "language": "ja",
            "summary_vi": "Tóm tắt trang 2",
            "ocr_corrections": [],
            "vocabulary": [{"term": "từ2", "reading_or_ipa": "yomi2", "part_of_speech": "n", "meaning_vi": "nghĩa2", "level": "N4", "example_from_text": "từ2 trong bài"}],
            "grammar": [],
            "kanji": [],
            "connectors": [],
            "sentence_patterns": [],
        },
        "review": {"verdict": "approved", "confidence": 0.9, "issues": [], "missing_items": [], "review_note_vi": "ok"},
        "quality_status": "verified",
        "warnings": [],
    }

    with patch.object(pipeline, "run_verified_analysis") as mock_run:
        mock_run.side_effect = [pipeline_res_1, pipeline_res_2]
        
        result = pipeline.run_page_analyses_pipeline(pages, "japanese")

    assert result["summary"] == "**Trang 1: Page One:** Tóm tắt trang 1\n\n**Trang 2: Page Two:** Tóm tắt trang 2"
    assert len(result["page_analyses"]) == 2
    assert result["page_analyses"][0]["page_name"] == "Page One"
    assert result["page_analyses"][1]["page_name"] == "Page Two"
    assert result["_pipeline_result"]["quality_status"] == "verified"
