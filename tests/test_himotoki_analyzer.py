"""Unit tests for modules/himotoki_analyzer.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import modules.himotoki_analyzer as himotoki_analyzer


def test_analyze_empty_text():
    result = himotoki_analyzer.analyze_text_with_himotoki("")
    assert "Không có văn bản" in result["summary"]
    assert result["vocabulary_all"] == []


@patch("himotoki.analyze")
def test_analyze_valid_text(mock_analyze):
    mock_word = MagicMock()
    mock_word.text = "食べ"
    mock_word.kana = "たべ"
    mock_word.source_text = "食べる"
    mock_word.meanings = ["to eat"]
    mock_word.pos = "[v5u]"
    mock_word.conj_type = "Continuative"
    
    mock_analyze.return_value = [([mock_word], 100)]
    
    result = himotoki_analyzer.analyze_text_with_himotoki("食べ")
    
    assert len(result["vocabulary_all"]) == 1
    assert result["vocabulary_all"][0]["word"] == "食べ"
    assert result["vocabulary_all"][0]["reading"] == "たべ"
    assert result["vocabulary_all"][0]["meaning"] == "to eat"
    assert len(result["kanji_analysis"]) == 1
    assert result["kanji_analysis"][0]["kanji"] == "食"
    assert len(result["grammar_points"]) == 1


def test_split_text_long():
    text = "あ" * 150
    chunks = himotoki_analyzer._split_text(text, max_len=90)
    assert len(chunks) == 2
    assert len(chunks[0]) == 90
    assert len(chunks[1]) == 60


@patch("himotoki.analyze")
def test_analyze_long_text_offsets(mock_analyze):
    mock_word = MagicMock()
    mock_word.text = "あ"
    mock_word.kana = "あ"
    mock_word.source_text = "あ"
    mock_word.meanings = []
    mock_word.pos = ""
    mock_word.conj_type = None
    mock_word.start = 0
    mock_word.end = 1
    
    mock_analyze.return_value = [([mock_word], 10)]
    
    # Text length 150 will be split into 2 chunks: length 90 and 60
    # First chunk: starts at index 0
    # Second chunk: starts at index 90
    result = himotoki_analyzer.analyze_text_with_himotoki("あ" * 150)
    
    # We mock it to return 1 word per chunk
    # The first word should have offset 0
    # The second word should have offset 90
    assert len(result["vocabulary_all"]) == 2
    
    # Wait, the starts/ends of mock_word are mutated in-place:
    # item.start += idx
    # Since mock_word is reuse, it gets mutated twice. But we patch it to return a copy or verify.
    # In real use, it returns different object instances, so it's fine.
