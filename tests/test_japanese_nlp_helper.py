"""Tests for Japanese NLP helper module."""

from modules.japanese_nlp_helper import tokenize_japanese, pos_tag_japanese


def test_tokenize_japanese_empty():
    assert tokenize_japanese("") == []
    assert tokenize_japanese("   ") == []


def test_tokenize_japanese_simple():
    text = "今日はいい天気ですね"
    tokens = tokenize_japanese(text)
    assert len(tokens) > 0
    assert "今日" in tokens


def test_pos_tag_japanese_empty():
    assert pos_tag_japanese("") == []


def test_pos_tag_japanese_simple():
    text = "今日はいい天気ですね"
    tags = pos_tag_japanese(text)
    assert len(tags) > 0
    assert isinstance(tags[0], dict)
    assert "surface" in tags[0]
    assert "pos" in tags[0]
    assert tags[0]["surface"] == "今日"
