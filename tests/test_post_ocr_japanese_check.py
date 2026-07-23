"""Tests for post-OCR Japanese check module."""

from modules.post_ocr_japanese_check import analyze_post_ocr_japanese


def test_analyze_post_ocr_japanese_empty():
    res = analyze_post_ocr_japanese("")
    assert res["tokens"] == []
    assert res["pos_tags"] == []
    assert res["contains_japanese"] is False
    assert res["token_count"] == 0


def test_analyze_post_ocr_japanese_simple():
    text = "こんにちは世界"
    res = analyze_post_ocr_japanese(text)
    
    assert res["contains_japanese"] is True
    assert res["token_count"] > 0
    assert isinstance(res["tokens"], list)
    assert len(res["tokens"]) == res["token_count"]
    assert isinstance(res["pos_tags"], list)


def test_analyze_post_ocr_not_japanese():
    text = "Hello world 123!"
    res = analyze_post_ocr_japanese(text)
    
    assert res["contains_japanese"] is False
    assert isinstance(res["tokens"], list)


def test_conditional_japanese_check():
    from modules.post_ocr_japanese_check import conditional_japanese_check
    
    # Branch 1: text is Japanese
    res_ja = conditional_japanese_check("こんにちは", "japanese")
    assert res_ja is not None
    assert res_ja["contains_japanese"] is True
    
    # Branch 2: text is not Japanese (e.g. English)
    res_en = conditional_japanese_check("こんにちは", "english")
    assert res_en is None
