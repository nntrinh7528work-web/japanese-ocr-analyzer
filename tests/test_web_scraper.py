"""Unit tests for web_scraper — no real HTTP calls."""

import pytest
from unittest.mock import patch, MagicMock
from modules.web_scraper import detect_language, fetch_article, _has_japanese, _scrape_with_bs4


def test_detect_japanese():
    assert detect_language("これは日本語のテキストです。") == "ja"


def test_detect_english():
    assert detect_language("This is an English article about technology.") == "en"


def test_has_japanese_true():
    assert _has_japanese("東京は大きい都市です") is True


def test_has_japanese_false():
    assert _has_japanese("Hello world") is False


def test_scrape_bs4_extracts_paragraphs():
    html = "<html><body><article><p>This is the first paragraph of the article body.</p><p>This is the second paragraph of the article body.</p></article></body></html>"
    result = _scrape_with_bs4(html)
    assert "first paragraph" in result
    assert "second paragraph" in result


def test_scrape_bs4_removes_nav():
    html = "<html><body><nav>Menu</nav><article><p>This is the main article text that should be kept.</p></article></body></html>"
    result = _scrape_with_bs4(html)
    assert "Menu" not in result
    assert "main article text" in result


def test_fetch_article_invalid_url():
    with pytest.raises(ValueError, match="http"):
        fetch_article("not-a-url")


def test_fetch_article_uses_trafilatura(monkeypatch):
    mock_result = {
        "title": "Test Title",
        "clean_text": "This is a test article with enough content to pass validation checks. " * 3,
        "lang": "en",
        "source_url": "https://example.com/article",
        "word_count": 39,
        "char_count": 210,
    }
    with patch("modules.web_scraper._scrape_with_trafilatura", return_value=mock_result["clean_text"]):
        with patch("requests.get") as mock_get:
            mock_get.return_value.text = "<html><head><title>Test Title</title></head></html>"
            mock_get.return_value.status_code = 200
            result = fetch_article("https://example.com/article")
            assert result["lang"] == "en"
            assert len(result["clean_text"]) > 10


def test_fetch_article_short_content_raises():
    with patch("modules.web_scraper._scrape_with_trafilatura", return_value="too short"):
        with patch("requests.get") as mock_get:
            mock_get.return_value.text = "<html><body><p>x</p></body></html>"
            mock_get.return_value.status_code = 200
            mock_get.return_value.encoding = "utf-8"
            mock_get.return_value.apparent_encoding = "utf-8"
            with pytest.raises(RuntimeError):
                fetch_article("https://example.com/short")
