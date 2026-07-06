"""Unit tests for web_scraper — no real HTTP calls."""

import pytest
from unittest.mock import patch, MagicMock
from modules.web_scraper import detect_language, fetch_article, _has_japanese


def test_detect_japanese():
    assert detect_language("これは日本語のテキストです。") == "ja"


def test_detect_english():
    assert detect_language("This is an English article about technology.") == "en"


def test_has_japanese_true():
    assert _has_japanese("東京は大きい都市です") is True


def test_has_japanese_false():
    assert _has_japanese("Hello world") is False


def test_fetch_article_invalid_url():
    with pytest.raises(ValueError, match="URL"):
        fetch_article("not-a-url")


@patch("modules.web_scraper._try_fundus")
@patch("modules.web_scraper._try_newspaper")
@patch("modules.web_scraper._try_trafilatura")
@patch("modules.web_scraper._get_html")
@patch("modules.web_scraper._try_readability")
def test_fetch_article_success(
    mock_readability, mock_get_html, mock_trafilatura, mock_newspaper, mock_fundus
):
    # Set all tiers except readability to raise an exception
    mock_fundus.side_effect = Exception("Fundus error")
    mock_newspaper.side_effect = Exception("Newspaper error")
    mock_trafilatura.side_effect = Exception("Trafilatura error")
    mock_get_html.return_value = ("<html><body>Mock HTML</body></html>", "Mock Title")
    mock_readability.return_value = ("Mock Title", "This is a mock article content with sufficient length to be valid. " * 3)

    result = fetch_article("https://example.com/article")
    assert result["title"] == "Mock Title"
    assert "mock article content" in result["clean_text"]
    assert result["lang"] == "en"
    assert result["extraction_method"] == "readability-lxml"


@patch("modules.web_scraper._try_fundus")
@patch("modules.web_scraper._try_newspaper")
@patch("modules.web_scraper._try_trafilatura")
@patch("modules.web_scraper._get_html")
@patch("modules.web_scraper._try_readability")
def test_fetch_article_all_fail(
    mock_readability, mock_get_html, mock_trafilatura, mock_newspaper, mock_fundus
):
    mock_fundus.side_effect = Exception("Fundus error")
    mock_newspaper.side_effect = Exception("Newspaper error")
    mock_trafilatura.side_effect = Exception("Trafilatura error")
    mock_get_html.side_effect = Exception("HTML fetch error")
    mock_readability.side_effect = Exception("Readability error")

    with pytest.raises(RuntimeError, match="Không thể trích xuất nội dung bài báo"):
        fetch_article("https://example.com/article")
