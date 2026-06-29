"""Web article scraper — extracts clean article text from news URLs."""

from __future__ import annotations

import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from langdetect import detect as _detect_lang
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ArticleAnalyzer/1.0)",
    "Accept-Language": "ja,en;q=0.9",
}


def _has_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text))


def detect_language(text: str) -> str:
    if _has_japanese(text):
        return "ja"
    if HAS_LANGDETECT:
        try:
            lang = _detect_lang(text[:2000])
            return lang if lang in ("ja", "en") else "en"
        except Exception:
            pass
    return "en"


def _scrape_with_trafilatura(url: str) -> str | None:
    if not HAS_TRAFILATURA:
        return None
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None
    return trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )


def _scrape_with_bs4(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    article = soup.find("article") or soup.find("main") or soup.find("body")
    if not article:
        return soup.get_text(separator="\n")
    paragraphs = article.find_all(["p", "h1", "h2", "h3"])
    return "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))


def _get_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()
    title_tag = soup.find("title")
    return title_tag.get_text(strip=True) if title_tag else ""


def fetch_article(url: str) -> dict[str, Any]:
    """Fetch and extract article content from a URL.

    Returns:
        dict with keys: title, clean_text, lang, source_url, word_count, char_count
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL phải bắt đầu bằng http:// hoặc https://")

    clean_text = _scrape_with_trafilatura(url)
    title_text = ""

    if not clean_text or len(clean_text.strip()) < 100:
        last_error = None
        for attempt in range(2):
            try:
                response = requests.get(url, headers=HEADERS, timeout=15)
                response.encoding = response.apparent_encoding
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")
                clean_text = _scrape_with_bs4(response.text)
                title_text = _get_title(response.text)
                break
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    time.sleep(1)
        if not clean_text and last_error:
            raise RuntimeError(f"Không thể tải trang: {last_error}")
    else:
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            title_text = _get_title(response.text)
        except Exception:
            title_text = ""

    if not clean_text or len(clean_text.strip()) < 50:
        raise RuntimeError(
            "Không thể trích xuất nội dung. "
            "Trang này có thể yêu cầu đăng nhập (paywall) hoặc dùng JavaScript động."
        )

    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text.strip())
    lang = detect_language(clean_text)

    return {
        "title": title_text,
        "clean_text": clean_text,
        "lang": lang,
        "source_url": url,
        "word_count": len(clean_text.split()),
        "char_count": len(clean_text),
    }
