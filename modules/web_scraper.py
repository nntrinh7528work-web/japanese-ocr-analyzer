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
    # favor_precision=True helps trafilatura focus on core text and discard secondary elements
    return trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        favor_precision=True,
    )


def _scrape_with_bs4(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    
    # 1. Decompose completely irrelevant tags
    garbage_tags = ["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript", "svg"]
    for tag in soup(garbage_tags):
        tag.decompose()
        
    # 2. Decompose elements with classes or IDs matching common boilerplate/noise patterns
    noise_pattern = re.compile(
        r"sidebar|menu|footer|header|nav|widget|ad-|advertisement|share|social|comment|feedback|"
        r"recommend|related|popular|trending|tag-cloud|newsletter|author-profile|breadcrumb|pagination|"
        r"cookie|banner|modal|popup|sub-link|button|btn|subscribe",
        re.I
    )
    
    for element in soup.find_all(attrs={"class": noise_pattern}):
        element.decompose()
    for element in soup.find_all(attrs={"id": noise_pattern}):
        element.decompose()

    # 3. Try to locate the core article body container
    article = (
        soup.find("article") 
        or soup.find("main") 
        or soup.find(class_=re.compile(r"article-body|entry-content|post-content|story-content|article-content|main-content", re.I))
    )
    
    if not article:
        article = soup.find("body") or soup

    # 4. Extract text from paragraph and heading elements
    paragraphs = article.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    
    cleaned_paras = []
    for p in paragraphs:
        text = p.get_text(strip=True)
        if not text:
            continue
        # Skip very short paragraphs (usually sharing cues or UI noise) unless they are headings
        if len(text) < 20 and not p.name.startswith("h"):
            continue
        # Skip paragraphs containing common social/action triggers
        if any(trigger in text.lower() for trigger in ["chia sẻ", "share this", "theo dõi", "đăng ký", "click here", "read more", "xem thêm"]):
            continue
        cleaned_paras.append(text)
        
    return "\n\n".join(cleaned_paras)


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
