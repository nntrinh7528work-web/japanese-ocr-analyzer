"""Web article scraper — 4-tier fallback chain for accurate news extraction."""

from __future__ import annotations

import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

# Tier 1: Fundus — highest accuracy for major news sites
try:
    from fundus import PublicCorpus, Crawler
    HAS_FUNDUS = True
except ImportError:
    HAS_FUNDUS = False

# Tier 2: Newspaper4k — broad coverage + auto NLP
try:
    from newspaper import Article as NewspaperArticle
    HAS_NEWSPAPER = True
except ImportError:
    HAS_NEWSPAPER = False

# Tier 3: Trafilatura
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

# Tier 4: readability-lxml
try:
    from readability import Document
    HAS_READABILITY = True
except ImportError:
    HAS_READABILITY = False

try:
    from langdetect import detect as _detect_lang
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

MIN_CONTENT_LENGTH = 150  # ký tự tối thiểu để coi là hợp lệ


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


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize newlines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_valid(text: str | None) -> bool:
    return bool(text and len(text.strip()) >= MIN_CONTENT_LENGTH)


# ── Tier 1: Fundus ──────────────────────────────────────────
def _try_fundus(url: str) -> tuple[str, str]:
    """Returns (title, text). Raises if not supported or failed."""
    if not HAS_FUNDUS:
        raise RuntimeError("fundus not installed")
    from fundus import Crawler, PublicCorpus
    crawler = Crawler(PublicCorpus.ALL)
    articles = list(crawler.crawl(max_articles=1, url_filter=lambda u: u == url))
    if not articles:
        raise RuntimeError("Fundus: no articles found for this URL")
    a = articles[0]
    text = a.plaintext or ""
    if not _is_valid(text):
        raise RuntimeError("Fundus: content too short")
    return getattr(a, "title", ""), text


# ── Tier 2: Newspaper4k ──────────────────────────────────────
def _try_newspaper(url: str) -> tuple[str, str]:
    if not HAS_NEWSPAPER:
        raise RuntimeError("newspaper4k not installed")
    article = NewspaperArticle(url, language="ja" if "nhk" in url or ".jp" in url else "en")
    article.download()
    article.parse()
    text = article.text or ""
    if not _is_valid(text):
        raise RuntimeError("Newspaper4k: content too short")
    return article.title or "", text


# ── Tier 3: Trafilatura ──────────────────────────────────────
def _try_trafilatura(url: str, html: str = "") -> tuple[str, str]:
    if not HAS_TRAFILATURA:
        raise RuntimeError("trafilatura not installed")
    if not html:
        downloaded = trafilatura.fetch_url(url)
    else:
        downloaded = html
    if not downloaded:
        raise RuntimeError("Trafilatura: could not fetch URL")
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_recall=True,      # quan trọng: lấy nhiều nội dung hơn
        no_fallback=False,
    )
    if not _is_valid(text):
        raise RuntimeError("Trafilatura: content too short")
    # Get title separately
    meta = trafilatura.extract_metadata(downloaded)
    title = meta.title if meta else ""
    return title or "", text


# ── Tier 4: readability-lxml ─────────────────────────────────
def _try_readability(html: str) -> tuple[str, str]:
    if not HAS_READABILITY:
        raise RuntimeError("readability-lxml not installed")
    doc = Document(html)
    title = doc.title() or ""
    content_html = doc.summary()
    soup = BeautifulSoup(content_html, "html.parser")
    text = soup.get_text(separator="\n")
    text = _clean_text(text)
    if not _is_valid(text):
        raise RuntimeError("Readability: content too short")
    return title, text


def _get_html(url: str) -> tuple[str, str]:
    """Fetch raw HTML + detect title. Returns (html, title)."""
    for attempt in range(2):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.encoding = response.apparent_encoding
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            soup = BeautifulSoup(response.text, "html.parser")
            og = soup.find("meta", property="og:title")
            title = (og.get("content") if og else None) or (
                soup.title.get_text(strip=True) if soup.title else ""
            )
            return response.text, title
        except Exception as exc:
            if attempt == 0:
                time.sleep(1.5)
            else:
                raise RuntimeError(f"Không thể tải HTML: {exc}") from exc
    return "", ""


def fetch_article(url: str) -> dict[str, Any]:
    """
    Fetch article using 4-tier fallback chain:
    Fundus → Newspaper4k → Trafilatura → readability-lxml
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL phải bắt đầu bằng http:// hoặc https://")

    title, text = "", ""
    errors = []

    # Tier 1 — Fundus
    try:
        title, text = _try_fundus(url)
    except Exception as e:
        errors.append(f"Fundus: {e}")

    # Tier 2 — Newspaper4k
    if not _is_valid(text):
        try:
            title, text = _try_newspaper(url)
        except Exception as e:
            errors.append(f"Newspaper4k: {e}")

    # Tier 3 — Trafilatura (cần HTML)
    if not _is_valid(text):
        try:
            title, text = _try_trafilatura(url)
        except Exception as e:
            errors.append(f"Trafilatura: {e}")

    # Fetch HTML for tier 4 (và lấy title nếu chưa có)
    html = ""
    if not _is_valid(text) or not title:
        try:
            html, html_title = _get_html(url)
            if not title:
                title = html_title
        except Exception as e:
            errors.append(f"HTML fetch: {e}")

    # Tier 4 — readability-lxml
    if not _is_valid(text) and html:
        try:
            _, text = _try_readability(html)
        except Exception as e:
            errors.append(f"Readability: {e}")

    # Tất cả đều thất bại
    if not _is_valid(text):
        error_summary = " | ".join(errors)
        raise RuntimeError(
            f"Không thể trích xuất nội dung bài báo.\n"
            f"Nguyên nhân có thể: paywall, trang dùng JavaScript động, hoặc cần đăng nhập.\n"
            f"Chi tiết lỗi: {error_summary}"
        )

    text = _clean_text(text)
    lang = detect_language(text)

    return {
        "title": title.strip(),
        "clean_text": text,
        "lang": lang,
        "source_url": url,
        "word_count": len(text.split()),
        "char_count": len(text),
        "extraction_method": _which_tier_succeeded(errors),
    }


def _which_tier_succeeded(errors: list[str]) -> str:
    """Infer which tier succeeded based on which ones failed."""
    failed = len(errors)
    return ["Fundus", "Newspaper4k", "Trafilatura", "readability-lxml", "unknown"][
        min(failed, 4)
    ]
