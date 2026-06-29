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

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en;q=0.9",
}

# ── helpers ──────────────────────────────────────────────────────────────

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


# ── inner noise / stop‑word patterns ────────────────────────────────────

_INNER_NOISE_RE = re.compile(
    r"share|social|comment|relation|keyword|recommend|subscribe|newsletter|"
    r"advertisement|banner|popup|author[_-]bio|breadcrumb|pagination|"
    r"tag[_-]list|more[_-]articles|read[_-]next|popular|ranking|pickup|"
    r"latest|utility|sidebar|widget|cookie|"
    r"関連|注目|おすすめ|アクセス|ランキング|シェア|話題",
    re.I,
)

_STOP_PHRASES = [
    "あわせて読みたい", "関連記事", "関連ニュース", "注目ワード",
    "おすすめ記事", "関連リンク", "おすすめのニュース", "話題のワード",
    "アクセスランキング", "深掘りコンテンツ", "もっと読む",
    "related stories", "related articles", "read more",
    "you may also like", "recommended for you", "more on this story",
]


# ── BS4 scraper (static HTML) ───────────────────────────────────────────

def _scrape_with_bs4(html: str) -> str:
    """Extract article body from *static* HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    # 1. remove completely irrelevant tags
    for tag in soup(["script", "style", "form", "iframe", "noscript", "svg"]):
        tag.decompose()

    # 2. locate article container
    container = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_=re.compile(
            r"article[_-]?body|entry[_-]?content|post[_-]?content|"
            r"story[_-]?body|article[_-]?content|main[_-]?content|"
            r"post[_-]?body|news[_-]?body|article[_-]?text|"
            r"content[_-]?body|story[_-]?text",
            re.I,
        ))
        or soup.find(id=re.compile(
            r"article[_-]?body|entry[_-]?content|post[_-]?content|"
            r"article[_-]?content|main[_-]?content|post[_-]?body|"
            r"news[_-]?body|article[_-]?text|content[_-]?body",
            re.I,
        ))
    )

    if container:
        # remove noise *inside* the article container only
        for el in container.find_all(attrs={"class": _INNER_NOISE_RE}):
            el.decompose()
        for el in container.find_all(attrs={"id": _INNER_NOISE_RE}):
            el.decompose()
        for tag in container(["nav", "footer", "header", "aside"]):
            tag.decompose()
    else:
        # fallback: clean whole page then use <body>
        for tag in soup(["nav", "footer", "header", "aside"]):
            tag.decompose()
        for el in soup.find_all(attrs={"class": _INNER_NOISE_RE}):
            el.decompose()
        for el in soup.find_all(attrs={"id": _INNER_NOISE_RE}):
            el.decompose()
        container = soup.find("body") or soup

    # 3. truncate at "related articles" / boilerplate sections
    _truncate_at_boilerplate(container)

    # 4. collect paragraphs & headings
    return _collect_text(container)


def _truncate_at_boilerplate(container) -> None:
    """Remove everything after a boilerplate heading like 'あわせて読みたい'."""
    for tag in container.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "div", "span"]
    ):
        text = tag.get_text(strip=True)
        if len(text) > 100:
            continue
        low = text.lower()
        if any(phrase in low for phrase in _STOP_PHRASES):
            for sib in list(tag.next_siblings):
                if hasattr(sib, "decompose"):
                    sib.decompose()
            tag.decompose()
            break


def _collect_text(container) -> str:
    """Return cleaned paragraph text from *container*."""
    paras: list[str] = []
    seen: set[str] = set()            # de‑duplicate
    for el in container.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"]):
        text = el.get_text(strip=True)
        if not text or text in seen:
            continue
        # skip very short non‑headings (usually UI noise)
        if len(text) < 10 and not el.name.startswith("h"):
            continue
        # skip if the <p> is inside an <li> that looks like a link list
        parent_li = el.find_parent("li")
        if parent_li and parent_li.find("a"):
            # likely a "related article" list item — skip
            continue
        seen.add(text)
        paras.append(text)
    return "\n\n".join(paras)


# ── Playwright scraper (JS‑rendered pages) ──────────────────────────────

def _scrape_with_playwright(url: str) -> str | None:
    """Render a page with headless Chromium, then extract article text."""
    if not HAS_PLAYWRIGHT:
        return None
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({"Accept-Language": "ja,en;q=0.9"})
            page.goto(url, wait_until="networkidle", timeout=30_000)
            # some NHK pages show a consent/login overlay — try to dismiss it
            try:
                accept_btn = page.locator(
                    "button:has-text('確認'), "
                    "button:has-text('同意'), "
                    "button:has-text('閉じる'), "
                    "button:has-text('次へ')"
                ).first
                if accept_btn.is_visible(timeout=2000):
                    accept_btn.click()
                    page.wait_for_timeout(1500)
            except Exception:
                pass

            rendered_html = page.content()
            browser.close()

        # now parse the fully rendered DOM with BS4
        return _scrape_with_bs4(rendered_html)
    except Exception:
        return None


# ── Trafilatura fallback ────────────────────────────────────────────────

def _scrape_with_trafilatura_html(html: str) -> str | None:
    if not HAS_TRAFILATURA:
        return None
    return trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        deduplicate=True,
    )


# ── title extraction ───────────────────────────────────────────────────

def _get_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"].strip()
    title_tag = soup.find("title")
    return title_tag.get_text(strip=True) if title_tag else ""


# ── quality check ───────────────────────────────────────────────────────

def _looks_like_article(text: str) -> bool:
    """Return True if *text* looks like a real article body (not just titles)."""
    if not text or len(text.strip()) < 100:
        return False
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    # an article should have at least some paragraphs longer than 40 chars
    long_lines = [l for l in lines if len(l) > 40]
    return len(long_lines) >= 2


# ── main entry point ───────────────────────────────────────────────────

def fetch_article(url: str) -> dict[str, Any]:
    """Fetch and extract article content from a URL.

    Pipeline:
      1. GET static HTML  →  BS4 extraction
      2. If too short / noisy  →  trafilatura on same HTML
      3. If still bad  →  playwright (headless browser for JS‑rendered sites)
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError("URL phải bắt đầu bằng http:// hoặc https://")

    # ── step 1: fetch static HTML ───────────────────────────────────────
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.encoding = response.apparent_encoding
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}")

    html_text = response.text
    title_text = _get_title(html_text)

    # ── step 2: try BS4 on static HTML ──────────────────────────────────
    clean_text = _scrape_with_bs4(html_text)

    # ── step 3: if BS4 result is weak, try trafilatura ──────────────────
    if not _looks_like_article(clean_text):
        traf = _scrape_with_trafilatura_html(html_text)
        if traf and _looks_like_article(traf):
            clean_text = traf
        elif traf and len(traf.strip()) > len((clean_text or "").strip()):
            clean_text = traf

    # ── step 4: if still weak, try playwright (JS rendering) ────────────
    if not _looks_like_article(clean_text):
        pw_text = _scrape_with_playwright(url)
        if pw_text and len(pw_text.strip()) > len((clean_text or "").strip()):
            clean_text = pw_text

    # ── final validation ────────────────────────────────────────────────
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
