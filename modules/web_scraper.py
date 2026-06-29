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


def _scrape_with_trafilatura_html(html: str) -> str | None:
    if not HAS_TRAFILATURA:
        return None
    return trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        deduplicate=True,
    )


def _scrape_with_bs4(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # 1. Xóa các thẻ hoàn toàn không liên quan tới nội dung
    for tag in soup(["script", "style", "form", "iframe", "noscript", "svg"]):
        tag.decompose()

    # 2. Tìm container chứa bài báo TRƯỚC
    article_container = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_=re.compile(
            r"article[_-]body|entry[_-]content|post[_-]content|story[_-]body|"
            r"article[_-]content|main[_-]content|post[_-]body|news[_-]body|"
            r"article[_-]text|content[_-]body|story[_-]text",
            re.I,
        ))
        or soup.find(id=re.compile(
            r"article[_-]body|entry[_-]content|post[_-]content|article[_-]content|"
            r"main[_-]content|post[_-]body|news[_-]body|article[_-]text|content[_-]body",
            re.I,
        ))
    )

    if article_container:
        # 3a. Chỉ lọc noise bên TRONG container bài báo (các block phụ nhúng vào bài)
        inner_noise = re.compile(
            r"share|social|comment|relation|keyword|recommend|subscribe|newsletter|"
            r"advertisement|banner|popup|author[_-]bio|breadcrumb|pagination|"
            r"tag[_-]list|more[_-]articles|read[_-]next|popular|ranking|pickup|latest|utility|"
            r"関連|注目|おすすめ|アクセス|ランキング|シェア|話題",
            re.I,
        )
        for el in article_container.find_all(attrs={"class": inner_noise}):
            el.decompose()
        for el in article_container.find_all(attrs={"id": inner_noise}):
            el.decompose()
        for tag in article_container(["nav", "footer", "header", "aside"]):
            tag.decompose()
    else:
        # 3b. Fallback: lọc noise toàn trang rồi dùng body
        for tag in soup(["nav", "footer", "header", "aside"]):
            tag.decompose()
        broad_noise = re.compile(
            r"sidebar|menu|footer|header|nav|widget|share|social|comment|"
            r"related|popular|newsletter|cookie|banner|pagination",
            re.I,
        )
        for el in soup.find_all(attrs={"class": broad_noise}):
            el.decompose()
        for el in soup.find_all(attrs={"id": broad_noise}):
            el.decompose()
        article_container = soup.find("body") or soup

    # 3. Cắt bỏ các phần kết bài chứa bài viết liên quan (rất phổ biến ở báo Nhật/Anh)
    stop_words = [
        "あわせて読みたい", "関連記事", "関連ニュース", "注目ワード", "おすすめ記事", 
        "関連リンク", "おすすめのニュース", "話題のワード", "アクセスランキング",
        "related stories", "related articles", "read more", "you may also like",
        "recommended for you", "more on this story"
    ]
    
    for tag in article_container.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "div"]):
        text = tag.get_text(strip=True)
        if len(text) < 100 and any(word in text.lower() for word in stop_words):
            # Decompose tag này và tất cả các sibling phía sau nó
            siblings = [tag] + list(tag.next_siblings)
            for sib in siblings:
                if hasattr(sib, "decompose"):
                    sib.decompose()
            break

    # 4. Trích xuất đoạn văn bản từ container
    paragraphs = article_container.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
    cleaned_paras = []
    for p in paragraphs:
        text = p.get_text(strip=True)
        if not text:
            continue
        # Bỏ đoạn quá ngắn (trừ heading) — ngưỡng 10 để tương thích tiếng Nhật (kanji)
        if len(text) < 10 and not p.name.startswith("h"):
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

    # 1. Tải HTML qua requests trước
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.encoding = response.apparent_encoding
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}")
    
    html_text = response.text
    title_text = _get_title(html_text)

    # 2. Sử dụng BS4 trước vì độ chính xác cao và được tối ưu hóa cho các trang báo
    clean_text = _scrape_with_bs4(html_text)

    # 3. Nếu BS4 không tìm được hoặc nội dung quá ngắn, fallback sang Trafilatura
    if not clean_text or len(clean_text.strip()) < 100:
        trafilatura_text = _scrape_with_trafilatura_html(html_text)
        if trafilatura_text and len(trafilatura_text.strip()) > len(clean_text or ""):
            clean_text = trafilatura_text

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
