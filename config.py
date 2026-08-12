"""Application-wide configuration."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _get_streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        return None


GEMINI_API_KEY = _get_streamlit_secret("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY == "your_api_key_here":
    GEMINI_API_KEY = None

GEMINI_MODEL_VISION = _get_streamlit_secret("GEMINI_MODEL_VISION") or os.getenv("GEMINI_MODEL_VISION") or "gemini-3.5-flash"
GEMINI_MODEL_TEXT = _get_streamlit_secret("GEMINI_MODEL_TEXT") or os.getenv("GEMINI_MODEL_TEXT") or "gemini-3.5-flash"

# Notion credentials are deployment secrets.  Database/data-source IDs are
# optional because the app can bootstrap them below NOTION_PARENT_PAGE_ID.
NOTION_TOKEN = _get_streamlit_secret("NOTION_TOKEN") or os.getenv("NOTION_TOKEN")
NOTION_PARENT_PAGE_ID = _get_streamlit_secret("NOTION_PARENT_PAGE_ID") or os.getenv("NOTION_PARENT_PAGE_ID")
NOTION_LESSONS_DATABASE_ID = _get_streamlit_secret("NOTION_LESSONS_DATABASE_ID") or os.getenv("NOTION_LESSONS_DATABASE_ID")
NOTION_LESSONS_DATA_SOURCE_ID = _get_streamlit_secret("NOTION_LESSONS_DATA_SOURCE_ID") or os.getenv("NOTION_LESSONS_DATA_SOURCE_ID")
NOTION_ITEMS_DATABASE_ID = _get_streamlit_secret("NOTION_ITEMS_DATABASE_ID") or os.getenv("NOTION_ITEMS_DATABASE_ID")
NOTION_ITEMS_DATA_SOURCE_ID = _get_streamlit_secret("NOTION_ITEMS_DATA_SOURCE_ID") or os.getenv("NOTION_ITEMS_DATA_SOURCE_ID")
NOTION_SENTENCES_DATABASE_ID = _get_streamlit_secret("NOTION_SENTENCES_DATABASE_ID") or os.getenv("NOTION_SENTENCES_DATABASE_ID")
NOTION_SENTENCES_DATA_SOURCE_ID = _get_streamlit_secret("NOTION_SENTENCES_DATA_SOURCE_ID") or os.getenv("NOTION_SENTENCES_DATA_SOURCE_ID")
NOTION_KANJI_DATABASE_ID = _get_streamlit_secret("NOTION_KANJI_DATABASE_ID") or os.getenv("NOTION_KANJI_DATABASE_ID")
NOTION_KANJI_DATA_SOURCE_ID = _get_streamlit_secret("NOTION_KANJI_DATA_SOURCE_ID") or os.getenv("NOTION_KANJI_DATA_SOURCE_ID")
NOTION_LANGUAGE_DATABASE_ID = _get_streamlit_secret("NOTION_LANGUAGE_DATABASE_ID") or os.getenv("NOTION_LANGUAGE_DATABASE_ID")
NOTION_LANGUAGE_DATA_SOURCE_ID = _get_streamlit_secret("NOTION_LANGUAGE_DATA_SOURCE_ID") or os.getenv("NOTION_LANGUAGE_DATA_SOURCE_ID")
PUBLIC_APP_URL = (
    _get_streamlit_secret("PUBLIC_APP_URL")
    or os.getenv("PUBLIC_APP_URL")
    or "https://japanese-ocr-analyzer-vn.streamlit.app/"
)
MAX_UPLOAD_SIZE_MB = 50
MAX_IMAGE_SIZE_MB = 20
MAX_PDF_SIZE_MB = 50
MAX_PDF_PAGES = 60
MAX_VIDEO_SIZE_MB = 100
MAX_VIDEO_DURATION_SECONDS = 30 * 60
SUPPORTED_VIDEO_FORMATS = ["mp4", "mov", "webm", "mpeg", "mpg", "avi"]
GEMINI_MODEL_VIDEO = _get_streamlit_secret("GEMINI_MODEL_VIDEO") or os.getenv("GEMINI_MODEL_VIDEO") or "gemini-3.6-flash"
# Gemini 2.5 Flash-Lite is no longer available to new Gemini API users.
# Keep this separate from the regular text model because video segments favor
# low-cost, high-throughput structured extraction.
GEMINI_MODEL_VIDEO_BATCH = _get_streamlit_secret("GEMINI_MODEL_VIDEO_BATCH") or os.getenv("GEMINI_MODEL_VIDEO_BATCH") or "gemini-3.5-flash-lite"
GEMINI_MODEL_AUDIO = _get_streamlit_secret("GEMINI_MODEL_AUDIO") or os.getenv("GEMINI_MODEL_AUDIO") or "gemini-3.5-flash-lite"
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]
SUPPORTED_UPLOAD_FORMATS = [*SUPPORTED_FORMATS, "pdf"]
