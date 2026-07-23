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

GEMINI_MODEL_VISION = "gemini-2.5-flash"
GEMINI_MODEL_TEXT = "gemini-2.5-flash"
MAX_UPLOAD_SIZE_MB = 50
MAX_IMAGE_SIZE_MB = 20
MAX_PDF_SIZE_MB = 50
MAX_PDF_PAGES = 60
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]
SUPPORTED_UPLOAD_FORMATS = [*SUPPORTED_FORMATS, "pdf"]

