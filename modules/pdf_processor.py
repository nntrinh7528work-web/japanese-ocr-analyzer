"""Render uploaded PDF pages into images for the existing OCR workflow."""

from __future__ import annotations

import gc
from pathlib import Path
from collections.abc import Iterator

import fitz

from config import MAX_PDF_PAGES, MAX_PDF_SIZE_MB


def iter_pdf_image_sources(
    data: bytes,
    filename: str,
    max_pages: int = MAX_PDF_PAGES,
) -> Iterator[tuple[str, bytes]]:
    """Yield compact JPEG pages one at a time.

    Memory optimisation: each page pixmap is explicitly deleted and garbage
    collected immediately after conversion so that large multi-page PDFs
    (20-50 MB) do not accumulate RAM on servers with limited memory.
    """
    if not data:
        raise ValueError("PDF trống.")
    if len(data) > MAX_PDF_SIZE_MB * 1024 * 1024:
        raise ValueError(f"PDF vượt quá giới hạn {MAX_PDF_SIZE_MB} MB.")

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError("File không phải là PDF hợp lệ.") from exc

    try:
        if document.needs_pass:
            raise ValueError("PDF có mật khẩu nên chưa thể xử lý.")
        if document.page_count == 0:
            raise ValueError("PDF không có trang.")
        if document.page_count > max_pages:
            raise ValueError(f"PDF có {document.page_count} trang, vượt giới hạn {max_pages} trang.")

        stem = Path(filename).stem or "document"
        digits = max(2, len(str(document.page_count)))
        # Use lower scaling for large documents to reduce memory usage.
        scale = 1.1 if document.page_count > 20 else 1.3
        matrix = fitz.Matrix(scale, scale)
        for page_number, page in enumerate(document, 1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            page_name = f"{stem} - trang {page_number:0{digits}d}.jpg"
            page_bytes = pixmap.tobytes("jpeg", jpg_quality=85)
            # Free pixmap memory immediately to avoid accumulation.
            del pixmap
            gc.collect()
            yield page_name, page_bytes
    finally:
        document.close()


def pdf_to_image_sources(
    data: bytes,
    filename: str,
    max_pages: int = MAX_PDF_PAGES,
) -> list[tuple[str, bytes]]:
    """Return all rendered pages for callers that explicitly need a list."""
    return list(iter_pdf_image_sources(data, filename, max_pages=max_pages))
