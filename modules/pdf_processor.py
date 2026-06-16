"""Render uploaded PDF pages into images for the existing OCR workflow."""

from __future__ import annotations

from pathlib import Path

import fitz

from config import MAX_PDF_PAGES, MAX_PDF_SIZE_MB


def pdf_to_image_sources(
    data: bytes,
    filename: str,
    max_pages: int = MAX_PDF_PAGES,
) -> list[tuple[str, bytes]]:
    """Return PDF pages as compact named JPEG byte sources."""
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
        sources = []
        matrix = fitz.Matrix(1.3, 1.3)
        for page_number, page in enumerate(document, 1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            page_name = f"{stem} - trang {page_number:0{digits}d}.jpg"
            sources.append((page_name, pixmap.tobytes("jpeg", jpg_quality=85)))
        return sources
    finally:
        document.close()
