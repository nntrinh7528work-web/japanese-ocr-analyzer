import fitz
import pytest

from modules.pdf_processor import pdf_to_image_sources


def _pdf_bytes(page_count=2):
    document = fitz.open()
    for index in range(page_count):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {index + 1}")
    data = document.tobytes()
    document.close()
    return data


def test_pdf_pages_are_rendered_as_named_jpeg_sources():
    sources = pdf_to_image_sources(_pdf_bytes(), "lesson.pdf")

    assert [name for name, _ in sources] == ["lesson - trang 01.jpg", "lesson - trang 02.jpg"]
    assert all(data.startswith(b"\xff\xd8\xff") for _, data in sources)


def test_pdf_page_limit_is_enforced():
    with pytest.raises(ValueError, match="vượt giới hạn 1 trang"):
        pdf_to_image_sources(_pdf_bytes(2), "lesson.pdf", max_pages=1)


def test_invalid_pdf_is_rejected():
    with pytest.raises(ValueError, match="PDF hợp lệ"):
        pdf_to_image_sources(b"not a pdf", "broken.pdf")
