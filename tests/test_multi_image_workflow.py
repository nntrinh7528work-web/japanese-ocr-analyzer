import io

from PIL import Image

import fitz

from modules.multi_image_workflow import add_image_items, add_upload_items, combined_notes, combined_text, move_image_item


def image_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (100, 60), color).save(buffer, "PNG")
    return buffer.getvalue()


def test_add_multiple_images_and_ignore_duplicates():
    first = image_bytes("white")
    second = image_bytes("black")
    items, added, errors = add_image_items([], [("one.png", first), ("two.png", second), ("copy.png", first)])

    assert len(items) == 2
    assert added == ["one.png", "two.png"]
    assert errors == []


def test_combine_text_and_notes_in_image_order():
    items = [
        {"name": "one.png", "edited_text": "一番", "ocr_result": {"ocr_notes": ["note 1"]}},
        {"name": "two.png", "edited_text": "二番", "ocr_result": {"ocr_notes": ["note 2"]}},
        {"name": "empty.png", "edited_text": "", "ocr_result": None},
    ]

    text = combined_text(items)
    assert text.index("一番") < text.index("二番")
    assert "=== ẢNH 1: one.png ===" in text
    assert combined_notes(items) == ["Ảnh 1 (one.png): note 1", "Ảnh 2 (two.png): note 2"]


def test_move_image_changes_combined_order():
    items = [
        {"id": "one", "name": "one.png", "edited_text": "一番", "ocr_result": None},
        {"id": "two", "name": "two.png", "edited_text": "二番", "ocr_result": None},
    ]
    reordered = move_image_item(items, "two", -1)
    assert [item["id"] for item in reordered] == ["two", "one"]
    assert combined_text(reordered).index("二番") < combined_text(reordered).index("一番")


def test_add_pdf_expands_pages_into_image_items():
    document = fitz.open()
    for text in ("first", "second"):
        page = document.new_page()
        page.insert_text((72, 72), text)
    pdf = document.tobytes()
    document.close()

    items, added, errors = add_upload_items([], [("lesson.pdf", pdf)])

    assert len(items) == 2
    assert added == ["lesson - trang 01.jpg", "lesson - trang 02.jpg"]
    assert errors == []
