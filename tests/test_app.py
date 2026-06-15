import io

from PIL import Image
from streamlit.testing.v1 import AppTest

from modules.multi_image_workflow import create_image_item


def _image_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (100, 60), color).save(buffer, "PNG")
    return buffer.getvalue()


def test_app_starts_without_upload():
    app = AppTest.from_file("app.py").run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "🔍 Japanese OCR Analyzer"
    assert any("một hoặc nhiều ảnh" in item.value for item in app.info)
    assert len(app.tabs) == 2


def test_app_renders_two_independent_image_flows():
    app = AppTest.from_file("app.py")
    app.session_state["image_items"] = [
        create_image_item(_image_bytes("white"), "page-1.png"),
        create_image_item(_image_bytes("black"), "page-2.png"),
    ]
    app.session_state["analysis"] = None
    app.session_state["uploader_version"] = 0
    app.session_state["camera_version"] = 0
    app.run(timeout=20)

    assert not app.exception
    labels = [button.label for button in app.button]
    assert labels.count("🔍 OCR ảnh này") == 2
    assert "🔍 OCR tất cả ảnh chưa xử lý" in labels
    assert "Ảnh trong bộ phân tích (2)" in [heading.value for heading in app.subheader]
