import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from modules.image_processor import (
    analyze_quality,
    apply_rotation,
    load_image,
    process_image,
    resize_if_needed,
)


def make_image_bytes(fmt="JPEG", size=(500, 300)):
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for y in range(30, size[1] - 20, 35):
        draw.text((20, y), "Japanese OCR test 12345", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def test_process_path_and_bytes(tmp_path):
    path = tmp_path / "test_horizontal.jpg"
    path.write_bytes(make_image_bytes())

    from_path = process_image(str(path))
    from_bytes = process_image(path.read_bytes())

    assert from_path["processed_image_bytes"].startswith(b"\x89PNG")
    assert isinstance(from_bytes["processed_image_bytes"], bytes)
    assert from_path["report"]["quality_level"] in {"good", "medium", "poor"}
    assert from_path["report"]["original_size"] == (500, 300)


def test_helpers_and_validation():
    image = load_image(make_image_bytes("PNG", (3000, 1000)))
    quality = analyze_quality(image)
    resized = resize_if_needed(image)
    rotated = apply_rotation(image, 90)

    assert set(("brightness", "contrast", "blur_score", "is_blurry", "quality_level")) <= quality.keys()
    assert resized.shape[1] == 2000
    assert rotated.shape[:2] == (3000, 1000)
    with pytest.raises(ValueError):
        load_image(b"not an image")
    with pytest.raises(ValueError):
        apply_rotation(np.zeros((10, 10, 3), dtype=np.uint8), 45)

