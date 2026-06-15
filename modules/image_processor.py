"""Image loading, quality analysis, and preprocessing."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from config import MAX_IMAGE_SIZE_MB, SUPPORTED_FORMATS


def _source_bytes(source: str | bytes | bytearray) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower().lstrip(".") not in SUPPORTED_FORMATS:
            raise ValueError(f"Định dạng ảnh không được hỗ trợ: {path.suffix or 'không xác định'}")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Không thể đọc ảnh: {exc}") from exc
    raise ValueError("Nguồn ảnh phải là đường dẫn hoặc bytes.")


def load_image(source: str | bytes | bytearray) -> np.ndarray:
    """Load a supported image source and return an RGB numpy array."""
    data = _source_bytes(source)
    if len(data) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"Ảnh vượt quá giới hạn {MAX_IMAGE_SIZE_MB} MB.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            detected_format = (image.format or "").lower()
            if detected_format not in SUPPORTED_FORMATS:
                raise ValueError(f"Định dạng ảnh không được hỗ trợ: {detected_format or 'không xác định'}")
            return np.array(image.convert("RGB"))
    except UnidentifiedImageError as exc:
        raise ValueError("Dữ liệu không phải là ảnh hợp lệ.") from exc


def analyze_quality(img: np.ndarray) -> dict[str, Any]:
    """Measure brightness, contrast, blur, and derive a quality label."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score >= 100 and contrast >= 30:
        quality_level = "good"
    elif blur_score >= 50 or contrast >= 20:
        quality_level = "medium"
    else:
        quality_level = "poor"
    return {
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "is_blurry": blur_score < 100,
        "quality_level": quality_level,
    }


def detect_rotation(img: np.ndarray) -> int:
    """Detect a right-angle rotation using Tesseract OSD, then an OpenCV fallback."""
    try:
        import pytesseract

        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        rotation = int(osd.get("rotate", 0))
        if rotation in (0, 90, 180, 270):
            return rotation
    except Exception:
        pass

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        points = np.vstack([contour.reshape(-1, 2) for contour in contours if cv2.contourArea(contour) > 20])
        if len(points) < 5:
            return 0
        angle = float(cv2.minAreaRect(points.astype(np.float32))[-1])
        if angle > 45:
            angle -= 90
        if abs(angle) < 15:
            return 0
        if 75 <= abs(angle) <= 90:
            return 90 if angle < 0 else 270
    except (ValueError, cv2.error):
        pass
    return 0


def apply_rotation(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate an image by a supported right angle."""
    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if angle == 0:
        return img
    if angle not in rotations:
        raise ValueError("Góc xoay phải là 0, 90, 180 hoặc 270.")
    return cv2.rotate(img, rotations[angle])


def denoise(img: np.ndarray) -> np.ndarray:
    """Denoise an image according to its measured quality."""
    quality = analyze_quality(img)["quality_level"]
    if quality == "poor":
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    if quality == "medium":
        return cv2.GaussianBlur(img, (3, 3), 0)
    return img


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to images with contrast below 30."""
    if analyze_quality(img)["contrast"] >= 30:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def resize_if_needed(img: np.ndarray, max_px: int = 2000) -> np.ndarray:
    """Resize while preserving aspect ratio if either dimension exceeds max_px."""
    if max_px <= 0:
        raise ValueError("max_px phải lớn hơn 0.")
    height, width = img.shape[:2]
    longest = max(height, width)
    if longest <= max_px:
        return img
    scale = max_px / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)


def process_image(source: str | bytes | bytearray) -> dict[str, Any]:
    """Run the complete preprocessing pipeline and return PNG bytes plus a report."""
    original_bytes = _source_bytes(source)
    img = load_image(original_bytes)
    height, width = img.shape[:2]
    quality = analyze_quality(img)
    angle = detect_rotation(img)

    issues: list[str] = []
    fixes: list[str] = []
    if quality["is_blurry"]:
        issues.append("Ảnh bị mờ")
    if quality["contrast"] < 30:
        issues.append("Độ tương phản thấp")
    if quality["brightness"] < 50:
        issues.append("Ảnh quá tối")
    elif quality["brightness"] > 220:
        issues.append("Ảnh quá sáng")
    if angle:
        issues.append(f"Ảnh bị xoay {angle}°")
        img = apply_rotation(img, angle)
        fixes.append(f"Đã xoay {angle}°")
    if quality["quality_level"] != "good":
        img = denoise(img)
        fixes.append("Đã giảm nhiễu")
    if quality["contrast"] < 30:
        img = enhance_contrast(img)
        fixes.append("Đã tăng tương phản")

    before_resize = img.shape[:2]
    img = resize_if_needed(img)
    if img.shape[:2] != before_resize:
        fixes.append("Đã thu nhỏ ảnh")

    success, encoded = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("Không thể mã hóa ảnh đã xử lý sang PNG.")

    return {
        "processed_image_bytes": encoded.tobytes(),
        "original_image_bytes": original_bytes,
        "report": {
            "original_size": (width, height),
            "file_size_kb": len(original_bytes) / 1024,
            **quality,
            "rotation_detected": angle,
            "needs_rotation": angle != 0,
            "issues": issues,
            "applied_fixes": fixes,
        },
    }

