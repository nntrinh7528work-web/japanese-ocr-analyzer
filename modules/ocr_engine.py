"""Gemini Vision OCR integration."""

from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Any

from PIL import Image


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "ocr_prompt.txt"


def init_gemini():
    """Initialize and return the configured Gemini vision model."""
    import google.generativeai as genai

    from config import GEMINI_API_KEY, GEMINI_MODEL_VISION

    if not GEMINI_API_KEY:
        raise ValueError("Thiếu GEMINI_API_KEY. Hãy cấu hình key trong .env hoặc Streamlit secrets.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL_VISION)


def build_ocr_prompt(preprocessing_report: dict) -> str:
    """Build the OCR prompt with preprocessing context."""
    issues = preprocessing_report.get("issues") or []
    image_info = "\n".join(
        (
            f"- Chất lượng ảnh: {preprocessing_report.get('quality_level', 'unknown')}",
            f"- Rotation đã chỉnh: {preprocessing_report.get('rotation_detected', 0)}°",
            f"- Các vấn đề đã phát hiện: {', '.join(issues) if issues else 'Không có'}",
        )
    )
    return PROMPT_PATH.read_text(encoding="utf-8").replace("{image_info}", image_info)


def _header_value(text: str, name: str, default: str) -> str:
    match = re.search(rf"(?im)^\s*{re.escape(name)}\s*:\s*\[?([^\]\r\n]+)\]?", text)
    return match.group(1).strip().lower() if match else default


def _section(text: str, name: str) -> str:
    match = re.search(rf"---{name}_START---\s*(.*?)\s*---{name}_END---", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_ocr_response(response_text: str) -> dict[str, Any]:
    """Parse Gemini's structured OCR response."""
    direction = _header_value(response_text, "TEXT_DIRECTION", "horizontal")
    if direction not in {"horizontal", "vertical", "mixed"}:
        direction = "horizontal"
    confidence = _header_value(response_text, "CONFIDENCE", "low")
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    try:
        regions = int(_header_value(response_text, "TEXT_REGIONS", "1"))
    except ValueError:
        regions = 1

    raw_text = _section(response_text, "OCR")
    notes_text = _section(response_text, "NOTES")
    notes = [
        re.sub(r"^\s*[-*•]\s*", "", line).strip()
        for line in notes_text.splitlines()
        if line.strip() and line.strip().lower() not in {"なし", "none", "không có"}
    ]
    clean_text = re.sub(r"【要確認:.*?】", "", raw_text, flags=re.DOTALL)
    clean_text = re.sub(r"[ \t]+\n", "\n", clean_text).strip()
    return {
        "raw_text": raw_text,
        "clean_text": clean_text,
        "ocr_notes": notes,
        "text_direction": direction,
        "text_regions": max(1, regions),
        "has_furigana": _header_value(response_text, "HAS_FURIGANA", "no") == "yes",
        "confidence": confidence,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _usage(response: Any) -> dict[str, int]:
    metadata = getattr(response, "usage_metadata", None)
    candidate_tokens = int(getattr(metadata, "candidates_token_count", 0) or 0)
    thinking_tokens = int(getattr(metadata, "thoughts_token_count", 0) or 0)
    return {
        "input_tokens": int(getattr(metadata, "prompt_token_count", 0) or 0),
        "output_tokens": candidate_tokens + thinking_tokens,
        "candidate_tokens": candidate_tokens,
        "thinking_tokens": thinking_tokens,
    }


def run_ocr(image_bytes: bytes, preprocessing_report: dict) -> dict[str, Any]:
    """Run OCR through Gemini Vision, retrying transient failures twice."""
    if not image_bytes:
        raise ValueError("Ảnh đầu vào không được rỗng.")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Dữ liệu ảnh đầu vào không hợp lệ.") from exc

    model = init_gemini()
    prompt = build_ocr_prompt(preprocessing_report)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = model.generate_content([prompt, image])
            result = parse_ocr_response(response.text)
            if not result["raw_text"]:
                raise ValueError("Gemini không trả về phần OCR hợp lệ.")
            result["usage"] = _usage(response)
            return result
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"OCR thất bại sau 3 lần thử: {last_error}") from last_error
