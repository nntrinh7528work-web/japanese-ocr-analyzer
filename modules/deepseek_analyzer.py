"""DeepSeek-based text analysis after OCR, returning structured JSON."""

from __future__ import annotations

import json
from pathlib import Path

from config import DEEPSEEK_MODEL
from modules.deepseek_client import get_deepseek_client

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "deepseek_analysis_prompt.txt"

_REQUIRED_KEYS = {"language", "summary_vi", "ocr_corrections", "vocabulary", "grammar", "kanji"}
_LIST_KEYS = {"ocr_corrections", "vocabulary", "grammar", "kanji"}

_LANGUAGE_NAMES = {
    "ja": "tiếng Nhật",
    "en": "tiếng Anh",
}


def _validate_analysis(data: dict) -> None:
    """Raise ValueError if the analysis dict is missing required keys or has wrong types."""
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Thiếu key bắt buộc: {missing}")
    for key in _LIST_KEYS:
        if not isinstance(data[key], list):
            raise ValueError(f"Key '{key}' phải là list, nhận được {type(data[key]).__name__}")


def analyze_with_deepseek(
    source_text: str,
    language: str,
) -> dict:
    """Analyze OCR text using DeepSeek and return validated JSON dict.

    Args:
        source_text: The text extracted from OCR to analyze.
        language: ``"ja"`` for Japanese, ``"en"`` for English.

    Returns:
        A dict conforming to the DeepSeek analysis schema.

    Raises:
        ValueError: If *language* is not ``"ja"``/``"en"`` or *source_text* is empty.
        RuntimeError: If the API fails to return valid JSON after retries.
    """
    if language not in _LANGUAGE_NAMES:
        raise ValueError(f"language phải là 'ja' hoặc 'en', nhận được '{language}'")
    if not source_text or not source_text.strip():
        raise ValueError("source_text không được rỗng.")

    language_name = _LANGUAGE_NAMES[language]
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    built_prompt = prompt_template.replace("{language_name}", language_name).replace("{source_text}", source_text.strip())

    client = get_deepseek_client()

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": built_prompt},
                ],
                temperature=0.1,
                max_tokens=30000,
                response_format={"type": "json_object"},
            )

            content = (response.choices[0].message.content or "").strip()
            if not content:
                raise ValueError("DeepSeek trả về nội dung rỗng.")

            data = json.loads(content)
            _validate_analysis(data)
            return data

        except Exception as exc:
            last_error = exc
            # Retry once on first failure.
            if attempt == 0:
                continue

    raise RuntimeError(
        f"DeepSeek phân tích thất bại sau 2 lần thử: {last_error}"
    ) from last_error
